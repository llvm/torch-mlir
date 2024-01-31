//===- backend_impl.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <c10/core/DeviceType.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/shape.h>

#include <base_lazy_backend/backend_impl.h>
#include <base_lazy_backend/generated/LazyNativeFunctions.h>
#include <base_lazy_backend/mlir_lowering_context.h>
#include <base_lazy_backend/utils/debug.h>
#include <base_lazy_backend/utils/exception.h>
#include <base_lazy_backend/utils/string_utils.h>

#include "backend_impl.h"

using namespace torch::lazy;

namespace torch {
namespace lazy {

/// Returns true if a string begins with another.
inline bool beginswith(const std::string& s, const std::string& t) {
  return s.size() >= t.size() && s.compare(0, t.size(), t) == 0;
}

struct ReferenceLazyBackendDeviceType : public BackendDeviceType {
  ReferenceLazyBackendDeviceType(c10::DeviceType device_type)
      : device_type_(device_type) {}
  ReferenceLazyBackendDeviceType(int8_t device_type)
      : device_type_(static_cast<c10::DeviceType>(device_type)) {}

  std::string toString() const override {
    return c10::DeviceTypeName(device_type_);
  }

  c10::DeviceType device_type_;
};

class ReferenceLazyBackendImpl : public torch::lazy::TorchMlirBackendImpl {
public:
  ReferenceLazyBackendImpl() : default_device_type_(c10::DeviceType::Lazy) {}

  /**
   * Configuration
   * */
  void SetRngSeed(size_t seed) const override {
    std::cout << "RNG Seed Set to: " << seed << std::endl;
  }

  /**
   * Lowering, Compilation, Execution
   * */
  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  };

  std::vector<ComputationPtr>
  Compile(std::vector<ComputationPtr> instances) const override {
    PRINT_FUNCTION();

    // Vendor backend specific lowering can be exec here before returning.
    for (const auto& instance : instances) {
      TORCH_CHECK(
          instance->in_mark_step, "Compile outside of mark step:\n",
          GetComputationBackendText(instance));
      // Store computation instance for external access after compilation.
      GetLatestComputation() = instance;
    }

    std::cout << "Received " << instances.size()
              << " computation instances at Compile!" << std::endl;

    return instances;
  }

  std::vector<BackendDataPtr> ExecuteComputation(
      torch::lazy::ComputationPtr computation,
      c10::ArrayRef<BackendDataPtr> arguments,
      const BackendDevice& device) const override {
    PRINT_FUNCTION();

    // `arguments` maps 1:1 with the parameters in the generated MLIR. In this
    // function, we will generate a list of BackendData that corresponds to the
    // return values in the MLIR.

    auto mlir_computation =
        static_cast<TorchMlirComputation*>(computation.get());

    int num_inputs = 0;

    // Vendor backend specific execution can be inserted here.
    //
    // We don't have a way to execute a computation based on the generated MLIR,
    // so we'll fallback to the implementation used by the TS LTC backend.
    //
    // JIT Execution adopted from:
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_backend_impl.cpp
    std::shared_ptr<torch::jit::Graph> graph = mlir_computation->graph();
    for (auto* node : graph->nodes()) {
      // Convert any lazy devices to cpu devices to ensure
      // that the values are actually computed
      if (node->outputs().size() == 1 &&
          node->output()->type()->kind() == c10::TypeKind::DeviceObjType) {
        auto value_sym = torch::jit::Symbol::attr("value");
        TORCH_CHECK(
            node->hasAttribute(value_sym),
            "Expected node to have 'value' attribute.");
        TORCH_CHECK(
            node->kindOf(value_sym) == torch::jit::AttributeKind::s,
            "Expected 'value' attribute to be a string.");
        if (beginswith(node->s(value_sym), "lazy")) {
          node->s_(value_sym, "cpu");
        }
      }
    }

    torch::jit::GraphExecutor graph_executor(graph, "");
    std::vector<torch::jit::IValue> stack;
    for (const auto& argument : arguments) {
      const auto mlir_data =
          std::static_pointer_cast<TorchMlirBackendData>(argument);
      auto* info =
          dynamic_cast<TorchMlirBackendData::Info*>(mlir_data->mlir_info());
      TORCH_CHECK(info);
      if (info->scalar.has_value()) {
        stack.emplace_back(info->scalar.value());
      } else {
        at::Tensor tensor = info->tensor;
        stack.emplace_back(tensor);
      }

      // count number of inputs
      auto name = info->name;
      if (startswith(name, "input_")) {
        // Printing tensor name for testing purposes
        std::cout << "Input tensor: " << name << std::endl;
        ++num_inputs;
      }
    }
    // Printing number of input tensors for testing purposes
    std::cout << num_inputs << " input tensors found" << std::endl;
    graph_executor.run(stack);
    std::vector<torch::lazy::BackendDataPtr> results;
    for (torch::jit::IValue component : stack) {
      at::Tensor result = component.toTensor();
      at::IntArrayRef result_sizes = result.sizes();
      torch::lazy::Shape shape(
          result.scalar_type(),
          std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
      results.push_back(
          std::make_shared<TorchMlirBackendData>(result, device, shape));
    }

    std::cout << "Received " << arguments.size() << " arguments, and returned "
              << results.size() << " results during ExecuteCompile!"
              << std::endl;

    return results;
  }

  /**
   * Device Configuration
   * */
  std::shared_ptr<torch::lazy::BackendDeviceType>
  GetDefaultDeviceType() const override {
    return std::make_shared<BackendDeviceType>(default_device_type_);
  }

  void SetDefaultDeviceType(int8_t device_type) override {
    default_device_type_ = ReferenceLazyBackendDeviceType(device_type);
  }

  /**
   * Debug/Metrics
   * */
  std::string
  GetComputationBackendText(const ComputationPtr computation) const override {
    // Store computation instance for external access after compilation.
    // We do this in GetComputationBackendText since there may be instances
    // where a ComputationPtr does not pass through Compile (e.g. when using
    // DumpUtil::ToBackend.)
    GetLatestComputation() = computation;

    return computation->to_string();
  }

private:
  ReferenceLazyBackendDeviceType default_device_type_;
};

BackendImplInterface* GetReferenceLazyBackendImpl() {
  static ReferenceLazyBackendImpl* reference_lazy_backend_impl =
      new ReferenceLazyBackendImpl();
  return reference_lazy_backend_impl;
}

void InitReferenceLazyBackend() {
  at::RegisterTorchMlirLazyNativeFunctions();
  static std::unique_ptr<BackendRegistrar> g_registrar;
  g_registrar.reset(new BackendRegistrar(GetReferenceLazyBackendImpl()));

  static LazyGraphExecutor* executor = new LazyGraphExecutor();
  LazyGraphExecutor::Register(executor);
}

ComputationPtr& GetLatestComputation() {
  // Store the computation from the most recent compile.
  static ComputationPtr computation;
  return computation;
}

} // namespace lazy
} // namespace torch
