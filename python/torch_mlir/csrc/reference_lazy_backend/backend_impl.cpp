//===- backend_impl.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/shape.h>

#include <torch_mlir/csrc/base_lazy_backend/backend_impl.h>
#include <torch_mlir/csrc/base_lazy_backend/generated/LazyNativeFunctions.h>
#include <torch_mlir/csrc/base_lazy_backend/mlir_lowering_context.h>
#include <torch_mlir/csrc/base_lazy_backend/utils/debug.h>
#include <torch_mlir/csrc/base_lazy_backend/utils/exception.h>

#include "backend_impl.h"

using namespace torch::lazy;

namespace torch {
namespace lazy {

struct ReferenceLazyBackendDeviceType : public BackendDeviceType {
  ReferenceLazyBackendDeviceType(std::string device_type)
      : device_type_(device_type) {}

  std::string toString() const override { return device_type_; }

  std::string device_type_;
};

class ReferenceLazyBackendImpl : public torch::lazy::TorchMlirBackendImpl {
public:
  ReferenceLazyBackendImpl() : default_device_type_("Magic") {}

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

    // Vendor backend specific execution can be inserted here.
    //
    // We don't have a way to execute a computation based on the generated MLIR,
    // so we'll fallback to the implementation used by the TS LTC backend.
    //
    // JIT Execution adopted from:
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_backend_impl.cpp
    torch::jit::GraphExecutor graph_executor(mlir_computation->graph(), "");
    std::vector<torch::jit::IValue> stack;
    for (const auto& argument : arguments) {
      const auto mlir_data =
          std::static_pointer_cast<TorchMlirBackendData>(argument);
      if (mlir_data->mlir_info()->scalar.has_value()) {
        stack.emplace_back(mlir_data->mlir_info()->scalar.value());
      } else {
        at::Tensor tensor = mlir_data->mlir_info()->tensor;
        stack.emplace_back(tensor);
      }
    }
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
  std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType() const {
    return std::make_shared<BackendDeviceType>(default_device_type_);
  }

  void SetDefaultDeviceType(std::string device_type) {
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
}

ComputationPtr& GetLatestComputation() {
  // Store the computation from the most recent compile.
  static ComputationPtr computation;
  return computation;
}

} // namespace lazy
} // namespace torch
