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

#include <torch_mlir/csrc/base_lazy_backend/LazyNativeFunctions.h>
#include <torch_mlir/csrc/base_lazy_backend/backend_impl.h>
#include <torch_mlir/csrc/base_lazy_backend/mlir_lowering_context.h>
#include <torch_mlir/csrc/utils/debug.h>
#include <torch_mlir/csrc/utils/exception.h>

#include "backend_impl.h"

using namespace torch::lazy;

namespace torch {
namespace lazy {

struct ExampleMlirBackendDeviceType : public BackendDeviceType {
  ExampleMlirBackendDeviceType(std::string device_type)
      : device_type_(device_type) {}

  std::string toString() const override { return device_type_; }

  std::string device_type_;
};

class ExampleMlirBackendImpl : public torch::lazy::TorchMlirBackendImpl {
public:
  ExampleMlirBackendImpl() : default_device_type_("Magic") {}

  /**
   * Configuration
   * */
  void SetRngSeed(size_t seed) const override {
    std::cout << "RNG Seed Set to: " << seed << std::endl;
  }

  /**
   * Lowering, Compilation, Execution
   * */
  std::vector<std::string>
  GetCompilationDevices(const std::string &device,
                        c10::ArrayRef<std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  };

  std::vector<ComputationPtr>
  Compile(std::vector<ComputationPtr> instances) const override {
    PRINT_FUNCTION();

    // Vendor backend specific lowering can be exec here before returning.
    for (const auto &instance : instances) {
      std::cout << "Instance received at Compile: \n"
                << GetComputationBackendText(instance) << std::endl;
    }

    return instances;
  }

  std::vector<BackendDataPtr>
  ExecuteComputation(Computation &computation,
                     c10::ArrayRef<BackendDataPtr> arguments,
                     const BackendDevice &device) const override {
    PRINT_FUNCTION();

    // `arguments` maps 1:1 with the parameters in the generated MLIR. In this
    // function, we will generate a list of BackendData that corresponds to the
    // return values in the MLIR.
    std::vector<torch::lazy::BackendDataPtr> results;

    // "Borrow" some tensor data from arguments to reuse in return. This ensures
    // that the tensor device is correctly configured.
    TORCH_CHECK(arguments.size() > 0,
                "Need at least one argument for example execution.");
    const TorchMlirBackendData *torch_mlir_data =
        dynamic_cast<const TorchMlirBackendData *>(arguments[0].get());
    TORCH_CHECK(torch_mlir_data,
                "Invalid Backend Data Pointer. Expected TorchMlirBackendData.");

    // For this demo we aren't performing a legitimate execution, so we generate
    // some dummy data to return based on the expected number of return values.
    auto mlir_computation = static_cast<TorchMlirComputation *>(&computation);
    for (unsigned i = 0; i < mlir_computation->num_results(); i++) {
      results.push_back(std::make_shared<TorchMlirBackendData>(
          torch_mlir_data->mlir_info()->tensor, device,
          torch_mlir_data->shape()));
    }

    return results;
  }

  /**
   * Device Configuration
   * */
  std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType() const {
    return std::make_shared<BackendDeviceType>(default_device_type_);
  }

  void SetDefaultDeviceType(std::string device_type) {
    default_device_type_ = ExampleMlirBackendDeviceType(device_type);
  }

  /**
   * Debug/Metrics
   * */
  std::string
  GetComputationBackendText(const ComputationPtr computation) const override {
    auto mlir_computation =
        static_cast<TorchMlirComputation *>(computation.get());
    return mlir_computation->to_string();
  }

private:
  ExampleMlirBackendDeviceType default_device_type_;
};

BackendImplInterface *GetExampleMlirBackendImpl() {
  static ExampleMlirBackendImpl *example_mlir_backend_impl =
      new ExampleMlirBackendImpl();
  return example_mlir_backend_impl;
}

void InitExampleMlirBackend() {
  at::RegisterTorchMlirLazyNativeFunctions();
  static std::unique_ptr<BackendRegistrar> g_registrar;
  g_registrar.reset(new BackendRegistrar(GetExampleMlirBackendImpl()));
}

} // namespace lazy
} // namespace torch
