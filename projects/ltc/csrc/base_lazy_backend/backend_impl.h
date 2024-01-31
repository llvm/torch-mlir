//===- backend_impl.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// The Torch-MLIR backend class API that handles lowering LTC ATen ops to MLIR
// using the Torch-MLIR ATen dialect
//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_backend_impl.h
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <sstream>

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/shape.h>

namespace torch {
namespace lazy {

class TORCH_API TorchMlirBackendData : public BackendData {
public:
  struct Info : public BackendData::Info {
    at::Tensor tensor;
    c10::optional<at::Scalar> scalar;
    bool requires_grad;
    std::string name;

    Info() {
      static int i = 0;
      std::stringstream ss;
      ss << "placeholder" << i;
      name = ss.str();
      ++i;
    }
    Info(const Info &other)
        : tensor{other.tensor}, scalar{other.scalar},
          requires_grad{other.requires_grad}, name{other.name} {}
    Info(const at::Tensor &tensor)
        : tensor{tensor}, requires_grad{tensor.requires_grad()} {}
    Info(const at::Scalar &scalar) : scalar{scalar}, requires_grad(false) {}
  };

  TorchMlirBackendData(BackendDevice device, Shape shape);
  TorchMlirBackendData(BackendDevice device, Shape shape,
                       std::shared_ptr<BackendData::Info> info);
  TorchMlirBackendData(const at::Scalar &scalar, BackendDevice device);
  TorchMlirBackendData(const at::Tensor &tensor, BackendDevice device,
                       Shape shape);

  virtual BackendData::Handle GetHandle() override;

  virtual void Assign(const BackendData &data) override;

  virtual bool HasValue() const override;

  BackendData::Info *mlir_info() const;

protected:
  std::shared_ptr<BackendData::Info> info_;
};

class TORCH_API TorchMlirBackendImpl : public BackendImplInterface {
public:
  virtual ~TorchMlirBackendImpl() = default;

  /**
   * Initialization/Teardown
   * */
  virtual void PrepareToExit() const override;

  /**
   * IR Tracing
   * */

  const IrBuilder *GetIrBuilder() const override;

  /**
   * Configuration
   * */
  // virtual void SetRngSeed(size_t seed) const = 0;

  /**
   * Data Transfer
   * */

  virtual BackendDataPtr
  MakeComputationDataFromTensor(const at::Tensor &tensor, const Shape &shape,
                                const BackendDevice &device) const override;

  virtual BackendDataPtr
  MakeComputationDataFromScalar(const at::Scalar &scalar,
                                const BackendDevice &device) const override;

  virtual BackendDataPtr
  CreateDataPlaceholder(const BackendDevice &device,
                        const Shape &shape) const override;

  // Gets backend data if the node is a device data node. Otherwise returns
  // nullptr.
  virtual BackendDataPtr
  GetComputationDataFromNode(const Node *) const override;

  virtual at::Tensor MakeTensorFromComputationData(
      const BackendDataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override;

  /**
   * Lowering, Compilation, Execution
   * */

  virtual std::unique_ptr<LoweringContext>
  CreateLoweringContext(const std::string &name, BackendDevice device,
                        c10::ArrayRef<const Node *> post_order,
                        Util::EmissionMap emit_status) const override;

  virtual std::unique_ptr<LoweringContext>
  CreateLoweringContext(const std::string &name,
                        BackendDevice device) const override;

  // TODO(whc) need to keep this?
  // virtual std::vector<std::string> GetCompilationDevices(
  //     const std::string& device, c10::ArrayRef<std::string> devices
  // ) const = 0;

  // virtual std::vector<ComputationPtr> Compile(
  //     std::vector<ComputationPtr> instances
  // ) const = 0;

  // virtual std::vector<BackendDataPtr> ExecuteComputation(
  //     Computation& computation,
  //     c10::ArrayRef<BackendDataPtr> arguments,
  //     const BackendDevice& device
  // ) const = 0;

  /**
   * Device Configuration
   * */

  // Set or get the default device type.
  // For backends used with virtual c10:: Devices, this configures what real
  // device type the backend should use, and matters if the backend supports
  // more than one type of real device.

  // virtual std::shared_ptr<BackendDeviceType> GetDefaultDeviceType() const =
  // 0;
  // virtual void SetDefaultDeviceType(std::string device_type) = 0;

  // Specify which aten device should be used for eager fallback
  // may change depending on current 'Default' DeviceType
  virtual at::DeviceType EagerFallbackDeviceType() const override;

  // Query all available backend devices
  virtual std::vector<BackendDevice> GetBackendDevices() const override;

  // Map a particular c10:: device to a concrete backend device
  // Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
  // virtual devices, meaning they may map to a gpu, tpu, etc. behind the
  // scenes. In the future, non-virtual c10:: devices may also use lazy tensors
  // through a mode, in which case these APIs should still work, but should be
  // identity mappings.
  virtual BackendDevice GetBackendDevice(c10::Device device) const override;

  virtual int64_t GetDefaultDeviceOrdinal() const override;

  virtual void SetDefaultDeviceOrdinal(int64_t ordinal) override;

  /**
   * Debug/Metrics
   * */

  // virtual std::map<std::string, Metric> GetMetrics() const = 0;

  // virtual MemoryInfo GetMemoryInfo(const std::string& device) = 0;

  // virtual std::string GetComputationBackendText(
  //     const ComputationPtr computation
  // ) const = 0;

protected:
  int64_t default_device_ordinal = 0;
};

} // namespace lazy
} // namespace torch
