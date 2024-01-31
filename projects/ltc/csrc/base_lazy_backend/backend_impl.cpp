//===- backend_impl.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_backend_impl.cpp
//===----------------------------------------------------------------------===//

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/shape.h>

#include "backend_impl.h"
#include "ir_builder.h"
#include "mlir_lowering_context.h"
#include "ops/device_data.h"
#include "utils/debug.h"
#include "utils/exception.h"

namespace torch {
namespace lazy {

TorchMlirBackendData::TorchMlirBackendData(BackendDevice device, Shape shape)
    : BackendData(device, shape),
      info_(std::make_shared<TorchMlirBackendData::Info>()) {
  PRINT_FUNCTION();
}
TorchMlirBackendData::TorchMlirBackendData(
    BackendDevice device, Shape shape, std::shared_ptr<BackendData::Info> info)
    : BackendData(device, shape), info_(info) {
  PRINT_FUNCTION();
}
TorchMlirBackendData::TorchMlirBackendData(const at::Scalar &scalar,
                                           BackendDevice device)
    : BackendData(device, Shape(scalar.type(), {})),
      info_(std::make_shared<TorchMlirBackendData::Info>(scalar)) {
  PRINT_FUNCTION();
}
TorchMlirBackendData::TorchMlirBackendData(const at::Tensor &tensor,
                                           BackendDevice device, Shape shape)
    : BackendData(device, shape),
      info_(std::make_shared<TorchMlirBackendData::Info>(tensor)) {
  PRINT_FUNCTION();
}

BackendData::Handle TorchMlirBackendData::GetHandle() {
  return reinterpret_cast<int64_t>(this);
}

void TorchMlirBackendData::Assign(const BackendData &data) {
  const TorchMlirBackendData *torch_mlir_data =
      dynamic_cast<const TorchMlirBackendData *>(&data);
  TORCH_CHECK(torch_mlir_data,
              "Invalid Backend Data Pointer. Expected TorchMlirBackendData.");

  info_ = torch_mlir_data->info_;
}

bool TorchMlirBackendData::HasValue() const { return bool(info_); }

BackendData::Info *TorchMlirBackendData::mlir_info() const {
  return info_.get();
}

/**
 * Initialization/Teardown
 * */
void TorchMlirBackendImpl::PrepareToExit() const {}

/**
 * IR Tracing
 * */

const IrBuilder *TorchMlirBackendImpl::GetIrBuilder() const {
  static const IrBuilder *builder = new TorchMlirIrBuilder();
  return builder;
}

/**
 * Data Transfer
 * */

BackendDataPtr TorchMlirBackendImpl::MakeComputationDataFromTensor(
    const at::Tensor &tensor, const Shape &shape,
    const BackendDevice &device) const {
  PRINT_FUNCTION();
  return std::make_shared<TorchMlirBackendData>(tensor, device, shape);
}

BackendDataPtr TorchMlirBackendImpl::MakeComputationDataFromScalar(
    const at::Scalar &scalar, const BackendDevice &device) const {
  PRINT_FUNCTION();
  return std::make_shared<TorchMlirBackendData>(scalar, device);
}

BackendDataPtr
TorchMlirBackendImpl::CreateDataPlaceholder(const BackendDevice &device,
                                            const Shape &shape) const {
  PRINT_FUNCTION();
  return std::make_shared<TorchMlirBackendData>(device, shape);
}

BackendDataPtr
TorchMlirBackendImpl::GetComputationDataFromNode(const Node *node) const {
  PRINT_FUNCTION();
  const auto *device_data_node = dynamic_cast<const DeviceData *>(node);
  if (!device_data_node) {
    return nullptr;
  }
  return device_data_node->data();
}

at::Tensor TorchMlirBackendImpl::MakeTensorFromComputationData(
    const BackendDataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type) const {
  PRINT_FUNCTION();

  TorchMlirBackendData *torch_mlir_data =
      dynamic_cast<TorchMlirBackendData *>(data.get());
  TORCH_CHECK(torch_mlir_data,
              "Invalid Backend Data Pointer. Expected TorchMlirBackendData.");

  TorchMlirBackendData::Info *info =
      dynamic_cast<TorchMlirBackendData::Info *>(torch_mlir_data->mlir_info());
  TORCH_CHECK(
      info,
      "Invalid Backend Data Pointer. Expected TorchMlirBackendData::Info.");

  return info->tensor;
}

/**
 * Lowering, Compilation, Execution
 * */

std::unique_ptr<LoweringContext> TorchMlirBackendImpl::CreateLoweringContext(
    const std::string &name, BackendDevice device,
    c10::ArrayRef<const Node *> post_order,
    Util::EmissionMap emit_status) const {
  PRINT_FUNCTION();
  return std::make_unique<TorchMlirLoweringContext>(
      name, std::forward<BackendDevice>(device),
      std::forward<c10::ArrayRef<const Node *>>(post_order),
      std::forward<Util::EmissionMap>(emit_status));
}

std::unique_ptr<LoweringContext>
TorchMlirBackendImpl::CreateLoweringContext(const std::string &name,
                                            BackendDevice device) const {
  PRINT_FUNCTION();
  return std::make_unique<TorchMlirLoweringContext>(
      name, std::forward<BackendDevice>(device));
}

/**
 * Device Configuration
 * */

// Set or get the default device type.
// For backends used with virtual c10:: Devices, this configures what real
// device type the backend should use, and matters if the backend supports
// more than one type of real device.

// Specify which aten device should be used for eager fallback
// may change depending on current 'Default' DeviceType
at::DeviceType TorchMlirBackendImpl::EagerFallbackDeviceType() const {
  PRINT_FUNCTION();
  return at::DeviceType::CPU;
}

// Query all available backend devices
std::vector<BackendDevice> TorchMlirBackendImpl::GetBackendDevices() const {
  PRINT_FUNCTION();
  return {GetBackendDevice(c10::Device(c10::kLazy, 0)),
          GetBackendDevice(c10::Device(c10::kCPU, 0))};
}

// Map a particular c10:: device to a concrete backend device
// Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
// virtual devices, meaning they may map to a gpu, tpu, etc. behind the
// scenes. In the future, non-virtual c10:: devices may also use lazy tensors
// through a mode, in which case these APIs should still work, but should be
// identity mappings.
BackendDevice TorchMlirBackendImpl::GetBackendDevice(c10::Device device) const {
  PRINT_FUNCTION();
  return BackendDevice(GetDefaultDeviceType(), device.index());
}

int64_t TorchMlirBackendImpl::GetDefaultDeviceOrdinal() const {
  return default_device_ordinal;
}

void TorchMlirBackendImpl::SetDefaultDeviceOrdinal(int64_t ordinal) {
  default_device_ordinal = ordinal;
}

} // namespace lazy
} // namespace torch
