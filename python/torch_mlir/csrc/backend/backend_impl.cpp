//===- backend_impl.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/backend_impl.cpp
//===----------------------------------------------------------------------===//

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/shape.h>

#include "../utils/debug.h"
#include "../utils/exception.h"
#include "backend_impl.h"
#include "mlir_lowering_context.h"

namespace torch {
namespace lazy {

MlirBackendData::MlirBackendData(BackendDevice device, Shape shape)
    : BackendData(device, shape) {
  PRINT_FUNCTION();
  auto info = std::make_shared<MlirBackendData::Info>();
  SetInfo(info);
}
MlirBackendData::MlirBackendData(const at::Scalar& scalar, BackendDevice device)
    : BackendData(device, Shape(scalar.type(), {})) {
  PRINT_FUNCTION();
  auto info = std::make_shared<MlirBackendData::Info>(scalar);
  SetInfo(info);
}
MlirBackendData::MlirBackendData(
    const at::Tensor& tensor, BackendDevice device, Shape shape)
    : BackendData(device, shape) {
  PRINT_FUNCTION();
  auto info = std::make_shared<MlirBackendData::Info>(tensor);
  SetInfo(info);
}

BackendData::Handle MlirBackendData::GetHandle() {
  return reinterpret_cast<int64_t>(this);
}

void MlirBackendData::Assign(const BackendData& data) {
  MlirBackendData::Info* info =
      dynamic_cast<MlirBackendData::Info*>(data.info());
  TORCH_CHECK(
      info, "Invalid Backend Data Pointer. Expected MlirBackendData::Info.");
  auto new_info = std::make_shared<MlirBackendData::Info>(*info);
  SetInfo(new_info);
}

bool MlirBackendData::HasValue() const { return bool(info()); }

/**
 * Initialization/Teardown
 * */
void MlirBackendImpl::PrepareToExit() const {}

/**
 * Data Transfer
 * */

BackendDataPtr MlirBackendImpl::MakeComputationDataFromTensor(
    const at::Tensor& tensor, const Shape& shape,
    const BackendDevice& device) const {
  PRINT_FUNCTION();
  return std::make_shared<MlirBackendData>(tensor, device, shape);
}

BackendDataPtr MlirBackendImpl::MakeComputationDataFromScalar(
    const at::Scalar& scalar, const BackendDevice& device) const {
  PRINT_FUNCTION();
  return std::make_shared<MlirBackendData>(scalar, device);
}

BackendDataPtr MlirBackendImpl::CreateDataPlaceholder(
    const BackendDevice& device, const Shape& shape) const {
  PRINT_FUNCTION();
  return std::make_shared<MlirBackendData>(device, shape);
}

at::Tensor MlirBackendImpl::MakeTensorFromComputationData(
    const BackendDataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type) const {
  PRINT_FUNCTION();
  MlirBackendData::Info* info =
      dynamic_cast<MlirBackendData::Info*>(data->info());
  TORCH_CHECK(
      info, "Invalid Backend Data Pointer. Expected MlirBackendData::Info.");
  return info->tensor;
}

/**
 * Lowering, Compilation, Execution
 * */

std::unique_ptr<LoweringContext> MlirBackendImpl::CreateLoweringContext(
    const std::string& name, BackendDevice device,
    c10::ArrayRef<Node*> post_order, Util::EmissionMap emit_status) const {
  PRINT_FUNCTION();
  return std::make_unique<MlirLoweringContext>(
      name, std::forward<BackendDevice>(device),
      std::forward<c10::ArrayRef<Node*>>(post_order),
      std::forward<Util::EmissionMap>(emit_status));
}

std::unique_ptr<LoweringContext> MlirBackendImpl::CreateLoweringContext(
    const std::string& name, BackendDevice device) const {
  PRINT_FUNCTION();
  return std::make_unique<MlirLoweringContext>(
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
at::DeviceType MlirBackendImpl::EagerFallbackDeviceType() const {
  PRINT_FUNCTION();
  return at::DeviceType::CPU;
}

// Query all available backend devices
std::vector<BackendDevice> MlirBackendImpl::GetBackendDevices() const {
  PRINT_FUNCTION();
  return {
      GetBackendDevice(c10::Device(c10::kLazy, 0)),
      GetBackendDevice(c10::Device(c10::kCPU, 0))};
}

// Map a particular c10:: device to a concrete backend device
// Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
// virtual devices, meaning they may map to a gpu, tpu, etc. behind the
// scenes. In the future, non-virtual c10:: devices may also use lazy tensors
// through a mode, in which case these APIs should still work, but should be
// identity mappings.
BackendDevice MlirBackendImpl::GetBackendDevice(c10::Device device) const {
  PRINT_FUNCTION();
  return BackendDevice(GetDefaultDeviceType(), device.index());
}

} // namespace lazy
} // namespace torch
