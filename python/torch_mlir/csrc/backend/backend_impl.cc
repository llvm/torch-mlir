#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/shape.h>

#include "backend_impl.h"
#include "mlir_lowering_context.h"
#include "../utils/exception.h"

namespace torch {
namespace lazy {

struct MlirBackendData::Info : public BackendData::Info {
    at::Tensor tensor;
    c10::optional<at::Scalar> scalar;

    Info() {}
    Info(const Info& other) :
        tensor{other.tensor}, scalar{other.scalar} {}
    Info(const at::Tensor& tensor) : tensor{tensor} {}
    Info(const at::Scalar& scalar) : scalar{scalar} {}
};

MlirBackendData::MlirBackendData(BackendDevice device, Shape shape) :
    BackendData(device, shape) {
    auto info = std::make_shared<MlirBackendData::Info>();
    SetInfo(info);
}
MlirBackendData::MlirBackendData(const at::Scalar& scalar, BackendDevice device) :
    BackendData(device, torch::lazy::Shape(scalar.type(), {})) {
    auto info = std::make_shared<MlirBackendData::Info>(scalar);
    SetInfo(info);
}
MlirBackendData::MlirBackendData(const at::Tensor& tensor, BackendDevice device, Shape shape) :
    BackendData(device, shape) {
    auto info = std::make_shared<MlirBackendData::Info>(tensor);
    SetInfo(info);
}

BackendData::Handle MlirBackendData::GetHandle() { return reinterpret_cast<int64_t>(this); }

void MlirBackendData::Assign(const BackendData& data) {
    MlirBackendData::Info* info = 
        dynamic_cast<MlirBackendData::Info*>(data.info());
    TORCH_CHECK(
        info, "Invalid Backend Data Pointer. Expected MlirBackendData::Info."
    );
    auto new_info = std::make_shared<MlirBackendData::Info>(*info);
    SetInfo(new_info);
}

bool MlirBackendData::HasValue() const {
    return bool(info());
}

/**
 * Initialization/Teardown
 * */
void MlirBackendImpl::PrepareToExit() const {}

/**
 * Data Transfer
 * */

BackendDataPtr MlirBackendImpl::MakeComputationDataFromTensor(
    const at::Tensor& tensor,
    const Shape& shape,
    const BackendDevice& device
) const {
    return std::make_shared<MlirBackendData>(tensor, device, shape);
}

BackendDataPtr MlirBackendImpl::MakeComputationDataFromScalar(
    const at::Scalar& scalar,
    const torch::lazy::BackendDevice& device
) const {
    return std::make_shared<MlirBackendData>(scalar, device);
}

BackendDataPtr MlirBackendImpl::CreateDataPlaceholder(
    const BackendDevice& device, const Shape& shape
) const {
    return std::make_shared<MlirBackendData>(device, shape);
}

at::Tensor MlirBackendImpl::MakeTensorFromComputationData(
    const BackendDataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type
) const {
    MlirBackendData::Info* info =
        dynamic_cast<MlirBackendData::Info*>(data->info());
    TORCH_CHECK(
        info, "Invalid Backend Data Pointer. Expected MlirBackendData::Info."
    );
    return info->tensor;
}

/**
 * Lowering, Compilation, Execution
 * */

std::unique_ptr<LoweringContext> MlirBackendImpl::CreateLoweringContext(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<torch::lazy::Node*> post_order,
    Util::EmissionMap emit_status
) const {
    return std::make_unique<MlirLoweringContext>(
        name,
        std::forward<BackendDevice>(device),
        std::forward<c10::ArrayRef<torch::lazy::Node*>>(post_order),
        std::forward<Util::EmissionMap>(emit_status)
    );
}

std::unique_ptr<LoweringContext> MlirBackendImpl::CreateLoweringContext(
    const std::string& name, BackendDevice device
) const {
    return std::make_unique<MlirLoweringContext>(
        name, std::forward<BackendDevice>(device)
    );
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
    return at::DeviceType::CPU;
}


// Query all available backend devices
std::vector<BackendDevice> MlirBackendImpl::GetBackendDevices() const {
    return {
        GetBackendDevice(c10::Device(c10::kCPU, 0)),
        GetBackendDevice(c10::Device(c10::kLazy, 0))
    };
}

// Map a particular c10:: device to a concrete backend device
// Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
// virtual devices, meaning they may map to a gpu, tpu, etc. behind the
// scenes. In the future, non-virtual c10:: devices may also use lazy tensors
// through a mode, in which case these APIs should still work, but should be
// identity mappings.
BackendDevice MlirBackendImpl::GetBackendDevice(c10::Device device) const {
    return torch::lazy::BackendDevice(GetDefaultDeviceType(), device.index());
}

}  // lazy
}  // torch
