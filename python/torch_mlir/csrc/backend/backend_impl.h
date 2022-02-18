#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/shape.h>

namespace torch {
namespace lazy {

class MlirBackendData : public torch::lazy::BackendData {
  public:
    struct Info;

    MlirBackendData(torch::lazy::BackendDevice device, torch::lazy::Shape shape);
    MlirBackendData(const at::Scalar& scalar, torch::lazy::BackendDevice device);
    MlirBackendData(const at::Tensor& tensor, torch::lazy::BackendDevice device, torch::lazy::Shape shape);

    virtual torch::lazy::BackendData::Handle GetHandle() override;
    
    virtual void Assign(const torch::lazy::BackendData& data) override;
    
    virtual bool HasValue() const override;
};

class MlirBackendImpl : public torch::lazy::BackendImplInterface {
public:
    /**
     * Initialization/Teardown
     * */
    virtual void PrepareToExit() const override;

    /**
     * Configuration
     * */
    // virtual void SetRngSeed(size_t seed) const = 0;

    /**
     * Data Transfer
     * */

    virtual torch::lazy::BackendDataPtr MakeComputationDataFromTensor(
        const at::Tensor& tensor,
        const torch::lazy::Shape& shape,
        const torch::lazy::BackendDevice& device
    ) const override;

    virtual torch::lazy::BackendDataPtr MakeComputationDataFromScalar(
        const at::Scalar& scalar,
        const torch::lazy::BackendDevice& device
    ) const override;

    virtual torch::lazy::BackendDataPtr CreateDataPlaceholder(
        const torch::lazy::BackendDevice& device, const torch::lazy::Shape& shape
    ) const override;

    virtual at::Tensor MakeTensorFromComputationData(
        const torch::lazy::BackendDataPtr data,
        c10::optional<at::ScalarType> logical_scalar_type
    ) const override;

    /**
     * Lowering, Compilation, Execution
     * */

    virtual std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
        const std::string& name,
        torch::lazy::BackendDevice device,
        c10::ArrayRef<torch::lazy::Node*> post_order,
        torch::lazy::Util::EmissionMap emit_status
    ) const override;

    virtual std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
        const std::string& name, torch::lazy::BackendDevice device
    ) const override;

    // TODO(whc) need to keep this?
    // virtual std::vector<std::string> GetCompilationDevices(
    //     const std::string& device, c10::ArrayRef<std::string> devices
    // ) const = 0;

    // virtual std::vector<torch::lazy::ComputationPtr> Compile(
    //     std::vector<torch::lazy::ComputationPtr> instances
    // ) const = 0;

    // virtual std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
    //     torch::lazy::Computation& computation,
    //     c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    //     const torch::lazy::BackendDevice& device
    // ) const = 0;

    /**
     * Device Configuration
     * */

    // Set or get the default device type.
    // For backends used with virtual c10:: Devices, this configures what real
    // device type the backend should use, and matters if the backend supports
    // more than one type of real device.

    // virtual std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType() const = 0;
    // virtual void SetDefaultDeviceType(std::string device_type) = 0;

    // Specify which aten device should be used for eager fallback
    // may change depending on current 'Default' DeviceType
    virtual at::DeviceType EagerFallbackDeviceType() const override;


    // Query all available backend devices
    virtual std::vector<torch::lazy::BackendDevice> GetBackendDevices() const override;

    // Map a particular c10:: device to a concrete backend device
    // Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
    // virtual devices, meaning they may map to a gpu, tpu, etc. behind the
    // scenes. In the future, non-virtual c10:: devices may also use lazy tensors
    // through a mode, in which case these APIs should still work, but should be
    // identity mappings.
    virtual torch::lazy::BackendDevice GetBackendDevice(c10::Device device) const override;


    /**
     * Debug/Metrics
     * */

    // virtual std::map<std::string, Metric> GetMetrics() const = 0;

    // virtual MemoryInfo GetMemoryInfo(const std::string& device) = 0;

    // virtual std::string GetComputationBackendText(
    //     const torch::lazy::ComputationPtr computation
    // ) const = 0;

};

}  // lazy
}  // torch
