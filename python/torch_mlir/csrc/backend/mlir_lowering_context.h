#pragma once

#include <vector>

#include <torch/csrc/lazy/backend/lowering_context.h>

namespace torch {
namespace lazy {

class MlirComputation : public torch::lazy::Computation {
  public:
    int parameters_size() const override;

    virtual const std::vector<torch::lazy::Shape>& parameter_shapes() const override;

    virtual const std::vector<std::string>& parameter_names() const override;

    virtual const torch::lazy::Shape& result_shape() const override;
};

class MlirLoweringContext : public torch::lazy::LoweringContext {
  public:

    MlirLoweringContext(const std::string& name, torch::lazy::BackendDevice device);
    MlirLoweringContext(const std::string& name,
                        torch::lazy::BackendDevice device,
                        c10::ArrayRef<torch::lazy::Node*> post_order,
                        torch::lazy::Util::EmissionMap emit_status);
    
    // Get the shape of the result tuple component, given by index.
    virtual torch::lazy::Shape GetResultShape(size_t index) const override;

    // Adds the given output as a component of the result tuple and returns its
    // assigned position within the tuple.
    virtual size_t AddResult(const torch::lazy::Output& output) override;

    // Associates the given output with the input parameter of the given index and
    // shape. Only used for the operator-by-operator execution, mostly for
    // debugging purposes.
    virtual void AddParameter(const torch::lazy::Output& output,
                      size_t index,
                      const torch::lazy::Shape& shape,
                      const std::string& name) override;

    // Build the computation capturing all the operations created with the
    // embedded builder (returned by the builder() API).
    virtual torch::lazy::ComputationPtr Build() override;

  private:
    std::vector<const torch::lazy::Node*> result_tuple_;
    torch::lazy::OutputMap<const torch::lazy::Node*> emitted_outputs_;
};

}  // namespace lazy
}  // namespace torch
