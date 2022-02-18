#include <iostream>

#include "mlir_lowering_context.h"
#include "../utils/exception.h"


namespace torch {
namespace lazy {

MlirLoweringContext::MlirLoweringContext(
    const std::string& name, BackendDevice device
) : LoweringContext(name, std::forward<BackendDevice>(device)) {}

MlirLoweringContext::MlirLoweringContext(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<torch::lazy::Node*> post_order,
    Util::EmissionMap emit_status
) : LoweringContext(
    name,
    std::forward<BackendDevice>(device),
    std::forward<c10::ArrayRef<torch::lazy::Node*>>(post_order),
    std::forward<Util::EmissionMap>(emit_status)
) {}

int MlirComputation::parameters_size() const {
    UNIMPLEMENTED_ERROR("MlirComputation::parameters_size");
}

const std::vector<torch::lazy::Shape>& MlirComputation::parameter_shapes() const {
    UNIMPLEMENTED_ERROR("MlirComputation::parameter_shapes");
}

const std::vector<std::string>& MlirComputation::parameter_names() const {
    UNIMPLEMENTED_ERROR("MlirComputation::parameter_names");
}

const torch::lazy::Shape& MlirComputation::result_shape() const {
    UNIMPLEMENTED_ERROR("MlirComputation::result_shape");
}


// Get the shape of the result tuple component, given by index.
torch::lazy::Shape MlirLoweringContext::GetResultShape(size_t index) const {
    UNIMPLEMENTED_ERROR("MlirLoweringContext::GetResultShape( " << index << " )");
}

// Adds the given output as a component of the result tuple and returns its
// assigned position within the tuple.
size_t MlirLoweringContext::AddResult(const torch::lazy::Output& output) {
    const torch::lazy::Node* node;
    auto it = emitted_outputs_.find(output);
    if (it == emitted_outputs_.end()) {
        node = output.node;

        auto post_order = Util::ComputePostOrder(node, &emit_status_);
        for (auto po_node : post_order) {
            // TODO: uncomment after lowering is implemented
            // bool ok = lowering_->Lower(node);
            // TORCH_CHECK(ok, "Failed to lower: ", node->ToString());
        }
        emitted_outputs_[output] = node;
    } else {
        node = it->second;
    }
    result_tuple_.emplace_back(node);
    return result_tuple_.size() - 1;
}

// Associates the given output with the input parameter of the given index and
// shape. Only used for the operator-by-operator execution, mostly for
// debugging purposes.
void MlirLoweringContext::AddParameter(
    const torch::lazy::Output& output,
    size_t index,
    const torch::lazy::Shape& shape,
    const std::string& name
) {
    UNIMPLEMENTED_ERROR("MlirLoweringContext::AddParameter");
}

// Build the computation capturing all the operations created with the
// embedded builder (returned by the builder() API).
ComputationPtr MlirLoweringContext::Build() {
    for (const torch::lazy::Node* output : result_tuple_) {

    }
    return std::make_shared<MlirComputation>();
}


}  // namespace lazy
}  // namespace torch
