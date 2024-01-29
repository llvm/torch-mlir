#include "jit_utils.h"

#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <ATen/core/type_factory.h>

namespace torch {
namespace jit {

void ConvertScalarImplicit(std::shared_ptr<Graph> &graph) {
  DepthFirstGraphNodeIterator it(graph);
  for (auto *node = it.next(); node != nullptr; node = it.next()) {
    if (node->kind() != c10::aten::ScalarImplicit) {
      continue;
    }

    auto input = node->input(0);
    auto scalar_type = input->type()->cast<c10::TensorType>()->scalarType();
    TORCH_CHECK(scalar_type, "scalar type is not defined for input value");

    NodeKind node_type;
    TypePtr output_type;
    if (c10::isIntegralType(*scalar_type, true)) {
      node_type = c10::aten::IntImplicit;
      output_type = IntType::get();
    } else if (c10::isFloatingType(*scalar_type)) {
      node_type = c10::aten::FloatImplicit;
      output_type = FloatType::get();
    } else {
      throw std::runtime_error("Expected isIntegralType or isFloatingType");
    }

    Value *output = graph->create(node_type, {input})
                        ->insertBefore(node)
                        ->output()
                        ->setType(output_type);
    node->output()->replaceAllUsesWith(output);
    node->destroy();
  }
}

} // namespace jit
} // namespace torch
