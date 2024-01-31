#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Convert ScalarImplicit to IntImplicit or FloatImplicit.
TORCH_API void ConvertScalarImplicit(std::shared_ptr<Graph> &graph);

} // namespace jit
} // namespace torch
