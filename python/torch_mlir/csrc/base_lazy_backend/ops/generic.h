#pragma once

#include "../mlir_node.h"

namespace torch {
namespace lazy {

// Generic IR Node implementation for nodes which can simply be described by a
// specific OpKind and a lowering function. IR nodes carrying
// metadata should not be using this class TORCH_API (and have the metadata
// captured by the LowerFn), but they should instead create a dedicated IR node.
// Doing the former would limit IR introspection.
class TORCH_API Generic : public TorchMlirNode {
 public:
  Generic(
      OpKind op,
      OpList operands,
      Shape shape,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

 private:
  hash_t hash_seed_;
};

} // namespace lazy
} // namespace torch
