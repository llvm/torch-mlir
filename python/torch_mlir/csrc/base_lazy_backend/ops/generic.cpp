#include <torch/csrc/lazy/ts_backend/ops/generic.h>

namespace torch {
namespace lazy {

Generic::Generic(
    OpKind op,
    OpList operands,
    Shape shape,
    size_t num_outputs,
    hash_t hash_seed)
    : TorchMlirNode(op, operands, {std::move(shape)}, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

} // namespace lazy
} // namespace torch
