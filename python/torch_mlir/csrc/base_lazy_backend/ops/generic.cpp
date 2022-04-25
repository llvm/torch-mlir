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

Generic::Generic(
    OpKind op,
    OpList operands,
    const std::function<Shape()>& shape_fn,
    size_t num_outputs,
    hash_t hash_seed)
    : TorchMlirNode(op, operands, shape_fn, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(
    OpKind op,
    OpList operands,
    size_t num_outputs,
    hash_t hash_seed)
    : TorchMlirNode(op, operands, num_outputs, hash_seed), hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : TorchMlirNode(op, std::move(shape), num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

} // namespace lazy
} // namespace torch
