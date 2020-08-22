//===- torch_util.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "torch_util.h"

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

namespace torch_mlir {
namespace util {

at::Tensor Zeros(at::IntArrayRef sizes, at::ScalarType type) {
  return at::zeros(sizes, type);
}

at::Tensor CopyTensor(const at::Tensor &ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

// Same as above, with an additional cast.
at::Tensor CopyTensor(const at::Tensor &ref, at::ScalarType dest_type) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false,
                /*copy=*/true);
}

at::ScalarType GetScalarType(at::Scalar scalar) {
  if (scalar.isFloatingPoint()) {
    return at::kDouble;
  } else if (scalar.isIntegral(/*includeBool=*/false)) {
    return at::kLong;
  } else if (scalar.isBoolean()) {
    return at::kBool;
  } else if (scalar.isComplex()) {
    return at::kComplexDouble;
  }
  assert(0 && "Unknown type for scalar");
}

} // namespace util
} // namespace torch_mlir
