//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

namespace mlir {
namespace torch {
namespace torch_to_stablehlo {

// Convert a scalar type to the corresponding builtin type in the
// stablehlo backend.
FailureOr<Type>
getBackendTypeForScalarType(MLIRContext *context,
                            torch_upstream::ScalarType dtypeInt);

} // namespace torch_to_stablehlo
} // namespace torch
} // namespace mlir
