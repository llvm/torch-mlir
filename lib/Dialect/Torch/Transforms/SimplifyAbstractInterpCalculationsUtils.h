//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCH_TRANSFORMS_SIMPLIFY_ABSTRACT_INTERP_CALCULATIONS_UTILS_H
#define TORCHMLIR_DIALECT_TORCH_TRANSFORMS_SIMPLIFY_ABSTRACT_INTERP_CALCULATIONS_UTILS_H

#include "mlir/IR/PatternMatch.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace mlir {
namespace torch {
namespace Torch {

// Updates the type of result `resultNum` of both `calculateOp` and the torch op
// being wrapped by `calculateOp` to the type `newResultType`.
LogicalResult updateCalculateOpResultTypes(Operation *calculateOp,
                                           int resultNum, Type newResultType,
                                           PatternRewriter &rewriter);

void populateFoldPrimUncheckedCastOpPattern(RewritePatternSet &patterns,
                                            MLIRContext *context);
void populateFullyUnrollPrimLoopOpPattern(RewritePatternSet &patterns,
                                          MLIRContext *context);
void populateAbstractlyInterpretListOpsWithinABlockPattern(
    RewritePatternSet &patterns, MLIRContext *context);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_SIMPLIFY_ABSTRACT_INTERP_CALCULATIONS_UTILS_H
