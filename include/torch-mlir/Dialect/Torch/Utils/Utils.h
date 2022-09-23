//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHMLIR_DIALECT_TORCH_UTILS_H
#define TORCHMLIR_DIALECT_TORCH_UTILS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

namespace mlir {
namespace torch {
namespace Torch {

int64_t toPositiveDim(int64_t dim, int64_t inputRank);
bool isValidDim(int64_t dim, int64_t inputRank);
bool getListConstructElements(Value v, SmallVectorImpl<Value> &elems);
/// Returns the index indicated by `v` for a list of given `length`.
/// If the index is negative, it is adjusted to `length` + `v`.
/// `None` is returned the index is not an integer in the range [0,`length).
llvm::Optional<int64_t> matchLegalConstantIndexIntoListOfSize(Value v,
                                                              int64_t length);
torch_upstream::ScalarType getScalarTypeForType(Type type);
Type getTypeForScalarType(
    MLIRContext *context, torch_upstream::ScalarType dtypeInt,
    mlir::IntegerType::SignednessSemantics signedness = IntegerType::Signed);

Type getTypeForTorchType(
    MLIRContext *context, Type type,
    mlir::IntegerType::SignednessSemantics signedness = IntegerType::Signed);

Type getTorchTypeForScalarType(MLIRContext *context,
                               torch_upstream::ScalarType dtypeInt);

Value getDtypeIntValueForType(PatternRewriter &rewriter, Location loc,
                              Type dtype);
// Helper to convert a tensor to a specific scalar type.
Value convertTensorToDtype(PatternRewriter &rewriter, Location loc, Value input,
                           Type dtype);

bool isBuiltInType(Type type);

// Helper funtion to get rank of `Base tensor type`.
// -1 is returned if the tensorRank can't be determined.
int getTensorRank(Value tensor);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_UTILS_H
