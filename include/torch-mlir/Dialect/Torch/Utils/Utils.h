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
std::optional<int64_t> matchLegalConstantIndexIntoListOfSize(Value v,
                                                             int64_t length);
torch_upstream::ScalarType getScalarTypeForType(Type type);
FailureOr<Type> getTypeForScalarType(
    MLIRContext *context, torch_upstream::ScalarType dtypeInt,
    mlir::IntegerType::SignednessSemantics signedness = IntegerType::Signed);

Type getTypeForTorchType(
    MLIRContext *context, Type type,
    mlir::IntegerType::SignednessSemantics signedness = IntegerType::Signed);

FailureOr<Type> getTorchTypeForScalarType(MLIRContext *context,
                                          torch_upstream::ScalarType dtypeInt);

// This is the type rule used for deciding dtype for:
// 1. A new tensor created from given data.
// 2. The scalar type for type promotion when a scalar is an operand of a tensor
// operation (such as AtenMulScalarOp, AtenAddScalarOp etc)
// If the data is floating-point, the `dtype` is inferred to be the
// default dtype, see `torch.get_default_dtype`.
Type getDefaultDtypeForTorchScalar(Type type);

// This is the type rule used for deciding builtin type for:
// 1. The dtype of the result tensor when converting a Scalar into a Tensor like
// PrimNumToTensorScalarOp.
// 2. The scalar type for type promotion when a scalar is an operand of scalar
// only operation like AtenAddOp.
Type getBuiltInTypeForTorchScalar(Type type);

Value getDtypeIntValueForType(PatternRewriter &rewriter, Location loc,
                              Type dtype);
// Helper to convert a tensor to a specific scalar type.
Value convertTensorToDtype(PatternRewriter &rewriter, Location loc, Value input,
                           Type dtype);

bool isBuiltInType(Type type);

// Helper funtion to get rank of `Base tensor type`.
// std::nullopt is returned if the tensorRank can't be determined.
std::optional<unsigned> getTensorRank(Value tensor);

bool isViewLikeOp(Operation *op);

Value getConstantWithGivenDtypeAndValue(PatternRewriter &rewriter, Location loc,
                                        float value, Type dtype);

// Return the number of elements of a tensor if the shape is static; otherwise,
// return -1.
int64_t getNumberOfElements(RankedTensorType inputType);

SmallVector<int64_t> makeShapeLLVMCompatible(ArrayRef<int64_t> shape);
SmallVector<int64_t> makeShapeTorchCompatible(ArrayRef<int64_t> shape);

// Helper function to squeeze the input tensor at given dim.
// Return the squeezed tensor or failure.
FailureOr<Value> squeezeTensor(PatternRewriter &rewriter, Operation *op,
                               Location loc, int64_t dim, Value input);

// Helper function to unsqueeze the input tensor at given dim.
// Return the unsqueezed tensor or failure.
FailureOr<Value> unsqueezeTensor(PatternRewriter &rewriter, Operation *op,
                                 Value input, Value dim);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_UTILS_H
