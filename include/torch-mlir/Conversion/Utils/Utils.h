//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_UTILS_H
#define TORCHMLIR_CONVERSION_UTILS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace torch {
namespace Torch {

LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                          PatternRewriter &rewriter);

LogicalResult checkNotNone(PatternRewriter &rewriter, Operation *op, Value v);

Value toPositiveDimDynamic(OpBuilder &b, Location loc, Value dim,
                           Value inputRank);

void assertIsValidDim(OpBuilder &b, Location loc, Value dim, Value inputRank);

bool isConstantIntListMatching(Value value, SmallVectorImpl<int64_t> &expects);

void checkDimEqualHelper(OpBuilder &b, Location loc, Value lhsDim,
                         Value rhsDim);

// Creates a tensor with required `sizes` and `elemTy` and fills it with
// initElem.
Value createInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                       Type elemTy, Value initElem);

Value createZeroInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                           Type elemTy);

Value castIntToIndex(OpBuilder &b, Location loc, Value v);

Value castIndexToInt64(OpBuilder &b, Location loc, Value idx);

SmallVector<Value>
castIntVectorToIndexVector(OpBuilder &b, Location loc,
                           SmallVectorImpl<Value> &intValues);

SmallVector<Value>
castIndexVectorToInt64Vector(OpBuilder &b, Location loc,
                             SmallVectorImpl<Value> &indexValues);

Value getDimOp(OpBuilder &b, Location loc, Value v, int dim);

SmallVector<Value> getTensorSizesUntilDim(OpBuilder &b, Location loc,
                                          Value tensor, int dim);

SmallVector<Value> getTensorSizes(OpBuilder &b, Location loc, Value tensor);

Value getTensorSize(OpBuilder &b, Location loc, Value tensor);

// Creates a constant of type `elemType` with value `val`.
Value getConstant(OpBuilder &b, Location loc, int64_t val, Type elemType);

SmallVector<Value> getAsConstantIntValues(OpBuilder &b, Location loc,
                                          SmallVectorImpl<int64_t> &ints);

SmallVector<Value> getAsConstantIndexValues(OpBuilder &b, Location loc,
                                            SmallVectorImpl<int64_t> &ints);

// This is a temporary solution to deal with types that are not fully supported
// like list, dict. For those container tyes, this helper can be used to
// convert their elements to valid target type.
// TODO: remove this when list gets full support.
SmallVector<Value> getTypeConvertedValues(OpBuilder &b, Location loc,
                                          TypeConverter *converter,
                                          SmallVectorImpl<Value> &vs);

mlir::RankedTensorType GetTypeFromTensorShape(llvm::ArrayRef<int64_t> shape,
                                              mlir::Type elementType,
                                              mlir::Attribute encoding = {});

// Convert a scalar value to the target type. The scalar value can be an element
// from a tensor or a scalar in the pytorch dialect. Both the scalar and dtype
// should be converted builtin types.
Value convertScalarToDtype(OpBuilder &b, Location loc, Value scalar, Type dtype,
                           std::optional<Type> srcOriginalDtype = std::nullopt);

Value toPositiveValidDim(ConversionPatternRewriter &rewriter, Location loc,
                         Value torchOptionalInt, Value builtinInt,
                         Value defaultValue, Value dimSize);

// Checks whether the `inputA` and `inputB` are broadcast compatible or not. If
// yes, then computes the final broadcast shape.
void computeBroadcastShape(ConversionPatternRewriter &rewriter, Location loc,
                           Value inputA, Value inputB,
                           SmallVector<int64_t> &resultShape,
                           SmallVector<Value> &resultShapeValue);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_UTILS_H
