//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir::torch::torch_upstream;

namespace mlir {
namespace torch {
namespace Torch {

int64_t toPositiveDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 ? dim : dim + inputRank;
}

bool isValidDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 && dim < inputRank;
}

bool getListConstructElements(Value v, SmallVectorImpl<Value> &elems) {
  auto listConstruct = v.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return false;
  elems = llvm::to_vector<4>(listConstruct.elements());
  return true;
}

ScalarType getScalarTypeForType(Type type) {
  if (type.isa<Float32Type>())
    return ScalarType::Float;
  if (type.isa<Float64Type>())
    return ScalarType::Double;
  if (type.isSignedInteger(64))
    return ScalarType::Long;
  if (type.isSignedInteger(32))
    return ScalarType::Int;
  if (type.isUnsignedInteger(1))
    return ScalarType::Bool;
  llvm::report_fatal_error("unhandled type for getScalarTypeForType");
}

Value getDtypeIntValueForType(PatternRewriter &rewriter, Location loc,
                              Type dtype) {
  int intType = (int)getScalarTypeForType(dtype);
  return rewriter.create<ConstantIntOp>(loc,
                                        rewriter.getI64IntegerAttr(intType));
}

// Helper to convert a tensor to a specific scalar type.
Value convertTensorToDtype(PatternRewriter &rewriter, Location loc, Value input,
                           Type dtype) {
  BaseTensorType origType = input.getType().cast<BaseTensorType>();
  Type newType = origType.getWithSizesAndDtype(origType.getSizes(), dtype);
  // `convertIntVal` contains the corresponding integer for the dtype which is
  // used by the aten.to.dtype op.
  Value convertIntVal = getDtypeIntValueForType(rewriter, loc, dtype);
  Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  Value converted = rewriter.create<AtenToDtypeOp>(
      loc, newType, input, convertIntVal, falseVal, falseVal, noneVal);
  return converted;
}

} // namespace Torch
} // namespace torch
} // namespace mlir
