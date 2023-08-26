//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

namespace mlir {
namespace torch {
namespace Torch {

LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                          PatternRewriter &rewriter) {
  // Check the value tensor is ranked as expected by Linalg.
  // TODO: Remove this check but use a separate verification pass to verify the
  // invariants expected by later passes.
  auto isValidLinalgType = [](Type type) {
    if (type.isa<NonValueTensorType>())
      return false;
    auto tensor = type.dyn_cast<ValueTensorType>();
    return !tensor ||
           tensor.toBuiltinTensor().dyn_cast_or_null<RankedTensorType>();
  };

  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

LogicalResult checkNotNone(PatternRewriter &rewriter, Operation *op, Value v) {
  Type type = v.getType();
  if (type.isa<OptionalType>() || type.isa<Torch::NoneType>() ||
      type.isa<mlir::NoneType>())
    return rewriter.notifyMatchFailure(op, "unimplemented None type arg");
  return success();
}

// Generate IR: dim = dim >= 0 ? dim : dim + inputRank
Value toPositiveDimDynamic(OpBuilder &b, Location loc, Value dim,
                           Value inputRank) {
  assert(dim.getType().isa<IntegerType>() &&
         "dim arg of toPositiveDim must be integer type");
  Value dimAddInputRank = b.create<arith::AddIOp>(loc, dim, inputRank);
  Value cst0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(inputRank.getType()));
  Value predDimGEZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, dim, cst0);
  Value dimInt =
      b.create<arith::SelectOp>(loc, predDimGEZero, dim, dimAddInputRank);
  return dimInt;
}

// Generate IR: assert(dim >= 0 && dim < inputRank)
void assertIsValidDim(OpBuilder &b, Location loc, Value dim, Value inputRank) {
  assert(dim.getType().isa<IntegerType>() &&
         "dim arg of assertIsValidDim must be integer type");
  Value cst0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(inputRank.getType()));
  Value predGEZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, dim, cst0);
  b.create<cf::AssertOp>(
      loc, predGEZero, b.getStringAttr("dim must be greater or equal to zero"));
  Value predLTInputRank =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, dim, inputRank);
  b.create<cf::AssertOp>(loc, predLTInputRank,
                         b.getStringAttr("dim must be smaller than inputRank"));
}

// Hack to deal with the Torch list type arguments which is not supported end
// to end. Constant values can be be extracted directly and non constant
// list values are not supported.
// TODO: loose this constraint when properly support list type
bool isConstantIntListMatching(Value value, SmallVectorImpl<int64_t> &expects) {
  SmallVector<int64_t> intValues;
  if (!matchPattern(value, m_TorchListOfConstantInts(intValues)))
    return false;

  if (intValues.size() != expects.size())
    return false;

  for (auto it : llvm::zip(intValues, expects)) {
    if (std::get<0>(it) != std::get<1>(it))
      return false;
  }
  return true;
}

void checkDimEqualHelper(OpBuilder &b, Location loc, Value lhsDim,
                         Value rhsDim) {
  Type lhsType = lhsDim.getType();
  Type rhsType = rhsDim.getType();
  auto checkIntOrIndex = [](Type type) {
    assert(type.isa<IntegerType>() ||
           type.isa<IndexType>() && "must be either integer or index type");
  };
  checkIntOrIndex(lhsType);
  checkIntOrIndex(rhsType);
  Value lhsDimInt =
      lhsType.isIndex() ? castIndexToInt64(b, loc, lhsDim) : lhsDim;
  Value rhsDimInt =
      rhsType.isIndex() ? castIndexToInt64(b, loc, rhsDim) : rhsDim;
  Value contractingDimEqual = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lhsDimInt, rhsDimInt);
  b.create<cf::AssertOp>(loc, contractingDimEqual,
                         b.getStringAttr("mismatching contracting dimension"));
}

// Creates a tensor with required `sizes` and `elemTy` and fills it with
// initElem.
Value createInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                       Type elemTy, Value initElem) {
  Value initTensor =
      b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(sizes), elemTy);
  return b.create<linalg::FillOp>(loc, initElem, initTensor).getResult(0);
}

Value createZeroInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                           Type elemTy) {
  Value initTensor =
      b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(sizes), elemTy);
  RankedTensorType type = initTensor.getType().cast<RankedTensorType>();
  Value c0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(type.getElementType()));
  return b.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
}

Value castIntToIndex(OpBuilder &b, Location loc, Value v) {
  assert(v.getType().isa<IntegerType>() && "must be called with integer type");
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

Value castIndexToInt64(OpBuilder &b, Location loc, Value idx) {
  assert(idx.getType().isa<IndexType>() && "must be called with integer type");
  return b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
}

SmallVector<Value>
castIntVectorToIndexVector(OpBuilder &b, Location loc,
                           SmallVectorImpl<Value> &intValues) {
  SmallVector<Value> indexValues;
  for (Value v : intValues)
    indexValues.push_back(castIntToIndex(b, loc, v));
  return indexValues;
}

SmallVector<Value>
castIndexVectorToInt64Vector(OpBuilder &b, Location loc,
                             SmallVectorImpl<Value> &indexValues) {
  SmallVector<Value> intValues;
  for (Value v : indexValues)
    intValues.push_back(castIndexToInt64(b, loc, v));
  return intValues;
}

Value getDimOp(OpBuilder &b, Location loc, Value v, int dim) {
  return b.createOrFold<tensor::DimOp>(loc, v, dim);
}

SmallVector<Value> getTensorSizesUntilDim(OpBuilder &b, Location loc,
                                          Value tensor, int dim) {
  RankedTensorType type = tensor.getType().cast<RankedTensorType>();
  assert(dim < type.getRank() &&
         "The given dim must be smaller than tensor rank");
  (void)type;
  SmallVector<Value> sizes;
  for (int i = 0; i <= dim; i++)
    sizes.push_back(getDimOp(b, loc, tensor, i));
  return sizes;
}

SmallVector<Value> getTensorSizes(OpBuilder &b, Location loc, Value tensor) {
  RankedTensorType type = tensor.getType().cast<RankedTensorType>();
  return getTensorSizesUntilDim(b, loc, tensor, type.getRank() - 1);
}

Value getTensorSize(OpBuilder &b, Location loc, Value tensor) {
  SmallVector<Value> sizes(getTensorSizes(b, loc, tensor));
  Value productResult = b.create<arith::ConstantOp>(loc, b.getIndexAttr(1));
  for (Value size : sizes)
    productResult = b.create<arith::MulIOp>(loc, productResult, size);
  return castIndexToInt64(b, loc, productResult);
}

// Creates a constant of type `elemType` with value `val`.
Value getConstant(OpBuilder &b, Location loc, int64_t val, Type elemType) {
  TypedAttr attr = {};
  if (elemType.isa<mlir::FloatType>())
    attr = b.getFloatAttr(elemType, val);
  if (elemType.isa<mlir::IndexType>())
    attr = b.getIndexAttr(val);
  if (elemType.isa<mlir::IntegerType>())
    attr = b.getIntegerAttr(
        elemType, APInt(elemType.cast<IntegerType>().getWidth(), val));
  if (!attr)
    return nullptr;
  return b.create<arith::ConstantOp>(loc, elemType, attr);
}

SmallVector<Value> getAsConstantIntValues(OpBuilder &b, Location loc,
                                          SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<arith::ConstantOp>(loc,
                                       b.getIntegerAttr(b.getI64Type(), val));
  }));
}

SmallVector<Value> getAsConstantIndexValues(OpBuilder &b, Location loc,
                                            SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<arith::ConstantOp>(loc, b.getIndexAttr(val));
  }));
}

// This is a temporary solution to deal with types that are not fully supported
// like list, dict. For those container tyes, this helper can be used to
// convert their elements to valid target type.
// TODO: remove this when list gets full support.
SmallVector<Value> getTypeConvertedValues(OpBuilder &b, Location loc,
                                          const TypeConverter *converter,
                                          SmallVectorImpl<Value> &vs) {
  return llvm::to_vector<4>(llvm::map_range(vs, [&](Value v) {
    return converter->materializeTargetConversion(
        b, loc, converter->convertType(v.getType()), v);
  }));
}

mlir::RankedTensorType GetTypeFromTensorShape(llvm::ArrayRef<int64_t> shape,
                                              mlir::Type elementType,
                                              mlir::Attribute encoding) {
  return mlir::RankedTensorType::get(makeShapeLLVMCompatible(shape),
                                     elementType, encoding);
}

// Convert a scalar value to the target type. The scalar value can be an element
// from a tensor or a scalar in the pytorch dialect. Both the scalar and dtype
// should be converted builtin types.
Value convertScalarToDtype(OpBuilder &b, Location loc, Value scalar, Type dtype,
                           std::optional<Type> srcOriginalDtype) {
  Type scalarType = scalar.getType();
  if (scalarType == dtype)
    return scalar;

  auto isByteOrChar = [](Type type) {
    if (auto integerTy = type.dyn_cast<mlir::IntegerType>()) {
      return integerTy.getWidth() == 8;
    }
    return false;
  };

  // We only support conversion from Byte or Char scalarType not to Byte or Char
  // dtype.
  if (isByteOrChar(dtype)) {
    mlir::emitError(loc) << "unsupported: conversion to byte or char type for "
                            "convertScalarToDtype "
                         << scalarType << "(scalar type) -> " << dtype
                         << "(dtype)";
    return nullptr;
  }

  // If the dtype is i1, i.e., a boolean type.
  if (dtype.isSignlessInteger(1)) {
    Type scalarType = scalar.getType();
    Value cstZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(scalarType));
    if (scalarType.isa<mlir::FloatType>()) {
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, scalar,
                                     cstZero);
    } else if (scalarType.isa<mlir::IntegerType>()) {
      return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, scalar,
                                     cstZero);
    } else {
      mlir::emitError(loc)
          << "unsupported scalar type for convertScalarToDtype " << scalarType
          << "(scalar type) -> " << dtype << "(dtype)";
      return nullptr;
    }
  }

  if (auto dtypeFloat = dtype.dyn_cast<mlir::FloatType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>()) {
      if (scalarFloat.getWidth() > dtypeFloat.getWidth())
        return b.create<arith::TruncFOp>(loc, dtype, scalar);
      // Only scalarFloat width < dtypeFloat width can reach here.
      return b.create<arith::ExtFOp>(loc, dtype, scalar);
    }
    assert(scalarType.isa<mlir::IntegerType>());
    if (scalarType.isSignlessInteger(1) ||
        (srcOriginalDtype.has_value() && srcOriginalDtype->isUnsignedInteger()))
      return b.create<arith::UIToFPOp>(loc, dtype, scalar);
    // It's safe to use SIToFPOp because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::SIToFPOp>(loc, dtype, scalar);
  }

  if (auto dtypeInteger = dtype.dyn_cast<mlir::IntegerType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>())
      return b.create<arith::FPToSIOp>(loc, dtype, scalar);
    assert(scalarType.isa<mlir::IntegerType>());
    auto scalarInteger = scalarType.cast<mlir::IntegerType>();
    if (scalarInteger.getWidth() > dtypeInteger.getWidth())
      return b.create<arith::TruncIOp>(loc, dtype, scalar);
    if (scalarType.isSignlessInteger(1) ||
        (srcOriginalDtype.has_value() && srcOriginalDtype->isUnsignedInteger()))
      return b.create<arith::ExtUIOp>(loc, dtype, scalar);
    // Only scalarInteger width < dtypeInteger width can reach here.
    // It's safe to use ExtSIOp here because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::ExtSIOp>(loc, dtype, scalar);
  }

  llvm_unreachable("convertScalarToDtype should handle all the types");
}

Value toPositiveValidDim(ConversionPatternRewriter &rewriter, Location loc,
                         Value torchOptionalInt, Value builtinInt,
                         Value defaultValue, Value dimSize) {
  if (torchOptionalInt.getType().isa<Torch::NoneType>())
    return defaultValue;
  auto dimSizeAsInt = castIndexToInt64(rewriter, loc, dimSize);
  Value positiveDim =
      toPositiveDimDynamic(rewriter, loc, builtinInt, dimSizeAsInt);
  // positiveDim < 0 ? 0 : positiveDim
  Value cst0 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(dimSizeAsInt.getType()));
  Value predDimSltZero = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, positiveDim, cst0);
  Value atLeastZero =
      rewriter.create<arith::SelectOp>(loc, predDimSltZero, cst0, positiveDim);
  // atLeastZero > dimSizeAsInt ? dimSizeAsInt : atLeastZero
  Value sgtDimSize = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, atLeastZero, dimSizeAsInt);
  Value boundedByDimSize = rewriter.create<arith::SelectOp>(
      loc, sgtDimSize, dimSizeAsInt, atLeastZero);

  return castIntToIndex(rewriter, loc, boundedByDimSize);
}

// Checks whether the `shapeA` and `shapeB` are broadcast compatible or not. If
// yes, then computes the final broadcast shape.
void computeBroadcastShape(ConversionPatternRewriter &rewriter, Location loc,
                           Value inputA, Value inputB,
                           SmallVector<int64_t> &resultShape,
                           SmallVector<Value> &resultShapeValue) {
  SmallVector<int64_t> shapeA{
      inputA.getType().cast<BaseTensorType>().getSizes()};
  SmallVector<int64_t> shapeB{
      inputB.getType().cast<BaseTensorType>().getSizes()};
  unsigned rankA = shapeA.size();
  unsigned rankB = shapeB.size();
  unsigned minRank = rankA > rankB ? rankB : rankA;
  // Check whether the shapes of the tensors are broadcastable or not.
  // Two tensors are “broadcastable” if the following rules hold:
  // 1.) Each tensor has at least one dimension.
  // 2.) When iterating over the dimension sizes, starting at the trailing
  // dimension, the dimension sizes must either be equal, one of them is 1, or
  // one of them does not exist.
  for (unsigned i = 0; i < minRank; i++) {
    Value sizeDimA = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankA - i - 1));
    Value sizeDimB = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankB - i - 1));
    Value sizeInputA =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputA, sizeDimA);
    Value sizeInputB =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputB, sizeDimB);
    Value torchCstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value cmpSizeAEqualsSizeB =
        rewriter.create<Torch::AtenEqIntOp>(loc, sizeInputA, sizeInputB);
    Value cmpSizeAEqualsOne =
        rewriter.create<Torch::AtenEqIntOp>(loc, sizeInputA, torchCstOne);
    Value cmpSizeBEqualsOne =
        rewriter.create<Torch::AtenEqIntOp>(loc, sizeInputB, torchCstOne);
    Value anyBoolOpList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(cmpSizeAEqualsOne.getType()),
        SmallVector<Value>{cmpSizeAEqualsSizeB, cmpSizeAEqualsOne,
                           cmpSizeBEqualsOne});
    Value cmp = rewriter.create<Torch::AtenAnyBoolOp>(loc, anyBoolOpList);
    rewriter.create<Torch::RuntimeAssertOp>(
        loc, cmp, "tensors are not broadcast compatible");
  }
  // If we reach here then it means both the shapes are broadcast compatible.
  resultShape = rankA >= rankB ? shapeA : shapeB;
  Value shapeTensor = rankA >= rankB ? inputA : inputB;
  for (unsigned i = 0; i < resultShape.size(); i++) {
    Value sizeDim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(i));
    resultShapeValue.push_back(
        rewriter.createOrFold<AtenSizeIntOp>(loc, shapeTensor, sizeDim));
  }

  unsigned resultRank = resultShape.size();
  for (unsigned i = 0; i < minRank; i++) {
    Value sizeDimA = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankA - i - 1));
    Value sizeDimB = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankB - i - 1));
    Value sizeInputA =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputA, sizeDimA);
    Value sizeInputB =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputB, sizeDimB);
    resultShapeValue[resultRank - i - 1] =
        rewriter.create<PrimMaxIntOp>(loc, sizeInputA, sizeInputB);
    if (shapeA[rankA - i - 1] == kUnknownSize ||
        shapeB[rankB - i - 1] == kUnknownSize) {
      resultShape[resultRank - i - 1] = kUnknownSize;
    } else {
      resultShape[resultRank - i - 1] =
          std::max(shapeA[rankA - i - 1], shapeB[rankB - i - 1]);
    }
  }
}

} // namespace Torch
} // namespace torch
} // namespace mlir
