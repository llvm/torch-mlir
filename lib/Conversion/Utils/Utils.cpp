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
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace torch {
namespace Torch {

LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                          PatternRewriter &rewriter) {
  // Check the value tensor is ranked as expected by Linalg.
  // TODO: Remove this check but use a separate verification pass to verify the
  // invariants expected by later passes.
  auto isValidLinalgType = [](Type type) {
    if (isa<NonValueTensorType>(type))
      return false;
    auto tensor = dyn_cast<ValueTensorType>(type);
    return !tensor ||
           dyn_cast_or_null<RankedTensorType>(tensor.toBuiltinTensor());
  };

  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

LogicalResult checkNotNone(PatternRewriter &rewriter, Operation *op, Value v) {
  Type type = v.getType();
  if (isa<OptionalType>(type) || isa<Torch::NoneType>(type) ||
      isa<mlir::NoneType>(type))
    return rewriter.notifyMatchFailure(op, "unimplemented None type arg");
  return success();
}

// Generate IR: dim = dim >= 0 ? dim : dim + inputRank
Value toPositiveDimDynamic(OpBuilder &b, Location loc, Value dim,
                           Value inputRank) {
  assert(isa<IntegerType>(dim.getType()) &&
         "dim arg of toPositiveDim must be integer type");
  Value dimAddInputRank = arith::AddIOp::create(b, loc, dim, inputRank);
  Value cst0 =
      arith::ConstantOp::create(b, loc, b.getZeroAttr(inputRank.getType()));
  Value predDimGEZero =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, dim, cst0);
  Value dimInt =
      arith::SelectOp::create(b, loc, predDimGEZero, dim, dimAddInputRank);
  return dimInt;
}

// Generate IR: assert(dim >= 0 && dim < inputRank)
void assertIsValidDim(OpBuilder &b, Location loc, Value dim, Value inputRank) {
  assert(isa<IntegerType>(dim.getType()) &&
         "dim arg of assertIsValidDim must be integer type");
  Value cst0 =
      arith::ConstantOp::create(b, loc, b.getZeroAttr(inputRank.getType()));
  Value predGEZero =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, dim, cst0);
  cf::AssertOp::create(b, loc, predGEZero,
                       b.getStringAttr("dim must be greater or equal to zero"));
  Value predLTInputRank =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, dim, inputRank);
  cf::AssertOp::create(b, loc, predLTInputRank,
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
    assert((isa<IntegerType>(type) || isa<IndexType>(type)) &&
           "must be either integer or index type");
  };
  checkIntOrIndex(lhsType);
  checkIntOrIndex(rhsType);
  Value lhsDimInt =
      lhsType.isIndex() ? castIndexToInt64(b, loc, lhsDim) : lhsDim;
  Value rhsDimInt =
      rhsType.isIndex() ? castIndexToInt64(b, loc, rhsDim) : rhsDim;
  Value contractingDimEqual = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::eq, lhsDimInt, rhsDimInt);
  cf::AssertOp::create(b, loc, contractingDimEqual,
                       b.getStringAttr("mismatching contracting dimension"));
}

// Creates a tensor with required `sizes` and `elemTy` and fills it with
// initElem.
Value createInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                       Type elemTy, Value initElem) {
  Value initTensor =
      tensor::EmptyOp::create(b, loc, getAsOpFoldResult(sizes), elemTy);
  return linalg::FillOp::create(b, loc, initElem, initTensor).getResult(0);
}

Value createZeroInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                           Type elemTy) {
  Value initTensor =
      tensor::EmptyOp::create(b, loc, getAsOpFoldResult(sizes), elemTy);

  Value c0;
  if (auto dtypeComplex = dyn_cast<mlir::ComplexType>(elemTy)) {
    // For complex types, create a complex zero (0.0 + 0.0j)
    Type floatType = cast<mlir::FloatType>(dtypeComplex.getElementType());
    Value realZero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(floatType));
    Value imagZero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(floatType));
    c0 = complex::CreateOp::create(b, loc, elemTy, realZero, imagZero);
  } else {
    c0 = arith::ConstantOp::create(b, loc, b.getZeroAttr(elemTy));
  }
  return linalg::FillOp::create(b, loc, c0, initTensor).getResult(0);
}

Value createOneInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                          Type elemTy) {
  Value initTensor =
      tensor::EmptyOp::create(b, loc, getAsOpFoldResult(sizes), elemTy);

  Value c1;
  if (auto dtypeComplex = dyn_cast<mlir::ComplexType>(elemTy)) {
    // For complex types, create a complex one (1.0 + 0.0j)
    Type floatType = cast<mlir::FloatType>(dtypeComplex.getElementType());
    Value realOne = arith::ConstantOp::create(b, loc, b.getOneAttr(floatType));
    Value imagZero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(floatType));
    c1 = complex::CreateOp::create(b, loc, elemTy, realOne, imagZero);
  } else {
    c1 = arith::ConstantOp::create(b, loc, b.getOneAttr(elemTy));
  }
  return linalg::FillOp::create(b, loc, c1, initTensor).getResult(0);
}

Value castIntToIndex(OpBuilder &b, Location loc, Value v) {
  assert(isa<IntegerType>(v.getType()) && "must be called with integer type");
  return b.createOrFold<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

Value castIndexToInt64(OpBuilder &b, Location loc, Value idx) {
  assert(isa<IndexType>(idx.getType()) && "must be called with integer type");
  return b.createOrFold<arith::IndexCastOp>(loc, b.getI64Type(), idx);
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
  RankedTensorType type = cast<RankedTensorType>(tensor.getType());
  assert(dim < type.getRank() &&
         "The given dim must be smaller than tensor rank");
  (void)type;
  SmallVector<Value> sizes;
  for (int i = 0; i <= dim; i++)
    sizes.push_back(getDimOp(b, loc, tensor, i));
  return sizes;
}

SmallVector<Value> getTensorSizes(OpBuilder &b, Location loc, Value tensor) {
  RankedTensorType type = cast<RankedTensorType>(tensor.getType());
  return getTensorSizesUntilDim(b, loc, tensor, type.getRank() - 1);
}

Value getTensorSize(OpBuilder &b, Location loc, Value tensor) {
  SmallVector<Value> sizes(getTensorSizes(b, loc, tensor));
  Value productResult = arith::ConstantOp::create(b, loc, b.getIndexAttr(1));
  for (Value size : sizes)
    productResult = arith::MulIOp::create(b, loc, productResult, size);
  return castIndexToInt64(b, loc, productResult);
}

// Creates a constant of type `elemType` with value `val`.
Value getConstant(OpBuilder &b, Location loc, int64_t val, Type elemType) {
  TypedAttr attr = {};
  if (isa<mlir::FloatType>(elemType))
    attr = b.getFloatAttr(elemType, val);
  if (isa<mlir::IndexType>(elemType))
    attr = b.getIndexAttr(val);
  if (isa<mlir::IntegerType>(elemType))
    attr = b.getIntegerAttr(elemType,
                            APInt(cast<IntegerType>(elemType).getWidth(), val));
  if (!attr)
    return nullptr;
  return arith::ConstantOp::create(b, loc, elemType, attr);
}

SmallVector<Value> getAsConstantIntValues(OpBuilder &b, Location loc,
                                          SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return arith::ConstantOp::create(b, loc,
                                     b.getIntegerAttr(b.getI64Type(), val));
  }));
}

SmallVector<Value> getAsConstantIndexValues(OpBuilder &b, Location loc,
                                            SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return arith::ConstantOp::create(b, loc, b.getIndexAttr(val));
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

static std::optional<int64_t> getIntegerValue(Value scalar) {
  if (auto constOp = scalar.getDefiningOp<Torch::ConstantIntOp>()) {
    return std::optional<int64_t>(constOp.getValue());
  }
  return std::optional<int64_t>();
}

// Convert a scalar value to the target type. The scalar value can be an element
// from a tensor or a scalar in the pytorch dialect. Both the scalar and dtype
// should be converted builtin types.
Value convertScalarToDtype(OpBuilder &b, Location loc, Value scalar, Type dtype,
                           std::optional<Type> srcOriginalDtype,
                           std::optional<Type> dstOriginalDtype,
                           std::optional<Value> originalScalar) {
  Type scalarType = scalar.getType();
  if (scalarType == dtype)
    return scalar;

  auto isByteOrChar = [](Type type) {
    if (auto integerTy = dyn_cast<mlir::IntegerType>(type)) {
      return integerTy.getWidth() == 8;
    }
    return false;
  };

  // We support conversion to Byte dtype only if the original scalar is an
  // integer constant with value lying between 0 - 63.
  if (isByteOrChar(dtype)) {
    if (!dstOriginalDtype.has_value()) {
      mlir::emitError(loc)
          << "unimplemented: for conversion to byte or char type "
             "dstOriginalDtype has to be passed to convertScalarToDtype";
      return nullptr;
    }
    if (dstOriginalDtype->isUnsignedInteger()) {
      if (originalScalar.has_value()) {
        std::optional<int64_t> optConstVal =
            getIntegerValue(originalScalar.value());
        if (optConstVal.has_value()) {
          int64_t constVal = optConstVal.value();
          if (constVal < 0 || constVal > 63) {
            // Do the conversion only if the original integer value is between
            // 0 - 63.
            mlir::emitError(loc)
                << "unsupported: conversion to byte type for "
                   "convertScalarToDtype "
                << scalarType << "(scalar type) -> " << dtype << "(dtype)";
            return nullptr;
          }
        }
      }
    }
  }

  // If the dtype is i1, i.e., a boolean type.
  if (dtype.isSignlessInteger(1)) {
    Type scalarType = scalar.getType();
    Value cstZero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(scalarType));
    if (isa<mlir::FloatType>(scalarType)) {
      return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UNE, scalar,
                                   cstZero);
    } else if (isa<mlir::IntegerType>(scalarType)) {
      return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, scalar,
                                   cstZero);
    } else {
      mlir::emitError(loc)
          << "unsupported scalar type for convertScalarToDtype " << scalarType
          << "(scalar type) -> " << dtype << "(dtype)";
      return nullptr;
    }
  }

  if (auto dtypeFloat = dyn_cast<mlir::FloatType>(dtype)) {
    if (auto scalarFloat = dyn_cast<mlir::FloatType>(scalarType)) {
      if (scalarFloat.getWidth() == 16 && dtypeFloat.getWidth() == 16) {
        auto scalarF32 = arith::ExtFOp::create(b, loc, b.getF32Type(), scalar);
        return arith::TruncFOp::create(b, loc, dtype, scalarF32);
      }
      if (scalarFloat.getWidth() > dtypeFloat.getWidth())
        return arith::TruncFOp::create(b, loc, dtype, scalar);
      // Only scalarFloat width < dtypeFloat width can reach here.
      return arith::ExtFOp::create(b, loc, dtype, scalar);
    }
    assert(isa<mlir::IntegerType>(scalarType));
    if (scalarType.isSignlessInteger(1) ||
        (srcOriginalDtype.has_value() && srcOriginalDtype->isUnsignedInteger()))
      return arith::UIToFPOp::create(b, loc, dtype, scalar);
    // It's safe to use SIToFPOp because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return arith::SIToFPOp::create(b, loc, dtype, scalar);
  }

  if (auto dtypeInteger = dyn_cast<mlir::IntegerType>(dtype)) {
    if (auto scalarFloat = dyn_cast<mlir::FloatType>(scalarType))
      return arith::FPToSIOp::create(b, loc, dtype, scalar);
    assert(isa<mlir::IntegerType>(scalarType));
    auto scalarInteger = cast<mlir::IntegerType>(scalarType);
    if (scalarInteger.getWidth() > dtypeInteger.getWidth())
      return arith::TruncIOp::create(b, loc, dtype, scalar);
    if (scalarType.isSignlessInteger(1) ||
        (srcOriginalDtype.has_value() && srcOriginalDtype->isUnsignedInteger()))
      return arith::ExtUIOp::create(b, loc, dtype, scalar);
    // Only scalarInteger width < dtypeInteger width can reach here.
    // It's safe to use ExtSIOp here because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return arith::ExtSIOp::create(b, loc, dtype, scalar);
  }

  if (auto dtypeComplex = dyn_cast<mlir::ComplexType>(dtype)) {

    // Complex to complex.
    if (auto scalarComplex = dyn_cast<mlir::ComplexType>(scalarType)) {
      auto dtypeElemType = dtypeComplex.getElementType();

      // Extract the real and imaginary parts of the scalar.
      // Cast them to the target element type, and create a new complex
      // value with the target complex type.
      Value realVal = complex::ReOp::create(b, loc, scalar);
      Value imgVal = complex::ImOp::create(b, loc, scalar);

      realVal = convertScalarToDtype(b, loc, realVal, dtypeElemType);
      imgVal = convertScalarToDtype(b, loc, imgVal, dtypeElemType);

      return complex::CreateOp::create(b, loc, dtypeComplex, realVal, imgVal);
    }

    // Float to complex type.
    if (auto dtypeFloat = dyn_cast<mlir::FloatType>(scalarType)) {
      auto complexElementType =
          cast<mlir::FloatType>(dtypeComplex.getElementType());
      Value realVal;
      Value imgVal =
          arith::ConstantOp::create(b, loc, b.getZeroAttr(complexElementType));

      if (complexElementType.getWidth() > dtypeFloat.getWidth()) {
        realVal = arith::ExtFOp::create(b, loc, complexElementType, scalar);
      } else if (complexElementType.getWidth() < dtypeFloat.getWidth()) {
        realVal = arith::TruncFOp::create(b, loc, complexElementType, scalar);
      } else {
        realVal = scalar;
      }

      return complex::CreateOp::create(b, loc, dtypeComplex, realVal, imgVal);
    }

    // Int to complex type.
    if (auto dtypeInt = dyn_cast<mlir::IntegerType>(scalarType)) {
      auto complexElementType =
          cast<mlir::FloatType>(dtypeComplex.getElementType());

      Value realVal =
          arith::SIToFPOp::create(b, loc, complexElementType, scalar);
      Value imgVal =
          arith::ConstantOp::create(b, loc, b.getZeroAttr(complexElementType));

      return complex::CreateOp::create(b, loc, dtypeComplex, realVal, imgVal);
    }

    mlir::emitError(loc) << "unsupported scalar type for convertScalarToDtype "
                         << scalarType << "(scalar type) -> " << dtype
                         << "(dtype)";
  }

  llvm_unreachable("convertScalarToDtype should handle all the types");
}

Value toPositiveValidDim(ConversionPatternRewriter &rewriter, Location loc,
                         Value torchOptionalInt, Value builtinInt,
                         Value defaultValue, Value dimSize) {
  if (isa<Torch::NoneType>(torchOptionalInt.getType()))
    return defaultValue;
  auto dimSizeAsInt = castIndexToInt64(rewriter, loc, dimSize);
  Value positiveDim =
      toPositiveDimDynamic(rewriter, loc, builtinInt, dimSizeAsInt);
  // positiveDim < 0 ? 0 : positiveDim
  Value cst0 = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(dimSizeAsInt.getType()));
  Value predDimSltZero = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, positiveDim, cst0);
  Value atLeastZero =
      arith::SelectOp::create(rewriter, loc, predDimSltZero, cst0, positiveDim);
  // atLeastZero > dimSizeAsInt ? dimSizeAsInt : atLeastZero
  Value sgtDimSize = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sgt, atLeastZero, dimSizeAsInt);
  Value boundedByDimSize = arith::SelectOp::create(rewriter, loc, sgtDimSize,
                                                   dimSizeAsInt, atLeastZero);

  return castIntToIndex(rewriter, loc, boundedByDimSize);
}

// Helper function to unsqueeze the input tensor at given dim.
// Returns the unsqueezed tensor or failure.
FailureOr<Value> unsqueezeTensor(PatternRewriter &rewriter, Operation *op,
                                 Value input, int64_t dim) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // `input` has a reduced rank. Hence add 1.
  int64_t unsqueezedRank = inputShape.size() + 1;
  dim = toPositiveDim(dim, unsqueezedRank);
  if (!isValidDim(dim, unsqueezedRank)) {
    return rewriter.notifyMatchFailure(op, "dim is not a valid dim");
  }

  SmallVector<int64_t> unsqueezedShape{inputShape};
  unsqueezedShape.insert(unsqueezedShape.begin() + dim, 1);
  Type unsqueezedType =
      RankedTensorType::get(unsqueezedShape, inputType.getElementType());

  SmallVector<ReassociationIndices> reassociationMap(inputRank);
  // From the perspective of the reassociation map, the situation of
  // unsqueezing before or after the last dimension is symmetrical.
  // Normalize it to the "before" case.
  // The 0 case is special here, since there is no last dimension to insert
  // before -- we simply rely on the loop below iterating 0 times.
  if (dim == inputRank && inputRank != 0)
    dim = inputRank - 1;
  bool alreadyCrossedExpandedDim = false;
  for (int i = 0; i != inputRank; i++) {
    if (alreadyCrossedExpandedDim) {
      reassociationMap[i].push_back(i + 1);
    } else {
      reassociationMap[i].push_back(i);
      if (i == dim) {
        reassociationMap[i].push_back(i + 1);
        alreadyCrossedExpandedDim = true;
      }
    }
  }
  Value unsqueezed = tensor::ExpandShapeOp::create(
      rewriter, op->getLoc(), unsqueezedType, input, reassociationMap);
  return unsqueezed;
}

// Helper function to squeeze the input tensor at given dim.
// Returns the squeezed tensor or failure.
FailureOr<Value> squeezeTensor(PatternRewriter &rewriter, Operation *op,
                               Value input, int64_t dim) {
  Location loc = op->getLoc();
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();

  // No scope for squeezing the input.
  if (inputRank == 0)
    return input;

  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  // assert dynamic squeeze dim size == 1
  if (inputType.isDynamicDim(dim)) {
    Value cstDim = arith::ConstantIndexOp::create(rewriter, loc, dim);
    Value dimVal = tensor::DimOp::create(rewriter, loc, input, cstDim);
    Value cstOne = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      dimVal, cstOne);
    cf::AssertOp::create(
        rewriter, loc, cmp,
        rewriter.getStringAttr(
            "Expected dynamic squeeze dim size to be statically 1"));
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t> squeezedShape;
  squeezedShape.append(inputShape.begin(), inputShape.begin() + dim);
  squeezedShape.append(inputShape.begin() + dim + 1, inputShape.end());
  int64_t squeezedRank = inputRank - 1;
  Type squeezedType =
      RankedTensorType::get(squeezedShape, inputType.getElementType());

  // If the dim(th) dimension of operand tensor type is not statically unit,
  // squeeze will behave as an identity operation.
  if (inputType.getDimSize(dim) != 1 && !inputType.isDynamicDim(dim)) {
    return input;
  }

  SmallVector<ReassociationIndices> reassociationMap(squeezedRank);
  bool alreadyCrossedSqueezedDim = false;
  for (int i = 0; i != squeezedRank; i++) {
    if (alreadyCrossedSqueezedDim) {
      reassociationMap[i].push_back(i + 1);
    } else {
      reassociationMap[i].push_back(i);
      if (dim != 0 && i != dim - 1)
        continue;

      alreadyCrossedSqueezedDim = true;
      if (dim == 0)
        reassociationMap[0].push_back(1);
      if (i == dim - 1)
        reassociationMap[i].push_back(dim);
    }
  }
  // Note: In case the operand tensor type is of unit rank and is statically
  // shaped with unit dimension, the `reassociationMap` will be empty and the
  // input will be collapsed to a 0-D tensor.
  Value squeezed = tensor::CollapseShapeOp::create(
      rewriter, op->getLoc(), squeezedType, input, reassociationMap);
  return squeezed;
}

void getZeroPoint(Value value, Value &zeropoint) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return;

  // Extract and set the zero point from a given op.
  auto getZp = [&](auto op) { zeropoint = op.getZeroPoint(); };

  llvm::TypeSwitch<Operation *>(definingOp)
      .Case<Aten_MakePerTensorQuantizedTensorOp>(getZp)
      .Case<AtenQuantizePerTensorOp>(getZp)
      .Case<Aten_MakePerChannelQuantizedTensorOp>(getZp);
}

LogicalResult getQuantizationParams(Value value, Value &zeropoint, Value &scale,
                                    int64_t &axis) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return failure();

  // Extract and set the common parameters from a given op.
  auto setParams = [&](auto op) -> LogicalResult {
    zeropoint = op.getZeroPoint();
    scale = op.getScale();
    // Axis must be constant scalar int for Aten_MakePerChannelQuantizedTensorOp
    if constexpr (std::is_same_v<decltype(op),
                                 Aten_MakePerChannelQuantizedTensorOp>) {
      return success(matchPattern(op.getAxis(), m_TorchConstantInt(&axis)));
    } else {
      // Other ops don't have axis parameter
      axis = -1;
      return success();
    }
  };

  return llvm::TypeSwitch<Operation *, LogicalResult>(definingOp)
      .Case<Aten_MakePerTensorQuantizedTensorOp>(setParams)
      .Case<AtenQuantizePerTensorOp>(setParams)
      .Case<Aten_MakePerChannelQuantizedTensorOp>(setParams)
      .Default([](auto) { return failure(); });
}

APFloat getFloatInf(mlir::FloatType fpType, bool negative,
                    bool supportsNonFinites) {
  return supportsNonFinites
             ? APFloat::getInf(fpType.getFloatSemantics(), negative)
             : APFloat::getLargest(fpType.getFloatSemantics(), negative);
}

} // namespace Torch
} // namespace torch
} // namespace mlir
