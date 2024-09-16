//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>
#include <type_traits>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Check if a ranked-tensor has the specified element type.
template <typename elementType> static bool hasElementType(Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  Type tensorElementType = tensorType.getElementType();
  return isa<elementType>(tensorElementType);
}

template <arith::CmpFPredicate fpred, arith::CmpIPredicate iupred,
          arith::CmpIPredicate ispred>
static Value createComparisonTemplate(OpBuilder &b, Location loc, Type type,
                                      Value lhs, Value rhs) {
  if (isa<mlir::FloatType>(type))
    return b.create<arith::CmpFOp>(loc, fpred, lhs, rhs);
  if (IntegerType intType = dyn_cast<mlir::IntegerType>(type)) {
    if (intType.isUnsigned())
      return b.create<arith::CmpIOp>(loc, iupred, lhs, rhs);
    if (intType.isSigned())
      return b.create<arith::CmpIOp>(loc, ispred, lhs, rhs);
    assert(intType.getWidth() == 1);
    return b.create<arith::CmpIOp>(loc, iupred, lhs, rhs);
  }
  llvm_unreachable("Unhandled element type for comparison");
}

static Value getZeroPoint(Value value) {
  if (auto make = value.getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
    return make.getZeroPoint();
  }
  return nullptr;
}

static Value createGreaterThan(OpBuilder &b, Location loc, Type elementalType,
                               Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::OGT,
                                  arith::CmpIPredicate::ugt,
                                  arith::CmpIPredicate::sgt>(
      b, loc, elementalType, lhs, rhs);
}

static Value createGreaterThanOrEqual(OpBuilder &b, Location loc,
                                      Type elementalType, Value lhs,
                                      Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::OGE,
                                  arith::CmpIPredicate::uge,
                                  arith::CmpIPredicate::sge>(
      b, loc, elementalType, lhs, rhs);
}

static Value createLessThan(OpBuilder &b, Location loc, Type elementalType,
                            Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::OLT,
                                  arith::CmpIPredicate::ult,
                                  arith::CmpIPredicate::slt>(
      b, loc, elementalType, lhs, rhs);
}

static Value createLessThanOrEqual(OpBuilder &b, Location loc,
                                   Type elementalType, Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::OLE,
                                  arith::CmpIPredicate::ule,
                                  arith::CmpIPredicate::sle>(
      b, loc, elementalType, lhs, rhs);
}

static Value createEqual(OpBuilder &b, Location loc, Type elementalType,
                         Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::OEQ,
                                  arith::CmpIPredicate::eq,
                                  arith::CmpIPredicate::eq>(
      b, loc, elementalType, lhs, rhs);
}

static Value createNotEqual(OpBuilder &b, Location loc, Type elementalType,
                            Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::UNE,
                                  arith::CmpIPredicate::ne,
                                  arith::CmpIPredicate::ne>(
      b, loc, elementalType, lhs, rhs);
}

static Value buildNormalCdf(OpBuilder &b, Location &loc, Value x, Value mean,
                            Value sigma) {
  Type elementType = x.getType();
  Value xMinusMean = b.create<arith::SubFOp>(loc, x, mean);
  Value two = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 2));
  Value sqrt2 = b.create<math::SqrtOp>(loc, two);
  Value erfArg = b.create<arith::DivFOp>(loc, xMinusMean, sqrt2);
  Value erf = b.create<math::ErfOp>(loc, erfArg);
  Value one = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1));
  Value erfPlus1 = b.create<arith::AddFOp>(loc, one, erf);
  Value oneHalf =
      b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.5));
  Value normalCdf = b.create<arith::MulFOp>(loc, oneHalf, erfPlus1);
  return normalCdf;
}

static Value buildUnitNormalCdf(OpBuilder &b, Location &loc, Value x) {
  Type elementType = x.getType();
  Value zero = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0));
  Value one = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1));
  return buildNormalCdf(b, loc, x, zero, one);
}

template <typename MathOpTy>
static Value createFpOpWithDtype(OpBuilder &b, const TypeConverter *converter,
                                 Value payloadArg, Operation *op) {
  Type inTTy = cast<ValueTensorType>(op->getOperand(0).getType()).getDtype();
  Type outTTy = cast<ValueTensorType>(op->getResult(0).getType()).getDtype();
  Type outTy =
      cast<RankedTensorType>(converter->convertType(op->getResult(0).getType()))
          .getElementType();
  Type computeTy = outTy;
  if (isa<IntegerType>(computeTy))
    computeTy = b.getF32Type();
  Location loc = op->getLoc();
  Value arg = convertScalarToDtype(b, loc, payloadArg, computeTy, inTTy);
  auto newOp = b.create<MathOpTy>(loc, arg);
  return convertScalarToDtype(b, loc, newOp, outTy, std::nullopt, outTTy);
}

template <class T, class... Ts>
struct is_any_same : std::disjunction<std::is_same<T, Ts>...> {};

template <typename OpTy>
static Value createCompareOp(OpBuilder &b, Location loc, OpTy op, Value lhs,
                             Value rhs) {
  static_assert(
      is_any_same<OpTy, AtenLtScalarOp, AtenLeScalarOp, AtenEqScalarOp,
                  AtenNeScalarOp, AtenGtScalarOp, AtenGeScalarOp,
                  AtenLtTensorOp, AtenLeTensorOp, AtenGtTensorOp,
                  AtenGeTensorOp, AtenEqTensorOp, AtenNeTensorOp>(),
      "unimplemented: op type not supported");

  Type lhsDtype = lhs.getType();
  Type rhsDtype = rhs.getType();
  Type elementalType = cast<BaseTensorType>(op.getSelf().getType()).getDtype();

  if (lhsDtype.isIntOrFloat() && rhsDtype.isIntOrFloat()) {
    if (isa<mlir::FloatType>(lhsDtype) && isa<mlir::IntegerType>(rhsDtype)) {
      rhs = convertScalarToDtype(b, loc, rhs, lhsDtype);
      elementalType = lhsDtype;
    } else if (isa<mlir::IntegerType>(lhsDtype) &&
               isa<mlir::FloatType>(rhsDtype)) {
      lhs = convertScalarToDtype(b, loc, lhs, rhsDtype);
      elementalType = rhsDtype;
    } else {
      // Both are either Integer or Float types, but the bit width might be
      // different.
      if (lhsDtype.getIntOrFloatBitWidth() > rhsDtype.getIntOrFloatBitWidth()) {
        rhs = convertScalarToDtype(b, loc, rhs, lhsDtype);
      } else {
        lhs = convertScalarToDtype(b, loc, lhs, rhsDtype);
      }
    }
  } else {
    op.emitError("unimplemented: type promotion from tensor to scalar.");
    return nullptr;
  }

  if constexpr (is_any_same<OpTy, AtenLtScalarOp, AtenLtTensorOp>()) {
    return createLessThan(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (is_any_same<OpTy, AtenLeScalarOp, AtenLeTensorOp>()) {
    return createLessThanOrEqual(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (is_any_same<OpTy, AtenGtScalarOp, AtenGtTensorOp>()) {
    return createGreaterThan(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (is_any_same<OpTy, AtenGeScalarOp, AtenGeTensorOp>()) {
    return createGreaterThanOrEqual(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (is_any_same<OpTy, AtenEqScalarOp, AtenEqTensorOp>()) {
    return createEqual(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (is_any_same<OpTy, AtenNeScalarOp, AtenNeTensorOp>()) {
    return createNotEqual(b, loc, elementalType, lhs, rhs);
  }
  llvm_unreachable("unimplemented: op type not supported");
}

template <arith::CmpIPredicate predicate>
static LogicalResult
createTriangularMatrix(OpBuilder &b, Location loc, ValueRange payloadArgs,
                       Operation *op, ArrayRef<Value> operands, Value &result) {
  auto inputType = cast<RankedTensorType>(operands[0].getType());
  uint64_t inputRank = inputType.getRank();

  // Use the indices of the two innermost dimensions.
  auto rowIndex = b.create<linalg::IndexOp>(loc, inputRank - 2);
  Value rowIndexI64 = castIndexToInt64(b, loc, rowIndex);
  auto colIndex = b.create<linalg::IndexOp>(loc, inputRank - 1);
  Value colIndexI64 = castIndexToInt64(b, loc, colIndex);

  // columnIndex >= rowIndex + diagonal?
  auto sum =
      b.create<arith::AddIOp>(loc, rowIndexI64, /*diagonal=*/operands[1]);
  auto pred = b.create<arith::CmpIOp>(loc, predicate, colIndexI64, sum);

  Value scalar = payloadArgs[0];
  Type elementType = inputType.getElementType();
  Value zero = getConstant(b, loc, 0, elementType);
  result = b.create<arith::SelectOp>(loc, pred, scalar, zero);
  return success();
}

template <typename OpT>
Value createDivModePayload(OpBuilder &b, Location loc,
                           const TypeConverter *converter,
                           ValueRange payloadArgs, OpT op,
                           ArrayRef<Value> operands) {
  static_assert(std::is_same_v<OpT, AtenDivTensorModeOp> ||
                    std::is_same_v<OpT, AtenDivScalarModeOp>,
                "template type must be a tensor/scalar div mode");
  typename OpT::Adaptor adaptor(operands);
  Type dtype = cast<RankedTensorType>(converter->convertType(op.getType()))
                   .getElementType();
  Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
  Value rhs = convertScalarToDtype(
      b, loc,
      std::is_same_v<OpT, AtenDivScalarModeOp> ? operands[1] : payloadArgs[1],
      dtype);

  Value quotient;
  if (isa<mlir::FloatType>(dtype)) {
    quotient = b.create<arith::DivFOp>(loc, lhs, rhs);
  } else if (dtype.isUnsignedInteger()) {
    quotient = b.create<arith::DivUIOp>(loc, lhs, rhs);
  } else {
    assert(dtype.isInteger() &&
           "dtype should be an integer (signless or signed)");
    quotient = b.create<arith::DivSIOp>(loc, lhs, rhs);
  }

  if (isa<Torch::NoneType>(op.getRoundingMode().getType()))
    return quotient;

  std::string roundingMode;
  if (!matchPattern(op.getRoundingMode(), m_TorchConstantStr(roundingMode))) {
    op.emitError("only support constant str rounding mode");
    return nullptr;
  }
  assert((roundingMode == "trunc" || roundingMode == "floor") &&
         "unsupported rounding mode");
  if (roundingMode == "trunc") {
    // "trunc" - rounds the results of the division towards zero. Equivalent
    // to C-style integer division.
    if (!isa<mlir::FloatType>(dtype)) {
      // nothing to do for integers
      return quotient;
    }

    // float
    Value ceil = b.create<math::CeilOp>(loc, quotient);
    Value floor = b.create<math::FloorOp>(loc, quotient);
    Value cstZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(dtype));
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,
                                         quotient, cstZero);
    return b.create<arith::SelectOp>(loc, pred, ceil, floor);
  }
  if (roundingMode == "floor") {
    // "floor" - rounds the results of the division down. Equivalent to
    // floor division in Python (the // operator)
    if (isa<mlir::FloatType>(dtype))
      return b.create<math::FloorOp>(loc, quotient);
    if (!dtype.isUnsignedInteger()) {
      Type defaultIntToFloatType = b.getF64Type();
      lhs = convertScalarToDtype(b, loc, lhs, defaultIntToFloatType);
      rhs = convertScalarToDtype(b, loc, rhs, defaultIntToFloatType);
      quotient = b.create<arith::DivFOp>(loc, lhs, rhs);
      Value floor = b.create<math::FloorOp>(loc, quotient);
      Value convert = convertScalarToDtype(b, loc, floor, dtype);
      return convert;
    }
  }
  return quotient;
}

template <typename OpT>
Value createRemainderPayload(OpBuilder &b, Location loc,
                             const TypeConverter *converter,
                             ValueRange payloadArgs, OpT op,
                             ArrayRef<Value> operands) {
  static_assert(
      llvm::is_one_of<OpT, AtenRemainderScalarOp, AtenRemainderTensorOp>(),
      "op must be a tensor/scalar remainder op");
  typename OpT::Adaptor adaptor(operands);
  Type dtype = cast<RankedTensorType>(converter->convertType(op.getType()))
                   .getElementType();
  Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
  Value rhs = convertScalarToDtype(
      b, loc,
      std::is_same_v<OpT, AtenRemainderScalarOp> ? operands[1] : payloadArgs[1],
      dtype);

  // The remainder op we wish to create would look roughly like this:
  // rem = a % b
  // if rem != 0 AND (rem < 0 XOR b < 0) rem += b
  // This is how python calucates remainders for floats and longs:
  // https://github.com/python/cpython/blob/2afd1751dd9a35d4ec03b708e3e5cddd72c43f7e/Objects/floatobject.c#L645
  // https://github.com/python/cpython/blob/2afd1751dd9a35d4ec03b708e3e5cddd72c43f7e/Objects/longobject.c#L3662
  Value result;
  if (isa<mlir::FloatType>(dtype)) {
    Value remainder = b.create<arith::RemFOp>(loc, lhs, rhs);

    Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(dtype));
    Value remainderNotEqualToZero = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, remainder, zero);
    Value otherLessThanZero =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, rhs, zero);
    Value remainderLessThanZero = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, remainder, zero);
    Value xorCondition =
        b.create<arith::XOrIOp>(loc, otherLessThanZero, remainderLessThanZero);
    Value condition =
        b.create<arith::AndIOp>(loc, remainderNotEqualToZero, xorCondition);
    Value fixedRemainder = b.create<arith::AddFOp>(loc, remainder, rhs);
    result =
        b.create<arith::SelectOp>(loc, condition, fixedRemainder, remainder);
  } else {
    assert(dtype.isInteger() &&
           "dtype should be a float or integer (signless or signed)");
    Value remainder = b.create<arith::RemSIOp>(loc, lhs, rhs);

    Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(dtype));
    Value remainderNotEqualToZero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, remainder, zero);
    Value otherLessThanZero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rhs, zero);
    Value remainderLessThanZero = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, remainder, zero);
    Value xorCondition =
        b.create<arith::XOrIOp>(loc, otherLessThanZero, remainderLessThanZero);
    Value condition =
        b.create<arith::AndIOp>(loc, remainderNotEqualToZero, xorCondition);
    Value fixedRemainder = b.create<arith::AddIOp>(loc, remainder, rhs);
    result =
        b.create<arith::SelectOp>(loc, condition, fixedRemainder, remainder);
  }
  return result;
}

static Value createLinalgPayloadCalculationForElementwiseOp(
    OpBuilder &b, Location loc, const TypeConverter *converter,
    ValueRange payloadArgs, Operation *op, ArrayRef<Value> operands) {
  if (isa<AtenFloorOp>(op))
    return b.create<math::FloorOp>(loc, payloadArgs[0]);
  if (isa<AtenCeilOp>(op))
    return b.create<math::CeilOp>(loc, payloadArgs[0]);
  if (isa<AtenExpOp>(op)) {
    return createFpOpWithDtype<math::ExpOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenExpm1Op>(op)) {
    return createFpOpWithDtype<math::ExpM1Op>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLogOp>(op)) {
    return createFpOpWithDtype<math::LogOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLog2Op>(op)) {
    return createFpOpWithDtype<math::Log2Op>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLog10Op>(op)) {
    return createFpOpWithDtype<math::Log10Op>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLog1pOp>(op)) {
    return createFpOpWithDtype<math::Log1pOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenErfOp>(op)) {
    return createFpOpWithDtype<math::ErfOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenSqrtOp>(op)) {
    return createFpOpWithDtype<math::SqrtOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenRsqrtOp>(op)) {
    return createFpOpWithDtype<math::RsqrtOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenNegOp>(op)) {
    return createFpOpWithDtype<arith::NegFOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenSinOp>(op)) {
    return createFpOpWithDtype<math::SinOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenSinhOp>(op)) {
    return createFpOpWithDtype<math::SinhOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAsinOp>(op)) {
    return createFpOpWithDtype<math::AsinOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAsinhOp>(op)) {
    return createFpOpWithDtype<math::AsinhOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenCosOp>(op)) {
    return createFpOpWithDtype<math::CosOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenCoshOp>(op)) {
    return createFpOpWithDtype<math::CoshOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAcosOp>(op)) {
    return createFpOpWithDtype<math::AcosOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAcoshOp>(op)) {
    return createFpOpWithDtype<math::AcoshOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenTanOp>(op)) {
    return createFpOpWithDtype<math::TanOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenTanhOp>(op)) {
    return createFpOpWithDtype<math::TanhOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAtanOp>(op)) {
    return createFpOpWithDtype<math::AtanOp>(b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAtanhOp>(op)) {
    return createFpOpWithDtype<math::AtanhOp>(b, converter, payloadArgs[0], op);
  }
  if (auto clone = dyn_cast<AtenCloneOp>(op)) {
    int64_t memoryFormat;
    if (!isa<Torch::NoneType>(clone.getMemoryFormat().getType()) &&
        (!matchPattern(clone.getMemoryFormat(),
                       m_TorchConstantInt(&memoryFormat)) ||
         (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
          memoryFormat != torch_upstream::MemoryFormat::ChannelsLast))) {
      clone.emitError("unimplemented: only contiguous and channels last memory "
                      "format is supported");
      return nullptr;
    }
    return payloadArgs[0];
  }
  if (auto bitwiseAndTensor = dyn_cast<AtenBitwiseAndTensorOp>(op)) {
    if (isa<mlir::FloatType>(
            cast<ValueTensorType>(bitwiseAndTensor.getType()).getDtype())) {
      bitwiseAndTensor.emitError(
          "Bitwise_And does not support floating point dtype");
      return nullptr;
    }
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(bitwiseAndTensor.getType()))
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::AndIOp>(loc, lhs, rhs);
  }
  if (auto bitwiseAndScalar = dyn_cast<AtenBitwiseAndScalarOp>(op)) {
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(bitwiseAndScalar.getType()))
                     .getElementType();
    if (!isa<mlir::IntegerType>(dtype)) {
      bitwiseAndScalar.emitError(
          "bitwise_and.Scalar does not support non-integer input dtype.");
      return nullptr;
    }
    Type resultElementType =
        cast<BaseTensorType>(bitwiseAndScalar.getType()).getDtype();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype,
                                      /*srcOriginalDtype=*/std::nullopt,
                                      /*dstOriginalDtype=*/resultElementType);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    return b.create<arith::AndIOp>(loc, self, other);
  }
  if (auto bitwiseOrTensor = dyn_cast<AtenBitwiseOrTensorOp>(op)) {
    if (isa<mlir::FloatType>(
            cast<ValueTensorType>(bitwiseOrTensor.getType()).getDtype())) {
      bitwiseOrTensor.emitError(
          "Bitwise_Or does not support floating point dtype");
      return nullptr;
    }
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(bitwiseOrTensor.getType()))
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::OrIOp>(loc, lhs, rhs);
  }
  if (auto bitwiseXorTensor = dyn_cast<AtenBitwiseXorTensorOp>(op)) {
    if (isa<mlir::FloatType>(
            cast<ValueTensorType>(bitwiseXorTensor.getType()).getDtype())) {
      bitwiseXorTensor.emitError(
          "Bitwise_Xor does not support floating point dtype");
      return nullptr;
    }
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(bitwiseXorTensor.getType()))
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::XOrIOp>(loc, lhs, rhs);
  }
  if (auto bitwiseRightShiftTensor =
          dyn_cast<AtenBitwiseRightShiftTensorOp>(op)) {
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(bitwiseRightShiftTensor.getType()))
                     .getElementType();
    if (!isa<mlir::IntegerType>(dtype)) {
      bitwiseRightShiftTensor.emitError(
          "Bitwise_Right_Shift op does not support non-integer input dtype.");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::ShRSIOp>(loc, lhs, rhs);
  }
  if (auto bitwiseLeftShiftTensor =
          dyn_cast<AtenBitwiseLeftShiftTensorOp>(op)) {
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(bitwiseLeftShiftTensor.getType()))
                     .getElementType();
    if (!isa<mlir::IntegerType>(dtype)) {
      bitwiseLeftShiftTensor.emitError(
          "Bitwise_Left_Shift op does not support non-integer input dtype.");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::ShLIOp>(loc, lhs, rhs);
  }
  if (isa<AtenLogicalOrOp, AtenLogicalAndOp, AtenLogicalXorOp>(op)) {
    MLIRContext *context = op->getContext();
    Type floatDtype = mlir::FloatType::getF64(context);
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], floatDtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], floatDtype);
    Value zero =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(floatDtype, 0));
    Value lhsTest = createNotEqual(b, loc, floatDtype, lhs, zero);
    Value rhsTest = createNotEqual(b, loc, floatDtype, rhs, zero);
    if (isa<AtenLogicalOrOp>(op)) {
      return b.create<arith::OrIOp>(loc, lhsTest, rhsTest);
    }
    if (isa<AtenLogicalAndOp>(op)) {
      return b.create<arith::AndIOp>(loc, lhsTest, rhsTest);
    }
    if (isa<AtenLogicalXorOp>(op)) {
      return b.create<arith::XOrIOp>(loc, lhsTest, rhsTest);
    }
    llvm_unreachable("Unknown op type");
  }
  if (isa<AtenLogicalNotOp>(op)) {
    MLIRContext *context = op->getContext();
    Type floatDtype = mlir::FloatType::getF64(context);
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], floatDtype);
    Value zero =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(floatDtype, 0));
    return createEqual(b, loc, floatDtype, self, zero);
  }
  if (isa<AtenAbsOp>(op)) {
    if (isa<IntegerType>(payloadArgs[0].getType()))
      return b.create<math::AbsIOp>(loc, payloadArgs[0]);
    return b.create<math::AbsFOp>(loc, payloadArgs[0]);
  }
  if (isa<AtenIsinfOp>(op)) {
    Value abs = b.create<math::AbsFOp>(loc, payloadArgs[0]);
    Value infinity = b.create<arith::ConstantOp>(
        loc,
        b.getFloatAttr(abs.getType(), std::numeric_limits<double>::infinity()));
    return createEqual(b, loc, abs.getType(), abs, infinity);
  }
  if (isa<AtenSigmoidOp>(op)) {
    Type inTTy = cast<ValueTensorType>(op->getOperand(0).getType()).getDtype();
    Type outTTy = cast<ValueTensorType>(op->getResult(0).getType()).getDtype();
    Type outTy = cast<RankedTensorType>(
                     converter->convertType(op->getResult(0).getType()))
                     .getElementType();
    Type computeTy = outTy;
    if (isa<IntegerType>(computeTy))
      computeTy = b.getF32Type();

    Value arg = payloadArgs[0];
    arg = convertScalarToDtype(b, loc, payloadArgs[0], computeTy, inTTy);
    auto negate = b.create<arith::NegFOp>(loc, arg);
    auto one =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(negate.getType(), 1));
    auto exp = b.create<math::ExpOp>(loc, negate);
    auto added = b.create<arith::AddFOp>(loc, exp, one);
    auto div = b.create<arith::DivFOp>(loc, one, added);
    return convertScalarToDtype(b, loc, div, outTy, std::nullopt, outTTy);
  }
  if (auto relu = dyn_cast<AtenReluOp>(op)) {
    Value zeroPoint = getZeroPoint(relu.getSelf());
    Value arg = payloadArgs[0];
    auto intType = dyn_cast<mlir::IntegerType>(arg.getType());
    if (zeroPoint && !intType) {
      relu.emitError("unimplemented: non-integer quantized Relu.");
      return nullptr;
    }
    auto reluTorchType = cast<ValueTensorType>(relu.getType());
    bool isUnsigned =
        torch_to_linalg::isUnsignedTorchType(reluTorchType.getDtype());
    if (zeroPoint) {
      int64_t zeroPointInt;
      int64_t width = intType.getWidth();
      assert(width < 64);
      int64_t minForIntType = isUnsigned ? 0 : -(1 << (width - 1));
      int64_t maxForIntType =
          isUnsigned ? (1 << (width + 1)) - 1 : (1 << (width - 1)) - 1;
      // check for constant zero point edge-cases:
      if (matchPattern(zeroPoint, m_TorchConstantInt(&zeroPointInt))) {
        if (zeroPointInt > maxForIntType) {
          // TODO: figure out how to handle this case:
          // current impl. quantizes output like input.
          // If zero point > maxForIntType, ordinary relu should return 0.
          // However, 0 isn't represented in such a quantization scheme.
          relu.emitError(
              "unimplemented: quantized relu for zero-point > max qint");
          return nullptr;
        }
        if (zeroPointInt < minForIntType)
          return arg;
      }
      zeroPoint = converter->materializeTargetConversion(
          b, loc, converter->convertType(zeroPoint.getType()), zeroPoint);
      auto minForIntTypeValue = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(zeroPoint.getType(), minForIntType));
      auto maxForIntTypeValue = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(zeroPoint.getType(), maxForIntType));
      auto zpLtMax = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             zeroPoint, maxForIntTypeValue);
      b.create<cf::AssertOp>(
          loc, zpLtMax,
          b.getStringAttr("Invalid Quantization: quantized relu with "
                          "zero-point > max qint"));
      auto zpLtMin = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             zeroPoint, minForIntTypeValue);
      zeroPoint = b.create<arith::SelectOp>(loc, zpLtMin, minForIntTypeValue,
                                            zeroPoint);
      zeroPoint = b.create<arith::TruncIOp>(loc, arg.getType(), zeroPoint);
    } else {
      zeroPoint =
          b.create<arith::ConstantOp>(loc, b.getZeroAttr(arg.getType()));
    }
    Value cmp;
    if (intType) {
      auto pred =
          isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
      cmp = b.create<arith::CmpIOp>(loc, pred, arg, zeroPoint);
    } else {
      cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, arg,
                                    zeroPoint);
    }
    return b.create<arith::SelectOp>(loc, cmp, arg, zeroPoint);
  }
  if (auto round = dyn_cast<AtenRoundOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(round.getType()).getDtype())) {
      round.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    return b.create<math::RoundEvenOp>(loc, payloadArgs[0]);
  }
  if (auto prelu = dyn_cast<AtenPreluOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(prelu.getType()).getDtype())) {
      prelu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type elementType = payloadArgs[0].getType();
    Value constZero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                         payloadArgs[0], constZero);
    Value positivePart =
        b.create<arith::SelectOp>(loc, pred, payloadArgs[0], constZero);
    Value negativePart =
        b.create<arith::SelectOp>(loc, pred, constZero, payloadArgs[0]);
    Value scale = convertScalarToDtype(b, loc, payloadArgs[1], elementType);
    Value scaledNegativePart =
        b.create<arith::MulFOp>(loc, negativePart, scale);
    return b.create<arith::AddFOp>(loc, positivePart, scaledNegativePart);
  }
  if (auto gelu = dyn_cast<AtenGeluOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(gelu.getType()).getDtype())) {
      gelu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    // TODO: Take approximation into account.
    std::string approximate;
    if (!matchPattern(gelu.getApproximate(), m_TorchConstantStr(approximate))) {
      gelu.emitError(
          "unimplemented: expected approximate to be a constant str");
      return nullptr;
    }
    if (approximate == "none") {
      Value multiplier = buildUnitNormalCdf(b, loc, payloadArgs[0]);
      return b.create<arith::MulFOp>(loc, payloadArgs[0], multiplier);
    }
    if (approximate == "tanh") {
      // GELU(x)=0.5∗x∗(1+Tanh((2/π)^1/2 * (x+0.044715∗x^3)))
      // Ref: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
      Value cstThree = b.create<arith::ConstantOp>(
          loc, IntegerAttr::get(IntegerType::get(op->getContext(), 64), 3));
      Value xCube = b.create<math::FPowIOp>(loc, payloadArgs[0], cstThree);
      Type elementType = payloadArgs[0].getType();
      Value cstAlpha = b.create<arith::ConstantOp>(
          loc, FloatAttr::get(elementType, 0.044715));
      Value xCubeMulAlpha = b.create<arith::MulFOp>(loc, xCube, cstAlpha);
      Value xPlusXCubeMulAlpha =
          b.create<arith::AddFOp>(loc, payloadArgs[0], xCubeMulAlpha);
      Value cstBeta = b.create<arith::ConstantOp>(
          loc, FloatAttr::get(elementType, 0.7977240352174656));
      Value betaMulX =
          b.create<arith::MulFOp>(loc, cstBeta, xPlusXCubeMulAlpha);
      Value tanh = b.create<math::TanhOp>(loc, betaMulX);
      Value cstOne =
          b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1.0));
      Value onePlusTanh = b.create<arith::AddFOp>(loc, cstOne, tanh);
      Value cstHalf =
          b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.5));
      Value multiplier = b.create<arith::MulFOp>(loc, cstHalf, onePlusTanh);
      return b.create<arith::MulFOp>(loc, payloadArgs[0], multiplier);
    }
    gelu.emitError("unimplemented: approximate value should be none or tanh");
    return nullptr;
  }
  if (auto geluBackward = dyn_cast<AtenGeluBackwardOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(geluBackward.getType()).getDtype())) {
      geluBackward.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    // TODO: Take approximation into account.
    std::string approximate;
    if (!matchPattern(geluBackward.getApproximate(),
                      m_TorchConstantStr(approximate)) ||
        approximate != "none")
      return nullptr;
    Type elementType = payloadArgs[1].getType();
    Value cstAlpha0 = b.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 1.12837916709551257390));
    Value cstAlpha1 = b.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.70710678118654752440));
    Value oneHalf =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.5));
    Value kAlpha = b.create<arith::MulFOp>(loc, cstAlpha0, cstAlpha1);
    Value kAlphaHalf = b.create<arith::MulFOp>(loc, kAlpha, oneHalf);
    Value negOneHalf =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, -0.5));
    Value inputSquared =
        b.create<arith::MulFOp>(loc, payloadArgs[1], payloadArgs[1]);
    Value negHalfInputSquared =
        b.create<arith::MulFOp>(loc, inputSquared, negOneHalf);
    Value dinput = b.create<math::ExpOp>(loc, negHalfInputSquared);
    Value cdf = buildUnitNormalCdf(b, loc, payloadArgs[1]);
    Value dinputInput = b.create<arith::MulFOp>(loc, dinput, payloadArgs[1]);
    Value dinputInputAlpha =
        b.create<arith::MulFOp>(loc, dinputInput, kAlphaHalf);
    Value cdfExt = b.create<arith::AddFOp>(loc, dinputInputAlpha, cdf);
    return b.create<arith::MulFOp>(loc, payloadArgs[0], cdfExt);
  }
  if (auto hardtanhBackward = dyn_cast<AtenHardtanhBackwardOp>(op)) {
    AtenHardtanhBackwardOp::Adaptor adaptor(operands);
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(hardtanhBackward.getType()).getDtype())) {
      hardtanhBackward.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value gradOutput = payloadArgs[0];
    Type elementType = gradOutput.getType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[1], elementType);
    Value constantZero =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    Value min = convertScalarToDtype(b, loc, adaptor.getMinVal(), elementType);
    Value max = convertScalarToDtype(b, loc, adaptor.getMaxVal(), elementType);
    Value lesser =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT, self, min);
    Value greater =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, self, max);
    Value cmp = b.create<arith::OrIOp>(loc, lesser, greater);
    return b.create<arith::SelectOp>(loc, cmp, constantZero, gradOutput);
  }
  if (auto add = dyn_cast<AtenAddTensorOp>(op)) {
    AtenAddTensorOp::Adaptor adaptor(operands);
    Type resultElementType = cast<BaseTensorType>(add.getType()).getDtype();
    Type dtype = cast<RankedTensorType>(converter->convertType(add.getType()))
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype,
                                     /*srcOriginalDtype=*/std::nullopt,
                                     /*dstOriginalDtype=*/resultElementType);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype,
                                     /*srcOriginalDtype=*/std::nullopt,
                                     /*dstOriginalDtype=*/resultElementType);
    Value alpha = convertScalarToDtype(b, loc, adaptor.getAlpha(), dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    if (isa<mlir::FloatType>(dtype)) {
      Value scaled = b.create<arith::MulFOp>(loc, rhs, alpha);
      return b.create<arith::AddFOp>(loc, lhs, scaled);
    } else {
      Value scaled = b.create<arith::MulIOp>(loc, rhs, alpha);
      return b.create<arith::AddIOp>(loc, lhs, scaled);
    }
  }
  if (auto sub = dyn_cast<AtenSubTensorOp>(op)) {
    AtenSubTensorOp::Adaptor adaptor(operands);
    Type dtype = cast<RankedTensorType>(converter->convertType(sub.getType()))
                     .getElementType();
    Type resultElementType = cast<BaseTensorType>(sub.getType()).getDtype();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype,
                                     /*srcOriginalDtype=*/std::nullopt,
                                     /*dstOriginalDtype=*/resultElementType);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype,
                                     /*srcOriginalDtype=*/std::nullopt,
                                     /*dstOriginalDtype=*/resultElementType);
    Value alpha = convertScalarToDtype(b, loc, adaptor.getAlpha(), dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType,
                                       /*originalScalar=*/sub.getAlpha());
    if (isa<mlir::FloatType>(dtype)) {
      Value scaled = b.create<arith::MulFOp>(loc, rhs, alpha);
      return b.create<arith::SubFOp>(loc, lhs, scaled);
    } else {
      Value scaled = b.create<arith::MulIOp>(loc, rhs, alpha);
      return b.create<arith::SubIOp>(loc, lhs, scaled);
    }
  }
  if (auto lshiftScalar = dyn_cast<Aten__Lshift__ScalarOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(lshiftScalar.getType()))
            .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other =
        convertScalarToDtype(b, loc, operands[1], dtype,
                             /*srcOriginalDtype=*/operands[1].getType(),
                             /*dstOriginalDtype=*/dtype);
    return b.create<arith::ShLIOp>(loc, self, other);
  }
  if (auto rshiftScalar = dyn_cast<Aten__Rshift__ScalarOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(rshiftScalar.getType()))
            .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other =
        convertScalarToDtype(b, loc, operands[1], dtype,
                             /*srcOriginalDtype=*/operands[1].getType(),
                             /*dstOriginalDtype=*/dtype);
    return b.create<arith::ShRUIOp>(loc, self, other);
  }
  if (auto subScalar = dyn_cast<AtenSubScalarOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(subScalar.getType()))
            .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(
        b, loc, operands[2], dtype, /*srcOriginalDtype=*/operands[2].getType(),
        /*dstOriginalDtype=*/dtype);
    if (isa<mlir::FloatType>(dtype)) {
      Value mult = b.create<arith::MulFOp>(loc, other, alpha);
      return b.create<arith::SubFOp>(loc, self, mult);
    } else if (isa<mlir::IntegerType>(dtype)) {
      Value mult = b.create<arith::MulIOp>(loc, other, alpha);
      return b.create<arith::SubIOp>(loc, self, mult);
    }
    subScalar.emitError("unimplemented: dtype other than float and integer "
                        "types are not supported.");
    return nullptr;
  }
  if (auto addScalar = dyn_cast<AtenAddScalarOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(addScalar.getType()))
            .getElementType();
    Type resultElementType =
        cast<BaseTensorType>(addScalar.getType()).getDtype();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype,
                                      /*srcOriginalDtype=*/std::nullopt,
                                      /*dstOriginalDtype=*/resultElementType);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    if (isa<mlir::FloatType>(dtype)) {
      Value mult = b.create<arith::MulFOp>(loc, other, alpha);
      return b.create<arith::AddFOp>(loc, self, mult);
    } else if (isa<mlir::IntegerType>(dtype)) {
      Value mult = b.create<arith::MulIOp>(loc, other, alpha);
      return b.create<arith::AddIOp>(loc, self, mult);
    }
    addScalar.emitError("unimplemented: dtype other than float and integer "
                        "types are not supported.");
    return nullptr;
  }
  if (auto mul = dyn_cast<AtenMulTensorOp>(op)) {
    AtenMulTensorOp::Adaptor adaptor(operands);
    Type dtype = cast<RankedTensorType>(converter->convertType(mul.getType()))
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    if (isa<mlir::FloatType>(dtype)) {
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    } else if (isa<mlir::ComplexType>(dtype)) {
      return b.create<complex::MulOp>(loc, lhs, rhs);
    } else {
      return b.create<arith::MulIOp>(loc, lhs, rhs);
    }
  }
  if (auto atan2 = dyn_cast<AtenAtan2Op>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(atan2.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      atan2.emitError("Atan2 requires floating point result type");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<math::Atan2Op>(loc, lhs, rhs);
  }
  if (auto ltTensor = dyn_cast<AtenLtTensorOp>(op)) {
    return createCompareOp(b, loc, ltTensor, payloadArgs[0], payloadArgs[1]);
  }
  if (auto leTensor = dyn_cast<AtenLeTensorOp>(op)) {
    return createCompareOp(b, loc, leTensor, payloadArgs[0], payloadArgs[1]);
  }
  if (auto gtTensor = dyn_cast<AtenGtTensorOp>(op)) {
    return createCompareOp(b, loc, gtTensor, payloadArgs[0], payloadArgs[1]);
  }
  if (auto geTensor = dyn_cast<AtenGeTensorOp>(op)) {
    return createCompareOp(b, loc, geTensor, payloadArgs[0], payloadArgs[1]);
  }
  if (auto eqTensor = dyn_cast<AtenEqTensorOp>(op)) {
    return createCompareOp(b, loc, eqTensor, payloadArgs[0], payloadArgs[1]);
  }
  if (auto neTensor = dyn_cast<AtenNeTensorOp>(op)) {
    return createCompareOp(b, loc, neTensor, payloadArgs[0], payloadArgs[1]);
  }
  if (auto div = dyn_cast<AtenDivTensorOp>(op)) {
    AtenDivTensorOp::Adaptor adaptor(operands);
    Type dtype = cast<RankedTensorType>(converter->convertType(div.getType()))
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    if (isa<mlir::FloatType>(dtype))
      return b.create<arith::DivFOp>(loc, lhs, rhs);
    else if (isa<mlir::IntegerType>(dtype)) {
      if (dtype.isUnsignedInteger())
        return b.create<arith::DivUIOp>(loc, lhs, rhs);
      return b.create<arith::DivSIOp>(loc, lhs, rhs);
    }
    div.emitError("unimplemented: non-floating point and non-integer dtype");
    return nullptr;
  }
  if (auto divScalarMode = dyn_cast<AtenDivScalarModeOp>(op)) {
    return createDivModePayload(b, loc, converter, payloadArgs, divScalarMode,
                                operands);
  }
  if (auto divTensorMode = dyn_cast<AtenDivTensorModeOp>(op)) {
    return createDivModePayload(b, loc, converter, payloadArgs, divTensorMode,
                                operands);
  }
  if (auto pow = dyn_cast<AtenPowScalarOp>(op)) {
    Type dtype = cast<ValueTensorType>(pow.getType()).getDtype();
    if (!isa<mlir::FloatType>(dtype)) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value selfPromoted = convertScalarToDtype(b, loc, operands[0], dtype);
    Value expPromoted = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    return b.create<math::PowFOp>(loc, selfPromoted, expPromoted);
  }

  if (auto pow = dyn_cast<AtenPowTensorScalarOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(pow.getType()).getDtype())) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type dtype = cast<ValueTensorType>(pow.getSelf().getType()).getDtype();
    Value expPromoted = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<math::PowFOp>(loc, payloadArgs[0], expPromoted);
  }

  if (auto pow = dyn_cast<AtenPowTensorTensorOp>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(pow.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<math::PowFOp>(loc, lhs, rhs);
  }

  if (auto imag = dyn_cast<AtenImagOp>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(imag.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      imag.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value imagVal = b.create<complex::ImOp>(loc, payloadArgs[0]);
    return imagVal;
  }

  if (auto real = dyn_cast<AtenRealOp>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(real.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      real.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value realVal = b.create<complex::ReOp>(loc, payloadArgs[0]);
    return realVal;
  }

  if (auto gtScalar = dyn_cast<AtenGtScalarOp>(op)) {
    return createCompareOp(b, loc, gtScalar, payloadArgs[0], operands[1]);
  }

  if (auto geScalar = dyn_cast<AtenGeScalarOp>(op)) {
    return createCompareOp(b, loc, geScalar, payloadArgs[0], operands[1]);
  }

  if (auto eqScalar = dyn_cast<AtenEqScalarOp>(op)) {
    return createCompareOp(b, loc, eqScalar, payloadArgs[0], operands[1]);
  }

  if (auto neScalar = dyn_cast<AtenNeScalarOp>(op)) {
    return createCompareOp(b, loc, neScalar, payloadArgs[0], operands[1]);
  }

  if (auto ltScalar = dyn_cast<AtenLtScalarOp>(op)) {
    return createCompareOp(b, loc, ltScalar, payloadArgs[0], operands[1]);
  }

  if (auto leScalar = dyn_cast<AtenLeScalarOp>(op)) {
    return createCompareOp(b, loc, leScalar, payloadArgs[0], operands[1]);
  }

  if (auto whereSelf = dyn_cast<AtenWhereSelfOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(whereSelf.getType()))
            .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[2], dtype);
    return b.create<arith::SelectOp>(loc, payloadArgs[0], lhs, rhs);
  }

  if (auto lerp = dyn_cast<AtenLerpTensorOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(lerp.getType()).getDtype())) {
      lerp.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    AtenLerpTensorOp::Adaptor adaptor(payloadArgs);
    auto start = adaptor.getSelf();
    auto end = adaptor.getEnd();
    auto weight = adaptor.getWeight();
    auto delta = b.create<arith::SubFOp>(loc, end, start);
    auto weightedDelta = b.create<arith::MulFOp>(loc, delta, weight);
    return b.create<arith::AddFOp>(loc, start, weightedDelta);
  }
  if (auto minimum = dyn_cast<AtenMinimumOp>(op)) {
    Type dtype = cast<BaseTensorType>(minimum.getType()).getDtype();
    Type elemTy =
        cast<RankedTensorType>(converter->convertType(minimum.getType()))
            .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], elemTy);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], elemTy);
    Value pred = createLessThan(b, loc, dtype, lhs, rhs);
    return b.create<arith::SelectOp>(loc, pred, lhs, rhs);
  }
  if (auto maximum = dyn_cast<AtenMaximumOp>(op)) {
    Type dtype = cast<BaseTensorType>(maximum.getType()).getDtype();
    Type elemTy =
        cast<RankedTensorType>(converter->convertType(maximum.getType()))
            .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], elemTy);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], elemTy);
    Value pred = createGreaterThan(b, loc, dtype, lhs, rhs);
    return b.create<arith::SelectOp>(loc, pred, lhs, rhs);
  }
  if (auto clamp = dyn_cast<AtenClampOp>(op)) {
    AtenClampOp::Adaptor adaptor(operands);
    auto min = adaptor.getMin();
    auto max = adaptor.getMax();
    if (isa<Torch::OptionalType>(min.getType()) ||
        isa<Torch::OptionalType>(max.getType())) {
      clamp.emitError("unimplemented: runtime optional type");
      return nullptr;
    }

    Type dtype = cast<RankedTensorType>(converter->convertType(clamp.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType, mlir::IntegerType>(dtype)) {
      clamp.emitError("unimplement type for clamp");
      return nullptr;
    }

    Type dstOriginalDtype = cast<BaseTensorType>(clamp.getType()).getDtype();
    bool isUnsigned = isa<QUInt8Type>(dstOriginalDtype);
    if (auto intTy = dyn_cast<IntegerType>(dstOriginalDtype)) {
      isUnsigned = intTy.isUnsigned();
    }
    auto cmpSelect = [&](Value input, Value clamp, bool getMax) -> Value {
      clamp = convertScalarToDtype(b, loc, clamp, dtype,
                                   /*srcOriginalDtype=*/std::nullopt,
                                   /*dstOriginalDtype=*/dstOriginalDtype);

      Value pred;
      if (isa<mlir::FloatType>(dtype)) {
        auto cmp =
            getMax ? arith::CmpFPredicate::UGT : arith::CmpFPredicate::ULT;
        pred = b.create<arith::CmpFOp>(loc, cmp, input, clamp);
      } else if (isa<mlir::IntegerType>(dtype)) {
        auto cmp =
            isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
        if (getMax)
          cmp = arith::invertPredicate(cmp);
        pred = b.create<arith::CmpIOp>(loc, cmp, input, clamp);
      }
      return b.create<arith::SelectOp>(loc, pred, clamp, input);
    };

    auto result = payloadArgs[0];
    if (!isa<Torch::NoneType>(min.getType()))
      result = cmpSelect(result, min, /*getMax=*/false);
    if (!isa<Torch::NoneType>(max.getType()))
      result = cmpSelect(result, max, /*getMax=*/true);
    return result;
  }
  if (auto clampTensor = dyn_cast<AtenClampTensorOp>(op)) {
    AtenClampTensorOp::Adaptor adaptor(operands);
    auto min = adaptor.getMin();
    auto max = adaptor.getMax();
    if (isa<Torch::OptionalType>(min.getType()) ||
        isa<Torch::OptionalType>(max.getType())) {
      clampTensor.emitError("unimplemented: runtime optional type");
      return nullptr;
    }
    Type dtype =
        cast<RankedTensorType>(converter->convertType(clampTensor.getType()))
            .getElementType();
    bool isMinNone = true;
    auto result = payloadArgs[0];
    if (!isa<Torch::NoneType>(min.getType())) {
      isMinNone = false;
      auto minPromoted = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
      Value pred;
      if (isa<mlir::FloatType>(dtype)) {
        pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT, result,
                                       minPromoted);
      } else if (isa<mlir::IntegerType>(dtype)) {
        pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, result,
                                       minPromoted);
      } else {
        clampTensor.emitError(
            "unimplemented: dtype other than float and integer "
            "types are not supported.");
        return nullptr;
      }
      result = b.create<arith::SelectOp>(loc, pred, minPromoted, result);
    }
    if (!isa<Torch::NoneType>(max.getType())) {
      max = isMinNone ? payloadArgs[1] : payloadArgs[2];
      auto maxPromoted = convertScalarToDtype(b, loc, max, dtype);
      Value pred;
      if (isa<mlir::FloatType>(dtype)) {
        pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, result,
                                       maxPromoted);
      } else if (isa<mlir::IntegerType>(dtype)) {
        pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, result,
                                       maxPromoted);
      } else {
        clampTensor.emitError(
            "unimplemented: dtype other than float and integer "
            "types are not supported.");
        return nullptr;
      }
      result = b.create<arith::SelectOp>(loc, pred, maxPromoted, result);
    }
    return result;
  }
  if (auto rsub = dyn_cast<AtenRsubScalarOp>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(rsub.getType()))
                     .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(
        b, loc, operands[2], dtype, /*srcOriginalDtype=*/operands[2].getType(),
        /*dstOriginalDtype=*/dtype);
    if (isa<mlir::FloatType>(dtype)) {
      Value mult = b.create<arith::MulFOp>(loc, self, alpha);
      return b.create<arith::SubFOp>(loc, other, mult);
    } else if (isa<mlir::IntegerType>(dtype)) {
      Value mult = b.create<arith::MulIOp>(loc, self, alpha);
      return b.create<arith::SubIOp>(loc, other, mult);
    }
    rsub.emitError("unimplemented: dtype other than float and integer "
                   "types are not supported.");
    return nullptr;
  }
  if (auto mulScalar = dyn_cast<AtenMulScalarOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(mulScalar.getType()))
            .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, operands[1], dtype);
    if (isa<mlir::FloatType>(dtype))
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    if (isa<mlir::IntegerType>(dtype))
      return b.create<arith::MulIOp>(loc, lhs, rhs);
    mulScalar.emitError("unimplemented: Only integer/float dtype supported");
    return nullptr;
  }
  if (auto atenToDtype = dyn_cast<AtenToDtypeOp>(op)) {
    Value input = payloadArgs[0];
    Type inputElementType =
        cast<BaseTensorType>(atenToDtype.getSelf().getType()).getDtype();
    Type dtype =
        cast<RankedTensorType>(converter->convertType(atenToDtype.getType()))
            .getElementType();
    Type resultElementType;
    int64_t dtypeInt;
    if (!matchPattern(atenToDtype.getDtype(), m_TorchConstantInt(&dtypeInt))) {
      atenToDtype.emitError("unimplemented: dtype must be a constant integer");
      return nullptr;
    }
    FailureOr<Type> maybeResultElementType =
        torch_to_linalg::getBackendTypeForScalarType(
            atenToDtype->getContext(), (torch_upstream::ScalarType)dtypeInt);
    if (failed(maybeResultElementType)) {
      atenToDtype.emitError("unable to convert `dtypeInt` to builtin type");
      return nullptr;
    }
    resultElementType = *maybeResultElementType;
    Value result = convertScalarToDtype(b, loc, input, dtype,
                                        /*srcOriginalDtype=*/inputElementType,
                                        /*dstOriginalDtype=*/resultElementType);
    return result;
  }
  if (auto divScalar = dyn_cast<AtenDivScalarOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(divScalar.getType()))
            .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      divScalar.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<arith::DivFOp>(loc, self, other);
  }
  if (auto remScalar = dyn_cast<AtenRemainderScalarOp>(op)) {
    return createRemainderPayload(b, loc, converter, payloadArgs, remScalar,
                                  operands);
  }
  if (auto remTensor = dyn_cast<AtenRemainderTensorOp>(op)) {
    return createRemainderPayload(b, loc, converter, payloadArgs, remTensor,
                                  operands);
  }
  if (auto reciprocal = dyn_cast<AtenReciprocalOp>(op)) {
    Type dtype =
        cast<RankedTensorType>(converter->convertType(reciprocal.getType()))
            .getElementType();
    Value arg = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Type elementType = arg.getType();
    // assert(element != 0)
    auto zero =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    auto pred =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, arg, zero);
    b.create<cf::AssertOp>(
        loc, pred, b.getStringAttr("unimplemented: tensor with zero element"));

    auto one =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1.0));
    return b.create<arith::DivFOp>(loc, one, arg);
  }
  if (auto thresholdOp = dyn_cast<AtenThresholdOp>(op)) {
    // The approach used here is as follows:
    //        result = self <= threshold ? value : self
    AtenThresholdOp::Adaptor adaptor(operands);
    Type dtype =
        cast<RankedTensorType>(converter->convertType(thresholdOp.getType()))
            .getElementType();

    Value self = payloadArgs[0];
    Value threshold =
        convertScalarToDtype(b, loc, adaptor.getThreshold(), dtype);
    Value value = convertScalarToDtype(b, loc, adaptor.getValue(), dtype);

    Value predicate;
    if (isa<mlir::FloatType>(dtype))
      predicate = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE, self,
                                          threshold);
    else
      predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, self,
                                          threshold);
    return b.create<arith::SelectOp>(loc, predicate, value, self);
  }
  if (auto thresholdBackward = dyn_cast<AtenThresholdBackwardOp>(op)) {
    // The approach used here is as follows:
    //        result = self <= threshold ? 0 : grad
    AtenThresholdBackwardOp::Adaptor adaptor(operands);
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(thresholdBackward.getType()))
                     .getElementType();

    Value grad = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value self = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value threshold =
        convertScalarToDtype(b, loc, adaptor.getThreshold(), dtype);
    Value constantZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(dtype));

    Value predicate;
    if (isa<mlir::FloatType>(dtype))
      predicate = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE, self,
                                          threshold);
    else
      predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, self,
                                          threshold);
    return b.create<arith::SelectOp>(loc, predicate, constantZero, grad);
  }
  if (auto fillScalar = dyn_cast<AtenFillScalarOp>(op)) {
    AtenFillScalarOp::Adaptor adaptor(operands);
    Type dtype =
        cast<RankedTensorType>(converter->convertType(fillScalar.getType()))
            .getElementType();
    return convertScalarToDtype(b, loc, adaptor.getValue(), dtype);
  }
  if (auto maskedFillTensor = dyn_cast<AtenMaskedFillTensorOp>(op)) {
    AtenMaskedFillScalarOp::Adaptor adaptor(operands);
    Type dtype = cast<RankedTensorType>(
                     converter->convertType(maskedFillTensor.getType()))
                     .getElementType();

    Value input = payloadArgs[0];
    Value mask = payloadArgs[1];
    Value fillValue = convertScalarToDtype(b, loc, payloadArgs[2], dtype);
    return b.create<arith::SelectOp>(loc, mask, fillValue, input);
  }
  if (auto fillTensor = dyn_cast<AtenFillTensorOp>(op)) {
    AtenFillTensorOp::Adaptor adaptor(operands);
    Type dtype =
        cast<RankedTensorType>(converter->convertType(fillTensor.getType()))
            .getElementType();
    return convertScalarToDtype(b, loc, payloadArgs[1], dtype);
  }

  if (auto triu = dyn_cast<AtenTriuOp>(op)) {
    Value result;
    if (failed(createTriangularMatrix<arith::CmpIPredicate::sge>(
            b, loc, payloadArgs, op, operands, result)))
      return nullptr;
    return result;
  }

  if (auto tril = dyn_cast<AtenTrilOp>(op)) {
    Value result;
    if (failed(createTriangularMatrix<arith::CmpIPredicate::sle>(
            b, loc, payloadArgs, op, operands, result)))
      return nullptr;
    return result;
  }

  if (auto bitwiseNot = dyn_cast<AtenBitwiseNotOp>(op)) {
    Type elementType =
        cast<RankedTensorType>(converter->convertType(bitwiseNot.getType()))
            .getElementType();
    if (isa<mlir::FloatType>(elementType)) {
      bitwiseNot.emitError("Bitwise_Not does not support floating point dtype");
      return nullptr;
    }

    Value allOnesVal = b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(
                 elementType,
                 APSInt::getAllOnes(elementType.getIntOrFloatBitWidth())));
    return b.create<arith::XOrIOp>(loc, payloadArgs[0], allOnesVal);
  }

  if (isa<AtenDequantizeTensorOp, AtenDequantizeSelfOp>(op)) {
    auto value = payloadArgs[0];
    auto valueTy = value.getType();
    auto qtensor = op->getOperand(0);
    auto qtensorTy = cast<ValueTensorType>(qtensor.getType()).getDtype();

    Value zp, scale;
    if (auto makeQTensor =
            qtensor.getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
      zp = makeQTensor.getZeroPoint();
      scale = makeQTensor.getScale();
    }

    if (auto quant = qtensor.getDefiningOp<AtenQuantizePerTensorOp>()) {
      zp = quant.getZeroPoint();
      scale = quant.getScale();
    }

    if (!zp || !scale) {
      return nullptr;
    }

    auto outFpTy = payloadArgs[1].getType();
    auto outBw = outFpTy.getIntOrFloatBitWidth();
    auto outIntTy = b.getIntegerType(outBw);

    if (valueTy != outIntTy) {
      if (torch_to_linalg::isUnsignedTorchType(qtensorTy)) {
        value = b.create<arith::ExtUIOp>(loc, outIntTy, value);
      } else {
        value = b.create<arith::ExtSIOp>(loc, outIntTy, value);
      }
    }

    zp = converter->materializeTargetConversion(
        b, loc, converter->convertType(zp.getType()), zp);
    auto zpTy = zp.getType();

    if (zpTy != outIntTy) {
      zp = b.create<arith::TruncIOp>(loc, outIntTy, zp);
    }

    value = b.create<arith::SubIOp>(loc, value, zp);
    // treat the i32 as a signed int regardless of original signed-ness
    // this will prevent overflow from subtraction for unsigned quantizations.
    value = b.create<arith::SIToFPOp>(loc, outFpTy, value);

    scale = converter->materializeTargetConversion(
        b, loc, converter->convertType(scale.getType()), scale);
    if (scale.getType() != value.getType()) {
      scale = b.create<arith::TruncFOp>(loc, value.getType(), scale);
    }
    value = b.create<arith::MulFOp>(loc, value, scale);
    return value;
  }

  if (auto quant = dyn_cast<AtenQuantizePerTensorOp>(op)) {
    Value value = payloadArgs[0];
    Value scale = quant.getScale();
    Value zp = quant.getZeroPoint();
    auto valueTy = value.getType();

    zp = converter->materializeTargetConversion(
        b, loc, converter->convertType(zp.getType()), zp);
    zp = b.create<arith::SIToFPOp>(loc, valueTy, zp);

    scale = converter->materializeTargetConversion(
        b, loc, converter->convertType(scale.getType()), scale);
    scale = b.create<arith::TruncFOp>(loc, valueTy, scale);

    value = b.create<arith::DivFOp>(loc, value, scale);
    value = b.create<math::RoundEvenOp>(loc, value);
    value = b.create<arith::AddFOp>(loc, value, zp);

    auto destTy = payloadArgs[1].getType();
    auto bitwidth = destTy.getIntOrFloatBitWidth();
    bool isUnsigned = torch_to_linalg::isUnsignedTorchType(quant.getType());
    APInt min = isUnsigned ? APInt::getMinValue(bitwidth)
                           : APInt::getSignedMinValue(bitwidth);
    APInt max = isUnsigned ? APInt::getMaxValue(bitwidth)
                           : APInt::getSignedMaxValue(bitwidth);

    double minI = isUnsigned ? static_cast<double>(min.getZExtValue())
                             : static_cast<double>(min.getSExtValue());
    double maxI = isUnsigned ? static_cast<double>(max.getZExtValue())
                             : static_cast<double>(max.getSExtValue());
    Value minVal =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(valueTy, minI));
    Value maxVal =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(valueTy, maxI));
    value = b.create<arith::MaximumFOp>(loc, value, minVal);
    value = b.create<arith::MinimumFOp>(loc, value, maxVal);

    if (isUnsigned) {
      value = b.create<arith::FPToUIOp>(loc, destTy, value);
    } else {
      value = b.create<arith::FPToSIOp>(loc, destTy, value);
    }

    return value;
  }

  if (auto isClose = dyn_cast<AtenIscloseOp>(op)) {
    double rtol, atol;
    bool equalNan;
    if (!matchPattern(isClose.getRtol(), m_TorchConstantFloat(&rtol))) {
      isClose.emitError("rtol must be a scalar constant");
      return nullptr;
    }
    if (!matchPattern(isClose.getAtol(), m_TorchConstantFloat(&atol))) {
      isClose.emitError("atol must be a scalar constant");
      return nullptr;
    }
    if (!matchPattern(isClose.getEqualNan(), m_TorchConstantBool(&equalNan))) {
      isClose.emitError("unimplemented: equal_nan is expected to be false");
      return nullptr;
    }
    auto lhsType = mlir::dyn_cast<mlir::FloatType>(payloadArgs[0].getType());
    auto rhsType = mlir::dyn_cast<mlir::FloatType>(payloadArgs[1].getType());
    if (!lhsType || !rhsType) {
      isClose.emitError("unimplemented: only FP element type is supported");
      return nullptr;
    }
    // Choose the widest float type as compute type.
    auto computeType =
        lhsType.getWidth() > rhsType.getWidth() ? lhsType : rhsType;
    computeType = computeType.getWidth() >= 32 ? computeType : b.getF32Type();
    auto cvtArg0 = convertScalarToDtype(b, loc, payloadArgs[0], computeType);
    auto cvtArg1 = convertScalarToDtype(b, loc, payloadArgs[1], computeType);
    // Reference to the definition of torch.isclose:
    //   ∣input − other∣ <= atol + rtol × ∣other∣
    auto diff = b.create<arith::SubFOp>(loc, computeType, cvtArg0, cvtArg1);
    auto absDiff = b.create<math::AbsFOp>(loc, computeType, diff);
    auto cstRtol =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(computeType, rtol));
    auto absOther = b.create<math::AbsFOp>(loc, computeType, cvtArg1);
    auto mul = b.create<arith::MulFOp>(loc, computeType, cstRtol, absOther);
    auto cstAtol =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(computeType, atol));
    auto threshold = b.create<arith::AddFOp>(loc, computeType, cstAtol, mul);
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE, absDiff,
                                   threshold);
  }

  op->emitError("unimplemented lowering in "
                "createLinalgPayloadCalculationForElementwiseOp");
  return nullptr;
}

namespace {
// Converts an elementwise op.
// This specifically includes:
// - converting elementwise ops of any tensor arity
// - converting elementwise ops with any number of scalar captures (such as a
//   scalar alpha to torch.aten.Add)
// - broadcasting of static size-1 dimensions
//
// Currently, we adopt the behavior that "size 1" broadcasting is a runtime
// error if it happens dynamically.
//
// Looking forward a bit, eventually, it probably makes sense to have
// a "linalg.generic-like" op for modeling a fused subgraph of numpy-broadcasted
// operands. Modeling elementwise ops that way is potentially useful to allow a
// more centralized reasoning about multiversioning. However a cost model will
// be needed for "pre-fusing" elementwise ops that way, as it can potentially be
// a pessimization. A mild extension of this pattern should work for such a
// general op.
class ConvertElementwiseOp : public ConversionPattern {
public:
  ConvertElementwiseOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<AtenTanOp, AtenTanhOp, AtenSinhOp, AtenCoshOp, AtenReluOp,
             AtenPreluOp, AtenGeluOp, AtenGeluBackwardOp, AtenAddTensorOp,
             AtenMulTensorOp, AtenDivTensorOp, AtenDivTensorModeOp,
             AtenDivScalarModeOp, AtenSubTensorOp, AtenAtan2Op,
             AtenLerpTensorOp, AtenSigmoidOp, AtenExpOp, AtenExpm1Op,
             AtenMinimumOp, AtenMaximumOp, AtenToDtypeOp, AtenClampOp,
             AtenClampTensorOp, AtenRsubScalarOp, AtenMulScalarOp, AtenLogOp,
             AtenErfOp, AtenSqrtOp, AtenFloorOp, AtenPowScalarOp,
             AtenPowTensorScalarOp, AtenPowTensorTensorOp, AtenLog2Op,
             AtenLog10Op, AtenLog1pOp, AtenRsqrtOp, AtenDivScalarOp,
             AtenRemainderScalarOp, AtenRemainderTensorOp, AtenAbsOp,
             AtenReciprocalOp, AtenBitwiseAndTensorOp, AtenBitwiseAndScalarOp,
             AtenBitwiseOrTensorOp, AtenBitwiseXorTensorOp,
             AtenBitwiseLeftShiftTensorOp, AtenBitwiseRightShiftTensorOp,
             Aten__Lshift__ScalarOp, Aten__Rshift__ScalarOp, AtenGtScalarOp,
             AtenGeScalarOp, AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp,
             AtenWhereSelfOp, AtenCeilOp, AtenGtTensorOp, AtenGeTensorOp,
             AtenEqTensorOp, AtenNeTensorOp, AtenLtTensorOp, AtenLeTensorOp,
             AtenSubScalarOp, AtenAddScalarOp, AtenThresholdOp,
             AtenThresholdBackwardOp, AtenHardtanhBackwardOp, AtenCloneOp,
             AtenSinOp, AtenCosOp, AtenNeScalarOp, AtenNegOp,
             AtenMaskedFillTensorOp, AtenLogicalOrOp, AtenLogicalAndOp,
             AtenLogicalXorOp, AtenLogicalNotOp, AtenIsinfOp, AtenTriuOp,
             AtenTrilOp, AtenBitwiseNotOp, AtenRoundOp, AtenFillScalarOp,
             AtenFillTensorOp, AtenAtanOp, AtenAcosOp, AtenAtanhOp, AtenAcoshOp,
             AtenAsinOp, AtenAsinhOp, AtenRealOp, AtenImagOp,
             AtenDequantizeSelfOp, AtenDequantizeTensorOp,
             AtenQuantizePerTensorOp, AtenIscloseOp>(op))
      return rewriter.notifyMatchFailure(op, "not a supported elementwise op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    auto tensorOperands = llvm::to_vector<6>(llvm::make_filter_range(
        operands, [](Value v) { return isa<RankedTensorType>(v.getType()); }));
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    bool hadErrorCreatingPayload = false;
    Value generic = torch_to_linalg::createElementwiseLinalgGeneric(
        rewriter, loc, tensorOperands, resultType.getElementType(),
        [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
          Value result = createLinalgPayloadCalculationForElementwiseOp(
              b, loc, getTypeConverter(), payloadArgs, op, operands);
          if (!result) {
            hadErrorCreatingPayload = true;
            return;
          }
          b.create<linalg::YieldOp>(loc, result);
        });
    if (hadErrorCreatingPayload)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, generic);
    return success();
  }
};
} // namespace

// Given `input`, `target`, `nll_loss_forward` is given by:
//   for i in range(0, len(target)):
//     indi = target[i];
//     nll_loss_forward[i] = -(input[i][indi]);
// TODO: `weight`operand is still to be taken care of.
namespace {

class ConvertAtenNllLossForwardOp
    : public OpConversionPattern<AtenNllLossForwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNllLossForwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value input = adaptor.getSelf();
    Value target = adaptor.getTarget();
    Value weight = adaptor.getWeight();

    int64_t reduction;
    if (!matchPattern(op.getReduction(), m_TorchConstantInt(&reduction)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    // TODO: Incorporate the weight argument.
    if (!isa<mlir::torch::Torch::NoneType>(weight.getType()))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    Value ignoreIndex = adaptor.getIgnoreIndex();
    Value ignoreIndexVal = castIntToIndex(rewriter, loc, ignoreIndex);

    unsigned inputRank = cast<RankedTensorType>(input.getType()).getRank();
    unsigned targetRank = cast<RankedTensorType>(target.getType()).getRank();

    // TODO: Add support for k-dim loss.
    if (inputRank > 2) {
      return rewriter.notifyMatchFailure(
          op, "expected input and target to be rank <= 2");
    }
    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    Type elementType = resultType.getElementType();

    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    Value finalRes = torch_to_linalg::createElementwiseLinalgGeneric(
        rewriter, loc, {target}, elementType,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value targetVal = args[0];
          Value indTarget = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), targetVal);

          // The final result is given by:
          // final_res = (indTarget == ignoreIndexVal) ? 0 :
          // input[indI][IndTarget]
          Value cmpEq = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, indTarget, ignoreIndexVal);

          SmallVector<Value> extractionIndices{indTarget};
          if (inputRank == 2) {
            Value indI = rewriter.create<linalg::IndexOp>(loc, 0);
            extractionIndices.insert(extractionIndices.begin(), indI);
          }

          Value result =
              rewriter.create<tensor::ExtractOp>(loc, input, extractionIndices);

          Value negate =
              rewriter.create<arith::NegFOp>(loc, elementType, result);
          Value selectFinal =
              rewriter.create<arith::SelectOp>(loc, cmpEq, zeroVal, negate);
          b.create<linalg::YieldOp>(loc, selectFinal);
        });

    llvm::iota_range<int64_t> dimsToReduce(0, targetRank,
                                           /*inclusive=*/false);
    DenseSet<int64_t> dimSet(dimsToReduce.begin(), dimsToReduce.end());

    if (reduction == torch_upstream::Reduction::Sum ||
        reduction == torch_upstream::Reduction::Mean) {

      Value zeroIVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(rewriter.getI32Type()));
      auto countInfo = torch_to_linalg::ReductionOpInfo{false, target, dimSet};
      Value numOfElems = torch_to_linalg::createReductionLinalgGeneric(
          rewriter, loc, countInfo,
          /*initElem=*/zeroIVal,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value targetVal = args[0];
            Value indTarget = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getIndexType(), targetVal);
            Value cmpEq = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ne, indTarget, ignoreIndexVal);
            cmpEq = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(),
                                                    cmpEq);
            Value add = rewriter.create<arith::AddIOp>(loc, args[1], cmpEq);
            rewriter.create<linalg::YieldOp>(loc, add);
          });

      numOfElems = rewriter.create<tensor::ExtractOp>(
          loc, rewriter.getI32Type(), numOfElems, ArrayRef<Value>{});
      numOfElems = convertScalarToDtype(rewriter, loc, numOfElems, elementType);

      auto opInfo = torch_to_linalg::ReductionOpInfo{false, finalRes, dimSet};
      finalRes = torch_to_linalg::createReductionLinalgGeneric(
          rewriter, loc, opInfo,
          /*initElem=*/zeroVal,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value newVal = args[0];
            Value accumulator = args[1];
            if (reduction == torch_upstream::Reduction::Mean)
              newVal = b.create<arith::DivFOp>(loc, newVal, numOfElems);
            Value result = b.create<arith::AddFOp>(loc, newVal, accumulator);
            b.create<linalg::YieldOp>(loc, result);
          });
    }

    // The implementation for the `total_weight` has been adopted from here:
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp#L154-L294
    // As per the ref link, the `total_weight` value when the `weight` is
    // `None`, is equal to `total_weight = batch_size - num_ignored_index`,
    // where `batch_size` is equal to `target.shape[0]` when rank(target) > 0,
    // otherwise 1. The value `num_ignored_index` is the number of elements of
    // the `target` tensors that have been ignored.

    if (reduction == torch_upstream::Reduction::None && inputRank == 2) {
      Value totalWeight = createZeroInitTensor(rewriter, loc, {}, elementType);
      rewriter.replaceOp(op, {finalRes, totalWeight});
      return success();
    }

    Value numIgnoredIndex;
    if (targetRank == 0) {
      Value targetVal = rewriter.create<tensor::ExtractOp>(loc, target);
      numIgnoredIndex = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, targetVal, ignoreIndex);
      numIgnoredIndex = convertScalarToDtype(rewriter, loc, numIgnoredIndex,
                                             ignoreIndex.getType());
    } else {
      Value zeroCstInt = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(ignoreIndex.getType()));

      auto opInfo =
          torch_to_linalg::ReductionOpInfo{/*keepDim=*/false, target, dimSet};
      numIgnoredIndex = torch_to_linalg::createReductionLinalgGeneric(
          rewriter, loc, opInfo,
          /*initElem=*/zeroCstInt,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value targetVal = args[0];
            Value accumulator = args[1];
            Value result = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, targetVal, ignoreIndex);
            result = b.create<arith::AddIOp>(
                loc,
                convertScalarToDtype(rewriter, loc, result,
                                     ignoreIndex.getType()),
                accumulator);
            b.create<linalg::YieldOp>(loc, result);
          });

      numIgnoredIndex =
          rewriter.create<tensor::ExtractOp>(loc, numIgnoredIndex);
    }

    Value numtargetElems = getTensorSize(rewriter, loc, target);
    Value totalWeightVal =
        rewriter.create<arith::SubIOp>(loc, numtargetElems, numIgnoredIndex);
    Value totalWeight = createInitTensor(
        rewriter, loc, {}, elementType,
        convertScalarToDtype(rewriter, loc, totalWeightVal, elementType));

    rewriter.replaceOp(op, {finalRes, totalWeight});
    return success();
  }
};
} // namespace

/// Inverted STD: rSTD = 1 / sqrt(var + eps).
static Value calculateRSTD(OpBuilder &b, Location loc, Type elemTy, Value eps,
                           Value var) {
  // The eps is always f64.
  Value truncatedEps = b.create<arith::TruncFOp>(loc, elemTy, eps);
  Value varPlusEps = b.create<arith::AddFOp>(loc, var, truncatedEps);
  Value rSTD = b.create<math::RsqrtOp>(loc, varPlusEps);
  return rSTD;
}

// Normalization formula:
//   ((input - mean) * rSTD * weight + bias
static Value createLinalgPayloadCalculationForNormOpsWithRSTD(
    OpBuilder &b, Location loc, Type elemTy, Value input, Value mean,
    Value rSTD, Value eps, Value weight, Value bias) {
  Value inputSubMean = b.create<arith::SubFOp>(loc, input, mean);
  Value temp = b.create<arith::MulFOp>(loc, inputSubMean, rSTD);
  Value timesWeight = b.create<arith::MulFOp>(loc, temp, weight);
  Value plusBias = b.create<arith::AddFOp>(loc, timesWeight, bias);
  return plusBias;
}

static Value createLinalgPayloadCalculationForNormOpsWithVar(
    OpBuilder &b, Location loc, Type elemTy, Value input, Value mean, Value var,
    Value eps, Value weight, Value bias) {
  Value rSTD = calculateRSTD(b, loc, elemTy, eps, var);
  Value result = createLinalgPayloadCalculationForNormOpsWithRSTD(
      b, loc, elemTy, input, mean, rSTD, eps, weight, bias);
  return result;
}

namespace {
class ConvertAtenBatchNormOp : public OpConversionPattern<AtenBatchNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
    Value runningMean = adaptor.getRunningMean();
    Value runningVar = adaptor.getRunningVar();
    Value training = adaptor.getTraining();
    Value eps = adaptor.getEps();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Handle the None cases for the optional parameters:
    // weight, bias.
    if (failed(checkNotNone(rewriter, op, weight)) ||
        failed(checkNotNone(rewriter, op, bias)) ||
        failed(checkNotNone(rewriter, op, runningMean)) ||
        failed(checkNotNone(rewriter, op, runningVar)))
      return failure();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto weightType = cast<RankedTensorType>(weight.getType());
    auto biasType = cast<RankedTensorType>(bias.getType());
    auto runningMeanType = cast<RankedTensorType>(runningMean.getType());
    auto runningVarType = cast<RankedTensorType>(runningVar.getType());

    auto inputRank = inputType.getRank();
    if (inputRank < 2)
      return rewriter.notifyMatchFailure(
          op, "input should have rank larger than 1");

    if (weightType.getRank() != 1 || biasType.getRank() != 1 ||
        runningMeanType.getRank() != 1 || runningVarType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expect weight, bias, running_mean and running_var to be rank 1");
    }

    // TODO: Add support for training.
    auto constFalse = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(IntegerType::get(context, 1), 0));
    auto trainingFalse = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, training, constFalse);
    rewriter.create<cf::AssertOp>(
        loc, trainingFalse,
        rewriter.getStringAttr("training is not supported for now"));

    // num_features – C from an expected input of size (N,C,D,H,W ...)
    Value numFeatures = rewriter.create<tensor::DimOp>(loc, input, 1);
    auto contractingDim0EqualsNumFeatures = [&](Value v) {
      auto dim0 = rewriter.create<tensor::DimOp>(loc, v, 0);
      auto dim0Equal = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, numFeatures, dim0);
      rewriter.create<cf::AssertOp>(
          loc, dim0Equal,
          rewriter.getStringAttr(
              "expect the size of dim 0 equal to the number of features"));
    };
    if (!isAssumingStrictSymbolicShapes(rewriter)) {
      contractingDim0EqualsNumFeatures(weight);
      contractingDim0EqualsNumFeatures(bias);
      contractingDim0EqualsNumFeatures(runningMean);
      contractingDim0EqualsNumFeatures(runningVar);
    }

    auto indexingMap = AffineMap::get(
        /*dimCount=*/inputRank,
        /*symbolCount=*/0, rewriter.getAffineDimExpr(1), context);
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputRank), // input
        indexingMap,                                // weight
        indexingMap,                                // bias
        indexingMap,                                // runningMean
        indexingMap,                                // runningVar
        rewriter.getMultiDimIdentityMap(inputRank), // output
    };
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
    Value batchNorm =
        rewriter
            .create<linalg::GenericOp>(
                loc, input.getType(),
                ValueRange{input, weight, bias, runningMean, runningVar}, input,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0], weight = args[1], bias = args[2],
                        mean = args[3], var = args[4];
                  Value result =
                      createLinalgPayloadCalculationForNormOpsWithVar(
                          b, loc, var.getType(), input, mean, var, eps, weight,
                          bias);
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, batchNorm);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenNllLossBackwardOp
    : public OpConversionPattern<AtenNllLossBackwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNllLossBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    Value gradOutput = adaptor.getGradOutput();
    Value input = adaptor.getSelf();
    Value target = adaptor.getTarget();
    Value weight = adaptor.getWeight();
    bool weightIsNone = isa<Torch::NoneType>(op.getWeight().getType());
    Value ignoreIndex = castIntToIndex(rewriter, loc, adaptor.getIgnoreIndex());
    Value totalWeight = adaptor.getTotalWeight();

    auto inputType = cast<RankedTensorType>(input.getType());
    int inputRank = inputType.getRank();
    auto gradOutputType = cast<RankedTensorType>(gradOutput.getType());
    Type resultElementType = gradOutputType.getElementType();

    int64_t reduction;
    if (!matchPattern(op.getReduction(), m_TorchConstantInt(&reduction)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    if (!hasElementType<mlir::FloatType>(gradOutput) ||
        !hasElementType<mlir::FloatType>(gradOutput) ||
        (!weightIsNone && !hasElementType<mlir::FloatType>(weight))) {
      return rewriter.notifyMatchFailure(
          op, "`gradOutput`, 'weight', and `totalWeight` must be tensors of "
              "type float");
    }

    if (!hasElementType<mlir::IntegerType>(target)) {
      return rewriter.notifyMatchFailure(
          op, "`target` must be a tensor of integer type");
    }

    auto outputSize = getTensorSizes(rewriter, loc, input);
    Value gradInputTensor =
        createZeroInitTensor(rewriter, loc, outputSize, resultElementType);

    auto getAffineMapForSingleElementTensor = [&](Value tensor) {
      auto tensorType = cast<RankedTensorType>(tensor.getType());
      SmallVector<AffineExpr> affineExprs(tensorType.getRank(),
                                          rewriter.getAffineConstantExpr(0));
      return AffineMap::get(inputRank, /*symbolCount=*/0, affineExprs,
                            op->getContext());
    };

    AffineMap gradOutMap = AffineMap::get(inputRank, /*symbolCount=*/0,
                                          rewriter.getAffineDimExpr(0));
    if (reduction != torch_upstream::Reduction::None || inputRank == 1)
      gradOutMap = getAffineMapForSingleElementTensor(gradOutput);
    AffineMap targetMap = AffineMap::get(inputRank, /*symbolCount=*/0,
                                         rewriter.getAffineDimExpr(0));
    if (inputRank == 1)
      targetMap = getAffineMapForSingleElementTensor(target);
    AffineMap totalWeightMap = getAffineMapForSingleElementTensor(totalWeight);
    AffineMap resultMap = rewriter.getMultiDimIdentityMap(inputRank);

    SmallVector<AffineMap> indexingMaps{gradOutMap, targetMap, totalWeightMap,
                                        resultMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);

    // The code generation is equivalent to the following pseudo-code:
    //
    // for batch_index in len(input.size(0)):
    //     for class_index in len(input.size(1)):
    //         target_elem = target[batch_index]
    //
    //         if reduction == None:
    //             grad_out_elem = grad_output[batchIndex]
    //         else:
    //             grad_out_elem = grad_output[0]
    //
    //         if reduction == Mean:
    //             total_weight_elem = total_weight[0]
    //             grad_out_elem /= total_weight_elem
    //
    //         weight_elem = weight[target_elem] if weight != None else 1
    //
    //         if target_elem != class_index or target_elem == ignore_index:
    //             grad_input_elem = -weight_elem * grad_out_elem
    //         else:
    //             grad_input_elem = 0
    //         grad_input[batch_index, target_elem] = grad_input_elem
    //
    // NOTE: In the case of not batch dimension, `batch_index` essentially
    // becomes zero.
    Value gradInput =
        rewriter
            .create<linalg::GenericOp>(
                loc, gradInputTensor.getType(),
                ValueRange{gradOutput, target, totalWeight}, gradInputTensor,
                indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value gradOutElem = args[0];
                  Value targetElem = castIntToIndex(b, loc, args[1]);
                  Value totalWeightElem = args[2];
                  Value classIndex =
                      b.create<linalg::IndexOp>(loc, inputRank - 1);

                  if (reduction == torch_upstream::Reduction::Mean) {
                    gradOutElem = b.create<arith::DivFOp>(loc, gradOutElem,
                                                          totalWeightElem);
                  }

                  Value negGradOutElem =
                      b.create<arith::NegFOp>(loc, gradOutElem);
                  Value weightElem = getConstant(b, loc, 1, resultElementType);
                  if (!weightIsNone) {
                    weightElem =
                        b.create<tensor::ExtractOp>(loc, weight, targetElem);
                  }
                  Value weightedNegGradOutElem =
                      b.create<arith::MulFOp>(loc, weightElem, negGradOutElem);

                  Value targetNeqClassIndex = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::ne, targetElem, classIndex);
                  Value targetEqIgnoreIndex = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, targetElem, ignoreIndex);
                  Value gradInputIsZero = b.create<arith::OrIOp>(
                      loc, targetNeqClassIndex, targetEqIgnoreIndex);

                  Value zero = getConstant(b, loc, 0, resultElementType);
                  Value gradInElem = b.create<arith::SelectOp>(
                      loc, gradInputIsZero, zero, weightedNegGradOutElem);
                  b.create<linalg::YieldOp>(loc, gradInElem);
                })
            ->getResult(0);

    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, gradInput);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenDetachOp : public OpConversionPattern<AtenDetachOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDetachOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getSelf());
    return success();
  }
};
} // namespace

namespace {
class ConvertPrimsSplitDimOp : public OpConversionPattern<PrimsSplitDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimsSplitDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto aRankedTensorType = cast<RankedTensorType>(adaptor.getA().getType());

    const TypeConverter *typeConverter = getTypeConverter();

    auto resultRankedTensorType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));

    // The dimension being split must be statically known.

    int64_t dimInt;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
      return failure();

    SmallVector<ReassociationIndices> associations;
    associations.reserve(aRankedTensorType.getRank());

    for (unsigned i = 0; i < dimInt; ++i) {
      associations.push_back(ReassociationIndices{i});
    }
    associations.push_back(ReassociationIndices{dimInt, dimInt + 1});
    for (int i = dimInt + 2; i < resultRankedTensorType.getRank(); ++i) {
      associations.push_back(ReassociationIndices{i});
    }

    auto expanded = rewriter.createOrFold<tensor::ExpandShapeOp>(
        op.getLoc(), resultRankedTensorType, adaptor.getA(), associations);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultRankedTensorType,
                                                expanded);
    return success();
  }
};
} // namespace

namespace {
class ConvertPrimsCollapseOp : public OpConversionPattern<PrimsCollapseOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimsCollapseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto aRankedTensorType = cast<RankedTensorType>(adaptor.getA().getType());
    const TypeConverter *typeConverter = getTypeConverter();

    auto resultRankedTensorType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));

    // Collapse range must be statically known.
    int64_t startInt;
    if (!matchPattern(op.getStart(), m_TorchConstantInt(&startInt)))
      return failure();

    int64_t endInt;
    if (!matchPattern(op.getEnd(), m_TorchConstantInt(&endInt)))
      return failure();

    // Upstream MLIR is overly strict -- it fails verification if the
    // collapse_shape is the identity op (i.e. when no dimensions are
    // collapsed). We manually fold this case here.
    if (startInt == endInt) {
      rewriter.replaceOp(op, adaptor.getA());
      return success();
    }

    SmallVector<ReassociationIndices> associations;
    associations.reserve(resultRankedTensorType.getRank());

    // An example of is where input shape is [3,4,5,6] and
    // start = 1, and end = 2. The collapsed shape is then [3,4*5,6],
    // with reassociation indices of [0], [1,2], and [3].

    // Append the singleton dimensions before the collapsed dimensions.
    for (unsigned i = 0; i < startInt; ++i) {
      associations.push_back(ReassociationIndices{i});
    }

    // Append the collapsed dimensions.
    ReassociationIndices collapseDims(endInt + 1 - startInt);
    std::iota(collapseDims.begin(), collapseDims.end(), startInt);
    associations.push_back(collapseDims);

    // Append the singleton dimensions after the collapsed dimensions.
    for (int i = endInt + 1; i < aRankedTensorType.getRank(); ++i) {
      associations.push_back(ReassociationIndices{i});
    }

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        op, resultRankedTensorType, adaptor.getA(), associations);

    return success();
  }
};
} // namespace

namespace {
class ConvertTensorStaticInfoCastOp
    : public OpConversionPattern<TensorStaticInfoCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorStaticInfoCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getOperand());
    return success();
  }
};
} // namespace

namespace {
class ConvertLogitOp : public OpConversionPattern<AtenLogitOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenLogitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    Value input = adaptor.getSelf();
    Value eps = adaptor.getEps();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    bool handleEps = false;
    if (succeeded(checkNotNone(rewriter, op, eps)))
      handleEps = true;

    if (handleEps && !isa<mlir::FloatType>(eps.getType())) {
      op.emitError("Logit does not support non-floating point type");
      return failure();
    }

    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();

    if (!isa<mlir::FloatType>(inputElementType)) {
      op.emitError("Logit does not support non-floating point type");
      return failure();
    }

    auto inputRank = inputType.getRank();

    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputRank), // input
        rewriter.getMultiDimIdentityMap(inputRank), // output
    };
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
    Value logit =
        rewriter
            .create<linalg::GenericOp>(
                loc, input.getType(),
                /*ins=*/input,
                /*outs=*/input,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0];

                  TypedAttr oneAttr = b.getFloatAttr(inputElementType, 1.0);
                  Value oneValue = b.create<arith::ConstantOp>(loc, oneAttr);

                  Value zI;
                  if (!handleEps) {
                    zI = input;
                  } else {
                    Value truncEps =
                        b.create<arith::TruncFOp>(loc, inputElementType, eps);
                    Value oneMinusEps =
                        b.create<arith::SubFOp>(loc, oneValue, truncEps);

                    Value min =
                        b.create<arith::MinimumFOp>(loc, input, oneMinusEps);
                    Value clampedInput =
                        b.create<arith::MaximumFOp>(loc, min, truncEps);

                    zI = clampedInput;
                  }

                  Value probability =
                      b.create<arith::SubFOp>(loc, oneValue, zI);
                  Value odds = b.create<arith::DivFOp>(loc, zI, probability);
                  Value result = b.create<math::LogOp>(loc, odds);

                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, logit);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenIntReprOp : public OpConversionPattern<AtenIntReprOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenIntReprOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getSelf());
    return success();
  }
};
} // namespace

namespace {
class ConvertDequantizePerChannel
    : public OpConversionPattern<AtenDequantizeSelfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDequantizeSelfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto qoperand = op.getOperand();
    auto make = qoperand.getDefiningOp<Aten_MakePerChannelQuantizedTensorOp>();
    if (!make) {
      return rewriter.notifyMatchFailure(op, "did not find per channel qint");
    }

    auto converter = getTypeConverter();
    auto operand = make.getOperand(0);
    auto scale = make.getScale();
    auto zeropoint = make.getZeroPoint();
    auto axis = make.getAxis();

    IntegerAttr axisAttr;
    if (!matchPattern(axis, m_Constant(&axisAttr))) {
      return failure();
    }

    auto operandDTy = cast<ValueTensorType>(operand.getType()).getDtype();
    auto zeropointDTy = cast<ValueTensorType>(zeropoint.getType()).getDtype();
    operand = converter->materializeTargetConversion(
        rewriter, loc, converter->convertType(operand.getType()), operand);
    scale = converter->materializeTargetConversion(
        rewriter, loc, converter->convertType(scale.getType()), scale);
    zeropoint = converter->materializeTargetConversion(
        rewriter, loc, converter->convertType(zeropoint.getType()), zeropoint);

    auto resultType = cast<RankedTensorType>(
        converter->convertType(op->getResult(0).getType()));

    llvm::SmallVector<Value> dynSizes;
    for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
      if (ShapedType::isDynamic(dim)) {
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, operand, index));
      }
    }

    llvm::SmallVector<utils::IteratorType> iterators(
        resultType.getRank(), utils::IteratorType::parallel);
    llvm::SmallVector<AffineMap> maps(
        4, {rewriter.getMultiDimIdentityMap(resultType.getRank())});
    auto broadcastMap = AffineMap::get(
        resultType.getRank(), /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(axisAttr.getInt())}, rewriter.getContext());
    maps[1] = broadcastMap;
    maps[2] = broadcastMap;

    auto empty =
        rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType, dynSizes);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{operand, scale, zeropoint},
        ValueRange{empty}, maps, iterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value operand = args[0];
          Value scale = args[1];
          Value zeropoint = args[2];
          if (operandDTy.isUnsignedInteger(8)) {
            operand = b.create<arith::ExtUIOp>(loc, b.getI32Type(), operand);
          } else if (operandDTy.isSignedInteger(8)) {
            operand = b.create<arith::ExtSIOp>(loc, b.getI32Type(), operand);
          }

          if (zeropointDTy.isUnsignedInteger(8)) {
            zeropoint =
                b.create<arith::ExtUIOp>(loc, b.getI32Type(), zeropoint);
          } else if (zeropointDTy.isSignedInteger(8)) {
            zeropoint =
                b.create<arith::ExtSIOp>(loc, b.getI32Type(), zeropoint);
          } else if (zeropointDTy.isInteger(64)) {
            zeropoint =
                b.create<arith::TruncIOp>(loc, b.getI32Type(), zeropoint);
            op->emitWarning() << "truncated zero point from 64 to 32 bit";
          }

          Value sub = rewriter.create<arith::SubIOp>(loc, operand, zeropoint);
          Value fp =
              rewriter.create<arith::SIToFPOp>(loc, args[3].getType(), sub);
          Value mul = rewriter.create<arith::MulFOp>(loc, fp, scale);
          b.create<linalg::YieldOp>(loc, mul);
        });
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};
} // namespace

namespace {

template <typename OpTy>
class ConvertCastEquivalentOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    RankedTensorType resultType = cast<RankedTensorType>(
        converter->convertType(op->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getSelf());
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenGridSamplerOp : public OpConversionPattern<AtenGridSamplerOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenGridSamplerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type int64type = rewriter.getI64Type();
    Type floatType = rewriter.getF32Type();
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value zeroFloat = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(floatType, 0.0));
    Value oneFloat = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(floatType, 1.0));
    Value twoFloat = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(floatType, 2.0));
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    Value innerDim0a = rewriter.create<tensor::DimOp>(loc, input, 2);
    Value innerDim1a = rewriter.create<tensor::DimOp>(loc, input, 3);
    Value innerDim0b =
        rewriter.create<arith::SubIOp>(loc, innerDim0a, oneIndex);
    Value innerDim1b =
        rewriter.create<arith::SubIOp>(loc, innerDim1a, oneIndex);
    Value innerDim0c =
        rewriter.create<arith::IndexCastOp>(loc, int64type, innerDim0b);
    Value innerDim1c =
        rewriter.create<arith::IndexCastOp>(loc, int64type, innerDim1b);
    Value innerDim0d =
        rewriter.create<arith::SIToFPOp>(loc, floatType, innerDim0c);
    Value innerDim1d =
        rewriter.create<arith::SIToFPOp>(loc, floatType, innerDim1c);
    Value innerDim0e =
        rewriter.create<arith::DivFOp>(loc, innerDim0d, twoFloat);
    Value innerDim1e =
        rewriter.create<arith::DivFOp>(loc, innerDim1d, twoFloat);
    Value grid = adaptor.getGrid();
    auto gridType = cast<RankedTensorType>(grid.getType());
    auto gridRank = gridType.getRank();
    SmallVector<AffineMap> gridMaps{
        AffineMap::get(
            4, 0,
            {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(2),
             rewriter.getAffineDimExpr(3), rewriter.getAffineConstantExpr(0)},
            op->getContext()),
        AffineMap::get(
            4, 0,
            {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(2),
             rewriter.getAffineDimExpr(3), rewriter.getAffineConstantExpr(1)},
            op->getContext()),
        rewriter.getMultiDimIdentityMap(inputType.getRank())};
    SmallVector<utils::IteratorType> gridIterators(
        gridRank, utils::IteratorType::parallel);
    auto lambdaExtract = [](OpBuilder &b, Location loc, Value input, Value idxA,
                            Value idxB, Value idxC, Value idxD) -> Value {
      SmallVector<Value> index{idxA, idxB, idxC, idxD};
      Value result = b.create<tensor::ExtractOp>(loc, input, index);
      return result;
    };

    auto lambdaLinear = [&](OpBuilder &b, Location loc, Value x, Value y,
                            Value d) -> Value {
      Value dm = b.create<arith::SubFOp>(loc, oneFloat, d);
      Value ra = b.create<arith::MulFOp>(loc, x, dm);
      Value rb = b.create<arith::MulFOp>(loc, y, d);
      Value res = b.create<arith::AddFOp>(loc, ra, rb);
      return res;
    };

    auto lambdaNearest = [&](OpBuilder &b, Location loc, Value x, Value y,
                             Value d) -> Value {
      Value halfConst = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(floatType, 0.5));
      Value checkClosest =
          b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, d, halfConst);
      Value res = b.create<arith::SelectOp>(loc, checkClosest, x, y);
      return res;
    };

    auto lambdaInterpolate = [&](OpBuilder &b, Location loc, Value iMode,
                                 Value x, Value y, Value d) -> Value {
      Value linear = lambdaLinear(b, loc, x, y, d);
      Value nearest = lambdaNearest(b, loc, x, y, d);
      Value zeroInt =
          b.create<arith::ConstantOp>(loc, b.getIntegerAttr(int64type, 0));
      Value checkMode = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                iMode, zeroInt);
      Value res = b.create<arith::SelectOp>(loc, checkMode, linear, nearest);
      return res;
    };

    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    Value alignCorners = adaptor.getAlignCorners();
    Value interMode = adaptor.getInterpolationMode();
    SmallVector<Value> dynamicSizes{};
    if (resultType.isDynamicDim(0))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
    if (resultType.isDynamicDim(1))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, input, 1));
    if (resultType.isDynamicDim(2))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, grid, 1));
    if (resultType.isDynamicDim(3))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, grid, 2));
    tensor::EmptyOp emptyOp =
        rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicSizes);
    auto sGrid = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{grid, grid}, ValueRange(emptyOp),
        gridMaps, gridIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value gr0 = args[1];
          Value gr1 = args[0];
          Value gr0Half = b.create<arith::DivFOp>(loc, gr0, twoFloat);
          Value gr1Half = b.create<arith::DivFOp>(loc, gr1, twoFloat);
          Value gr0HalfSelect =
              b.create<arith::SelectOp>(loc, alignCorners, zeroFloat, gr0Half);
          Value gr1HalfSelect =
              b.create<arith::SelectOp>(loc, alignCorners, zeroFloat, gr1Half);
          Value gplus0 = b.create<arith::AddFOp>(loc, gr0, oneFloat);
          Value gplus1 = b.create<arith::AddFOp>(loc, gr1, oneFloat);
          Value gPlusMul0 = b.create<arith::MulFOp>(loc, gplus0, innerDim0e);
          Value gPlusMul1 = b.create<arith::MulFOp>(loc, gplus1, innerDim1e);
          Value result0 =
              b.create<arith::AddFOp>(loc, gPlusMul0, gr0HalfSelect);
          Value result1 =
              b.create<arith::AddFOp>(loc, gPlusMul1, gr1HalfSelect);
          Value checkLowerBound0 = b.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OLT, result0, zeroFloat);
          Value checkLowerBound1 = b.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OLT, result1, zeroFloat);
          Value lowerOrig0 = b.create<arith::FPToSIOp>(loc, int64type, result0);
          Value lowerOrig1 = b.create<arith::FPToSIOp>(loc, int64type, result1);
          Value zeroInt =
              b.create<arith::ConstantOp>(loc, b.getIntegerAttr(int64type, 0));
          Value oneInt =
              b.create<arith::ConstantOp>(loc, b.getIntegerAttr(int64type, 1));
          Value lowerSub0 = b.create<arith::SubIOp>(loc, lowerOrig0, oneInt);
          Value lowerSub1 = b.create<arith::SubIOp>(loc, lowerOrig1, oneInt);
          Value lower0 = b.create<arith::SelectOp>(loc, checkLowerBound0,
                                                   lowerSub0, lowerOrig0);
          Value lower1 = b.create<arith::SelectOp>(loc, checkLowerBound1,
                                                   lowerSub1, lowerOrig1);
          Value lowerValid0 =
              b.create<arith::SelectOp>(loc, checkLowerBound0, zeroInt, lower0);
          Value lowerValid1 =
              b.create<arith::SelectOp>(loc, checkLowerBound1, zeroInt, lower1);
          Value upper0 =
              b.create<arith::AddIOp>(loc, int64type, lower0, oneInt);
          Value upper1 =
              b.create<arith::AddIOp>(loc, int64type, lower1, oneInt);
          Value notValidUpper0 = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, upper0, innerDim0c);
          Value notValidUpper1 = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, upper1, innerDim1c);
          Value upperValid0 =
              b.create<arith::SelectOp>(loc, notValidUpper0, lower0, upper0);
          Value upperValid1 =
              b.create<arith::SelectOp>(loc, notValidUpper1, lower1, upper1);
          Value lw0 =
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), lowerValid0);
          Value lw1 =
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), lowerValid1);
          Value up0 =
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), upperValid0);
          Value up1 =
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), upperValid1);
          Value N = b.create<linalg::IndexOp>(loc, 0);
          Value C = b.create<linalg::IndexOp>(loc, 1);
          Value result00 = lambdaExtract(b, loc, input, N, C, lw0, lw1);
          Value result00a = b.create<arith::SelectOp>(loc, checkLowerBound0,
                                                      zeroFloat, result00);
          Value result00b = b.create<arith::SelectOp>(loc, checkLowerBound1,
                                                      zeroFloat, result00a);
          Value result01 = lambdaExtract(b, loc, input, N, C, lw0, up1);
          Value result01a = b.create<arith::SelectOp>(loc, notValidUpper1,
                                                      zeroFloat, result01);
          Value result01b = b.create<arith::SelectOp>(loc, checkLowerBound0,
                                                      zeroFloat, result01a);
          Value result10 = lambdaExtract(b, loc, input, N, C, up0, lw1);
          Value result10a = b.create<arith::SelectOp>(loc, notValidUpper0,
                                                      zeroFloat, result10);
          Value result10b = b.create<arith::SelectOp>(loc, checkLowerBound1,
                                                      zeroFloat, result10a);
          Value result11 = lambdaExtract(b, loc, input, N, C, up0, up1);
          Value result11a = b.create<arith::SelectOp>(loc, notValidUpper0,
                                                      zeroFloat, result11);
          Value result11b = b.create<arith::SelectOp>(loc, notValidUpper1,
                                                      zeroFloat, result11a);
          Value lw0a = b.create<arith::SIToFPOp>(loc, floatType, lower0);
          Value lw1a = b.create<arith::SIToFPOp>(loc, floatType, lower1);
          Value d1 = b.create<arith::SubFOp>(loc, result0, lw0a);
          Value d0 = b.create<arith::SubFOp>(loc, result1, lw1a);
          Value resultScaled0 =
              lambdaInterpolate(b, loc, interMode, result00b, result01b, d0);
          Value resultScaled1 =
              lambdaInterpolate(b, loc, interMode, result10b, result11b, d0);
          Value resultScaled = lambdaInterpolate(
              b, loc, interMode, resultScaled0, resultScaled1, d1);
          b.create<linalg::YieldOp>(loc, resultScaled);
        });
    rewriter.replaceOp(op, sGrid.getResults());
    return success();
  }
};
} // namespace

static Value NearestInterpolate(OpBuilder &b, Location loc,
                                SmallVector<Value> outputSizes, Value input,
                                SmallVector<Value> inputSizes,
                                SmallVector<Value> scaleValues,
                                std::string coordStr, std::string nearestMode) {

  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();

  SmallVector<Value> indices;
  for (unsigned i = 0; i < inputRank; i++) {
    indices.push_back(b.create<linalg::IndexOp>(loc, i));
  }

  for (unsigned i = 2; i < inputRank; i++) {
    Value outIndex = indices[i];

    Value inputSizeFP =
        b.create<arith::SIToFPOp>(loc, b.getF32Type(), inputSizes[i - 2]);

    Value outputSizeFP =
        b.create<arith::SIToFPOp>(loc, b.getF32Type(), outputSizes[i - 2]);

    // scale = length_resized / length_original
    // x_original = x_resized / scale
    Value scale;
    if (scaleValues.empty())
      scale = b.create<arith::DivFOp>(loc, outputSizeFP, inputSizeFP);
    else
      scale = scaleValues[i - 2];

    Value outInt = b.create<arith::IndexCastOp>(loc, b.getI64Type(), outIndex);
    Value outFP = b.create<arith::SIToFPOp>(loc, b.getF32Type(), outInt);
    Value proj;
    if (coordStr.empty() || coordStr == "_asymmetric") {
      proj = b.create<arith::DivFOp>(loc, outFP, scale);
    } else if (coordStr == "_half_pixel") {
      Value cstHalf = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.5));
      Value add = b.create<arith::AddFOp>(loc, outFP, cstHalf);
      Value div = b.create<arith::DivFOp>(loc, add, scale);
      proj = b.create<arith::SubFOp>(loc, div, cstHalf);
    } else {
      llvm_unreachable("Unsupported coordination transformation mode");
    }

    Value nearestFP;
    // get nearest pixel using floor
    if (nearestMode == "floor" || nearestMode == "") {
      nearestFP = b.create<math::FloorOp>(loc, proj);
    } else if (nearestMode == "round_prefer_floor") {
      Value cstHalf = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.5));
      Value floor = b.create<math::FloorOp>(loc, proj);
      Value ceil = b.create<math::CeilOp>(loc, proj);
      Value decimal = b.create<arith::SubFOp>(loc, proj, floor);
      Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE,
                                          decimal, cstHalf);
      nearestFP = b.create<arith::SelectOp>(loc, cmp, floor, ceil);
    } else if (nearestMode == "round_prefer_ceil") {
      Value cstHalf = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.5));
      Value cstOne = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1));
      Value floor = b.create<math::FloorOp>(loc, proj);
      Value ceil = b.create<math::CeilOp>(loc, proj);
      Value decimal = b.create<arith::SubFOp>(loc, proj, floor);
      Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGE,
                                          decimal, cstHalf);
      nearestFP = b.create<arith::SelectOp>(loc, cmp, ceil, floor);
      Value inputSizeMOne = b.create<arith::SubFOp>(loc, inputSizeFP, cstOne);
      // don't extract out of bounds
      nearestFP = b.create<arith::MinimumFOp>(loc, nearestFP, inputSizeMOne);
    } else if (nearestMode == "ceil") {
      Value cstOne = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1));
      Value inputSizeMOne = b.create<arith::SubFOp>(loc, inputSizeFP, cstOne);
      nearestFP = b.create<math::CeilOp>(loc, proj);
      nearestFP = b.create<arith::MinimumFOp>(loc, nearestFP, inputSizeMOne);
    } else {
      llvm_unreachable("Unsupported nearest mode");
    }
    Value nearestInt =
        b.create<arith::FPToSIOp>(loc, b.getI64Type(), nearestFP);
    Value nearest =
        b.create<arith::IndexCastOp>(loc, b.getIndexType(), nearestInt);

    indices[i] = nearest;
  }
  Value retVal = b.create<tensor::ExtractOp>(loc, input, indices);
  return retVal;
}

static Value BilinearInterpolate(OpBuilder &b,
                                 Aten__InterpolateSizeListScaleListOp op,
                                 Location loc, SmallVector<Value> outputSizes,
                                 Value input, SmallVector<Value> inputSizes,
                                 SmallVector<Value> scaleValues,
                                 std::string coordStr) {
  unsigned dimOffset = 2;
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();

  Value cstOneFloat = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1.0));
  Value cstHalf = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.5));
  Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));

  bool alignCornersBool;
  matchPattern(op.getAlignCorners(), m_TorchConstantBool(&alignCornersBool));

  SmallVector<Value> indices;
  for (unsigned i = 0; i < inputRank; i++) {
    indices.push_back(b.create<linalg::IndexOp>(loc, i));
  }

  SmallVector<Value> proj, projEps, high, low, highFP, lowFP;
  for (unsigned i = 0; i < inputRank - dimOffset; i++) {
    // length_original
    Value inputFP =
        b.create<arith::SIToFPOp>(loc, b.getF32Type(), inputSizes[i]);
    // length_resized
    Value outputSizeFP =
        b.create<arith::SIToFPOp>(loc, b.getF32Type(), outputSizes[i]);
    // scale = length_resized/length_original
    Value scale;
    if (alignCornersBool) {
      // x_original = x_resized * (length_original - 1) / (length_resized - 1)
      Value inputSubOne = b.create<arith::SubFOp>(loc, inputFP, cstOneFloat);
      Value outputSizeSubOne =
          b.create<arith::SubFOp>(loc, outputSizeFP, cstOneFloat);
      Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ,
                                          outputSizeSubOne, zero);
      scale = b.create<arith::DivFOp>(loc, inputSubOne, outputSizeSubOne);
      scale = b.create<arith::SelectOp>(loc, cmp, zero, scale);
      coordStr = "_align_corners";
    } else if (scaleValues.empty())
      scale = b.create<arith::DivFOp>(loc, outputSizeFP, inputFP);
    else
      scale = scaleValues[i];
    // y_resized
    Value outInt = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
                                                indices[i + dimOffset]);
    Value outFP = b.create<arith::SIToFPOp>(loc, b.getF32Type(), outInt);
    Value preClip;
    if (coordStr == "_align_corners") {
      preClip = b.create<arith::MulFOp>(loc, outFP, scale);
    }
    if (coordStr == "_asymmetric") {
      preClip = b.create<arith::DivFOp>(loc, outFP, scale);
    }
    if (coordStr == "_pytorch_half_pixel" || coordStr == "" ||
        coordStr == "_half_pixel_symmetric") {
      // half-pixel modes
      // y_resized + 0.5
      Value outPlusHalf = b.create<arith::AddFOp>(loc, outFP, cstHalf);
      // (y_resized + 0.5) / scale
      Value outDivScale = b.create<arith::DivFOp>(loc, outPlusHalf, scale);
      // _ - 0.5
      preClip = b.create<arith::SubFOp>(loc, outDivScale, cstHalf);
    }
    // for half_pixel_symmetric, need to compute offset from raw scales
    if (coordStr == "_half_pixel_symmetric" && !scaleValues.empty()) {
      Value outputSizeFromScale = b.create<arith::MulFOp>(loc, inputFP, scale);
      Value adjustment =
          b.create<arith::DivFOp>(loc, outputSizeFP, outputSizeFromScale);
      Value cstTwo = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(2.0));
      Value center = b.create<arith::DivFOp>(loc, inputFP, cstTwo);
      Value oneMAdjustment =
          b.create<arith::SubFOp>(loc, cstOneFloat, adjustment);
      Value offset = b.create<arith::MulFOp>(loc, center, oneMAdjustment);
      preClip = b.create<arith::AddFOp>(loc, offset, preClip);
    }
    // for pytorch half pixel , special case for length_resized == 1:
    if (coordStr == "_pytorch_half_pixel") {
      Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ,
                                          outputSizeFP, cstOneFloat);
      preClip = b.create<arith::SelectOp>(loc, cmp, zero, preClip);
    }
    // preClip is the fp position inside the input image to extract from.
    // clip to [0,inf)
    Value max = b.create<arith::MaximumFOp>(loc, preClip, zero);
    Value inputSubOne = b.create<arith::SubFOp>(loc, inputFP, cstOneFloat);
    // clip to [0,length_original - 1].
    // proj is properly within the input image.
    proj.push_back(b.create<arith::MinimumFOp>(loc, max, inputSubOne));

    // for bilinear interpolation, we look for the nearest indices below and
    // above proj
    lowFP.push_back(b.create<math::FloorOp>(loc, proj[i]));
    Value projPlusOne = b.create<arith::AddFOp>(loc, cstOneFloat, proj[i]);
    highFP.push_back(b.create<math::FloorOp>(loc, projPlusOne));

    Value lowInt = b.create<arith::FPToSIOp>(loc, b.getI64Type(), lowFP[i]);
    low.push_back(b.create<arith::IndexCastOp>(loc, b.getIndexType(), lowInt));

    // highFP could be out-of-bounds, so make sure to clip it down before
    // extracting. If highFP actually gets clipped here, then high[i] will
    // extract at the last pixel, but will treat it as if it were extracted from
    // one further position when computing the interpolation weights.
    Value highExtract =
        b.create<arith::MinimumFOp>(loc, projPlusOne, inputSubOne);
    highExtract = b.create<arith::FPToSIOp>(loc, b.getI64Type(), highExtract);
    high.push_back(
        b.create<arith::IndexCastOp>(loc, b.getIndexType(), highExtract));
  }

  indices[dimOffset] = low[0];
  indices[dimOffset + 1] = low[1];
  Value p00 = b.create<tensor::ExtractOp>(loc, input, indices);

  indices[dimOffset] = low[0];
  indices[dimOffset + 1] = high[1];
  Value p01 = b.create<tensor::ExtractOp>(loc, input, indices);

  indices[dimOffset] = high[0];
  indices[dimOffset + 1] = low[1];
  Value p10 = b.create<tensor::ExtractOp>(loc, input, indices);

  indices[dimOffset] = high[0];
  indices[dimOffset + 1] = high[1];
  Value p11 = b.create<tensor::ExtractOp>(loc, input, indices);

  // Let Aij := area rect((yProj,xProj) <-> (y_i*,x_j*)),
  // where i* = i+1 mod 2 and x_0 = xLow, x_1 = xHigh etc.
  // We interpolate via the weighted average of pij by weights Aij
  // the formula is retval = Sum(pij*Aij for i and j in range(2))
  // Note: we do not need to divide by total rect area == 1

  // lengths : Aij == dyi*dxj
  Value dy0 = b.create<arith::SubFOp>(loc, highFP[0], proj[0]);
  Value dy1 = b.create<arith::SubFOp>(loc, proj[0], lowFP[0]);
  Value dx0 = b.create<arith::SubFOp>(loc, highFP[1], proj[1]);
  Value dx1 = b.create<arith::SubFOp>(loc, proj[1], lowFP[1]);

  // left = A00*p00 + A01*p01 = dy0(dx0p00 + dx1p01)
  Value dx0p00 = b.create<arith::MulFOp>(loc, dx0, p00);
  Value dx1p01 = b.create<arith::MulFOp>(loc, dx1, p01);
  Value sum = b.create<arith::AddFOp>(loc, dx0p00, dx1p01);
  Value left = b.create<arith::MulFOp>(loc, dy0, sum);
  // right = A10*p10 + A11*p11 = dy1(dx0p10 + dx1p11)
  Value dx0p10 = b.create<arith::MulFOp>(loc, dx0, p10);
  Value dx1p11 = b.create<arith::MulFOp>(loc, dx1, p11);
  sum = b.create<arith::AddFOp>(loc, dx0p10, dx1p11);
  Value right = b.create<arith::MulFOp>(loc, dy1, sum);

  return b.create<arith::AddFOp>(loc, left, right);
}

namespace {
class ConvertInterpolateOp
    : public OpConversionPattern<Aten__InterpolateSizeListScaleListOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Aten__InterpolateSizeListScaleListOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    std::string mode;
    // note: to support onnx.Resize, we are passing some extra options through
    // the mode attribute. For example, onnx.Resize with mode="linear" and
    // coordinate_transformation_mode="asymmetric" will lower to an interpolate
    // op with the non-standard mode="bilinear_asymmetric".
    matchPattern(op.getMode(), m_TorchConstantStr(mode));
    if (mode.substr(0, 8) != "bilinear" && mode.substr(0, 7) != "nearest") {
      return failure();
    }

    Location loc = op->getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();
    if (mode.substr(0, 8) == "bilinear" && inputRank != 4)
      return rewriter.notifyMatchFailure(
          op,
          "cannot perform bilinear interpolation when input spatial dims != 2");

    SmallVector<Value> outputSizeIntValues;
    SmallVector<Value> inputSizes;
    SmallVector<Value> ScaleFactorFloatValues;
    for (unsigned i = 2; i < inputRank; i++) {
      Value inputSize = getDimOp(rewriter, loc, input, i);
      inputSizes.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIntegerType(64), inputSize));
    }

    if (!isa<Torch::NoneType>(op.getScaleFactor().getType())) {
      bool recompScale;
      if (!matchPattern(op.getRecomputeScaleFactor(),
                        m_TorchConstantBool(&recompScale)))
        recompScale = false;
      SmallVector<Value> ScaleFactorTorchFloat;
      if (!getListConstructElements(op.getScaleFactor(), ScaleFactorTorchFloat))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the output_size is not constructed from "
                "ListConstruct");
      ScaleFactorFloatValues = getTypeConvertedValues(
          rewriter, loc, getTypeConverter(), ScaleFactorTorchFloat);
      for (unsigned i = 0; i < inputRank - 2; i++) {
        Value inputSizeFP = rewriter.create<arith::SIToFPOp>(
            loc, rewriter.getF32Type(), inputSizes[i]);
        ScaleFactorFloatValues[i] = rewriter.create<arith::TruncFOp>(
            loc, inputSizeFP.getType(), ScaleFactorFloatValues[i]);
        Value outputSize = rewriter.create<arith::MulFOp>(
            loc, inputSizeFP, ScaleFactorFloatValues[i]);
        outputSize = rewriter.create<math::FloorOp>(loc, outputSize);
        outputSize = rewriter.create<arith::FPToSIOp>(
            loc, rewriter.getI64Type(), outputSize);
        outputSizeIntValues.push_back(outputSize);
      }
      if (recompScale)
        ScaleFactorFloatValues.clear();
    } else {
      SmallVector<Value> outputSizeTorchInt;
      if (!getListConstructElements(op.getSize(), outputSizeTorchInt))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the output_size is not constructed from "
                "ListConstruct");
      outputSizeIntValues = getTypeConvertedValues(
          rewriter, loc, getTypeConverter(), outputSizeTorchInt);
    }
    SmallVector<Value> dims = getTensorSizesUntilDim(rewriter, loc, input, 1);
    for (unsigned i = 2; i < inputRank; i++) {
      dims.push_back(castIntToIndex(rewriter, loc, outputSizeIntValues[i - 2]));
    }

    Value outTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(dims), inputType.getElementType());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(inputRank);
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, outTensor.getType(), ValueRange{}, outTensor,
                /*indexingMaps=*/idMap,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value retVal;
                  if (mode.substr(0, 7) == "nearest") {
                    std::string coordTfMode =
                        mode.substr(7, mode.find(",") - 7);
                    std::string nearestMode =
                        (mode.find(",") == std::string::npos)
                            ? ""
                            : mode.substr(mode.find(",") + 1);
                    retVal = NearestInterpolate(
                        b, loc, outputSizeIntValues, input, inputSizes,
                        ScaleFactorFloatValues, coordTfMode, nearestMode);
                  } else if (mode.substr(0, 8) == "bilinear") {
                    retVal = BilinearInterpolate(
                        b, op, loc, outputSizeIntValues, input, inputSizes,
                        ScaleFactorFloatValues, mode.substr(8));
                  }
                  b.create<linalg::YieldOp>(loc, retVal);
                })
            .getResult(0);
    Type newResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, finalRes);
    return success();
  }
};
} // namespace

namespace {
// This pattern row reduces a matrix, then returns the product of it's diagonal
// elements
class ConvertAtenLinalgDetOp : public OpConversionPattern<AtenLinalgDetOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenLinalgDetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.getA();
    auto inputType = cast<RankedTensorType>(input.getType());
    unsigned inputRank = inputType.getRank();
    auto elemTy = inputType.getElementType();
    bool isBatched = (inputRank == 3);
    Value cstZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cstOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cstZeroF = getConstant(rewriter, loc, 0, elemTy);
    // get some shapes
    SmallVector<int64_t> inputShape(inputType.getShape());

    SmallVector<int64_t> sliceShape(inputShape);
    sliceShape[sliceShape.size() - 2] = 1;

    SmallVector<int64_t> diagShape(inputType.getShape());
    diagShape[diagShape.size() - 2] = 1;
    diagShape[diagShape.size() - 1] = 1;

    ArrayRef<int64_t> diagCollapseShape(diagShape);
    diagCollapseShape = diagCollapseShape.drop_back();

    auto sliceTy = RankedTensorType::get(sliceShape, elemTy);
    auto diagTy = RankedTensorType::get(diagShape, elemTy);
    auto diagCollapseTy = RankedTensorType::get(diagCollapseShape, elemTy);

    SmallVector<ReassociationIndices> diagReassociations;
    diagReassociations.reserve(diagCollapseShape.size());
    int64_t diagRank = diagCollapseShape.size();
    for (int i = 0, s = diagRank - 1; i < s; ++i)
      diagReassociations.push_back(ReassociationIndices{i});
    diagReassociations.push_back(ReassociationIndices{diagRank - 1, diagRank});

    // get some sizes
    SmallVector<Value> inputSizes = getTensorSizes(rewriter, loc, input);
    Value chDim = isBatched ? inputSizes[0] : cstOne;
    Value matDim = inputSizes[inputRank - 1];
    Value matDimMinusOne = rewriter.create<arith::SubIOp>(loc, matDim, cstOne);
    ArrayRef<Value> sliceSizes(inputSizes.begin(), inputSizes.end() - 1);
    // initialize a tensor to store the diagonal elements found during row
    // reduction
    Value initDiags = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(sliceSizes), elemTy);
    // loop over each pivot row in A. Get the diagonal, then reduce the
    // subdiagonal Don't perform the loop on the last row since no further
    // reduction is needed.
    auto rowReductionLoop = rewriter.create<scf::ForOp>(
        loc, /*start=*/cstZero, /*end=*/matDimMinusOne, /*step=*/cstOne,
        /*yeild_to=*/ValueRange{input, initDiags}, /*body_lambda=*/
        [&](OpBuilder &b, Location loc, Value row, ValueRange vals) {
          // extract row i from input Tensor of shape CxNxN or shape
          // NxN.
          OpFoldResult cstOneFold = getAsOpFoldResult(cstOne);
          OpFoldResult cstZeroFold = getAsOpFoldResult(cstZero);
          SmallVector<OpFoldResult> offsets(inputRank, cstZeroFold);
          offsets[inputRank - 2] = row;
          SmallVector<OpFoldResult> strides(inputRank, cstOneFold);
          auto sizes = getAsOpFoldResult(inputSizes);
          sizes[inputRank - 2] = cstOneFold;
          // offsets = [0, row, 0], sizes = [C, 1, N] -> pivot row
          Value pivot = b.create<tensor::ExtractSliceOp>(
              loc, sliceTy, vals[0], offsets, sizes, strides);
          // extract diagonal elements and insert them into vals[1]
          offsets.back() = row;
          sizes.back() = cstOneFold;
          // offsets = [0, row, row], sizes = [C, 1, 1] -> diag(row,row)
          Value diag = b.create<tensor::ExtractSliceOp>(
              loc, diagTy, vals[0], offsets, sizes, strides);

          Value diagCollapse = b.create<tensor::CollapseShapeOp>(
              loc, diagCollapseTy, diag, diagReassociations);

          SmallVector<OpFoldResult> diagOffsets(inputRank - 1, cstZeroFold);
          diagOffsets.back() = row;
          SmallVector<OpFoldResult> diagStrides(inputRank - 1, cstOneFold);
          SmallVector<OpFoldResult> diagSizes = getAsOpFoldResult(sliceSizes);
          diagSizes.back() = cstOneFold;
          // offsets = [0, row], sizes = [C, 1] insert to [C,N]
          Value updatedDiags = b.create<tensor::InsertSliceOp>(
              loc, diagCollapse, vals[1], diagOffsets, diagSizes, diagStrides);
          // the subpivot matrix column size, as a Value, is matDim - row -
          // cstOne. This can't be statically converted to an int64_t, since row
          // is the loop index, so this is left as a dynamic dim.
          SmallVector<int64_t> subPivotShape(inputType.getShape());
          subPivotShape[inputRank - 2] = ShapedType::kDynamic;
          ArrayRef<int64_t> subDiagShape(subPivotShape.begin(),
                                         subPivotShape.end() - 1);
          auto subPivotTy = RankedTensorType::get(subPivotShape, elemTy);
          auto subDiagTy = RankedTensorType::get(subDiagShape, elemTy);
          Value rowPlusOne = b.create<arith::AddIOp>(loc, row, cstOne);
          offsets[inputRank - 2] = getAsOpFoldResult(rowPlusOne);
          sizes[inputRank - 2] = getAsOpFoldResult(
              b.create<arith::SubIOp>(loc, matDim, rowPlusOne));
          // offsets = [0, row + 1, row], sizes = [C, N - row - 1, 1] -> A_j,row
          // with j > row
          Value subDiag = b.create<tensor::ExtractSliceOp>(
              loc, subDiagTy, vals[0], offsets, sizes, strides);
          offsets.back() = cstZeroFold;
          sizes.back() = getAsOpFoldResult(matDim);
          // offsets = [0, row + 1, 0], sizes = [C, N - row - 1, N] -> elements
          // below pivot row
          Value subPivot = b.create<tensor::ExtractSliceOp>(
              loc, subPivotTy, vals[0], offsets, sizes, strides);
          Value initResult = b.create<tensor::EmptyOp>(loc, sizes, elemTy);
          // write a generic op to perform subpivot = subpivot -
          // (subdiag/diag)*pivot
          // d0 = batches, d1 = row, d2 = column -> pivot(d0,d2), diag(d0),
          // subPivot(d0,d1,d2), subDiag(d0, d1); output(d0,d1,d2)
          SmallVector<AffineExpr> allDims;
          for (unsigned i = 0; i < inputRank; i++)
            allDims.push_back(b.getAffineDimExpr(i));
          SmallVector<AffineExpr> rowIterator(1, allDims[0]);
          SmallVector<AffineExpr> colIterator;
          SmallVector<AffineExpr> batchIterator;
          if (isBatched) {
            rowIterator.push_back(allDims[1]);
            colIterator.push_back(allDims[0]);
            colIterator.push_back(rewriter.getAffineConstantExpr(0));
            colIterator.push_back(allDims[2]);
            batchIterator.push_back(allDims[0]);
            batchIterator.push_back(getAffineConstantExpr(0, context));
            batchIterator.push_back(getAffineConstantExpr(0, context));
          } else {
            colIterator.push_back(rewriter.getAffineConstantExpr(0));
            colIterator.push_back(allDims[1]);
            batchIterator.push_back(getAffineConstantExpr(0, context));
            batchIterator.push_back(getAffineConstantExpr(0, context));
          }
          SmallVector<AffineMap> indexingMaps;
          indexingMaps.push_back(
              AffineMap::get(inputRank, 0, colIterator, context));
          indexingMaps.push_back(
              AffineMap::get(inputRank, 0, batchIterator, context));
          indexingMaps.push_back(b.getMultiDimIdentityMap(inputRank));
          indexingMaps.push_back(
              AffineMap::get(inputRank, 0, rowIterator, context));
          indexingMaps.push_back(b.getMultiDimIdentityMap(inputRank));
          SmallVector<utils::IteratorType> iteratorTypes(
              inputRank, utils::IteratorType::parallel);
          Value reducedSubPivot =
              b.create<linalg::GenericOp>(
                   loc, subPivotTy, ValueRange{pivot, diag, subPivot, subDiag},
                   initResult, indexingMaps, iteratorTypes,
                   [&](OpBuilder &b, Location loc, ValueRange args) {
                     // for d0 in batches, d1 in subpivotrows, d2 in columns
                     // let i represent the pivot row index (scf loop index)
                     Value pivotd0d2 = args[0];
                     Value diagd0 = args[1];
                     Value subPivotd0d1d2 = args[2];
                     Value subDiagd0d1 = args[3];
                     // coeff = A_d1,i / A_i,i
                     Value coeff =
                         b.create<arith::DivFOp>(loc, subDiagd0d1, diagd0);
                     auto cmp = b.create<arith::CmpFOp>(
                         loc, arith::CmpFPredicate::ONE, diagd0, cstZeroF);
                     b.create<cf::AssertOp>(
                         loc, cmp,
                         b.getStringAttr(
                             "unimplemented: determinants requiring "
                             "permutations and singular matrices"));
                     // coeff*A_i,d2
                     Value scaledPivotValue =
                         b.create<arith::MulFOp>(loc, coeff, pivotd0d2);
                     // result = A_d1,d2 - (A_d1,i/A_i,i)*A_i,d2
                     // so that when d2 = i, A_d1,i - (A_d1,i/A_i,i) * A_i,i = 0
                     Value result = b.create<arith::SubFOp>(loc, subPivotd0d1d2,
                                                            scaledPivotValue);
                     b.create<linalg::YieldOp>(loc, result);
                   })
                  .getResult(0);
          Value rowReductionResult = b.create<tensor::InsertSliceOp>(
              loc, reducedSubPivot, vals[0], offsets, sizes, strides);
          b.create<scf::YieldOp>(loc,
                                 ValueRange{rowReductionResult, updatedDiags});
        });
    Value allDiagsExceptLast = rowReductionLoop.getResult(1);
    SmallVector<OpFoldResult> offsets(inputRank,
                                      getAsOpFoldResult(matDimMinusOne));
    SmallVector<OpFoldResult> strides(inputRank, getAsOpFoldResult(cstOne));
    SmallVector<OpFoldResult> sizes(inputRank, getAsOpFoldResult(cstOne));
    sizes[0] = getAsOpFoldResult(chDim);
    if (isBatched)
      offsets[0] = getAsOpFoldResult(cstZero);
    Value lastDiag = rewriter.create<tensor::ExtractSliceOp>(
        loc, diagTy, rowReductionLoop.getResult(0), offsets, sizes, strides);
    offsets.pop_back();
    strides.pop_back();
    sizes.pop_back();

    lastDiag = rewriter.create<tensor::CollapseShapeOp>(
        loc, diagCollapseTy, lastDiag, diagReassociations);

    Value allDiags = rewriter.create<tensor::InsertSliceOp>(
        loc, lastDiag, allDiagsExceptLast, offsets, sizes, strides);
    // linalg generic to do reduce prod for allDiags along back dim.
    // the result of that generic will be the determinant
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(inputRank - 1));
    AffineExpr resultExpr = isBatched ? rewriter.getAffineDimExpr(0)
                                      : getAffineConstantExpr(0, context);
    indexingMaps.push_back(AffineMap::get(inputRank - 1, 0, resultExpr));
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank - 2, utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);
    Value initDet = createInitTensor(rewriter, loc, ValueRange{chDim}, elemTy,
                                     getConstant(rewriter, loc, 1.0, elemTy));
    Value determinant =
        rewriter
            .create<linalg::GenericOp>(
                loc, initDet.getType(), ValueRange{allDiags}, initDet,
                indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value prod = b.create<arith::MulFOp>(loc, args[0], args[1]);
                  b.create<linalg::YieldOp>(loc, prod);
                })
            .getResult(0);
    Type newResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (isBatched) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
                                                  determinant);
      return success();
    }

    determinant = rewriter.create<tensor::CollapseShapeOp>(
        loc, newResultType, determinant,
        llvm::ArrayRef<ReassociationIndices>{});
    rewriter.replaceOp(op, ValueRange{determinant});
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenPolarOp : public OpConversionPattern<AtenPolarOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenPolarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();
    MLIRContext *context = rewriter.getContext();

    Value absTensor = adaptor.getAbs();
    Value angleTensor = adaptor.getAngle();

    RankedTensorType resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    auto elementType = resultType.getElementType();

    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < resultType.getRank(); i++) {
      auto currentDimSize = rewriter.create<tensor::DimOp>(loc, absTensor, i);
      resultShape.push_back(currentDimSize);
    }

    Value outTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultShape), elementType);

    SmallVector<AffineExpr> outputExpr;
    for (unsigned i = 0; i < resultType.getRank(); i++) {
      outputExpr.push_back(getAffineDimExpr(i, context));
    }

    AffineMap identityMap =
        AffineMap::get(resultType.getRank(), 0, outputExpr, op->getContext());

    SmallVector<AffineMap> indexingMaps{identityMap, identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);
    auto complexVar =
        rewriter
            .create<linalg::GenericOp>(
                loc, outTensor.getType(), ValueRange{absTensor, angleTensor},
                outTensor, indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  // out = abs⋅cos(angle) + abs⋅sin(angle)⋅j
                  Value abs = args[0];
                  Value angle = args[1];
                  Value realVal = b.create<math::CosOp>(loc, angle);
                  Value imagVal = b.create<math::SinOp>(loc, angle);
                  realVal = b.create<arith::MulFOp>(loc, abs, realVal);
                  imagVal = b.create<arith::MulFOp>(loc, abs, imagVal);
                  Value complexVal = b.create<complex::CreateOp>(
                      loc, elementType, realVal, imagVal);
                  b.create<linalg::YieldOp>(loc, complexVal);
                })
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, complexVar);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateUncategorizedOpsLegality(
    ConversionTarget &target) {
  target.addIllegalOp<
      AtenTanOp, AtenTanhOp, AtenSinhOp, AtenCoshOp, AtenAtanhOp, AtenAcoshOp,
      AtenAsinOp, AtenAsinhOp, AtenReluOp, AtenGeluOp, AtenGeluBackwardOp,
      AtenAddTensorOp, AtenMulTensorOp, AtenDivTensorOp, AtenDivTensorModeOp,
      AtenDivScalarModeOp, AtenSubTensorOp, AtenLerpTensorOp, AtenSigmoidOp,
      AtenMinimumOp, AtenAtan2Op, AtenMaximumOp, AtenToDtypeOp, AtenClampOp,
      AtenClampTensorOp, AtenRsubScalarOp, AtenLogOp, AtenErfOp, AtenSqrtOp,
      AtenFloorOp, AtenCeilOp, AtenPreluOp, AtenPowScalarOp,
      AtenPowTensorScalarOp, AtenPowTensorTensorOp, AtenLog2Op, AtenLog10Op,
      AtenLog1pOp, AtenRsqrtOp, AtenAbsOp, AtenReciprocalOp,
      AtenBitwiseAndTensorOp, AtenBitwiseAndScalarOp, AtenBitwiseOrTensorOp,
      AtenBitwiseXorTensorOp, AtenBitwiseLeftShiftTensorOp,
      AtenBitwiseRightShiftTensorOp, Aten__Lshift__ScalarOp,
      Aten__Rshift__ScalarOp, AtenGtScalarOp, AtenGeScalarOp, AtenEqScalarOp,
      AtenLtScalarOp, AtenLeScalarOp, AtenWhereSelfOp, AtenGtTensorOp,
      AtenGeTensorOp, AtenEqTensorOp, AtenNeTensorOp, AtenLtTensorOp,
      AtenLeTensorOp, AtenThresholdOp, AtenThresholdBackwardOp,
      AtenHardtanhBackwardOp, AtenCloneOp, AtenSinOp, AtenCosOp, AtenNeScalarOp,
      AtenMaskedFillTensorOp, AtenLogicalOrOp, AtenLogicalAndOp, AtenAtanOp,
      AtenAcosOp, AtenLogicalXorOp, AtenLogicalNotOp, AtenIsinfOp, AtenTriuOp,
      AtenTrilOp, AtenRemainderScalarOp, AtenRemainderTensorOp,
      AtenBitwiseNotOp, AtenRoundOp, AtenFillScalarOp, AtenFillTensorOp,
      AtenRealOp, AtenImagOp, AtenDequantizeSelfOp, AtenDequantizeTensorOp,
      AtenQuantizePerTensorOp, AtenIscloseOp, AtenNllLossForwardOp,
      AtenDetachOp, AtenBatchNormOp, AtenLogitOp, PrimsCollapseOp,
      PrimsSplitDimOp, AtenNllLossBackwardOp, TensorStaticInfoCastOp,
      AtenIntReprOp, Aten_MakePerChannelQuantizedTensorOp,
      Aten_MakePerTensorQuantizedTensorOp, AtenGridSamplerOp,
      Aten__InterpolateSizeListScaleListOp, AtenLinalgDetOp, AtenPolarOp>();
}

void mlir::torch::torch_to_linalg::populateUncategorizedPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<ConvertElementwiseOp>(typeConverter, context);
  patterns.add<ConvertAtenDetachOp>(typeConverter, context);
  patterns.add<ConvertAtenNllLossForwardOp>(typeConverter, context);
  patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
  patterns.add<ConvertLogitOp>(typeConverter, context);
  patterns.add<ConvertPrimsCollapseOp>(typeConverter, context);
  patterns.add<ConvertPrimsSplitDimOp>(typeConverter, context);
  patterns.add<ConvertAtenNllLossBackwardOp>(typeConverter, context);
  patterns.add<ConvertTensorStaticInfoCastOp>(typeConverter, context);
  patterns.add<ConvertAtenIntReprOp>(typeConverter, context);
  patterns.add<ConvertCastEquivalentOp<Aten_MakePerChannelQuantizedTensorOp>>(
      typeConverter, context);
  patterns.add<ConvertCastEquivalentOp<Aten_MakePerTensorQuantizedTensorOp>>(
      typeConverter, context);
  patterns.add<ConvertDequantizePerChannel>(typeConverter, context);
  patterns.add<ConvertAtenGridSamplerOp>(typeConverter, context);
  patterns.add<ConvertInterpolateOp>(typeConverter, context);
  patterns.add<ConvertAtenLinalgDetOp>(typeConverter, context);
  patterns.add<ConvertAtenPolarOp>(typeConverter, context);
}
