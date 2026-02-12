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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>
#include <string>
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
    return arith::CmpFOp::create(b, loc, fpred, lhs, rhs);
  if (IntegerType intType = dyn_cast<mlir::IntegerType>(type)) {
    if (intType.isUnsigned())
      return arith::CmpIOp::create(b, loc, iupred, lhs, rhs);
    if (intType.isSigned())
      return arith::CmpIOp::create(b, loc, ispred, lhs, rhs);
    assert(intType.getWidth() == 1);
    return arith::CmpIOp::create(b, loc, iupred, lhs, rhs);
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
  Value xMinusMean = arith::SubFOp::create(b, loc, x, mean);
  Value two = arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 2));
  Value sqrt2 = math::SqrtOp::create(b, loc, two);
  Value erfArg = arith::DivFOp::create(b, loc, xMinusMean, sqrt2);
  Value erf = math::ErfOp::create(b, loc, erfArg);
  Value one = arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 1));
  Value erfPlus1 = arith::AddFOp::create(b, loc, one, erf);
  Value oneHalf =
      arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 0.5));
  Value normalCdf = arith::MulFOp::create(b, loc, oneHalf, erfPlus1);
  return normalCdf;
}

static Value buildUnitNormalCdf(OpBuilder &b, Location &loc, Value x) {
  Type elementType = x.getType();
  Value zero =
      arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 0));
  Value one = arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 1));
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
  auto newOp = MathOpTy::create(b, loc, arg);
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
  auto rowIndex = linalg::IndexOp::create(b, loc, inputRank - 2);
  Value rowIndexI64 = castIndexToInt64(b, loc, rowIndex);
  auto colIndex = linalg::IndexOp::create(b, loc, inputRank - 1);
  Value colIndexI64 = castIndexToInt64(b, loc, colIndex);

  // columnIndex >= rowIndex + diagonal?
  auto sum =
      arith::AddIOp::create(b, loc, rowIndexI64, /*diagonal=*/operands[1]);
  auto pred = arith::CmpIOp::create(b, loc, predicate, colIndexI64, sum);

  Value scalar = payloadArgs[0];
  Type elementType = inputType.getElementType();
  Value zero = getConstant(b, loc, 0, elementType);
  result = arith::SelectOp::create(b, loc, pred, scalar, zero);
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
    quotient = arith::DivFOp::create(b, loc, lhs, rhs);
  } else if (dtype.isUnsignedInteger()) {
    quotient = arith::DivUIOp::create(b, loc, lhs, rhs);
  } else {
    assert(dtype.isInteger() &&
           "dtype should be an integer (signless or signed)");
    quotient = arith::DivSIOp::create(b, loc, lhs, rhs);
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
    Value ceil = math::CeilOp::create(b, loc, quotient);
    Value floor = math::FloorOp::create(b, loc, quotient);
    Value cstZero = arith::ConstantOp::create(b, loc, b.getZeroAttr(dtype));
    Value pred = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULT,
                                       quotient, cstZero);
    return arith::SelectOp::create(b, loc, pred, ceil, floor);
  }
  if (roundingMode == "floor") {
    // "floor" - rounds the results of the division down. Equivalent to
    // floor division in Python (the // operator)
    if (isa<mlir::FloatType>(dtype))
      return math::FloorOp::create(b, loc, quotient);
    if (!dtype.isUnsignedInteger()) {
      Type defaultIntToFloatType = b.getF64Type();
      lhs = convertScalarToDtype(b, loc, lhs, defaultIntToFloatType);
      rhs = convertScalarToDtype(b, loc, rhs, defaultIntToFloatType);
      quotient = arith::DivFOp::create(b, loc, lhs, rhs);
      Value floor = math::FloorOp::create(b, loc, quotient);
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
    Value remainder = arith::RemFOp::create(b, loc, lhs, rhs);

    Value zero = arith::ConstantOp::create(b, loc, b.getZeroAttr(dtype));
    Value remainderNotEqualToZero = arith::CmpFOp::create(
        b, loc, arith::CmpFPredicate::ONE, remainder, zero);
    Value otherLessThanZero =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, rhs, zero);
    Value remainderLessThanZero = arith::CmpFOp::create(
        b, loc, arith::CmpFPredicate::OLT, remainder, zero);
    Value xorCondition =
        arith::XOrIOp::create(b, loc, otherLessThanZero, remainderLessThanZero);
    Value condition =
        arith::AndIOp::create(b, loc, remainderNotEqualToZero, xorCondition);
    Value fixedRemainder = arith::AddFOp::create(b, loc, remainder, rhs);
    result =
        arith::SelectOp::create(b, loc, condition, fixedRemainder, remainder);
  } else {
    assert(dtype.isInteger() &&
           "dtype should be a float or integer (signless or signed)");
    Value remainder = arith::RemSIOp::create(b, loc, lhs, rhs);

    Value zero = arith::ConstantOp::create(b, loc, b.getZeroAttr(dtype));
    Value remainderNotEqualToZero = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ne, remainder, zero);
    Value otherLessThanZero =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, rhs, zero);
    Value remainderLessThanZero = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::slt, remainder, zero);
    Value xorCondition =
        arith::XOrIOp::create(b, loc, otherLessThanZero, remainderLessThanZero);
    Value condition =
        arith::AndIOp::create(b, loc, remainderNotEqualToZero, xorCondition);
    Value fixedRemainder = arith::AddIOp::create(b, loc, remainder, rhs);
    result =
        arith::SelectOp::create(b, loc, condition, fixedRemainder, remainder);
  }
  return result;
}

static Value createLinalgPayloadCalculationForElementwiseOp(
    OpBuilder &b, Location loc, const TypeConverter *converter,
    ValueRange payloadArgs, Operation *op, ArrayRef<Value> operands) {
  if (isa<AtenFloorOp>(op))
    return math::FloorOp::create(b, loc, payloadArgs[0]);
  if (isa<AtenCeilOp>(op))
    return math::CeilOp::create(b, loc, payloadArgs[0]);
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
    return arith::AndIOp::create(b, loc, lhs, rhs);
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
    return arith::AndIOp::create(b, loc, self, other);
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
    return arith::OrIOp::create(b, loc, lhs, rhs);
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
    return arith::XOrIOp::create(b, loc, lhs, rhs);
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
    return arith::ShRSIOp::create(b, loc, lhs, rhs);
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
    return arith::ShLIOp::create(b, loc, lhs, rhs);
  }
  if (isa<AtenLogicalOrOp, AtenLogicalAndOp, AtenLogicalXorOp>(op)) {
    MLIRContext *context = op->getContext();
    Type floatDtype = mlir::Float64Type::get(context);
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], floatDtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], floatDtype);
    Value zero =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(floatDtype, 0));
    Value lhsTest = createNotEqual(b, loc, floatDtype, lhs, zero);
    Value rhsTest = createNotEqual(b, loc, floatDtype, rhs, zero);
    if (isa<AtenLogicalOrOp>(op)) {
      return arith::OrIOp::create(b, loc, lhsTest, rhsTest);
    }
    if (isa<AtenLogicalAndOp>(op)) {
      return arith::AndIOp::create(b, loc, lhsTest, rhsTest);
    }
    if (isa<AtenLogicalXorOp>(op)) {
      return arith::XOrIOp::create(b, loc, lhsTest, rhsTest);
    }
    llvm_unreachable("Unknown op type");
  }
  if (isa<AtenLogicalNotOp>(op)) {
    MLIRContext *context = op->getContext();
    Type floatDtype = mlir::Float64Type::get(context);
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], floatDtype);
    Value zero =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(floatDtype, 0));
    return createEqual(b, loc, floatDtype, self, zero);
  }
  if (auto complex = dyn_cast<AtenComplexOp>(op)) {
    auto ctype = cast<ComplexType>(
        cast<RankedTensorType>(converter->convertType(complex.getType()))
            .getElementType());
    Type stype = ctype.getElementType();

    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], stype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], stype);
    return complex::CreateOp::create(b, loc, ctype, lhs, rhs);
  }
  if (isa<AtenAbsOp>(op)) {
    if (isa<IntegerType>(payloadArgs[0].getType()))
      return math::AbsIOp::create(b, loc, payloadArgs[0]);
    return math::AbsFOp::create(b, loc, payloadArgs[0]);
  }
  if (isa<AtenIsinfOp>(op)) {
    Value abs = math::AbsFOp::create(b, loc, payloadArgs[0]);
    Value infinity = arith::ConstantOp::create(
        b, loc,
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
    auto negate = arith::NegFOp::create(b, loc, arg);
    auto one =
        arith::ConstantOp::create(b, loc, FloatAttr::get(negate.getType(), 1));
    auto exp = math::ExpOp::create(b, loc, negate);
    auto added = arith::AddFOp::create(b, loc, exp, one);
    auto div = arith::DivFOp::create(b, loc, one, added);
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
      auto minForIntTypeValue = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(zeroPoint.getType(), minForIntType));
      auto maxForIntTypeValue = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(zeroPoint.getType(), maxForIntType));
      auto zpLtMax = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                           zeroPoint, maxForIntTypeValue);
      cf::AssertOp::create(
          b, loc, zpLtMax,
          b.getStringAttr("Invalid Quantization: quantized relu with "
                          "zero-point > max qint"));
      auto zpLtMin = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                           zeroPoint, minForIntTypeValue);
      zeroPoint = arith::SelectOp::create(b, loc, zpLtMin, minForIntTypeValue,
                                          zeroPoint);
      zeroPoint = arith::TruncIOp::create(b, loc, arg.getType(), zeroPoint);
    } else {
      zeroPoint =
          arith::ConstantOp::create(b, loc, b.getZeroAttr(arg.getType()));
    }
    Value cmp;
    if (intType) {
      auto pred =
          isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
      cmp = arith::CmpIOp::create(b, loc, pred, arg, zeroPoint);
    } else {
      cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UGT, arg,
                                  zeroPoint);
    }
    return arith::SelectOp::create(b, loc, cmp, arg, zeroPoint);
  }
  if (auto round = dyn_cast<AtenRoundOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(round.getType()).getDtype())) {
      round.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    return math::RoundEvenOp::create(b, loc, payloadArgs[0]);
  }
  if (auto prelu = dyn_cast<AtenPreluOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(prelu.getType()).getDtype())) {
      prelu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type elementType = payloadArgs[0].getType();
    Value constZero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(elementType));
    Value pred = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UGT,
                                       payloadArgs[0], constZero);
    Value positivePart =
        arith::SelectOp::create(b, loc, pred, payloadArgs[0], constZero);
    Value negativePart =
        arith::SelectOp::create(b, loc, pred, constZero, payloadArgs[0]);
    Value scale = convertScalarToDtype(b, loc, payloadArgs[1], elementType);
    Value scaledNegativePart =
        arith::MulFOp::create(b, loc, negativePart, scale);
    return arith::AddFOp::create(b, loc, positivePart, scaledNegativePart);
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
      return arith::MulFOp::create(b, loc, payloadArgs[0], multiplier);
    }
    if (approximate == "tanh") {
      // GELU(x)=0.5∗x∗(1+Tanh((2/π)^1/2 * (x+0.044715∗x^3)))
      // Ref: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
      Value cstThree = arith::ConstantOp::create(
          b, loc, IntegerAttr::get(IntegerType::get(op->getContext(), 64), 3));
      Value xCube = math::FPowIOp::create(b, loc, payloadArgs[0], cstThree);
      Type elementType = payloadArgs[0].getType();
      Value cstAlpha = arith::ConstantOp::create(
          b, loc, FloatAttr::get(elementType, 0.044715));
      Value xCubeMulAlpha = arith::MulFOp::create(b, loc, xCube, cstAlpha);
      Value xPlusXCubeMulAlpha =
          arith::AddFOp::create(b, loc, payloadArgs[0], xCubeMulAlpha);
      Value cstBeta = arith::ConstantOp::create(
          b, loc, FloatAttr::get(elementType, 0.7977240352174656));
      Value betaMulX =
          arith::MulFOp::create(b, loc, cstBeta, xPlusXCubeMulAlpha);
      Value tanh = math::TanhOp::create(b, loc, betaMulX);
      Value cstOne =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 1.0));
      Value onePlusTanh = arith::AddFOp::create(b, loc, cstOne, tanh);
      Value cstHalf =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 0.5));
      Value multiplier = arith::MulFOp::create(b, loc, cstHalf, onePlusTanh);
      return arith::MulFOp::create(b, loc, payloadArgs[0], multiplier);
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
    Value cstAlpha0 = arith::ConstantOp::create(
        b, loc, FloatAttr::get(elementType, 1.12837916709551257390));
    Value cstAlpha1 = arith::ConstantOp::create(
        b, loc, FloatAttr::get(elementType, 0.70710678118654752440));
    Value oneHalf =
        arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 0.5));
    Value kAlpha = arith::MulFOp::create(b, loc, cstAlpha0, cstAlpha1);
    Value kAlphaHalf = arith::MulFOp::create(b, loc, kAlpha, oneHalf);
    Value negOneHalf =
        arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, -0.5));
    Value inputSquared =
        arith::MulFOp::create(b, loc, payloadArgs[1], payloadArgs[1]);
    Value negHalfInputSquared =
        arith::MulFOp::create(b, loc, inputSquared, negOneHalf);
    Value dinput = math::ExpOp::create(b, loc, negHalfInputSquared);
    Value cdf = buildUnitNormalCdf(b, loc, payloadArgs[1]);
    Value dinputInput = arith::MulFOp::create(b, loc, dinput, payloadArgs[1]);
    Value dinputInputAlpha =
        arith::MulFOp::create(b, loc, dinputInput, kAlphaHalf);
    Value cdfExt = arith::AddFOp::create(b, loc, dinputInputAlpha, cdf);
    return arith::MulFOp::create(b, loc, payloadArgs[0], cdfExt);
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
        arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 0.0));
    Value min = convertScalarToDtype(b, loc, adaptor.getMinVal(), elementType);
    Value max = convertScalarToDtype(b, loc, adaptor.getMaxVal(), elementType);
    Value lesser =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULT, self, min);
    Value greater =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UGT, self, max);
    Value cmp = arith::OrIOp::create(b, loc, lesser, greater);
    return arith::SelectOp::create(b, loc, cmp, constantZero, gradOutput);
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
      Value scaled = arith::MulFOp::create(b, loc, rhs, alpha);
      return arith::AddFOp::create(b, loc, lhs, scaled);
    } else if (dtype.isInteger(1)) {
      Value scaled = arith::MulIOp::create(b, loc, rhs, alpha);
      return arith::OrIOp::create(b, loc, lhs, scaled);
    } else {
      Value scaled = arith::MulIOp::create(b, loc, rhs, alpha);
      return arith::AddIOp::create(b, loc, lhs, scaled);
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
      Value scaled = arith::MulFOp::create(b, loc, rhs, alpha);
      return arith::SubFOp::create(b, loc, lhs, scaled);
    } else {
      Value scaled = arith::MulIOp::create(b, loc, rhs, alpha);
      return arith::SubIOp::create(b, loc, lhs, scaled);
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
    return arith::ShLIOp::create(b, loc, self, other);
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
    return arith::ShRUIOp::create(b, loc, self, other);
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
      Value mult = arith::MulFOp::create(b, loc, other, alpha);
      return arith::SubFOp::create(b, loc, self, mult);
    } else if (isa<mlir::IntegerType>(dtype)) {
      Value mult = arith::MulIOp::create(b, loc, other, alpha);
      return arith::SubIOp::create(b, loc, self, mult);
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
      Value mult = arith::MulFOp::create(b, loc, other, alpha);
      return arith::AddFOp::create(b, loc, self, mult);
    } else if (isa<mlir::IntegerType>(dtype)) {
      Value mult = arith::MulIOp::create(b, loc, other, alpha);
      return arith::AddIOp::create(b, loc, self, mult);
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
      return arith::MulFOp::create(b, loc, lhs, rhs);
    } else if (isa<mlir::ComplexType>(dtype)) {
      return complex::MulOp::create(b, loc, lhs, rhs);
    } else {
      return arith::MulIOp::create(b, loc, lhs, rhs);
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
    return math::Atan2Op::create(b, loc, lhs, rhs);
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
      return arith::DivFOp::create(b, loc, lhs, rhs);
    else if (isa<mlir::IntegerType>(dtype)) {
      if (dtype.isUnsignedInteger())
        return arith::DivUIOp::create(b, loc, lhs, rhs);
      return arith::DivSIOp::create(b, loc, lhs, rhs);
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
    return math::PowFOp::create(b, loc, selfPromoted, expPromoted);
  }

  if (auto pow = dyn_cast<AtenPowTensorScalarOp>(op)) {
    if (!isa<mlir::FloatType>(
            cast<ValueTensorType>(pow.getType()).getDtype())) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value exp = operands[1];
    Type expType = exp.getType();
    if (!expType.isIntOrFloat()) {
      pow.emitError("unimplemented: exp type neither float nor int");
      return nullptr;
    }
    if (isa<mlir::IntegerType>(expType)) {
      return math::FPowIOp::create(b, loc, payloadArgs[0], exp);
    }
    Type dtype = cast<ValueTensorType>(pow.getSelf().getType()).getDtype();
    Value expPromoted = convertScalarToDtype(b, loc, operands[1], dtype);
    return math::PowFOp::create(b, loc, payloadArgs[0], expPromoted);
  }

  if (auto pow = dyn_cast<AtenPowTensorTensorOp>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(pow.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      // The result type is integer when both operands are integer.
      // Torch then uses the following implementation:
      // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Pow.h
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type powType = dtype;
    if (payloadArgs[0].getType().isInteger() ||
        payloadArgs[1].getType().isInteger())
      powType = mlir::Float64Type::get(op->getContext());
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], powType);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], powType);
    auto powOp = math::PowFOp::create(b, loc, lhs, rhs);
    return convertScalarToDtype(b, loc, powOp, dtype);
  }

  if (auto imag = dyn_cast<AtenImagOp>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(imag.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      imag.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value imagVal = complex::ImOp::create(b, loc, payloadArgs[0]);
    return imagVal;
  }

  if (auto real = dyn_cast<AtenRealOp>(op)) {
    Type dtype = cast<RankedTensorType>(converter->convertType(real.getType()))
                     .getElementType();
    if (!isa<mlir::FloatType>(dtype)) {
      real.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value realVal = complex::ReOp::create(b, loc, payloadArgs[0]);
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
    return arith::SelectOp::create(b, loc, payloadArgs[0], lhs, rhs);
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
    auto delta = arith::SubFOp::create(b, loc, end, start);
    auto weightedDelta = arith::MulFOp::create(b, loc, delta, weight);
    return arith::AddFOp::create(b, loc, start, weightedDelta);
  }
  if (auto minimum = dyn_cast<AtenMinimumOp>(op)) {
    Type dtype = cast<BaseTensorType>(minimum.getType()).getDtype();
    Type elemTy =
        cast<RankedTensorType>(converter->convertType(minimum.getType()))
            .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], elemTy);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], elemTy);
    Value pred = createLessThan(b, loc, dtype, lhs, rhs);
    return arith::SelectOp::create(b, loc, pred, lhs, rhs);
  }
  if (auto maximum = dyn_cast<AtenMaximumOp>(op)) {
    Type dtype = cast<BaseTensorType>(maximum.getType()).getDtype();
    Type elemTy =
        cast<RankedTensorType>(converter->convertType(maximum.getType()))
            .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], elemTy);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], elemTy);
    Value pred = createGreaterThan(b, loc, dtype, lhs, rhs);
    return arith::SelectOp::create(b, loc, pred, lhs, rhs);
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
        pred = arith::CmpFOp::create(b, loc, cmp, input, clamp);
      } else if (isa<mlir::IntegerType>(dtype)) {
        auto cmp =
            isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
        if (getMax)
          cmp = arith::invertPredicate(cmp);
        pred = arith::CmpIOp::create(b, loc, cmp, input, clamp);
      }
      return arith::SelectOp::create(b, loc, pred, clamp, input);
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
        pred = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULT, result,
                                     minPromoted);
      } else if (isa<mlir::IntegerType>(dtype)) {
        pred = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, result,
                                     minPromoted);
      } else {
        clampTensor.emitError(
            "unimplemented: dtype other than float and integer "
            "types are not supported.");
        return nullptr;
      }
      result = arith::SelectOp::create(b, loc, pred, minPromoted, result);
    }
    if (!isa<Torch::NoneType>(max.getType())) {
      max = isMinNone ? payloadArgs[1] : payloadArgs[2];
      auto maxPromoted = convertScalarToDtype(b, loc, max, dtype);
      Value pred;
      if (isa<mlir::FloatType>(dtype)) {
        pred = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UGT, result,
                                     maxPromoted);
      } else if (isa<mlir::IntegerType>(dtype)) {
        pred = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, result,
                                     maxPromoted);
      } else {
        clampTensor.emitError(
            "unimplemented: dtype other than float and integer "
            "types are not supported.");
        return nullptr;
      }
      result = arith::SelectOp::create(b, loc, pred, maxPromoted, result);
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
      Value mult = arith::MulFOp::create(b, loc, self, alpha);
      return arith::SubFOp::create(b, loc, other, mult);
    } else if (isa<mlir::IntegerType>(dtype)) {
      Value mult = arith::MulIOp::create(b, loc, self, alpha);
      return arith::SubIOp::create(b, loc, other, mult);
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
      return arith::MulFOp::create(b, loc, lhs, rhs);
    if (isa<mlir::IntegerType>(dtype))
      return arith::MulIOp::create(b, loc, lhs, rhs);
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
    return arith::DivFOp::create(b, loc, self, other);
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
        arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 0.0));
    auto pred =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ONE, arg, zero);
    cf::AssertOp::create(
        b, loc, pred,
        b.getStringAttr("unimplemented: tensor with zero element"));

    auto one =
        arith::ConstantOp::create(b, loc, FloatAttr::get(elementType, 1.0));
    return arith::DivFOp::create(b, loc, one, arg);
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
      predicate = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULE, self,
                                        threshold);
    else
      predicate = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, self,
                                        threshold);
    return arith::SelectOp::create(b, loc, predicate, value, self);
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
    Value constantZero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(dtype));

    Value predicate;
    if (isa<mlir::FloatType>(dtype))
      predicate = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULE, self,
                                        threshold);
    else
      predicate = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, self,
                                        threshold);
    return arith::SelectOp::create(b, loc, predicate, constantZero, grad);
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
    return arith::SelectOp::create(b, loc, mask, fillValue, input);
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

    Value allOnesVal = arith::ConstantOp::create(
        b, loc,
        b.getIntegerAttr(
            elementType,
            APSInt::getAllOnes(elementType.getIntOrFloatBitWidth())));
    return arith::XOrIOp::create(b, loc, payloadArgs[0], allOnesVal);
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
        value = arith::ExtUIOp::create(b, loc, outIntTy, value);
      } else {
        value = arith::ExtSIOp::create(b, loc, outIntTy, value);
      }
    }

    zp = converter->materializeTargetConversion(
        b, loc, converter->convertType(zp.getType()), zp);
    auto zpTy = zp.getType();

    if (zpTy != outIntTy) {
      zp = arith::TruncIOp::create(b, loc, outIntTy, zp);
    }

    value = arith::SubIOp::create(b, loc, value, zp);
    // treat the i32 as a signed int regardless of original signed-ness
    // this will prevent overflow from subtraction for unsigned quantizations.
    value = arith::SIToFPOp::create(b, loc, outFpTy, value);

    scale = converter->materializeTargetConversion(
        b, loc, converter->convertType(scale.getType()), scale);
    if (scale.getType() != value.getType()) {
      scale = arith::TruncFOp::create(b, loc, value.getType(), scale);
    }
    value = arith::MulFOp::create(b, loc, value, scale);
    return value;
  }

  if (auto quant = dyn_cast<AtenQuantizePerTensorOp>(op)) {
    Value value = payloadArgs[0];
    Value scale = quant.getScale();
    Value zp = quant.getZeroPoint();
    auto valueTy = value.getType();

    zp = converter->materializeTargetConversion(
        b, loc, converter->convertType(zp.getType()), zp);
    zp = arith::SIToFPOp::create(b, loc, valueTy, zp);

    scale = converter->materializeTargetConversion(
        b, loc, converter->convertType(scale.getType()), scale);
    scale = arith::TruncFOp::create(b, loc, valueTy, scale);

    value = arith::DivFOp::create(b, loc, value, scale);
    value = math::RoundEvenOp::create(b, loc, value);
    value = arith::AddFOp::create(b, loc, value, zp);

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
        arith::ConstantOp::create(b, loc, b.getFloatAttr(valueTy, minI));
    Value maxVal =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(valueTy, maxI));
    value = arith::MaximumFOp::create(b, loc, value, minVal);
    value = arith::MinimumFOp::create(b, loc, value, maxVal);

    if (isUnsigned) {
      value = arith::FPToUIOp::create(b, loc, destTy, value);
    } else {
      value = arith::FPToSIOp::create(b, loc, destTy, value);
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
    auto diff = arith::SubFOp::create(b, loc, computeType, cvtArg0, cvtArg1);
    auto absDiff = math::AbsFOp::create(b, loc, computeType, diff);
    auto cstRtol =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(computeType, rtol));
    auto absOther = math::AbsFOp::create(b, loc, computeType, cvtArg1);
    auto mul = arith::MulFOp::create(b, loc, computeType, cstRtol, absOther);
    auto cstAtol =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(computeType, atol));
    auto threshold = arith::AddFOp::create(b, loc, computeType, cstAtol, mul);
    return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULE, absDiff,
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
             AtenComplexOp, AtenReciprocalOp, AtenBitwiseAndTensorOp,
             AtenBitwiseAndScalarOp, AtenBitwiseOrTensorOp,
             AtenBitwiseXorTensorOp, AtenBitwiseLeftShiftTensorOp,
             AtenBitwiseRightShiftTensorOp, Aten__Lshift__ScalarOp,
             Aten__Rshift__ScalarOp, AtenGtScalarOp, AtenGeScalarOp,
             AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp, AtenWhereSelfOp,
             AtenCeilOp, AtenGtTensorOp, AtenGeTensorOp, AtenEqTensorOp,
             AtenNeTensorOp, AtenLtTensorOp, AtenLeTensorOp, AtenSubScalarOp,
             AtenAddScalarOp, AtenThresholdOp, AtenThresholdBackwardOp,
             AtenHardtanhBackwardOp, AtenCloneOp, AtenSinOp, AtenCosOp,
             AtenNeScalarOp, AtenNegOp, AtenMaskedFillTensorOp, AtenLogicalOrOp,
             AtenLogicalAndOp, AtenLogicalXorOp, AtenLogicalNotOp, AtenIsinfOp,
             AtenTriuOp, AtenTrilOp, AtenBitwiseNotOp, AtenRoundOp,
             AtenFillScalarOp, AtenFillTensorOp, AtenAtanOp, AtenAcosOp,
             AtenAtanhOp, AtenAcoshOp, AtenAsinOp, AtenAsinhOp, AtenRealOp,
             AtenImagOp, AtenDequantizeSelfOp, AtenDequantizeTensorOp,
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
          linalg::YieldOp::create(b, loc, result);
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

    Value zeroVal = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(elementType));

    Value finalRes = torch_to_linalg::createElementwiseLinalgGeneric(
        rewriter, loc, {target}, elementType,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value targetVal = args[0];
          Value indTarget = arith::IndexCastOp::create(
              rewriter, loc, rewriter.getIndexType(), targetVal);

          // The final result is given by:
          // final_res = (indTarget == ignoreIndexVal) ? 0 :
          // input[indI][IndTarget]
          Value cmpEq =
              arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    indTarget, ignoreIndexVal);

          SmallVector<Value> extractionIndices{indTarget};
          if (inputRank == 2) {
            Value indI = linalg::IndexOp::create(rewriter, loc, 0);
            extractionIndices.insert(extractionIndices.begin(), indI);
          }

          Value result = tensor::ExtractOp::create(rewriter, loc, input,
                                                   extractionIndices);

          Value negate =
              arith::NegFOp::create(rewriter, loc, elementType, result);
          Value selectFinal =
              arith::SelectOp::create(rewriter, loc, cmpEq, zeroVal, negate);
          linalg::YieldOp::create(b, loc, selectFinal);
        });

    llvm::iota_range<int64_t> dimsToReduce(0, targetRank,
                                           /*inclusive=*/false);
    DenseSet<int64_t> dimSet(dimsToReduce.begin(), dimsToReduce.end());

    if (reduction == torch_upstream::Reduction::Sum ||
        reduction == torch_upstream::Reduction::Mean) {

      Value zeroIVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(rewriter.getI32Type()));
      auto countInfo = torch_to_linalg::ReductionOpInfo{false, target, dimSet};
      Value numOfElems = torch_to_linalg::createReductionLinalgGeneric(
          rewriter, loc, countInfo,
          /*initElem=*/zeroIVal,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value targetVal = args[0];
            Value indTarget = arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), targetVal);
            Value cmpEq =
                arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                      indTarget, ignoreIndexVal);
            cmpEq = arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(),
                                           cmpEq);
            Value add = arith::AddIOp::create(rewriter, loc, args[1], cmpEq);
            linalg::YieldOp::create(rewriter, loc, add);
          });

      numOfElems = tensor::ExtractOp::create(
          rewriter, loc, rewriter.getI32Type(), numOfElems, ArrayRef<Value>{});
      numOfElems = convertScalarToDtype(rewriter, loc, numOfElems, elementType);

      auto opInfo = torch_to_linalg::ReductionOpInfo{false, finalRes, dimSet};
      finalRes = torch_to_linalg::createReductionLinalgGeneric(
          rewriter, loc, opInfo,
          /*initElem=*/zeroVal,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value newVal = args[0];
            Value accumulator = args[1];
            if (reduction == torch_upstream::Reduction::Mean)
              newVal = arith::DivFOp::create(b, loc, newVal, numOfElems);
            Value result = arith::AddFOp::create(b, loc, newVal, accumulator);
            linalg::YieldOp::create(b, loc, result);
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
      Value targetVal = tensor::ExtractOp::create(rewriter, loc, target);
      numIgnoredIndex = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, targetVal, ignoreIndex);
      numIgnoredIndex = convertScalarToDtype(rewriter, loc, numIgnoredIndex,
                                             ignoreIndex.getType());
    } else {
      Value zeroCstInt = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(ignoreIndex.getType()));

      auto opInfo =
          torch_to_linalg::ReductionOpInfo{/*keepDim=*/false, target, dimSet};
      numIgnoredIndex = torch_to_linalg::createReductionLinalgGeneric(
          rewriter, loc, opInfo,
          /*initElem=*/zeroCstInt,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value targetVal = args[0];
            Value accumulator = args[1];
            Value result = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::eq, targetVal, ignoreIndex);
            result = arith::AddIOp::create(
                b, loc,
                convertScalarToDtype(rewriter, loc, result,
                                     ignoreIndex.getType()),
                accumulator);
            linalg::YieldOp::create(b, loc, result);
          });

      numIgnoredIndex =
          tensor::ExtractOp::create(rewriter, loc, numIgnoredIndex);
    }

    Value numtargetElems = getTensorSize(rewriter, loc, target);
    Value totalWeightVal =
        arith::SubIOp::create(rewriter, loc, numtargetElems, numIgnoredIndex);
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
  Value truncatedEps = arith::TruncFOp::create(b, loc, elemTy, eps);
  Value varPlusEps = arith::AddFOp::create(b, loc, var, truncatedEps);
  Value rSTD = math::RsqrtOp::create(b, loc, varPlusEps);
  return rSTD;
}

// Normalization formula:
//   ((input - mean) * rSTD * weight + bias
static Value createLinalgPayloadCalculationForNormOpsWithRSTD(
    OpBuilder &b, Location loc, Type elemTy, Value input, Value mean,
    Value rSTD, Value eps, Value weight, Value bias) {
  Value inputSubMean = arith::SubFOp::create(b, loc, input, mean);
  Value temp = arith::MulFOp::create(b, loc, inputSubMean, rSTD);
  Value timesWeight = arith::MulFOp::create(b, loc, temp, weight);
  Value plusBias = arith::AddFOp::create(b, loc, timesWeight, bias);
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
    auto constFalse = arith::ConstantOp::create(
        rewriter, loc, IntegerAttr::get(IntegerType::get(context, 1), 0));
    auto trainingFalse = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, training, constFalse);
    cf::AssertOp::create(
        rewriter, loc, trainingFalse,
        rewriter.getStringAttr("training is not supported for now"));

    // num_features – C from an expected input of size (N,C,D,H,W ...)
    Value numFeatures = tensor::DimOp::create(rewriter, loc, input, 1);
    auto contractingDim0EqualsNumFeatures = [&](Value v) {
      auto dim0 = tensor::DimOp::create(rewriter, loc, v, 0);
      auto dim0Equal = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, numFeatures, dim0);
      cf::AssertOp::create(
          rewriter, loc, dim0Equal,
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
        linalg::GenericOp::create(
            rewriter, loc, input.getType(),
            ValueRange{input, weight, bias, runningMean, runningVar}, input,
            /*indexingMaps=*/indexingMaps,
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value input = args[0], weight = args[1], bias = args[2],
                    mean = args[3], var = args[4];
              Value result = createLinalgPayloadCalculationForNormOpsWithVar(
                  b, loc, var.getType(), input, mean, var, eps, weight, bias);
              linalg::YieldOp::create(b, loc, result);
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
        linalg::GenericOp::create(
            rewriter, loc, gradInputTensor.getType(),
            ValueRange{gradOutput, target, totalWeight}, gradInputTensor,
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value gradOutElem = args[0];
              Value targetElem = castIntToIndex(b, loc, args[1]);
              Value totalWeightElem = args[2];
              Value classIndex = linalg::IndexOp::create(b, loc, inputRank - 1);

              if (reduction == torch_upstream::Reduction::Mean) {
                gradOutElem =
                    arith::DivFOp::create(b, loc, gradOutElem, totalWeightElem);
              }

              Value negGradOutElem = arith::NegFOp::create(b, loc, gradOutElem);
              Value weightElem = getConstant(b, loc, 1, resultElementType);
              if (!weightIsNone) {
                weightElem =
                    tensor::ExtractOp::create(b, loc, weight, targetElem);
              }
              Value weightedNegGradOutElem =
                  arith::MulFOp::create(b, loc, weightElem, negGradOutElem);

              Value targetNeqClassIndex = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::ne, targetElem, classIndex);
              Value targetEqIgnoreIndex = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::eq, targetElem, ignoreIndex);
              Value gradInputIsZero = arith::OrIOp::create(
                  b, loc, targetNeqClassIndex, targetEqIgnoreIndex);

              Value zero = getConstant(b, loc, 0, resultElementType);
              Value gradInElem = arith::SelectOp::create(
                  b, loc, gradInputIsZero, zero, weightedNegGradOutElem);
              linalg::YieldOp::create(b, loc, gradInElem);
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
        linalg::GenericOp::create(
            rewriter, loc, input.getType(),
            /*ins=*/input,
            /*outs=*/input,
            /*indexingMaps=*/indexingMaps,
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value input = args[0];

              TypedAttr oneAttr = b.getFloatAttr(inputElementType, 1.0);
              Value oneValue = arith::ConstantOp::create(b, loc, oneAttr);

              Value zI;
              if (!handleEps) {
                zI = input;
              } else {
                Value truncEps =
                    arith::TruncFOp::create(b, loc, inputElementType, eps);
                Value oneMinusEps =
                    arith::SubFOp::create(b, loc, oneValue, truncEps);

                Value min =
                    arith::MinimumFOp::create(b, loc, input, oneMinusEps);
                Value clampedInput =
                    arith::MaximumFOp::create(b, loc, min, truncEps);

                zI = clampedInput;
              }

              Value probability = arith::SubFOp::create(b, loc, oneValue, zI);
              Value odds = arith::DivFOp::create(b, loc, zI, probability);
              Value result = math::LogOp::create(b, loc, odds);

              linalg::YieldOp::create(b, loc, result);
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
        dynSizes.push_back(
            tensor::DimOp::create(rewriter, loc, operand, index));
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
        tensor::EmptyOp::create(rewriter, op.getLoc(), resultType, dynSizes);
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{operand, scale, zeropoint},
        ValueRange{empty}, maps, iterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value operand = args[0];
          Value scale = args[1];
          Value zeropoint = args[2];
          if (operandDTy.isUnsignedInteger(8)) {
            operand = arith::ExtUIOp::create(b, loc, b.getI32Type(), operand);
          } else if (operandDTy.isSignedInteger(8)) {
            operand = arith::ExtSIOp::create(b, loc, b.getI32Type(), operand);
          }

          if (zeropointDTy.isUnsignedInteger(8)) {
            zeropoint =
                arith::ExtUIOp::create(b, loc, b.getI32Type(), zeropoint);
          } else if (zeropointDTy.isSignedInteger(8)) {
            zeropoint =
                arith::ExtSIOp::create(b, loc, b.getI32Type(), zeropoint);
          } else if (zeropointDTy.isInteger(64)) {
            zeropoint =
                arith::TruncIOp::create(b, loc, b.getI32Type(), zeropoint);
            op->emitWarning() << "truncated zero point from 64 to 32 bit";
          }

          Value sub = arith::SubIOp::create(rewriter, loc, operand, zeropoint);
          Value fp =
              arith::SIToFPOp::create(rewriter, loc, args[3].getType(), sub);
          Value mul = arith::MulFOp::create(rewriter, loc, fp, scale);
          linalg::YieldOp::create(b, loc, mul);
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
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value zeroFloat = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(floatType, 0.0));
    Value oneFloat = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(floatType, 1.0));
    Value twoFloat = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(floatType, 2.0));
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    Value innerDim0a = tensor::DimOp::create(rewriter, loc, input, 2);
    Value innerDim1a = tensor::DimOp::create(rewriter, loc, input, 3);
    Value innerDim0b =
        arith::SubIOp::create(rewriter, loc, innerDim0a, oneIndex);
    Value innerDim1b =
        arith::SubIOp::create(rewriter, loc, innerDim1a, oneIndex);
    Value innerDim0c =
        arith::IndexCastOp::create(rewriter, loc, int64type, innerDim0b);
    Value innerDim1c =
        arith::IndexCastOp::create(rewriter, loc, int64type, innerDim1b);
    Value innerDim0d =
        arith::SIToFPOp::create(rewriter, loc, floatType, innerDim0c);
    Value innerDim1d =
        arith::SIToFPOp::create(rewriter, loc, floatType, innerDim1c);
    Value innerDim0e =
        arith::DivFOp::create(rewriter, loc, innerDim0d, twoFloat);
    Value innerDim1e =
        arith::DivFOp::create(rewriter, loc, innerDim1d, twoFloat);
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
      Value result = tensor::ExtractOp::create(b, loc, input, index);
      return result;
    };

    auto lambdaLinear = [&](OpBuilder &b, Location loc, Value x, Value y,
                            Value d) -> Value {
      Value dm = arith::SubFOp::create(b, loc, oneFloat, d);
      Value ra = arith::MulFOp::create(b, loc, x, dm);
      Value rb = arith::MulFOp::create(b, loc, y, d);
      Value res = arith::AddFOp::create(b, loc, ra, rb);
      return res;
    };

    auto lambdaNearest = [&](OpBuilder &b, Location loc, Value x, Value y,
                             Value d) -> Value {
      Value halfConst = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(floatType, 0.5));
      Value checkClosest = arith::CmpFOp::create(
          b, loc, arith::CmpFPredicate::OLT, d, halfConst);
      Value res = arith::SelectOp::create(b, loc, checkClosest, x, y);
      return res;
    };

    auto lambdaInterpolate = [&](OpBuilder &b, Location loc, Value iMode,
                                 Value x, Value y, Value d) -> Value {
      Value linear = lambdaLinear(b, loc, x, y, d);
      Value nearest = lambdaNearest(b, loc, x, y, d);
      Value zeroInt =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(int64type, 0));
      Value checkMode = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                              iMode, zeroInt);
      Value res = arith::SelectOp::create(b, loc, checkMode, linear, nearest);
      return res;
    };

    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    Value alignCorners = adaptor.getAlignCorners();
    Value interMode = adaptor.getInterpolationMode();
    SmallVector<Value> dynamicSizes{};
    if (resultType.isDynamicDim(0))
      dynamicSizes.push_back(tensor::DimOp::create(rewriter, loc, input, 0));
    if (resultType.isDynamicDim(1))
      dynamicSizes.push_back(tensor::DimOp::create(rewriter, loc, input, 1));
    if (resultType.isDynamicDim(2))
      dynamicSizes.push_back(tensor::DimOp::create(rewriter, loc, grid, 1));
    if (resultType.isDynamicDim(3))
      dynamicSizes.push_back(tensor::DimOp::create(rewriter, loc, grid, 2));
    tensor::EmptyOp emptyOp =
        tensor::EmptyOp::create(rewriter, loc, resultType, dynamicSizes);
    auto sGrid = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{grid, grid},
        ValueRange(emptyOp), gridMaps, gridIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value gr0 = args[1];
          Value gr1 = args[0];
          Value gr0Half = arith::DivFOp::create(b, loc, gr0, twoFloat);
          Value gr1Half = arith::DivFOp::create(b, loc, gr1, twoFloat);
          Value gr0HalfSelect =
              arith::SelectOp::create(b, loc, alignCorners, zeroFloat, gr0Half);
          Value gr1HalfSelect =
              arith::SelectOp::create(b, loc, alignCorners, zeroFloat, gr1Half);
          Value gplus0 = arith::AddFOp::create(b, loc, gr0, oneFloat);
          Value gplus1 = arith::AddFOp::create(b, loc, gr1, oneFloat);
          Value gPlusMul0 = arith::MulFOp::create(b, loc, gplus0, innerDim0e);
          Value gPlusMul1 = arith::MulFOp::create(b, loc, gplus1, innerDim1e);
          Value result0 =
              arith::AddFOp::create(b, loc, gPlusMul0, gr0HalfSelect);
          Value result1 =
              arith::AddFOp::create(b, loc, gPlusMul1, gr1HalfSelect);
          Value checkLowerBound0 = arith::CmpFOp::create(
              b, loc, arith::CmpFPredicate::OLT, result0, zeroFloat);
          Value checkLowerBound1 = arith::CmpFOp::create(
              b, loc, arith::CmpFPredicate::OLT, result1, zeroFloat);
          Value lowerOrig0 =
              arith::FPToSIOp::create(b, loc, int64type, result0);
          Value lowerOrig1 =
              arith::FPToSIOp::create(b, loc, int64type, result1);
          Value zeroInt =
              arith::ConstantOp::create(b, loc, b.getIntegerAttr(int64type, 0));
          Value oneInt =
              arith::ConstantOp::create(b, loc, b.getIntegerAttr(int64type, 1));
          Value lowerSub0 = arith::SubIOp::create(b, loc, lowerOrig0, oneInt);
          Value lowerSub1 = arith::SubIOp::create(b, loc, lowerOrig1, oneInt);
          Value lower0 = arith::SelectOp::create(b, loc, checkLowerBound0,
                                                 lowerSub0, lowerOrig0);
          Value lower1 = arith::SelectOp::create(b, loc, checkLowerBound1,
                                                 lowerSub1, lowerOrig1);
          Value lowerValid0 = arith::SelectOp::create(b, loc, checkLowerBound0,
                                                      zeroInt, lower0);
          Value lowerValid1 = arith::SelectOp::create(b, loc, checkLowerBound1,
                                                      zeroInt, lower1);
          Value upper0 =
              arith::AddIOp::create(b, loc, int64type, lower0, oneInt);
          Value upper1 =
              arith::AddIOp::create(b, loc, int64type, lower1, oneInt);
          Value notValidUpper0 = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::sgt, upper0, innerDim0c);
          Value notValidUpper1 = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::sgt, upper1, innerDim1c);
          Value upperValid0 =
              arith::SelectOp::create(b, loc, notValidUpper0, lower0, upper0);
          Value upperValid1 =
              arith::SelectOp::create(b, loc, notValidUpper1, lower1, upper1);
          Value lw0 =
              arith::IndexCastOp::create(b, loc, b.getIndexType(), lowerValid0);
          Value lw1 =
              arith::IndexCastOp::create(b, loc, b.getIndexType(), lowerValid1);
          Value up0 =
              arith::IndexCastOp::create(b, loc, b.getIndexType(), upperValid0);
          Value up1 =
              arith::IndexCastOp::create(b, loc, b.getIndexType(), upperValid1);
          Value N = linalg::IndexOp::create(b, loc, 0);
          Value C = linalg::IndexOp::create(b, loc, 1);
          Value result00 = lambdaExtract(b, loc, input, N, C, lw0, lw1);
          Value result00a = arith::SelectOp::create(b, loc, checkLowerBound0,
                                                    zeroFloat, result00);
          Value result00b = arith::SelectOp::create(b, loc, checkLowerBound1,
                                                    zeroFloat, result00a);
          Value result01 = lambdaExtract(b, loc, input, N, C, lw0, up1);
          Value result01a = arith::SelectOp::create(b, loc, notValidUpper1,
                                                    zeroFloat, result01);
          Value result01b = arith::SelectOp::create(b, loc, checkLowerBound0,
                                                    zeroFloat, result01a);
          Value result10 = lambdaExtract(b, loc, input, N, C, up0, lw1);
          Value result10a = arith::SelectOp::create(b, loc, notValidUpper0,
                                                    zeroFloat, result10);
          Value result10b = arith::SelectOp::create(b, loc, checkLowerBound1,
                                                    zeroFloat, result10a);
          Value result11 = lambdaExtract(b, loc, input, N, C, up0, up1);
          Value result11a = arith::SelectOp::create(b, loc, notValidUpper0,
                                                    zeroFloat, result11);
          Value result11b = arith::SelectOp::create(b, loc, notValidUpper1,
                                                    zeroFloat, result11a);
          Value lw0a = arith::SIToFPOp::create(b, loc, floatType, lower0);
          Value lw1a = arith::SIToFPOp::create(b, loc, floatType, lower1);
          Value d1 = arith::SubFOp::create(b, loc, result0, lw0a);
          Value d0 = arith::SubFOp::create(b, loc, result1, lw1a);
          Value resultScaled0 =
              lambdaInterpolate(b, loc, interMode, result00b, result01b, d0);
          Value resultScaled1 =
              lambdaInterpolate(b, loc, interMode, result10b, result11b, d0);
          Value resultScaled = lambdaInterpolate(
              b, loc, interMode, resultScaled0, resultScaled1, d1);
          linalg::YieldOp::create(b, loc, resultScaled);
        });
    rewriter.replaceOp(op, sGrid.getResults());
    return success();
  }
};
} // namespace

static Value nearestInterpolate(OpBuilder &b, Location loc,
                                SmallVector<Value> outputSizes, Value input,
                                SmallVector<Value> inputSizes,
                                SmallVector<Value> scaleValues,
                                std::string coordStr, std::string nearestMode) {

  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();

  SmallVector<Value> indices;
  for (unsigned i = 0; i < inputRank; i++) {
    indices.push_back(linalg::IndexOp::create(b, loc, i));
  }

  for (unsigned i = 2; i < inputRank; i++) {
    Value outIndex = indices[i];

    Value inputSizeFP =
        arith::SIToFPOp::create(b, loc, b.getF32Type(), inputSizes[i - 2]);

    Value outputSizeFP =
        arith::SIToFPOp::create(b, loc, b.getF32Type(), outputSizes[i - 2]);

    // scale = length_resized / length_original
    // x_original = x_resized / scale
    Value scale;
    if (scaleValues.empty())
      scale = arith::DivFOp::create(b, loc, outputSizeFP, inputSizeFP);
    else
      scale = scaleValues[i - 2];

    Value outInt = arith::IndexCastOp::create(b, loc, b.getI64Type(), outIndex);
    Value outFP = arith::SIToFPOp::create(b, loc, b.getF32Type(), outInt);
    Value proj;
    if (coordStr.empty() || coordStr == "_asymmetric") {
      proj = arith::DivFOp::create(b, loc, outFP, scale);
    } else if (coordStr == "_half_pixel") {
      Value cstHalf = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.5));
      Value add = arith::AddFOp::create(b, loc, outFP, cstHalf);
      Value div = arith::DivFOp::create(b, loc, add, scale);
      proj = arith::SubFOp::create(b, loc, div, cstHalf);
    } else {
      llvm_unreachable("Unsupported coordination transformation mode");
    }

    Value nearestFP;
    // get nearest pixel using floor
    if (nearestMode == "floor" || nearestMode == "") {
      nearestFP = math::FloorOp::create(b, loc, proj);
    } else if (nearestMode == "round_prefer_floor") {
      Value cstHalf = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.5));
      Value floor = math::FloorOp::create(b, loc, proj);
      Value ceil = math::CeilOp::create(b, loc, proj);
      Value decimal = arith::SubFOp::create(b, loc, proj, floor);
      Value cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ULE,
                                        decimal, cstHalf);
      nearestFP = arith::SelectOp::create(b, loc, cmp, floor, ceil);
    } else if (nearestMode == "round_prefer_ceil") {
      Value cstHalf = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.5));
      Value cstOne = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1));
      Value floor = math::FloorOp::create(b, loc, proj);
      Value ceil = math::CeilOp::create(b, loc, proj);
      Value decimal = arith::SubFOp::create(b, loc, proj, floor);
      Value cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UGE,
                                        decimal, cstHalf);
      nearestFP = arith::SelectOp::create(b, loc, cmp, ceil, floor);
      Value inputSizeMOne = arith::SubFOp::create(b, loc, inputSizeFP, cstOne);
      // don't extract out of bounds
      nearestFP = arith::MinimumFOp::create(b, loc, nearestFP, inputSizeMOne);
    } else if (nearestMode == "ceil") {
      Value cstOne = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1));
      Value inputSizeMOne = arith::SubFOp::create(b, loc, inputSizeFP, cstOne);
      nearestFP = math::CeilOp::create(b, loc, proj);
      nearestFP = arith::MinimumFOp::create(b, loc, nearestFP, inputSizeMOne);
    } else {
      llvm_unreachable("Unsupported nearest mode");
    }
    Value nearestInt =
        arith::FPToSIOp::create(b, loc, b.getI64Type(), nearestFP);
    Value nearest =
        arith::IndexCastOp::create(b, loc, b.getIndexType(), nearestInt);

    indices[i] = nearest;
  }
  Value retVal = tensor::ExtractOp::create(b, loc, input, indices);
  return retVal;
}

static SmallVector<Value> coordinateTransform(
    OpBuilder &b, Aten__InterpolateSizeListScaleListOp op, Location loc,
    SmallVector<Value> outputSizes, Value input, SmallVector<Value> inputSizes,
    SmallVector<Value> scaleValues, std::string coordStr, bool alignCornersBool,
    SmallVector<Value> indices, bool clip) {

  unsigned dimOffset = 2;
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();

  Value cstOneFloat = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1.0));
  Value cstHalf = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.5));
  Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0));

  SmallVector<Value> proj;
  for (unsigned i = 0; i < inputRank - dimOffset; i++) {
    // length_original
    Value inputFP =
        arith::SIToFPOp::create(b, loc, b.getF32Type(), inputSizes[i]);
    // length_resized
    Value outputSizeFP =
        arith::SIToFPOp::create(b, loc, b.getF32Type(), outputSizes[i]);
    // scale = length_resized/length_original
    Value scale;
    if (alignCornersBool) {
      // x_original = x_resized * (length_original - 1) / (length_resized - 1)
      Value inputSubOne = arith::SubFOp::create(b, loc, inputFP, cstOneFloat);
      Value outputSizeSubOne =
          arith::SubFOp::create(b, loc, outputSizeFP, cstOneFloat);
      Value cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UEQ,
                                        outputSizeSubOne, zero);
      scale = arith::DivFOp::create(b, loc, inputSubOne, outputSizeSubOne);
      scale = arith::SelectOp::create(b, loc, cmp, zero, scale);
      coordStr = "_align_corners";
    } else if (scaleValues.empty())
      scale = arith::DivFOp::create(b, loc, outputSizeFP, inputFP);
    else
      scale = scaleValues[i];
    // y_resized
    Value outInt = arith::IndexCastOp::create(b, loc, b.getI64Type(),
                                              indices[i + dimOffset]);
    Value outFP = arith::SIToFPOp::create(b, loc, b.getF32Type(), outInt);
    Value preClip;
    if (coordStr == "_align_corners") {
      preClip = arith::MulFOp::create(b, loc, outFP, scale);
    }
    if (coordStr == "_asymmetric") {
      preClip = arith::DivFOp::create(b, loc, outFP, scale);
    }
    if (coordStr == "_pytorch_half_pixel" || coordStr == "" ||
        coordStr == "_half_pixel_symmetric") {
      // half-pixel modes
      // y_resized + 0.5
      Value outPlusHalf = arith::AddFOp::create(b, loc, outFP, cstHalf);
      // (y_resized + 0.5) / scale
      Value outDivScale = arith::DivFOp::create(b, loc, outPlusHalf, scale);
      // _ - 0.5
      preClip = arith::SubFOp::create(b, loc, outDivScale, cstHalf);
    }
    // for half_pixel_symmetric, need to compute offset from raw scales
    if (coordStr == "_half_pixel_symmetric" && !scaleValues.empty()) {
      Value outputSizeFromScale = arith::MulFOp::create(b, loc, inputFP, scale);
      Value adjustment =
          arith::DivFOp::create(b, loc, outputSizeFP, outputSizeFromScale);
      Value cstTwo = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(2.0));
      Value center = arith::DivFOp::create(b, loc, inputFP, cstTwo);
      Value oneMAdjustment =
          arith::SubFOp::create(b, loc, cstOneFloat, adjustment);
      Value offset = arith::MulFOp::create(b, loc, center, oneMAdjustment);
      preClip = arith::AddFOp::create(b, loc, offset, preClip);
    }
    // for pytorch half pixel , special case for length_resized == 1:
    if (coordStr == "_pytorch_half_pixel") {
      Value cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UEQ,
                                        outputSizeFP, cstOneFloat);
      preClip = arith::SelectOp::create(b, loc, cmp, zero, preClip);
    }
    if (clip) {
      // preClip is the fp position inside the input image to extract from.
      // clip to [0,inf)
      Value max = arith::MaximumFOp::create(b, loc, preClip, zero);
      Value inputSubOne = arith::SubFOp::create(b, loc, inputFP, cstOneFloat);
      // clip to [0,length_original - 1].
      // proj is properly within the input image.
      proj.push_back(arith::MinimumFOp::create(b, loc, max, inputSubOne));
    } else {
      proj.push_back(preClip);
    }
  }
  return proj;
}

static Value bilinearInterpolate(OpBuilder &b,
                                 Aten__InterpolateSizeListScaleListOp op,
                                 Location loc, SmallVector<Value> outputSizes,
                                 Value input, SmallVector<Value> inputSizes,
                                 SmallVector<Value> scaleValues,
                                 std::string coordStr) {
  unsigned dimOffset = 2;
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();

  Value cstOneFloat = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1.0));

  bool alignCornersBool;
  matchPattern(op.getAlignCorners(), m_TorchConstantBool(&alignCornersBool));

  SmallVector<Value> indices;
  for (unsigned i = 0; i < inputRank; i++) {
    indices.push_back(linalg::IndexOp::create(b, loc, i));
  }

  SmallVector<Value> proj, high, low, highFP, lowFP;
  proj = coordinateTransform(b, op, loc, outputSizes, input, inputSizes,
                             scaleValues, coordStr, alignCornersBool, indices,
                             true);
  for (unsigned i = 0; i < inputRank - dimOffset; i++) {
    // length_original
    Value inputFP =
        arith::SIToFPOp::create(b, loc, b.getF32Type(), inputSizes[i]);
    Value inputSubOne = arith::SubFOp::create(b, loc, inputFP, cstOneFloat);

    // for bilinear interpolation, we look for the nearest indices below and
    // above proj
    lowFP.push_back(math::FloorOp::create(b, loc, proj[i]));
    Value projPlusOne = arith::AddFOp::create(b, loc, cstOneFloat, proj[i]);
    highFP.push_back(math::FloorOp::create(b, loc, projPlusOne));

    Value lowInt = arith::FPToSIOp::create(b, loc, b.getI64Type(), lowFP[i]);
    low.push_back(arith::IndexCastOp::create(b, loc, b.getIndexType(), lowInt));

    // highFP could be out-of-bounds, so make sure to clip it down before
    // extracting. If highFP actually gets clipped here, then high[i] will
    // extract at the last pixel, but will treat it as if it were extracted from
    // one further position when computing the interpolation weights.
    Value highExtract =
        arith::MinimumFOp::create(b, loc, projPlusOne, inputSubOne);
    highExtract = arith::FPToSIOp::create(b, loc, b.getI64Type(), highExtract);
    high.push_back(
        arith::IndexCastOp::create(b, loc, b.getIndexType(), highExtract));
  }

  indices[dimOffset] = low[0];
  indices[dimOffset + 1] = low[1];
  Value p00 = tensor::ExtractOp::create(b, loc, input, indices);

  indices[dimOffset] = low[0];
  indices[dimOffset + 1] = high[1];
  Value p01 = tensor::ExtractOp::create(b, loc, input, indices);

  indices[dimOffset] = high[0];
  indices[dimOffset + 1] = low[1];
  Value p10 = tensor::ExtractOp::create(b, loc, input, indices);

  indices[dimOffset] = high[0];
  indices[dimOffset + 1] = high[1];
  Value p11 = tensor::ExtractOp::create(b, loc, input, indices);

  // Let Aij := area rect((yProj,xProj) <-> (y_i*,x_j*)),
  // where i* = i+1 mod 2 and x_0 = xLow, x_1 = xHigh etc.
  // We interpolate via the weighted average of pij by weights Aij
  // the formula is retval = Sum(pij*Aij for i and j in range(2))
  // Note: we do not need to divide by total rect area == 1

  // lengths : Aij == dyi*dxj
  Value dy0 = arith::SubFOp::create(b, loc, highFP[0], proj[0]);
  Value dy1 = arith::SubFOp::create(b, loc, proj[0], lowFP[0]);
  Value dx0 = arith::SubFOp::create(b, loc, highFP[1], proj[1]);
  Value dx1 = arith::SubFOp::create(b, loc, proj[1], lowFP[1]);

  // left = A00*p00 + A01*p01 = dy0(dx0p00 + dx1p01)
  Value dx0p00 = arith::MulFOp::create(b, loc, dx0, p00);
  Value dx1p01 = arith::MulFOp::create(b, loc, dx1, p01);
  Value sum = arith::AddFOp::create(b, loc, dx0p00, dx1p01);
  Value left = arith::MulFOp::create(b, loc, dy0, sum);
  // right = A10*p10 + A11*p11 = dy1(dx0p10 + dx1p11)
  Value dx0p10 = arith::MulFOp::create(b, loc, dx0, p10);
  Value dx1p11 = arith::MulFOp::create(b, loc, dx1, p11);
  sum = arith::AddFOp::create(b, loc, dx0p10, dx1p11);
  Value right = arith::MulFOp::create(b, loc, dy1, sum);

  return arith::AddFOp::create(b, loc, left, right);
}

static Value bicubicInterpolate(OpBuilder &b,
                                Aten__InterpolateSizeListScaleListOp op,
                                Location loc, SmallVector<Value> outputSizes,
                                Value input, SmallVector<Value> inputSizes,
                                SmallVector<Value> scaleValues,
                                std::string coordStr) {
  unsigned dimOffset = 2;
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();

  Value inputFPH =
      arith::SIToFPOp::create(b, loc, b.getF32Type(), inputSizes[0]);
  Value inputFPW =
      arith::SIToFPOp::create(b, loc, b.getF32Type(), inputSizes[1]);

  Value a = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(-0.75));
  Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0));
  Value cstOneFloat = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1.0));
  Value cstTwoFloat = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(2.0));
  Value cstThreeFloat =
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(3.0));
  Value cstFourFloat =
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(4.0));
  Value cstFiveFloat =
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(5.0));
  Value cstEightFloat =
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(8.0));

  // (a+2)|x|^3 - (a+3)|x|^2 + 1 for xDistance (|x| <= 1)
  auto WeightLessThanEqualOne = [&](Value xDistance) -> Value {
    Value xDistanceSquared =
        arith::MulFOp::create(b, loc, xDistance, xDistance);
    Value xDistanceCubed =
        arith::MulFOp::create(b, loc, xDistanceSquared, xDistance);

    Value lessEqualOne = arith::AddFOp::create(b, loc, a, cstTwoFloat);
    lessEqualOne = arith::MulFOp::create(b, loc, xDistanceCubed, lessEqualOne);
    Value aPlusThree = arith::AddFOp::create(b, loc, a, cstThreeFloat);
    aPlusThree = arith::MulFOp::create(b, loc, xDistanceSquared, aPlusThree);
    lessEqualOne = arith::SubFOp::create(b, loc, lessEqualOne, aPlusThree);
    lessEqualOne = arith::AddFOp::create(b, loc, lessEqualOne, cstOneFloat);

    return lessEqualOne;
  };

  // a|x|^3 - 5a|x|^2 + 8a|x| - 4a for xDistance (1 < |x| < 2)
  auto WeightLessThanTwo = [&](Value xDistance) -> Value {
    Value xDistanceSquared =
        arith::MulFOp::create(b, loc, xDistance, xDistance);
    Value xDistanceCubed =
        arith::MulFOp::create(b, loc, xDistanceSquared, xDistance);
    // a|x|^3
    Value lessThanTwo = arith::MulFOp::create(b, loc, xDistanceCubed, a);

    Value fiveA = arith::MulFOp::create(b, loc, xDistanceSquared, a);
    fiveA = arith::MulFOp::create(b, loc, fiveA, cstFiveFloat);
    // a|x|^3 - 5a|x|^2
    lessThanTwo = arith::SubFOp::create(b, loc, lessThanTwo, fiveA);

    Value eightA = arith::MulFOp::create(b, loc, a, xDistance);
    eightA = arith::MulFOp::create(b, loc, eightA, cstEightFloat);
    // a|x|^3 - 5a|x|^2 + 8a|x|
    lessThanTwo = arith::AddFOp::create(b, loc, eightA, lessThanTwo);

    Value fourA = arith::MulFOp::create(b, loc, a, cstFourFloat);
    // a|x|^3 - 5a|x|^2 + 8a|x| - 4a
    lessThanTwo = arith::SubFOp::create(b, loc, lessThanTwo, fourA);
    return lessThanTwo;
  };

  bool alignCornersBool;
  matchPattern(op.getAlignCorners(), m_TorchConstantBool(&alignCornersBool));

  SmallVector<Value> indices;
  for (unsigned i = 0; i < inputRank; i++) {
    indices.push_back(linalg::IndexOp::create(b, loc, i));
  }

  SmallVector<Value> proj;

  proj = coordinateTransform(b, op, loc, outputSizes, input, inputSizes,
                             scaleValues, coordStr, alignCornersBool, indices,
                             false);

  // get the nearest neighbors of proj
  Value x1 = math::CeilOp::create(b, loc, proj[1]);
  Value x_1 = arith::SubFOp::create(b, loc, x1, cstOneFloat);
  Value x_2 = arith::SubFOp::create(b, loc, x_1, cstOneFloat);
  Value x2 = arith::AddFOp::create(b, loc, x1, cstOneFloat);

  Value y1 = math::CeilOp::create(b, loc, proj[0]);
  Value y_1 = arith::SubFOp::create(b, loc, y1, cstOneFloat);
  Value y_2 = arith::SubFOp::create(b, loc, y_1, cstOneFloat);
  Value y2 = arith::AddFOp::create(b, loc, y1, cstOneFloat);

  // calculate the distance of nearest neighbors x and y to proj
  Value y2Distance = arith::SubFOp::create(b, loc, proj[0], y2);
  y2Distance = math::AbsFOp::create(b, loc, y2Distance);
  Value y1Distance = arith::SubFOp::create(b, loc, proj[0], y1);
  y1Distance = math::AbsFOp::create(b, loc, y1Distance);
  Value y_1Distance = arith::SubFOp::create(b, loc, proj[0], y_1);
  y_1Distance = math::AbsFOp::create(b, loc, y_1Distance);
  Value y_2Distance = arith::SubFOp::create(b, loc, proj[0], y_2);
  y_2Distance = math::AbsFOp::create(b, loc, y_2Distance);

  Value x2Distance = arith::SubFOp::create(b, loc, proj[1], x2);
  x2Distance = math::AbsFOp::create(b, loc, x2Distance);
  Value x1Distance = arith::SubFOp::create(b, loc, proj[1], x1);
  x1Distance = math::AbsFOp::create(b, loc, x1Distance);
  Value x_1Distance = arith::SubFOp::create(b, loc, proj[1], x_1);
  x_1Distance = math::AbsFOp::create(b, loc, x_1Distance);
  Value x_2Distance = arith::SubFOp::create(b, loc, proj[1], x_2);
  x_2Distance = math::AbsFOp::create(b, loc, x_2Distance);

  SmallVector<Value> y{y_2, y_1, y1, y2};
  SmallVector<Value> x{x_2, x_1, x1, x2};

  SmallVector<Value> wys{
      WeightLessThanTwo(y_2Distance), WeightLessThanEqualOne(y_1Distance),
      WeightLessThanEqualOne(y1Distance), WeightLessThanTwo(y2Distance)};
  SmallVector<Value> wxs{
      WeightLessThanTwo(x_2Distance), WeightLessThanEqualOne(x_1Distance),
      WeightLessThanEqualOne(x1Distance), WeightLessThanTwo(x2Distance)};

  // clip the nearest neighbors points to inside the original image
  for (int k = 0; k < 4; k++) {
    Value yClipped = arith::MaximumFOp::create(b, loc, y[k], zero);
    Value inputHSubOne = arith::SubFOp::create(b, loc, inputFPH, cstOneFloat);
    yClipped = arith::MinimumFOp::create(b, loc, yClipped, inputHSubOne);
    Value yInt = arith::FPToSIOp::create(b, loc, b.getI64Type(), yClipped);
    y[k] = arith::IndexCastOp::create(b, loc, b.getIndexType(), yInt);

    Value xClipped = arith::MaximumFOp::create(b, loc, x[k], zero);
    Value inputWSubOne = arith::SubFOp::create(b, loc, inputFPW, cstOneFloat);
    xClipped = arith::MinimumFOp::create(b, loc, xClipped, inputWSubOne);
    Value xInt = arith::FPToSIOp::create(b, loc, b.getI64Type(), xClipped);
    x[k] = arith::IndexCastOp::create(b, loc, b.getIndexType(), xInt);
  }
  // 1. Compute x_original and y_original (proj)
  // 2. Compute nearest x and y neighbors
  // 3. Compute Wx Wy
  // 4. Extract inputs at nearest neighbors (inputExtracts)
  // 5. Compute weighted sum (yield this)

  // 4 nearest x neighbors : [x_2, x_1, x1, x2] of x_original
  // 4 nearest y neighbors : [y_2, y_1, y1, y2] of y_original
  // Sum_x is over 4 nearest x neighbors (similar for Sum_y)
  // f(x_original, y_original) = Sum_y Sum_x W(x_original - x)*input[x,y]
  //                     * W(y_original - y)
  Value fxy = zero;

  for (int j = 0; j < 4; j++) {
    Value wy = wys[j];
    Value xInterpy = zero;

    indices[dimOffset] = y[j];

    for (int i = 0; i < 4; i++) {
      Value wx = wxs[i];

      indices[dimOffset + 1] = x[i];

      Value p = tensor::ExtractOp::create(b, loc, input, indices);

      Value wxp = arith::MulFOp::create(b, loc, wx, p);
      xInterpy = arith::AddFOp::create(b, loc, xInterpy, wxp);
    }
    Value wyXInterpy = arith::MulFOp::create(b, loc, wy, xInterpy);
    fxy = arith::AddFOp::create(b, loc, fxy, wyXInterpy);
  }

  return fxy;
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
    if (mode.substr(0, 8) != "bilinear" && mode.substr(0, 7) != "nearest" &&
        mode.substr(0, 5) != "cubic") {
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
      inputSizes.push_back(arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIntegerType(64), inputSize));
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
        Value inputSizeFP = arith::SIToFPOp::create(
            rewriter, loc, rewriter.getF32Type(), inputSizes[i]);
        ScaleFactorFloatValues[i] = arith::TruncFOp::create(
            rewriter, loc, inputSizeFP.getType(), ScaleFactorFloatValues[i]);
        Value outputSize = arith::MulFOp::create(rewriter, loc, inputSizeFP,
                                                 ScaleFactorFloatValues[i]);
        outputSize = math::FloorOp::create(rewriter, loc, outputSize);
        outputSize = arith::FPToSIOp::create(rewriter, loc,
                                             rewriter.getI64Type(), outputSize);
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

    Value outTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(dims), inputType.getElementType());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(inputRank);
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
    Value finalRes =
        linalg::GenericOp::create(
            rewriter, loc, outTensor.getType(), ValueRange{}, outTensor,
            /*indexingMaps=*/idMap,
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value retVal;
              if (mode.substr(0, 7) == "nearest") {
                std::string coordTfMode = mode.substr(7, mode.find(",") - 7);
                std::string nearestMode = (mode.find(",") == std::string::npos)
                                              ? ""
                                              : mode.substr(mode.find(",") + 1);
                retVal = nearestInterpolate(b, loc, outputSizeIntValues, input,
                                            inputSizes, ScaleFactorFloatValues,
                                            coordTfMode, nearestMode);
              } else if (mode.substr(0, 8) == "bilinear") {
                retVal = bilinearInterpolate(
                    b, op, loc, outputSizeIntValues, input, inputSizes,
                    ScaleFactorFloatValues, mode.substr(8));
              } else if (mode.substr(0, 5) == "cubic") {

                retVal = bicubicInterpolate(
                    b, op, loc, outputSizeIntValues, input, inputSizes,
                    ScaleFactorFloatValues, mode.substr(5));
              }
              linalg::YieldOp::create(b, loc, retVal);
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
    Value cstZero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value cstOne = arith::ConstantIndexOp::create(rewriter, loc, 1);
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
    Value matDimMinusOne = arith::SubIOp::create(rewriter, loc, matDim, cstOne);
    ArrayRef<Value> sliceSizes(inputSizes.begin(), inputSizes.end() - 1);
    // initialize a tensor to store the diagonal elements found during row
    // reduction
    Value initDiags = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(sliceSizes), elemTy);
    // loop over each pivot row in A. Get the diagonal, then reduce the
    // subdiagonal Don't perform the loop on the last row since no further
    // reduction is needed.
    auto rowReductionLoop = scf::ForOp::create(
        rewriter, loc, /*start=*/cstZero, /*end=*/matDimMinusOne,
        /*step=*/cstOne,
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
          Value pivot = tensor::ExtractSliceOp::create(b, loc, sliceTy, vals[0],
                                                       offsets, sizes, strides);
          // extract diagonal elements and insert them into vals[1]
          offsets.back() = row;
          sizes.back() = cstOneFold;
          // offsets = [0, row, row], sizes = [C, 1, 1] -> diag(row,row)
          Value diag = tensor::ExtractSliceOp::create(b, loc, diagTy, vals[0],
                                                      offsets, sizes, strides);

          Value diagCollapse = tensor::CollapseShapeOp::create(
              b, loc, diagCollapseTy, diag, diagReassociations);

          SmallVector<OpFoldResult> diagOffsets(inputRank - 1, cstZeroFold);
          diagOffsets.back() = row;
          SmallVector<OpFoldResult> diagStrides(inputRank - 1, cstOneFold);
          SmallVector<OpFoldResult> diagSizes = getAsOpFoldResult(sliceSizes);
          diagSizes.back() = cstOneFold;
          // offsets = [0, row], sizes = [C, 1] insert to [C,N]
          Value updatedDiags = tensor::InsertSliceOp::create(
              b, loc, diagCollapse, vals[1], diagOffsets, diagSizes,
              diagStrides);
          // the subpivot matrix column size, as a Value, is matDim - row -
          // cstOne. This can't be statically converted to an int64_t, since row
          // is the loop index, so this is left as a dynamic dim.
          SmallVector<int64_t> subPivotShape(inputType.getShape());
          subPivotShape[inputRank - 2] = ShapedType::kDynamic;
          ArrayRef<int64_t> subDiagShape(subPivotShape.begin(),
                                         subPivotShape.end() - 1);
          auto subPivotTy = RankedTensorType::get(subPivotShape, elemTy);
          auto subDiagTy = RankedTensorType::get(subDiagShape, elemTy);
          Value rowPlusOne = arith::AddIOp::create(b, loc, row, cstOne);
          offsets[inputRank - 2] = getAsOpFoldResult(rowPlusOne);
          sizes[inputRank - 2] = getAsOpFoldResult(
              arith::SubIOp::create(b, loc, matDim, rowPlusOne));
          // offsets = [0, row + 1, row], sizes = [C, N - row - 1, 1] -> A_j,row
          // with j > row
          Value subDiag = tensor::ExtractSliceOp::create(
              b, loc, subDiagTy, vals[0], offsets, sizes, strides);
          offsets.back() = cstZeroFold;
          sizes.back() = getAsOpFoldResult(matDim);
          // offsets = [0, row + 1, 0], sizes = [C, N - row - 1, N] -> elements
          // below pivot row
          Value subPivot = tensor::ExtractSliceOp::create(
              b, loc, subPivotTy, vals[0], offsets, sizes, strides);
          Value initResult = tensor::EmptyOp::create(b, loc, sizes, elemTy);
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
              linalg::GenericOp::create(
                  b, loc, subPivotTy,
                  ValueRange{pivot, diag, subPivot, subDiag}, initResult,
                  indexingMaps, iteratorTypes,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    // for d0 in batches, d1 in subpivotrows, d2 in columns
                    // let i represent the pivot row index (scf loop index)
                    Value pivotd0d2 = args[0];
                    Value diagd0 = args[1];
                    Value subPivotd0d1d2 = args[2];
                    Value subDiagd0d1 = args[3];
                    // coeff = A_d1,i / A_i,i
                    Value coeff =
                        arith::DivFOp::create(b, loc, subDiagd0d1, diagd0);
                    auto cmp = arith::CmpFOp::create(
                        b, loc, arith::CmpFPredicate::ONE, diagd0, cstZeroF);
                    cf::AssertOp::create(
                        b, loc, cmp,
                        b.getStringAttr("unimplemented: determinants requiring "
                                        "permutations and singular matrices"));
                    // coeff*A_i,d2
                    Value scaledPivotValue =
                        arith::MulFOp::create(b, loc, coeff, pivotd0d2);
                    // result = A_d1,d2 - (A_d1,i/A_i,i)*A_i,d2
                    // so that when d2 = i, A_d1,i - (A_d1,i/A_i,i) * A_i,i = 0
                    Value result = arith::SubFOp::create(b, loc, subPivotd0d1d2,
                                                         scaledPivotValue);
                    linalg::YieldOp::create(b, loc, result);
                  })
                  .getResult(0);
          Value rowReductionResult = tensor::InsertSliceOp::create(
              b, loc, reducedSubPivot, vals[0], offsets, sizes, strides);
          scf::YieldOp::create(b, loc,
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
    Value lastDiag = tensor::ExtractSliceOp::create(
        rewriter, loc, diagTy, rowReductionLoop.getResult(0), offsets, sizes,
        strides);
    offsets.pop_back();
    strides.pop_back();
    sizes.pop_back();

    lastDiag = tensor::CollapseShapeOp::create(rewriter, loc, diagCollapseTy,
                                               lastDiag, diagReassociations);

    Value allDiags = tensor::InsertSliceOp::create(
        rewriter, loc, lastDiag, allDiagsExceptLast, offsets, sizes, strides);
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
        linalg::GenericOp::create(
            rewriter, loc, initDet.getType(), ValueRange{allDiags}, initDet,
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value prod = arith::MulFOp::create(b, loc, args[0], args[1]);
              linalg::YieldOp::create(b, loc, prod);
            })
            .getResult(0);
    Type newResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (isBatched) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
                                                  determinant);
      return success();
    }

    determinant = tensor::CollapseShapeOp::create(
        rewriter, loc, newResultType, determinant,
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
      auto currentDimSize = tensor::DimOp::create(rewriter, loc, absTensor, i);
      resultShape.push_back(currentDimSize);
    }

    Value outTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(resultShape), elementType);

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
        linalg::GenericOp::create(
            rewriter, loc, outTensor.getType(),
            ValueRange{absTensor, angleTensor}, outTensor, indexingMaps,
            iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              // out = abs⋅cos(angle) + abs⋅sin(angle)⋅j
              Value abs = args[0];
              Value angle = args[1];
              Value realVal = math::CosOp::create(b, loc, angle);
              Value imagVal = math::SinOp::create(b, loc, angle);
              realVal = arith::MulFOp::create(b, loc, abs, realVal);
              imagVal = arith::MulFOp::create(b, loc, abs, imagVal);
              Value complexVal = complex::CreateOp::create(b, loc, elementType,
                                                           realVal, imagVal);
              linalg::YieldOp::create(b, loc, complexVal);
            })
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, complexVar);
    return success();
  }
};
} // namespace

namespace {
class ConvertSymConstrainRangeOp
    : public OpConversionPattern<AtenSymConstrainRangeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSymConstrainRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto loc = op.getLoc();
    auto min = op.getMin();
    auto max = op.getMax();

    int64_t minValue = std::numeric_limits<int64_t>::min();
    int64_t maxValue = std::numeric_limits<int64_t>::max();

    Type operandType = getTypeConverter()->convertType(op.getSize().getType());

    if (!isa<Torch::NoneType>(min.getType()))
      if (!matchPattern(min, m_TorchConstantInt(&minValue)))
        return rewriter.notifyMatchFailure(
            op, "Expected min value to be constant integer");

    if (!isa<Torch::NoneType>(max.getType()))
      if (!matchPattern(max, m_TorchConstantInt(&maxValue)))
        return rewriter.notifyMatchFailure(
            op, "Expected max value to be constant integer");

    if (maxValue < minValue) {
      std::string errorMsg =
          "Max must be greater than or equal to min, got min = " +
          std::to_string(minValue) + ", max = " + std::to_string(maxValue);
      return op.emitError(errorMsg);
    }

    min = getConstant(rewriter, loc, minValue, operandType);
    max = getConstant(rewriter, loc, maxValue, operandType);

    // Check min <= size <= max

    // FIXME:: Skip the below checks if constraint ops are already inserted as
    // part of symbol expr evaluation
    auto checkMin = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, min, adaptor.getSize());
    auto checkMax = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, adaptor.getSize(), max);
    auto compareVal = arith::AndIOp::create(rewriter, loc, checkMin, checkMax);

    std::string assertMessage = "Size constraint failed. Expected range: [" +
                                std::to_string(minValue) + ", " +
                                std::to_string(maxValue) + "]";
    cf::AssertOp::create(rewriter, loc, compareVal,
                         rewriter.getStringAttr(assertMessage));

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class ConvertOnnxVariantRotaryEmbeddingOp
    : public OpConversionPattern<OnnxVariantRotaryEmbeddingOp> {
public:
  using OpConversionPattern::OpConversionPattern;

private:
  struct RotaryParameters {
    int64_t batchSize;      // May be kDynamic
    int64_t sequenceLength; // May be kDynamic
    int64_t hiddenSize;
    int64_t headSize;
    int64_t rotaryEmbeddingDim;
    int64_t numHeads;
    int64_t maxSequenceLength;
    int64_t inputRank; // 3 or 4
  };

  static LogicalResult checkInputs(OnnxVariantRotaryEmbeddingOp op, Value input,
                                   Value positionIds, Value cosCache,
                                   Value sinCache, int64_t numHeads,
                                   int64_t rotaryEmbeddingDim,
                                   RotaryParameters &parameters,
                                   ConversionPatternRewriter &rewriter) {
    //    input        : (batch_size, num_heads, sequence_length, hidden_size)
    //    position_ids : (batch_size, sequence_length)
    //    cos_cache    : (max_sequence_length, head_size / 2) or
    //                   (max_sequence_length, rotary_embedding_dim / 2)
    //    sin_cache    : (max_sequence_length, head_size / 2) or
    //                   (max_sequence_length, rotary_embedding_dim / 2)

    // Check input - support both rank 3 and rank 4
    // Rank 3: (batch_size, sequence_length, hidden_size)
    // Rank 4: (batch_size, num_heads, sequence_length, head_size)
    // Dynamic batch/seq dimensions are allowed.

    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> inputShape{inputType.getShape()};
    int64_t inputRank = inputShape.size();

    if (inputRank != 3 && inputRank != 4)
      return rewriter.notifyMatchFailure(
          op, "input is expected to have rank 3 or 4");

    // For rank 3: hidden_size (dim 2) must be static for reshape computation
    // For rank 4: head_size (dim 3) must be static
    if (inputRank == 3 && inputShape[2] == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "hidden_size (dim 2) must be static for rank 3 input");
    if (inputRank == 4 && inputShape[3] == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "head_size (dim 3) must be static for rank 4 input");

    // Check position_ids - allow dynamic dims, just check rank
    RankedTensorType positionIdsType =
        cast<RankedTensorType>(positionIds.getType());

    SmallVector<int64_t> positionIdsShape{positionIdsType.getShape()};
    if (positionIdsShape.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "position_ids is expected to have rank 2");

    // Check cos_cache and sin_cache
    RankedTensorType cosCacheType = cast<RankedTensorType>(cosCache.getType());
    if (!cosCacheType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: expected cos_cache to have static shape");

    SmallVector<int64_t> cosCacheShape{cosCacheType.getShape()};
    if (cosCacheShape.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "cos_cache is expected to have rank 2");

    RankedTensorType sinCacheType = cast<RankedTensorType>(sinCache.getType());
    if (!sinCacheType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: expected sin_cache to have static shape");

    SmallVector<int64_t> sinCacheShape{sinCacheType.getShape()};
    if (sinCacheShape.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "sin_cache is expected to have rank 2");

    if (!llvm::equal(cosCacheShape, sinCacheShape))
      return rewriter.notifyMatchFailure(
          op,
          "Inputs cos_cache and sin_cache are expected to have the same shape");

    // Check for element types.
    bool areAllElementTypesEqual = llvm::all_equal(
        {inputType.getElementType(), cosCacheType.getElementType(),
         sinCacheType.getElementType()});
    if (!(areAllElementTypesEqual &&
          isa<mlir::FloatType>(inputType.getElementType())))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: expected input, cos_cache, and sin_cache to be "
              "of floating-point type");

    if (!isa<mlir::IntegerType>(positionIdsType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: expected position_ids to be of integer type");

    // Check num_heads and rotary_embedding_dim
    if (rotaryEmbeddingDim > 0 && numHeads == 0)
      return rewriter.notifyMatchFailure(
          op,
          "num_heads must be non-zero if rotary_embedding_dim is specified");

    // Compute parameters - headSize always comes from cos_cache (static)
    int64_t maxSequenceLength = cosCacheShape[0];
    int64_t headSize = cosCacheShape[1] * 2;

    // hiddenSize computation based on rank
    int64_t hiddenSize;
    if (inputRank == 3) {
      hiddenSize = inputShape[2]; // Must be static (checked above)
    } else {
      // For rank 4, hidden = num_heads * head_size
      if (inputShape[1] != ShapedType::kDynamic &&
          inputShape[3] != ShapedType::kDynamic) {
        hiddenSize = inputShape[1] * inputShape[3];
      } else if (numHeads > 0) {
        hiddenSize = numHeads * headSize;
      } else {
        return rewriter.notifyMatchFailure(
            op, "num_heads attribute required when input dims are dynamic");
      }
    }

    // Override headSize if rotaryEmbeddingDim is specified
    if (rotaryEmbeddingDim > 0 && numHeads > 0) {
      headSize = hiddenSize / numHeads;
    }

    if (rotaryEmbeddingDim > 0 && rotaryEmbeddingDim > headSize)
      return rewriter.notifyMatchFailure(
          op, "rotary_embedding_dim must be less than or equal to head_size");

    // numHeads computation
    int64_t computedNumHeads;
    if (numHeads > 0) {
      computedNumHeads = numHeads;
    } else if (hiddenSize != ShapedType::kDynamic) {
      computedNumHeads = hiddenSize / headSize;
    } else {
      return rewriter.notifyMatchFailure(
          op, "num_heads attribute required when hidden_size is dynamic");
    }

    // Check cos_cache input shapes
    if (cosCacheShape[1] != (headSize / 2) &&
        (rotaryEmbeddingDim > 0 &&
         (cosCacheShape[1] != (rotaryEmbeddingDim / 2))))
      return rewriter.notifyMatchFailure(
          op, "cos_cache shape dimension 1 should be equal to head_size / 2 "
              "or rotary_embedding_dim / 2");

    // batch/seq may be dynamic - store the static values or kDynamic
    int64_t batchSize = inputShape[0];
    int64_t sequenceLength = inputRank == 3 ? inputShape[1] : inputShape[2];

    parameters.batchSize = batchSize;
    parameters.sequenceLength = sequenceLength;
    parameters.hiddenSize = hiddenSize;
    parameters.headSize = headSize;
    parameters.numHeads = computedNumHeads;
    parameters.maxSequenceLength = maxSequenceLength;
    parameters.rotaryEmbeddingDim =
        rotaryEmbeddingDim > 0 ? rotaryEmbeddingDim : headSize;
    parameters.inputRank = inputRank;

    return success();
  }

  LogicalResult
  matchAndRewrite(OnnxVariantRotaryEmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();
    MLIRContext *context = rewriter.getContext();

    Value input = adaptor.getInput();
    Value positionIds = adaptor.getPositionIds();
    Value cosCache = adaptor.getCosCache();
    Value sinCache = adaptor.getSinCache();

    RankedTensorType resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    RankedTensorType inputType = cast<RankedTensorType>(input.getType());

    if (inputType.getElementType() != resultType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "Expected input and result to have the same element type");

    int64_t interleaved, isPackedBatching, numHeads, rotaryEmbeddingDim;
    if (!matchPattern(op.getInterleaved(), m_TorchConstantInt(&interleaved)))
      return rewriter.notifyMatchFailure(
          op, "interleaved must be constant integer");
    if (!matchPattern(op.getIsPackedBatching(),
                      m_TorchConstantInt(&isPackedBatching)))
      return rewriter.notifyMatchFailure(
          op, "is_packed_batching must be constant integer");
    if (!matchPattern(op.getNumHeads(), m_TorchConstantInt(&numHeads)))
      return rewriter.notifyMatchFailure(op,
                                         "num_heads must be constant integer");
    if (!matchPattern(op.getRotaryEmbeddingDim(),
                      m_TorchConstantInt(&rotaryEmbeddingDim)))
      return rewriter.notifyMatchFailure(
          op, "rotary_embedding_dim must be constant integer");

    double scale;
    if (!matchPattern(op.getScale(), m_TorchConstantFloat(&scale)))
      return rewriter.notifyMatchFailure(op, "scale must be constant float");

    // The `checkInputs` function verifies the validity of all the inputs for
    // the cases supported by this lowering and also computes the required
    // rotary parameters.
    RotaryParameters parameters;
    if (failed(checkInputs(op, input, positionIds, cosCache, sinCache, numHeads,
                           rotaryEmbeddingDim, parameters, rewriter)))
      return rewriter.notifyMatchFailure(
          op, "Failed to satisfy the constraints in checkInputs function");

    rotaryEmbeddingDim = parameters.rotaryEmbeddingDim;
    int64_t halfRotaryEmbDim = rotaryEmbeddingDim / 2;

    auto elementType = inputType.getElementType();
    SmallVector<int64_t> inputShape{inputType.getShape()};
    bool needsReshape = (parameters.inputRank == 3);

    // Capture original input dims for reshaping back later (only needed for
    // rank-3 inputs that require reshape)
    Value origBatchDim, origSeqDim, origHiddenDim;

    // processedInput will always be rank 4 for the linalg.generic
    Value processedInput = input;
    RankedTensorType processedInputType = inputType;
    if (needsReshape) {
      origBatchDim = getDimOp(rewriter, loc, input, 0);
      origSeqDim = getDimOp(rewriter, loc, input, 1);
      origHiddenDim = getDimOp(rewriter, loc, input, 2);

      // Result type: preserve dynamic markers for batch/seq
      auto reshapedType =
          RankedTensorType::get({inputShape[0], parameters.numHeads,
                                 inputShape[1], parameters.headSize},
                                elementType);

      // Build i64 shape tensor for tensor.reshape
      auto i64Type = rewriter.getI64Type();
      Value numHeadsVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(parameters.numHeads));
      Value headSizeVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(parameters.headSize));
      SmallVector<Value> reshapedDimVals = {
          arith::IndexCastOp::create(rewriter, loc, i64Type, origBatchDim),
          arith::IndexCastOp::create(rewriter, loc, i64Type, numHeadsVal),
          arith::IndexCastOp::create(rewriter, loc, i64Type, origSeqDim),
          arith::IndexCastOp::create(rewriter, loc, i64Type, headSizeVal)};
      auto shapeType =
          RankedTensorType::get({static_cast<int64_t>(reshapedDimVals.size())},
                                rewriter.getI64Type());
      Value shapeValue = tensor::FromElementsOp::create(
          rewriter, loc, shapeType, reshapedDimVals);
      processedInput = tensor::ReshapeOp::create(rewriter, loc, reshapedType,
                                                 input, shapeValue);
      processedInputType = reshapedType;
    }

    // Build result shape for the rank-4 linalg.generic output
    // processedInput is always rank 4 at this point
    SmallVector<OpFoldResult> resultDimsOFR =
        tensor::getMixedSizes(rewriter, loc, processedInput);

    // Create output tensor with mixed static/dynamic dims
    Value outTensor =
        tensor::EmptyOp::create(rewriter, loc, resultDimsOFR, elementType);
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(elementType));
    outTensor =
        linalg::FillOp::create(rewriter, loc, zero, outTensor).getResult(0);

    Value cstFloatOne = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elementType, 1.0));
    Value cstFloatMinusOne = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elementType, -1.0));
    Value cstIndexTwo =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(2));
    Value cstIndexOne =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
    Value cstRotaryEmbDim = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(rotaryEmbeddingDim));
    Value cstHalfRotaryEmbDim = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(halfRotaryEmbDim));

    // Always rank 4 after reshape
    unsigned processedRank = 4;
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(processedRank, context);
    // position_ids maps to (batch, seq) which is dims (0, 2) for rank 4
    AffineMap positionIdsMap = identityMap.getSubMap({0, 2});

    SmallVector<AffineMap> indexingMaps{identityMap, positionIdsMap,
                                        /*outputMap=*/identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        processedRank, utils::IteratorType::parallel);

    auto rotaryEmbedding =
        linalg::GenericOp::create(
            rewriter, loc, outTensor.getType(),
            ValueRange{processedInput, positionIds}, outTensor, indexingMaps,
            iteratorTypes,
            [&](OpBuilder &builder, Location loc, ValueRange args) {
              // This linalg.generic will be iterating over the 4 dimensions
              // of the input "b, n, s, h", respectively.
              //
              // if (interleaved):
              //     cache_idx = (h / 2) % half_rotary_emb_dim
              //     sign = h & 1
              //     j = sign ? h - 1: h + 1
              // else:
              //     cache_idx = h % half_rotary_emb_dim
              //     sign = (h >= rotary_emb_dim)
              //     j = (h + half_rotary_emb_dim) % rotary_emb_dim
              //
              // orig_input = input[b][n][s][h]
              // rotated_input = input[b][n][s][j]
              // position_id = position_ids[b][s]
              // cos_emb = cos_cache[position_id][cache_idx]
              // sin_emb = sin_cache[position_id][cache_idx]
              // out[b][n][s][h] = orig_input * cos_emb
              //                             +
              //                  (rotated_input * sin_emb) * sign

              Value b = linalg::IndexOp::create(builder, loc, 0);
              Value n = linalg::IndexOp::create(builder, loc, 1);
              Value s = linalg::IndexOp::create(builder, loc, 2);
              Value h = linalg::IndexOp::create(builder, loc, 3);

              Value cacheIdx, sign, rotatedInputLastIdx;
              if (interleaved) {
                cacheIdx = arith::DivSIOp::create(builder, loc, h, cstIndexTwo);
                cacheIdx = arith::RemSIOp::create(builder, loc, cacheIdx,
                                                  cstHalfRotaryEmbDim);
                sign = arith::AndIOp::create(builder, loc, h, cstIndexOne);
                // Converting sign value from index type to bool type.
                sign = arith::TruncIOp::create(builder, loc,
                                               rewriter.getI1Type(), sign);
                rotatedInputLastIdx = arith::SelectOp::create(
                    builder, loc, sign,
                    arith::SubIOp::create(builder, loc, h, cstIndexOne),
                    arith::AddIOp::create(builder, loc, h, cstIndexOne));
              } else {
                cacheIdx = arith::RemSIOp::create(builder, loc, h,
                                                  cstHalfRotaryEmbDim);
                sign = arith::CmpIOp::create(builder, loc,
                                             arith::CmpIPredicate::sge, h,
                                             cstHalfRotaryEmbDim);
                rotatedInputLastIdx =
                    arith::AddIOp::create(builder, loc, h, cstHalfRotaryEmbDim);
                rotatedInputLastIdx = arith::RemSIOp::create(
                    builder, loc, rotatedInputLastIdx, cstRotaryEmbDim);
              }

              Value positionId = castIntToIndex(builder, loc, args[1]);
              Value cosEmb = tensor::ExtractOp::create(
                  builder, loc, cosCache, ValueRange{positionId, cacheIdx});
              Value sinEmb = tensor::ExtractOp::create(
                  builder, loc, sinCache, ValueRange{positionId, cacheIdx});

              Value origInput = args[0];
              Value rotatedInput = tensor::ExtractOp::create(
                  builder, loc, processedInput,
                  ValueRange{b, n, s, rotatedInputLastIdx});

              Value signMultiplier = arith::SelectOp::create(
                  builder, loc, sign, cstFloatOne, cstFloatMinusOne);

              Value outputI =
                  arith::MulFOp::create(builder, loc, origInput, cosEmb);
              Value outputJ =
                  arith::MulFOp::create(builder, loc, rotatedInput, sinEmb);
              outputJ =
                  arith::MulFOp::create(builder, loc, outputJ, signMultiplier);
              Value out = arith::AddFOp::create(builder, loc, outputI, outputJ);
              linalg::YieldOp::create(builder, loc, out);
            })
            .getResult(0);

    Value result = rotaryEmbedding;

    // Apply scale if not 1.0
    if (scale != 1.0) {
      Value scaleVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(elementType, scale));
      // Create output tensor with same shape as input
      Value scaledOutTensor =
          tensor::EmptyOp::create(rewriter, loc, resultDimsOFR, elementType);
      result = linalg::GenericOp::create(
                   rewriter, loc, processedInputType, ValueRange{result},
                   scaledOutTensor,
                   SmallVector<AffineMap>{
                       AffineMap::getMultiDimIdentityMap(4, context),
                       AffineMap::getMultiDimIdentityMap(4, context)},
                   SmallVector<utils::IteratorType>(
                       4, utils::IteratorType::parallel),
                   [&](OpBuilder &builder, Location loc, ValueRange args) {
                     Value scaled =
                         arith::MulFOp::create(builder, loc, args[0], scaleVal);
                     linalg::YieldOp::create(builder, loc, scaled);
                   })
                   .getResult(0);
    }

    if (needsReshape) {
      // Reshape (batch, num_heads, seq, head_size) -> (batch, seq, hidden)
      // Use original input dims to preserve dynamic info
      // Build i64 shape tensor using original input dims
      auto i64Type = rewriter.getI64Type();
      SmallVector<Value> finalDimVals = {
          arith::IndexCastOp::create(rewriter, loc, i64Type, origBatchDim),
          arith::IndexCastOp::create(rewriter, loc, i64Type, origSeqDim),
          arith::IndexCastOp::create(rewriter, loc, i64Type, origHiddenDim)};
      auto shapeType = RankedTensorType::get(
          {static_cast<int64_t>(finalDimVals.size())}, rewriter.getI64Type());
      Value shapeValue = tensor::FromElementsOp::create(
          rewriter, loc, shapeType, finalDimVals);
      result = tensor::ReshapeOp::create(rewriter, loc, resultType, result,
                                         shapeValue);
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateUncategorizedPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<
      AtenTanOp, AtenTanhOp, AtenSinhOp, AtenCoshOp, AtenAtanhOp, AtenAcoshOp,
      AtenAsinOp, AtenAsinhOp, AtenReluOp, AtenGeluOp, AtenGeluBackwardOp,
      AtenAddTensorOp, AtenMulTensorOp, AtenDivTensorOp, AtenDivTensorModeOp,
      AtenDivScalarModeOp, AtenSubTensorOp, AtenLerpTensorOp, AtenSigmoidOp,
      AtenMinimumOp, AtenAtan2Op, AtenMaximumOp, AtenToDtypeOp, AtenClampOp,
      AtenClampTensorOp, AtenRsubScalarOp, AtenLogOp, AtenErfOp, AtenSqrtOp,
      AtenFloorOp, AtenCeilOp, AtenPreluOp, AtenPowScalarOp,
      AtenPowTensorScalarOp, AtenPowTensorTensorOp, AtenLog2Op, AtenLog10Op,
      AtenLog1pOp, AtenRsqrtOp, AtenAbsOp, AtenComplexOp, AtenReciprocalOp,
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
      AtenQuantizePerTensorOp, AtenIscloseOp>();
  patterns.add<ConvertElementwiseOp>(typeConverter, context);
  target.addIllegalOp<AtenNllLossForwardOp>();
  patterns.add<ConvertAtenDetachOp>(typeConverter, context);
  target.addIllegalOp<AtenDetachOp>();
  patterns.add<ConvertAtenNllLossForwardOp>(typeConverter, context);
  target.addIllegalOp<AtenBatchNormOp>();
  patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
  target.addIllegalOp<AtenLogitOp>();
  patterns.add<ConvertLogitOp>(typeConverter, context);
  target.addIllegalOp<PrimsCollapseOp>();
  patterns.add<ConvertPrimsCollapseOp>(typeConverter, context);
  target.addIllegalOp<PrimsSplitDimOp>();
  patterns.add<ConvertPrimsSplitDimOp>(typeConverter, context);
  target.addIllegalOp<AtenNllLossBackwardOp>();
  patterns.add<ConvertAtenNllLossBackwardOp>(typeConverter, context);
  patterns.add<ConvertTensorStaticInfoCastOp>(typeConverter, context);
  target.addIllegalOp<TensorStaticInfoCastOp>();
  patterns.add<ConvertAtenIntReprOp>(typeConverter, context);
  target.addIllegalOp<AtenIntReprOp>();
  patterns.add<ConvertCastEquivalentOp<Aten_MakePerChannelQuantizedTensorOp>>(
      typeConverter, context);
  target.addIllegalOp<Aten_MakePerChannelQuantizedTensorOp>();
  patterns.add<ConvertCastEquivalentOp<Aten_MakePerTensorQuantizedTensorOp>>(
      typeConverter, context);
  target.addIllegalOp<Aten_MakePerTensorQuantizedTensorOp>();
  patterns.add<ConvertDequantizePerChannel>(typeConverter, context);
  target.addIllegalOp<AtenGridSamplerOp>();
  patterns.add<ConvertAtenGridSamplerOp>(typeConverter, context);
  target.addIllegalOp<Aten__InterpolateSizeListScaleListOp>();
  patterns.add<ConvertInterpolateOp>(typeConverter, context);
  target.addIllegalOp<AtenLinalgDetOp>();
  patterns.add<ConvertAtenLinalgDetOp>(typeConverter, context);
  target.addIllegalOp<AtenPolarOp>();
  patterns.add<ConvertAtenPolarOp>(typeConverter, context);
  target.addIllegalOp<AtenSymConstrainRangeOp>();
  patterns.add<ConvertSymConstrainRangeOp>(typeConverter, context);
  target.addIllegalOp<OnnxVariantRotaryEmbeddingOp>();
  patterns.add<ConvertOnnxVariantRotaryEmbeddingOp>(typeConverter, context);
}
