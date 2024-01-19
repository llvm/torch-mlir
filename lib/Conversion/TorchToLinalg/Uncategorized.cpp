//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Check if a ranked-tensor has the specified element type.
template <typename elementType> static bool hasElementType(Value tensor) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  Type tensorElementType = tensorType.getElementType();
  return tensorElementType.isa<elementType>();
}

template <arith::CmpFPredicate fpred, arith::CmpIPredicate iupred,
          arith::CmpIPredicate ispred>
static Value createComparisonTemplate(OpBuilder &b, Location loc, Type type,
                                      Value lhs, Value rhs) {
  if (type.isa<mlir::FloatType>())
    return b.create<arith::CmpFOp>(loc, fpred, lhs, rhs);
  if (IntegerType intType = type.dyn_cast<mlir::IntegerType>()) {
    if (intType.isUnsigned())
      return b.create<arith::CmpIOp>(loc, iupred, lhs, rhs);
    if (intType.isSigned())
      return b.create<arith::CmpIOp>(loc, ispred, lhs, rhs);
    assert(intType.getWidth() == 1);
    return b.create<arith::CmpIOp>(loc, iupred, lhs, rhs);
  }
  llvm_unreachable("Unhandled element type for comparison");
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
static Value
createCalculationForMathOpWithDtypeConversion(OpBuilder &b,
                                              const TypeConverter *converter,
                                              Value payloadArg, Operation *op) {
  Type dtype = converter->convertType(op->getResult(0).getType())
                   .template cast<RankedTensorType>()
                   .getElementType();
  Location loc = op->getLoc();
  Value arg = convertScalarToDtype(b, loc, payloadArg, dtype);
  return b.create<MathOpTy>(loc, arg);
}

template <typename OpTy>
static Value createCompareTensorOp(OpBuilder &b, Location loc, OpTy op,
                                   Value lhs, Value rhs) {
  static_assert(std::is_same<OpTy, AtenLtTensorOp>() ||
                    std::is_same<OpTy, AtenLeTensorOp>() ||
                    std::is_same<OpTy, AtenGtTensorOp>() ||
                    std::is_same<OpTy, AtenGeTensorOp>() ||
                    std::is_same<OpTy, AtenEqTensorOp>() ||
                    std::is_same<OpTy, AtenNeTensorOp>(),
                "unimplemented: op type not supported");

  Type lhsDtype = lhs.getType();
  Type rhsDtype = rhs.getType();

  // TODO: Type promotion in case of different `lhsDtype` and `rhsDtype` needs
  // to be handled.
  if (lhsDtype != rhsDtype) {
    op.emitError("unimplemented: lhs and rhs dtype must be same");
    return nullptr;
  }

  Type elementalType =
      op.getSelf().getType().template cast<BaseTensorType>().getDtype();
  if constexpr (std::is_same<OpTy, AtenLtTensorOp>()) {
    return createLessThan(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (std::is_same<OpTy, AtenLeTensorOp>()) {
    return createLessThanOrEqual(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (std::is_same<OpTy, AtenGtTensorOp>()) {
    return createGreaterThan(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (std::is_same<OpTy, AtenGeTensorOp>()) {
    return createGreaterThanOrEqual(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (std::is_same<OpTy, AtenEqTensorOp>()) {
    return createEqual(b, loc, elementalType, lhs, rhs);
  }
  if constexpr (std::is_same<OpTy, AtenNeTensorOp>()) {
    return createNotEqual(b, loc, elementalType, lhs, rhs);
  }
  llvm_unreachable("unimplemented: op type not supported");
}

template <arith::CmpIPredicate predicate>
static LogicalResult
createTriangularMatrix(OpBuilder &b, Location loc, ValueRange payloadArgs,
                       Operation *op, ArrayRef<Value> operands, Value &result) {
  auto inputType = operands[0].getType().cast<RankedTensorType>();
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

static Value createLinalgPayloadCalculationForElementwiseOp(
    OpBuilder &b, Location loc, const TypeConverter *converter,
    ValueRange payloadArgs, Operation *op, ArrayRef<Value> operands) {
  if (isa<AtenFloorOp>(op))
    return b.create<math::FloorOp>(loc, payloadArgs[0]);
  if (isa<AtenCeilOp>(op))
    return b.create<math::CeilOp>(loc, payloadArgs[0]);
  if (isa<AtenTanOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::TanOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenTanhOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::TanhOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenSinhOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::SinhOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenCoshOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::CoshOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenExpOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::ExpOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenExpm1Op>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::ExpM1Op>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLogOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::LogOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLog2Op>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::Log2Op>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLog10Op>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::Log10Op>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenLog1pOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::Log1pOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenErfOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::ErfOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenSqrtOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::SqrtOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenRsqrtOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::RsqrtOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenNegOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<arith::NegFOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenSinOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::SinOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenCosOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::CosOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAtanOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::AtanOp>(
        b, converter, payloadArgs[0], op);
  }
  if (isa<AtenAcosOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::AcosOp>(
        b, converter, payloadArgs[0], op);
  }
  if (auto clone = dyn_cast<AtenCloneOp>(op)) {
    int64_t memoryFormat;
    if (!clone.getMemoryFormat().getType().isa<Torch::NoneType>() &&
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
    if (bitwiseAndTensor.getType()
            .cast<ValueTensorType>()
            .getDtype()
            .isa<mlir::FloatType>()) {
      bitwiseAndTensor.emitError(
          "Bitwise_And does not support floating point dtype");
      return nullptr;
    }
    Type dtype = converter->convertType(bitwiseAndTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::AndIOp>(loc, lhs, rhs);
  }
  if (auto bitwiseAndScalar = dyn_cast<AtenBitwiseAndScalarOp>(op)) {
    Type dtype = converter->convertType(bitwiseAndScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::IntegerType>()) {
      bitwiseAndScalar.emitError(
          "bitwise_and.Scalar does not support non-integer input dtype.");
      return nullptr;
    }
    Type resultElementType =
        bitwiseAndScalar.getType().cast<BaseTensorType>().getDtype();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype,
                                      /*srcOriginalDtype=*/std::nullopt,
                                      /*dstOriginalDtype=*/resultElementType);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    return b.create<arith::AndIOp>(loc, self, other);
  }
  if (auto bitwiseOrTensor = dyn_cast<AtenBitwiseOrTensorOp>(op)) {
    if (bitwiseOrTensor.getType()
            .cast<ValueTensorType>()
            .getDtype()
            .isa<mlir::FloatType>()) {
      bitwiseOrTensor.emitError(
          "Bitwise_Or does not support floating point dtype");
      return nullptr;
    }
    Type dtype = converter->convertType(bitwiseOrTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::OrIOp>(loc, lhs, rhs);
  }
  if (auto bitwiseXorTensor = dyn_cast<AtenBitwiseXorTensorOp>(op)) {
    if (bitwiseXorTensor.getType()
            .cast<ValueTensorType>()
            .getDtype()
            .isa<mlir::FloatType>()) {
      bitwiseXorTensor.emitError(
          "Bitwise_Xor does not support floating point dtype");
      return nullptr;
    }
    Type dtype = converter->convertType(bitwiseXorTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::XOrIOp>(loc, lhs, rhs);
  }
  if (auto bitwiseRightShiftTensor =
          dyn_cast<AtenBitwiseRightShiftTensorOp>(op)) {
    Type dtype = converter->convertType(bitwiseRightShiftTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::IntegerType>()) {
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
    Type dtype = converter->convertType(bitwiseLeftShiftTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::IntegerType>()) {
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
  if (isa<AtenAbsOp>(op))
    return b.create<math::AbsFOp>(loc, payloadArgs[0]);
  if (isa<AtenIsinfOp>(op)){
    Value abs = b.create<math::AbsFOp>(loc, payloadArgs[0]);
    Value infinity = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(abs.getType(), std::numeric_limits<double>::infinity()));
    return createEqual(b, loc, abs.getType(), abs, infinity);
  }
  if (isa<AtenSigmoidOp>(op)) {
    auto negate = createCalculationForMathOpWithDtypeConversion<arith::NegFOp>(
        b, converter, payloadArgs[0], op);
    auto one =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(negate.getType(), 1));
    auto exp = b.create<math::ExpOp>(loc, negate);
    auto added = b.create<arith::AddFOp>(loc, exp, one);
    return b.create<arith::DivFOp>(loc, one, added);
  }
  if (auto relu = dyn_cast<AtenReluOp>(op)) {
    if (!relu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      relu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type elementType = payloadArgs[0].getType();
    Value constZero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                         payloadArgs[0], constZero);
    return b.create<arith::SelectOp>(loc, pred, payloadArgs[0], constZero);
  }
  if (auto round = dyn_cast<AtenRoundOp>(op)) {
    if (!round.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      round.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    return b.create<math::RoundEvenOp>(loc, payloadArgs[0]);
  }
  if (auto prelu = dyn_cast<AtenPreluOp>(op)) {
    if (!prelu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
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
    if (!gelu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      gelu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    // TODO: Take approximation into account.
    std::string approximate;
    if (!matchPattern(gelu.getApproximate(), m_TorchConstantStr(approximate)) ||
        approximate != "none")
      return nullptr;
    Value cdf = buildUnitNormalCdf(b, loc, payloadArgs[0]);
    return b.create<arith::MulFOp>(loc, payloadArgs[0], cdf);
  }
  if (auto geluBackward = dyn_cast<AtenGeluBackwardOp>(op)) {
    if (!geluBackward.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
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
    if (!hardtanhBackward.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
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
    Type dtype = converter->convertType(add.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, adaptor.getAlpha(), dtype);
    if (dtype.isa<mlir::FloatType>()) {
      Value scaled = b.create<arith::MulFOp>(loc, rhs, alpha);
      return b.create<arith::AddFOp>(loc, lhs, scaled);
    } else {
      Value scaled = b.create<arith::MulIOp>(loc, rhs, alpha);
      return b.create<arith::AddIOp>(loc, lhs, scaled);
    }
  }
  if (auto sub = dyn_cast<AtenSubTensorOp>(op)) {
    AtenSubTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(sub.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Type resultElementType = sub.getType().cast<BaseTensorType>().getDtype();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype,
                                     /*srcOriginalDtype=*/std::nullopt,
                                     /*dstOriginalDtype=*/resultElementType);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype,
                                     /*srcOriginalDtype=*/std::nullopt,
                                     /*dstOriginalDtype=*/resultElementType);
    Value alpha = convertScalarToDtype(b, loc, adaptor.getAlpha(), dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    if (dtype.isa<mlir::FloatType>()) {
      Value scaled = b.create<arith::MulFOp>(loc, rhs, alpha);
      return b.create<arith::SubFOp>(loc, lhs, scaled);
    } else {
      Value scaled = b.create<arith::MulIOp>(loc, rhs, alpha);
      return b.create<arith::SubIOp>(loc, lhs, scaled);
    }
  }
  if (auto subScalar = dyn_cast<AtenSubScalarOp>(op)) {
    Type dtype = converter->convertType(subScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype);
    if (dtype.isa<mlir::FloatType>()) {
      Value mult = b.create<arith::MulFOp>(loc, other, alpha);
      return b.create<arith::SubFOp>(loc, self, mult);
    } else if (dtype.isa<mlir::IntegerType>()) {
      Value mult = b.create<arith::MulIOp>(loc, other, alpha);
      return b.create<arith::SubIOp>(loc, self, mult);
    }
    subScalar.emitError("unimplemented: dtype other than float and integer "
                        "types are not supported.");
    return nullptr;
  }
  if (auto addScalar = dyn_cast<AtenAddScalarOp>(op)) {
    Type dtype = converter->convertType(addScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Type resultElementType =
        addScalar.getType().cast<BaseTensorType>().getDtype();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype,
                                      /*srcOriginalDtype=*/std::nullopt,
                                      /*dstOriginalDtype=*/resultElementType);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype,
                                       /*srcOriginalDtype=*/std::nullopt,
                                       /*dstOriginalDtype=*/resultElementType);
    if (dtype.isa<mlir::FloatType>()) {
      Value mult = b.create<arith::MulFOp>(loc, other, alpha);
      return b.create<arith::AddFOp>(loc, self, mult);
    } else if (dtype.isa<mlir::IntegerType>()) {
      Value mult = b.create<arith::MulIOp>(loc, other, alpha);
      return b.create<arith::AddIOp>(loc, self, mult);
    }
    addScalar.emitError("unimplemented: dtype other than float and integer "
                        "types are not supported.");
    return nullptr;
  }
  if (auto mul = dyn_cast<AtenMulTensorOp>(op)) {
    AtenMulTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(mul.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    if (dtype.isa<mlir::FloatType>()) {
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    } else if (dtype.isa<mlir::ComplexType>()) {
      return b.create<complex::MulOp>(loc, lhs, rhs);
    } else {
      return b.create<arith::MulIOp>(loc, lhs, rhs);
    }
  }
  if (auto atan2 = dyn_cast<AtenAtan2Op>(op)) {
    Type dtype = converter->convertType(atan2.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      atan2.emitError("Atan2 requires floating point result type");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<math::Atan2Op>(loc, lhs, rhs);
  }
  if (auto ltTensor = dyn_cast<AtenLtTensorOp>(op)) {
    return createCompareTensorOp(b, loc, ltTensor, payloadArgs[0],
                                 payloadArgs[1]);
  }
  if (auto leTensor = dyn_cast<AtenLeTensorOp>(op)) {
    return createCompareTensorOp(b, loc, leTensor, payloadArgs[0],
                                 payloadArgs[1]);
  }
  if (auto gtTensor = dyn_cast<AtenGtTensorOp>(op)) {
    return createCompareTensorOp(b, loc, gtTensor, payloadArgs[0],
                                 payloadArgs[1]);
  }
  if (auto geTensor = dyn_cast<AtenGeTensorOp>(op)) {
    return createCompareTensorOp(b, loc, geTensor, payloadArgs[0],
                                 payloadArgs[1]);
  }
  if (auto eqTensor = dyn_cast<AtenEqTensorOp>(op)) {
    return createCompareTensorOp(b, loc, eqTensor, payloadArgs[0],
                                 payloadArgs[1]);
  }
  if (auto neTensor = dyn_cast<AtenNeTensorOp>(op)) {
    return createCompareTensorOp(b, loc, neTensor, payloadArgs[0],
                                 payloadArgs[1]);
  }
  if (auto div = dyn_cast<AtenDivTensorOp>(op)) {
    AtenDivTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(div.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      div.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::DivFOp>(loc, lhs, rhs);
  }
  if (auto divTensorMode = dyn_cast<AtenDivTensorModeOp>(op)) {
    AtenDivTensorModeOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(divTensorMode.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      divTensorMode.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value div = b.create<arith::DivFOp>(loc, lhs, rhs);

    if (divTensorMode.getRoundingMode().getType().isa<Torch::NoneType>())
      return div;

    std::string roundingMode;
    if (!matchPattern(divTensorMode.getRoundingMode(),
                      m_TorchConstantStr(roundingMode))) {
      divTensorMode.emitError("only support constant str rounding mode");
      return nullptr;
    }
    if (roundingMode == "trunc") {
      // "trunc" - rounds the results of the division towards zero. Equivalent
      // to C-style integer division.
      Value ceil = b.create<math::CeilOp>(loc, div);
      Value floor = b.create<math::FloorOp>(loc, div);
      Value cstZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(dtype));
      Value pred =
          b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT, div, cstZero);
      return b.create<arith::SelectOp>(loc, pred, ceil, floor);
    }
    if (roundingMode == "floor") {
      // "floor" - rounds the results of the division down. Equivalent to
      // floor division in Python (the // operator)
      return b.create<math::FloorOp>(loc, div);
    }
    divTensorMode.emitError("invalid rounding mode");
    return nullptr;
  }

  if (auto pow = dyn_cast<AtenPowScalarOp>(op)) {
    Type dtype = pow.getType().cast<ValueTensorType>().getDtype();
    if (!dtype.isa<mlir::FloatType>()) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value selfPromoted = convertScalarToDtype(b, loc, operands[0], dtype);
    Value expPromoted = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    return b.create<math::PowFOp>(loc, selfPromoted, expPromoted);
  }

  if (auto pow = dyn_cast<AtenPowTensorScalarOp>(op)) {
    if (!pow.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type dtype = pow.getSelf().getType().cast<ValueTensorType>().getDtype();
    Value expPromoted = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<math::PowFOp>(loc, payloadArgs[0], expPromoted);
  }

  if (auto pow = dyn_cast<AtenPowTensorTensorOp>(op)) {
    Type dtype = converter->convertType(pow.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<math::PowFOp>(loc, lhs, rhs);
  }

  if (auto imag = dyn_cast<AtenImagOp>(op)) {
    Type dtype = converter->convertType(imag.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      imag.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value imagVal = b.create<complex::ImOp>(loc, payloadArgs[0]);
    return imagVal;
  }

  if (auto real = dyn_cast<AtenRealOp>(op)) {
    Type dtype = converter->convertType(real.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      real.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value realVal = b.create<complex::ReOp>(loc, payloadArgs[0]);
    return realVal;
  }

  if (auto gtScalar = dyn_cast<AtenGtScalarOp>(op)) {
    Type dtype = gtScalar.getSelf().getType().cast<BaseTensorType>().getDtype();

    // TODO: `gtTensor` and `gtScalar` share similar code and can be called from
    // one static function.
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor args from integer to float.
        gtScalar.emitError(
            "unimplemented: type promotion from tensor to scalar.");
        return nullptr;
      }

      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                       payloadArgs[0], otherPromoted);
    }
    gtScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto geScalar = dyn_cast<AtenGeScalarOp>(op)) {
    Type dtype = geScalar.getSelf().getType().cast<BaseTensorType>().getDtype();

    // TODO: The `AtenGeScalarOp` and `AtenGtScalarOp` share a lot of code that
    // can be refactored.
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGE,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor args from integer to float.
        geScalar.emitError(
            "unimplemented: type promotion from tensor to scalar.");
        return nullptr;
      }

      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                       payloadArgs[0], otherPromoted);
    }
    geScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto eqScalar = dyn_cast<AtenEqScalarOp>(op)) {
    Type dtype = eqScalar.getSelf().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        eqScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
    }
    return createEqual(b, loc, dtype, payloadArgs[0], otherPromoted);
  }

  if (auto neScalar = dyn_cast<AtenNeScalarOp>(op)) {
    Type dtype = neScalar.getSelf().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        neScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
    }
    return createNotEqual(b, loc, dtype, payloadArgs[0], otherPromoted);
  }

  if (auto ltScalar = dyn_cast<AtenLtScalarOp>(op)) {
    Type dtype = ltScalar.getSelf().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    // TODO:  Both tensor and scalar variants of `aten.gt` and `aten.lt` share
    // a lot of code that can be refactored.
    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        ltScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       payloadArgs[0], otherPromoted);
    }
    ltScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto leScalar = dyn_cast<AtenLeScalarOp>(op)) {
    Type dtype = leScalar.getSelf().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    // TODO: The `AtenLeScalarOp` and `AtenLtScalarOp` share a lot of code
    // that can be refactored.
    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        leScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                       payloadArgs[0], otherPromoted);
    }
    leScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto whereSelf = dyn_cast<AtenWhereSelfOp>(op)) {
    Type dtype = converter->convertType(whereSelf.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[2], dtype);
    return b.create<arith::SelectOp>(loc, payloadArgs[0], lhs, rhs);
  }

  if (auto lerp = dyn_cast<AtenLerpTensorOp>(op)) {
    if (!lerp.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
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
    Type dtype = minimum.getType().cast<BaseTensorType>().getDtype();
    Type elemTy = converter->convertType(minimum.getType())
                      .cast<RankedTensorType>()
                      .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], elemTy);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], elemTy);
    Value pred = createLessThan(b, loc, dtype, lhs, rhs);
    return b.create<arith::SelectOp>(loc, pred, lhs, rhs);
  }
  if (auto maximum = dyn_cast<AtenMaximumOp>(op)) {
    Type dtype = maximum.getType().cast<BaseTensorType>().getDtype();
    Type elemTy = converter->convertType(maximum.getType())
                      .cast<RankedTensorType>()
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
    if (min.getType().isa<Torch::OptionalType>() ||
        max.getType().isa<Torch::OptionalType>()) {
      clamp.emitError("unimplemented: runtime optional type");
      return nullptr;
    }

    Type dtype = converter->convertType(clamp.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType, mlir::IntegerType>()) {
      clamp.emitError("unimplement type for clamp");
      return nullptr;
    }

    Type dstOriginalDtype = clamp.getType().cast<BaseTensorType>().getDtype();
    bool isUnsigned = isa<QUInt8Type>(dstOriginalDtype);
    if (auto intTy = dstOriginalDtype.dyn_cast<IntegerType>()) {
      isUnsigned = intTy.isUnsigned();
    }
    auto cmpSelect = [&](Value input, Value clamp, bool getMax) -> Value {
      clamp = convertScalarToDtype(b, loc, clamp, dtype,
                                   /*srcOriginalDtype=*/std::nullopt,
                                   /*dstOriginalDtype=*/dstOriginalDtype);

      Value pred;
      if (dtype.isa<mlir::FloatType>()) {
        auto cmp =
            getMax ? arith::CmpFPredicate::UGT : arith::CmpFPredicate::ULT;
        pred = b.create<arith::CmpFOp>(loc, cmp, input, clamp);
      } else if (dtype.isa<mlir::IntegerType>()) {
        auto cmp =
            isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
        if (getMax)
          cmp = arith::invertPredicate(cmp);
        pred = b.create<arith::CmpIOp>(loc, cmp, input, clamp);
      }
      return b.create<arith::SelectOp>(loc, pred, clamp, input);
    };

    auto result = payloadArgs[0];
    if (!min.getType().isa<Torch::NoneType>())
      result = cmpSelect(result, min, /*getMax=*/false);
    if (!max.getType().isa<Torch::NoneType>())
      result = cmpSelect(result, max, /*getMax=*/true);
    return result;
  }
  if (auto clampTensor = dyn_cast<AtenClampTensorOp>(op)) {
    AtenClampTensorOp::Adaptor adaptor(operands);
    auto min = adaptor.getMin();
    auto max = adaptor.getMax();
    if (min.getType().isa<Torch::OptionalType>() ||
        max.getType().isa<Torch::OptionalType>()) {
      clampTensor.emitError("unimplemented: runtime optional type");
      return nullptr;
    }
    Type dtype = converter->convertType(clampTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    bool isMinNone = true;
    auto result = payloadArgs[0];
    if (!min.getType().isa<Torch::NoneType>()) {
      isMinNone = false;
      auto minPromoted = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
      Value pred;
      if (dtype.isa<mlir::FloatType>()) {
        pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT, result,
                                       minPromoted);
      } else if (dtype.isa<mlir::IntegerType>()) {
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
    if (!max.getType().isa<Torch::NoneType>()) {
      max = isMinNone ? payloadArgs[1] : payloadArgs[2];
      auto maxPromoted = convertScalarToDtype(b, loc, max, dtype);
      Value pred;
      if (dtype.isa<mlir::FloatType>()) {
        pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, result,
                                       maxPromoted);
      } else if (dtype.isa<mlir::IntegerType>()) {
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
    Type dtype = converter->convertType(rsub.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype);
    if (dtype.isa<mlir::FloatType>()) {
      Value mult = b.create<arith::MulFOp>(loc, self, alpha);
      return b.create<arith::SubFOp>(loc, other, mult);
    } else if (dtype.isa<mlir::IntegerType>()) {
      Value mult = b.create<arith::MulIOp>(loc, self, alpha);
      return b.create<arith::SubIOp>(loc, other, mult);
    }
    rsub.emitError("unimplemented: dtype other than float and integer "
                   "types are not supported.");
    return nullptr;
  }
  if (auto mulScalar = dyn_cast<AtenMulScalarOp>(op)) {
    Type dtype = converter->convertType(mulScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, operands[1], dtype);
    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    if (dtype.isa<mlir::IntegerType>())
      return b.create<arith::MulIOp>(loc, lhs, rhs);
    mulScalar.emitError("unimplemented: Only integer/float dtype supported");
    return nullptr;
  }
  if (auto atenToDtype = dyn_cast<AtenToDtypeOp>(op)) {
    Value input = payloadArgs[0];
    Type dtype = converter->convertType(atenToDtype.getType())
                     .cast<RankedTensorType>()
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
                                        /*srcOriginalDtype=*/std::nullopt,
                                        /*dstOriginalDtype=*/resultElementType);
    return result;
  }
  if (auto divScalar = dyn_cast<AtenDivScalarOp>(op)) {
    Type dtype = converter->convertType(divScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      divScalar.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<arith::DivFOp>(loc, self, other);
  }
  if (auto remScalar = dyn_cast<AtenRemainderScalarOp>(op)) {
    Type newResultType = converter->convertType(remScalar.getType())
                             .cast<RankedTensorType>()
                             .getElementType();

    Value self = convertScalarToDtype(b, loc, payloadArgs[0], newResultType);
    Value other = convertScalarToDtype(b, loc, operands[1], newResultType);
    Value result;

    if (newResultType.isa<mlir::FloatType>()) {
      result = b.create<arith::RemFOp>(loc, self, other);
    } else if (newResultType.isa<mlir::IntegerType>()) {
      result = b.create<arith::RemSIOp>(loc, self, other);
    } else {
      remScalar.emitError(
          "Unsupported type encountered for AtenRemainderScalarOp.");
    }

    return result;
  }
  if (auto remTensor = dyn_cast<AtenRemainderTensorOp>(op)) {
    Type newResultType = converter->convertType(remTensor.getType())
                             .cast<RankedTensorType>()
                             .getElementType();

    Value self = convertScalarToDtype(b, loc, payloadArgs[0], newResultType);
    Value other = convertScalarToDtype(b, loc, payloadArgs[1], newResultType);
    Value result;

    if (newResultType.isa<mlir::FloatType>()) {
      result = b.create<arith::RemFOp>(loc, self, other);
    } else if (newResultType.isa<mlir::IntegerType>()) {
      result = b.create<arith::RemSIOp>(loc, self, other);
    } else {
      remTensor.emitError(
          "Unsupported type encountered for AtenRemainderTensorOp.");
    }

    return result;
  }
  if (auto reciprocal = dyn_cast<AtenReciprocalOp>(op)) {
    Type dtype = converter->convertType(reciprocal.getType())
                     .cast<RankedTensorType>()
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
    Type dtype = converter->convertType(thresholdOp.getType())
                     .cast<RankedTensorType>()
                     .getElementType();

    Value self = payloadArgs[0];
    Value threshold =
        convertScalarToDtype(b, loc, adaptor.getThreshold(), dtype);
    Value value = convertScalarToDtype(b, loc, adaptor.getValue(), dtype);

    Value predicate;
    if (dtype.isa<mlir::FloatType>())
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
    Type dtype = converter->convertType(thresholdBackward.getType())
                     .cast<RankedTensorType>()
                     .getElementType();

    Value grad = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value self = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value threshold =
        convertScalarToDtype(b, loc, adaptor.getThreshold(), dtype);
    Value constantZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(dtype));

    Value predicate;
    if (dtype.isa<mlir::FloatType>())
      predicate = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE, self,
                                          threshold);
    else
      predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, self,
                                          threshold);
    return b.create<arith::SelectOp>(loc, predicate, constantZero, grad);
  }
  if (auto fillScalar = dyn_cast<AtenFillScalarOp>(op)) {
    AtenFillScalarOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(fillScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    return convertScalarToDtype(b, loc, adaptor.getValue(), dtype);
  }
  if (auto maskedFillTensor = dyn_cast<AtenMaskedFillTensorOp>(op)) {
    AtenMaskedFillScalarOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(maskedFillTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();

    Value input = payloadArgs[0];
    Value mask = payloadArgs[1];
    Value fillValue = convertScalarToDtype(b, loc, payloadArgs[2], dtype);
    return b.create<arith::SelectOp>(loc, mask, fillValue, input);
  }
  if (auto fillTensor = dyn_cast<AtenFillTensorOp>(op)) {
    AtenFillTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(fillTensor.getType())
                     .cast<RankedTensorType>()
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
    Type elementType = converter->convertType(bitwiseNot.getType())
                           .cast<RankedTensorType>()
                           .getElementType();
    if (elementType.isa<mlir::FloatType>()) {
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
    auto qtensorTy = qtensor.getType().cast<ValueTensorType>().getDtype();
    auto makeQTensor =
        qtensor.getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>();
    if (!makeQTensor) {
      op->emitError(
          "unimplemented: dequantizing tensor of unknown scale / zero-point");
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

    Value zp = makeQTensor.getZeroPoint();
    zp = converter->materializeTargetConversion(
        b, loc, converter->convertType(zp.getType()),
        makeQTensor.getZeroPoint());
    auto zpTy = zp.getType();

    if (zpTy != outIntTy) {
      zp = b.create<arith::TruncIOp>(loc, outIntTy, zp);
    }

    value = b.create<arith::SubIOp>(loc, value, zp);

    if (torch_to_linalg::isUnsignedTorchType(qtensorTy)) {
      value = b.create<arith::UIToFPOp>(loc, outFpTy, value);
    } else {
      value = b.create<arith::SIToFPOp>(loc, outFpTy, value);
    }

    Value scale = makeQTensor.getScale();
    scale = converter->materializeTargetConversion(
        b, loc, converter->convertType(scale.getType()),
        makeQTensor.getScale());
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
    value = b.create<math::RoundOp>(loc, value);
    value = b.create<arith::AddFOp>(loc, value, zp);

    auto destTy = payloadArgs[1].getType();
    auto bitwidth = destTy.getIntOrFloatBitWidth();
    bool isUnsigned = torch_to_linalg::isUnsignedTorchType(quant.getType());
    APInt min = isUnsigned ? APInt::getMinValue(bitwidth)
                           : APInt::getSignedMinValue(bitwidth);
    APInt max = isUnsigned ? APInt::getMaxValue(bitwidth)
                           : APInt::getSignedMaxValue(bitwidth);

    Value minVal = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(valueTy, min.getSExtValue()));
    Value maxVal = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(valueTy, max.getSExtValue()));
    Value minCmp =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT, value, minVal);
    Value maxCmp =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, value, maxVal);
    value = b.create<arith::SelectOp>(loc, minCmp, minVal, value);
    value = b.create<arith::SelectOp>(loc, maxCmp, maxVal, value);

    if (isUnsigned) {
      value = b.create<arith::FPToUIOp>(loc, destTy, value);
    } else {
      value = b.create<arith::FPToSIOp>(loc, destTy, value);
    }

    return value;
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
             AtenSubTensorOp, AtenAtan2Op, AtenLerpTensorOp, AtenSigmoidOp,
             AtenExpOp, AtenExpm1Op, AtenMinimumOp, AtenMaximumOp,
             AtenToDtypeOp, AtenClampOp, AtenClampTensorOp, AtenRsubScalarOp,
             AtenMulScalarOp, AtenLogOp, AtenErfOp, AtenSqrtOp, AtenFloorOp,
             AtenPowScalarOp, AtenPowTensorScalarOp, AtenPowTensorTensorOp,
             AtenLog2Op, AtenLog10Op, AtenLog1pOp, AtenRsqrtOp, AtenDivScalarOp,
             AtenRemainderScalarOp, AtenRemainderTensorOp, AtenAbsOp,
             AtenReciprocalOp, AtenBitwiseAndTensorOp, AtenBitwiseAndScalarOp,
             AtenBitwiseOrTensorOp, AtenBitwiseXorTensorOp,
             AtenBitwiseLeftShiftTensorOp, AtenBitwiseRightShiftTensorOp,
             AtenGtScalarOp, AtenGeScalarOp, AtenEqScalarOp, AtenLtScalarOp,
             AtenLeScalarOp, AtenWhereSelfOp, AtenCeilOp, AtenGtTensorOp,
             AtenGeTensorOp, AtenEqTensorOp, AtenNeTensorOp, AtenLtTensorOp,
             AtenLeTensorOp, AtenSubScalarOp, AtenAddScalarOp, AtenThresholdOp,
             AtenThresholdBackwardOp, AtenHardtanhBackwardOp, AtenCloneOp,
             AtenSinOp, AtenCosOp, AtenNeScalarOp, AtenNegOp,
             AtenMaskedFillTensorOp, AtenLogicalOrOp, AtenLogicalAndOp,
             AtenLogicalXorOp, AtenLogicalNotOp, AtenIsinfOp, AtenTriuOp,
             AtenTrilOp, AtenBitwiseNotOp, AtenRoundOp, AtenFillScalarOp,
             AtenFillTensorOp, AtenAtanOp, AtenAcosOp, AtenRealOp, AtenImagOp,
             AtenDequantizeSelfOp, AtenDequantizeTensorOp,
             AtenQuantizePerTensorOp>(op))
      return rewriter.notifyMatchFailure(op, "not a supported elementwise op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    auto tensorOperands = llvm::to_vector<6>(llvm::make_filter_range(
        operands, [](Value v) { return v.getType().isa<RankedTensorType>(); }));
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
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
    if (!weight.getType().isa<mlir::torch::Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    Value ignoreIndex = adaptor.getIgnoreIndex();
    Value ignoreIndexVal = castIntToIndex(rewriter, loc, ignoreIndex);

    unsigned inputRank = input.getType().cast<RankedTensorType>().getRank();
    unsigned targetRank = target.getType().cast<RankedTensorType>().getRank();

    // TODO: Add support for k-dim loss.
    if (inputRank > 2) {
      return rewriter.notifyMatchFailure(
          op, "expected input and target to be rank <= 2");
    }
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
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
      Value numOfElems = getTensorSize(rewriter, loc, finalRes);
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

    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();
    auto runningMeanType = runningMean.getType().cast<RankedTensorType>();
    auto runningVarType = runningVar.getType().cast<RankedTensorType>();

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
    bool weightIsNone = op.getWeight().getType().isa<Torch::NoneType>();
    Value ignoreIndex = castIntToIndex(rewriter, loc, adaptor.getIgnoreIndex());
    Value totalWeight = adaptor.getTotalWeight();

    auto inputType = input.getType().cast<RankedTensorType>();
    int inputRank = inputType.getRank();
    auto gradOutputType = gradOutput.getType().cast<RankedTensorType>();
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
      auto tensorType = tensor.getType().cast<RankedTensorType>();
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

    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
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

    auto aRankedTensorType = adaptor.getA().getType().cast<RankedTensorType>();

    const TypeConverter *typeConverter = getTypeConverter();

    auto resultRankedTensorType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

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

    auto aRankedTensorType = adaptor.getA().getType().cast<RankedTensorType>();
    const TypeConverter *typeConverter = getTypeConverter();

    auto resultRankedTensorType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

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
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
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

    if (handleEps && !eps.getType().isa<mlir::FloatType>()) {
      op.emitError("Logit does not support non-floating point type");
      return failure();
    }

    auto inputType = input.getType().cast<RankedTensorType>();
    auto inputElementType = inputType.getElementType();

    if (!inputElementType.isa<mlir::FloatType>()) {
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
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getSelf());
    return success();
  }
};
} // namespace

namespace {
class ConvertMakePerTensorQuantizedTensorOp
    : public OpConversionPattern<Aten_MakePerTensorQuantizedTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Aten_MakePerTensorQuantizedTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getSelf());
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateUncategorizedPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<
      AtenTanOp, AtenTanhOp, AtenSinhOp, AtenCoshOp, AtenReluOp, AtenGeluOp,
      AtenGeluBackwardOp, AtenAddTensorOp, AtenMulTensorOp, AtenDivTensorOp,
      AtenDivTensorModeOp, AtenSubTensorOp, AtenLerpTensorOp, AtenSigmoidOp,
      AtenMinimumOp, AtenAtan2Op, AtenMaximumOp, AtenToDtypeOp, AtenClampOp,
      AtenClampTensorOp, AtenRsubScalarOp, AtenLogOp, AtenErfOp, AtenSqrtOp,
      AtenFloorOp, AtenCeilOp, AtenPreluOp, AtenPowScalarOp,
      AtenPowTensorScalarOp, AtenPowTensorTensorOp, AtenLog2Op, AtenLog10Op,
      AtenLog1pOp, AtenRsqrtOp, AtenAbsOp, AtenReciprocalOp,
      AtenBitwiseAndTensorOp, AtenBitwiseAndScalarOp, AtenBitwiseOrTensorOp,
      AtenBitwiseXorTensorOp, AtenBitwiseLeftShiftTensorOp,
      AtenBitwiseRightShiftTensorOp, AtenGtScalarOp, AtenGeScalarOp,
      AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp, AtenWhereSelfOp,
      AtenGtTensorOp, AtenGeTensorOp, AtenEqTensorOp, AtenNeTensorOp,
      AtenLtTensorOp, AtenLeTensorOp, AtenThresholdOp, AtenThresholdBackwardOp,
      AtenHardtanhBackwardOp, AtenCloneOp, AtenSinOp, AtenCosOp, AtenNeScalarOp,
      AtenMaskedFillTensorOp, AtenLogicalOrOp, AtenLogicalAndOp, AtenAtanOp,
      AtenAcosOp, AtenLogicalXorOp, AtenLogicalNotOp, AtenIsinfOp, AtenTriuOp,
      AtenTrilOp, AtenRemainderScalarOp, AtenRemainderTensorOp,
      AtenBitwiseNotOp, AtenRoundOp, AtenFillScalarOp, AtenFillTensorOp,
      AtenRealOp, AtenImagOp, AtenDequantizeSelfOp, AtenDequantizeTensorOp,
      AtenQuantizePerTensorOp>();
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
  patterns.add<ConvertMakePerTensorQuantizedTensorOp>(typeConverter, context);
  target.addIllegalOp<Aten_MakePerTensorQuantizedTensorOp>();
}
