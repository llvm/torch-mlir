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
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APSInt.h"

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
  }
  llvm_unreachable("Unhandled element type for comparison");
}

static Value createGreaterThan(OpBuilder &b, Location loc, Type elementalType,
                               Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::UGT,
                                  arith::CmpIPredicate::ugt,
                                  arith::CmpIPredicate::sgt>(
      b, loc, elementalType, lhs, rhs);
}

static Value createGreaterThanOrEqual(OpBuilder &b, Location loc,
                                      Type elementalType, Value lhs,
                                      Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::UGE,
                                  arith::CmpIPredicate::uge,
                                  arith::CmpIPredicate::sge>(
      b, loc, elementalType, lhs, rhs);
}

static Value createLessThan(OpBuilder &b, Location loc, Type elementalType,
                            Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::ULT,
                                  arith::CmpIPredicate::ult,
                                  arith::CmpIPredicate::slt>(
      b, loc, elementalType, lhs, rhs);
}

static Value createLessThanOrEqual(OpBuilder &b, Location loc,
                                   Type elementalType, Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::ULE,
                                  arith::CmpIPredicate::ule,
                                  arith::CmpIPredicate::sle>(
      b, loc, elementalType, lhs, rhs);
}

static Value createEqual(OpBuilder &b, Location loc, Type elementalType,
                         Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::UEQ,
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
static Value createCalculationForMathOpWithDtypeConversion(
    OpBuilder &b, TypeConverter *converter, Value payloadArg, Operation *op) {
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
                    std::is_same<OpTy, AtenEqTensorOp>(),
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
  llvm_unreachable("unimplemented: op type not supported");
}

static Value createLinalgPayloadCalculationForElementwiseOp(
    OpBuilder &b, Location loc, TypeConverter *converter,
    ValueRange payloadArgs, Operation *op, ArrayRef<Value> operands) {
  if (isa<AtenFloorOp>(op))
    return b.create<math::FloorOp>(loc, payloadArgs[0]);
  if (isa<AtenCeilOp>(op))
    return b.create<math::CeilOp>(loc, payloadArgs[0]);
  if (isa<AtenTanhOp>(op)) {
    return createCalculationForMathOpWithDtypeConversion<math::TanhOp>(
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
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, adaptor.getAlpha(), dtype);
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
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype);
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
    Type dtype = converter->convertType(clamp.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      clamp.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    AtenClampOp::Adaptor adaptor(operands);
    auto min = adaptor.getMin();
    auto max = adaptor.getMax();
    if (min.getType().isa<Torch::OptionalType>() ||
        max.getType().isa<Torch::OptionalType>()) {
      clamp.emitError("unimplemented: runtime optional type");
      return nullptr;
    }
    auto result = payloadArgs[0];
    if (!min.getType().isa<Torch::NoneType>()) {
      auto minPromoted = convertScalarToDtype(b, loc, min, dtype);
      auto pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,
                                          result, minPromoted);
      result = b.create<arith::SelectOp>(loc, pred, minPromoted, result);
    }
    if (!max.getType().isa<Torch::NoneType>()) {
      auto maxPromoted = convertScalarToDtype(b, loc, max, dtype);
      auto pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                          result, maxPromoted);
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
    Value result = convertScalarToDtype(b, loc, input, dtype);
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
    Value self = payloadArgs[0];
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
    Value threshold = convertScalarToDtype(b, loc, adaptor.getThreshold(), dtype);
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
    Value threshold = convertScalarToDtype(b, loc, adaptor.getThreshold(), dtype);
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
    // Check if the rank of the input tensor is valid.
    AtenTriuOp::Adaptor adaptor(operands);
    auto inputType = adaptor.getSelf().getType().cast<RankedTensorType>();
    uint64_t inputRank = inputType.getRank();
    if (inputRank < 2) {
      triu.emitError("too few dimensions to compute triangular part of matrix");
      return nullptr;
    }

    // Use the indices of the two innermost dimensions.
    auto rowIndex = b.create<linalg::IndexOp>(loc, inputRank - 2);
    Value rowIndexI64 = castIndexToInt64(b, loc, rowIndex);
    auto colIndex = b.create<linalg::IndexOp>(loc, inputRank - 1);
    Value colIndexI64 = castIndexToInt64(b, loc, colIndex);

    // columnIndex >= rowIndex + diagonal?
    auto sum = b.create<arith::AddIOp>(loc, rowIndexI64, adaptor.getDiagonal());
    auto pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                        colIndexI64, sum);

    Value scalar = payloadArgs[0];
    Type elementType = inputType.getElementType();
    Value zero = getConstant(b, loc, 0, elementType);
    return b.create<arith::SelectOp>(loc, pred, scalar, zero);
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
                 APSInt::getAllOnesValue(elementType.getIntOrFloatBitWidth())));
    return b.create<arith::XOrIOp>(loc, payloadArgs[0], allOnesVal);
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
    if (!isa<AtenTanhOp, AtenReluOp, AtenPreluOp, AtenGeluOp,
             AtenGeluBackwardOp, AtenAddTensorOp, AtenMulTensorOp,
             AtenDivTensorOp, AtenDivTensorModeOp, AtenSubTensorOp, AtenAtan2Op,
             AtenLerpTensorOp, AtenSigmoidOp, AtenExpOp, AtenExpm1Op,
             AtenMinimumOp, AtenMaximumOp, AtenToDtypeOp, AtenClampOp,
             AtenRsubScalarOp, AtenMulScalarOp, AtenLogOp, AtenErfOp,
             AtenSqrtOp, AtenFloorOp, AtenPowTensorScalarOp,
             AtenPowTensorTensorOp, AtenLog2Op, AtenLog1pOp, AtenRsqrtOp,
             AtenDivScalarOp, AtenRemainderScalarOp, AtenAbsOp,
             AtenReciprocalOp, AtenBitwiseAndTensorOp, AtenBitwiseOrTensorOp,
             AtenBitwiseXorTensorOp, AtenGtScalarOp, AtenGeScalarOp,
             AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp, AtenWhereSelfOp,
             AtenCeilOp, AtenGtTensorOp, AtenGeTensorOp, AtenEqTensorOp,
             AtenLtTensorOp, AtenLeTensorOp, AtenSubScalarOp, AtenAddScalarOp,
             AtenThresholdOp, AtenThresholdBackwardOp, AtenCloneOp, AtenSinOp,
             AtenCosOp, AtenNeScalarOp, AtenNegOp,
             AtenMaskedFillTensorOp, AtenLogicalOrOp, AtenLogicalAndOp,
             AtenLogicalXorOp, AtenLogicalNotOp, AtenTriuOp, AtenBitwiseNotOp,
             AtenRoundOp, AtenFillScalarOp, AtenFillTensorOp>(op))
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

    if (reduction == torch_upstream::Reduction::Sum ||
        reduction == torch_upstream::Reduction::Mean) {
      Value numOfElems = getTensorSize(rewriter, loc, finalRes);
      numOfElems = convertScalarToDtype(rewriter, loc, numOfElems, elementType);
      llvm::iota_range<int64_t> dimsToReduce(0, targetRank,
                                             /*inclusive=*/false);
      DenseSet<int64_t> dimSet(dimsToReduce.begin(), dimsToReduce.end());

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

    // TODO: Update the second result tensor.
    Value weightUpdated = createZeroInitTensor(rewriter, loc, {}, elementType);
    rewriter.replaceOp(op, {finalRes, weightUpdated});
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
    contractingDim0EqualsNumFeatures(weight);
    contractingDim0EqualsNumFeatures(bias);
    contractingDim0EqualsNumFeatures(runningMean);
    contractingDim0EqualsNumFeatures(runningVar);

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
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, adaptor.getSelf());
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

void mlir::torch::torch_to_linalg::populateUncategorizedPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<
      AtenTanhOp, AtenReluOp, AtenGeluOp, AtenGeluBackwardOp, AtenAddTensorOp,
      AtenMulTensorOp, AtenDivTensorOp, AtenDivTensorModeOp, AtenSubTensorOp,
      AtenLerpTensorOp, AtenSigmoidOp, AtenMinimumOp, AtenAtan2Op,
      AtenMaximumOp, AtenToDtypeOp, AtenClampOp, AtenRsubScalarOp, AtenLogOp,
      AtenErfOp, AtenSqrtOp, AtenFloorOp, AtenCeilOp, AtenPreluOp,
      AtenPowTensorScalarOp, AtenPowTensorTensorOp, AtenLog2Op, AtenLog1pOp,
      AtenRsqrtOp, AtenAbsOp, AtenReciprocalOp, AtenBitwiseAndTensorOp,
      AtenBitwiseOrTensorOp, AtenBitwiseXorTensorOp, AtenGtScalarOp,
      AtenGeScalarOp, AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp,
      AtenWhereSelfOp, AtenGtTensorOp, AtenGeTensorOp, AtenEqTensorOp,
      AtenLtTensorOp, AtenLeTensorOp, AtenThresholdOp, AtenThresholdBackwardOp,
      AtenCloneOp, AtenSinOp, AtenCosOp, AtenNeScalarOp,
      AtenMaskedFillTensorOp, AtenLogicalOrOp, AtenLogicalAndOp,
      AtenLogicalXorOp, AtenLogicalNotOp, AtenTriuOp, AtenRemainderScalarOp,
      AtenBitwiseNotOp, AtenRoundOp, AtenFillScalarOp, AtenFillTensorOp>();
  patterns.add<ConvertElementwiseOp>(typeConverter, context);
  target.addIllegalOp<AtenNllLossForwardOp>();
  patterns.add<ConvertAtenDetachOp>(typeConverter, context);
  target.addIllegalOp<AtenDetachOp>();
  patterns.add<ConvertAtenNllLossForwardOp>(typeConverter, context);
  target.addIllegalOp<AtenBatchNormOp>();
  patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
  target.addIllegalOp<AtenNllLossBackwardOp>();
  patterns.add<ConvertAtenNllLossBackwardOp>(typeConverter, context);
  patterns.add<ConvertTensorStaticInfoCastOp>(typeConverter, context);
  target.addIllegalOp<TensorStaticInfoCastOp>();
}
