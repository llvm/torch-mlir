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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static Value createElementwiseLinalgGeneric(
    OpBuilder &b, Location loc, ValueRange tensorOperands,
    Type resultElementType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  // The overall error handling strategy here is best viewed by thinking about
  // what happens for a single result dimension. This loop not structured that
  // way because it is hard to create the affine maps for each operand unless
  // we structure the loop to iterate over tensor operands as the outer loop
  // instead of inner loop. This valsemcode gives better intuition:
  // ```
  // for each result dimension:
  //   for each tensor operand:
  //     if it doesn't even have high enough rank relative to the result:
  //       continue
  //     if it is a static size-1 along this result dimension:
  //       continue
  //     if this is the first tensor operand that didn't continue above:
  //       take its dimension size as the size of the non-broadcasted
  //       traversal along this dimension (this may include a dynamic size-1,
  //       **non-broadcasted** traversal!)
  //     emit error check "if the size does not match the non-broadcasted
  //     traversal size along this dimension, error"
  // ```
  SmallVector<int64_t> operandRanks;
  operandRanks.resize(tensorOperands.size());
  llvm::transform(tensorOperands, operandRanks.begin(), [](Value tensor) {
    return tensor.getType().dyn_cast<RankedTensorType>().getRank();
  });

  auto resultRankIt =
      std::max_element(operandRanks.begin(), operandRanks.end());
  assert(resultRankIt != operandRanks.end() && "Unable to get result rank.");
  int64_t resultRank = *resultRankIt;

  // Initialize the resultShape to all 1's, as a fallback in case
  // all sizes along that result dimension are statically 1.
  auto c1 = b.create<arith::ConstantIndexOp>(loc, /*value=*/1);
  SmallVector<Value> resultShape(resultRank, c1);
  SmallVector<AffineMap> indexingMaps;
  for (Value tensorOperand : tensorOperands) {
    SmallVector<AffineExpr> exprs;
    auto type = tensorOperand.getType().cast<RankedTensorType>();
    for (auto size : llvm::enumerate(type.getShape())) {
      // If the size is statically known to be 1, we don't want any
      // error guards to be spuriously emitted, since we are specifically
      // allowing size-1 broadcasts in this case, as they correspond to a
      // constant-0 indexing map.
      if (size.value() == 1) {
        exprs.push_back(b.getAffineConstantExpr(0));
        continue;
      }

      // The rank of this operand might be smaller than the overall rank of
      // the broadcast. Add an offset to correlate it to the correct
      // dimension of the result.
      auto resultDim = size.index() + (resultRank - type.getRank());

      // The generated linalg op will now be iterating along the full size
      // of this dimension. Record that fact.
      exprs.push_back(b.getAffineDimExpr(resultDim));

      // Now, we need to ensure that such iteration is not going to trigger
      // undefined behavior, by doing appropriate checks against the current
      // dimension size.
      auto currentDimSize = getDimOp(b, loc, tensorOperand, size.index());

      // If the result size of this dimension has so far only hit the
      // statically-known-to-be-1 case above (i.e., we have not yet assigned a
      // new Value to `resultShape[resultDim]`), then we have no other dynamic
      // values to check against, and merely need to record the current
      // dimension size.
      if (resultShape[resultDim] == c1) {
        resultShape[resultDim] = currentDimSize;
        continue;
      }

      // We prohibit the size-1 dynamic broadcasting scenario, so just check
      // for exact equality with the running result size.
      // This is the check which protects against the undefined behavior of
      // the generated linalg op in the case of iterating two operands with
      // dimensions sizes that are expected to match.
      auto equalToRunning =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                  resultShape[resultDim], currentDimSize);
      b.create<cf::AssertOp>(loc, equalToRunning,
                             "mismatched size for broadcast");
    }
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultRank, /*symbolCount=*/0, exprs, b.getContext()));
  }

  SmallVector<StringRef> iteratorTypes(resultRank,
                                       getParallelIteratorTypeName());
  // Add the indexing map for the outs init tensor.
  indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));

  Value initTensor = b.create<linalg::InitTensorOp>(
      loc, getAsOpFoldResult(resultShape), resultElementType);
  return b
      .create<linalg::GenericOp>(loc,
                                 /*resultTensorTypes=*/initTensor.getType(),
                                 /*inputs=*/tensorOperands,
                                 /*outputs=*/initTensor, indexingMaps,
                                 iteratorTypes, bodyBuild)
      .getResult(0);
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
  assert(false && "Unhandled element type for comparison");
}

static Value createGreaterThan(OpBuilder &b, Location loc, Type elementalType,
                               Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::UGT,
                                  arith::CmpIPredicate::ugt,
                                  arith::CmpIPredicate::sgt>(
      b, loc, elementalType, lhs, rhs);
}

static Value createLessThan(OpBuilder &b, Location loc, Type elementalType,
                            Value lhs, Value rhs) {
  return createComparisonTemplate<arith::CmpFPredicate::ULT,
                                  arith::CmpIPredicate::ult,
                                  arith::CmpIPredicate::slt>(
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

static Value createLinalgPayloadCalculationForElementwiseOp(
    OpBuilder &b, Location loc, TypeConverter *converter,
    ValueRange payloadArgs, Operation *op, ArrayRef<Value> operands) {
  if (isa<AtenTanhOp>(op))
    return b.create<math::TanhOp>(loc, payloadArgs[0]);
  if (isa<AtenExpOp>(op))
    return b.create<math::ExpOp>(loc, payloadArgs[0]);
  if (isa<AtenFloorOp>(op))
    return b.create<math::FloorOp>(loc, payloadArgs[0]);
  if (isa<AtenCeilOp>(op))
    return b.create<math::CeilOp>(loc, payloadArgs[0]);
  if (isa<AtenLogOp>(op))
    return b.create<math::LogOp>(loc, payloadArgs[0]);
  if (isa<AtenErfOp>(op))
    return b.create<math::ErfOp>(loc, payloadArgs[0]);
  if (isa<AtenSqrtOp>(op))
    return b.create<math::SqrtOp>(loc, payloadArgs[0]);
  if (isa<AtenRsqrtOp>(op))
    return b.create<math::RsqrtOp>(loc, payloadArgs[0]);
  if (auto clone = dyn_cast<AtenCloneOp>(op)) {
    int64_t memoryFormat;
    if (!clone.memory_format().getType().isa<Torch::NoneType>() &&
        (!matchPattern(clone.memory_format(),
                       m_TorchConstantInt(&memoryFormat)) ||
         memoryFormat != torch_upstream::MemoryFormat::Contiguous)) {
      clone.emitError("unimplemented: only default memory format is supported");
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
  if (isa<AtenLog2Op>(op))
    return b.create<math::Log2Op>(loc, payloadArgs[0]);
  if (isa<AtenAbsOp>(op))
    return b.create<math::AbsOp>(loc, payloadArgs[0]);
  if (isa<AtenSigmoidOp>(op)) {
    Type elementType = payloadArgs[0].getType();
    auto one = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1));
    auto negate = b.create<arith::NegFOp>(loc, payloadArgs[0]);
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
  if (auto lrelu = dyn_cast<AtenLeakyReluOp>(op)) {
    if (!lrelu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      lrelu.emitError("unimplemented: non-floating point dtype");
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
    Value scale = convertScalarToDtype(b, loc, operands[1], elementType);
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
    if (!matchPattern(gelu.approximate(), m_TorchConstantStr(approximate)) ||
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
    if (!matchPattern(geluBackward.approximate(),
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
    Value alpha = convertScalarToDtype(b, loc, adaptor.alpha(), dtype);
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
    Value alpha = convertScalarToDtype(b, loc, adaptor.alpha(), dtype);
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
  if (auto gtTensor = dyn_cast<AtenGtTensorOp>(op)) {
    AtenGtTensorOp::Adaptor adaptor(operands);
    Type lhsDtype = payloadArgs[0].getType();
    Type rhsDtype = payloadArgs[1].getType();

    // TODO: Type promotion in case of different `lhsDtype` and `rhsDtype` needs
    // to be handled.
    if (lhsDtype != rhsDtype) {
      gtTensor.emitError("unimplemented: different lhs and rhs dtype");
      return nullptr;
    }

    Type elementalType =
        gtTensor.self().getType().cast<BaseTensorType>().getDtype();
    return createGreaterThan(b, loc, elementalType, payloadArgs[0],
                             payloadArgs[1]);
  }
  if (auto eqTensor = dyn_cast<AtenEqTensorOp>(op)) {
    AtenEqTensorOp::Adaptor adaptor(operands);
    Type lhsDtype = payloadArgs[0].getType();
    Type rhsDtype = payloadArgs[1].getType();

    // TODO: Type promotion in case of different `lhsDtype` and `rhsDtype` needs
    // to be handled.
    if (lhsDtype != rhsDtype) {
      eqTensor.emitError("unimplemented: lhs and rhs dtype must be same");
      return nullptr;
    }

    Type elementalType =
        eqTensor.self().getType().cast<BaseTensorType>().getDtype();

    if (elementalType.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ,
                                     payloadArgs[0], payloadArgs[1]);
    if (elementalType.isa<mlir::IntegerType>()) {
      return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                     payloadArgs[0], payloadArgs[1]);
    }
    eqTensor.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }
  if (auto ltTensor = dyn_cast<AtenLtTensorOp>(op)) {
    AtenLtTensorOp::Adaptor adaptor(operands);
    Type lhsDtype = payloadArgs[0].getType();
    Type rhsDtype = payloadArgs[1].getType();

    // TODO: Type promotion in case of different `lhsDtype` and `rhsDtype` needs
    // to be handled.
    if (lhsDtype != rhsDtype) {
      ltTensor.emitError("unimplemented: lhs and rhs dtype must be same");
      return nullptr;
    }

    Type elementalType =
        ltTensor.self().getType().cast<BaseTensorType>().getDtype();
    return createLessThan(b, loc, elementalType, payloadArgs[0],
                          payloadArgs[1]);
  }
  if (auto div = dyn_cast<AtenDivTensorOp>(op)) {
    AtenDivTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(div.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>())
      div.emitError("unimplemented: non-floating point dtype");
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::DivFOp>(loc, lhs, rhs);
  }
  if (auto pow = dyn_cast<AtenPowTensorScalarOp>(op)) {
    if (!pow.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type dtype = pow.self().getType().cast<ValueTensorType>().getDtype();
    Value expPromoted = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<math::PowFOp>(loc, payloadArgs[0], expPromoted);
  }

  if (auto gtScalar = dyn_cast<AtenGtScalarOp>(op)) {
    Type dtype = gtScalar.self().getType().cast<BaseTensorType>().getDtype();

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
    Type dtype = geScalar.self().getType().cast<BaseTensorType>().getDtype();

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
    Type dtype = eqScalar.self().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ,
                                     payloadArgs[0], otherPromoted);
    if (dtype.isa<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        eqScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
      return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                     payloadArgs[0], otherPromoted);
    }
    eqScalar.emitError("unimplemented: dtype isn't supported");
    return nullptr;
  }

  if (auto ltScalar = dyn_cast<AtenLtScalarOp>(op)) {
    Type dtype = ltScalar.self().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    // TODO:  Both tensor and scalar variants of `aten.gt` and `aten.lt` share a
    // lot of code that can be refactored.
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
    Type dtype = leScalar.self().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    // TODO: The `AtenLeScalarOp` and `AtenLtScalarOp` share a lot of code that
    // can be refactored.
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
    auto start = adaptor.self();
    auto end = adaptor.end();
    auto weight = adaptor.weight();
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
    auto min = adaptor.min();
    auto max = adaptor.max();
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
    if (!dtype.isa<mlir::FloatType>()) {
      rsub.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value self = payloadArgs[0];
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype);
    Value mult = b.create<arith::MulFOp>(loc, self, alpha);
    return b.create<arith::SubFOp>(loc, other, mult);
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
  if (auto reciprocal = dyn_cast<AtenReciprocalOp>(op)) {
    if (!reciprocal.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      reciprocal.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }

    Type elementType = payloadArgs[0].getType();
    // assert(element != 0)
    auto zero =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    auto pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE,
                                        payloadArgs[0], zero);
    b.create<cf::AssertOp>(
        loc, pred, b.getStringAttr("unimplemented: tensor with zero element"));

    auto one =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1.0));
    return b.create<arith::DivFOp>(loc, one, payloadArgs[0]);
  }
  if (auto thresholdOp = dyn_cast<AtenThresholdOp>(op)) {
    // The approach used here is as follows:
    //        result = self <= threshold ? value : self
    AtenThresholdOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(thresholdOp.getType())
                     .cast<RankedTensorType>()
                     .getElementType();

    Value self = payloadArgs[0];
    Value threshold = convertScalarToDtype(b, loc, adaptor.threshold(), dtype);
    Value value = convertScalarToDtype(b, loc, adaptor.value(), dtype);

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
    Value threshold = convertScalarToDtype(b, loc, adaptor.threshold(), dtype);
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
    if (!isa<AtenTanhOp, AtenReluOp, AtenLeakyReluOp, AtenGeluOp,
             AtenGeluBackwardOp, AtenAddTensorOp, AtenMulTensorOp,
             AtenDivTensorOp, AtenSubTensorOp, AtenLerpTensorOp, AtenSigmoidOp,
             AtenExpOp, AtenMinimumOp, AtenMaximumOp, AtenToDtypeOp,
             AtenClampOp, AtenRsubScalarOp, AtenMulScalarOp, AtenLogOp,
             AtenErfOp, AtenSqrtOp, AtenFloorOp, AtenPowTensorScalarOp,
             AtenLog2Op, AtenRsqrtOp, AtenDivScalarOp, AtenAbsOp,
             AtenReciprocalOp, AtenBitwiseAndTensorOp, AtenGtScalarOp,
             AtenGeScalarOp, AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp,
             AtenWhereSelfOp, AtenCeilOp, AtenGtTensorOp, AtenEqTensorOp,
             AtenLtTensorOp, AtenSubScalarOp, AtenAddScalarOp, AtenThresholdOp,
             AtenThresholdBackwardOp, AtenCloneOp>(op))
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
    Value generic = createElementwiseLinalgGeneric(
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
    Value input = adaptor.self();
    Value target = adaptor.target();
    Value weight = adaptor.weight();

    int64_t reduction;
    if (!matchPattern(op.reduction(), m_TorchConstantInt(&reduction)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    // TODO: Incorporate the weight argument.
    if (!weight.getType().isa<mlir::torch::Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    Value ignoreIndex = adaptor.ignore_index();
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

    Value finalRes = createElementwiseLinalgGeneric(
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

      finalRes = torch_to_linalg::createReductionLinalgGeneric(
          rewriter, loc, finalRes, dimSet, /*keepDim=*/false,
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
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    Value runningMean = adaptor.running_mean();
    Value runningVar = adaptor.running_var();
    Value training = adaptor.training();
    Value eps = adaptor.eps();

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
    if (inputRank <= 2)
      return rewriter.notifyMatchFailure(
          op, "input should have rank larger than 2");

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
    SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
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

// For layernorm, the mean and standard-deviation are calculated separately over
// the last certain number dimensions which have to be of the shape specified by
// normalized_shape.
//
// The shapes of different parts are as the following:
// +-------------------+--------------------+
// |  meanAndVarShape  |   normalizedShape  |
// +-------------------+---------------------
// <------------+ inputShape +-------------->
// There are the following steps:
// Step 1. Check if all the arguments meet the requirements.
// Step 2. Common parts to be used for getting mean and var.
//         This includes elements count, affineMap and iteratorTypes.
// Step 3. Get mean.
// Step 4. Get rSTD.
// Step 5. Get layernorm.
namespace {
class ConvertAtenNativeLayerNormOp
    : public OpConversionPattern<AtenNativeLayerNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNativeLayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    Value eps = adaptor.eps();
    Value normalizedShape = op.normalized_shape();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Handle the None cases for the optional parameters:
    // weight, bias.
    if (failed(checkNotNone(rewriter, op, weight)) ||
        failed(checkNotNone(rewriter, op, bias)))
      return failure();

    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();
    Type elemTy = inputType.getElementType();

    // Step 1. Check if all the arguments meet the requirements.
    SmallVector<Value> normalizedShapeSizesTorchInt;
    if (!getListConstructElements(normalizedShape,
                                  normalizedShapeSizesTorchInt)) {
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented normalized_shape not"
                                         "constructed from ListConstruct");
    }
    SmallVector<Value> normalizedShapeSizesInt = getTypeConvertedValues(
        rewriter, loc, getTypeConverter(), normalizedShapeSizesTorchInt);
    int64_t normalizedShapeRank = normalizedShapeSizesInt.size();
    if (weightType.getRank() != normalizedShapeRank ||
        biasType.getRank() != normalizedShapeRank ||
        inputRank < normalizedShapeRank || normalizedShapeRank < 1)
      return rewriter.notifyMatchFailure(op, "Input or weight or bias shape or"
                                             "normalized shape not compatible");

    // Check all the dimensions match the normalized_shape
    int64_t meanAndVarShapeRank = inputRank - normalizedShapeSizesInt.size();
    for (auto en : enumerate((normalizedShapeSizesInt))) {
      auto index = en.index();
      auto inputDim =
          getDimOp(rewriter, loc, input, index + meanAndVarShapeRank);
      auto weightDim = getDimOp(rewriter, loc, weight, index);
      auto biasDim = getDimOp(rewriter, loc, bias, index);

      auto expectedSize = en.value();
      checkDimEqualHelper(rewriter, loc, inputDim, expectedSize);
      checkDimEqualHelper(rewriter, loc, weightDim, expectedSize);
      checkDimEqualHelper(rewriter, loc, biasDim, expectedSize);
    }

    // Get iterator types for input shape.
    SmallVector<StringRef> normalizedShapeIteratorTypes(
        normalizedShapeRank, getReductionIteratorTypeName());
    SmallVector<StringRef> meanAndVarIterationTypes(
        meanAndVarShapeRank, getParallelIteratorTypeName());
    SmallVector<StringRef> inputShapeIteratorTypes = meanAndVarIterationTypes;
    inputShapeIteratorTypes.append(normalizedShapeIteratorTypes);

    // Step 2. Common parts to be used for getting mean and var.

    // Get sizes and affineMaps needed for mean and var.
    AffineMap inputShapeAffineMap = rewriter.getMultiDimIdentityMap(inputRank);
    SmallVector<AffineExpr> meanAndVarShapeExprs;
    for (int i = 0; i < meanAndVarShapeRank; i++)
      meanAndVarShapeExprs.push_back(mlir::getAffineDimExpr(i, context));
    auto meanAndVarShapeAffineMap = AffineMap::get(
        /*dimCount=*/inputRank,
        /*symbolCount=*/0, meanAndVarShapeExprs, context);
    SmallVector<Value> meanAndVarShapeSizes =
        getTensorSizesUntilDim(rewriter, loc, input, meanAndVarShapeRank - 1);

    // Get number of elements to be used for calculating mean and var.
    Value elemCnts = normalizedShapeSizesInt[0];
    for (int i = 1; i < normalizedShapeRank; i++) {
      elemCnts = rewriter.create<arith::MulIOp>(loc, elemCnts,
                                                normalizedShapeSizesInt[i]);
    }
    Value elemCntsFloat =
        rewriter.create<arith::SIToFPOp>(loc, elemTy, elemCnts);

    // Helper to calculate mean and var.
    auto genMeanOrVarCalculation = [&](Value sumOrSquareSum) {
      SmallVector<AffineMap> indexingMaps(
          2, rewriter.getMultiDimIdentityMap(meanAndVarShapeRank));
      Value initShapeTensor = rewriter.create<linalg::InitTensorOp>(
          loc, meanAndVarShapeSizes, elemTy);
      return rewriter
          .create<linalg::GenericOp>(
              loc, initShapeTensor.getType(), sumOrSquareSum, initShapeTensor,
              /*indexingMaps=*/indexingMaps,
              /*iteratorTypes=*/meanAndVarIterationTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                Value sumOrSqureSum = args[0];
                Value result =
                    b.create<arith::DivFOp>(loc, sumOrSqureSum, elemCntsFloat);
                b.create<linalg::YieldOp>(loc, result);
              })
          .getResult(0);
    };

    // Step 3. Get mean.

    // Get sum to be used for calculating mean.
    SmallVector<AffineMap, 2> sumIndexingMaps = {
        inputShapeAffineMap,      // input
        meanAndVarShapeAffineMap, // output
    };
    auto initSumTensor =
        createZeroInitTensor(rewriter, loc, meanAndVarShapeSizes, elemTy);
    Value sum = rewriter
                    .create<linalg::GenericOp>(
                        loc, initSumTensor.getType(), input, initSumTensor,
                        /*indexingMaps=*/sumIndexingMaps,
                        /*iteratorTypes=*/inputShapeIteratorTypes,
                        [&](OpBuilder &b, Location loc, ValueRange args) {
                          Value input = args[0], sum = args[1];
                          Value result =
                              rewriter.create<arith::AddFOp>(loc, sum, input);
                          b.create<linalg::YieldOp>(loc, result);
                        })
                    .getResult(0);
    Value mean = genMeanOrVarCalculation(sum);

    // Step 4. Get rSTD.

    // Calculate squareSum for the layer.
    SmallVector<AffineMap> squareSumIndexingMaps{
        inputShapeAffineMap,
        meanAndVarShapeAffineMap,
        meanAndVarShapeAffineMap,
    };
    auto initSquareSumTensor =
        createZeroInitTensor(rewriter, loc, meanAndVarShapeSizes, elemTy);
    Value squareSum =
        rewriter
            .create<linalg::GenericOp>(
                loc, initSquareSumTensor.getType(), ValueRange{input, mean},
                initSquareSumTensor,
                /*indexingMaps=*/squareSumIndexingMaps,
                /*iteratorTypes=*/inputShapeIteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0], mean = args[1], squareSum = args[2];
                  Value sub = rewriter.create<arith::SubFOp>(loc, input, mean);
                  Value square = rewriter.create<arith::MulFOp>(loc, sub, sub);
                  Value result =
                      rewriter.create<arith::AddFOp>(loc, squareSum, square);
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    Value var = genMeanOrVarCalculation(squareSum);
    Value rSTDTensor = rewriter.create<linalg::InitTensorOp>(
        loc, meanAndVarShapeSizes, elemTy);
    SmallVector<AffineMap> rSTDIndexingMap(
        2, rewriter.getMultiDimIdentityMap(meanAndVarShapeRank));

    Value rSTD = rewriter
                     .create<linalg::GenericOp>(
                         loc, rSTDTensor.getType(), var, rSTDTensor,
                         rSTDIndexingMap, meanAndVarIterationTypes,
                         [&](OpBuilder &b, Location loc, ValueRange args) {
                           Value result =
                               calculateRSTD(b, loc, elemTy, eps, args[0]);
                           b.create<linalg::YieldOp>(loc, result);
                         })
                     .getResult(0);

    // Step 5. Get layernorm.

    // Get affineMap for normalized shape.
    SmallVector<AffineExpr> normalizedShapeExprs;
    for (int i = meanAndVarShapeRank; i < inputRank; i++)
      normalizedShapeExprs.push_back(mlir::getAffineDimExpr(i, context));
    auto normalizedShapeAffineMap = AffineMap::get(
        /*dimCount=*/inputRank,
        /*symbolCount=*/0, normalizedShapeExprs, context);
    auto inputSizes = getTensorSizes(rewriter, loc, input);
    Value initLayerNormTensor =
        rewriter.create<linalg::InitTensorOp>(loc, inputSizes, elemTy);
    SmallVector<AffineMap> indexingMaps(1, inputShapeAffineMap);
    indexingMaps.resize(3, meanAndVarShapeAffineMap);
    indexingMaps.resize(5, normalizedShapeAffineMap);
    indexingMaps.push_back(inputShapeAffineMap);
    SmallVector<StringRef> layerNormIterationTypes(
        inputRank, getParallelIteratorTypeName());
    Value layerNorm =
        rewriter
            .create<linalg::GenericOp>(
                loc, initLayerNormTensor.getType(),
                ValueRange{input, mean, rSTD, weight, bias},
                initLayerNormTensor,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/layerNormIterationTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0], mean = args[1], rSTD = args[2],
                        weight = args[3], bias = args[4];
                  Value result =
                      createLinalgPayloadCalculationForNormOpsWithRSTD(
                          b, loc, elemTy, input, mean, rSTD, eps, weight, bias);
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    SmallVector<int64_t> expandShape(inputRank, 1);
    for (int i = 0; i < meanAndVarShapeRank; i++) {
      // `mean` and `rstd` are not yet casted, so they will be having dynamic
      // shape. Hence to match them, for each dimension corresponding to `mean`
      // or `rstd` assign -1.
      expandShape[i] = -1;
    }
    auto expandShapeType = RankedTensorType::get(expandShape, elemTy);
    SmallVector<ReassociationIndices> reassociation(meanAndVarShapeRank);
    for (auto i : llvm::seq<int64_t>(0, meanAndVarShapeRank)) {
      reassociation[i].push_back(i);
      if (i == meanAndVarShapeRank - 1) {
        for (auto j : llvm::seq<int64_t>(0, normalizedShapeRank))
          reassociation[i].push_back(i + j + 1);
      }
    }
    Value meanResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, mean, reassociation);
    Value rSTDResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, rSTD, reassociation);
    Type layerNormResultType = getTypeConverter()->convertType(op.getType(0));
    Type meanResultType = getTypeConverter()->convertType(op.getType(1));
    Type rSTDResultType = getTypeConverter()->convertType(op.getType(2));
    Value layerNorm_ =
        rewriter.create<tensor::CastOp>(loc, layerNormResultType, layerNorm);
    Value mean_ =
        rewriter.create<tensor::CastOp>(loc, meanResultType, meanResult);
    Value var_ =
        rewriter.create<tensor::CastOp>(loc, rSTDResultType, rSTDResult);
    rewriter.replaceOp(op, {layerNorm_, mean_, var_});
    return success();
  }
};
} // namespace

// Given `grad_output`, `input`, `target`, `nll_loss_backward` is given by:
//   for i in range(0, len(input[0])):
//      for j in range(0, len(input[1])):
//          nll_loss_backward[i][j] = (j == target[i]) ? -grad_output[i] : 0
// TODO: `weight` and `reduction` operands are still to be taken care of.
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
    Value input = adaptor.self();
    Value target = adaptor.target();
    Value weight = adaptor.weight();
    Value gradOutput = adaptor.grad_output();

    int64_t reduction;
    if (!matchPattern(op.reduction(), m_TorchConstantInt(&reduction)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    // TODO: Handle reduction.
    if (reduction != torch_upstream::Reduction::None)
      return rewriter.notifyMatchFailure(
          op, "reduction along dimensions is not supported.");

    // TODO: Incorporate the weight argument.
    if (!weight.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    Value ignoreIndex = adaptor.ignore_index();
    Value ignoreIndexVal = castIntToIndex(rewriter, loc, ignoreIndex);

    unsigned inputRank = input.getType().cast<RankedTensorType>().getRank();
    unsigned targetRank = target.getType().cast<RankedTensorType>().getRank();

    // TODO: Cases with targetRank != 1 where `Mean` or `Sum` reduction is
    // required.
    if (inputRank != 2 || targetRank != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected  input and target to be rank 2 and 1 respectively");
    }
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();

    Type elementType = resultType.getElementType();

    // Given there is no reduction `grad_input` size is equal to `input` size.
    auto outputSize = getTensorSizes(rewriter, loc, input);
    Value initTensor0 =
        createZeroInitTensor(rewriter, loc, outputSize, elementType);
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    SmallVector<AffineExpr> targetExpr{rewriter.getAffineDimExpr(0)};
    SmallVector<AffineExpr> resultExpr{rewriter.getAffineDimExpr(0),
                                       rewriter.getAffineDimExpr(1)};
    SmallVector<StringRef> iteratorTypes{getParallelIteratorTypeName(),
                                         getParallelIteratorTypeName()};
    auto indexingMaps =
        AffineMap::inferFromExprList({targetExpr, targetExpr, resultExpr});
    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor0.getType(), ValueRange{target, gradOutput},
                initTensor0,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value indTarget = rewriter.create<arith::IndexCastOp>(
                      loc, rewriter.getIndexType(), args[0]);
                  Value indJ = rewriter.create<linalg::IndexOp>(loc, 1);

                  // The final result is given by:
                  // grad_input[i][j] = (j == target[i]) ? -grad_output[i] : 0
                  Value cmpEq = rewriter.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, indJ, indTarget);

                  // The target index shouldn't be equal to `ignoreIndex`.
                  Value cmpNe = rewriter.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::ne, ignoreIndexVal, indTarget);
                  Value finalPredicate =
                      rewriter.create<arith::AndIOp>(loc, cmpEq, cmpNe);
                  Value negate =
                      rewriter.create<arith::NegFOp>(loc, elementType, args[1]);
                  Value selectFinal = rewriter.create<arith::SelectOp>(
                      loc, finalPredicate, negate, zeroVal);
                  b.create<linalg::YieldOp>(loc, selectFinal);
                })
            .getResult(0);

    rewriter.replaceOp(op, finalRes);
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
                                                adaptor.operand());
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateUncategorizedPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<
      AtenTanhOp, AtenReluOp, AtenLeakyReluOp, AtenGeluOp, AtenGeluBackwardOp,
      AtenAddTensorOp, AtenMulTensorOp, AtenDivTensorOp, AtenSubTensorOp,
      AtenLerpTensorOp, AtenSigmoidOp, AtenMinimumOp, AtenMaximumOp,
      AtenToDtypeOp, AtenClampOp, AtenRsubScalarOp, AtenLogOp, AtenErfOp,
      AtenSqrtOp, AtenFloorOp, AtenCeilOp, AtenPowTensorScalarOp, AtenLog2Op,
      AtenRsqrtOp, AtenAbsOp, AtenReciprocalOp, AtenBitwiseAndTensorOp,
      AtenGtScalarOp, AtenGeScalarOp, AtenEqScalarOp, AtenLtScalarOp,
      AtenLeScalarOp, AtenWhereSelfOp, AtenGtTensorOp, AtenEqTensorOp,
      AtenLtTensorOp, AtenThresholdOp, AtenThresholdBackwardOp, AtenCloneOp>();
  patterns.add<ConvertElementwiseOp>(typeConverter, context);
  target.addIllegalOp<AtenNllLossForwardOp>();
  patterns.add<ConvertAtenNllLossForwardOp>(typeConverter, context);
  target.addIllegalOp<AtenBatchNormOp>();
  patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
  target.addIllegalOp<AtenNativeLayerNormOp>();
  patterns.add<ConvertAtenNativeLayerNormOp>(typeConverter, context);
  target.addIllegalOp<AtenNllLossBackwardOp>();
  patterns.add<ConvertAtenNllLossBackwardOp>(typeConverter, context);
  patterns.add<ConvertTensorStaticInfoCastOp>(typeConverter, context);
  target.addIllegalOp<TensorStaticInfoCastOp>();
}
