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
#include "llvm/ADT/APSInt.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// Aten maxdim lowering represents the MaxDim op as an linalg.indexed_generic
// op, producing two output buffers.
//
// The first output buffer contains the maximum value found. It is initialized
// to the minimum representable value of the input element type.
//
// The second output buffer contains the index of the found maximum value. It is
// initialized to 0 and is resulting integer type.
//
// The indexed_generic op updates both the maximum value and index if the
// current value exceeds the running max.
class ConvertAtenMaxDimOp : public OpConversionPattern<AtenMaxDimOp> {
public:
  using OpConversionPattern<AtenMaxDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenMaxDimOp maxDimOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = maxDimOp.getLoc();
    Value input = adaptor.self();
    RankedTensorType valResultType =
        getTypeConverter()
            ->convertType(maxDimOp.getResult(0).getType())
            .cast<RankedTensorType>();
    RankedTensorType idxResultType =
        getTypeConverter()
            ->convertType(maxDimOp.getResult(1).getType())
            .cast<RankedTensorType>();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type idxElementType = idxResultType.getElementType();
    if (!idxElementType.isa<IntegerType>())
      return rewriter.notifyMatchFailure(
          maxDimOp,
          "aten.max_dim to linalg.* requires integer-like result type");

    bool keepDim = false;
    if (!matchPattern(maxDimOp.keepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(
          maxDimOp, "aten.max_dim requires boolean value for keepdim");

    int64_t dim;
    if (!matchPattern(maxDimOp.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          maxDimOp, "aten.max_dim to linalg.* requires int value for Dim");
    dim = toPositiveDim(dim, inputType.getRank());
    if (!isValidDim(dim, inputType.getRank()))
      return rewriter.notifyMatchFailure(maxDimOp, "dim is not a valid dim");

    Type inElementType = inputType.getElementType();
    if (!inElementType.isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          maxDimOp,
          "aten.max_dim to linalg.* requires Float input element type");
    }

    // Constant op to account for the reduction along dim.
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < inputType.getRank(); i++) {
      if (dim != i) {
        auto currentDimSize = rewriter.create<tensor::DimOp>(loc, input, i);
        resultShape.push_back(currentDimSize);
      } else if (keepDim)
        resultShape.push_back(c1);
    }
    // First fill the output buffer for the index.
    Value filledTensorIdx =
        createZeroInitTensor(rewriter, loc, resultShape, idxElementType);

    // Second fill the output buffer for the running max.
    Value initTensorMax =
        rewriter.create<linalg::InitTensorOp>(loc, resultShape, inElementType)
            .result();

    FloatAttr fillValueMaxAttr = rewriter.getFloatAttr(
        inElementType,
        APFloat::getLargest(
            inElementType.cast<mlir::FloatType>().getFloatSemantics(), true));

    Value fillValueMax =
        rewriter.create<arith::ConstantOp>(loc, fillValueMaxAttr);
    Value filledTensorMax =
        rewriter.create<linalg::FillOp>(loc, fillValueMax, initTensorMax)
            .result();

    // Create the affine expressions that will be used to
    // iterate over the input and output tensors.
    // Here we also set the type of iterator: parallel or reduction.
    SmallVector<AffineExpr> exprs;
    SmallVector<StringRef> iteratorTypes;
    SmallVector<AffineExpr> resultExprs;
    for (auto size : llvm::enumerate(inputType.getShape())) {
      exprs.push_back(rewriter.getAffineDimExpr(size.index()));

      if (unsigned(dim) == size.index()) {
        iteratorTypes.push_back(getReductionIteratorTypeName());
        // If `keepDim`, create affine map to the first element
        // in the current dimension.
        if (keepDim)
          resultExprs.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        iteratorTypes.push_back(getParallelIteratorTypeName());
        resultExprs.push_back(rewriter.getAffineDimExpr(size.index()));
      }
    }
    auto maps = AffineMap::inferFromExprList({exprs, resultExprs, resultExprs});
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        ArrayRef<Type>({filledTensorMax.getType(), filledTensorIdx.getType()}),
        input, ValueRange({filledTensorMax, filledTensorIdx}), maps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value newValue = blockArgs[0];
          Value oldValue = blockArgs[1];
          Value oldIndex = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
              rewriter.create<linalg::IndexOp>(loc, dim));

          Value predicate;
          if (inElementType.isa<mlir::FloatType>())
            predicate = rewriter.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);
          auto resultMax = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultMax, resultIndex}));
        });

    // This cast is required to fix the shape in the case of keepDim=True
    Value maxValuesCast = rewriter.create<tensor::CastOp>(
        loc, valResultType, linalgOp.getResult(0));
    Value maxIdxCast = rewriter.create<tensor::CastOp>(loc, idxResultType,
                                                       linalgOp.getResult(1));
    rewriter.replaceOp(maxDimOp, {maxValuesCast, maxIdxCast});
    return success();
  }
};
} // namespace

static Value createInitElementForReduceOp(OpBuilder &b, Location loc,
                                          Operation *op, Type elementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op))
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));

  if (isa<AtenMaxOp>(op)) {
    if (elementType.isa<mlir::FloatType>())
      return b.create<arith::ConstantOp>(
          loc, b.getFloatAttr(
                   elementType,
                   APFloat::getLargest(
                       elementType.cast<mlir::FloatType>().getFloatSemantics(),
                       /*Negative=*/true)));
    else if (elementType.isa<mlir::IntegerType>() &&
             elementType.getIntOrFloatBitWidth() != 8)
      return b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(elementType,
                                APSInt::getSignedMinValue(
                                    elementType.getIntOrFloatBitWidth())));
  }

  if (isa<AtenLinalgVectorNormOp>(op))
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));

  op->emitError("unimplemented lowering in createInitElementForReduceOp");
  return nullptr;
}

static Value createLinalgPayloadForReduceOp(OpBuilder &b, Location loc,
                                            ValueRange payloadArgs,
                                            Operation *op,
                                            ArrayRef<Value> operands,
                                            Type resultElementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (resultElementType.isa<mlir::FloatType>())
      return b.create<arith::AddFOp>(loc, self, result);
    else if (resultElementType.isa<mlir::IntegerType>())
      return b.create<arith::AddIOp>(loc, self, result);
  } else if (auto max = dyn_cast<AtenMaxOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (resultElementType.isa<mlir::FloatType>())
      return b.create<arith::MaxFOp>(loc, self, result);
    else if (resultElementType.isa<mlir::IntegerType>()) {
      IntegerType intType = max.self()
                                .getType()
                                .cast<BaseTensorType>()
                                .getDtype()
                                .dyn_cast<mlir::IntegerType>();
      if (intType.isUnsigned())
        return b.create<arith::MaxUIOp>(loc, self, result);
      if (intType.isSigned())
        return b.create<arith::MaxSIOp>(loc, self, result);
    }
  } else if (isa<AtenLinalgVectorNormOp>(op)) {
    // This creates payload for only the first of the two linalg.generic ops.
    // TODO: Short-circuit operations if `ord` is zero or one.
    Value elem = payloadArgs[0];
    Value result = payloadArgs[1];
    Value self = convertScalarToDtype(b, loc, elem, resultElementType);
    auto abs = b.create<math::AbsOp>(loc, self);
    AtenLinalgVectorNormOp::Adaptor adaptor(operands);
    Value ord = convertScalarToDtype(b, loc, adaptor.ord(), resultElementType);
    auto pow = b.create<math::PowFOp>(loc, abs, ord);
    return b.create<arith::AddFOp>(loc, pow, result);
  }
  op->emitError("unimplemented lowering in createLinalgPayloadForReduceOp");
  return nullptr;
}

namespace {
class ConvertReductionOp : public ConversionPattern {
private:
  /// Given a reduction operation that has the `keepdim` attribute and the
  /// (optional) `dim` attribute, return the source tensor operand and the
  /// literal values of the attributes or failure otherwise.
  template <typename T>
  FailureOr<torch_to_linalg::ReductionOpInfo>
  computeReductionOpInfoForDimVariantOp(
      T op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const {
    auto opInfo = torch_to_linalg::ReductionOpInfo{false, Value{}, {}};
    typename T::Adaptor adaptor(operands);
    opInfo.tensorOperand = adaptor.self();
    auto inputType = opInfo.tensorOperand.getType().cast<RankedTensorType>();

    if (!matchPattern(op.keepdim(), m_TorchConstantBool(&opInfo.keepDim)))
      return rewriter.notifyMatchFailure(op,
                                         "`keepdim` must be a constant bool");

    SmallVector<int64_t> dimList;
    if (matchPattern(op.dim(), m_TorchConstantIntList(dimList))) {
      // Fix negative dimensions, if any, before adding to the list.
      for (int64_t dim : dimList) {
        dim = toPositiveDim(dim, inputType.getRank());
        // Drop invalid dimensions
        if (isValidDim(dim, inputType.getRank()))
          opInfo.dimSet.insert(dim);
      }
    } else if (op.dim().getType().template isa<Torch::NoneType>()) {
      // If no dimensions were specified, reduce along all dimensions
      for (int64_t i = 0; i < inputType.getRank(); i++)
        opInfo.dimSet.insert(i);
    } else {
      return rewriter.notifyMatchFailure(
          op, "`dim` argument must be a constant int list or None");
    }

    return opInfo;
  }

  /// Given a reduction operation, return the source tensor operand and the
  /// literal values of the `keepdim` and `dim` attributes, if any, or failure
  /// otherwise.
  FailureOr<torch_to_linalg::ReductionOpInfo>
  computeReductionOpInfo(Operation *op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const {
    auto opInfo = torch_to_linalg::ReductionOpInfo{false, Value{}, {}};

    if (isa<AtenMaxOp, AtenSumOp>(op)) {
      opInfo.tensorOperand = operands[0];
      auto inputType = opInfo.tensorOperand.getType().cast<RankedTensorType>();

      // `AtenSumOp` and `AtenMaxOp` reduces along all the dimensions of the
      // input tensor.
      for (int64_t i = 0; i < inputType.getRank(); i++)
        opInfo.dimSet.insert(i);

      return opInfo;
    }

    if (auto sumOp = dyn_cast<AtenSumDimIntListOp>(op))
      return computeReductionOpInfoForDimVariantOp(sumOp, operands, rewriter);

    if (auto normOp = dyn_cast<AtenLinalgVectorNormOp>(op))
      return computeReductionOpInfoForDimVariantOp(normOp, operands, rewriter);

    return rewriter.notifyMatchFailure(op, "not a supported reduce op");
  }

  /// Generate a linalg.generic operation for pointwise exponentiation of each
  /// element.
  Value createElementwiseExp(Location loc, Type elemType, Value exponent,
                             Value inputTensor,
                             const torch_to_linalg::ReductionOpInfo &opInfo,
                             ConversionPatternRewriter &rewriter) const {
    bool err = false;
    auto powBodyBuilder = [&](OpBuilder &builder, Location loc,
                              ValueRange payloadArgs) {
      Value elem = convertScalarToDtype(builder, loc, payloadArgs[0], elemType);
      auto result = builder.create<math::PowFOp>(loc, elem, exponent);
      if (result)
        builder.create<linalg::YieldOp>(loc, Value{result});
      err = !result;
    };

    Value powOp = torch_to_linalg::createElementwiseLinalgGeneric(
        rewriter, loc, {inputTensor}, elemType, powBodyBuilder);
    return err ? Value{} : powOp;
  }

  FailureOr<Value> createSecondReductionForVectorNormOp(
      Location loc, Type elemType, AtenLinalgVectorNormOp op, Value ordOp,
      Value firstReduction, const torch_to_linalg::ReductionOpInfo &opInfo,
      ConversionPatternRewriter &rewriter) const {
    // Cast `ord` to float so that we can readily pass it math.powf.
    Value ordValue = convertScalarToDtype(rewriter, loc, ordOp, elemType);

    // TODO: Add support for ord = {0, +inf, -inf}.
    auto epsilon = 1e-5;
    auto ordLiteral = 0.0;
    if (matchPattern(ordValue, m_TorchConstantFloat(&ordLiteral)) &&
        fabs(ordLiteral) < epsilon)
      return rewriter.notifyMatchFailure(op, "unimplemented: L0 norm");

    if (std::isinf(ordLiteral))
      return rewriter.notifyMatchFailure(op, "unimplemented: ord = +/- inf");

    // Raise each summed value to the inverse of the order of the norm.
    Attribute oneAttr = rewriter.getFloatAttr(elemType, 1.0);
    auto oneValue = rewriter.create<arith::ConstantOp>(loc, oneAttr);
    auto inverseOrdValue =
        rewriter.create<arith::DivFOp>(loc, oneValue, ordValue);

    // Use the results of the first reduction operation from above to generate
    // a second reduction operation.
    Value reduceOp = createElementwiseExp(loc, elemType, inverseOrdValue,
                                          firstReduction, opInfo, rewriter);
    if (!reduceOp)
      return rewriter.notifyMatchFailure(
          op, "failed to create linalg.generic operation for element-wise "
              "exponentiation");

    return reduceOp;
  }

  /// Generate a linalg.generic operation for a reduction.
  Value createReductionOp(Location loc, Type elemType, Operation *op,
                          ArrayRef<Value> operands,
                          const torch_to_linalg::ReductionOpInfo &opInfo,
                          ConversionPatternRewriter &rewriter) const {
    bool err = false;
    auto reductionBodyBuilder = [&](OpBuilder &builder, Location loc,
                                    ValueRange payloadArgs) {
      Value result = createLinalgPayloadForReduceOp(builder, loc, payloadArgs,
                                                    op, operands, elemType);
      if (result)
        builder.create<linalg::YieldOp>(loc, result);
      err = !result;
    };

    Value initElem = createInitElementForReduceOp(rewriter, loc, op, elemType);
    Value reduceOp = torch_to_linalg::createReductionLinalgGeneric(
        rewriter, loc, opInfo, initElem, reductionBodyBuilder);
    return err ? Value{} : reduceOp;
  }

  /// Depending on the operation, check validity of the result's element type.
  LogicalResult
  validateReductionElementType(Operation *op, Type elemType,
                               ConversionPatternRewriter &rewriter) const {
    if (isa<AtenLinalgVectorNormOp>(op) && !elemType.isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "only float types are valid for vector norm ops");
    // No checks for all other reduction operations
    return success();
  }

public:
  ConvertReductionOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return rewriter.notifyMatchFailure(
          op, "invalid operand or result types to use with linalg on tensors");

    FailureOr<torch_to_linalg::ReductionOpInfo> opInfo =
        computeReductionOpInfo(op, operands, rewriter);
    if (failed(opInfo))
      return opInfo;

    Location loc = op->getLoc();
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    Type elemType = resultType.getElementType();
    LogicalResult elemTypeCheck =
        validateReductionElementType(op, elemType, rewriter);
    if (failed(elemTypeCheck))
      return elemTypeCheck;

    Value reduceOp =
        createReductionOp(loc, elemType, op, operands, *opInfo, rewriter);
    if (!reduceOp)
      return rewriter.notifyMatchFailure(
          op, "failed to create linalg.generic operation for reduction");

    // If this is aten.linalg_vector_norm op, then we need to generate another
    // linalg.generic op that references the first linalg.generic op.
    if (auto normOp = dyn_cast<AtenLinalgVectorNormOp>(op)) {
      AtenLinalgVectorNormOp::Adaptor adaptor(operands);
      FailureOr<Value> secondReduceOp = createSecondReductionForVectorNormOp(
          loc, elemType, normOp, adaptor.ord(), reduceOp, *opInfo, rewriter);
      if (failed(secondReduceOp))
        return secondReduceOp;
      reduceOp = *secondReduceOp;
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, reduceOp);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateReductionPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenMaxDimOp>();
  patterns.add<ConvertAtenMaxDimOp>(typeConverter, context);
  target.addIllegalOp<AtenSumOp>();
  target.addIllegalOp<AtenSumDimIntListOp>();
  target.addIllegalOp<AtenMaxOp>();
  target.addIllegalOp<AtenLinalgVectorNormOp>();
  patterns.add<ConvertReductionOp>(typeConverter, context);
}
