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

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// Aten max.dim (min.dim) lowering represents the MaxDimOp (MinDimOp) as an
// linalg.indexed_generic op, producing two output buffers.
//
// The first output buffer contains the maximum (minium) value found. It is
// initialized to the minimum (maximum) representable value of the input
// element type.
//
// The second output buffer contains the index of the found maximum (minimum)
// value. It is initialized to 0 and is resulting integer type.
//
// The indexed_generic op updates both the maximum (minimum) value and index
// if the current value exceeds the running max (min).
template <typename OpTy>
class ConvertAtenMinMaxDimOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpConversionPattern<OpTy>::getTypeConverter;

  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(std::is_same<OpTy, AtenMaxDimOp>() ||
                  std::is_same<OpTy, AtenMinDimOp>());
    constexpr bool isMax = std::is_same<OpTy, AtenMaxDimOp>();
    const llvm::StringRef opName = op->getName().getStringRef();

    Location loc = op.getLoc();
    Value input = adaptor.getSelf();
    RankedTensorType valResultType =
        getTypeConverter()
            ->convertType(op.getResult(0).getType())
            .template cast<RankedTensorType>();

    RankedTensorType idxResultType =
        this->getTypeConverter()
            ->convertType(op.getResult(1).getType())
            .template cast<RankedTensorType>();
    RankedTensorType inputType =
        input.getType().template cast<RankedTensorType>();
    Type idxElementType = idxResultType.getElementType();
    if (!idxElementType.isa<IntegerType>())
      return rewriter.notifyMatchFailure(
          op, opName + " to linalg.* requires integer-like result type");

    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(
          op, opName + " requires boolean value for keepdim");

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          op, opName + " to linalg.* requires int value for Dim");
    dim = toPositiveDim(dim, inputType.getRank());
    if (!isValidDim(dim, inputType.getRank()))
      return rewriter.notifyMatchFailure(op, "dim is not a valid dim");

    Type inElementType = inputType.getElementType();
    if (!inElementType.isa<mlir::FloatType>()) {
      if (inElementType.isa<mlir::IntegerType>()) {
        auto integerTy = op.getSelf()
                             .getType()
                             .template cast<BaseTensorType>()
                             .getDtype()
                             .template dyn_cast<mlir::IntegerType>();
        if (integerTy.isUnsigned())
          return rewriter.notifyMatchFailure(
              op, opName + " to linalg.* requires input element type "
                           "to be signed in case of integer");
      } else {
        return rewriter.notifyMatchFailure(
            op, opName + " to linalg.* requires Float or Integer "
                         "input element type");
      }
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

    // Second fill the output buffer for the running max or min.
    Value initTensorVal = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultShape), inElementType);

    Value fillValue;
    if (inElementType.isa<mlir::FloatType>()) {
      fillValue = rewriter.create<arith::ConstantOp>(
          loc,
          rewriter.getFloatAttr(
              inElementType,
              APFloat::getInf(
                  inElementType.cast<mlir::FloatType>().getFloatSemantics(),
                  /*Negative=*/isMax)));
    } else {
      auto width = inElementType.cast<mlir::IntegerType>().getWidth();
      auto init = isMax ? APSInt::getSignedMinValue(width)
                        : APSInt::getSignedMaxValue(width);
      fillValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(inElementType, init));
    }

    Value filledTensorVal =
        rewriter.create<linalg::FillOp>(loc, fillValue, initTensorVal).result();

    // Create the affine expressions that will be used to
    // iterate over the input and output tensors.
    // Here we also set the type of iterator: parallel or reduction.
    SmallVector<AffineExpr> exprs;
    SmallVector<utils::IteratorType> iteratorTypes;
    SmallVector<AffineExpr> resultExprs;
    for (auto size :
         llvm::enumerate(makeShapeTorchCompatible(inputType.getShape()))) {
      exprs.push_back(rewriter.getAffineDimExpr(size.index()));

      if (unsigned(dim) == size.index()) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
        // If `keepDim`, create affine map to the first element
        // in the current dimension.
        if (keepDim)
          resultExprs.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        iteratorTypes.push_back(utils::IteratorType::parallel);
        resultExprs.push_back(rewriter.getAffineDimExpr(size.index()));
      }
    }
    auto maps = AffineMap::inferFromExprList({exprs, resultExprs, resultExprs},
                                             rewriter.getContext());
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        ArrayRef<Type>({filledTensorVal.getType(), filledTensorIdx.getType()}),
        input, ValueRange({filledTensorVal, filledTensorIdx}), maps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value newValue = blockArgs[0];
          Value oldValue = blockArgs[1];
          Value oldIndex = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
              rewriter.create<linalg::IndexOp>(loc, dim));

          Value resultVal, predicate;
          if (inElementType.isa<mlir::FloatType>()) {
            arith::CmpFPredicate predType;
            if (isMax) {
              predType = arith::CmpFPredicate::OGT;
              resultVal = rewriter.create<arith::MaximumFOp>(
                  nestedLoc, newValue, oldValue);
            } else {
              predType = arith::CmpFPredicate::OLT;
              resultVal = rewriter.create<arith::MinimumFOp>(
                  nestedLoc, newValue, oldValue);
            }

            predicate = rewriter.create<arith::CmpFOp>(nestedLoc, predType,
                                                       newValue, oldValue);
          } else {
            arith::CmpIPredicate predType;
            if (isMax) {
              predType = arith::CmpIPredicate::sgt;
              resultVal = rewriter.create<arith::MaxSIOp>(nestedLoc, newValue,
                                                          oldValue);
            } else {
              predType = arith::CmpIPredicate::slt;
              resultVal = rewriter.create<arith::MinSIOp>(nestedLoc, newValue,
                                                          oldValue);
            }
            predicate = rewriter.create<arith::CmpIOp>(nestedLoc, predType,
                                                       newValue, oldValue);
          }
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultVal, resultIndex}));
        });

    // This cast is required to fix the shape in the case of keepDim=True
    Value valuesCast = rewriter.create<tensor::CastOp>(loc, valResultType,
                                                       linalgOp.getResult(0));
    Value idxCast = rewriter.create<tensor::CastOp>(loc, idxResultType,
                                                    linalgOp.getResult(1));
    rewriter.replaceOp(op, {valuesCast, idxCast});
    return success();
  }
};

} // namespace

static Value createInitElementForReduceOp(OpBuilder &b, Location loc,
                                          Operation *op, Type elementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op))
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));

  if (isa<AtenProdDimIntOp>(op)) {
    if (elementType.isa<mlir::FloatType>())
      return b.create<arith::ConstantOp>(loc, b.getFloatAttr(elementType, 1.0));
    else if (elementType.isa<mlir::IntegerType>())
      return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(elementType, 1));
  }

  if (isa<AtenMaxOp>(op)) {
    if (elementType.isa<mlir::FloatType>())
      return b.create<arith::ConstantOp>(
          loc, b.getFloatAttr(
                   elementType,
                   APFloat::getInf(
                       elementType.cast<mlir::FloatType>().getFloatSemantics(),
                       /*Negative=*/true)));
    else if (elementType.isa<mlir::IntegerType>() &&
             elementType.getIntOrFloatBitWidth() != 8)
      return b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(elementType,
                                APSInt::getSignedMinValue(
                                    elementType.getIntOrFloatBitWidth())));
  }

  if (isa<AtenMinOp>(op)) {
    if (elementType.isa<mlir::FloatType>())
      return b.create<arith::ConstantOp>(
          loc, b.getFloatAttr(
                   elementType,
                   APFloat::getInf(
                       elementType.cast<mlir::FloatType>().getFloatSemantics(),
                       /*Negative=*/false)));
    else if (elementType.isa<mlir::IntegerType>() &&
             elementType.getIntOrFloatBitWidth() != 8)
      return b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(elementType,
                                APSInt::getSignedMaxValue(
                                    elementType.getIntOrFloatBitWidth())));
  }

  if (isa<AtenLinalgVectorNormOp>(op) || isa<AtenFrobeniusNormDimOp>(op))
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));

  if (isa<AtenAllDimOp>(op)) {
    return b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
  }

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
  } else if (isa<AtenProdDimIntOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (resultElementType.isa<mlir::FloatType>())
      return b.create<arith::MulFOp>(loc, self, result);
    else if (resultElementType.isa<mlir::IntegerType>())
      return b.create<arith::MulIOp>(loc, self, result);
  } else if (auto max = dyn_cast<AtenMaxOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (resultElementType.isa<mlir::FloatType>())
      return b.create<arith::MaximumFOp>(loc, self, result);
    else if (resultElementType.isa<mlir::IntegerType>()) {
      IntegerType intType = max.getSelf()
                                .getType()
                                .cast<BaseTensorType>()
                                .getDtype()
                                .dyn_cast<mlir::IntegerType>();
      if (intType.isUnsigned())
        return b.create<arith::MaxUIOp>(loc, self, result);
      if (intType.isSigned())
        return b.create<arith::MaxSIOp>(loc, self, result);
    }
  } else if (auto min = dyn_cast<AtenMinOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (resultElementType.isa<mlir::FloatType>())
      return b.create<arith::MinimumFOp>(loc, self, result);
    else if (resultElementType.isa<mlir::IntegerType>()) {
      IntegerType intType = min.getSelf()
                                .getType()
                                .cast<BaseTensorType>()
                                .getDtype()
                                .dyn_cast<mlir::IntegerType>();
      if (intType.isUnsigned())
        return b.create<arith::MinUIOp>(loc, self, result);
      if (intType.isSigned())
        return b.create<arith::MinSIOp>(loc, self, result);
    }
  } else if (isa<AtenLinalgVectorNormOp>(op)) {
    // This creates payload for only the first of the two linalg.generic ops.
    // TODO: Short-circuit operations if `ord` is zero or one.
    Value elem = payloadArgs[0];
    Value result = payloadArgs[1];
    Value self = convertScalarToDtype(b, loc, elem, resultElementType);
    auto abs = b.create<math::AbsFOp>(loc, self);
    AtenLinalgVectorNormOp::Adaptor adaptor(operands);
    Value ord =
        convertScalarToDtype(b, loc, adaptor.getOrd(), resultElementType);
    auto pow = b.create<math::PowFOp>(loc, abs, ord);
    return b.create<arith::AddFOp>(loc, pow, result);
  } else if (isa<AtenFrobeniusNormDimOp>(op)) {
    Value elem = payloadArgs[0];
    Value result = payloadArgs[1];
    Value self = convertScalarToDtype(b, loc, elem, resultElementType);
    auto abs = b.create<math::AbsFOp>(loc, self);
    TypedAttr twoAttr = b.getFloatAttr(resultElementType, 2.0);
    auto ord = b.create<arith::ConstantOp>(loc, twoAttr);
    auto pow = b.create<math::PowFOp>(loc, abs, ord);
    return b.create<arith::AddFOp>(loc, pow, result);
  } else if (isa<AtenAllDimOp>(op)) {
    Value elem = payloadArgs[0];
    Value result = payloadArgs[1];
    Value self = convertScalarToDtype(b, loc, elem, resultElementType);
    return b.create<arith::MulIOp>(loc, self, result);
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
    opInfo.tensorOperand = adaptor.getSelf();
    auto inputType = opInfo.tensorOperand.getType().cast<RankedTensorType>();

    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&opInfo.keepDim)))
      return rewriter.notifyMatchFailure(op,
                                         "`keepdim` must be a constant bool");

    SmallVector<int64_t> dimList;
    int64_t dim;
    bool isNoneOrEmptyDimList =
        op.getDim().getType().template isa<Torch::NoneType>();
    if (matchPattern(op.getDim(), m_TorchListOfConstantInts(dimList))) {
      // Fix negative dimensions, if any, before adding to the list.
      for (int64_t dim : dimList) {
        dim = toPositiveDim(dim, inputType.getRank());
        // Drop invalid dimensions
        if (isValidDim(dim, inputType.getRank()))
          opInfo.dimSet.insert(dim);
      }
      if (dimList.empty())
        isNoneOrEmptyDimList = true;
    } else if (matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
      dim = toPositiveDim(dim, inputType.getRank());
      if (!isValidDim(dim, inputType.getRank()))
        return rewriter.notifyMatchFailure(
            op, "`dim` argument must be valid, invalid received.");
      opInfo.dimSet.insert(dim);
    } else if (!isNoneOrEmptyDimList) {
      return rewriter.notifyMatchFailure(
          op, "`dim` argument must be a constant int list or None");
    }
    if (isNoneOrEmptyDimList) {
      // If no dimensions were specified, reduce along all dimensions
      for (int64_t i = 0; i < inputType.getRank(); i++)
        opInfo.dimSet.insert(i);
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

    if (isa<AtenMaxOp, AtenMinOp, AtenSumOp>(op)) {
      opInfo.tensorOperand = operands[0];
      auto inputType = opInfo.tensorOperand.getType().cast<RankedTensorType>();

      // `AtenSumOp`, `AtenMaxOp`, and `AtenMinOp` each reduce along all the
      // dimensions of the input tensor.
      for (int64_t i = 0; i < inputType.getRank(); i++)
        opInfo.dimSet.insert(i);

      return opInfo;
    }

    if (auto sumOp = dyn_cast<AtenSumDimIntListOp>(op))
      return computeReductionOpInfoForDimVariantOp(sumOp, operands, rewriter);

    if (auto prodOp = dyn_cast<AtenProdDimIntOp>(op))
      return computeReductionOpInfoForDimVariantOp(prodOp, operands, rewriter);

    if (auto normOp = dyn_cast<AtenLinalgVectorNormOp>(op))
      return computeReductionOpInfoForDimVariantOp(normOp, operands, rewriter);

    if (auto normOp = dyn_cast<AtenFrobeniusNormDimOp>(op))
      return computeReductionOpInfoForDimVariantOp(normOp, operands, rewriter);

    if (auto allOp = dyn_cast<AtenAllDimOp>(op))
      return computeReductionOpInfoForDimVariantOp(allOp, operands, rewriter);

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
    TypedAttr oneAttr = rewriter.getFloatAttr(elemType, 1.0);
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
    if ((isa<AtenLinalgVectorNormOp>(op) || isa<AtenFrobeniusNormDimOp>(op)) &&
        !elemType.isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "only float types are valid for vector norm ops");
    if (isa<AtenAllDimOp>(op) && elemType.isa<mlir::IntegerType>() &&
        elemType.getIntOrFloatBitWidth() == 8)
      return rewriter.notifyMatchFailure(op, "uint8 is not supported");
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
          loc, elemType, normOp, adaptor.getOrd(), reduceOp, *opInfo, rewriter);
      if (failed(secondReduceOp))
        return secondReduceOp;
      reduceOp = *secondReduceOp;
    }

    // If it is aten.frobenius_norm.dim op, take the square root of reduceOp as
    // the final result
    if (auto normOp = dyn_cast<AtenFrobeniusNormDimOp>(op)) {
      auto halfAttr = rewriter.getFloatAttr(elemType, 0.5);
      auto exp = rewriter.create<arith::ConstantOp>(loc, halfAttr);
      reduceOp =
          createElementwiseExp(loc, elemType, exp, reduceOp, *opInfo, rewriter);
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
  patterns.add<ConvertAtenMinMaxDimOp<AtenMaxDimOp>>(typeConverter, context);
  target.addIllegalOp<AtenMinDimOp>();
  patterns.add<ConvertAtenMinMaxDimOp<AtenMinDimOp>>(typeConverter, context);
  target.addIllegalOp<AtenSumOp>();
  target.addIllegalOp<AtenSumDimIntListOp>();
  target.addIllegalOp<AtenProdDimIntOp>();
  target.addIllegalOp<AtenMaxOp>();
  target.addIllegalOp<AtenMinOp>();
  target.addIllegalOp<AtenAllDimOp>();
  target.addIllegalOp<AtenLinalgVectorNormOp>();
  target.addIllegalOp<AtenFrobeniusNormDimOp>();
  patterns.add<ConvertReductionOp>(typeConverter, context);
}
