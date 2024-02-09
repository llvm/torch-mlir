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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

static void getZeroPoint(Value value, Value &zeropoint) {
  if (auto make = value.getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
    zeropoint = make.getZeroPoint();
  }
}

static Value transposeValue(Location loc, Value value, ArrayRef<int64_t> perms,
                            PatternRewriter &rewriter) {
  auto valueTy = value.getType().cast<RankedTensorType>();
  auto inShape = valueTy.getShape();
  llvm::SmallVector<int64_t> outShape;
  llvm::SmallVector<Value> dynDims;
  for (size_t i = 0; i < perms.size(); ++i) {
    outShape.push_back(inShape[perms[i]]);
    if (ShapedType::isDynamic(inShape[perms[i]])) {
      dynDims.push_back(rewriter.create<tensor::DimOp>(loc, value, perms[i]));
    }
  }

  auto outTy = RankedTensorType::get(outShape, valueTy.getElementType());
  Value empty = rewriter.create<tensor::EmptyOp>(loc, outTy, dynDims);
  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, value, empty, perms)
          ->getResult(0);
  return transpose;
}

class ConvertAtenMmOp : public OpConversionPattern<AtenMmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.getSelf();
    Value rhs = adaptor.getMat2();

    // A user can write an errorneous program where `aten.mm` is in fact called
    // with operands of invalid rank or dtype. We cannot convert to linalg in
    // this case or we will get a verifier error, which corresponds to breaking
    // of *internal* compiler invariants, and for a user manifests as a compiler
    // crash in the worst case (such as we try to canonicalize/fold/print the
    // invalid op before the verifier gets to see it -- also release builds of a
    // mature compiler usually have the verifier turned off for compile time
    // reasons).
    //
    // The compiler cannot crash even if the user wrote an erroneous program!
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.mm to be rank 2");
    }

    ValueTensorType lhsTorchType =
        op.getSelf().getType().cast<ValueTensorType>();
    ValueTensorType rhsTorchType =
        op.getMat2().getType().cast<ValueTensorType>();

    Value lhsZeroPoint, rhsZeroPoint;
    getZeroPoint(op.getSelf(), lhsZeroPoint);
    getZeroPoint(op.getMat2(), rhsZeroPoint);

    if (static_cast<bool>(lhsZeroPoint) != static_cast<bool>(lhsZeroPoint)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported: aten.mm with mixed quantization");
    }

    if (lhsTorchType.getDtype() != rhsTorchType.getDtype()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported: aten.mm with different input element types");
    }

    bool isUnsigned = torch_to_linalg::isUnsignedTorchType(lhsTorchType);
    if (lhsZeroPoint && isUnsigned) {
      return rewriter.notifyMatchFailure(
          op, "unsupported: unsigned quantized matmul not supported");
    }

    Value lhsDim0 = rewriter.create<tensor::DimOp>(loc, lhs, 0);
    Value rhsDim1 = rewriter.create<tensor::DimOp>(loc, rhs, 1);

    if (!isAssumingStrictSymbolicShapes(rewriter)) {
      Value lhsDim1 = rewriter.create<tensor::DimOp>(loc, lhs, 1);
      Value rhsDim0 = rewriter.create<tensor::DimOp>(loc, rhs, 0);
      Value contractingDimEqual = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, lhsDim1, rhsDim0);
      rewriter.create<cf::AssertOp>(
          loc, contractingDimEqual,
          rewriter.getStringAttr(
              "mismatching contracting dimension for torch.aten.mm"));
    }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value zeroFill = createZeroInitTensor(
        rewriter, loc, ValueRange{lhsDim0, rhsDim1}, elementType);

    Value matmul;
    if (lhsZeroPoint && !isUnsigned) {
      lhsZeroPoint = typeConverter->materializeTargetConversion(
          rewriter, loc,
          getTypeConverter()->convertType(lhsZeroPoint.getType()),
          lhsZeroPoint);
      rhsZeroPoint = typeConverter->materializeTargetConversion(
          rewriter, loc,
          getTypeConverter()->convertType(rhsZeroPoint.getType()),
          rhsZeroPoint);
      lhsZeroPoint = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getI32Type(), lhsZeroPoint);
      rhsZeroPoint = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getI32Type(), rhsZeroPoint);
      matmul =
          rewriter
              .create<linalg::QuantizedMatmulOp>(
                  loc, zeroFill.getType(),
                  ValueRange{lhs, rhs, lhsZeroPoint, rhsZeroPoint}, zeroFill)
              .getResult(0);
    } else if (isUnsigned) {
      matmul = rewriter
                   .create<linalg::MatmulUnsignedOp>(
                       loc, zeroFill.getType(), ValueRange{lhs, rhs}, zeroFill)
                   .getResult(0);
    } else {
      matmul = rewriter
                   .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                             ValueRange{lhs, rhs}, zeroFill)
                   .getResult(0);
    }
    // When constructed with just dynamic sizes, EmptyOp will have a result
    // type which has all `?`'s for dimensions, which might not be the result
    // type of `op`. The constraints on later linalg ops means that the result
    // of the MatmulOp will have this type too. So cast it to the desired type
    // so that in the end we have the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenFlipOp : public OpConversionPattern<AtenFlipOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenFlipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    MLIRContext *context = op.getContext();
    Value self = adaptor.getSelf();
    auto selfRank =
        adaptor.getSelf().getType().cast<RankedTensorType>().getRank();
    Type elementType =
        adaptor.getSelf().getType().cast<RankedTensorType>().getElementType();
    Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    SmallVector<int64_t> axis;
    if (!matchPattern(adaptor.getDims(), m_TorchListOfConstantInts(axis)))
      return rewriter.notifyMatchFailure(op,
                                         "only constant dim lists supported");
    for (unsigned i = 0, e = axis.size(); i < e; i++) {
      axis[i] = toPositiveDim(axis[i], selfRank);
      if (!isValidDim(axis[i], selfRank)) {
        return rewriter.notifyMatchFailure(op, "axis is statically invalid");
      }
    }

    // Only used to calculate flipped values, i.e. those on the flip axes. Other
    // dims won't be used.
    SmallVector<Value> dims = getTensorSizes(rewriter, loc, self);
    for (auto flipDim : axis)
      dims[flipDim] = rewriter.create<arith::SubIOp>(loc, dims[flipDim], c1);

    Value initTensor = createZeroInitTensor(
        rewriter, loc, getTensorSizes(rewriter, loc, self), elementType);

    SmallVector<utils::IteratorType> iteratorTypes(
        selfRank, utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(selfRank, context));
    Value flipped =
        rewriter
            .create<linalg::GenericOp>(
                loc, self.getType(), self, initTensor, indexingMaps,
                iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  SmallVector<Value> indices;
                  for (auto i = 0; i < selfRank; i++)
                    indices.push_back(b.create<linalg::IndexOp>(loc, i));
                  for (auto flipDim : axis) {
                    indices[flipDim] = b.create<arith::SubIOp>(
                        loc, dims[flipDim], indices[flipDim]);
                  }
                  Value res = b.create<tensor::ExtractOp>(loc, self, indices)
                                  .getResult();
                  b.create<linalg::YieldOp>(loc, res);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, self.getType(), flipped);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenMatmulOp : public OpConversionPattern<AtenMatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.getSelf();
    Value rhs = adaptor.getOther();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto rhsType = rhs.getType().cast<RankedTensorType>();

    // Get the rank of both matrix.
    unsigned lhsRank = lhsType.getRank();
    unsigned rhsRank = rhsType.getRank();

    Type newResultType = getTypeConverter()->convertType(op.getType());
    auto resultType = newResultType.cast<RankedTensorType>();
    Type elementType = resultType.getElementType();

    // The different cases of torch_matmul op is mentioned here:
    // https://pytorch.org/docs/stable/generated/torch.matmul.html

    // First Case: Dot Product.
    if (lhsRank == 1 && rhsRank == 1) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);

      checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

      Value zeroTensor = createZeroInitTensor(rewriter, loc, {}, elementType);
      Value dotProd =
          rewriter
              .create<linalg::DotOp>(loc, zeroTensor.getType(),
                                     ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, dotProd);
      return success();
    }

    // Second Case: Vec-Mat Multiplication.
    if (lhsRank == 1 && rhsRank == 2) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
      checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

      Value zeroTensor =
          createZeroInitTensor(rewriter, loc, ValueRange{rhsDim1}, elementType);
      Value matmul =
          rewriter
              .create<linalg::VecmatOp>(loc, zeroTensor.getType(),
                                        ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Third Case: Matrix-Vec Multiplication.
    if (lhsRank == 2 && rhsRank == 1) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      Value zeroTensor =
          createZeroInitTensor(rewriter, loc, ValueRange{lhsDim0}, elementType);
      Value matmul =
          rewriter
              .create<linalg::MatvecOp>(loc, zeroTensor.getType(),
                                        ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Fourth Case: Vec-Vec Multiplication.
    if (lhsRank == 2 && rhsRank == 2) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      Value zeroTensor = createZeroInitTensor(
          rewriter, loc, ValueRange{lhsDim0, rhsDim1}, elementType);
      Value matmul =
          rewriter
              .create<linalg::MatmulOp>(loc, zeroTensor.getType(),
                                        ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Fifth Case: Batch-Matrix Multiplication.
    // TODO: Handle batch matrix multiplication when one of the matrix is unity
    // rank and the other has batch dimension.
    if (lhsRank > 1 && rhsRank > 1) {
      unsigned maxRank = std::max(lhsRank, rhsRank);
      unsigned minRank = std::min(lhsRank, rhsRank);
      unsigned batchRank = maxRank - 2;

      // At least one of the matrix must have rank greater than 2.
      if (batchRank <= 0) {
        return rewriter.notifyMatchFailure(op, "expected batch dimensions");
      }

      // The `broadcastedBatchShape` contains batch dimensions of the resultant
      // matrix.
      SmallVector<Value> broadcastedBatchShape(batchRank);
      Value maxRankMatrix = (lhsRank > rhsRank) ? lhs : rhs;
      Value maxDim;
      // Compute broadcasted batch dimensions if the batch dimensions of
      // the matrices are broadcastable.
      for (unsigned i = 1; i <= batchRank; i++) {
        if (i <= minRank - 2) {
          Value lhsDim = getDimOp(rewriter, loc, lhs, lhsRank - 2 - i);
          Value rhsDim = getDimOp(rewriter, loc, rhs, rhsRank - 2 - i);
          maxDim = rewriter.createOrFold<arith::MaxUIOp>(loc, lhsDim, rhsDim);
        } else {
          maxDim = getDimOp(rewriter, loc, maxRankMatrix, maxRank - 2 - i);
        }
        broadcastedBatchShape[batchRank - i] = maxDim;
      }

      Value lhsDim0 = getDimOp(rewriter, loc, lhs, lhsRank - 2);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, lhsRank - 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, rhsRank - 2);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, rhsRank - 1);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      // Compute broadcasted shape of both the matrices in integer format.
      SmallVector<Value> lhsBroadcastToShape(broadcastedBatchShape);
      lhsBroadcastToShape.push_back(lhsDim0);
      lhsBroadcastToShape.push_back(lhsDim1);
      SmallVector<Value> rhsBroadcastToShape(broadcastedBatchShape);
      rhsBroadcastToShape.push_back(rhsDim0);
      rhsBroadcastToShape.push_back(rhsDim1);
      for (unsigned i = 0; i < maxRank; i++) {
        lhsBroadcastToShape[i] =
            castIndexToInt64(rewriter, loc, lhsBroadcastToShape[i]);
        rhsBroadcastToShape[i] =
            castIndexToInt64(rewriter, loc, rhsBroadcastToShape[i]);
      }

      // Broadcast the batch dimensions of both the matrices.
      Value broadcastedLhs, broadcastedRhs;
      // TODO: Improve usage of static shape information.
      SmallVector<int64_t> lhsTargetShape(lhsBroadcastToShape.size(),
                                          ShapedType::kDynamic);
      auto lhsBroadcastType = RankedTensorType::get(
          lhsTargetShape, lhsType.getElementType(), lhsType.getEncoding());
      if (failed(torch_to_linalg::broadcastToGivenShape(
              op, rewriter, lhs, lhsBroadcastToShape, lhsBroadcastType,
              broadcastedLhs))) {
        return rewriter.notifyMatchFailure(
            op, "unable to perform broadcast operation");
      }
      SmallVector<int64_t> rhsTargetShape(rhsBroadcastToShape.size(),
                                          ShapedType::kDynamic);
      auto rhsBroadcastType = RankedTensorType::get(
          rhsTargetShape, rhsType.getElementType(), rhsType.getEncoding());
      if (failed(torch_to_linalg::broadcastToGivenShape(
              op, rewriter, rhs, rhsBroadcastToShape, rhsBroadcastType,
              broadcastedRhs))) {
        return rewriter.notifyMatchFailure(
            op, "unable to perform broadcast operation");
      }

      if (maxRank == 3) {
        Value zeroTensor = createZeroInitTensor(
            rewriter, loc,
            ValueRange{broadcastedBatchShape[0], lhsDim0, rhsDim1},
            elementType);
        Value matmul =
            rewriter
                .create<linalg::BatchMatmulOp>(
                    loc, zeroTensor.getType(),
                    ValueRange{broadcastedLhs, broadcastedRhs}, zeroTensor)
                .getResult(0);
        rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
        return success();
      }

      // Check if the result of the matrix multiplication has more than one
      // dynamic batch dimensions.
      SmallVector<int64_t> batchDimsInt =
          makeShapeTorchCompatible(resultType.getShape());
      batchDimsInt.pop_back();
      batchDimsInt.pop_back();
      bool multipleDynamicBatchDims =
          llvm::count(batchDimsInt, kUnknownSize) > 1;

      // TODO: Lowering to `linalg.BatchMatmul` is only possible when there is
      // at most one dynamic batch dimension due to limited support of the
      // `tensor.ExpandShape` op.
      if (!multipleDynamicBatchDims) {
        // Collapse the batch dimensions into one dimension. The resultant rank
        // will always be 3.
        SmallVector<ReassociationIndices> reassociation(3);
        for (unsigned i = 0, j = 0; i < maxRank; i++) {
          if (i >= batchRank)
            j++;
          reassociation[j].push_back(i);
        }
        Value collapsedLhs = rewriter.create<tensor::CollapseShapeOp>(
            op->getLoc(), broadcastedLhs, reassociation);
        Value collapsedRhs = rewriter.create<tensor::CollapseShapeOp>(
            op->getLoc(), broadcastedRhs, reassociation);

        // Compute the result shape after collapsing the batch dimensions.
        SmallVector<Value> collapsedResultShape;
        collapsedResultShape.push_back(broadcastedBatchShape[0]);
        for (unsigned i = 1; i < batchRank; i++) {
          collapsedResultShape[0] = rewriter.createOrFold<arith::MulIOp>(
              loc, collapsedResultShape[0], broadcastedBatchShape[i]);
        }
        collapsedResultShape.push_back(lhsDim0);
        collapsedResultShape.push_back(rhsDim1);
        SmallVector<OpFoldResult> updatedCollapseResultShape =
            getAsOpFoldResult(collapsedResultShape);

        Value initTensor = rewriter.create<tensor::EmptyOp>(
            loc, updatedCollapseResultShape, elementType);
        Value c0 = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(elementType));
        Value zeroTensor =
            rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);

        Value batchMatMul =
            rewriter
                .create<linalg::BatchMatmulOp>(
                    loc, zeroTensor.getType(),
                    ValueRange{collapsedLhs, collapsedRhs}, zeroTensor)
                .getResult(0);
        Value expandResult = rewriter.create<tensor::ExpandShapeOp>(
            loc, resultType, batchMatMul, reassociation);
        rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
                                                    expandResult);
        return success();
      }

      SmallVector<AffineExpr> lhsExpr;
      SmallVector<AffineExpr> rhsExpr;
      SmallVector<AffineExpr> outExpr;
      SmallVector<utils::IteratorType> iteratorTypes(
          batchRank, utils::IteratorType::parallel);
      for (unsigned i = 0; i < batchRank; i++) {
        lhsExpr.push_back(rewriter.getAffineDimExpr(i));
        rhsExpr.push_back(rewriter.getAffineDimExpr(i));
        outExpr.push_back(rewriter.getAffineDimExpr(i));
      }
      lhsExpr.insert(lhsExpr.end(), {rewriter.getAffineDimExpr(batchRank),
                                     rewriter.getAffineDimExpr(batchRank + 1)});
      rhsExpr.insert(rhsExpr.end(), {rewriter.getAffineDimExpr(batchRank + 1),
                                     rewriter.getAffineDimExpr(batchRank + 2)});
      outExpr.insert(outExpr.end(), {rewriter.getAffineDimExpr(batchRank),
                                     rewriter.getAffineDimExpr(batchRank + 2)});

      SmallVector<Value> resultShape(broadcastedBatchShape);
      resultShape.insert(resultShape.end(), {lhsDim0, rhsDim1});
      Value zeroTensor =
          createZeroInitTensor(rewriter, loc, resultShape, elementType);
      auto indexingMaps = AffineMap::inferFromExprList(
          {lhsExpr, rhsExpr, outExpr}, rewriter.getContext());
      iteratorTypes.insert(iteratorTypes.end(),
                           {utils::IteratorType::parallel,
                            utils::IteratorType::reduction,
                            utils::IteratorType::parallel});

      Value finalRes =
          rewriter
              .create<linalg::GenericOp>(
                  loc, zeroTensor.getType(),
                  ValueRange{broadcastedLhs, broadcastedRhs}, zeroTensor,
                  /*indexingMaps=*/indexingMaps,
                  /*iteratorTypes=*/iteratorTypes,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value l = args[0], r = args[1], res = args[2];
                    Value mul = b.create<arith::MulFOp>(loc, l, r);
                    Value add = b.create<arith::AddFOp>(loc, mul, res);
                    b.create<linalg::YieldOp>(loc, add);
                  })
              .getResult(0);

      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, finalRes);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
class ConvertAtenBmmOp : public OpConversionPattern<AtenBmmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value lhs = adaptor.getSelf();
    Value rhs = adaptor.getMat2();
    RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();
    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type resultElementType =
        newResultType.cast<RankedTensorType>().getElementType();
    Type lhsElementType = lhsType.cast<RankedTensorType>().getElementType();
    Type rhsElementType = rhsType.cast<RankedTensorType>().getElementType();

    if (lhsType.getRank() != 3 || rhsType.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.bmm to be rank 3");
    }

    // Convert the inputs element type equivalent to the result' element type.
    if (lhsElementType != rhsElementType) {
      if (lhsElementType != resultElementType) {
        // True if the lhs element type is not equal to the result' element
        // type.
        lhs = torch_to_linalg::convertTensorToElementType(rewriter, loc, lhs,
                                                          resultElementType);
      } else {
        // True if the rhs element type is not equal to the result' element
        // type.
        rhs = torch_to_linalg::convertTensorToElementType(rewriter, loc, rhs,
                                                          resultElementType);
      }
    }

    Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
    Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
    Value lhsDim2 = getDimOp(rewriter, loc, lhs, 2);
    Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
    Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
    Value rhsDim2 = getDimOp(rewriter, loc, rhs, 2);

    // Check the batch numbers are equal.
    checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

    // Check the matrixs shapes are valid for mulplication.
    checkDimEqualHelper(rewriter, loc, lhsDim2, rhsDim1);

    Value initTensor0 = createZeroInitTensor(
        rewriter, loc, ValueRange{lhsDim0, lhsDim1, rhsDim2},
        resultElementType);

    Value bmm =
        rewriter
            .create<linalg::BatchMatmulOp>(loc, initTensor0.getType(),
                                           ValueRange{lhs, rhs}, initTensor0)
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, bmm);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenConvolutionOp : public OpConversionPattern<AtenConvolutionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.getInput();   /* in form of N*C*H*W */
    Value weight = adaptor.getWeight(); /* in form of F*C*H*W */
    Value bias = adaptor.getBias();
    auto resultTy = op.getType().cast<ValueTensorType>();

    Value inputZp, weightZp;
    if (auto make = op.getInput()
                        .getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
      input = make.getSelf();
      inputZp = make.getZeroPoint();
      input = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(input.getType()), input);
      inputZp = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(inputZp.getType()),
          inputZp);
    }

    if (auto make = op.getWeight()
                        .getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
      weight = make.getSelf();
      weightZp = make.getZeroPoint();

      weight = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(weight.getType()), weight);
      weightZp = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(weightZp.getType()),
          weightZp);
    }

    if (static_cast<bool>(inputZp) != static_cast<bool>(weightZp)) {
      return rewriter.notifyMatchFailure(
          op, "lhs and rhs of convolution must either be both int or fp");
    }

    if (inputZp && weightZp && !isa<Torch::NoneType>(bias.getType())) {
      auto biasDTy = bias.getType().cast<RankedTensorType>().getElementType();
      if (!biasDTy.isInteger(32)) {
        return rewriter.notifyMatchFailure(
            op, "quantized result ty should be i32 accumulator");
      }
    }

    bool transposed = true;
    if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant transposed supported");

    auto inputDTy = input.getType().cast<RankedTensorType>().getElementType();
    auto weightDTy = weight.getType().cast<RankedTensorType>().getElementType();
    auto resultDTy = resultTy.toBuiltinTensor().getElementType();

    if (!inputDTy.isa<mlir::FloatType, mlir::IntegerType>() ||
        !weightDTy.isa<mlir::FloatType, mlir::IntegerType>() ||
        !resultDTy.isa<mlir::FloatType, mlir::IntegerType>())
      return op.emitError("unimplemented: non-fp not-int type");
    size_t inRank = input.getType().cast<RankedTensorType>().getRank();
    size_t numSpatialDims = inRank - 2;
    if (numSpatialDims < 1 || numSpatialDims > 3)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only 1d-3d convolution currently supported");

    Type intType = IntegerType::get(context, 64);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(loc, intType, v);
    };

    SmallVector<Value> paddingIntValues;
    if (!getListConstructElements(op.getPadding(), paddingIntValues))
      return rewriter.notifyMatchFailure(
          op, "only support padding from a list construct");
    paddingIntValues = getTypeConvertedValues(rewriter, loc, getTypeConverter(),
                                              paddingIntValues);
    SmallVector<Value> outputPaddingIntValues;
    if (!getListConstructElements(op.getOutputPadding(),
                                  outputPaddingIntValues))
      return rewriter.notifyMatchFailure(
          op, "only support output_padding from a list construct");
    outputPaddingIntValues = getTypeConvertedValues(
        rewriter, loc, getTypeConverter(), outputPaddingIntValues);
    SmallVector<int64_t> strideInts;
    if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");
    SmallVector<int64_t> dilationInts;
    if (!matchPattern(op.getDilation(),
                      m_TorchListOfConstantInts(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    Value inBatch = getDimOp(rewriter, loc, input, 0);
    Value inChannels = getDimOp(rewriter, loc, input, 1);
    SmallVector<Value> inDims;
    for (size_t i = 2; i < inRank; i++)
      inDims.push_back(getDimOp(rewriter, loc, input, i));
    Value weightBatch = getDimOp(rewriter, loc, weight, 0);
    Value weightChannels = getDimOp(rewriter, loc, weight, 1);
    SmallVector<Value> weightDims;
    for (size_t i = 2; i < inRank; i++)
      weightDims.push_back(getDimOp(rewriter, loc, weight, i));

    // Checks for valid group size
    int64_t groupSize;
    if (!matchPattern(op.getGroups(), m_TorchConstantInt(&groupSize)))
      return rewriter.notifyMatchFailure(op,
                                         "only constant group size supported.");
    Value groups = castIntToIndex(rewriter, loc, adaptor.getGroups());

    auto validate = [&](Value toValidate, std::string err) {
      Value c0 =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      Value inputValid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, c0,
          rewriter.create<arith::RemSIOp>(loc, toValidate, groups));
      rewriter.create<cf::AssertOp>(loc, inputValid,
                                    rewriter.getStringAttr(err));
    };
    validate(inChannels,
             "invalid: groups must divide input channel size evenly.");
    validate(weightBatch,
             "invalid: groups must divide weight batch size evenly.");
    SmallVector<Value> dilationIntValues =
        getAsConstantIntValues(rewriter, loc, dilationInts);
    SmallVector<Value> strideIntValues =
        getAsConstantIntValues(rewriter, loc, strideInts);

    // Pad the input tensor according to padding.
    SmallVector<Value> outDims{inBatch, weightBatch};
    Value paddedInput;
    if (transposed) {
      if (!inputDTy.isa<mlir::FloatType>() ||
          !weightDTy.isa<mlir::FloatType>() ||
          !resultDTy.isa<mlir::FloatType>())
        return rewriter.notifyMatchFailure(
            op, "transpose does not support non-fp type yet");

      Value c0 =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      Value c1 =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      Value c2 =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(2));

      // Transpose and flip weight
      SmallVector<Value> weightInitDims = getTensorSizes(rewriter, loc, weight);
      std::iter_swap(weightInitDims.begin(), weightInitDims.begin() + 1);
      outDims[1] = weightInitDims[0];
      Value weightInitTensor =
          createZeroInitTensor(rewriter, loc, weightInitDims, weightDTy);
      SmallVector<utils::IteratorType> iteratorTypes(
          inRank, utils::IteratorType::parallel);
      SmallVector<AffineMap> indexingMaps{
          AffineMap::getMultiDimIdentityMap(inRank, context)};
      weight = rewriter
                   .create<linalg::GenericOp>(
                       loc, weightInitTensor.getType(), ValueRange{},
                       weightInitTensor, indexingMaps, iteratorTypes,
                       [&](OpBuilder &b, Location loc, ValueRange args) {
                         SmallVector<Value> indices;
                         for (size_t i = 0; i < inRank; i++)
                           indices.push_back(b.create<linalg::IndexOp>(loc, i));
                         std::iter_swap(indices.begin(), indices.begin() + 1);
                         // Flip only the spatial dimensions (from 2 to inRank)
                         for (size_t flipDim = 2; flipDim < inRank; flipDim++) {
                           indices[flipDim] = b.create<arith::SubIOp>(
                               loc,
                               b.create<arith::SubIOp>(
                                   loc, weightInitDims[flipDim], c1),
                               indices[flipDim]);
                         }
                         Value res =
                             b.create<tensor::ExtractOp>(loc, weight, indices)
                                 .getResult();
                         b.create<linalg::YieldOp>(loc, res);
                       })
                   .getResult(0);

      // Calculate padded input size, allocate tensor
      SmallVector<Value> outerSizes{inBatch, inChannels};
      SmallVector<Value> innerSizes{inBatch, inChannels};
      SmallVector<Value> offsets{c0, c0};
      for (size_t i = 0; i < numSpatialDims; i++) {
        Value innerSize = rewriter.create<arith::SubIOp>(loc, inDims[i], c1);
        innerSize = rewriter.create<arith::MulIOp>(
            loc, innerSize, castIntToIndex(rewriter, loc, strideIntValues[i]));
        innerSize = rewriter.create<arith::AddIOp>(loc, innerSize, c1);

        Value offset = rewriter.create<arith::SubIOp>(loc, weightDims[i], c1);
        offset = rewriter.create<arith::MulIOp>(
            loc, offset, castIntToIndex(rewriter, loc, dilationIntValues[i]));
        offset = rewriter.create<arith::SubIOp>(
            loc, offset, castIntToIndex(rewriter, loc, paddingIntValues[i]));

        Value outerSize = rewriter.create<arith::MulIOp>(loc, offset, c2);
        outerSize = rewriter.create<arith::AddIOp>(loc, outerSize, innerSize);
        outerSize = rewriter.create<arith::AddIOp>(
            loc, outerSize,
            castIntToIndex(rewriter, loc, outputPaddingIntValues[i]));

        outerSizes.push_back(outerSize);
        offsets.push_back(offset);
      }

      // Allocate padded input tensor
      Value initTensor =
          createZeroInitTensor(rewriter, loc, outerSizes, inputDTy);

      // Insert input into allocated tensor
      SmallVector<Value> strideIndexValues{c1, c1};
      for (auto stride : strideIntValues)
        strideIndexValues.push_back(castIntToIndex(rewriter, loc, stride));
      SmallVector<Value> insertSizes = getTensorSizes(rewriter, loc, input);

      paddedInput = rewriter.create<tensor::InsertSliceOp>(
          loc, torch_to_linalg::removeSizeInformation(rewriter, loc, input),
          initTensor, offsets, insertSizes, strideIndexValues);

      // Calculate output dims
      for (size_t i = 0; i < numSpatialDims; i++)
        outDims.push_back(torch_to_linalg::getOutputDimForConvTransposeOps(
            rewriter, loc, inDims[i], paddingIntValues[i], dilationIntValues[i],
            castIndexToInt(weightDims[i]), strideIntValues[i],
            outputPaddingIntValues[i]));

      // Set stride to 1
      strideInts.clear();
      strideInts.append(numSpatialDims, 1);
    } else {
      Value pad = inputZp;
      if (!pad) {
        if (isa<mlir::FloatType>(inputDTy))
          pad = rewriter.create<arith::ConstantOp>(
              op.getLoc(), rewriter.getFloatAttr(inputDTy, 0.0));
        if (isa<mlir::IntegerType>(inputDTy))
          pad = rewriter.create<arith::ConstantOp>(
              op.getLoc(), rewriter.getIntegerAttr(inputDTy, 0));
      }

      if (pad.getType() != inputDTy) {
        if (isa<mlir::FloatType>(inputDTy))
          pad = rewriter.create<arith::TruncFOp>(op.getLoc(), inputDTy, pad);

        if (isa<mlir::IntegerType>(inputDTy))
          pad = rewriter.create<arith::TruncIOp>(op.getLoc(), inputDTy, pad);
      }

      // Pad input
      paddedInput = torch_to_linalg::getDynamicZeroPaddedTensor(
          op, rewriter, input, paddingIntValues, /*unpaddedDims=*/2, pad);

      // Calculate output dims
      for (size_t i = 0; i < numSpatialDims; i++)
        outDims.push_back(torch_to_linalg::getOutputDimForConvOps(
            rewriter, loc, inDims[i], paddingIntValues[i], dilationIntValues[i],
            castIndexToInt(weightDims[i]), strideIntValues[i]));
    }

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outDims), resultDTy);

    Value outputTensor;
    if (bias.getType().isa<Torch::NoneType>()) {
      Value c0;
      if (resultDTy.isa<mlir::FloatType>()) {
        c0 = rewriter.create<arith::ConstantOp>(
            loc, FloatAttr::get(resultDTy, 0.0));
      } else if (resultDTy.isa<mlir::IntegerType>()) {
        c0 = rewriter.create<arith::ConstantOp>(
            loc, IntegerAttr::get(resultDTy, 0));
      }
      outputTensor = rewriter.create<linalg::FillOp>(loc, c0, initTensor)
                         .getResult(0);

    } else {
      auto biasType = bias.getType().cast<RankedTensorType>();
      if (biasType.getRank() != 1)
        return rewriter.notifyMatchFailure(op, "expect bias to be rank 1");

      auto resultRank = initTensor.getType().cast<RankedTensorType>().getRank();
      SmallVector<AffineMap> indexingMaps = {
          // bias is used to initialize the channels - dimension 1 of output
          AffineMap::get(/*dimCount=*/resultRank, /*symbolCount=*/0,
                         rewriter.getAffineDimExpr(1), context),
          rewriter.getMultiDimIdentityMap(resultRank)};
      SmallVector<utils::IteratorType> iteratorTypes(
          resultRank, utils::IteratorType::parallel);
      outputTensor = rewriter
                         .create<linalg::GenericOp>(
                             loc, initTensor.getType(), bias, initTensor,
                             indexingMaps, iteratorTypes,
                             [](OpBuilder &b, Location loc, ValueRange args) {
                               b.create<linalg::YieldOp>(loc, args[0]);
                             })
                         .getResult(0);
    }

    auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
    auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);

    Value inputStride =
        rewriter.create<arith::FloorDivSIOp>(loc, inChannels, groups);
    Value weightStride =
        rewriter.create<arith::FloorDivSIOp>(loc, weightBatch, groups);

    SmallVector<Value> zeroOffsets(inRank, rewriter.create<arith::ConstantOp>(
                                               loc, rewriter.getIndexAttr(0)));
    SmallVector<Value> unitStrides(inRank, rewriter.create<arith::ConstantOp>(
                                               loc, rewriter.getIndexAttr(1)));
    SmallVector<Value> outDimSlice(outDims);
    outDimSlice[1] = weightStride;
    SmallVector<Value> inputSliceSizes{inBatch, inputStride};
    inputSliceSizes.append(inDims);
    SmallVector<Value> weightSliceSizes{weightStride, weightChannels};
    weightSliceSizes.append(weightDims);

    Value conv;
    // the code so far is able to respect all numSpatialDims
    // the code below this point is numSpatialDims specific and groupSize
    // specific
    // TODO: factor out the above code into a helper function, and then separate
    // convolution into:
    // - grouped 1d-3d
    // - grouped 1d-3d (quantized)
    // - ungrouped 1d-3d
    if (groupSize == 1 && !inputZp && !weightZp) {
      switch (numSpatialDims) {
      case 1:
        conv = rewriter
                   .create<linalg::Conv1DNcwFcwOp>(
                       loc, outputTensor.getType(),
                       ValueRange{paddedInput, weight}, outputTensor,
                       stridesAttr, dilationAttr)
                   .getResult(0);
        break;
      case 2:
        conv = rewriter
                   .create<linalg::Conv2DNchwFchwOp>(
                       loc, outputTensor.getType(),
                       ValueRange{paddedInput, weight}, outputTensor,
                       stridesAttr, dilationAttr)
                   .getResult(0);
        break;
      case 3:
        conv = rewriter
                   .create<linalg::Conv3DNcdhwFcdhwOp>(
                       loc, outputTensor.getType(),
                       ValueRange{paddedInput, weight}, outputTensor,
                       stridesAttr, dilationAttr)
                   .getResult(0);
        break;
      default:
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only 1D, 2D, and 3D convolution supported");
      };
      Type newResultType = getTypeConverter()->convertType(op.getType());
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
      return success();
    }

    if (groupSize == 1 && inputZp && weightZp) {
      // The quantized version uses a different channel ordering so we need to
      // permute the tensors in order to use the existing path. We should
      // eventually directly support this channel ordering.
      llvm::SmallVector<int64_t> inPerms, weightPerms;
      inPerms.push_back(0); // N stays at the front for input.
      // Then we expect the spatial dimensions
      for (size_t i = 0; i < numSpatialDims; ++i) {
        inPerms.push_back(i + 2);
        weightPerms.push_back(i + 2);
      }
      inPerms.push_back(1);
      weightPerms.append({1, 0});

      paddedInput = transposeValue(op.getLoc(), paddedInput, inPerms, rewriter);
      weight = transposeValue(op.getLoc(), weight, weightPerms, rewriter);
      outputTensor =
          transposeValue(op.getLoc(), outputTensor, inPerms, rewriter);

      switch (numSpatialDims) {
      case 2:
        conv = rewriter
                   .create<linalg::Conv2DNhwcHwcfQOp>(
                       loc, outputTensor.getType(),
                       ValueRange{paddedInput, weight, inputZp, weightZp},
                       outputTensor, stridesAttr, dilationAttr)
                   .getResult(0);
        break;
      case 3:
        conv = rewriter
                   .create<linalg::Conv3DNdhwcDhwcfQOp>(
                       loc, outputTensor.getType(),
                       ValueRange{paddedInput, weight, inputZp, weightZp},
                       outputTensor, stridesAttr, dilationAttr)
                   .getResult(0);
        break;
      default:
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only 1D, 2D, and 3D convolution supported");
      };

      llvm::SmallVector<int64_t> outPerms;
      outPerms.push_back(0);
      outPerms.push_back(inPerms.size() - 1);
      for (size_t i = 0; i < numSpatialDims; ++i) {
        outPerms.push_back(i + 1);
      }
      conv = transposeValue(op.getLoc(), conv, outPerms, rewriter);

      Type newResultType = getTypeConverter()->convertType(op.getType());
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
      return success();
    }

    if (inputZp || weightZp)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: quantized grouped convolutions");

    if (numSpatialDims != 2)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only 2D grouped convolution supported");

    // Special depthwise case
    auto inShape = makeShapeTorchCompatible(
        input.getType().cast<RankedTensorType>().getShape());
    auto weightShape = makeShapeTorchCompatible(
        weight.getType().cast<RankedTensorType>().getShape());
    if (weightShape[0] != kUnknownSize && inShape[1] == groupSize &&
        weightShape[0] % inShape[1] == 0 && weightShape[1] == 1) {
      // Collapse weight shape
      SmallVector<ReassociationIndices, 4> collapsedDims = {{0, 1}, {2}, {3}};
      SmallVector<int64_t> collapsedShape{
          (weightShape[0] == kUnknownSize ? kUnknownSize
                                          : weightShape[0] * weightShape[1]),
          weightShape[2], weightShape[3]};
      Type collapsedType = RankedTensorType::get(
          makeShapeLLVMCompatible(collapsedShape), weightDTy);
      Value collapsedWeight = rewriter.create<tensor::CollapseShapeOp>(
          loc, collapsedType, weight, collapsedDims);

      conv = rewriter
                .create<linalg::DepthwiseConv2DNchwChwOp>(
                    loc, outputTensor.getType(),
                    ValueRange{paddedInput, collapsedWeight}, outputTensor,
                    stridesAttr, dilationAttr)
                .getResult(0);

      Type newResultType = getTypeConverter()->convertType(op.getType());
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
      return success();
    }

    // Grouped case, use the grouped conv linalg op
    auto expandGroups = [&](Value tensor, size_t dim) {
      auto inType = tensor.getType().cast<RankedTensorType>();
      auto inShape = makeShapeTorchCompatible(inType.getShape());

      SmallVector<int64_t> outShape;
      for (auto i = 0; i < (long)inShape.size(); i++) {
        if (i == 1) {
          outShape.push_back(groupSize);
        }
        if (i == (long)dim) {
          outShape.push_back(inShape[i] == kUnknownSize
                                 ? kUnknownSize
                                 : inShape[i] / groupSize);
        } else {
          outShape.push_back(inShape[i]);
        }
      }

      SmallVector<ReassociationIndices> indices;
      for (auto i = 0; i <= (long)inShape.size(); i++) {
        if (i == (long)dim) {
          indices.push_back({i, ++i});
          continue;
        }
        indices.push_back({i});
      }

      auto retType = inType.clone(makeShapeLLVMCompatible(outShape));
      return rewriter.create<tensor::ExpandShapeOp>(loc, retType, tensor,
                                                    indices);
    };

    // expand F,C,H,W -> G,F/G,C,H,W
    auto expandWeight = [&](Value tensor) {
      auto inType = tensor.getType().cast<RankedTensorType>();
      auto inShape = makeShapeTorchCompatible(inType.getShape());

      SmallVector<int64_t> outShape{
          groupSize,
          (inShape[0] == kUnknownSize ? kUnknownSize : inShape[0] / groupSize)};
      outShape.append(inShape.begin() + 1, inShape.end());

      SmallVector<ReassociationIndices> indices{{0, 1}};
      for (auto i = 2; i <= (long)inShape.size(); i++)
        indices.push_back({i});

      auto retType = inType.clone(makeShapeLLVMCompatible(outShape));
      return rewriter.create<tensor::ExpandShapeOp>(loc, retType, tensor,
                                                    indices);
    };

    Value paddedInputExpanded = expandGroups(paddedInput, 1);
    Value weightExpanded = expandWeight(weight);
    auto expandOutputTensor = expandGroups(outputTensor, 1);

    // TODO: add 1D and 3D case
    conv = rewriter
               .create<linalg::Conv2DNgchwGfchwOp>(
                   loc, expandOutputTensor.getResultType(),
                   ValueRange{paddedInputExpanded, weightExpanded},
                   expandOutputTensor.getResult(), stridesAttr, dilationAttr)
               .getResult(0);

    conv = rewriter.create<tensor::CollapseShapeOp>(
        loc, outputTensor.getType(), conv,
        expandOutputTensor.getReassociationIndices());
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateLinearPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenMmOp>();
  patterns.add<ConvertAtenMmOp>(typeConverter, context);
  target.addIllegalOp<AtenFlipOp>();
  patterns.add<ConvertAtenFlipOp>(typeConverter, context);
  target.addIllegalOp<AtenMatmulOp>();
  patterns.add<ConvertAtenMatmulOp>(typeConverter, context);
  target.addIllegalOp<AtenBmmOp>();
  patterns.add<ConvertAtenBmmOp>(typeConverter, context);
  target.addIllegalOp<AtenConvolutionOp>();
  patterns.add<ConvertAtenConvolutionOp>(typeConverter, context);
}
