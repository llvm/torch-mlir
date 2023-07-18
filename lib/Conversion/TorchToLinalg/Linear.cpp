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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
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
    if (lhs.getType().cast<RankedTensorType>().getRank() != 2 ||
        rhs.getType().cast<RankedTensorType>().getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.mm to be rank 2");
    }

    Value lhsDim0 = rewriter.create<tensor::DimOp>(loc, lhs, 0);
    Value lhsDim1 = rewriter.create<tensor::DimOp>(loc, lhs, 1);
    Value rhsDim0 = rewriter.create<tensor::DimOp>(loc, rhs, 0);
    Value rhsDim1 = rewriter.create<tensor::DimOp>(loc, rhs, 1);
    Value contractingDimEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhsDim1, rhsDim0);
    rewriter.create<cf::AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for torch.aten.mm"));

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<OpFoldResult>{lhsDim0, rhsDim1}, elementType);
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.0));
    Value zeroFill =
        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                                 ValueRange{lhs, rhs}, zeroFill)
                       .getResult(0);
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
    auto selfRank = adaptor.getSelf().getType().cast<RankedTensorType>().getRank();
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

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
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

    // Fourth Case: Batch-Matrix Multiplication.
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
      if (failed(torch_to_linalg::broadcastToGivenShape(
              op, rewriter, lhs, lhsBroadcastToShape, broadcastedLhs))) {
        return rewriter.notifyMatchFailure(
            op, "unable to perform broadcast operation");
      }
      if (failed(torch_to_linalg::broadcastToGivenShape(
              op, rewriter, rhs, rhsBroadcastToShape, broadcastedRhs))) {
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
      auto indexingMaps =
          AffineMap::inferFromExprList({lhsExpr, rhsExpr, outExpr});
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
class ConvertCustomQuantizedMatmulOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "brevitas.matmul_rhs_group_quant") {
      return failure();
    }
    Location loc = op->getLoc();
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }

    // get inputs: lhs, q_rhs, scales, zps
    Value lhs = adaptor.getOperands()[0];
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    if (!lhsType) {
      return failure();
    }
    auto lhsShape = lhsType.getShape();
    int lhs_reduct_dim_size = lhsShape.back();

    Value q_rhs = adaptor.getOperands()[1];
    auto rhsType = q_rhs.getType().cast<RankedTensorType>();
    if (!rhsType) {
      return failure();
    }
    auto rhsShape = rhsType.getShape();
    int rhs_reduct_dim_size = rhsShape.back();
    Type rhs_elementType = rhsType.getElementType();

    Value scales = adaptor.getOperands()[2];
    Value zps = adaptor.getOperands()[3];
    Value unpacked_type_width = adaptor.getOperands()[4];
    Value group_size = adaptor.getOperands()[5];

    auto getConstantIntegerFromDefiningOp = [](Value operand,
                                               int &extractedInt) {
      auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(operand.getDefiningOp());
      if (!castOp) {
        return failure();
      }
      auto constOp =
          dyn_cast<Torch::ConstantIntOp>(castOp.getOperand(0).getDefiningOp());
      if (!constOp) {
        return failure();
      }
      extractedInt = constOp.getValue();
      return success();
    };

    int gs;
    if (failed(getConstantIntegerFromDefiningOp(group_size, gs))) {
      return failure();
    }
    int unpackedBitWidth;
    if (failed(getConstantIntegerFromDefiningOp(unpacked_type_width, unpackedBitWidth))) {
      return failure();
    }
    if (unpackedBitWidth != rhs_elementType.getIntOrFloatBitWidth()) {
      return failure();
    }

    // get outputs
    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    auto resultType = newResultType.cast<RankedTensorType>();
    if (!resultType) {
      return failure();
    }
    auto resultShape = resultType.getShape();
    Type elementType = resultType.getElementType();

    // expand lhs
    std::vector<int64_t> lhs_expandedShape = {lhsShape[0], lhsShape[1],
                                              lhs_reduct_dim_size / gs, gs};
    RankedTensorType lhs_expandedType = RankedTensorType::get(lhs_expandedShape, elementType);
    SmallVector<ReassociationIndices, 4> lhs_reassociation = {{0}, {1}, {2, 3}};
    Value expanded_lhs = rewriter.create<tensor::ExpandShapeOp>(
      loc, lhs_expandedType, lhs, lhs_reassociation);

    // expand rhs
    std::vector<int64_t> expandedShape = {rhsShape[0], rhs_reduct_dim_size/gs, gs};
    RankedTensorType expandedType = RankedTensorType::get(expandedShape, rhs_elementType);
    SmallVector<ReassociationIndices, 4> reassociation = {{0}, {1, 2}};
    Value expanded_rhs = rewriter.create<tensor::ExpandShapeOp>(
      loc, expandedType, q_rhs, reassociation);
    Value cst_0 = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(elementType, 0.0));

    Value dq_empty = rewriter.create<tensor::EmptyOp>(
      loc, expandedShape, elementType);
    SmallVector<Value> dynDims;
    for (int i = 0; i < lhsType.getRank(); i++) {
      if (lhsType.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, lhs, i));
      }
    }
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, resultShape, elementType, dynDims);
    Value output = rewriter.create<linalg::FillOp>(
      loc, cst_0, empty).getResult(0);

    AffineExpr d0, d1, d2, d3, d4;
    bindDims(getContext(), d0, d1, d2, d3, d4);
    auto c0 = rewriter.getAffineConstantExpr(0);
    auto map = AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext());
    auto map1 = AffineMap::get(3, 0, {d0, d1, c0}, rewriter.getContext());
    auto map2 = AffineMap::get(5, 0, {d0, d1, d3, d4}, rewriter.getContext());
    auto map3 = AffineMap::get(5, 0, {d2, d3, d4}, rewriter.getContext());
    auto map4 = AffineMap::get(5, 0, {d0, d1, d2}, rewriter.getContext());
    SmallVector<AffineMap, 4> dq_indexingMaps = {map, map1, map1, map};
    SmallVector<AffineMap, 4> mat_indexingMaps = {map2, map3, map4};

    SmallVector<utils::IteratorType> dq_iteratorTypes(3, utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> mat_iteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::reduction,
      utils::IteratorType::reduction
    };

    Value dq_rhs =
        rewriter
            .create<linalg::GenericOp>(
                loc, dq_empty.getType(),
                ValueRange{expanded_rhs, scales, zps}, dq_empty,
                /*indexingMaps=*/dq_indexingMaps,
                /*iteratorTypes=*/dq_iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value w = args[0], scale = args[1], zeroPoint = args[2];
                  Value extw = b.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), w);
                  Value fp_extw = b.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), extw);
                  Value shifted = b.create<arith::SubFOp>(loc, fp_extw, zeroPoint);
                  Value dqw = b.create<arith::MulFOp>(loc, shifted, scale);
                  b.create<linalg::YieldOp>(loc, dqw);
                })
            .getResult(0);

    Value quantMat =
        rewriter
            .create<linalg::GenericOp>(
                loc, output.getType(),
                ValueRange{expanded_lhs, dq_rhs}, output,
                /*indexingMaps=*/mat_indexingMaps,
                /*iteratorTypes=*/mat_iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value l = args[0], r = args[1], out = args[2];
                  Value pd = b.create<arith::MulFOp>(loc, l, r);
                  Value ac = b.create<arith::AddFOp>(loc, pd, out);
                  b.create<linalg::YieldOp>(loc, ac);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, quantMat);
    return success();
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

    if (lhsType.getRank() != 3 || rhsType.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.bmm to be rank 3");
    }
    if (!lhsType.getElementType().isa<mlir::FloatType>() ||
        lhsType.getElementType() != rhsType.getElementType())
      return op.emitError(
          "unimplemented: non floating point operands or operands of "
          "different types");

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

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor0 = createZeroInitTensor(
        rewriter, loc, ValueRange{lhsDim0, lhsDim1, rhsDim2}, elementType);

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

    bool transposed = true;
    if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant transposed supported");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");
    size_t inRank = input.getType().cast<RankedTensorType>().getRank();
    size_t numSpacialDims = inRank - 2;
    if (numSpacialDims != 2)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only 2D convolution currently supported");

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
    if (!matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilationInts)))
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
          createZeroInitTensor(rewriter, loc, weightInitDims, elementType);
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
      for (size_t i = 0; i < numSpacialDims; i++) {
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
          createZeroInitTensor(rewriter, loc, outerSizes, elementType);

      // Insert input into allocated tensor
      SmallVector<Value> strideIndexValues{c1, c1};
      for (auto stride : strideIntValues)
        strideIndexValues.push_back(castIntToIndex(rewriter, loc, stride));
      SmallVector<Value> insertSizes = getTensorSizes(rewriter, loc, input);

      paddedInput = rewriter.create<tensor::InsertSliceOp>(
          loc, torch_to_linalg::removeSizeInformation(rewriter, loc, input),
          initTensor, offsets, insertSizes, strideIndexValues);

      // Calculate output dims
      for (size_t i = 0; i < numSpacialDims; i++)
        outDims.push_back(torch_to_linalg::getOutputDimForConvTransposeOps(
            rewriter, loc, inDims[i], paddingIntValues[i], dilationIntValues[i],
            castIndexToInt(weightDims[i]), strideIntValues[i],
            outputPaddingIntValues[i]));

      // Set stride to 1
      strideInts.clear();
      strideInts.append(numSpacialDims, 1);

    } else {
      // Pad input
      paddedInput = torch_to_linalg::getDynamicZeroPaddedTensor(
          op, rewriter, input, paddingIntValues, /*unpaddedDims=*/2);

      // Calculate output dims
      for (size_t i = 0; i < numSpacialDims; i++)
        outDims.push_back(torch_to_linalg::getOutputDimForConvOps(
            rewriter, loc, inDims[i], paddingIntValues[i], dilationIntValues[i],
            castIndexToInt(weightDims[i]), strideIntValues[i]));
    }

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outDims), elementType);

    Value bias = adaptor.getBias();
    Value outputTensor;
    if (bias.getType().isa<Torch::NoneType>()) {
      Value c0float = rewriter.create<arith::ConstantOp>(
          loc, FloatAttr::get(elementType, 0.0));
      outputTensor = rewriter.create<linalg::FillOp>(loc, c0float, initTensor)
                         .getResult(0);
    } else {
      auto biasType = bias.getType().cast<RankedTensorType>();
      if (biasType.getRank() != 1)
        return rewriter.notifyMatchFailure(op, "expect bias to be rank 1");
      if (elementType != biasType.getElementType())
        return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");

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
    if (groupSize == 1) {
      // TODO: add 1D and 3D case
      conv =
          rewriter
              .create<linalg::Conv2DNchwFchwOp>(
                  loc, outputTensor.getType(), ValueRange{paddedInput, weight},
                  outputTensor, stridesAttr, dilationAttr)
              .getResult(0);
    } else {
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
            makeShapeLLVMCompatible(collapsedShape), elementType);
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

      auto expandWeight = [&](Value tensor) {
        auto inType = tensor.getType().cast<RankedTensorType>();
        auto inShape = makeShapeTorchCompatible(inType.getShape());

        SmallVector<int64_t> outShape{
            groupSize, (inShape[0] == kUnknownSize ? kUnknownSize
                                                   : inShape[0] / groupSize)};
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
      Value outputTensorExpanded = expandGroups(outputTensor, 1);

      // TODO: add 1D and 3D case
      conv = rewriter
                 .create<linalg::Conv2DNgchwFgchwOp>(
                     loc, outputTensorExpanded.getType(),
                     ValueRange{paddedInputExpanded, weightExpanded},
                     outputTensorExpanded, stridesAttr, dilationAttr)
                 .getResult(0);

      SmallVector<ReassociationIndices> indices{{0}, {1, 2}};
      for (auto dim = 3; dim <= (int64_t)inRank; dim++)
        indices.push_back({dim});
      conv = rewriter.create<tensor::CollapseShapeOp>(
          loc, outputTensor.getType(), conv, indices);
    }

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
  target.addIllegalOp<OperatorOp>();
  patterns.add<ConvertCustomQuantizedMatmulOp>(typeConverter, context);
  target.addIllegalOp<AtenBmmOp>();
  patterns.add<ConvertAtenBmmOp>(typeConverter, context);
  target.addIllegalOp<AtenConvolutionOp>();
  patterns.add<ConvertAtenConvolutionOp>(typeConverter, context);
}
