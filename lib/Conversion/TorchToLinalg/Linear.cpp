//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
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

// for uint8 types, we shift down by 128 so that we can faithfully
// represent the quantization with signed i8 types.
static void signShift(PatternRewriter &rewriter, Location loc, Value &arg,
                      Value &zp, bool isUnsignedType, int64_t numBits) {
  if (!isUnsignedType)
    return;
  int64_t minSI = -(1 << (numBits - 1));
  Value minSIValue = arith::ConstantIntOp::create(
      rewriter, loc, minSI, cast<mlir::IntegerType>(zp.getType()).getWidth());
  zp = arith::AddIOp::create(rewriter, loc, zp, minSIValue);
  minSIValue = arith::ConstantIntOp::create(rewriter, loc, minSI, numBits);
  arg = torch_to_linalg::createElementwiseLinalgGeneric(
      rewriter, loc, ValueRange{arg},
      cast<TensorType>(arg.getType()).getElementType(),
      [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
        Value result =
            arith::AddIOp::create(rewriter, loc, payloadArgs[0], minSIValue);
        linalg::YieldOp::create(b, loc, result);
      });
}

static Value transposeValue(Location loc, Value value, ArrayRef<int64_t> perms,
                            PatternRewriter &rewriter) {
  auto valueTy = cast<RankedTensorType>(value.getType());
  auto inShape = valueTy.getShape();
  llvm::SmallVector<int64_t> outShape;
  llvm::SmallVector<Value> dynDims;
  for (size_t i = 0; i < perms.size(); ++i) {
    outShape.push_back(inShape[perms[i]]);
    if (ShapedType::isDynamic(inShape[perms[i]])) {
      dynDims.push_back(tensor::DimOp::create(rewriter, loc, value, perms[i]));
    }
  }

  auto outTy = RankedTensorType::get(outShape, valueTy.getElementType());
  Value empty = tensor::EmptyOp::create(rewriter, loc, outTy, dynDims);
  Value transpose =
      linalg::TransposeOp::create(rewriter, loc, value, empty, perms)
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

    RankedTensorType lhsType = cast<RankedTensorType>(lhs.getType());
    RankedTensorType rhsType = cast<RankedTensorType>(rhs.getType());

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.mm to be rank 2");
    }

    ValueTensorType lhsTorchType =
        cast<ValueTensorType>(op.getSelf().getType());
    ValueTensorType rhsTorchType =
        cast<ValueTensorType>(op.getMat2().getType());

    Value lhsZeroPoint, rhsZeroPoint;
    getZeroPoint(op.getSelf(), lhsZeroPoint);
    getZeroPoint(op.getMat2(), rhsZeroPoint);

    if (static_cast<bool>(lhsZeroPoint) != static_cast<bool>(rhsZeroPoint)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported: aten.mm with mixed quantization");
    }

    if (lhsTorchType.getDtype() != rhsTorchType.getDtype()) {
      if (!lhsZeroPoint) {
        return rewriter.notifyMatchFailure(
            op, "unsupported: aten.mm with different input element types");
      }
      // Allows quantized types to mismatch since they will be cast to the same
      // type.
    }

    bool isUnsigned = torch_to_linalg::isUnsignedTorchType(lhsTorchType);
    bool isUnsignedR = torch_to_linalg::isUnsignedTorchType(rhsTorchType);

    Value lhsDim0 = tensor::DimOp::create(rewriter, loc, lhs, 0);
    Value rhsDim1 = tensor::DimOp::create(rewriter, loc, rhs, 1);

    if (!isAssumingStrictSymbolicShapes(rewriter)) {
      Value lhsDim1 = tensor::DimOp::create(rewriter, loc, lhs, 1);
      Value rhsDim0 = tensor::DimOp::create(rewriter, loc, rhs, 0);
      Value contractingDimEqual = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, lhsDim1, rhsDim0);
      cf::AssertOp::create(
          rewriter, loc, contractingDimEqual,
          rewriter.getStringAttr(
              "mismatching contracting dimension for torch.aten.mm"));
    }

    TensorType resultType =
        cast<TensorType>(getTypeConverter()->convertType(op.getType()));
    Type elementType = resultType.getElementType();
    auto accumulatorDType =
        getDefaultAccType(rewriter, lhsType.getElementType());
    if (accumulatorDType != resultType.getElementType()) {
      elementType = accumulatorDType;
    }
    Value zeroFill = createZeroInitTensor(
        rewriter, loc, ValueRange{lhsDim0, rhsDim1}, elementType);

    Value matmul;
    if (lhsZeroPoint) {
      lhsZeroPoint = typeConverter->materializeTargetConversion(
          rewriter, loc,
          getTypeConverter()->convertType(lhsZeroPoint.getType()),
          lhsZeroPoint);
      rhsZeroPoint = typeConverter->materializeTargetConversion(
          rewriter, loc,
          getTypeConverter()->convertType(rhsZeroPoint.getType()),
          rhsZeroPoint);
      lhsZeroPoint = arith::TruncIOp::create(
          rewriter, loc, rewriter.getI32Type(), lhsZeroPoint);
      rhsZeroPoint = arith::TruncIOp::create(
          rewriter, loc, rewriter.getI32Type(), rhsZeroPoint);

      // change uint8 quantization -> int8 quantization
      int64_t numBits =
          cast<mlir::IntegerType>(lhsType.getElementType()).getWidth();
      signShift(rewriter, loc, lhs, lhsZeroPoint, isUnsigned, numBits);
      numBits = cast<mlir::IntegerType>(rhsType.getElementType()).getWidth();
      signShift(rewriter, loc, rhs, rhsZeroPoint, isUnsignedR, numBits);

      matmul = linalg::QuantizedMatmulOp::create(
                   rewriter, loc, zeroFill.getType(),
                   ValueRange{lhs, rhs, lhsZeroPoint, rhsZeroPoint}, zeroFill)
                   .getResult(0);
    } else if (isUnsigned) {
      auto matmulOp = linalg::MatmulOp::create(
          rewriter, loc, zeroFill.getType(), ValueRange{lhs, rhs}, zeroFill);
      matmulOp.setCast(linalg::TypeFn::cast_unsigned);
      matmul = matmulOp->getResult(0);
    } else {
      matmul = linalg::MatmulOp::create(rewriter, loc, zeroFill.getType(),
                                        ValueRange{lhs, rhs}, zeroFill)
                   .getResult(0);
    }

    if (accumulatorDType != resultType.getElementType()) {
      matmul = torch_to_linalg::convertTensorToElementType(
          rewriter, loc, matmul, resultType.getElementType());
    }
    // When constructed with just dynamic sizes, EmptyOp will have a result
    // type which has all `?`'s for dimensions, which might not be the result
    // type of `op`. The constraints on later linalg ops means that the result
    // of the MatmulOp will have this type too. So cast it to the desired type
    // so that in the end we have the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, matmul);

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
    Value self = adaptor.getSelf();
    auto selfRank =
        cast<RankedTensorType>(adaptor.getSelf().getType()).getRank();

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

    Value flipped = torch_to_linalg::flipTensor(rewriter, loc, self, axis);
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
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());

    auto lhsTorchType = cast<ValueTensorType>(op.getSelf().getType());
    auto rhsTorchType = cast<ValueTensorType>(op.getOther().getType());

    // Get the rank of both matrix.
    unsigned lhsRank = lhsType.getRank();
    unsigned rhsRank = rhsType.getRank();

    Value lhsZeroPoint, rhsZeroPoint;
    getZeroPoint(op.getSelf(), lhsZeroPoint);
    getZeroPoint(op.getOther(), rhsZeroPoint);

    if (static_cast<bool>(lhsZeroPoint) != static_cast<bool>(rhsZeroPoint)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported: aten.matmul with mixed quantization");
    }

    bool isUnsigned = torch_to_linalg::isUnsignedTorchType(lhsTorchType);
    bool isUnsignedR = torch_to_linalg::isUnsignedTorchType(rhsTorchType);

    if (!lhsZeroPoint && lhsTorchType.getDtype() != rhsTorchType.getDtype()) {
      // Allows quantized types to mismatch
      return rewriter.notifyMatchFailure(
          op, "unsupported: aten.matmul with different input element types");
    }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    auto resultType = cast<RankedTensorType>(newResultType);
    Type elementType = resultType.getElementType();

    if (lhsZeroPoint) {
      // get each zero point ready to pass to a quantized_matmul
      lhsZeroPoint = typeConverter->materializeTargetConversion(
          rewriter, loc,
          getTypeConverter()->convertType(lhsZeroPoint.getType()),
          lhsZeroPoint);
      rhsZeroPoint = typeConverter->materializeTargetConversion(
          rewriter, loc,
          getTypeConverter()->convertType(rhsZeroPoint.getType()),
          rhsZeroPoint);
      lhsZeroPoint = arith::TruncIOp::create(
          rewriter, loc, rewriter.getI32Type(), lhsZeroPoint);
      rhsZeroPoint = arith::TruncIOp::create(
          rewriter, loc, rewriter.getI32Type(), rhsZeroPoint);

      // change uint8 quantization -> int8 quantization
      int64_t numBits =
          cast<mlir::IntegerType>(lhsType.getElementType()).getWidth();
      signShift(rewriter, loc, lhs, lhsZeroPoint, isUnsigned, numBits);
      numBits = cast<mlir::IntegerType>(rhsType.getElementType()).getWidth();
      signShift(rewriter, loc, rhs, rhsZeroPoint, isUnsignedR, numBits);

      // for quantized vec-vec, vec-mat, and mat-vec cases, lower to
      // expand/collapse + quantized_matmul
      bool lhsVec = (lhsRank == 1 && rhsRank <= 2);
      bool rhsVec = (lhsRank <= 2 && rhsRank == 1);

      if (lhsVec || rhsVec) {
        SmallVector<ReassociationIndices> reassociation(1);
        reassociation[0].push_back(0);
        reassociation[0].push_back(1);

        if (lhsVec) {
          // unsqueeze lhs to a matrix
          int64_t lhsDim = lhsType.getShape()[0];
          auto lhsUnsqueezeType = RankedTensorType::get(
              ArrayRef<int64_t>{1, lhsDim}, lhsType.getElementType());
          lhs = tensor::ExpandShapeOp::create(rewriter, loc, lhsUnsqueezeType,
                                              lhs, reassociation);
        }
        if (rhsVec) {
          // unsqueeze rhs to a matrix
          int64_t rhsDim = rhsType.getShape()[0];
          auto rhsUnsqueezeType = RankedTensorType::get(
              ArrayRef<int64_t>{rhsDim, 1}, rhsType.getElementType());
          rhs = tensor::ExpandShapeOp::create(rewriter, loc, rhsUnsqueezeType,
                                              rhs, reassociation);
        }
        // get quantized_matmul and squeeze result
        Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
        Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
        Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
        Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
        checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

        Value zeroTensor = createZeroInitTensor(
            rewriter, loc, ValueRange{lhsDim0, rhsDim1}, elementType);
        Value matmul =
            linalg::QuantizedMatmulOp::create(
                rewriter, loc, zeroTensor.getType(),
                ValueRange{lhs, rhs, lhsZeroPoint, rhsZeroPoint}, zeroTensor)
                .getResult(0);
        int64_t resultRank = resultType.getRank();
        if (resultRank == 0) {
          // in vec-vec case, need to collapse result to a scalar
          reassociation.clear();
        }
        matmul = tensor::CollapseShapeOp::create(rewriter, loc, resultType,
                                                 matmul, reassociation);
        rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
        return success();
      }
      // the remaining quantized cases (Mat-Mat and broadcast -> BMM) are
      // covered in the relevant section below
    }

    // The different cases of torch_matmul op is mentioned here:
    // https://pytorch.org/docs/stable/generated/torch.matmul.html

    // First Case: Dot Product.
    if (lhsRank == 1 && rhsRank == 1) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);

      checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

      Value zeroTensor = createZeroInitTensor(rewriter, loc, {}, elementType);
      Value dotProd = linalg::DotOp::create(rewriter, loc, zeroTensor.getType(),
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
          linalg::VecmatOp::create(rewriter, loc, zeroTensor.getType(),
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
          linalg::MatvecOp::create(rewriter, loc, zeroTensor.getType(),
                                   ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Fourth Case: Mat-Mat Multiplication.
    if (lhsRank == 2 && rhsRank == 2) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      Value zeroTensor = createZeroInitTensor(
          rewriter, loc, ValueRange{lhsDim0, rhsDim1}, elementType);
      Value matmul;
      if (lhsZeroPoint) {
        matmul =
            linalg::QuantizedMatmulOp::create(
                rewriter, loc, zeroTensor.getType(),
                ValueRange{lhs, rhs, lhsZeroPoint, rhsZeroPoint}, zeroTensor)
                .getResult(0);
      } else {
        matmul = linalg::MatmulOp::create(rewriter, loc, zeroTensor.getType(),
                                          ValueRange{lhs, rhs}, zeroTensor)
                     .getResult(0);
      }
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
      SmallVector<int64_t> lhsTargetShape =
          llvm::to_vector(llvm::map_range(lhsBroadcastToShape, [](Value v) {
            return getConstantIntValue(v).value_or(ShapedType::kDynamic);
          }));

      auto lhsBroadcastType = RankedTensorType::get(
          lhsTargetShape, lhsType.getElementType(), lhsType.getEncoding());
      if (failed(torch_to_linalg::broadcastToGivenShape(
              op, rewriter, lhs, lhsBroadcastToShape, lhsBroadcastType,
              broadcastedLhs))) {
        return rewriter.notifyMatchFailure(
            op, "unable to perform broadcast operation");
      }
      SmallVector<int64_t> rhsTargetShape =
          llvm::to_vector(llvm::map_range(rhsBroadcastToShape, [](Value v) {
            return getConstantIntValue(v).value_or(ShapedType::kDynamic);
          }));
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
        Value matmul;
        if (lhsZeroPoint) {
          matmul = linalg::QuantizedBatchMatmulOp::create(
                       rewriter, loc, zeroTensor.getType(),
                       ValueRange{broadcastedLhs, broadcastedRhs, lhsZeroPoint,
                                  rhsZeroPoint},
                       zeroTensor)
                       .getResult(0);
          rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
                                                      matmul);
          return success();
        }
        matmul = linalg::BatchMatmulOp::create(
                     rewriter, loc, zeroTensor.getType(),
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
        Value collapsedLhs = tensor::CollapseShapeOp::create(
            rewriter, op->getLoc(), broadcastedLhs, reassociation);
        Value collapsedRhs = tensor::CollapseShapeOp::create(
            rewriter, op->getLoc(), broadcastedRhs, reassociation);

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

        Value initTensor = tensor::EmptyOp::create(
            rewriter, loc, updatedCollapseResultShape, elementType);
        Value c0 = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getZeroAttr(elementType));
        Value zeroTensor =
            linalg::FillOp::create(rewriter, loc, c0, initTensor).getResult(0);
        Value batchMatMul;

        if (lhsZeroPoint) {
          batchMatMul = linalg::QuantizedBatchMatmulOp::create(
                            rewriter, loc, zeroTensor.getType(),
                            ValueRange{collapsedLhs, collapsedRhs, lhsZeroPoint,
                                       rhsZeroPoint},
                            zeroTensor)
                            .getResult(0);
        } else {
          batchMatMul = linalg::BatchMatmulOp::create(
                            rewriter, loc, zeroTensor.getType(),
                            ValueRange{collapsedLhs, collapsedRhs}, zeroTensor)
                            .getResult(0);
        }
        Value expandResult = tensor::ExpandShapeOp::create(
            rewriter, loc, resultType, batchMatMul, reassociation);
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
          linalg::GenericOp::create(
              rewriter, loc, zeroTensor.getType(),
              ValueRange{broadcastedLhs, broadcastedRhs}, zeroTensor,
              /*indexingMaps=*/indexingMaps,
              /*iteratorTypes=*/iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                Value l = args[0], r = args[1], res = args[2];
                Value mul = arith::MulFOp::create(b, loc, l, r);
                Value add = arith::AddFOp::create(b, loc, mul, res);
                linalg::YieldOp::create(b, loc, add);
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
    RankedTensorType lhsType = cast<RankedTensorType>(lhs.getType());
    RankedTensorType rhsType = cast<RankedTensorType>(rhs.getType());
    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type resultElementType =
        cast<RankedTensorType>(newResultType).getElementType();
    Type lhsElementType = cast<RankedTensorType>(lhsType).getElementType();
    Type rhsElementType = cast<RankedTensorType>(rhsType).getElementType();

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

    Type accumulatorDType = getDefaultAccType(rewriter, resultElementType);
    Value initTensor0 = createZeroInitTensor(
        rewriter, loc, ValueRange{lhsDim0, lhsDim1, rhsDim2}, accumulatorDType);

    Value bmm =
        linalg::BatchMatmulOp::create(rewriter, loc, initTensor0.getType(),
                                      ValueRange{lhs, rhs}, initTensor0)
            .getResult(0);

    if (accumulatorDType != resultElementType) {
      bmm = torch_to_linalg::convertTensorToElementType(rewriter, loc, bmm,
                                                        resultElementType);
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, bmm);
    return success();
  }
};
} // namespace

namespace {
bool isValueNegative(mlir::Value value) {
  // Try to fold the operation to a constant
  mlir::Operation *definingOp = value.getDefiningOp();

  if (!definingOp)
    return false;

  // Attempt to fold the operation
  mlir::SmallVector<mlir::OpFoldResult, 1> results;
  if (failed(definingOp->fold(results)) || results.empty())
    return false;

  // Check if the folded result is a constant
  if (auto attr = results.front().dyn_cast<mlir::Attribute>()) {
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(attr)) {
      int64_t intValue = intAttr.getInt();
      return intValue < 0;
    }
  }

  return false;
}
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
    Value weight = adaptor.getWeight(); /* in form of F*C/G*H*W */
    Value bias = adaptor.getBias();
    auto resultTy = cast<ValueTensorType>(op.getType());

    Value inputZp, weightZp;
    bool inputUnsigned = false;
    bool weightUnsigned = false;
    if (auto make = op.getInput()
                        .getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
      input = make.getSelf();
      inputZp = make.getZeroPoint();
      input = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(input.getType()), input);
      inputZp = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(inputZp.getType()),
          inputZp);
      inputZp = arith::TruncIOp::create(rewriter, loc, rewriter.getI32Type(),
                                        inputZp);
      auto torchDtype = cast<ValueTensorType>(make.getType()).getDtype();
      inputUnsigned = torch_to_linalg::isUnsignedTorchType(torchDtype);
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
      weightZp = arith::TruncIOp::create(rewriter, loc, rewriter.getI32Type(),
                                         weightZp);
      auto torchDtype = cast<ValueTensorType>(make.getType()).getDtype();
      weightUnsigned = torch_to_linalg::isUnsignedTorchType(torchDtype);
    }

    if (static_cast<bool>(inputZp) != static_cast<bool>(weightZp)) {
      return rewriter.notifyMatchFailure(
          op, "lhs and rhs of convolution must either be both int or fp");
    }

    if (inputZp && !isa<Torch::NoneType>(bias.getType())) {
      auto biasDTy = cast<RankedTensorType>(bias.getType()).getElementType();
      if (!biasDTy.isInteger(32)) {
        return rewriter.notifyMatchFailure(
            op, "quantized result ty should be i32 accumulator");
      }
    }

    bool transposed = true;
    if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant transposed supported");

    auto inputDTy = cast<RankedTensorType>(input.getType()).getElementType();
    auto weightDTy = cast<RankedTensorType>(weight.getType()).getElementType();
    auto resultDTy = resultTy.toBuiltinTensor().getElementType();

    if (!isa<mlir::FloatType, mlir::IntegerType>(inputDTy) ||
        !isa<mlir::FloatType, mlir::IntegerType>(weightDTy) ||
        !isa<mlir::FloatType, mlir::IntegerType>(resultDTy))
      return op.emitError("unimplemented: non-fp not-int type");
    size_t inRank = cast<RankedTensorType>(input.getType()).getRank();
    size_t numSpatialDims = inRank - 2;
    if (numSpatialDims < 1 || numSpatialDims > 3)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only 1d-3d convolution currently supported");

    Type intType = IntegerType::get(context, 64);
    auto castIndexToInt = [&](Value v) {
      return rewriter.createOrFold<arith::IndexCastOp>(loc, intType, v);
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

    // Checks for valid group size
    int64_t numGroups;
    if (!matchPattern(op.getGroups(), m_TorchConstantInt(&numGroups)))
      return rewriter.notifyMatchFailure(op,
                                         "only constant group size supported.");
    Value groups = castIntToIndex(rewriter, loc, adaptor.getGroups());

    // Adding support for 1d group convolution by converting the 1d-conv to
    // 2d-conv.
    // TODO: Replace this logic with the appropriate linalg op for 1-d group
    // convolution once that support is added.
    bool is1DGroupConv = (numSpatialDims == 1 && numGroups != 1);
    if (is1DGroupConv) {
      // Unsqueezing the last dim of input and weight. Also extending the
      // dilation, stride, padding, and output padding lists.
      auto unsqueezeInputInfo =
          unsqueezeTensor(rewriter, op, input, /*dim=*/-1);
      if (failed(unsqueezeInputInfo)) {
        return rewriter.notifyMatchFailure(op,
                                           "cannot generate unsqueeze tensor");
      }
      input = unsqueezeInputInfo.value();

      auto unsqueezeWeightInfo =
          unsqueezeTensor(rewriter, op, weight, /*dim=*/-1);
      if (failed(unsqueezeWeightInfo)) {
        return rewriter.notifyMatchFailure(op,
                                           "cannot generate unsqueeze tensor");
      }
      weight = unsqueezeWeightInfo.value();

      Value cstZero = arith::ConstantOp::create(rewriter, loc,
                                                rewriter.getI64IntegerAttr(0));
      paddingIntValues.push_back(cstZero);
      outputPaddingIntValues.push_back(cstZero);
      strideInts.push_back(1);
      dilationInts.push_back(1);

      inRank++;
      numSpatialDims++;
    }

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

    auto validate = [&](Value toValidate, std::string err) {
      Value c0 =
          arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
      Value inputValid = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, c0,
          arith::RemSIOp::create(rewriter, loc, toValidate, groups));
      cf::AssertOp::create(rewriter, loc, inputValid,
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

    // convert any uint8 quantization to int8 quantization
    if (auto integerType = dyn_cast<mlir::IntegerType>(inputDTy)) {
      int64_t width = integerType.getWidth();
      signShift(rewriter, loc, input, inputZp, inputUnsigned, width);
    }
    if (auto integerType = dyn_cast<mlir::IntegerType>(weightDTy)) {
      int64_t width = integerType.getWidth();
      signShift(rewriter, loc, weight, weightZp, weightUnsigned, width);
    }
    // Pad the input tensor according to padding.
    SmallVector<Value> outDims{inBatch, weightBatch};
    Value paddedInput;
    Value pad = inputZp;
    if (!pad) {
      if (isa<mlir::FloatType>(inputDTy))
        pad = arith::ConstantOp::create(rewriter, op.getLoc(),
                                        rewriter.getFloatAttr(inputDTy, 0.0));
      if (isa<mlir::IntegerType>(inputDTy))
        pad = arith::ConstantOp::create(rewriter, op.getLoc(),
                                        rewriter.getIntegerAttr(inputDTy, 0));
    }
    if (pad.getType() != inputDTy) {
      if (isa<mlir::FloatType>(inputDTy))
        pad = arith::TruncFOp::create(rewriter, op.getLoc(), inputDTy, pad);

      if (isa<mlir::IntegerType>(inputDTy))
        pad = arith::TruncIOp::create(rewriter, op.getLoc(), inputDTy, pad);
    }

    // The expandWeight lambda function below is used to expand the group
    // dimension. For the normal case the group dimension is expanded out
    // of the output filter dimension:
    // expand F,C,H,W -> G,F/G,C,H,W
    //
    // Note that the group dimension has to be the first dimension. For the
    // transposed convolution case, the group convolution is extracted out
    // of the input channel dimension. But note that the input channel
    // dimension is interchanged with the output filter dimension (due to
    // the transposed operation). Because of this the group and input
    // channel dimensions will not be adjacent and the expand_shape
    // operation will not work.
    //
    // For this reason, in the transposed convolution case the expandWeight
    // lambda needs to be executed before this dimension flipping by doing
    // these two steps:
    //
    // Expansion:    C,F,H,W -> G,C/G,F,H,W
    //
    // Dimension interchange: G,C/G,F,H,W -> G,F,C/G,H,W
    //
    auto expandWeight = [&](Value tensor) {
      auto inType = cast<RankedTensorType>(tensor.getType());
      auto inShape = makeShapeTorchCompatible(inType.getShape());

      SmallVector<int64_t> outShape{numGroups,
                                    (inShape[0] == kUnknownSize
                                         ? kUnknownSize
                                         : (inShape[0] / numGroups)),
                                    inShape[1]};
      outShape.append(inShape.begin() + 2, inShape.end());

      SmallVector<ReassociationIndices> indices{};
      int currIndex = 0;
      indices.push_back({0, 1});
      currIndex += 2;
      for (int i = currIndex; i <= (long)inShape.size(); i++)
        indices.push_back({i});

      auto retType = inType.clone(makeShapeLLVMCompatible(outShape));
      return tensor::ExpandShapeOp::create(rewriter, loc, retType, tensor,
                                           indices);
    };

    if (transposed) {
      bool isGroupedConv = numGroups > 1;
      weight = isGroupedConv ? expandWeight(weight) : weight;

      Value c0 =
          arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
      Value c1 =
          arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

      // Transpose and flip weight
      SmallVector<Value> weightInitDims = getTensorSizes(rewriter, loc, weight);
      if (isGroupedConv) {
        // We need to skip the first dimension (group) in this case, also the
        // output dimension needs to consider the number of groups.
        std::iter_swap(weightInitDims.begin() + 1, weightInitDims.begin() + 2);
        auto numGroupsVal =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, numGroups);
        outDims[1] = rewriter.createOrFold<mlir::arith::MulIOp>(
            loc, weightInitDims[1], numGroupsVal);
      } else {
        std::iter_swap(weightInitDims.begin(), weightInitDims.begin() + 1);
        outDims[1] = weightInitDims[0];
      }
      auto weightRank = weightInitDims.size();
      Value weightInitTensor =
          createZeroInitTensor(rewriter, loc, weightInitDims, weightDTy);
      SmallVector<utils::IteratorType> iteratorTypes(
          weightRank, utils::IteratorType::parallel);
      SmallVector<AffineMap> indexingMaps{
          AffineMap::getMultiDimIdentityMap(weightRank, context)};
      weight = linalg::GenericOp::create(
                   rewriter, loc, weightInitTensor.getType(), ValueRange{},
                   weightInitTensor, indexingMaps, iteratorTypes,
                   [&](OpBuilder &b, Location loc, ValueRange args) {
                     SmallVector<Value> indices;
                     for (size_t i = 0; i < weightRank; i++)
                       indices.push_back(linalg::IndexOp::create(b, loc, i));
                     auto fcIdxSwapOffset = isGroupedConv ? 1 : 0;
                     std::iter_swap(indices.begin() + fcIdxSwapOffset,
                                    indices.begin() + fcIdxSwapOffset + 1);
                     // Flip only the spatial dimensions (from 2 to
                     // weightRank)
                     for (size_t flipDim = fcIdxSwapOffset + 2;
                          flipDim < weightRank; flipDim++) {
                       indices[flipDim] = arith::SubIOp::create(
                           b, loc,
                           arith::SubIOp::create(b, loc,
                                                 weightInitDims[flipDim], c1),
                           indices[flipDim]);
                     }
                     Value res =
                         tensor::ExtractOp::create(b, loc, weight, indices)
                             .getResult();
                     linalg::YieldOp::create(b, loc, res);
                   })
                   .getResult(0);

      paddedInput = createTransposedInputPadding(
          inBatch, inChannels, inDims, weightDims, paddingIntValues,
          strideIntValues, dilationIntValues, outputPaddingIntValues, input,
          inputDTy, pad, rewriter, loc, numSpatialDims, c0, c1);

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
      // Pad input
      paddedInput = torch_to_linalg::getDynamicZeroPaddedTensor(
          op, rewriter, input, paddingIntValues, /*unpaddedDims=*/2, pad);

      // Calculate output dims
      for (size_t i = 0; i < numSpatialDims; i++)
        outDims.push_back(torch_to_linalg::getOutputDimForConvOps(
            rewriter, loc, inDims[i], paddingIntValues[i], dilationIntValues[i],
            castIndexToInt(weightDims[i]), strideIntValues[i]));
    }

    Type accumulatorDType = getDefaultAccType(rewriter, inputDTy);
    Value initTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(outDims), accumulatorDType);

    Value outputTensor;
    if (accumulatorDType != resultDTy && !isa<Torch::NoneType>(bias.getType()))
      bias = torch_to_linalg::convertTensorToElementType(rewriter, loc, bias,
                                                         accumulatorDType);
    if (isa<Torch::NoneType>(bias.getType())) {
      Value c0;
      if (isa<mlir::FloatType>(accumulatorDType)) {
        c0 = arith::ConstantOp::create(rewriter, loc,
                                       FloatAttr::get(accumulatorDType, 0.0));
      } else if (isa<mlir::IntegerType>(accumulatorDType)) {
        c0 = arith::ConstantOp::create(rewriter, loc,
                                       IntegerAttr::get(accumulatorDType, 0));
      }
      outputTensor =
          linalg::FillOp::create(rewriter, loc, c0, initTensor).getResult(0);

    } else {
      auto biasType = cast<RankedTensorType>(bias.getType());
      if (biasType.getRank() != 1)
        return rewriter.notifyMatchFailure(op, "expect bias to be rank 1");

      auto resultRank = cast<RankedTensorType>(initTensor.getType()).getRank();
      SmallVector<int64_t, 4> addedDimensions;
      // bias is used to initialize the channels - dimension 1 of
      // output
      for (int i = 0; i < resultRank; ++i)
        if (i != 1)
          addedDimensions.push_back(i);
      outputTensor = linalg::BroadcastOp::create(rewriter, loc, bias,
                                                 initTensor, addedDimensions)
                         ->getResult(0);
    }

    auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
    auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);

    Value inputStride =
        arith::FloorDivSIOp::create(rewriter, loc, inChannels, groups);
    Value weightStride =
        arith::FloorDivSIOp::create(rewriter, loc, weightBatch, groups);

    SmallVector<Value> zeroOffsets(
        inRank,
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0)));
    SmallVector<Value> unitStrides(
        inRank,
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1)));
    SmallVector<Value> outDimSlice(outDims);
    outDimSlice[1] = weightStride;
    SmallVector<Value> inputSliceSizes{inBatch, inputStride};
    inputSliceSizes.append(inDims);
    SmallVector<Value> weightSliceSizes{weightStride, weightChannels};
    weightSliceSizes.append(weightDims);

    Value conv;
    // the code so far is able to respect all numSpatialDims
    // the code below this point is numSpatialDims specific and numGroups
    // specific
    // TODO: factor out the above code into a helper function, and then separate
    // convolution into:
    // - grouped 1d-3d
    // - grouped 1d-3d (quantized)
    // - ungrouped 1d-3d
    if (numGroups == 1 && !inputZp) {
      switch (numSpatialDims) {
      case 1:
        conv = linalg::Conv1DNcwFcwOp::create(
                   rewriter, loc, outputTensor.getType(),
                   ValueRange{paddedInput, weight}, outputTensor, stridesAttr,
                   dilationAttr)
                   .getResult(0);
        break;
      case 2:
        conv = linalg::Conv2DNchwFchwOp::create(
                   rewriter, loc, outputTensor.getType(),
                   ValueRange{paddedInput, weight}, outputTensor, stridesAttr,
                   dilationAttr)
                   .getResult(0);
        break;
      case 3:
        conv = linalg::Conv3DNcdhwFcdhwOp::create(
                   rewriter, loc, outputTensor.getType(),
                   ValueRange{paddedInput, weight}, outputTensor, stridesAttr,
                   dilationAttr)
                   .getResult(0);
        break;
      default:
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only 1D, 2D, and 3D convolution supported");
      };
      Type newResultType = getTypeConverter()->convertType(op.getType());
      if (accumulatorDType != resultDTy) {
        Type resultElementType =
            cast<RankedTensorType>(newResultType).getElementType();
        conv = torch_to_linalg::convertTensorToElementType(rewriter, loc, conv,
                                                           resultElementType);
      }
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
      return success();
    }

    if (numGroups == 1 && inputZp) {
      switch (numSpatialDims) {
      case 2:
        conv = linalg::Conv2DNchwFchwQOp::create(
                   rewriter, loc, outputTensor.getType(),
                   ValueRange{paddedInput, weight, inputZp, weightZp},
                   outputTensor, stridesAttr, dilationAttr)
                   .getResult(0);
        break;
      case 3: {
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

        paddedInput =
            transposeValue(op.getLoc(), paddedInput, inPerms, rewriter);
        weight = transposeValue(op.getLoc(), weight, weightPerms, rewriter);
        outputTensor =
            transposeValue(op.getLoc(), outputTensor, inPerms, rewriter);

        conv = linalg::Conv3DNdhwcDhwcfQOp::create(
                   rewriter, loc, outputTensor.getType(),
                   ValueRange{paddedInput, weight, inputZp, weightZp},
                   outputTensor, stridesAttr, dilationAttr)
                   .getResult(0);

        llvm::SmallVector<int64_t> outPerms;
        outPerms.push_back(0);
        outPerms.push_back(inPerms.size() - 1);
        for (size_t i = 0; i < numSpatialDims; ++i) {
          outPerms.push_back(i + 1);
        }
        conv = transposeValue(op.getLoc(), conv, outPerms, rewriter);

        break;
      }
      default:
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only 1D, 2D, and 3D convolution supported");
      };

      Type newResultType = getTypeConverter()->convertType(op.getType());
      if (accumulatorDType != resultDTy) {
        Type resultElementType =
            cast<RankedTensorType>(newResultType).getElementType();
        conv = torch_to_linalg::convertTensorToElementType(rewriter, loc, conv,
                                                           resultElementType);
      }
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
      return success();
    }

    // Special depthwise case: Cin = Cout = groups.
    // Note: pytorch considers Cin == groups (Cout possibly a non-zero multiple
    // of groups) to be depthwise in their documentation, but the linalg ops
    // apparently disagree.
    auto inShape = makeShapeTorchCompatible(
        cast<RankedTensorType>(input.getType()).getShape());
    auto weightShape = makeShapeTorchCompatible(
        cast<RankedTensorType>(weight.getType()).getShape());
    if (inShape[1] == numGroups && weightShape[0] == numGroups &&
        weightShape[1] == 1) {
      // Collapse weight shape (C/G == 1)
      SmallVector<ReassociationIndices> collapsedDims = {{0, 1}};
      SmallVector<int64_t> collapsedShape{weightShape[0] * weightShape[1]};
      for (unsigned i = 0; i < numSpatialDims; i++) {
        collapsedDims.push_back({i + 2});
        collapsedShape.push_back(weightShape[i + 2]);
      }
      Type collapsedType = RankedTensorType::get(
          makeShapeLLVMCompatible(collapsedShape), weightDTy);
      Value collapsedWeight = tensor::CollapseShapeOp::create(
          rewriter, loc, collapsedType, weight, collapsedDims);
      if (!inputZp) {
        switch (numSpatialDims) {
        case 1:
          conv = linalg::DepthwiseConv1DNcwCwOp::create(
                     rewriter, loc, outputTensor.getType(),
                     ValueRange{paddedInput, collapsedWeight}, outputTensor,
                     stridesAttr, dilationAttr)
                     .getResult(0);
          break;
        case 2:
          conv = linalg::DepthwiseConv2DNchwChwOp::create(
                     rewriter, loc, outputTensor.getType(),
                     ValueRange{paddedInput, collapsedWeight}, outputTensor,
                     stridesAttr, dilationAttr)
                     .getResult(0);
          break;
        default:
          return rewriter.notifyMatchFailure(
              op, "unimplemented: only 1D and 2D depthwise convolution "
                  "supported for special case of group convolution");
        };
      } else {
        if (numSpatialDims != 2)
          return rewriter.notifyMatchFailure(
              op, "unimplemented: only 2D depthwise quantized convolution "
                  "supported for special case of group convolution");

        // currently, the only named depthwise qconv op is nhwc_hwc
        // input: nchw -> nhwc; weight (collapsed): chw -> hwc
        // linalg conv result nhwc -> nchw
        // inPerms = [0, 2, 3, 1]
        // weightPerms = [1, 2, 0]
        // resultPerms = [0, 3, 1, 2]
        llvm::SmallVector<int64_t> inPerms, weightPerms, resultPerms;
        inPerms.push_back(0);
        resultPerms.append({0, static_cast<int64_t>(numSpatialDims + 1)});
        for (size_t i = 0; i < numSpatialDims; ++i) {
          inPerms.push_back(i + 2);
          weightPerms.push_back(i + 1);
          resultPerms.push_back(i + 1);
        }
        inPerms.push_back(1);
        weightPerms.push_back(0);

        paddedInput =
            transposeValue(op.getLoc(), paddedInput, inPerms, rewriter);
        collapsedWeight =
            transposeValue(op.getLoc(), collapsedWeight, weightPerms, rewriter);
        outputTensor =
            transposeValue(op.getLoc(), outputTensor, inPerms, rewriter);

        conv = linalg::DepthwiseConv2DNhwcHwcQOp::create(
                   rewriter, loc, outputTensor.getType(),
                   ValueRange{paddedInput, collapsedWeight, inputZp, weightZp},
                   outputTensor, stridesAttr, dilationAttr)
                   .getResult(0);
        // convert output nhwc -> nchw
        conv = transposeValue(op.getLoc(), conv, resultPerms, rewriter);
      }

      Type newResultType = getTypeConverter()->convertType(op.getType());
      if (accumulatorDType != resultDTy) {
        Type resultElementType =
            cast<RankedTensorType>(newResultType).getElementType();
        conv = torch_to_linalg::convertTensorToElementType(rewriter, loc, conv,
                                                           resultElementType);
      }

      if (is1DGroupConv) {
        // Squeezing the last dim of the result of conv.
        auto squeezeOutputInfo = squeezeTensor(rewriter, op, conv, /*dim=*/-1);
        if (failed(squeezeOutputInfo)) {
          return rewriter.notifyMatchFailure(op,
                                             "cannot generate squeeze tensor");
        }
        conv = squeezeOutputInfo.value();
      }

      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
      return success();
    }

    if (numSpatialDims != 2 && numSpatialDims != 3)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only 2D and 3D grouped convolution supported");
    if (numSpatialDims == 3 && inputZp) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: quantized 3D grouped convolution not supported");
    }

    // Grouped case, use the grouped conv linalg op
    auto expandGroups = [&](Value tensor, size_t dim) {
      auto inType = cast<RankedTensorType>(tensor.getType());
      auto inShape = makeShapeTorchCompatible(inType.getShape());

      SmallVector<int64_t> outShape;
      for (auto i = 0; i < (long)inShape.size(); i++) {
        if (i == 1) {
          outShape.push_back(numGroups);
        }
        if (i == (long)dim) {
          outShape.push_back(inShape[i] == kUnknownSize
                                 ? kUnknownSize
                                 : inShape[i] / numGroups);
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
      return tensor::ExpandShapeOp::create(rewriter, loc, retType, tensor,
                                           indices);
    };

    Value paddedInputExpanded = expandGroups(paddedInput, 1);
    // If we have a transposed convolution, this needs to be handled before
    // dimension permutation. See comments in the expandWeight lambda definition
    // for details.
    weight = transposed ? weight : expandWeight(weight);
    auto expandOutputTensor = expandGroups(outputTensor, 1);

    if (numSpatialDims == 2) {
      // 2D grouped convolution
      if (!inputZp) {
        conv = linalg::Conv2DNgchwGfchwOp::create(
                   rewriter, loc, expandOutputTensor.getResultType(),
                   ValueRange{paddedInputExpanded, weight},
                   expandOutputTensor.getResult(), stridesAttr, dilationAttr)
                   .getResult(0);
      } else {
        conv = linalg::Conv2DNgchwGfchwQOp::create(
                   rewriter, loc, expandOutputTensor.getResultType(),
                   ValueRange{paddedInputExpanded, weight, inputZp, weightZp},
                   expandOutputTensor.getResult(), stridesAttr, dilationAttr)
                   .getResult(0);
      }
    } else if (numSpatialDims == 3) {
      // MLIR does not have a named 3D grouped convolution op, so we use
      // linalg.generic instead.
      AffineExpr d0, d1, d2, d3, d4, d5, d6, d7, d8, d9;
      bindDims(context, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9);

      SmallVector<AffineExpr> inputExprs = {
          d0,                                        // N
          d1,                                        // G
          d6,                                        // C/G
          d3 * strideInts[0] + d7 * dilationInts[0], // D
          d4 * strideInts[1] + d8 * dilationInts[1], // H
          d5 * strideInts[2] + d9 * dilationInts[2]  // W
      };

      SmallVector<AffineExpr> weightExprs = {
          d1, // G
          d2, // F/G
          d6, // C/G
          d7, // KD
          d8, // KH
          d9  // KW
      };

      SmallVector<AffineExpr> outputExprs = {
          d0, // N
          d1, // G
          d2, // F/G
          d3, // OD
          d4, // OH
          d5, // OW
      };

      SmallVector<AffineMap> indexingMaps = {
          AffineMap::get(10, 0, inputExprs, rewriter.getContext()),
          AffineMap::get(10, 0, weightExprs, rewriter.getContext()),
          AffineMap::get(10, 0, outputExprs, rewriter.getContext())};

      SmallVector<utils::IteratorType> iteratorTypes = {
          utils::IteratorType::parallel,  // N
          utils::IteratorType::parallel,  // G
          utils::IteratorType::parallel,  // F/G
          utils::IteratorType::parallel,  // OD
          utils::IteratorType::parallel,  // OH
          utils::IteratorType::parallel,  // OW
          utils::IteratorType::reduction, // C/G
          utils::IteratorType::reduction, // KD
          utils::IteratorType::reduction, // KH
          utils::IteratorType::reduction  // KW
      };

      conv = linalg::GenericOp::create(
                 rewriter, loc, expandOutputTensor.getResultType(),
                 ValueRange{paddedInputExpanded, weight},
                 expandOutputTensor.getResult(), indexingMaps, iteratorTypes,
                 [&](OpBuilder &b, Location loc, ValueRange args) {
                   Value input = args[0];
                   Value weight = args[1];
                   Value output = args[2];

                   // Convert input and weight to accumulator type if needed
                   Type accType = output.getType();
                   if (input.getType() != accType) {
                     input = arith::ExtFOp::create(b, loc, accType, input);
                   }
                   if (weight.getType() != accType) {
                     weight = arith::ExtFOp::create(b, loc, accType, weight);
                   }

                   Value mul = arith::MulFOp::create(b, loc, input, weight);
                   Value add = arith::AddFOp::create(b, loc, mul, output);
                   linalg::YieldOp::create(b, loc, add);
                 })
                 .getResult(0);
    }
    conv = tensor::CollapseShapeOp::create(
        rewriter, loc, outputTensor.getType(), conv,
        expandOutputTensor.getReassociationIndices());
    Type newResultType = getTypeConverter()->convertType(op.getType());
    if (accumulatorDType != resultDTy) {
      Type resultElementType =
          cast<RankedTensorType>(newResultType).getElementType();
      conv = torch_to_linalg::convertTensorToElementType(rewriter, loc, conv,
                                                         resultElementType);
    }

    if (is1DGroupConv) {
      // Squeezing the last dim of the result of conv.
      auto squeezeOutputInfo = squeezeTensor(rewriter, op, conv, /*dim=*/-1);
      if (failed(squeezeOutputInfo)) {
        return rewriter.notifyMatchFailure(op,
                                           "cannot generate squeeze tensor");
      }
      conv = squeezeOutputInfo.value();
    }
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
    return success();
  }

  static Value createTransposedInputPadding(
      Value inBatch, Value inChannels, SmallVector<Value> &inDims,
      SmallVector<Value> &weightDims, SmallVector<Value> &paddingIntValues,
      SmallVector<Value> &strideIntValues,
      SmallVector<Value> &dilationIntValues,
      SmallVector<Value> &outputPaddingIntValues, Value input, Type inputDTy,
      Value pad, PatternRewriter &rewriter, Location loc, size_t numSpatialDims,
      Value c0, Value c1);
};
} // namespace

Value ConvertAtenConvolutionOp::createTransposedInputPadding(
    Value inBatch, Value inChannels, SmallVector<Value> &inDims,
    SmallVector<Value> &weightDims, SmallVector<Value> &paddingIntValues,
    SmallVector<Value> &strideIntValues, SmallVector<Value> &dilationIntValues,
    SmallVector<Value> &outputPaddingIntValues, Value input, Type inputDTy,
    Value pad, PatternRewriter &rewriter, Location loc, size_t numSpatialDims,
    Value c0, Value c1) {
  // Calculate padded input size, allocate tensor
  SmallVector<Value> outerSizes{inBatch, inChannels};
  SmallVector<Value> innerSizes{inBatch, inChannels};
  SmallVector<Value> insertSliceOffsets{c0, c0};

  SmallVector<Value> inputSizes = getTensorSizes(rewriter, loc, input);
  SmallVector<Value> sliceSizes{inputSizes[0], inputSizes[1]};

  // For the case in which the padding dimension value is negative,
  // we will need to shrink the dimension. Note in the PyTorch
  // ConvTranspose2d operator documentation that the padding is
  // defined by dilation * (kernel_size - 1) - padding. If the
  // resulting padding is negative, PyTorch will extract elements
  // from both sides of the dimension.
  SmallVector<Value> extractSliceOffsets{c0, c0};
  bool anyDimensionPaddingIsNegative = false;

  Value c2 = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(2));

  for (size_t i = 0; i < numSpatialDims; i++) {
    Value innerSize = rewriter.createOrFold<arith::SubIOp>(loc, inDims[i], c1);
    innerSize = rewriter.createOrFold<arith::MulIOp>(
        loc, innerSize, castIntToIndex(rewriter, loc, strideIntValues[i]));
    innerSize = rewriter.createOrFold<arith::AddIOp>(loc, innerSize, c1);

    Value offset = rewriter.createOrFold<arith::SubIOp>(loc, weightDims[i], c1);
    offset = rewriter.createOrFold<arith::MulIOp>(
        loc, offset, castIntToIndex(rewriter, loc, dilationIntValues[i]));
    offset = rewriter.createOrFold<arith::SubIOp>(
        loc, offset, castIntToIndex(rewriter, loc, paddingIntValues[i]));

    Value outerSize = rewriter.createOrFold<arith::MulIOp>(loc, offset, c2);
    outerSize = rewriter.createOrFold<arith::AddIOp>(loc, outerSize, innerSize);
    outerSize = rewriter.createOrFold<arith::AddIOp>(
        loc, outerSize,
        castIntToIndex(rewriter, loc, outputPaddingIntValues[i]));

    outerSizes.push_back(outerSize);
    if (isValueNegative(offset)) {
      // Make the negative value positive by multiplying by -1.
      anyDimensionPaddingIsNegative = true;
      auto offsetType = offset.getType();
      auto negOneConst = rewriter.createOrFold<arith::ConstantOp>(
          loc, offsetType, rewriter.getIntegerAttr(offsetType, -1));
      auto posOffset =
          rewriter.createOrFold<arith::MulIOp>(loc, offset, negOneConst);

      // Compute the reduced dimension size due to negative padding.
      auto sizeReduction =
          rewriter.createOrFold<arith::MulIOp>(loc, posOffset, c2);
      sliceSizes.push_back(rewriter.createOrFold<arith::SubIOp>(
          loc, inputSizes[i + 2], sizeReduction));

      extractSliceOffsets.push_back(posOffset);
      insertSliceOffsets.push_back(c0);
    } else {
      sliceSizes.push_back(inputSizes[i + 2]);
      extractSliceOffsets.push_back(c0);
      insertSliceOffsets.push_back(offset);
    }
  }
  Value initTensor = createInitTensor(rewriter, loc, outerSizes, inputDTy, pad);

  // Insert input into allocated tensor
  SmallVector<Value> strideIndexValues{c1, c1};
  for (auto stride : strideIntValues)
    strideIndexValues.push_back(castIntToIndex(rewriter, loc, stride));

  auto insertSliceOpInput = input;
  if (anyDimensionPaddingIsNegative) {
    insertSliceOpInput = tensor::ExtractSliceOp::create(
        rewriter, loc,
        torch_to_linalg::removeSizeInformation(rewriter, loc, input),
        extractSliceOffsets, sliceSizes, strideIndexValues);
  }

  auto paddedInput = tensor::InsertSliceOp::create(
      rewriter, loc,
      torch_to_linalg::removeSizeInformation(rewriter, loc, insertSliceOpInput),
      initTensor, insertSliceOffsets, sliceSizes, strideIndexValues);
  return paddedInput;
}

namespace {

/// Creates coefficients based on DFT definition, see
/// https://en.wikipedia.org/wiki/Discrete_Fourier_transform.
Value getDFTMatmulCoeff(OpBuilder b, Location loc,
                        RankedTensorType matrixType) {

  ComplexType complexTy = llvm::cast<ComplexType>(matrixType.getElementType());
  mlir::FloatType floatType =
      llvm::cast<mlir::FloatType>(complexTy.getElementType());

  // scale = 2 * pi / N
  double scale = 2 * M_PI / matrixType.getDimSize(0);

  SmallVector<std::complex<APFloat>> values;
  for (auto i : llvm::seq<unsigned>(0, matrixType.getDimSize(0))) {
    for (auto j : llvm::seq<unsigned>(0, matrixType.getDimSize(1))) {
      double v = scale * i * j;
      double realV = cos(v);
      double imagV = -sin(v);

      bool unused;
      APFloat real(realV);
      real.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &unused);
      APFloat imag(imagV);
      imag.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &unused);

      values.push_back(std::complex<APFloat>(real, imag));
    }
  }
  return arith::ConstantOp::create(b, loc, matrixType,
                                   DenseElementsAttr::get(matrixType, values));
}

struct ConvertAtenFftRfftOp final : OpConversionPattern<AtenFftRfftOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenFftRfftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = adaptor.getSelf();

    int64_t dim;
    auto dimVal = op.getDim();
    if (isa<torch::Torch::NoneType>(dimVal.getType())) {
      dim = -1;
    } else if (!matchPattern(dimVal, torch::Torch::m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires dim to be constant");
    }

    if (!isa<torch::Torch::NoneType>(op.getN().getType())) {
      return rewriter.notifyMatchFailure(op, "unimplemented: parameter n");
    }

    if (!isa<torch::Torch::NoneType>(op.getNorm().getType())) {
      return rewriter.notifyMatchFailure(op, "unimplemented: parameter norm");
    }

    RankedTensorType inputType =
        cast<RankedTensorType>(adaptor.getSelf().getType());
    if (!inputType.hasRank()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported: only ranked tensors are supported");
    }

    const ArrayRef<int64_t> inputShape = inputType.getShape();
    dim += dim < 0 ? inputShape.size() : 0;

    const int64_t fftLength = inputShape[dim];
    if (fftLength == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(
          op, "unsupported: FFT signal length must be static");
    }
    const int64_t rank = inputType.getRank();
    const int64_t lastDim = rank - 1;
    const int64_t outputFftDim = fftLength / 2 + 1;
    const bool needTranspose = dim != lastDim;

    // Transpose if FFT dimension is not the last one
    llvm::SmallVector<int64_t> perms = llvm::to_vector(llvm::seq(rank));
    std::swap(perms[dim], perms[lastDim]);
    if (needTranspose) {
      self = transposeValue(loc, self, perms, rewriter);
    }

    RankedTensorType newResultType = llvm::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    ComplexType complexElemType =
        llvm::cast<ComplexType>(newResultType.getElementType());
    Type elemType = complexElemType.getElementType();

    // coeffMatrix : tensor<fftLength x outputFftDim x complex<f32>>
    RankedTensorType coeffType =
        RankedTensorType::get({fftLength, outputFftDim}, complexElemType);
    // coeffMatrix(n,m) = cos(2 pi n m / N) - j sin(2 pi n m / N)
    Value coeffMatrix = getDFTMatmulCoeff(rewriter, loc, coeffType);

    // #matmul_trait = {
    //   indexing_maps = [
    //     affine_map<(d_0, ... d_m, f, o) -> (d_0, ... d_m, f)>,
    //     affine_map<(d_0, ... d_m, f, o) -> (f, o)>,
    //     affine_map<(d_0, ... d_m, f, o) -> (d_0, ... d_m, o)>
    //   ],
    //   iterator_types = ["parallel", ..., "parallel", "reduction", "parallel"]
    // }
    // linalg.generic #matmul_trait
    //   ins(%A, %B : tensor<D_0 x ... x D_m x fftLength x f32>,
    //                tensor<fftLength x outputFftDim x complex<f32>>)
    //   outs(%C : tensor<D_0 x ... x D_m x outputFftDim x complex<f32>>) {
    //   ^bb0(%a: f32, %b: complex<f32>, %c: complex<f32>) :
    //     %re = complex.re %b : f32
    //     %im = complex.im %b : f32
    //     %mulre = arith.mulf %a, %re: f32
    //     %mulim = arith.mulf %a, %im: f32
    //     %mulcplx = complex.create %mulre, %mulim : complex<f32>
    //     %add = complex.add %c, %mulcplx: complex<f32>
    //     linalg.yield %add : complex<f32>
    // } -> (tensor<D_0 x ... x D_m x outputFftDim x complex<f32>>)

    Value lhs = self;
    Value rhs = coeffMatrix;
    RankedTensorType lhsType = llvm::cast<RankedTensorType>(lhs.getType());
    ArrayRef<int64_t> lhsShape(lhsType.getShape());
    ArrayRef<int64_t> rhsShape(coeffType.getShape());

    unsigned batchRank = lhsShape.size() - 1;

    SmallVector<AffineExpr> lhsExpr;
    SmallVector<AffineExpr> rhsExpr;
    SmallVector<AffineExpr> outExpr;
    SmallVector<utils::IteratorType> iteratorTypes(
        batchRank, utils::IteratorType::parallel);
    SmallVector<Value> resultShape;
    for (unsigned i = 0; i < batchRank; i++) {
      lhsExpr.push_back(rewriter.getAffineDimExpr(i));
      outExpr.push_back(rewriter.getAffineDimExpr(i));
      resultShape.push_back(getDimOp(rewriter, loc, lhs, i));
    }
    unsigned fIdx = batchRank, oIdx = batchRank + 1;
    lhsExpr.insert(lhsExpr.end(), {rewriter.getAffineDimExpr(fIdx)});
    rhsExpr.insert(rhsExpr.end(), {rewriter.getAffineDimExpr(fIdx),
                                   rewriter.getAffineDimExpr(oIdx)});
    outExpr.insert(outExpr.end(), {rewriter.getAffineDimExpr(oIdx)});
    resultShape.insert(resultShape.end(),
                       {getDimOp(rewriter, loc, rhs, rhsShape.size() - 1)});

    Value zeroTensor =
        createZeroInitTensor(rewriter, loc, resultShape, complexElemType);
    auto indexingMaps = AffineMap::inferFromExprList(
        {lhsExpr, rhsExpr, outExpr}, rewriter.getContext());
    iteratorTypes.insert(iteratorTypes.end(), {utils::IteratorType::reduction,
                                               utils::IteratorType::parallel});

    Value complexRes =
        linalg::GenericOp::create(
            rewriter, loc, zeroTensor.getType(),
            /*inputs=*/ValueRange{lhs, rhs},
            /*outputs=*/zeroTensor, indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value l = args[0], r = args[1], res = args[2];
              Value re = complex::ReOp::create(b, loc, elemType, r);
              Value im = complex::ImOp::create(b, loc, elemType, r);
              Value mulRe = arith::MulFOp::create(b, loc, l, re);
              Value mulIm = arith::MulFOp::create(b, loc, l, im);
              Value mulCplx = complex::CreateOp::create(b, loc, complexElemType,
                                                        mulRe, mulIm);
              Value add = complex::AddOp::create(b, loc, mulCplx, res);
              linalg::YieldOp::create(b, loc, add);
            })
            .getResult(0);

    // Transpose back
    if (needTranspose) {
      complexRes = transposeValue(loc, complexRes, perms, rewriter);
    }

    rewriter.replaceOp(op, complexRes);
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
  target.addIllegalOp<AtenFftRfftOp>();
  patterns.add<ConvertAtenFftRfftOp>(typeConverter, context);
}
