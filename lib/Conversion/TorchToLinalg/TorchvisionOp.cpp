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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
// #include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
// #include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
// #include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
static Value calculateIoU(OpBuilder &b, Location loc, Value box1, Value box2) {
  // box format: [x1, y1, x2, y2] with 0 <= x1 < x2 and 0 <= y1 < y2
  Value idx0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value idx1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value idx2 = b.create<arith::ConstantIndexOp>(loc, 2);
  Value idx3 = b.create<arith::ConstantIndexOp>(loc, 3);
  Value b1x1 = b.create<tensor::ExtractOp>(loc, box1, ValueRange{idx0});
  Value b1y1 = b.create<tensor::ExtractOp>(loc, box1, ValueRange{idx1});
  Value b1x2 = b.create<tensor::ExtractOp>(loc, box1, ValueRange{idx2});
  Value b1y2 = b.create<tensor::ExtractOp>(loc, box1, ValueRange{idx3});
  Value b2x1 = b.create<tensor::ExtractOp>(loc, box2, ValueRange{idx0});
  Value b2y1 = b.create<tensor::ExtractOp>(loc, box2, ValueRange{idx1});
  Value b2x2 = b.create<tensor::ExtractOp>(loc, box2, ValueRange{idx2});
  Value b2y2 = b.create<tensor::ExtractOp>(loc, box2, ValueRange{idx3});

  // Calculate intersection width and height
  Value intersectX1 = b.create<arith::MaximumFOp>(loc, b1x1, b2x1);
  Value intersectY1 = b.create<arith::MaximumFOp>(loc, b1y1, b2y1);
  Value intersectX2 = b.create<arith::MinimumFOp>(loc, b1x2, b2x2);
  Value intersectY2 = b.create<arith::MinimumFOp>(loc, b1y2, b2y2);
  // Width = max(0, intersectX2 - intersectX1)
  Value zero = b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(0.0));
  Value width = b.create<arith::SubFOp>(loc, intersectX2, intersectX1);
  width = b.create<arith::MaximumFOp>(loc, width, zero);
  // Height = max(0, intersectY2 - intersectY1)
  Value height = b.create<arith::SubFOp>(loc, intersectY2, intersectY1);
  height = b.create<arith::MaximumFOp>(loc, height, zero);
  // Intersection area = width * height
  Value intersectionArea = b.create<arith::MulFOp>(loc, width, height);

  // Calculate area of box1: (b1x2 - b1x1) * (b1y2 - b1y1)
  Value width1 = b.create<arith::SubFOp>(loc, b1x2, b1x1);
  Value height1 = b.create<arith::SubFOp>(loc, b1y2, b1y1);
  Value area1 = b.create<arith::MulFOp>(loc, width1, height1);
  // Calculate area of box2: (b2x2 - b2x1) * (b2y2 - b2y1)
  Value width2 = b.create<arith::SubFOp>(loc, b2x2, b2x1);
  Value height2 = b.create<arith::SubFOp>(loc, b2y2, b2y1);
  Value area2 = b.create<arith::MulFOp>(loc, width2, height2);
  // Union area = area1 + area2 - intersectionArea
  Value unionArea = b.create<arith::AddFOp>(loc, area1, area2);
  unionArea = b.create<arith::SubFOp>(loc, unionArea, intersectionArea);

  return b.create<arith::DivFOp>(loc, intersectionArea, unionArea);
}

class ConvertTorchvisionNmsOp : public OpConversionPattern<TorchvisionNmsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TorchvisionNmsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }

    Location loc = op->getLoc();
    Value boxes = adaptor.getOperands()[0];
    Value scores = adaptor.getOperands()[1];
    Value iouThreshold = adaptor.getOperands()[2];

    auto boxesType = cast<RankedTensorType>(boxes.getType());
    auto scoresType = cast<RankedTensorType>(scores.getType());
    if (!boxesType || !scoresType) {
      return failure();
    }

    // Calculate IoU for each pair of boxes
    Type boxesElementType = boxesType.getElementType();
    Value cst0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(boxesElementType));
    int64_t boxesSize = boxesType.getShape()[0];
    Value iouEmpty =
        rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{boxesSize, boxesSize}, boxesElementType);
    Value iouOutput =
        rewriter.create<linalg::FillOp>(loc, cst0, iouEmpty).getResult(0);


    // AffineExpr d0, d1;
    // bindDims(getContext(), d0, d1);
    // auto c0 = rewriter.getAffineConstantExpr(0);
    // auto map = AffineMap::get(2, 0, {d0, d1}, rewriter.getContext());
    // auto map1 = AffineMap::get(2, 0, {d0, d1}, rewriter.getContext());
    // auto map2 = AffineMap::get(2, 0, {d0, d1}, rewriter.getContext());
    // SmallVector<AffineMap> indexingMaps = {map, map1, map2};
    AffineMap inputMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)}, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)}, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {inputMap, inputMap, outputMap};
    // SmallVector<utils::IteratorType> iteratorTypes(
    //     2, utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::reduction};
    // Create the linalg.generic operation
    Value result =
        rewriter
            .create<linalg::GenericOp>(
                loc,
                /*resultTypes=*/iouOutput.getType(),
                /*inputs=*/ValueRange{boxes, boxes},
                /*outputs=*/iouOutput,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value box1 = args[0], box2 = args[1], out = args[2];
                  box1.getType().dump();
                  box2.getType().dump();
                  out.getType().dump();
                  Value resultValue = calculateIoU(b, loc, box1, box2);
                  b.create<linalg::YieldOp>(loc, resultValue);
                })
            .getResult(0);


    // Create a mask tensor where we mark suppressed boxes (0 for keep, 1 for suppress)
    // Value maskEmpty = rewriter.create<scoresType>(loc, scoresType.getShape(), rewriter.getI1Type());
    // Value maskOutput =
    //     rewriter.create<linalg::FillOp>(loc, cst0, maskEmpty).getResult(0);
    // maskOutput.dump();

    // Value maskTensor =
    //     rewriter
    //         .create<linalg::GenericOp>(
    //             loc, maskOutput.getType(), ValueRange{iouTensor, scores}, maskOutput,
    //             /*indexing_maps=*/AffineMap::inferFromExprList({
    //                 {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)}
    //             }),
    //             /*iterator_types=*/ArrayRef<StringRef>{"parallel", "parallel"},
    //             [&](OpBuilder &b, Location loc, ValueRange args) {
    //               Value iouValue = args[0];
    //               Value score = args[1];
    //               // Check conditions to suppress based on IoU and score thresholds.
    //               Value isSuppressed = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, iouValue, iouThreshold);
    //               b.create<linalg::YieldOp>(loc, isSuppressed);
    //             })
    //         .getResult(0);

    // // Convert the mask tensor to the desired output format (filtered boxes).
    // rewriter.replaceOp(op, maskTensor);


    llvm::outs() << "!!!\n";
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateTorchvisionPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<TorchvisionNmsOp>();
  patterns.add<ConvertTorchvisionNmsOp>(typeConverter, context);
}
