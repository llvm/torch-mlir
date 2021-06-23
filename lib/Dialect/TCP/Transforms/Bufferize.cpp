//===- Bufferize.cpp - Bufferization for TCP dialect -------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Refback/IR/RefbackDialect.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"
#include "npcomp/Dialect/TCP/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;

// TODO: Don't just open-code all shape transfer functions here.
static SmallVector<Value, 6> bypassResultShapes(Operation &op) {
  OpBuilder builder(&op);

  if (auto broadcastTo = dyn_cast<tcp::BroadcastToOp>(op)) {
    return {broadcastTo.shape()};
  }

  if (auto splatted = dyn_cast<tcp::SplattedOp>(op)) {
    return {splatted.shape()};
  }

  if (auto pad = dyn_cast<tcp::PadOp>(op)) {
    SmallVector<Value, 6> outDims;
    auto inputType = pad.operand().getType().cast<RankedTensorType>();
    for (int i = 0, e = inputType.getRank(); i < e; i++) {
      auto dimIndex = builder.create<ConstantIndexOp>(op.getLoc(), i);
      auto lowerExpansion =
        builder.create<tensor::ExtractOp>(op.getLoc(), pad.lowerExpansion(),
            ValueRange({dimIndex}));
      auto upperExpansion =
        builder.create<tensor::ExtractOp>(op.getLoc(), pad.upperExpansion(),
            ValueRange({dimIndex}));
      auto operandDim =
          builder.create<memref::DimOp>(op.getLoc(), pad.operand(), i);
      auto totalExpansion =
        builder.create<AddIOp>(op.getLoc(), lowerExpansion, upperExpansion);
      auto outDim =
        builder.create<AddIOp>(op.getLoc(), totalExpansion, operandDim);
      outDims.push_back(outDim);
    }
    Value outDimTensor = builder.create<tensor::FromElementsOp>(op.getLoc(), ValueRange(outDims));
    return {outDimTensor};
  }

  // No shape transfer function.
  return {};
}

static FailureOr<SmallVector<Value, 6>>
allocateResults(Operation *op, ConversionPatternRewriter &rewriter,
                Location loc,
                SmallVectorImpl<Value> *resultShapesOut = nullptr) {
  auto resultShapes = bypassResultShapes(*op);
  SmallVector<Value, 6> results;
  for (auto t : llvm::zip(op->getResults(), resultShapes)) {
    auto result = std::get<0>(t);
    auto resultShape = std::get<1>(t);
    auto tensorType = result.getType().cast<RankedTensorType>();
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto memref =
        rewriter.create<refback::AllocMemRefOp>(loc, memrefType, resultShape);
    results.push_back(memref);
  }
  if (resultShapesOut)
    resultShapesOut->append(resultShapes.begin(), resultShapes.end());
  return results;
}

namespace {
// TODO: Lower to a "buffer version" of tcp::BroadcastTo instead of directly to
// loops.
class LowerBroadcastToToLoopsPattern
    : public OpConversionPattern<tcp::BroadcastToOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::BroadcastToOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<RankedTensorType>();
    auto inputType = op.operand().getType().cast<RankedTensorType>();
    SmallVector<Value, 6> resultShapes;
    auto resultsOrFailure =
        allocateResults(op, rewriter, op.getLoc(), &resultShapes);
    if (failed(resultsOrFailure))
      return failure();
    Value resultMemref = (*resultsOrFailure)[0];
    auto resultShape = resultShapes[0];
    Value inputMemref = operands[0];

    SmallVector<Value, 6> outputExtents;
    for (int i = 0, e = resultType.getRank(); i < e; i++) {
      Value dimIndex = rewriter.create<ConstantIndexOp>(op.getLoc(), i);
      Value outputExtent = rewriter.create<tensor::ExtractOp>(
          op.getLoc(), resultShape, ValueRange({dimIndex}));
      outputExtents.push_back(outputExtent);
    }
    int rankDiff = resultType.getRank() - inputType.getRank();
    SmallVector<Value, 6> inputDimRequiresBroadcasting;
    for (int i = 0, e = inputType.getRank(); i < e; i++) {
      // Calculate the relevant extents.
      Value inputExtent =
          rewriter.create<memref::DimOp>(op.getLoc(), op.operand(), i);
      inputDimRequiresBroadcasting.push_back(
          rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::ne, inputExtent,
                                  outputExtents[rankDiff + i]));
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      Value c0 = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
      Value c1 = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);

      SmallVector<Value, 6> inductionVariables;
      // Create the (perfectly nested) loops.
      // Loop invariant: At the start of iteration `i`, the rewriter insertion
      // point is inside `i` nested loops.
      for (int i = 0, e = resultType.getRank(); i < e; i++) {
        auto loop = rewriter.create<scf::ForOp>(
            op.getLoc(), c0, outputExtents[i], c1, ValueRange({}));
        Block *body = loop.getBody();
        inductionVariables.push_back(body->getArgument(0));
        // Leave the insertion point at the beginning of the body.
        rewriter.setInsertionPointToStart(body);
      }

      // Create the inner loop body.
      // When reading from the input, clamp any indices for dimensions that are
      // being broadcast.
      SmallVector<Value, 6> inputIndices;
      for (int i = 0, e = inputType.getRank(); i < e; i++) {
        auto c0 = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
        auto select = rewriter.create<SelectOp>(
            op.getLoc(), inputDimRequiresBroadcasting[i], c0,
            inductionVariables[rankDiff + i]);
        inputIndices.push_back(select);
      }
      Value load = rewriter.create<memref::LoadOp>(op.getLoc(), inputMemref,
                                                   inputIndices);
      rewriter.create<memref::StoreOp>(op.getLoc(), load, resultMemref,
                                       inductionVariables);
    }
    rewriter.replaceOp(op, resultMemref);
    return success();
  }
};
} // namespace

namespace {
class BufferizeSplattedOp : public OpConversionPattern<tcp::SplattedOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::SplattedOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultsOrFailure = allocateResults(op, rewriter, op.getLoc());
    if (failed(resultsOrFailure))
      return failure();
    auto results = *resultsOrFailure;
    rewriter.create<linalg::FillOp>(op.getLoc(), op.splatVal(), results[0]);
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

namespace {
class BufferizePadOp : public OpConversionPattern<tcp::PadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::PadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultsOrFailure = allocateResults(op, rewriter, op.getLoc());
    if (failed(resultsOrFailure))
      return failure();
    auto results = *resultsOrFailure;
    auto c1 =
      rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(
            rewriter.getIndexType(), 1));
    SmallVector<Value, 6> offsets, sizes, strides;
    auto resultType = op.getType().cast<RankedTensorType>();
    for (int i = 0, e = resultType.getRank(); i < e; i++) {
      auto dimIndex = rewriter.create<ConstantIndexOp>(op.getLoc(), i);
      auto offset =
        rewriter.create<tensor::ExtractOp>(op.getLoc(), op.lowerExpansion(),
            ValueRange({dimIndex}));
      auto size = rewriter.create<memref::DimOp>(op.getLoc(), op.operand(), i);
      auto stride   = c1;
      offsets.push_back(offset);
      sizes.push_back(size);
      strides.push_back(stride);
    }
    rewriter.create<linalg::FillOp>(op.getLoc(), op.fillVal(), results[0]);
    auto unpadded = rewriter.create<memref::SubViewOp>(
        op.getLoc(), results[0], ValueRange(offsets), ValueRange(sizes),
        ValueRange(strides));
    auto inputMemref = operands[0];
    rewriter.create<linalg::CopyOp>(op.getLoc(), inputMemref, unpadded);
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

namespace {
class TCPBufferizePass : public TCPBufferizeBase<TCPBufferizePass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<refback::RefbackDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;

    RewritePatternSet patterns(context);

    ConversionTarget target(*context);

    // All lowering to buffers involves refback.alloc_memref ops.
    // TODO: This makes the tests cleaner, but otherwise isn't too essential as
    // we can just open-code the extents for the alloc.
    target.addLegalOp<refback::AllocMemRefOp>();

    patterns.add<LowerBroadcastToToLoopsPattern>(typeConverter, context);
    target.addIllegalOp<tcp::BroadcastToOp>();
    patterns.add<BufferizeSplattedOp>(typeConverter, context);
    target.addIllegalOp<tcp::SplattedOp>();
    patterns.add<BufferizePadOp>(typeConverter, context);
    target.addIllegalOp<tcp::PadOp>();

    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::NPCOMP::createTCPBufferizePass() {
  return std::make_unique<TCPBufferizePass>();
}
