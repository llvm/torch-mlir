//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/E2E/E2E.h"
#include "PassDetail.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
class LowerTensorStoreOp : public OpConversionPattern<TensorStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorStoreOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorStoreOp::OperandAdaptor adaptor(operands);
    // The tensor has been converted to an unranked memref. We need to cast
    // it to the original memref type and copy it to the destination.
    //
    // TODO: Can we have a conversion infrastructure that doesn't have
    // patterns that doesn't couple type conversions and the patterns. That
    // is, patterns should be "context free" and locally expand to always
    // valid IR without relying on some side-channel TypeConverter to do
    // something else to make the IR valid.
    auto memref = rewriter.create<MemRefCastOp>(
        op.getLoc(), op.memref().getType(), adaptor.tensor());
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, memref, adaptor.memref());
    return success();
  }
};
} // namespace

namespace {
class LowerTensorLoadOp : public OpConversionPattern<TensorLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorLoadOp::OperandAdaptor adaptor(operands);
    auto type = UnrankedMemRefType::get(op.getType().getElementType(), 0);
    // TODO: This won't work. The LLVM unranked memref calling convention
    // doesn't allow returning an unranked memref becuase it lowers it to
    // 'int64 rank, void *descriptor' but in this case the descriptor will
    // likely be on the stack, so when returning the descriptor pointer it
    // will be use-after-return.
    //
    // We could directly emit LLVM IR mallocing the memref struct on the
    // heap or do a conversion to out params and require a preallocated
    // memref out descriptor (perhaps preallocated to a fixed upper bound
    // rank).
    //
    // But a more holistic approach seems needed:
    // 1. Use custom npcomp runtime types at function boundaries. These can
    // be approximately like IREE's !hal.buffer_view, namely a type-erased,
    // shape-erased, ref-counted multidimensional array of dense primitive
    // types. (Something like Py_buffer from the python buffer protocol is
    // another potential inspiration)
    //   - [IREE HAL buffer view](https://github.com/google/iree/blob/634136f03c144ad3acd2f28cd87785b0b6b572ac/iree/hal/api_detail.h#L26)
    //   - [Python buffer protocol](https://docs.python.org/3/c-api/buffer.html)
    // 2. Use a custom LLVM conversion that creates the memref types.
    // For example, have an op
    // ```
    // npcomp_rt.to_memref %buf_view : !npcomp_rt.buffer_view -> memref<?xf32>
    // ```
    // with a custom LLVM lowering that expands to all the right stuff.
    rewriter.replaceOpWithNewOp<MemRefCastOp>(op, type, adaptor.memref());
    return success();
  }
};
} // namespace

namespace {
class LowerShapeOfOp : public OpConversionPattern<shape::ShapeOfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::ShapeOfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::ShapeOfOp::OperandAdaptor adaptor(operands);
    auto tensorType = op.arg().getType().cast<RankedTensorType>();
    auto rankedMemRefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto rankedMemRef = rewriter.create<MemRefCastOp>(
        op.getLoc(), rankedMemRefType, adaptor.arg());
    SmallVector<Value, 6> extents;
    for (int i = 0, e = tensorType.getRank(); i < e; i++)
      extents.push_back(rewriter.create<DimOp>(op.getLoc(), rankedMemRef, i));
    rewriter.replaceOpWithNewOp<tcp::ShapeFromExtentsOp>(op, extents);
    return success();
  }
};
} // namespace

namespace {
// This pass lowers tensor types to a calling convention where all tensors
// are passed as UnrankedMemRefType. This allows the current StandardToLLVM
// lowering to return them as `size_t rank, void *descriptor` which is easy
// to bridge across a fixed C ABI. (otherwise it specializes the signature
// to the memref rank, which is very difficult to interoperate with).
class LowerToMemRefABI : public LowerToMemRefABIBase<LowerToMemRefABI> {
  void runOnOperation() {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter converter;
    converter.addConversion([](TensorType type) {
      return UnrankedMemRefType::get(type.getElementType(), /*memorySpace=*/0);
    });
    // Mark UnrankedMemRefType as "legal". This is the awkward way of doing
    // that.
    // TODO: Commenting this out causes a seemingly unrelated crash.
    // Redesign MLIR's type conversion system to have a clearer mental
    // model and not be so flaky.
    converter.addConversion([](UnrankedMemRefType type) { return type; });

    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    populateFuncOpTypeConversionPattern(patterns, context, converter);
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
      return converter.isSignatureLegal(op.getType());
    });

    patterns.insert<LowerTensorStoreOp>(context);
    target.addIllegalOp<TensorStoreOp>();
    target.addLegalOp<DimOp>();
    target.addLegalOp<MemRefCastOp>();
    target.addLegalOp<linalg::CopyOp>();

    patterns.insert<LowerTensorLoadOp>(context);
    target.addIllegalOp<TensorLoadOp>();

    patterns.insert<LowerShapeOfOp>(context);
    target.addIllegalOp<shape::ShapeOfOp>();
    target.addLegalOp<tcp::ShapeFromExtentsOp>();

    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerToMemRefABIPass() {
  return std::make_unique<LowerToMemRefABI>();
}
