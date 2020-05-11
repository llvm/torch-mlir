//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base file for our "end-to-end" npcomp lowering pipeline.
// At the moment, the first "end" is TCF ops and the second "end" is `llvm`
// dialect suitable for jitting.
//
// This is still work-in-progress and not even working end-to-end for the
// most trivial examples, see TODO's in createE2ELoweringPipeline for the
// status.
//
// As a pragmatic matter, I generally tend to drop random passes and stuff
// inside this top-level file and then shard it out to separate files once
// a clear organizing principle arises (to avoid premature organizing).
//
// Once we have end-to-end functionality working, we will throw
// increasingly complex programs and augment this pass pipeline, likely
// introducing better structure and more clear principles.
//
// I wish I had a clear view of how this pipeline should perfectly layer
// ahead of time, but unfortunately I don't since it crosses half a dozen
// abstraction levels / dialects, some of which have no precedent that I'm
// aware of (dynamic-shape-aware, error-aware TCF -> TCP) or very little
// (tensor -> memref/buffer with dynamic shapes, shape -> SSA values for
// ranked shape extents).
//
// Right now there's lots of stuff in this pipeline that is limited to
// special cases where I have an idea of how to elaborate it to the general
// case. The priority is getting and end-to-end flow working that we can
// grow out organically to a curriculum of more complex cases, elaborating
// on the design principles and layering as necessitated by the curriculum.
//
// This should be fun :)
//
//===----------------------------------------------------------------------===//

#include "npcomp/E2E/E2E.h"
#include "PassDetail.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"
#include "npcomp/Conversion/TCPToLinalg/TCPToLinalg.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// ResolveShapeOfOps
//===----------------------------------------------------------------------===//

class ResolveShapeOfOpViaAllocMemRefOp : public OpRewritePattern<shape::ShapeOfOp> {
  public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                  PatternRewriter &rewriter) const override {
      if (auto tensorLoad = llvm::dyn_cast_or_null<TensorLoadOp>(
              op.getOperand().getDefiningOp())) {
        if (auto allocMemRef = llvm::dyn_cast_or_null<tcp::AllocMemRefOp>(
                tensorLoad.getOperand().getDefiningOp())) {
          rewriter.replaceOp(op, allocMemRef.getOperand());
          return success();
        }
      }
      return failure();
    }
};

class ResolveShapeOfOps : public ResolveShapeOfOpsBase<ResolveShapeOfOps> {
  void runOnOperation() {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<ResolveShapeOfOpViaAllocMemRefOp>(context);
    ConversionTarget target(*context);
    //target.addIllegalOp<shape::ShapeOfOp>();
    target.addDynamicallyLegalOp<shape::ShapeOfOp>(
        [](shape::ShapeOfOp shapeOf) {
          // Only shape.shape_of on arguments to the entry block are legal at
          // this point. They are assumed to be resolved eventually via
          // the lowering of the tensor argument to some ABI that has the
          // relevant information available. But this is ABI dependent.
          // TODO: Convince myself that we never need to deal with general
          // block operands, or implement general handling of block
          // operands (need to add new bb operands of !shape.shape type).
          if (auto blockArg = shapeOf.getOperand().dyn_cast<BlockArgument>()) {
            Block *block = blockArg.getOwner();
            if (&block->getParent()->front() == block) {
              return true;
            }
          }
          return false;
        });
    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createResolveShapeOfOpsPass() {
  return std::make_unique<ResolveShapeOfOps>();
}

//===----------------------------------------------------------------------===//
// ResolveTensorLoadStoreOps
//===----------------------------------------------------------------------===//

class ReplaceTensorStoreWithCopyPattern
    : public OpRewritePattern<TensorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorLoad =
        llvm::dyn_cast_or_null<TensorLoadOp>(op.tensor().getDefiningOp());
    if (!tensorLoad)
      return rewriter.notifyMatchFailure(op, "not fed by tensor_load op");
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, tensorLoad.memref(),
                                                op.memref());
    return success();
  }
};

class EraseUnusedTensorLoadOpPattern : public OpRewritePattern<TensorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.use_empty())
      return rewriter.notifyMatchFailure(op, "has uses");
    rewriter.eraseOp(op);
    return success();
  }
};

class ResolveTensorLoadStoreOps
    : public ResolveTensorLoadStoreOpsBase<ResolveTensorLoadStoreOps> {
  void runOnOperation() {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<ReplaceTensorStoreWithCopyPattern>(context);
    patterns.insert<EraseUnusedTensorLoadOpPattern>(context);
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addDynamicallyLegalOp<TensorLoadOp>([](TensorLoadOp op) {
      for (auto user : op.getResult().getUsers())
        if (!isa<tcp::YieldOp>(user))
          return false;
      return true;
    });
    target.addDynamicallyLegalOp<TensorStoreOp>(
        [](TensorStoreOp op) { return op.tensor().isa<BlockArgument>(); });
    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createResolveTensorLoadStoreOpsPass() {
  return std::make_unique<ResolveTensorLoadStoreOps>();
}

//===----------------------------------------------------------------------===//
// createE2ELoweringPipeline
//===----------------------------------------------------------------------===//

void mlir::NPCOMP::createE2ELoweringPipeline(OpPassManager &pm) {
  // Input IR is TCF ops.

  // Convert to TCP.
  pm.addPass(createConvertTCFToTCPPass());

  // TODO: Do tcp.island coarsening here.

  // --------------------------------------------------------------------------
  // Tensor to buffer (memref) conversion.
  // --------------------------------------------------------------------------

  // Convert tcp ops to Linalg where possible, as we want generic linalg
  // tensor->memref to do most of the mechanical work of rewriting ops in
  // terms of tensors to ops in terms of memrefs (since it is easy on that
  // representation).
  pm.addPass(createConvertTCPToLinalgPass());

  // Lower to hybrid tensor/memref
  //
  // The hybrid tensor/memref representation guarantees:
  // - every use of a tensor is a tensor_store op writing it into a memref
  // - every def of a tensor is a tensor_load op loading out of some memref.
  // - every memref is allocated by a `tcp.alloc_memref(%shape)` op.
  // - every memref is only ever writen once, and never mutated
  //
  // Exceptions: "boundaries" such as function arguments and island
  // live-outs.
  //
  // Or, another way to say this: the hybrid tensor/memref representation
  // doesn't attempt to eliminate the original tensors from the program,
  // but rather locally expands operations on tensors to be small subgraphs
  // with tensor_load/tensor_store at the boundaries, leaving enough
  // invariants that we can clean it up later.
  //
  // The core invariants that are needed for this step are that the
  // tensor-level ops we receive as input have a way of calculating the
  // sizes for their outputs. This is equivalent to saying that
  // `shape.shape_of` on the result of an op must be calculatable in terms
  // of the shapes of the inputs to the op.
  createLowerToHybridTensorMemRefPipeline(pm);

  // At this point, the invariants of the hybrid tensor/memref
  // representation allow us to resolve `shape.shape_of` ops to shape
  // computations earlier in the program. Specifically, every
  // `shape.shape_of` can be resolved to the shape argument to the
  // corresponding `tcp.alloc_memref` op of the tensor_load that produced
  // that tensor.
  pm.addPass(createResolveShapeOfOpsPass());

  // Now, we use the hybrid tensor/memref invariants to replace the
  // tensor_store ops with memref copy operations and erase the
  // tensor_load/tensor_store ops.
  pm.addPass(createResolveTensorLoadStoreOpsPass());

  // At this point, the IR is in a form where the interiors of islands
  // don't have tensor ops (except tensor_store's of arguments and
  // tensor_load's of returns).
  //
  // This is a reasonable representation for doing buffer assignment.
  //
  // TODO: Might want a different kind of island to better represent this.
  // This island op would explicitly capture all tensors as inputs, and it
  // would establish a more formalized ABI with the interior of the body
  // region (much like IREE does with dispatch regions). For now, we are
  // planning on just inlining the islands, so there is little value in
  // doing this, but we should look at the layering aspects here later.

  // At this point, we have loose shape calculations floating around, so
  // it's a good time to do some general cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // TODO:
  // lower linalg to loops: mlir::createConvertLinalgToLoopsPass()
  // lower shape stuff to ssa values.
  // Convert all of it to LLVM?
}
