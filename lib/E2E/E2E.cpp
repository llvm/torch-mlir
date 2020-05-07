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
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"
#include "npcomp/Conversion/TCPToLinalg/TCPToLinalg.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

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
// createE2ELoweringPipeline
//===----------------------------------------------------------------------===//

void mlir::NPCOMP::createE2ELoweringPipeline(OpPassManager &pm) {
  // Input IR is TCF ops.

  // Convert to TCP.
  pm.addPass(createConvertTCFToTCPPass());
  // Convert tcp ops to Linalg where possible.
  pm.addPass(createConvertTCPToLinalgPass());

  // TODO: legalize `dim` to shape.shape_of + tcp.get_extent

  // --------------------------------------------------------------------------
  // Tensor to buffer (memref) conversion.
  // --------------------------------------------------------------------------

  // Lower to hybrid tensor/memref
  createLowerToHybridTensorMemRefPipeline(pm);

  // At this point, every tensor in the program is the result of a
  // `tensor_load` of an `alloc_memref` op (or is an argument). Therefore,
  // every shape_of can be resolved by looking at the corresponding
  // alloc_memref of the tensor.
  pm.addPass(createResolveShapeOfOpsPass());


  // TODO:
  // forward tensor_load/tensor_store (which leaves all tensors with no
  // uses)
  // lower linalg to loops: mlir::createConvertLinalgToLoopsPass()
  // lower shape stuff to rshape?
  // lower rshape to SSA values?
  // Convert all of it to LLVM?
}
