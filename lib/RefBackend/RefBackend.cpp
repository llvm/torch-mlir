//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base file for npcomp's "reference backend".
//
// The input to this backend is a layer that consists of linalg-on-tensors
// together with std scalar ops and control flow.
//
// The output of this backend is LLVM IR suitable for JITing.
//
// We expect that other backends will appear that have a similar kind of
// interface. IREE already uses this layering.
//
//===----------------------------------------------------------------------===//

#include "npcomp/RefBackend/RefBackend.h"
#include "PassDetail.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "npcomp/RefBackend/Passes.h.inc"
} // end namespace

void mlir::NPCOMP::registerRefBackendPasses() {
  ::registerPasses();

  mlir::PassPipelineRegistration<RefBackendLoweringPipelineOptions>(
      "refback-lowering-pipeline", "RefBackend lowering pipeline.",
      mlir::NPCOMP::createRefBackendLoweringPipeline);
}

//===----------------------------------------------------------------------===//
// LowerAllocMemRefOps
//===----------------------------------------------------------------------===//

namespace {
class LowerAllocMemRefOp : public OpRewritePattern<refback::AllocMemRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(refback::AllocMemRefOp op,
                                PatternRewriter &rewriter) const override {
    auto memrefType = op.getType().cast<MemRefType>();
    auto shape = op.getOperand();
    // std.alloc only accepts the dynamic extents as operands, so only
    // collect those.
    SmallVector<Value, 6> dynamicExtents;
    for (int i = 0, e = memrefType.getRank(); i < e; i++) {
      if (memrefType.isDynamicDim(i)) {
        auto ci = rewriter.create<ConstantIndexOp>(op.getLoc(), i);
        auto extent = rewriter.create<tensor::ExtractOp>(op.getLoc(), shape,
                                                         ValueRange({ci}));
        dynamicExtents.push_back(extent);
      }
    }
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType,
                                                 dynamicExtents);
    return success();
  }
};
} // namespace

namespace {
class LowerAllocMemRefOps
    : public LowerAllocMemRefOpsBase<LowerAllocMemRefOps> {

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerAllocMemRefOp>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<refback::AllocMemRefOp>();
    target.addLegalOp<tensor::ExtractOp>();
    target.addLegalOp<memref::AllocOp>();
    target.addLegalOp<ConstantOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerAllocMemRefOpsPass() {
  return std::make_unique<LowerAllocMemRefOps>();
}

//===----------------------------------------------------------------------===//
// createRefBackendLoweringPipeline
//===----------------------------------------------------------------------===//

void mlir::NPCOMP::createRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options) {

  // Convert all elementwise ops to linalg.
  //
  // Considering correctness, this lets us reuse the linalg bufferization, which
  // applies uniformly to all linalg structured ops.
  //
  // Also, converting to linalg herevopens up a lot of optimization
  // opportunities.
  pm.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());

  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createLinalgElementwiseOpFusionPass());
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // Lower shape constraints before we enter tensor->memref conversion.
  // That is, we expand shape.cstr_* ops to eager error handling code.
  pm.addNestedPass<FuncOp>(createConvertShapeConstraintsPass());
  // Run shape canonicalizations. In particular, this erases shape.assuming,
  // now that we have converted shape constraints.
  // TODO: Don't canonicalize everything.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Lower shape ops to std.
  pm.addPass(createConvertShapeToStandardPass());

  // --------------------------------------------------------------------------
  // Lower the `tensor` type to `memref`.
  // --------------------------------------------------------------------------
  // We make a conscious effort here to do this as a sequence of separate passes
  // rather than a single mega dialect conversion pass.
  //
  // This means that intermediate steps have source/target materializations
  // (memref.tensor_load / memref.buffer_cast) in the IR.

  // Run tensor constant bufferization.
  // This pass has to run on a module op, and so does the final
  // FuncBufferizePass. But everything else can run in parallel on functions,
  // so we try to bracket the entire bufferization pipeline with the module
  // passes to allow maximum parallelism.
  pm.addPass(createTensorConstantBufferizePass());
  // refback::AllocMemRefOp takes a shape (i.e. extent tensor) as an argument.
  // We need to resolve this to std.alloc which takes individual extents.
  pm.addNestedPass<FuncOp>(createLowerAllocMemRefOpsPass());
  pm.addNestedPass<FuncOp>(createSCFBufferizePass());
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<FuncOp>(createStdBufferizePass());
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());
  pm.addPass(createFuncBufferizePass());
  pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());

  // TODO: Do buffer deallocation. We should be able to just drop in the
  // upstream pass?

  // At this point, we have lots of loose stuff floating around from lowering,
  // so it's a good time to do some general cleanups.
  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // --------------------------------------------------------------------------
  // Preparation for converting to an LLVM module.
  // --------------------------------------------------------------------------
  // Now, we begin the process of lowering to LLVM's level of abstraction
  // (after which LLVM will take over lowering to machine code).

  // Lower linalg ops to loops.
  // TODO: Do some linalg optimizations like tiling here.
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());

  // Run a some cleanups.
  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // --------------------------------------------------------------------------
  // Final conversion to an LLVM module.
  // --------------------------------------------------------------------------

  // Convert affine to std control flow in preparation for going to LLVM.
  pm.addNestedPass<FuncOp>(createLowerAffinePass());

  // Convert scf to std control flow in preparation for going to LLVM.
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());

  // Convert functions signatures and other constructs that interface with the
  // runtime to the `refbackrt` dialect.
  pm.addPass(createLowerToRefbackrtABIPass());

  // Finally, convert to LLVM dialect using our custom LowerToLLVM pass
  // which reuses the upstream patterns and gives us a place to add our own
  // patterns for our own custom ops like the refbackrt ops.
  pm.addPass(createLowerToLLVMPass());

  // Although LLVM will clean everything up eventually, for the sake of IR
  // clarity while still in MLIR, run some cleanups.
  if (options.optimize) {
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
  }
}
