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
//===----------------------------------------------------------------------===//

#include "npcomp/RefBackend/RefBackend.h"
#include "PassDetail.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

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
        auto extent =
            rewriter.create<shape::GetExtentOp>(op.getLoc(), shape, i);
        dynamicExtents.push_back(extent);
      }
    }
    rewriter.replaceOpWithNewOp<AllocOp>(op, memrefType, dynamicExtents);
    return success();
  }
};
} // namespace

namespace {
class LowerAllocMemRefOps
    : public LowerAllocMemRefOpsBase<LowerAllocMemRefOps> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<LowerAllocMemRefOp>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<refback::AllocMemRefOp>();
    target.addLegalOp<shape::GetExtentOp>();
    target.addLegalOp<AllocOp>();
    target.addLegalOp<ConstantOp>();
    if (failed(applyPartialConversion(func, target, patterns))) {
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
// RestrictedCanonicalizer
//===----------------------------------------------------------------------===//

namespace {
struct RestrictedCanonicalizer
    : public RestrictedCanonicalizerBase<RestrictedCanonicalizer> {
  void runOnOperation() override {
    auto *context = &getContext();

    // Find the dialects from their names.
    DenseSet<StringRef> neededDialects;
    for (const std::string &dialectName : includedDialects)
      neededDialects.insert(dialectName);
    DenseSet<Dialect *> dialectsToCanonicalize;
    for (Dialect *dialect : context->getLoadedDialects()) {
      if (neededDialects.count(dialect->getNamespace())) {
        dialectsToCanonicalize.insert(dialect);
        // Erase the dialect so that we can report an error below for any
        // dialect names that are not loaded.
        neededDialects.erase(dialect->getNamespace());
      }
    }

    // Report a helpful error if a dialect is not found.
    auto missingDialects = llvm::to_vector<6>(neededDialects);
    if (!missingDialects.empty()) {
      llvm::sort(missingDialects);
      std::string buf;
      llvm::raw_string_ostream os(buf);
      llvm::interleaveComma(missingDialects, os);
      llvm::report_fatal_error("restricted-canonicalize: unknown dialects: " +
                               os.str());
    }

    // Collect all canonicalization patterns from ops in the included dialects.
    OwningRewritePatternList patterns;
    for (AbstractOperation *op : context->getRegisteredOperations())
      if (dialectsToCanonicalize.count(&op->dialect))
        op->getCanonicalizationPatterns(patterns, context);

    Operation *op = getOperation();
    applyPatternsAndFoldGreedily(op->getRegions(), patterns);
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::NPCOMP::createRestrictedCanonicalizerPass() {
  return std::make_unique<RestrictedCanonicalizer>();
}

//===----------------------------------------------------------------------===//
// createRefBackendLoweringPipeline
//===----------------------------------------------------------------------===//

void mlir::NPCOMP::createRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options) {
  // This "end to end" lowering pipline loewrings from approximately the "numpy"
  // level of abstraction (which is a dialect we call "TCF", or "Tensor Compute
  // Frontend") all the way down to LLVM IR.

  // Convert from TCF to TCP.
  //
  // TCF has implicit broadcasting, and issues errors "inside the ops" in the
  // case of invalid broadcasts.
  //
  // TCP does not. So we need to reify the broadcasting and error checking.
  pm.addPass(createConvertTCFToTCPPass());

  // For operations with a shape transfer function, explicitly bypass their
  // shape computations with refback.shaped_results ops.
  //
  // Right now, our lowering flow depends heavily on descriptors, so technically
  // we don't need to bypass shapes -- we can just splat out the shape
  // calculations when lowering the ops themselves. However, this design keeps
  // the door open to various future directions, and is an interesting example
  // in its own right.
  //
  // For example, if we want to lower to command-buffer style API's like Vulkan,
  // then we need (for correctness) to bypass the shapes (actually,
  // something more sophisticated than just that) if we want to do command
  // buffer formation while we are still on tensors (e.g. to record workgroup
  // sizes). We might not care about pursuing that direction here though. So
  // consider this pass as purely advisory now.
  //
  // One case where we might still be interested in this is dealing with
  // linalg.generic ops and other types of "fusions" that have shape transfer
  // functions that are not easily reconstructible and thus we have to capture
  // the shape transfer functions earlier in the pipeline.
  pm.addPass(createBypassShapesPass());

  // Lower shape constraints before we enter tensor->memref conversion.
  // That is, we expand shape.cstr_* ops to eager error handling code.
  pm.addPass(createConvertShapeConstraintsPass());
  // Run shape canonicalizations. In particular, this erases shape.assuming,
  // now that we have converted shape constraints.
  // TODO: This is kind of ugly. Either we use pass options or a constructor
  // that takes C++ data structures. The former makes the pass usable on the
  // command line (including reproducers), the latter makes the pass more
  // convenient.
  std::unique_ptr<Pass> shapeCanonicalizer =
      createRestrictedCanonicalizerPass();
  if (failed(shapeCanonicalizer->initializeOptions("included-dialects=shape")))
    llvm::report_fatal_error("couldn't initialize restricted-canonicalize");
  pm.addPass(std::move(shapeCanonicalizer));

  // --------------------------------------------------------------------------
  // Lower the `tensor` type to `memref`.
  // --------------------------------------------------------------------------
  // We make a conscious effort here to do this as a sequence of separate passes
  // rather than a single mega dialect conversion pass.
  //
  // This means that intermediate steps have source/target materializations
  // (refback.memref_to_tensor / refback.tensor_to_memref) in the IR.

  // Lower ops enclosed in refback.shaped_results regions.
  // For now, this is covering the "tensor compute" ops like tcp.add /
  // tcp.broadcast_to (the former being handled via a special subset of
  // linalg.generic) -- we only handle those two, so having an isolated pass
  // that hardcodes all of them is fine -- eventually we might want something
  // more pluggable. The exact interface for this pluggability depends on
  // what design we want to settle on for bypassing shape computations.
  pm.addPass(createLowerShapedResultsToMemrefPass());
  // Lower tensor-valued constants to refback.global.
  pm.addPass(createLowerConstantTensorsToMemrefPass());
  // refback::AllocMemRefOp takes a shape (i.e. extent tensor) as an argument.
  // We need to resolve this to std.alloc which takes individual extents.
  pm.addPass(createLowerAllocMemRefOpsPass());
  // Lower shape ops to std.
  // TODO: This should in principle be moved before tensor->memref conversion.
  // But some of the tensor->memref lowerings above use shape.get_extent. For
  // example, when lowering a broadcast, we need to get an extent from its shape
  // operand to allocate the output.
  pm.addPass(createConvertShapeToStandardPass());
  // Lower std ops to memref.
  // This includes ops like extract_element.
  pm.addPass(createLowerStdToMemrefPass());
  // Lower control flow and other "structural" ops.
  //
  // These ops are generally not sensitive to the types that they operate on
  // (e.g. the types of block operands, function arguments, etc.). But they all
  // need to be converted consistently. So it makes sense to do this as the
  // final step of conversion, which also finalizes the elimination of all
  // stray source/target materializations introduced by the incremental
  // tensor->memref lowering.
  //
  // This completes conversion to memref. There are no `tensor`'s after
  // this point.
  pm.addPass(createLowerStructuralToMemrefPass());

  // TODO: Do buffer assignment. We should be able to just drop in the upstream
  // pass?

  // At this point, we have lots of loose stuff floating around from lowering,
  // so it's a good time to do some general cleanups.
  if (options.optimize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // --------------------------------------------------------------------------
  // Preparation for converting to an LLVM module.
  // --------------------------------------------------------------------------
  // Now, we begin the process of lowering to LLVM's level of abstraction
  // (after which LLVM will take over lowering to machine code).

  // Lower linalg ops to loops.
  // TODO: Do some linalg optimizations like tiling here.
  pm.addPass(createConvertLinalgToLoopsPass());

  // Run a some cleanups.
  if (options.optimize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // --------------------------------------------------------------------------
  // Final conversion to an LLVM module.
  // --------------------------------------------------------------------------

  // Convert scf to std control flow in preparation for going to LLVM.
  pm.addPass(createLowerToCFGPass());

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
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
}
