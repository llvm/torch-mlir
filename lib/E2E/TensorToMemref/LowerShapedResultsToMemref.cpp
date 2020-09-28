//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "npcomp/E2E/E2E.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

static FailureOr<SmallVector<Value, 6>>
allocateResults(Operation *op, ConversionPatternRewriter &rewriter,
                Location loc,
                SmallVectorImpl<Value> *resultShapesOut = nullptr) {
  // TODO: This is really fragile. Can we have a better story?
  auto shapedResults = dyn_cast<tcp::ShapedResultsOp>(op->getParentOp());
  if (!shapedResults)
    return rewriter.notifyMatchFailure(op, "parent not tcp.shaped_results");
  if (op->getResults() !=
      shapedResults.getBody()->getTerminator()->getOperands())
    return rewriter.notifyMatchFailure(
        op, "only limited forms of tcp.shaped_results allowed");
  auto resultShapes = shapedResults.resultShapes();
  SmallVector<Value, 6> results;
  for (auto t : llvm::zip(op->getResults(), resultShapes)) {
    auto result = std::get<0>(t);
    auto resultShape = std::get<1>(t);
    auto tensorType = result.getType().cast<RankedTensorType>();
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto memref =
        rewriter.create<tcp::AllocMemRefOp>(loc, memrefType, resultShape);
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
      Value outputExtent = rewriter.create<shape::GetExtentOp>(
          op.getLoc(), rewriter.getIndexType(), resultShape, dimIndex);
      outputExtents.push_back(outputExtent);
    }
    int rankDiff = resultType.getRank() - inputType.getRank();
    SmallVector<Value, 6> inputDimRequiresBroadcasting;
    for (int i = 0, e = inputType.getRank(); i < e; i++) {
      // Calculate the relevant extents.
      Value inputExtent = rewriter.create<DimOp>(op.getLoc(), op.operand(), i);
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
      Value load =
          rewriter.create<LoadOp>(op.getLoc(), inputMemref, inputIndices);
      rewriter.create<StoreOp>(op.getLoc(), load, resultMemref,
                               inductionVariables);
    }
    rewriter.replaceOp(op, resultMemref);
    return success();
  }
};
} // namespace

static Value createLinalgBodyCalculationForElementwiseOp(Operation *op,
                                                         ValueRange bodyArgs,
                                                         OpBuilder &builder,
                                                         Location loc) {
  if (isa<tcp::AddOp>(op))
    return builder.create<AddFOp>(loc, bodyArgs[0], bodyArgs[1]);

  if (isa<tcp::MaxOp>(op)) {
    auto greater = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, bodyArgs[0],
                                          bodyArgs[1]);
    return builder.create<SelectOp>(loc, greater, bodyArgs[0], bodyArgs[1]);
  }

  if (isa<tcp::ExpOp>(op))
    return builder.create<ExpOp>(loc, bodyArgs[0]);

  if (isa<tcp::TanhOp>(op))
    return builder.create<TanhOp>(loc, bodyArgs[0]);

  op->dump();
  llvm::report_fatal_error("unhandled op (see dump above): linalg body "
                           "calculation for elementwise op");
}

static LogicalResult
matchAndRewriteElementwiseOp(Operation *op, ArrayRef<Value> operands,
                             ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  Value result = op->getResult(0);

  auto resultsOrFailure = allocateResults(op, rewriter, loc);
  if (failed(resultsOrFailure))
    return failure();
  auto results = *resultsOrFailure;

  SmallVector<Value, 6> args;
  args.append(operands.begin(), operands.end());
  args.append(results.begin(), results.end());

  size_t rank = result.getType().cast<RankedTensorType>().getRank();
  SmallVector<StringRef, 6> iterators(rank, getParallelIteratorTypeName());
  // TODO: Generalize this to other elementwise ops.
  // All we need to do is to have a mapping of tcp.foo to scalar.foo.
  // TODO: Should we just use linalg named ops for most of TCP?
  // Doing so would make tcp very consistent, but also it would, at this early
  // stage, make most non-trivial changes also require co-design with the
  // linalg ODS generator, which would be a very slow process.
  auto argsIn = operands.size();
  auto argsOut = results.size();
  SmallVector<AffineMap, 3> accesses(argsIn + argsOut,
                                     rewriter.getMultiDimIdentityMap(rank));
  rewriter.create<linalg::GenericOp>(
      loc, /*inputs=*/operands, /*outputBuffers=*/results,
      /*indexingMaps=*/accesses,
      /*iteratorTypes=*/iterators,
      /*bodyBuilder=*/
      [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
        auto scalarResult = createLinalgBodyCalculationForElementwiseOp(
            op, regionArgs, builder, loc);
        builder.create<linalg::YieldOp>(loc, ValueRange({scalarResult}));
      });
  rewriter.replaceOp(op, results);
  return success();
}

namespace {
template <typename SourceOp>
class LowerElementwiseOp : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return matchAndRewriteElementwiseOp(op, operands, rewriter);
  }
};
} // namespace

namespace {
class LowerTcpMatmulOp : public OpConversionPattern<tcp::MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::MatmulOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultsOrFailure = allocateResults(op, rewriter, op.getLoc());
    if (failed(resultsOrFailure))
      return failure();
    auto results = *resultsOrFailure;
    rewriter.create<linalg::MatmulOp>(op.getLoc(), operands, results);
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

namespace {
// TODO: Linalg and shape don't implement the inliner interface, which blocks us
// from using mlir::inlineRegion. Locally override it here.
class LocallyOverrideLegalityInlinerInterface : public InlinerInterface {
public:
  using InlinerInterface::InlinerInterface;
  bool isLegalToInline(Operation *op, Region *dest,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
};
} // namespace

namespace {
// This pass is responsible for lowering regions wrapped by
// tcp.shaped_results (which operate on tensors) to memrefs.
// This includes any ops potentially contained within them.
// This is somewhat analogous to IREE's backend compilation of a single dispatch
// region, except that for now, we only allow a single op in the
// tcp.shaped_results, and we don't have any notion of "backend" layered at all.
// Nor is it clear if we really want any of that here.
//
// The tcp.shaped_results ops provide precisely the information needed to
// allocate output buffers when converting to memref.
// For now, this process eliminates the original tcp.shaped_results op since we
// don't have any host/device distinction or other structure that would require
// retaining that sort of IR structure.
//
// TODO: Do "shape_of" resolution while still on tensors.
// Here we spew out tons of shape_of and rely on dim ops on descriptors to make
// it work. The key difference is that we need tcp.shaped_results (or its
// successor / something it gets lowered to) to not be IsolatedFromAbove, and
// explicitly capture all input tensors along with their shapes. That allows
// shape_of ops on inputs to be trivially resolved. Unfortunately, this opens up
// the whole "dispatch region formation" can of worms like exists in IREE --
// once you have multiple ops inside a "dispatch region", you need to somehow
// lower them without allocating intermediate buffers.
//
// TODO: Don't hardcode the lowering for every op in this one pass.
class LowerShapedResultsToMemref
    : public LowerShapedResultsToMemrefBase<LowerShapedResultsToMemref> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<linalg::LinalgDialect,
                    scf::SCFDialect,
                    shape::ShapeDialect>();
    // clang-format on
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });

    typeConverter.addSourceMaterialization([](OpBuilder &builder,
                                              RankedTensorType type,
                                              ValueRange inputs, Location loc) {
      assert(inputs.size() == 1);
      assert(inputs[0].getType().isa<MemRefType>());
      return (Value)builder.create<tcp::MemrefToTensorOp>(loc, type, inputs[0]);
    });
    typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                              MemRefType type,
                                              ValueRange inputs, Location loc) {
      assert(inputs.size() == 1);
      assert(inputs[0].getType().isa<RankedTensorType>());
      return (Value)builder.create<tcp::TensorToMemrefOp>(loc, type, inputs[0]);
    });

    OwningRewritePatternList patterns;

    ConversionTarget target(*context);

    // The shaped results ops themselves. They have to be legal since we delete
    // them later after the conversion process.
    target.addLegalOp<tcp::ShapedResultsOp>();
    target.addLegalOp<tcp::YieldOp>();
    // All lowering to buffers involves tcp.alloc_memref ops.
    target.addLegalOp<tcp::AllocMemRefOp>();
    // The casting ops are introduced by the type converter, so we should mark
    // them legal.
    target.addLegalOp<tcp::MemrefToTensorOp>();
    target.addLegalOp<tcp::TensorToMemrefOp>();

    patterns.insert<LowerBroadcastToToLoopsPattern>(typeConverter, context);
    target.addIllegalOp<tcp::BroadcastToOp>();
    patterns
        .insert<LowerElementwiseOp<tcp::AddOp>, LowerElementwiseOp<tcp::MaxOp>,
                LowerElementwiseOp<tcp::ExpOp>,
                LowerElementwiseOp<tcp::TanhOp>>(typeConverter, context);
    target.addIllegalOp<tcp::AddOp, tcp::MaxOp>();
    patterns.insert<LowerTcpMatmulOp>(typeConverter, context);
    target.addIllegalOp<tcp::MatmulOp>();

    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalOp<shape::GetExtentOp>();

    SmallVector<Operation *, 6> shapedResultsOps;
    func.walk([&](tcp::ShapedResultsOp op) { shapedResultsOps.push_back(op); });

    if (failed(applyFullConversion(shapedResultsOps, target, patterns)))
      return signalPassFailure();

    // Now inline the tcp.shaped_results ops.
    // This can't be done as part of the conversion since conversion visits
    // ops in preorder, and we need the tcp.shaped_results ops to be present
    // so that inner ops can get their shape.
    LocallyOverrideLegalityInlinerInterface interface(context);
    for (Operation *shapedResultsOp : shapedResultsOps) {
      auto op = cast<tcp::ShapedResultsOp>(shapedResultsOp);
      if (failed(inlineRegion(interface, &op.body(), op, ValueRange({}),
                              op.getResults(), /*inlineLoc=*/llvm::None,
                              /*shouldCloneInlinedRegion=*/false))) {
        op.emitError() << "could not inline body";
        return signalPassFailure();
      }
      op.erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerShapedResultsToMemrefPass() {
  return std::make_unique<LowerShapedResultsToMemref>();
}
