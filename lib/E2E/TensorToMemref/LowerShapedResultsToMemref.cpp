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
#include "npcomp/Conversion/TCPToLinalg/TCPToLinalg.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

static Value allocMemRefForTensor(OpBuilder &builder, Value tensor, Value shape,
                                  Location loc) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return builder.create<tcp::AllocMemRefOp>(loc, memrefType, shape);
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

    auto shapedResults = dyn_cast<tcp::ShapedResultsOp>(op.getParentOp());
    if (!shapedResults)
      return rewriter.notifyMatchFailure(op, "parent not tcp.shaped_results");
    if (op.getOperation()->getResults() !=
        shapedResults.getBody()->getTerminator()->getOperands())
      return rewriter.notifyMatchFailure(
          op, "only limited forms of tcp.shaped_results allowed");
    auto resultShape = shapedResults.resultShapes()[0];
    Value resultMemref =
        allocMemRefForTensor(rewriter, op.result(), resultShape, op.getLoc());
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

namespace {
class LowerLinalgGenericTensorToMemRef
    : public OpConversionPattern<linalg::GenericOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::GenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: Replace this with more generic code operating on named
    // structured ops too.

    // These checks mirror those in BypassShapes.
    if (!llvm::all_of(op.getOperandTypes(),
                      [](Type type) { return type.isa<RankedTensorType>(); })) {
      return rewriter.notifyMatchFailure(op, "all operands must be tensors");
    }
    if (!llvm::all_of(op.getResultTypes(),
                      [](Type type) { return type.isa<RankedTensorType>(); })) {
      return rewriter.notifyMatchFailure(op, "all results must be tensors");
    }
    if (!llvm::all_of(op.indexing_maps(), [](Attribute map) {
          return map.cast<AffineMapAttr>().getValue().isIdentity();
        })) {
      return rewriter.notifyMatchFailure(
          op, "all indexing maps must be identity maps");
    }
    if (!llvm::all_of(op.iterator_types(), [](Attribute str) {
          return str.cast<StringAttr>().getValue() ==
                 getParallelIteratorTypeName();
        })) {
      return rewriter.notifyMatchFailure(
          op, "all iterator types must be 'parallel'");
    }

    SmallVector<Value, 6> memrefs(operands.begin(), operands.end());
    SmallVector<Value, 6> resultMemrefs;
    SmallVector<Value, 6> operandShapes;

    auto shapedResults = dyn_cast<tcp::ShapedResultsOp>(op.getParentOp());
    if (!shapedResults)
      return rewriter.notifyMatchFailure(op, "parent not tcp.shaped_results");
    // TODO: What if there are multiple ops in the tcp.shaped_results region?
    // The IREE solution is "they have to be fused and create no allocations
    // ultimately". The non-IREE solution is to just not bypass shapes in the
    // first place.
    if (op.getResults() !=
        shapedResults.getBody()->getTerminator()->getOperands())
      return rewriter.notifyMatchFailure(
          op, "only limited forms of tcp.shaped_results allowed");

    for (auto t : llvm::zip(op.getResults(), shapedResults.resultShapes())) {
      auto tensor = std::get<0>(t);
      auto shape = std::get<1>(t);
      auto memref = allocMemRefForTensor(rewriter, tensor, shape, op.getLoc());
      memrefs.push_back(memref);
      resultMemrefs.push_back(memref);
    }
    auto newGeneric = rewriter.create<linalg::GenericOp>(
        op.getLoc(), llvm::None, ValueRange(memrefs), op.getAttrs());
    newGeneric.region().getBlocks().clear();
    BlockAndValueMapping mapper;
    op.region().cloneInto(&newGeneric.region(), mapper);
    for (auto memref : resultMemrefs) {
      newGeneric.region().front().addArgument(
          memref.getType().cast<MemRefType>().getElementType());
    }
    rewriter.replaceOp(op, resultMemrefs);
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
  void runOnOperation() {
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

    patterns.insert<LowerLinalgGenericTensorToMemRef>(typeConverter, context);
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      if (llvm::any_of(op.getOperandTypes(), [](Type type) {
            return type.isa<RankedTensorType>();
          })) {
        return false;
      }
      if (llvm::any_of(op.getResultTypes(), [](Type type) {
            return type.isa<RankedTensorType>();
          })) {
        return false;
      }
      return true;
    });

    patterns.insert<LowerBroadcastToToLoopsPattern>(typeConverter, context);
    target.addIllegalOp<tcp::BroadcastToOp>();
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
