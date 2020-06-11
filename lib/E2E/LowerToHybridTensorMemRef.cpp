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
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
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

static Value allocMemRefForTensor(OpBuilder &builder, Value tensor, Value shape,
                                  Location loc) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return builder.create<tcp::AllocMemRefOp>(loc, memrefType, shape);
}

//===----------------------------------------------------------------------===//
// LowerBroadcastTo
//===----------------------------------------------------------------------===//

// TODO: Lower to linalg.indexed_generic instead and let linalg do the expansion
// to loops?
namespace {
class LowerBroadcastToToLoopsPattern
    : public OpRewritePattern<tcp::BroadcastToOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcp::BroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<RankedTensorType>();
    auto inputType = op.operand().getType().cast<RankedTensorType>();
    Value resultMemref = rewriter.create<tcp::AllocMemRefOp>(
        op.getLoc(),
        MemRefType::get(resultType.getShape(), resultType.getElementType()),
        op.shape());
    Value inputMemref = allocMemRefForTensor(
        rewriter, op.operand(),
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.operand()),
        op.getLoc());
    rewriter.create<TensorStoreOp>(op.getLoc(), op.operand(), inputMemref);
    SmallVector<Value, 6> outputExtents;
    SmallVector<Value, 6> inputDimRequiresBroadcasting;

    // TODO: handle output rank > input rank.
    for (int i = 0, e = resultType.getRank(); i < e; i++) {

      Value outputExtent = rewriter.create<tcp::GetExtentOp>(
          op.getLoc(), op.shape(), rewriter.getI64IntegerAttr(i));
      outputExtents.push_back(outputExtent);

    }
    int rankDiff = resultType.getRank() - inputType.getRank();
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
      rewriter.create<StoreOp>(op.getLoc(), load, resultMemref, inductionVariables);
    }

    rewriter.replaceOpWithNewOp<TensorLoadOp>(op, resultMemref);
    return success();
  }
};
} // namespace

// TODO: This should be layered in better somewhere.
// We currently only create DimOp's during LowerBroadcastToToLoopsPattern,
// so for now just stuff it in here.
namespace {
class LowerDimOpToShape : public OpRewritePattern<DimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DimOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Remove this const pattern when lowering to shape.get_extent.
    auto constIndex = op.getConstantIndex();
    if (!constIndex)
      return failure();
    auto shape =
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.memrefOrTensor());
    rewriter.replaceOpWithNewOp<tcp::GetExtentOp>(op, shape, *constIndex);
    return success();
  }
};
} // namespace

namespace {
class LowerBroadcastToToLoops
    : public LowerBroadcastToToLoopsBase<LowerBroadcastToToLoops> {
  void runOnOperation() {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tcp::TCPDialect>();

    OwningRewritePatternList patterns;
    target.addIllegalOp<tcp::BroadcastToOp>();
    patterns.insert<LowerBroadcastToToLoopsPattern>(context);
    target.addIllegalOp<DimOp>();
    patterns.insert<LowerDimOpToShape>(context);

    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerBroadcastToToLoopsPass() {
  return std::make_unique<LowerBroadcastToToLoops>();
}

//===----------------------------------------------------------------------===//
// LowerLinalgOnTensorToLinalgOnMemref
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgGenericTensorToMemRef : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(linalg::GenericOp op,
                                  PatternRewriter &rewriter) const override {

      // TODO: Replace this with more generic code operating on named
      // structured ops too.

      // Only handle generic ops where all operands and results are tensors.
      if (!llvm::all_of(op.getOperandTypes(), [](Type type) {
            return type.isa<RankedTensorType>();
          })) {
        return rewriter.notifyMatchFailure(op, "all operands must be tensors");
      }
      if (!llvm::all_of(op.getResultTypes(), [](Type type) {
            return type.isa<RankedTensorType>();
          })) {
        return rewriter.notifyMatchFailure(op, "all results must be tensors");
      }

      // TODO: Loosen restrictions on indexing maps.
      // This will require more principled handling of shape reification
      // earlier in the compilation stack, as in general output shapes of a
      // linalg.generic cannot be inferred easily.
      // See:
      // https://llvm.discourse.group/t/computing-output-shapes-of-structured-ops-on-tensors/866
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

      SmallVector<Value, 6> memrefs;
      SmallVector<Value, 6> resultMemrefs;
      SmallVector<Value, 6> operandShapes;
      for (auto tensor : op.getOperands()) {
        auto shape = rewriter.create<shape::ShapeOfOp>(op.getLoc(), tensor);
        auto memref =
            allocMemRefForTensor(rewriter, tensor, shape, op.getLoc());
        rewriter.create<TensorStoreOp>(op.getLoc(), tensor, memref);
        memrefs.push_back(memref);
        operandShapes.push_back(shape);
      }
      auto shapeType = shape::ShapeType::get(rewriter.getContext());
      SmallVector<Type, 6> shapeTypes(op.getNumResults(), shapeType);
      // TODO: We need more principled handling of output shapes.
      // This assumes that all results have the same shape, which is justified
      // by checks above, but we really need a better story here.
      SmallVector<Value, 6> resultShapes(op.getNumResults(), operandShapes[0]);
      for (auto t : llvm::zip(op.getResults(), resultShapes)) {
        auto tensor = std::get<0>(t);
        auto shape = std::get<1>(t);
        auto memref =
            allocMemRefForTensor(rewriter, tensor, shape, op.getLoc());
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
      auto newResultTensors =
          llvm::to_vector<6>(llvm::map_range(resultMemrefs, [&](Value memref) {
            return rewriter.create<TensorLoadOp>(op.getLoc(), memref)
                .getResult();
          }));
      rewriter.replaceOp(op, newResultTensors);
      return success();
    }
};
}

namespace {
class LowerLinalgOnTensorToLinalgOnMemref
    : public LowerLinalgOnTensorToLinalgOnMemrefBase<
          LowerLinalgOnTensorToLinalgOnMemref> {
  void runOnOperation() {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    ConversionTarget target(*context);
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalOp<tcp::AllocMemRefOp>();
    patterns.insert<LowerLinalgGenericTensorToMemRef>(context);
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

    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerLinalgOnTensorToLinalgOnMemrefPass() {
  return std::make_unique<LowerLinalgOnTensorToLinalgOnMemref>();
}

void mlir::NPCOMP::createLowerToHybridTensorMemRefPipeline(OpPassManager &pm) {
  // Lower to hybrid tensor/memref.
  // The invariant of "hybrid tensor/memref" is that the core computation
  // ops operate on memref, but we launder in and out of tensors in such a
  // way that the original SSA tensor values remain and can be traced to
  // their corresponding memrefs (via tensor_load/tensor_store) which are
  // allocated with alloc_shape ops.
  // Thus, shape.shape_of ops on the original tensors in the program can be
  // resolved to the shapes in the alloc_memref calls.
  pm.addPass(createLowerLinalgOnTensorToLinalgOnMemrefPass());
  pm.addPass(createLowerBroadcastToToLoopsPass());
}
