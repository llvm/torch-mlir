//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/E2E/E2E.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
class LowerCstrBroadcastableOp
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
    // A shape.cstr_* op should be the result of lowering a !shape.shape; it
    // should not itself ever consume or produce a !shape.shape.
    //
    // There is no way to "sink" a !shape.shape type, because one cannot inspect
    // if it is an error. The only way to use it safely is to lower the op that
    // produced the value to a set of constraints and then use the witness to
    // guard a shape.assuming.
    //
    // Consider for example what we do when lowering TCF to TCP: we need to do a
    // shape calculation for the broadcasting. But we create the
    // shape.cstr_broadcastable and use its witness to guard a `shape.assuming {
    // ... shape.broadcast ...}`. There's never any need to create a
    // !shape.shape.
    //
    // The use of !shape.shape should be restricted to contexts like
    // declarations of shape transfer functions, with automatic utilities to
    // lower !shape.shape types to corresponding constraints + shape.assuming +
    // tensors. In this (npcomp e2e) lowering flow, we don't have any such
    // "declarative shape transfer functions" or utilities to expand them to
    // constraints. So !shape.shape should never exist in our IR.
    //
    // Historically, we used !shape.shape type for everything, and
    // shape.to_extent_tensor would abort in case of an error. But that's not a
    // useful semantics for lowering, since the error is defined to happen as
    // part of the shape.to_extent_tensor op, which requires materializing an
    // "is error" bit in the IR and carrying it around everywhere that the
    // original !shape.shape value was being used. In practice, nobody respects
    // that, which opens us up to miscompilations. That is, the lowering
    // strategy is either "not emit errors at all" or "emit errors as part of
    // lowering e.g. the shape.broadcast op itself" (which technically puts the
    // errors in some random location in the IR that is not the
    // shape.to_extent_tensor op). E.g. the following code would miscompile with
    // either of those ways that these ops get lowered in practice:
    // ```
    // %shape = shape.broadcast %lhs, %rhs : !shape.shape
    // if %cond:
    //     shape.to_extent_tensor(%shape)
    // ```
    // It's not possible to correctly compile this code without significant
    // contortions (such as carrying an "is error" bit). And to boot, we
    // shouldn't be getting into that situation in the first place! But the
    // `shape.to_extent_tensor : !shape.shape -> tensor<?xindex>` abstraction
    // opens up that possibility.
    //
    // shape.to_extent_tensor should not really be a thing, since it creates
    // these ill-defined situations about where errors are observed. A
    // !shape.shape type should only exist (for this compilation flow) as part
    // of a utility, something like "I want to do this shape calculation on
    // !shape.shape type, create IR that uses tensor<?xindex> and witnesses to
    // implement it, on the assumption that the error can be
    // observed anywhere inside the shape calculation".
    //
    // !shape.shape type would still be useful for lowerings that actually
    // result in a runtime type that carries an "is error" bit inside it, though
    // TBD if such use cases arise.
    if (op.getType().isa<shape::ShapeType>() ||
        op.lhs().getType().isa<shape::ShapeType>() ||
        op.rhs().getType().isa<shape::ShapeType>()) {
      return op.emitError() << "Error shapes should not exist at this point";
    }

    auto loc = op.getLoc();
    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<ConstantIndexOp>(loc, 1);

    // Find smaller and greater rank and extent tensor.
    Value lhsRank = rewriter.create<DimOp>(loc, op.lhs(), zero);
    Value rhsRank = rewriter.create<DimOp>(loc, op.rhs(), zero);
    Value lhsSmaller =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, lhsRank, rhsRank);
    Type indexTy = rewriter.getIndexType();
    Type extentTensorTy = op.lhs().getType();
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, TypeRange{indexTy, extentTensorTy, indexTy, extentTensorTy},
        lhsSmaller,
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(
              loc, ValueRange{lhsRank, op.lhs(), rhsRank, op.rhs()});
        },
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(
              loc, ValueRange{rhsRank, op.rhs(), lhsRank, op.lhs()});
        });
    Value lesserRank = ifOp.getResult(0);
    Value lesserRankOperand = ifOp.getResult(1);
    Value greaterRank = ifOp.getResult(2);
    Value greaterRankOperand = ifOp.getResult(3);

    Value rankDiff =
        rewriter.create<SubIOp>(loc, indexTy, greaterRank, lesserRank);

    // Compare the shapes extent by extent, and emit errors for
    // non-broadcast-compatible shapes.
    // Two extents are broadcast-compatible if
    // 1. they are both equal, or
    // 2. at least one of them is 1.

    rewriter.create<scf::ForOp>(
        loc, rankDiff, greaterRank, one, llvm::None,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
          Value greaterRankOperandExtent = b.create<ExtractElementOp>(
              loc, greaterRankOperand, ValueRange{iv});
          Value ivShifted = b.create<SubIOp>(loc, indexTy, iv, rankDiff);
          Value lesserRankOperandExtent = b.create<ExtractElementOp>(
              loc, lesserRankOperand, ValueRange{ivShifted});

          Value greaterRankOperandExtentIsOne = b.create<CmpIOp>(
              loc, CmpIPredicate::eq, greaterRankOperandExtent, one);
          Value lesserRankOperandExtentIsOne = b.create<CmpIOp>(
              loc, CmpIPredicate::eq, lesserRankOperandExtent, one);
          Value extentsAgree =
              b.create<CmpIOp>(loc, CmpIPredicate::eq, greaterRankOperandExtent,
                               lesserRankOperandExtent);
          auto broadcastIsValid =
              b.create<OrOp>(loc, b.getI1Type(), extentsAgree,
                             b.create<OrOp>(loc, greaterRankOperandExtentIsOne,
                                            lesserRankOperandExtentIsOne));
          b.create<AssertOp>(loc, broadcastIsValid, "invalid broadcast");
          b.create<scf::YieldOp>(loc);
        });

    // Now that we have emitted all the assertions, the witness is trivially
    // satisfied.
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};
} // namespace

namespace {
class LowerCstrRequireOp : public OpRewritePattern<shape::CstrRequireOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrRequireOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<AssertOp>(op.getLoc(), op.pred(), op.msgAttr());
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};
} // namespace

namespace {
// This pass eliminates shape constraints from the program.
//
// After this pass finishes, there are no !shape.witness types in the program,
// no shape.assuming, no shape.cstr_*.
//
// TODO: This should move to upstream ShapeToStandard conversions.
class LowerShapeConstraints
    : public LowerShapeConstraintsBase<LowerShapeConstraints> {
  void runOnOperation() {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<LowerCstrBroadcastableOp>(context);
    patterns.insert<LowerCstrRequireOp>(context);
    // Add in the canonicalization patterns for shape.assuming so that it gets
    // inlined when its witness becomes a true constant witness.
    shape::AssumingOp::getCanonicalizationPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(func, patterns)))
      return signalPassFailure();

    // TODO: Check that there are no remaining !shape.witness, shape.assuming,
    // shape.cstr_* ops, etc.
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerShapeConstraintsPass() {
  return std::make_unique<LowerShapeConstraints>();
}
