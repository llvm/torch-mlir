#include "torch-mlir-dialects/Dialect/Tcp/Transforms/FuseTcpOpsPass.h"
#include "./PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"
#include "torch-mlir-dialects/Dialect/Tcp/Transforms/Passes.h"

using namespace mlir;

namespace mlir::tcp {

namespace {

class GenericBottomUpFuser : public RewritePattern {
public:
  using CanFuseFuncType = std::function<bool(Operation *, Operation *)>;

  GenericBottomUpFuser(MLIRContext *context, CanFuseFuncType cf)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        canFuse(cf) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *use = op;
    for (auto operand : op->getOperands()) {
      if (operand.getDefiningOp()) {
        Operation *def = operand.getDefiningOp();
        if (canFuse(def, use)) {

          // Currently we are only fusing ops at the top-level.
          // This is to avoid recursing inside a group and ending up with
          // nested groups that contain the same ops.
          // Since we are iterating bottom up in a block, we only need to check
          // if the def op has a func parent.
          //
          // TODO: Remove this restriction to allow fusing in nested regions.
          if (!isa<func::FuncOp>(def->getParentOp())) {
            continue;
          }

          // We only support fusing def ops that have exactly one use, for now.
          if (!def->hasOneUse()) {
            continue;
          }

          // Fuse the def and use ops into a group.

          // * If both the ops have the same parent region, they must be part
          //   of the top-level func. So, we need to create a new group.
          // * The only other case is when the def op is part of the top-level
          //   func and the use is already inside a group.
          if (def->getParentRegion() == use->getParentRegion()) {
            auto groupOp =
                rewriter.create<GroupOp>(use->getLoc(), use->getResultTypes());
            Block *groupBlock = new Block();
            groupOp.getBody().push_back(groupBlock);
            for (unsigned num = 0; num < use->getNumResults(); ++num) {
              rewriter.replaceAllUsesWith(use->getResult(num),
                                          groupOp->getResult(num));
            }
            {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(groupBlock);
              auto yieldOp =
                  rewriter.create<YieldOp>(use->getLoc(), use->getResults());
              use->moveBefore(yieldOp);
              operand.getDefiningOp()->moveBefore(use);
            }
          } else if (auto groupOp = dyn_cast<GroupOp>(use->getParentOp())) {
            def->moveBefore(use);
          } else {
            llvm_unreachable("Unhandled case during fusion");
          }
        }
      }
    }
    return success();
  }

private:
  CanFuseFuncType canFuse;
};

class TcpFuseElementwiseOpsPass
    : public TcpFuseElementwiseOpsBase<TcpFuseElementwiseOpsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    auto canFuse = [](Operation *def, Operation *use) -> bool {
      return def->hasTrait<OpTrait::Elementwise>() &&
             use->hasTrait<OpTrait::Elementwise>();
    };
    patterns.add<GenericBottomUpFuser>(context, canFuse);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTcpFuseElementwiseOpsPass() {
  return std::make_unique<TcpFuseElementwiseOpsPass>();
}

} // namespace mlir::tcp
