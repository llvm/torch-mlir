#include "torch-mlir-dialects/Dialect/Tcp/Transforms/IsolateGroupOpsPass.h"
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
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir::tcp {

namespace {

class IsolateGroups : public OpRewritePattern<GroupOp> {
public:
  using OpRewritePattern<GroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GroupOp groupOp,
                                PatternRewriter &rewriter) const override {
    // Collect the values used in the given GroupOp. Those will be the inputs
    // to the IsolatedGroup op. The constants used in the GroupOp are collected
    // in a separate set and cloned into the body of the IsolatedGroupOp.
    llvm::SmallVector<Value> inputs;
    llvm::SmallDenseSet<Value> addedInputs;
    llvm::SmallDenseSet<Value> consts;
    llvm::SmallDenseSet<Value> defs;
    for (auto &op : groupOp.getBody().front()) {
      for (auto operand : op.getOperands()) {
        if (defs.find(operand) == defs.end()) {
          if (operand.getDefiningOp() &&
              operand.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
            consts.insert(operand);
          } else if (!addedInputs.contains(operand)) {
            inputs.push_back(operand);
            addedInputs.insert(operand);
          }
        }
      }
      defs.insert(op.getResults().begin(), op.getResults().end());
    }

    auto isolatedGroupOp = rewriter.create<IsolatedGroupOp>(
        groupOp.getLoc(), groupOp.getResultTypes(), inputs);
    isolatedGroupOp->setAttrs(groupOp->getAttrs());

    isolatedGroupOp.getBody().takeBody(groupOp.getBody());
    auto &isolatedGroupBlock = isolatedGroupOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&isolatedGroupBlock);
      auto belongsToIsolatedGroup = [&](OpOperand &opOperand) {
        return (opOperand.getOwner()->getParentOp() == isolatedGroupOp);
      };

      // Clone the constants at the start of the isolated group block.
      for (auto c : consts) {
        auto newConst = rewriter.clone(*c.getDefiningOp());
        rewriter.replaceUsesWithIf(c, newConst->getResult(0),
                                   belongsToIsolatedGroup);
      }

      // Add inputs as arguments to the isolated group block.
      for (size_t n = 0; n < inputs.size(); ++n) {
        isolatedGroupBlock.addArgument(inputs[n].getType(), groupOp.getLoc());
        rewriter.replaceUsesWithIf(inputs[n], isolatedGroupBlock.getArgument(n),
                                   belongsToIsolatedGroup);
      }
    }
    for (unsigned n = 0; n < groupOp.getNumResults(); ++n) {
      rewriter.replaceAllUsesWith(groupOp->getOpResult(n),
                                  isolatedGroupOp->getOpResult(n));
    }
    rewriter.eraseOp(groupOp);
    return success();
  }
};

class TcpIsolateGroupOpsPass
    : public TcpIsolateGroupOpsBase<TcpIsolateGroupOpsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    patterns.add<IsolateGroups>(context);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTcpIsolateGroupOpsPass() {
  return std::make_unique<TcpIsolateGroupOpsPass>();
}

} // namespace mlir::tcp
