//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TCPToLinalg/TCPToLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace NPCOMP;

namespace {
class ConvertAdd : public OpRewritePattern<tcp::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcp::AddOp op,
                                PatternRewriter &rewriter) const override {
    size_t rank = op.getType().cast<RankedTensorType>().getRank();
    SmallVector<StringRef, 6> iterators(rank, getParallelIteratorTypeName());
    SmallVector<AffineMap, 3> accesses(/*args in + args out*/ 3,
                                       rewriter.getMultiDimIdentityMap(rank));
    auto genericOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), llvm::makeArrayRef({op.getType()}),
        ValueRange({op.lhs(), op.rhs()}),
        /*args_in=*/2,
        /*args_out=*/1,
        /*indexing_maps=*/accesses,
        /*iterator_types=*/iterators,
        /*function_ref=*/nullptr);

    Region &region = genericOp.region();
    Block *block = rewriter.createBlock(&region, region.begin());
    for (auto operandType : op.getOperandTypes()) {
      block->addArgument(operandType.cast<RankedTensorType>().getElementType());
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    Value bodyValue = rewriter.create<AddFOp>(
        op.getLoc(), block->getArgument(0), block->getArgument(1));
    rewriter.create<linalg::YieldOp>(op.getLoc(), bodyValue);

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertTCPToLinalg : public ConvertTCPToLinalgBase<ConvertTCPToLinalg> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);

    OwningRewritePatternList patterns;

    patterns.insert<ConvertAdd>(context);
    target.addIllegalOp<tcp::AddOp>();

    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<StandardOpsDialect>();

    if (failed(applyPartialConversion(module, target, patterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::createConvertTCPToLinalgPass() {
  return std::make_unique<ConvertTCPToLinalg>();
}
