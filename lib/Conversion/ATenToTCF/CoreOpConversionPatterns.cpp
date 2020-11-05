//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/ATenToTCF/Patterns.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {

/// The ATen AddOp actually has three arguments:
///   self, other, alpha
/// Alpha is an integer that is multiplied by 'other' prior to adding.
class ConvertATenAdd : public OpRewritePattern<aten::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(aten::AddOp srcAddOp,
                                PatternRewriter &rewriter) const override {
    // Special case: Match when alpha is constant 1, which is the default,
    // quite common and maps directly to a TCF add. Note that regardless of
    // the type of self/other (i.e. if they are float), alpha emits as an
    // integer with value 1 when defaulted. It is this specific case that we
    // are detecting (default value) and will leave all others to the fully
    // generic conversion.
    APInt alphaValue;
    if (matchPattern(srcAddOp.alpha(), m_ConstantInt(&alphaValue)) &&
        alphaValue.getZExtValue() == 1) {
      rewriter.replaceOpWithNewOp<tcf::AddOp>(
          srcAddOp, srcAddOp.getResult().getType(), srcAddOp.self(),
          srcAddOp.other());
      return success();
    }

    return rewriter.notifyMatchFailure(
        srcAddOp, "aten.add to tcf.add currently only supports alpha == 1");
  }
};

/// Common conversion template for true binary elementwise ops.
/// This does not apply to the handful of not-actually-binary PyTorch ops that
/// have broadcastable self/other operands but may have additional parameters.
template <typename SourceOp, typename TargetOp>
class ConvertBinaryElementwise : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp srcOp,
                                PatternRewriter &rewriter) const override {
    auto operands = srcOp.getOperation()->getOperands();
    auto results = srcOp.getOperation()->getResults();
    assert(operands.size() == 2 && "expected true binary op");
    assert(results.size() == 1 && "expected single result op");
    Type resultType = results[0].getType();
    rewriter.replaceOpWithNewOp<TargetOp>(
        srcOp, resultType, srcOp.getOperand(0), srcOp.getOperand(1));
    return success();
  }
};

} // namespace

void mlir::NPCOMP::populateCoreATenToTCFPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<ConvertATenAdd>(context);
  patterns.insert<ConvertBinaryElementwise<aten::MulOp, tcf::MulOp>>(context);
  patterns.insert<ConvertBinaryElementwise<aten::MaximumOp, tcf::MaxOp>>(
      context);
}
