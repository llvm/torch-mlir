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
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
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

/// The ATen Conv2dOp has seven arguments:
///   input, weight, bias, stride, padding, dilation, groups

class ConvertATenConv2d : public OpRewritePattern<aten::Conv2dOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(aten::Conv2dOp srcConv2dOp,
                                PatternRewriter &rewriter) const override {
    auto results = srcConv2dOp.getOperation()->getResults();
    assert(srcConv2dOp.getNumOperands() == 7 && "expected seven (7) operands");
    assert(results.size() == 1 && "expected single result op");
    // TODO: Replace constant int-list constraints for stride, padding, and dilation; and, constant int constraint for groups.
    auto strideOp = srcConv2dOp.stride().getDefiningOp<Basicpy::BuildListOp>();
    if (!strideOp) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "expected basicpy.build_list to drive stride input");
    }
    if (strideOp.getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "expected stride length of 2");
    }
    auto *strideOperand0Op = strideOp.getOperand(0).getDefiningOp();
    auto *strideOperand1Op = strideOp.getOperand(1).getDefiningOp();
    if (!matchPattern(strideOperand0Op, m_One())
      || !matchPattern(strideOperand1Op, m_One())
      ) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "aten.conv2d to tcf.conv_2d_nchw currently only supports stride == [1, 1]");
    }
    auto paddingOp = srcConv2dOp.padding().getDefiningOp<Basicpy::BuildListOp>();
    if (!paddingOp) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "expected basicpy.build_list to drive padding input");
    }
    if (paddingOp.getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "expected padding length of 2");
    }
    auto *paddingOperand0Op = paddingOp.getOperand(0).getDefiningOp();
    auto *paddingOperand1Op = paddingOp.getOperand(1).getDefiningOp();
    if (!matchPattern(paddingOperand0Op, m_Zero())
      || !matchPattern(paddingOperand1Op, m_Zero())
      ) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "aten.conv2d to tcf.conv_2d_nchw currently only supports padding == [0, 0]");
    }
    auto dilationOp = srcConv2dOp.dilation().getDefiningOp<Basicpy::BuildListOp>();
    if (!dilationOp) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "expected basicpy.build_list to drive dilation input");
    }
    if (dilationOp.getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "expected dilation length of 2");
    }
    auto *dilationOperand0Op = dilationOp.getOperand(0).getDefiningOp();
    auto *dilationOperand1Op = dilationOp.getOperand(1).getDefiningOp();
    if (!matchPattern(dilationOperand0Op, m_One())
      || !matchPattern(dilationOperand1Op, m_One())
      ) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "aten.conv2d to tcf.conv_2d_nchw currently only supports dilation == [1, 1]");
    }
    if (!matchPattern(srcConv2dOp.groups(), m_One())
      ) {
      return rewriter.notifyMatchFailure(
          srcConv2dOp, "aten.conv2d to tcf.conv_2d_nchw currently only supports groups == 1");
    }
    auto tcfConvNCHWOp = rewriter.create<tcf::ConvNCHWOp>(
        srcConv2dOp.getLoc(), srcConv2dOp.getResult().getType(), srcConv2dOp.input(),
        srcConv2dOp.weight());
    // TODO: Reference Torch Conv2D's bias flag to conditionally create TCF Add.
    // (see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
    auto tcfConvNCHWBiasOp = rewriter.create<tcf::AddOp>(
        srcConv2dOp.getLoc(), srcConv2dOp.getResult().getType(), tcfConvNCHWOp.getResult(),
        srcConv2dOp.bias());
    rewriter.replaceOp(srcConv2dOp, tcfConvNCHWBiasOp.getResult());
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
  patterns.insert<ConvertBinaryElementwise<aten::MmOp, tcf::MatmulOp>>(context);
  patterns.insert<ConvertATenConv2d>(context);
}
