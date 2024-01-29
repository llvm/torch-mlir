//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "SimplifyAbstractInterpCalculationsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static LogicalResult refineDtypeCalculateResult(DtypeCalculateOp op,
                                                int resultNum,
                                                PatternRewriter &rewriter) {
  auto yieldDtypes = op.getCalculation().front().getTerminator();
  auto dtype = yieldDtypes->getOperand(resultNum);
  auto result = op->getResult(resultNum);

  int64_t dtypeInt;
  if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    return rewriter.notifyMatchFailure(
        op, "Expected result from the DtypeCalculateOp calculation to be a "
            "constant int");
  auto dtypeScalarType = static_cast<torch_upstream::ScalarType>(dtypeInt);

  // Calculate the updated type incorporating the new information.
  Type impliedTypeFromDtype;
  if (result.getType().isa<Torch::NumberType>()) {
    FailureOr<Type> torchType =
        getTorchTypeForScalarType(op->getContext(), dtypeScalarType);
    if (failed(torchType)) {
      return rewriter.notifyMatchFailure(
          op, "Failed to convert result dtype to `Torch::FloatType` or "
              "`Torch::IntType`");
    }
    impliedTypeFromDtype = *torchType;
  } else if (auto originalResultType =
                 result.getType().dyn_cast<BaseTensorType>()) {
    FailureOr<Type> builtinType =
        getTypeForScalarType(op->getContext(), dtypeScalarType);
    if (failed(builtinType)) {
      return rewriter.notifyMatchFailure(
          op, "Failed to convert `dtypeScalarType` to a builtin type");
    }
    impliedTypeFromDtype =
        originalResultType.cast<BaseTensorType>().getWithSizesAndDtype(
            originalResultType.getOptionalSizes(), *builtinType);
  } else {
    return rewriter.notifyMatchFailure(op,
                                       "Unimplemented: Expected result type to "
                                       "be `BaseTensorType` or `NumberType`");
  }
  return updateCalculateOpResultTypes(op, resultNum, impliedTypeFromDtype,
                                      rewriter);
}

namespace {
// This pattern propagates information out of the dtype calculation region and
// into the DtypeCalculateOp result types.
class RefineDtypeCalculateOp : public OpRewritePattern<DtypeCalculateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DtypeCalculateOp op,
                                PatternRewriter &rewriter) const override {
    LogicalResult result = failure();
    for (int i = 0, e = op->getNumResults(); i != e; i++) {
      if (succeeded(refineDtypeCalculateResult(op, i, rewriter)))
        result = success();
    }
    return result;
  }
};
} // namespace

namespace {
class DecomposePromoteDtypesOp : public OpRewritePattern<PromoteDtypesOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PromoteDtypesOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<std::optional<int64_t>> ranks;
    SmallVector<int64_t> dtypes;
    if (!matchPattern(op.getRanks(),
                      m_TorchListOfOptionalConstantInts(ranks))) {
      return rewriter.notifyMatchFailure(
          op, "Expected `ranks` to be a list of optional constant ints");
    }

    if (!matchPattern(op.getDtypes(), m_TorchListOfConstantInts(dtypes))) {
      return rewriter.notifyMatchFailure(
          op, "Expected `dtypes` to be a list of constant ints");
    }

    if (ranks.empty() || dtypes.empty()) {
      return rewriter.notifyMatchFailure(
          op, "`ranks` list and `dtypes` list must be non-empty");
    }

    if (ranks.size() != dtypes.size()) {
      return rewriter.notifyMatchFailure(
          op, "`ranks` list and `dtypes` list must have the same size");
    }

    torch_upstream::ResultTypeState state{};
    for (auto ranksAndDtypes : llvm::zip(ranks, dtypes)) {
      std::optional<int64_t> rank;
      int64_t dtype;
      std::tie(rank, dtype) = ranksAndDtypes;
      auto scalarType = static_cast<torch_upstream::ScalarType>(dtype);

      bool isScalarOnlyOp = llvm::all_of(
          ranks, [](std::optional<int64_t> rank) { return !rank.has_value(); });

      if (!rank.has_value()) {
        // If `rank` does not have a value, then we are dealing with a scalar
        // input. For the type promotion, the behavior of a scalar argument is
        // dependent on whether the op is performing an operation with only
        // scalars (such as AtenAddOp) or with scalars and tensors (such as
        // AtenAddScalarOp). Therefore, we convert back to the original torch
        // type of the scalar first, and then determine the right scalar type to
        // use for promotion based on whether the op involves only scalars or
        // scalars and tensors.
        FailureOr<Type> torchType =
            getTorchTypeForScalarType(op->getContext(), scalarType);
        if (failed(torchType)) {
          return rewriter.notifyMatchFailure(
              op, "Dtypes for arguments scalars must be convertible to "
                  "`Torch::FloatType` or `Torch::IntType`");
        }
        Type builtinType = isScalarOnlyOp
                               ? getBuiltInTypeForTorchScalar(*torchType)
                               : getDefaultDtypeForTorchScalar(*torchType);
        scalarType = getScalarTypeForType(builtinType);
        state.wrappedResult =
            promote_skip_undefined(state.wrappedResult, scalarType);
      } else if (rank.value() == 0) {
        state.zeroResult = promote_skip_undefined(state.zeroResult, scalarType);
      } else if (rank.value() > 0) {
        state.dimResult = promote_skip_undefined(state.dimResult, scalarType);
      } else {
        return rewriter.notifyMatchFailure(op, "Rank should not be negative");
      }
    }

    auto resultType = static_cast<int64_t>(result_type(state));
    rewriter.replaceOpWithNewOp<ConstantIntOp>(
        op, rewriter.getI64IntegerAttr(resultType));
    return success();
  }
};
} // namespace

namespace {
class RefineNumToTensorScalarOpType
    : public OpRewritePattern<PrimNumToTensorScalarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimNumToTensorScalarOp op,
                                PatternRewriter &rewriter) const override {
    auto originalResultType = op.getResult().getType().cast<BaseTensorType>();
    if (originalResultType.hasDtype())
      return rewriter.notifyMatchFailure(
          op, "`PrimNumToTensorScalarOp` already has a dtype");

    if (op.getA().getType().isa<Torch::NumberType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "`PrimNumToTensorScalarOp`'s input "
                                         "should have concrete Scalar Type.");
    }
    Type inputType = getBuiltInTypeForTorchScalar(op.getA().getType());
    auto impliedTypeFromInputType =
        originalResultType.cast<BaseTensorType>()
            .getWithSizesAndDtype(originalResultType.getOptionalSizes(),
                                  inputType)
            .cast<BaseTensorType>();

    op.getResult().setType(impliedTypeFromInputType);
    return success();
  }
};
} // namespace

namespace {
class SimplifyDtypeCalculationsPass
    : public SimplifyDtypeCalculationsBase<SimplifyDtypeCalculationsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateFullyUnrollPrimLoopOpPattern(patterns, context);
    populateAbstractlyInterpretListOpsWithinABlockPattern(patterns, context);
    populateFoldPrimUncheckedCastOpPattern(patterns, context);
    patterns.insert<RefineDtypeCalculateOp>(context);
    patterns.insert<DecomposePromoteDtypesOp>(context);
    patterns.insert<RefineNumToTensorScalarOpType>(context);

    PrimIfOp::getCanonicalizationPatterns(patterns, context);
    Aten__Getitem__TOp::getCanonicalizationPatterns(patterns, context);
    PrimTupleUnpackOp::getCanonicalizationPatterns(patterns, context);

    // TODO: Debug visitation order to make this more efficient.
    // A single linear scan should suffice.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createSimplifyDtypeCalculationsPass() {
  return std::make_unique<SimplifyDtypeCalculationsPass>();
}
