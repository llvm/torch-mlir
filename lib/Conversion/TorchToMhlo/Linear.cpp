//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include "../PassDetail.h"
#include "./MhloLegalizeUtils.h"
#include "./PopulatePatterns.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
Value getBroadcastTensor(PatternRewriter &rewriter, Operation *op, Value tensor,
                         ArrayRef<int64_t> shape, ArrayRef<Value> dimSizes,
                         ArrayRef<int64_t> broadcastDims) {
  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto loc = op->getLoc();
  Value mhloShape = rewriter.create<tensor::FromElementsOp>(loc, dimSizes);

  RankedTensorType outTy =
      RankedTensorType::get(shape, tensorTy.getElementType());

  RankedTensorType attrTy =
      RankedTensorType::get({static_cast<int64_t>(broadcastDims.size())},
                            rewriter.getIntegerType(64));
  auto broadcastAttr = DenseIntElementsAttr::get(attrTy, broadcastDims);

  auto broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
      loc, outTy, tensor, mhloShape, broadcastAttr);
  return broadcast;
}

Value getPermutedTensor(PatternRewriter &rewriter, Operation *op, Value input,
                        ArrayRef<int64_t> inpTransDims) {
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto rank = inputTy.getRank();
  auto transDims = mhlo::toPositiveDims(inpTransDims, rank);
  auto inpShape = inputTy.getShape();
  std::vector<int64_t> newShape;
  newShape.reserve(rank);

  for (auto d : transDims) {
    newShape.push_back(inpShape[d]);
  }

  auto attrTy = RankedTensorType::get({static_cast<int64_t>(transDims.size())},
                                      rewriter.getIntegerType(64));
  auto permuteAttr = DenseIntElementsAttr::get(attrTy, transDims);

  auto outTy = RankedTensorType::get(newShape, inputTy.getElementType());
  auto result = rewriter.create<mhlo::TransposeOp>(op->getLoc(), outTy, input,
                                                   permuteAttr);
  return result.getResult();
}

void getBmmBroadcast(PatternRewriter &rewriter, Operation *op, Value &inpLhs,
                     Value &inpRhs, int64_t leadingRank) {
  Value lhs = inpLhs;
  Value rhs = inpRhs;
  auto lhsRankTy = inpLhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankTy = inpRhs.getType().dyn_cast<RankedTensorType>();

  auto lhsRank = lhsRankTy.getRank();
  auto rhsRank = rhsRankTy.getRank();

  // The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
  // broadcastable).
  auto minRank = std::min(lhsRank, rhsRank);
  auto leadingDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, leadingRank));
  auto broadcastDims = llvm::to_vector<4>(
      llvm::seq<int64_t>(leadingRank, minRank + leadingRank));
  auto lhsShape = lhsRankTy.getShape();
  auto rhsShape = rhsRankTy.getShape();
  if (lhsRank < rhsRank) {
    std::vector<int64_t> newShape(rhsShape.begin(),
                                  rhsShape.begin() + leadingRank);
    newShape.insert(newShape.end(), lhsShape.begin(), lhsShape.end());
    auto newDimSizes =
        *mhlo::getDimSizesOfTensor(rewriter, op, rhs, leadingDims);
    auto lhsDimSizes = *mhlo::getDimSizesOfTensor(rewriter, op, lhs);
    newDimSizes.insert(newDimSizes.end(), lhsDimSizes.begin(),
                       lhsDimSizes.end());
    lhs = getBroadcastTensor(rewriter, op, lhs, newShape, newDimSizes,
                             broadcastDims);
  } else {
    std::vector<int64_t> newShape(lhsShape.begin(),
                                  lhsShape.begin() + leadingRank);
    newShape.insert(newShape.end(), rhsShape.begin(), rhsShape.end());
    auto newDimSizes =
        *mhlo::getDimSizesOfTensor(rewriter, op, lhs, leadingDims);
    auto rhsDimSizes = *mhlo::getDimSizesOfTensor(rewriter, op, rhs);
    newDimSizes.insert(newDimSizes.end(), rhsDimSizes.begin(),
                       rhsDimSizes.end());
    rhs = getBroadcastTensor(rewriter, op, rhs, newShape, newDimSizes,
                             broadcastDims);
  }

  inpLhs = lhs;
  inpRhs = rhs;
}

// Perform the basic n-dim matmul operation encompassing the handling of
// broadcasting and dynamic shape propagation.
// All PyTorch ops that leverage matrix multiplication will derive this and
// implement their specialized input processing (e.g transpose), and output
// processing, e.g. GEMM or fully connected bias handling.
template <typename AtenOpT>
class ConvertAtenMatmulBaseOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  // Each variant must implement corresponding parameter parsing options.
  // Maintain separate input read functions for each variant because it is not
  // necessarily true with all variants that the first two operands are the lhs
  // and rhs.
  virtual LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         Value &lhs, Value &rhs) const {
    return rewriter.notifyMatchFailure(
        op,
        "unimplemented matrix multiplication variant input parsing function");
  }
  LogicalResult performMatmul(AtenOpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &lhs,
                              Value &rhs, Value &output) const {
    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();
    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return op.emitError("matmul: input datatypes mismatched");
    if (lhsRank < 1 || rhsRank < 1) {
      return op.emitError("matmul: inputs can't be 0-rank");
    }

    if (lhsRank <= 2 && rhsRank <= 2) {
      output = rewriter.create<mhlo::DotOp>(op->getLoc(), lhs, rhs, nullptr);
      return success();
    }

    int64_t nBatchDims;
    if (rhsRank <= 2) {
      auto leadingRank = lhsRank - 2;
      getBmmBroadcast(rewriter, op, lhs, rhs, leadingRank);
      nBatchDims = leadingRank;
    } else if (lhsRank <= 2) {
      auto leadingRank = rhsRank - 2;
      getBmmBroadcast(rewriter, op, lhs, rhs, leadingRank);
      nBatchDims = leadingRank;
    } else {
      assert(rhsRank > 2 && lhsRank > 2);
      auto leadingRank = std::max(lhsRank - rhsRank, rhsRank - lhsRank);
      nBatchDims = std::max(lhsRank - 2, rhsRank - 2);
      getBmmBroadcast(rewriter, op, lhs, rhs, leadingRank);
    }
    auto batchDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, nBatchDims));
    auto lhsContractingDim = nBatchDims + 1;
    auto rhsContractingDim = nBatchDims;
    if (lhsRank == 1)
      lhsContractingDim = nBatchDims;

    mhlo::DotDimensionNumbersAttr dotDimensionNumbers =
        mhlo::DotDimensionNumbersAttr::get(
            rewriter.getContext(),
            /*lhsBatchingDimensions=*/batchDims,
            /*rhsBatchingDimensions=*/batchDims,
            /*lhsContractingDimensions=*/{lhsContractingDim},
            /*rhsContractingDimensions=*/{rhsContractingDim});
    auto resultTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(op.getType())
                        .template cast<RankedTensorType>();

    output = rewriter
                 .create<mhlo::DotGeneralOp>(op->getLoc(), resultTy, lhs, rhs,
                                             dotDimensionNumbers, nullptr)
                 .getResult();

    return success();
  }

  // The default version just reads two inputs, computes output and returns it.
  // Other versions may add a bias, apply GEMM-style alpha/beta scaling etc.
  virtual LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs, rhs;
    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return op.emitError("failed to read matmul inputs");

    Value output;

    if (failed(performMatmul(op, adaptor, rewriter, lhs, rhs, output)))
      return op.emitError("failed to perform matmul operation");

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>(),
        output);

    return success();
  }
};

// Legalizes the torch.matmul op for general n-dim matmul.
template <typename AtenOpT>
class ConvertAtenMatMulOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.other();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError(
          "only ranked tensor types are supported in MHLO matmul");

    return success();
  }
};

// Implements handling of aten.mm and aten.bmm ops.
template <typename AtenOpT>
class ConvertAtenMmOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.mat2();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError(
          "only ranked tensor types are supported in MHLO matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (isa<AtenMmOp>(op)) {
      // Mm takes two 2D tensors.
      if (lhsRank != 2 || rhsRank != 2)
        return op.emitError("aten.mm called but matrix rank != 2");
    } else if (isa<AtenBmmOp>(op)) {
      // Bmm takes two 3D tensors.
      if (lhsRank != 3 || rhsRank != 3)
        return op.emitError("aten.bmm called but matrix rank != 3");
    }

    return success();
  }
};

// Implements handling of aten.linear op.
template <typename AtenOpT>
class ConvertAtenLinearOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.input();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.weight();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError(
          "only ranked tensor types are supported in MHLO matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (lhsRank != 2 && lhsRank != 3)
      return op.emitError("aten.Linear called but input rank not 2 or 3");
    if (rhsRank != 2 && rhsRank != 3)
      return op.emitError("aten.Linear called but weight rank not 2 or 3");

    return success();
  }
  // Override the default rewriter to perform RHS transpose and bias addition
  // as well.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return op.emitError("failed to read matmul inputs");

    // The aten.Linear op has a bias tensor that is added to the matmul
    // output.
    auto bias = adaptor.bias();
    auto biasTy = bias.getType();

    // MHLO does not mandate that elementwise op tensors need to be ranked.
    if (!biasTy.template isa<Torch::NoneType>() &&
        !biasTy.template isa<RankedTensorType>())
      return op.emitError("only ranked tensor types are supported in MHLO "
                          "matmul for bias tensor");

    // weight.T
    rhs = getPermutedTensor(rewriter, op, rhs, {1, 0});

    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();
    auto leadingRank = std::max(lhsTy.getRank() - rhsTy.getRank(),
                                rhsTy.getRank() - lhsTy.getRank());
    getBmmBroadcast(rewriter, op, lhs, rhs, leadingRank);
    auto resultRank = std::max(lhsTy.getRank(), rhsTy.getRank());
    auto nBatchDims = resultRank - 2;
    auto batchDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, nBatchDims));
    auto lhsContractingDim = nBatchDims + 1;
    auto rhsContractingDim = nBatchDims;

    mhlo::DotDimensionNumbersAttr dotDimensionNumbers =
        mhlo::DotDimensionNumbersAttr::get(
            rewriter.getContext(),
            /*lhsBatchingDimensions=*/batchDims,
            /*rhsBatchingDimensions=*/batchDims,
            /*lhsContractingDimensions=*/{lhsContractingDim},
            /*rhsContractingDimensions=*/{rhsContractingDim});

    auto resultTy =
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType());

    Value matmulOutput = rewriter.create<mhlo::DotGeneralOp>(
        op->getLoc(), resultTy, lhs, rhs, dotDimensionNumbers, nullptr);

    Value matmulPlusBias = matmulOutput;
    if (!biasTy.template isa<Torch::NoneType>()) {
      // Bias addition broadcasts to the matmul output shape.
      matmulPlusBias =
          rewriter
              .create<chlo::BroadcastAddOp>(op->getLoc(), resultTy,
                                            matmulOutput, bias, nullptr)
              .getResult();
    }

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, resultTy, matmulPlusBias);
    return success();
  }
};

} // namespace

void mlir::torch::torch_to_mhlo::populateLinearOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

#define INSERT_MATMUL_ATENOP_PATTERN(AtenOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMatMulOp<AtenOp>>(typeConverter, context);
  INSERT_MATMUL_ATENOP_PATTERN(AtenMatmulOp);
#undef INSERT_MATMUL_ATEMOP_PATTERN

#define INSERT_MM_ATENOP_PATTERN(AtenOp)                                       \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMmOp<AtenOp>>(typeConverter, context);
  INSERT_MM_ATENOP_PATTERN(AtenMmOp);
  INSERT_MM_ATENOP_PATTERN(AtenBmmOp);
#undef INSERT_MM_ATEMOP_PATTERN

#define INSERT_LINEAR_ATENOP_PATTERN(AtenOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenLinearOp<AtenOp>>(typeConverter, context);
  INSERT_LINEAR_ATENOP_PATTERN(AtenLinearOp);
#undef INSERT_LINEAR_ATEMOP_PATTERN
}
