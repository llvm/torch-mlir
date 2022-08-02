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

namespace mlir {
namespace mhlo {
FailureOr<Value> getZeroRankTensor(PatternRewriter &rewriter, Operation *op,
                                   Value tensor) {
  auto rankTy = tensor.getType().dyn_cast<RankedTensorType>();
  if (!rankTy)
    return rewriter.notifyMatchFailure(
        op, "can not reshape a tensor that is not ranked to 0-rank");

  auto shape = rankTy.getShape();
  if (!(shape.size() == 1 && shape[0] == 1))
    return rewriter.notifyMatchFailure(op, "the shape must equal to [1]");

  return rewriter
      .create<mhlo::ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get(ArrayRef<int64_t>{}, rankTy.getElementType()),
          tensor)
      .getResult();
}

Value getReshapedTensor(PatternRewriter &rewriter, Operation *op, Value tensor,
                        ArrayRef<int64_t> shape, ArrayRef<Value> dimSizes) {
  // create mhlo::DynamicReshapeOp
  auto loc = op->getLoc();
  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto outRankTy = RankedTensorType::get(shape, tensorTy.getElementType());
  Value mhloShape = rewriter.create<tensor::FromElementsOp>(loc, dimSizes);
  return rewriter.create<mhlo::DynamicReshapeOp>(loc, outRankTy, tensor,
                                                 mhloShape);
}

Value getExpandedTensor(PatternRewriter &rewriter, Operation *op, Value tensor,
                        ArrayRef<Value> expandDimSizes, int64_t expandPos) {
  if (expandDimSizes.size() == 0) {
    return tensor;
  }

  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto dimSizes = *getDimSizesOfTensor(rewriter, op, tensor);
  int64_t rank = dimSizes.size();
  expandPos = (expandPos + rank) % rank;

  std::vector<Value> newDimSizes;
  std::vector<int64_t> newShape;
  for (int64_t k = 0; k < rank; ++k) {
    if (k == expandPos) {
      newDimSizes.insert(newDimSizes.end(), expandDimSizes.begin(),
                         expandDimSizes.end());
      for (size_t j = 0; j < expandDimSizes.size(); ++j) {
        newShape.push_back(ShapedType::kDynamicSize);
      }
    } else {
      newDimSizes.push_back(dimSizes[k]);
      newShape.push_back(tensorTy.getShape()[k]);
    }
  }

  return getReshapedTensor(rewriter, op, tensor, newShape, newDimSizes);
}

Value getProductOfDimSizes(PatternRewriter &rewriter, Operation *op,
                           ArrayRef<Value> dimSizes) {
  auto loc = op->getLoc();
  Type intTy = rewriter.getIntegerType(mhlo::kMhloDimSizeBits);
  auto prod =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intTy, 1))
          .getResult();

  for (auto &d : dimSizes) {
    prod = rewriter.create<arith::MulIOp>(loc, prod, d).getResult();
  }
  return prod;
}

FailureOr<std::tuple<Value, std::vector<Value>>>
getCollapsedTensor(PatternRewriter &rewriter, Operation *op, Value tensor,
                   ArrayRef<int64_t> inpCollapDims) {
  // Ref to XLA:Collapse:
  // https://www.tensorflow.org/xla/operation_semantics#collapse However we use
  // high to low dimension indices.
  //
  // Collapse replaces the given subset of the operand's dimensions by a single
  // dimension. The input arguments are an arbitrary array of type T and a
  // compile-time-constant vector of dimension indices. The dimension indices
  // must be an in-order (high to low dimension numbers), consecutive subset of
  // T's dimensions. Thus, {0, 1, 2}, {0, 1}, or {1, 2} are all valid dimension
  // sets, but {1, 0} or {0, 2} are not.
  auto nCollaps = inpCollapDims.size();
  std::vector<Value> collapDimSizes;
  if (nCollaps == 0) {
    return std::make_tuple(tensor, collapDimSizes);
  }

  // CHECK the input collapse dimensions are in-order, otherwise throw exception
  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  size_t rank = tensorTy.getRank();
  auto collapDims = toPositiveDims(inpCollapDims, rank);
  for (size_t k = 1; k < nCollaps; ++k)
    if (collapDims[k] != collapDims[k - 1] + 1)
      return rewriter.notifyMatchFailure(
          op, "collapse dimensions are not in consecutive order");

  // get original tensor shape in mlir standard dialect
  auto dimSizes = *getDimSizesOfTensor(rewriter, op, tensor);

  // calculate the collapse new_dim, which build the graph in mlir standard
  // dialect
  for (auto k : collapDims) {
    auto dsize = dimSizes[k];
    collapDimSizes.push_back(dsize);
  }

  // gather the new dim size values
  SmallVector<Value, 4> newDimSizes;
  SmallVector<int64_t, 4> newShape;
  for (size_t k = 0; k < collapDims[0]; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(tensorTy.getShape()[k]);
  }
  int64_t collapDimVal = 1;
  for (size_t k = collapDims[0]; k < collapDims[nCollaps - 1] + 1; ++k) {
    auto dsize = tensorTy.getShape()[k];
    if (dsize == ShapedType::kDynamicSize) {
      collapDimVal = ShapedType::kDynamicSize;
      break;
    }
    collapDimVal *= dsize;
  }
  newDimSizes.push_back(getProductOfDimSizes(rewriter, op, collapDimSizes));
  newShape.push_back(collapDimVal);
  for (size_t k = collapDims[nCollaps - 1] + 1; k < rank; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(tensorTy.getShape()[k]);
  }

  return std::make_tuple(
      getReshapedTensor(rewriter, op, tensor, newShape, newDimSizes),
      collapDimSizes);
}

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
  auto transDims = toPositiveDims(inpTransDims, rank);
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

FailureOr<Value> getDotProduct(PatternRewriter &rewriter, Operation *op,
                               Value lhs, Value rhs, int64_t rank) {
  if (rank < 2)
    return rewriter.notifyMatchFailure(
        op, "the input of DotProduct must has rank >= 2");

  std::vector<int64_t> batchDims;
  for (int64_t r = 0; r < rank - 2; ++r) {
    batchDims.push_back(r);
  }
  auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();

  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();

  // lhsShape[b, m, n], rhsShape[b', n', k] -> resultShape[b, m, k],
  // assert b == b' and n == n', but we could only verify it at runtime
  std::vector<int64_t> resultShape(lhsShape.begin(), lhsShape.end());
  resultShape[rank - 1] = rhsShape[rank - 1];

  auto loc = op->getLoc();
  auto resultTy = RankedTensorType::get(resultShape, lhsTy.getElementType());
  auto dotDimAttr = mhlo::DotDimensionNumbersAttr::get(
      op->getContext(), batchDims, batchDims, {rank - 1}, {rank - 2});
  auto result = rewriter.create<mhlo::DotGeneralOp>(
      loc, resultTy, lhs, rhs, dotDimAttr, /*precision_config*/ nullptr);
  return result.getResult();
}

FailureOr<Value> getBmmDotProduct(PatternRewriter &rewriter, Operation *op,
                                  Value inpLhs, Value inpRhs) {
  Value lhs = inpLhs;
  Value rhs = inpRhs;
  auto lhsRankTy = inpLhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankTy = inpRhs.getType().dyn_cast<RankedTensorType>();

  auto lhsRank = lhsRankTy.getRank();
  auto rhsRank = rhsRankTy.getRank();
  if (lhsRank < 2 || rhsRank < 2)
    return rewriter.notifyMatchFailure(
        op, "the input of batch-matmul must has rank >= 2");

  // The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
  // broadcastable).
  auto maxRank = std::max(lhsRank, rhsRank);
  auto minRank = std::min(lhsRank, rhsRank);
  if (maxRank != minRank) {
    auto leadingRank = maxRank - minRank;
    auto leadingDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, leadingRank));
    auto broadcastDims =
        llvm::to_vector<4>(llvm::seq<int64_t>(leadingRank, maxRank));
    auto lhsShape = lhsRankTy.getShape();
    auto rhsShape = rhsRankTy.getShape();
    if (lhsRank < rhsRank) {
      std::vector<int64_t> newShape(rhsShape.begin(),
                                    rhsShape.begin() + leadingRank);
      newShape.insert(newShape.end(), lhsShape.begin(), lhsShape.end());
      auto newDimSizes = *getDimSizesOfTensor(rewriter, op, rhs, leadingDims);
      auto lhsDimSizes = *getDimSizesOfTensor(rewriter, op, lhs);
      newDimSizes.insert(newDimSizes.end(), lhsDimSizes.begin(),
                         lhsDimSizes.end());
      lhs = getBroadcastTensor(rewriter, op, lhs, newShape, newDimSizes,
                               broadcastDims);
    } else {
      std::vector<int64_t> newShape(lhsShape.begin(),
                                    lhsShape.begin() + leadingRank);
      newShape.insert(newShape.end(), rhsShape.begin(), rhsShape.end());
      auto newDimSizes = *getDimSizesOfTensor(rewriter, op, lhs, leadingDims);
      auto rhsDimSizes = *getDimSizesOfTensor(rewriter, op, rhs);
      newDimSizes.insert(newDimSizes.end(), rhsDimSizes.begin(),
                         rhsDimSizes.end());
      rhs = getBroadcastTensor(rewriter, op, rhs, newShape, newDimSizes,
                               broadcastDims);
    }
  }

  // [?, ?, m, n] x [?, n, k] ==> batch_matmul([m,n], [n,k])
  return getDotProduct(rewriter, op, lhs, rhs, /*rank*/ maxRank);
}

FailureOr<Value> getMmDotProduct(PatternRewriter &rewriter, Operation *op,
                                 Value inpLhs, Value inpRhs) {
  auto lhsRankTy = inpLhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankTy = inpRhs.getType().dyn_cast<RankedTensorType>();

  auto lhsRank = lhsRankTy.getRank();
  auto rhsRank = rhsRankTy.getRank();
  if (lhsRank < 2)
    return rewriter.notifyMatchFailure(
        op, "the left hand-side input of matmul must has rank >= 2");
  if (rhsRank != 2)
    return rewriter.notifyMatchFailure(
        op, "the right hand-side input of matmul must has rank == 2");

  Value lhs = inpLhs;
  Value rhs = inpRhs;
  // [?, m, n] x [n, k] ==> [?xm, n] x [n, k]
  std::vector<Value> collapDimSizes;
  if (lhsRank > 2) {
    std::vector<int64_t> collapDims;
    for (int64_t d = 0; d < lhsRank - 1; ++d) {
      collapDims.push_back(d);
    }
    auto collapDimSizesInfo = getCollapsedTensor(rewriter, op, lhs, collapDims);
    if (failed(collapDimSizesInfo))
      return rewriter.notifyMatchFailure(
          op, "failed to construct matrix-matrix multiply");
    std::tie(lhs, collapDimSizes) = *collapDimSizesInfo;
  }
  auto result = getDotProduct(rewriter, op, lhs, rhs, /*rank*/ 2);
  if (failed(result))
    return rewriter.notifyMatchFailure(
        op, "failed to construct matrix-matrix multiply");

  return getExpandedTensor(rewriter, op, *result, collapDimSizes,
                           /*expandPos*/ 0);
}

FailureOr<Value> getMvDotProduct(PatternRewriter &rewriter, Operation *op,
                                 Value inpLhs, Value inpRhs) {
  auto lhsRankTy = inpLhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankTy = inpRhs.getType().dyn_cast<RankedTensorType>();

  auto lhsRank = lhsRankTy.getRank();
  auto rhsRank = rhsRankTy.getRank();

  if (rhsRank != 1)
    return rewriter.notifyMatchFailure(
        op, "the right hand-side input of matmul must has rank == 1");
  if (lhsRank < 2)
    return rewriter.notifyMatchFailure(
        op, "the left hand-side input of matmul must has rank >= 2");

  auto unsqzRhsInfo = mhlo::unsqueezeTensor(rewriter, op, inpRhs, {1});
  if (failed(unsqzRhsInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to unsqueeze right hand-side input to rank 2");

  auto unsqzRhs = *unsqzRhsInfo;
  auto product = getMmDotProduct(rewriter, op, inpLhs, unsqzRhs);
  if (failed(product))
    return rewriter.notifyMatchFailure(
        op, "failed to construct matrix-vector multiply");
  Value result = *product;
  std::vector<Value> collapDimSizes;
  auto collapDimSizesInfo = getCollapsedTensor(rewriter, op, result, {-2, -1});
  if (failed(collapDimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to construct matrix-vector multiply");
  std::tie(result, collapDimSizes) = *collapDimSizesInfo;
  return result;
}
} // namespace mhlo
} // namespace mlir

namespace {
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

    FailureOr<Value> product;
    if (rhsRank == 1) {
      if (lhsRank == 1) {
        // If both tensors are 1-dimensional, the dot product (scalar) is
        // returned.
        auto unsqzLhs = mhlo::unsqueezeTensor(rewriter, op, lhs, {0});
        product = mhlo::getMvDotProduct(rewriter, op, *unsqzLhs, rhs);
        product = mhlo::getZeroRankTensor(rewriter, op, *product);
      } else {
        // If the first argument is 2-dimensional and the second argument is
        // 1-dimensional, the matrix-vector product is returned.
        // NB: if lhsRank > 2 reshape it to rank 2.
        product = mhlo::getMvDotProduct(rewriter, op, lhs, rhs);
      }
    } else if (rhsRank == 2) {
      if (lhsRank == 1) {
        // If the first argument is 1-dimensional, a 1 is prepended to its
        // dimension for the purpose of the batched matrix multiply and removed
        // after.
        auto unsqzLhs = mhlo::unsqueezeTensor(rewriter, op, lhs, {0});
        product = mhlo::getMmDotProduct(rewriter, op, *unsqzLhs, rhs);
        auto collapDimSizesInfo =
            mhlo::getCollapsedTensor(rewriter, op, *product, {-2, -1});
        if (failed(collapDimSizesInfo))
          return op.emitError("failed to construct matrix-vector multiply");

        std::vector<Value> collapDimSizes;
        std::tie(product, collapDimSizes) = *collapDimSizesInfo;
      } else {
        // If both arguments are 2-dimensional, the matrix-matrix product is
        // returned. NB: if lhsRank > 2 reshape it to rank 2.
        product = mhlo::getMmDotProduct(rewriter, op, lhs, rhs);
      }
    } else {
      // rhsRank > 2
      if (lhsRank == 1) {
        // If the first argument is 1-dimensional, a 1 is prepended to its
        // dimension for the purpose of the batched matrix multiply and removed
        // after.
        auto unsqzLhs = mhlo::unsqueezeTensor(rewriter, op, lhs, {0});
        product = mhlo::getBmmDotProduct(rewriter, op, *unsqzLhs, rhs);
        auto collapDimSizesInfo =
            mhlo::getCollapsedTensor(rewriter, op, *product, {-2, -1});
        if (failed(collapDimSizesInfo))
          return op.emitError("failed to construct matrix-vector multiply");

        std::vector<Value> collapDimSizes;
        std::tie(product, collapDimSizes) = *collapDimSizesInfo;
      } else {
        product = mhlo::getBmmDotProduct(rewriter, op, lhs, rhs);
      }
    }
    if (failed(product))
      return op.emitError("matmul: conversion failed");
    output = *product;
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
    auto weightT = mhlo::getPermutedTensor(rewriter, op, rhs, {1, 0});
    auto product = mhlo::getMmDotProduct(rewriter, op, lhs, weightT);
    if (failed(product))
      return op.emitError("failed to perform matmul operation");

    Value matmulOutput = *product;
    Value matmulPlusBias = matmulOutput;

    if (!biasTy.template isa<Torch::NoneType>()) {
      // Bias addition broadcasts to the matmul output shape.
      matmulPlusBias = rewriter
                           .create<chlo::BroadcastAddOp>(
                               op->getLoc(), matmulOutput.getType(),
                               matmulOutput, bias, nullptr)
                           .getResult();
    }

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        matmulPlusBias);
    return success();
  }
};

} // namespace

void mlir::torch::torch_to_mhlo::populateMatmulOpPatternsAndLegality(
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
