//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;
using namespace mlir::torch::torch_to_stablehlo;

namespace {
// A dimension index from torch.dialect might outside the range [0, dimSize].
// The function is used to normalize the input index into the range.
Value getNormalizedDimSizeInternal(PatternRewriter &rewriter, Operation *op,
                                   Value index, Value dimSize) {
  auto loc = op->getLoc();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0));

  // To normalize index into range [-dimSize, dimSize]
  // index = min(max(-dimSize, index), dimSize)
  auto negDimSize = rewriter.create<arith::SubIOp>(loc, zero, dimSize);
  index = rewriter.create<arith::MaxSIOp>(loc, negDimSize, index);
  index = rewriter.create<arith::MinSIOp>(loc, dimSize, index);

  auto dimSizePlusIndex = rewriter.create<arith::AddIOp>(loc, dimSize, index);
  auto indexPositive = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, index, zero);
  // get positive index: (index >=0) ? index: index + dimSize
  return rewriter.create<arith::SelectOp>(loc, indexPositive, index,
                                          dimSizePlusIndex);
}

Value getDynamicSliceInternal(PatternRewriter &rewriter, Operation *op,
                              Type outTy, Value input, Value startIndex,
                              Value endIndex, Value step, size_t dimIndex,
                              ArrayRef<Value> dimSizes,
                              size_t dimSizeIndexBits) {
  auto loc = op->getLoc();
  // startIndex & endIndex has been normailized into range [0, dSize]
  Type intType = rewriter.getIntegerType(dimSizeIndexBits);
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 0));
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));

  SmallVector<Value, 4> startIndices;
  SmallVector<Value, 4> endIndices;
  SmallVector<Value, 4> strides;

  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  size_t rank = inputTy.getRank();
  startIndices.reserve(rank);
  endIndices.reserve(rank);
  strides.reserve(rank);

  auto endIndexIsZero = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, endIndex, zero);
  endIndex = rewriter.create<arith::SelectOp>(loc, endIndexIsZero,
                                              dimSizes[dimIndex], endIndex);

  for (size_t r = 0; r < rank; ++r) {
    if (r == dimIndex) {
      startIndices.push_back(startIndex);
      endIndices.push_back(endIndex);
      strides.push_back(step);
    } else {
      startIndices.push_back(zero);
      endIndices.push_back(dimSizes[r]);
      strides.push_back(one);
    }
  }

  auto startTensor =
      rewriter.create<tensor::FromElementsOp>(loc, startIndices).getResult();
  auto endTensor =
      rewriter.create<tensor::FromElementsOp>(loc, endIndices).getResult();
  auto stridesTensor =
      rewriter.create<tensor::FromElementsOp>(loc, strides).getResult();

  return rewriter.create<stablehlo::RealDynamicSliceOp>(
      loc, outTy, input, startTensor, endTensor, stridesTensor);
}

// Get a dynamic slice of the tensor from startIndex to endIndex with stride
// step on the specifed dimension. The input startIndex(default to 0),
// endIndex(default to dimSize), and step(default to 1) can be optional.
FailureOr<Value> getDynamicSlice(PatternRewriter &rewriter, Operation *op,
                                 Type outTy, Value input,
                                 std::optional<Value> startIndexOpt,
                                 std::optional<Value> endIndexOpt,
                                 std::optional<Value> stepOpt, int64_t dim,
                                 size_t dimSizeIndexBits) {
  auto loc = op->getLoc();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto rank = inputTy.getRank();

  dim = (dim + rank) % rank;
  Value dimSize = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(),
      rewriter.create<tensor::DimOp>(loc, input, dim));

  Value normStartIndex =
      startIndexOpt
          ? getNormalizedDimSizeInternal(rewriter, op, *startIndexOpt, dimSize)
          : rewriter.create<arith::ConstantOp>(
                loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
  Value normEndIndex =
      endIndexOpt
          ? getNormalizedDimSizeInternal(rewriter, op, *endIndexOpt, dimSize)
          : dimSize;
  Value step =
      stepOpt ? *stepOpt
              : rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 1));

  if (dimSizeIndexBits == 32) {
    Type intType = rewriter.getIntegerType(dimSizeIndexBits);
    normStartIndex =
        rewriter.create<arith::TruncIOp>(loc, intType, normStartIndex);
    normEndIndex = rewriter.create<arith::TruncIOp>(loc, intType, normEndIndex);
    step = rewriter.create<arith::TruncIOp>(loc, intType, step);
  }
  FailureOr<SmallVector<Value, 4>> dimSizesInfo =
      hlo::getDimSizesOfTensor(rewriter, op, input, dimSizeIndexBits);
  if (failed(dimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");

  auto dimSizes = *dimSizesInfo;
  return getDynamicSliceInternal(rewriter, op, outTy, input, normStartIndex,
                                 normEndIndex, step, dim, dimSizes,
                                 dimSizeIndexBits);
}

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenViewOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rankType =
        adaptor.getSelf().getType().template dyn_cast<RankedTensorType>();
    if (!rankType)
      return op.emitError("Only ranked tensor types are currently supported");

    SmallVector<Value, 4> dimSizes;
    if (!getAtenViewOpSizes(op, adaptor, rewriter, dimSizes)) {
      return op.emitError("Dims size must be a list of Scalar");
    }

    auto loc = op.getLoc();
    auto newRank = dimSizes.size();
    if (newRank == 0 || rankType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          adaptor.getSelf());
      return success();
    }

    std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value &dSize) {
      dSize = rewriter.create<ToI64Op>(loc, dSize).getResult();
      return dSize;
    });

    const auto &options = ConvertAtenOp<AtenOpT>::getOptions();
    Type intType = rewriter.getIntegerType(options.dimSizeIndexBits);
    if (options.dimSizeIndexBits == 32) {
      // The i64 calculation is much slower than i32 on some devices, such as
      // Nvidia GPU. One can truncate from i64 to i32 since dimension sizes are
      // unlikely to exceed the range of i32(4GiB)
      std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value &dSize) {
        // dimSize: cast i64 -> i32
        dSize = rewriter.create<arith::TruncIOp>(loc, intType, dSize);
        return dSize;
      });
    }

    Value numel = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(intType, 1));
    for (auto d : dimSizes) {
      numel = rewriter.create<arith::MulIOp>(loc, numel, d);
    }
    numel = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                numel);

    if (dimSizes.size() == 0) {
      rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          adaptor.getSelf());
      return success();
    }
    Value stablehloShape =
        rewriter.create<tensor::FromElementsOp>(loc, dimSizes);
    Value computedShape = rewriter.create<stablehlo::ComputeReshapeShapeOp>(
        loc, stablehloShape.getType(), numel, stablehloShape);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.getSelf(), computedShape);
    return success();
  }

  bool getAtenViewOpSizes(AtenOpT op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter,
                          SmallVector<Value, 4> &dimSizes) const;
};

template <>
bool ConvertAtenViewOp<AtenViewOp>::getAtenViewOpSizes(
    AtenViewOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    SmallVector<Value, 4> &dimSizes) const {
  return getListConstructElements(adaptor.getSize(), dimSizes);
}

template <>
bool ConvertAtenViewOp<AtenReshapeOp>::getAtenViewOpSizes(
    AtenReshapeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    SmallVector<Value, 4> &dimSizes) const {
  return getListConstructElements(adaptor.getShape(), dimSizes);
}
} // namespace

template <>
LogicalResult ConvertAtenOp<AtenSliceTensorOp>::matchAndRewrite(
    AtenSliceTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = self.getType().cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");
  auto outTy =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "only constant dim is currently supported");
  int64_t inputRank = selfTy.getRank();
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  auto getOptionalVal = [&](Value val) -> std::optional<Value> {
    if (val.getType().isa<Torch::NoneType>()) {
      return std::nullopt;
    } else {
      return val;
    }
  };

  std::optional<Value> start = getOptionalVal(adaptor.getStart());
  std::optional<Value> end = getOptionalVal(adaptor.getEnd());
  std::optional<Value> step = getOptionalVal(adaptor.getStep());

  FailureOr<Value> sliceInfo =
      getDynamicSlice(rewriter, op, outTy, self, start, end, step, dim,
                      options.dimSizeIndexBits);
  if (failed(sliceInfo))
    return op.emitError("can not create a dynmaic slice");

  auto slice = *sliceInfo;
  rewriter.replaceOp(op, slice);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqueezeOp>::matchAndRewrite(
    AtenSqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = self.getType().cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");

  auto rank = selfTy.getRank();
  if (rank == 0)
    return rewriter.notifyMatchFailure(
        op, "The rank of tensor must be greater than 0");

  SmallVector<int64_t, 4> dims;
  dims.reserve(rank);
  for (int r = 0; r < rank; ++r) {
    auto dSize = selfTy.getShape()[r];
    if (dSize == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "the size of the dimension being squeezed can't be unknown");
    if (dSize != 1)
      dims.push_back(r);
  }
  if (dims.size() == 0) {
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }

  auto newDimSizesInfo = hlo::getDimSizesOfTensor(rewriter, op, self, dims,
                                                  options.dimSizeIndexBits);
  if (failed(newDimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  auto newDimSizes = *newDimSizesInfo;
  auto stablehloShape =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), newDimSizes);
  rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), self, stablehloShape);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqueezeDimOp>::matchAndRewrite(
    AtenSqueezeDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = self.getType().cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");

  auto rank = selfTy.getRank();
  if (rank == 0)
    return rewriter.notifyMatchFailure(
        op, "the rank of tensor must be greater than 0");

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "only constant dim is currently supported");
  dim = toPositiveDim(dim, rank);
  if (!isValidDim(dim, rank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  if (selfTy.getShape()[dim] != 1) {
    if (selfTy.getShape()[dim] == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "the size of the dimension being squeezed is can't be unknown");

    rewriter.replaceOp(op, adaptor.getSelf());
    return success();
  }

  SmallVector<int64_t, 4> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  dims.erase(dims.begin() + dim);
  if (dims.size() == 0) {
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }
  auto newDimSizesInfo = hlo::getDimSizesOfTensor(rewriter, op, self, dims,
                                                  options.dimSizeIndexBits);
  if (failed(newDimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  auto newDimSizes = *newDimSizesInfo;
  auto stablehloShape =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), newDimSizes);
  rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), self, stablehloShape);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUnsqueezeOp>::matchAndRewrite(
    AtenUnsqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto selfType = adaptor.getSelf().getType().dyn_cast<TensorType>();
  if (!selfType) {
    return op.emitError("only tensor types are currently supported");
  }

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return op->emitError("dim must be a Scalar constant");
  int64_t inputRank =
      adaptor.getSelf().getType().cast<RankedTensorType>().getRank();
  dim = toPositiveDim(dim, inputRank + 1);
  if (!isValidDim(dim, inputRank + 1))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  auto unsqzTensorInfo = hlo::unsqueezeTensor(rewriter, op, adaptor.getSelf(),
                                              {dim}, options.dimSizeIndexBits);
  if (failed(unsqzTensorInfo))
    return rewriter.notifyMatchFailure(op,
                                       "failed to create unsqueezed tensor");

  rewriter.replaceOp(op, *unsqzTensorInfo);
  return success();
}

void mlir::torch::torch_to_stablehlo::populateViewLikeOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context, options)
  INSERT_ATENOP_PATTERN(AtenSliceTensorOp);
  INSERT_ATENOP_PATTERN(AtenSqueezeOp);
  INSERT_ATENOP_PATTERN(AtenSqueezeDimOp);
  INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_VIEW_OP_PATTERN(AtenOp)                                         \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenViewOp<AtenOp>>(typeConverter, context, options)
  INSERT_VIEW_OP_PATTERN(AtenViewOp);
  INSERT_VIEW_OP_PATTERN(AtenReshapeOp);
#undef INSERT_VIEW_OP_PATTERN
}
