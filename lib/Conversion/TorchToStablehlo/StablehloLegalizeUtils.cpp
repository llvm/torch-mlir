//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {
namespace hlo {

// Create chlo::ConstantLikeOp
template <typename T>
Value getConstantLike(OpBuilder &rewriter, Location loc, T constant,
                      Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  auto getAttr = [&]() -> Attribute {
    if (isa<mlir::IntegerType>(ty))
      return rewriter.getIntegerAttr(ty, constant);
    if (isa<mlir::FloatType>(ty))
      return rewriter.getFloatAttr(ty, constant);
    if (auto complexTy = dyn_cast<mlir::ComplexType>(ty))
      return mlir::complex::NumberAttr::get(complexTy, constant, 0);
    llvm_unreachable("unhandled element type");
  };
  return rewriter.create<mlir::chlo::ConstantLikeOp>(
      loc, cast<TypedAttr>(getAttr()), val);
}

// Template instantiation
template Value getConstantLike<int64_t>(OpBuilder &rewriter, Location loc,
                                        int64_t constant, Value val);

template Value getConstantLike<double>(OpBuilder &rewriter, Location loc,
                                       double constant, Value val);

// Create a 32-bit float constant operator from a float
Value getStablehloConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                       float val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 64-bit float constant operator from a double
Value getStablehloConstTensorSingleF64(PatternRewriter &rewriter, Operation *op,
                                       double val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF64Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  RankedTensorType const_type;
  if constexpr (std::is_same_v<T, APInt>) {
    const_type = RankedTensorType::get(
        shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  } else if constexpr (std::is_same_v<T, float>) {
    const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  } else if constexpr (std::is_same_v<T, double>) {
    const_type = RankedTensorType::get(shape, rewriter.getF64Type());
  } else {
    const_type =
        RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  }
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
template std::optional<Value> getConstTensor<APInt>(PatternRewriter &rewriter,
                                                    Operation *op,
                                                    ArrayRef<APInt> vec,
                                                    ArrayRef<int64_t> shape);

template std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                                    Operation *op,
                                                    ArrayRef<float> vec,
                                                    ArrayRef<int64_t> shape);

template std::optional<Value> getConstTensor<double>(PatternRewriter &rewriter,
                                                     Operation *op,
                                                     ArrayRef<double> vec,
                                                     ArrayRef<int64_t> shape);

template std::optional<Value> getConstTensor<int32_t>(PatternRewriter &,
                                                      Operation *,
                                                      ArrayRef<int32_t> vec,
                                                      ArrayRef<int64_t> shape);

template std::optional<Value> getConstTensor<int64_t>(PatternRewriter &,
                                                      Operation *,
                                                      ArrayRef<int64_t> vec,
                                                      ArrayRef<int64_t> shape);

template <typename T>
static bool isInValidRange(bool isFloat, const double &doubleValue, bool isInt,
                           const int64_t &intValue) {
  if (isFloat) {
    // Do a round-trip check here instead of numeric limits due to
    // compiler warnings around double <-> int conversion.
    return (doubleValue == static_cast<double>(static_cast<T>(doubleValue)));
  } else {
    assert(isInt);
    return (intValue >= std::numeric_limits<T>::min()) &&
           (intValue <= std::numeric_limits<T>::max());
  }
  return true;
}

template <typename T>
Value getSplatConstTensor(ConversionPatternRewriter &rewriter, Operation *op,
                          T val, Type dtype, llvm::ArrayRef<int64_t> dshape) {
  auto const_type = RankedTensorType::get(dshape, dtype);
  auto const_attr = SplatElementsAttr::get(const_type, val);
  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

Value scalarToStablehloTensor(ConversionPatternRewriter &rewriter,
                              Operation *op, Value scalarValue, Type dtype) {
  auto tensor = rewriter.create<tensor::FromElementsOp>(
      op->getLoc(), ArrayRef<Value>{scalarValue});
  auto dtype_tensor =
      rewriter.create<stablehlo::ConvertOp>(op->getLoc(), tensor, dtype);
  return rewriter.create<stablehlo::ReshapeOp>(
      op->getLoc(), RankedTensorType::get(mlir::ArrayRef<int64_t>{}, dtype),
      dtype_tensor);
}

Value promoteType(PatternRewriter &rewriter, Location loc, Value input,
                  Type outElementType) {
  TensorType inType = cast<TensorType>(input.getType());
  if (inType.getElementType() != outElementType) {
    return rewriter.create<stablehlo::ConvertOp>(loc, input, outElementType);
  }
  return input;
}

Value promoteAndBroadcast(ConversionPatternRewriter &rewriter, Value input,
                          TensorType outType,
                          std::optional<Value> bcastSizeTensor) {
  // Two tensors are “broadcastable” if the following rules hold:
  //   - Each tensor has at least one dimension.
  //   - When iterating over the dimension sizes, starting at the trailing
  //   dimension, the dimension sizes must either be equal, one of them is 1, or
  //   one of them does not exist.
  // If one provide bcastSizeTensor, we emit stablehlo::DynamicBroadcastInDimOp
  // instead of stablehlo::BroadcastInDimOp to support dynamic shape.
  Operation *op = input.getDefiningOp();
  TensorType in_type = dyn_cast<TensorType>(input.getType());

  if (in_type.getElementType() != outType.getElementType()) {
    TensorType promoted_type =
        in_type.cloneWith(in_type.getShape(), outType.getElementType());
    input = rewriter.create<stablehlo::ConvertOp>(op->getLoc(), promoted_type,
                                                  input);
  }

  ArrayRef<int64_t> inShape = in_type.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();

  bool do_bcast = (inShape.size() != outShape.size());
  SmallVector<int64_t> bcastDims;
  for (size_t i = 0; i < inShape.size(); ++i) {
    // iterating over the dimension sizes, starting at the trailing dimension
    size_t outPos = outShape.size() - 1 - i;
    size_t inPos = inShape.size() - 1 - i;
    int64_t outDim = outShape[outPos];
    int64_t inDim = inShape[inPos];
    if (inDim == outDim) {
      bcastDims.push_back(outPos);
    } else if (inDim != outDim && inDim == 1) {
      bcastDims.push_back(outPos);
      do_bcast = true;
    } else {
      op->emitError("The size of tensor a (")
          << inDim << ")" << "must match the size of tensor b (" << outDim
          << ")" << "at non-singleton dimension " << inPos;
    }
  }
  std::reverse(bcastDims.begin(), bcastDims.end());
  if (!do_bcast) {
    return input;
  }
  auto bcast_attr = rewriter.getDenseI64ArrayAttr(bcastDims);
  if (bcastSizeTensor.has_value()) {
    auto bcast_op = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
        op->getLoc(), outType, input, bcastSizeTensor.value(), bcast_attr);
    return bcast_op.getResult();
  }
  auto bcast_op = rewriter.create<stablehlo::BroadcastInDimOp>(
      op->getLoc(), outType, input, bcast_attr);
  return bcast_op.getResult();
}

SmallVector<int64_t> toPositiveDims(ArrayRef<int64_t> dims, int64_t rank) {
  SmallVector<int64_t> posDims;
  posDims.reserve(rank);
  std::transform(
      dims.begin(), dims.end(), std::back_inserter(posDims),
      [rank](int64_t d) -> int64_t { return toPositiveDim(d, rank); });
  return posDims;
}

FailureOr<SmallVector<Value, 4>> getDimSizesOfTensor(PatternRewriter &rewriter,
                                                     Operation *op, Value value,
                                                     ArrayRef<int64_t> inpDims,
                                                     size_t dimSizeIndexBits) {
  auto valueTy = dyn_cast<RankedTensorType>(value.getType());
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimSizesOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  auto dims = toPositiveDims(inpDims, rank);
  SmallVector<Value, 4> dimSizes;
  dimSizes.reserve(dims.size());

  auto loc = op->getLoc();
  for (auto d : dims) {
    dimSizes.emplace_back(rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIntegerType(dimSizeIndexBits),
        rewriter.create<tensor::DimOp>(loc, value, d)));
  }
  return dimSizes;
}

FailureOr<SmallVector<Value, 4>> getDimSizesOfTensor(PatternRewriter &rewriter,
                                                     Operation *op, Value value,
                                                     size_t dimSizeIndexBits) {
  auto valueTy = dyn_cast<RankedTensorType>(value.getType());
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimSizesOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  // Get int vector [0, 1, ..., rank-1]
  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  return getDimSizesOfTensor(rewriter, op, value, dims, dimSizeIndexBits);
}

// Get the dimension sizes of the input tensor, given the dimension axes
FailureOr<SmallVector<Value, 4>>
getDimIndexOfTensor(PatternRewriter &rewriter, Operation *op, Value value,
                    ArrayRef<int64_t> inpDims) {
  auto valueTy = dyn_cast<RankedTensorType>(value.getType());
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimIndexOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  auto dims = toPositiveDims(inpDims, rank);
  SmallVector<Value, 4> dimSizes;
  dimSizes.reserve(dims.size());

  auto loc = op->getLoc();
  for (auto d : dims) {
    dimSizes.emplace_back(rewriter.create<tensor::DimOp>(loc, value, d));
  }
  return dimSizes;
}

// Get the dimension sizes of the input tensor
FailureOr<SmallVector<Value, 4>>
getDimIndexOfTensor(PatternRewriter &rewriter, Operation *op, Value value) {
  auto valueTy = dyn_cast<RankedTensorType>(value.getType());
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimIndexOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  // Get int vector [0, 1, ..., rank-1]
  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  return getDimIndexOfTensor(rewriter, op, value, dims);
}

FailureOr<Value> getBroadcastResultShape(PatternRewriter &rewriter,
                                         Operation *op, ArrayRef<Value> tensors,
                                         size_t dimSizeIndexBits) {
  SmallVector<ArrayRef<int64_t>> tensorSizes;

  int maxRank = 0;
  for (auto tensor : tensors) {
    auto tensorType = cast<RankedTensorType>(tensor.getType());
    auto tensorRank = tensorType.getRank();

    tensorSizes.emplace_back(tensorType.getShape());
    maxRank = std::max(maxRank, static_cast<int>(tensorRank));
  }

  SmallVector<Value> bcastSizeTensors;
  for (int outDim = 0; outDim < maxRank; ++outDim) { // loop dimensions.
    int dynamicDimCnt = 0;
    int staticDimCnt = 0;
    int64_t staticDimSize;
    Value dimSizeTensor = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(dimSizeIndexBits), 1));

    for (size_t i = 0; i < tensorSizes.size(); ++i) { // loop tensors.
      int inDim = tensorSizes[i].size() - 1 - outDim;
      if (inDim < 0)
        continue;

      // dim size: 1
      if (tensorSizes[i][inDim] == 1)
        continue;
      // dim size: dynamic
      if (tensorSizes[i][inDim] == ShapedType::kDynamic ||
          tensorSizes[i][inDim] == kUnknownSize) {
        dynamicDimCnt++;
        auto dimSizeTensorInfo = hlo::getDimSizesOfTensor(
            rewriter, op, tensors[i], {inDim}, dimSizeIndexBits);
        if (failed(dimSizeTensorInfo)) {
          return failure();
        }
        dimSizeTensor = (*dimSizeTensorInfo)[0];
        continue;
      }
      // dim size: static
      // we already found dynamic dim size, fail.
      if (dynamicDimCnt > 0) {
        return failure();
      }
      // we already found static dim size not equal with this, fail.
      if (staticDimCnt > 0 && staticDimSize != tensorSizes[i][inDim]) {
        return failure();
      }

      staticDimCnt++;
      staticDimSize = tensorSizes[i][inDim];
      auto dimSizeTensorInfo = hlo::getDimSizesOfTensor(
          rewriter, op, tensors[i], {inDim}, dimSizeIndexBits);
      if (failed(dimSizeTensorInfo)) {
        return failure();
      }
      dimSizeTensor = (*dimSizeTensorInfo)[0];
    }

    // TODO: Relax this check, by assuming all dynamic shape is same.
    // if (dynamicDimCnt > 1) {
    //   return failure();
    // }

    bcastSizeTensors.push_back(dimSizeTensor);
  }
  std::reverse(bcastSizeTensors.begin(), bcastSizeTensors.end());
  return rewriter.create<tensor::FromElementsOp>(op->getLoc(), bcastSizeTensors)
      .getResult();
}

FailureOr<Value> unsqueezeTensor(PatternRewriter &rewriter, Operation *op,
                                 Value tensor,
                                 ArrayRef<int64_t> inputUnsqzDims) {
  // Returns a new tensor with dims of size 1 inserted at the specified
  // position.
  //
  // The position indices (must be high to low dimension number of the returned
  // tensor) are specified with unsqzDims. Indices must be in-order, and in
  // range of tensor rank. Thus, unsqueeze a rank 1 tensor with {0, 2}, {0, 1,
  // 3}, {0, 1, 2} are all valid dimension sets, but {0, 3}, {2} are not.
  auto dimSizesInfo = getDimIndexOfTensor(rewriter, op, tensor);
  if (failed(dimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");

  auto dimSizes = *dimSizesInfo;
  int64_t rank = dimSizes.size();
  int64_t newRank = rank + inputUnsqzDims.size();
  auto unsqzDims = toPositiveDims(inputUnsqzDims, newRank);
  for (int64_t k = 0, sz = unsqzDims.size(); k < sz; ++k)
    if (k > 1 && unsqzDims[k] <= unsqzDims[k - 1])
      return rewriter.notifyMatchFailure(
          op, "unsqueeze dimensions must be specified in order");

  auto loc = op->getLoc();
  auto rankTy = dyn_cast<RankedTensorType>(tensor.getType());
  auto oldShape = rankTy.getShape();
  auto one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  std::vector<Value> newDimSizes;
  std::vector<int64_t> newShape;
  newDimSizes.reserve(newRank);
  newShape.reserve(newRank);
  for (int64_t k = 0, i = 0, j = 0; k < newRank; ++k) {
    if (j < static_cast<int64_t>(unsqzDims.size()) && unsqzDims[j] == k) {
      newDimSizes.push_back(one);
      newShape.push_back(1);
      j++;
    } else {
      newDimSizes.push_back(dimSizes[i]);
      newShape.push_back(oldShape[i]);
      i++;
    }
  }

  auto outTy = RankedTensorType::get(newShape, rankTy.getElementType());
  auto shape = rewriter.create<tensor::FromElementsOp>(loc, newDimSizes);
  return rewriter.create<stablehlo::DynamicReshapeOp>(loc, outTy, tensor, shape)
      .getResult();
}

FailureOr<Value> collapseTensor(PatternRewriter &rewriter, Operation *op,
                                Value tensor, int64_t collapseStartDim,
                                int64_t collapseEndDim) {

  auto dimSizesInfo = getDimIndexOfTensor(rewriter, op, tensor);
  if (failed(dimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");

  auto dimSizes = *dimSizesInfo;
  int64_t rank = dimSizes.size();

  collapseStartDim = toPositiveDim(collapseStartDim, rank);
  collapseEndDim = toPositiveDim(collapseEndDim, rank);

  int64_t newRank = rank - (collapseEndDim - collapseStartDim + 1);

  auto loc = op->getLoc();
  auto rankTy = dyn_cast<RankedTensorType>(tensor.getType());
  auto oldShape = rankTy.getShape();

  std::vector<Value> newDimSizes;
  std::vector<int64_t> newShape;
  newDimSizes.reserve(newRank);
  newShape.reserve(newRank);

  Value collapseDimSize = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  int64_t collapseShape = 1;

  for (int64_t k = collapseStartDim; k <= collapseEndDim; ++k) {
    if (k < 0 || k >= rank) {
      return rewriter.notifyMatchFailure(
          op, "collapse dimensions must be within the rank of the tensor");
    }
    if (collapseShape == ShapedType::kDynamic ||
        oldShape[k] == ShapedType::kDynamic) {
      collapseShape = ShapedType::kDynamic;
    } else {
      collapseShape *= oldShape[k];
    }
    collapseDimSize =
        rewriter.create<arith::MulIOp>(loc, collapseDimSize, dimSizes[k]);
  }

  for (int64_t k = 0; k < collapseStartDim; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(oldShape[k]);
  }
  newDimSizes.push_back(collapseDimSize);
  newShape.push_back(collapseShape);
  for (int64_t k = collapseEndDim + 1; k < rank; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(oldShape[k]);
  }

  auto outTy = RankedTensorType::get(newShape, rankTy.getElementType());
  auto shape = rewriter.create<tensor::FromElementsOp>(loc, newDimSizes);
  return rewriter.create<stablehlo::DynamicReshapeOp>(loc, outTy, tensor, shape)
      .getResult();
}

// TODO: support splitDim & outerLength to be Value
FailureOr<Value> splitTensor(PatternRewriter &rewriter, Operation *op,
                             Value tensor, int64_t splitDim,
                             int64_t outerLength) {
  auto dimSizesInfo = getDimIndexOfTensor(rewriter, op, tensor);
  if (failed(dimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");

  auto dimSizes = *dimSizesInfo;
  int64_t rank = dimSizes.size();
  splitDim = toPositiveDim(splitDim, rank);

  auto loc = op->getLoc();
  auto rankTy = dyn_cast<RankedTensorType>(tensor.getType());
  auto oldShape = rankTy.getShape();

  if (splitDim < 0 || splitDim >= rank) {
    return rewriter.notifyMatchFailure(
        op, "split dimensions must be within the rank of the tensor");
  }

  int64_t newRank = rank + 1;
  auto outerLengthValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getIndexType(), outerLength));

  auto innerLengthValue = rewriter.create<arith::DivSIOp>(
      loc, dimSizes[splitDim], outerLengthValue);

  int64_t originShape = oldShape[splitDim];
  int64_t outerShape = outerLength;
  int64_t innerShape = originShape == ShapedType::kDynamic
                           ? ShapedType::kDynamic
                           : originShape / outerLength;

  std::vector<Value> newDimSizes;
  std::vector<int64_t> newShape;

  newDimSizes.reserve(newRank);
  newShape.reserve(newRank);

  for (int64_t k = 0; k < splitDim; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(oldShape[k]);
  }
  newDimSizes.push_back(outerLengthValue);
  newShape.push_back(outerShape);
  newDimSizes.push_back(innerLengthValue);
  newShape.push_back(innerShape);

  for (int64_t k = splitDim + 1; k < rank; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(oldShape[k]);
  }

  auto outTy = RankedTensorType::get(newShape, rankTy.getElementType());
  auto shape = rewriter.create<tensor::FromElementsOp>(loc, newDimSizes);
  return rewriter.create<stablehlo::DynamicReshapeOp>(loc, outTy, tensor, shape)
      .getResult();
}

Value getConstantOfShape(PatternRewriter &rewriter, Location loc,
                         const APFloat &constant, Value shape,
                         TensorType outType) {
  auto constAttr = rewriter.getFloatAttr(outType.getElementType(), constant);
  auto constTensor = rewriter.create<stablehlo::ConstantOp>(loc, constAttr);
  return rewriter
      .create<stablehlo::DynamicBroadcastInDimOp>(
          loc, outType, constTensor, shape, rewriter.getDenseI64ArrayAttr({}))
      .getResult();
}
} // namespace hlo
} // namespace mlir
