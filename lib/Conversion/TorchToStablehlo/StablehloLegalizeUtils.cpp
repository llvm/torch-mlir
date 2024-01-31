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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {
namespace hlo {

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

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for APInt
template <>
std::optional<Value> getConstTensor<APInt>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<APInt> vec,
                                           ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }
  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for float
template <>
std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<float> vec,
                                           ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

template <>
std::optional<Value> getConstTensor<double>(PatternRewriter &rewriter,
                                            Operation *op, ArrayRef<double> vec,
                                            ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF64Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
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
                  TensorType outType) {
  TensorType in_type = input.getType().cast<TensorType>();

  if (in_type.getElementType() != outType.getElementType()) {
    TensorType promotedType =
        in_type.cloneWith(in_type.getShape(), outType.getElementType());
    return rewriter.create<stablehlo::ConvertOp>(loc, promotedType, input);
  }
  return input;
}

Value promoteAndBroadcast(ConversionPatternRewriter &rewriter, Value input,
                          TensorType outType) {
  // Two tensors are “broadcastable” if the following rules hold:
  //   - Each tensor has at least one dimension.
  //   - When iterating over the dimension sizes, starting at the trailing
  //   dimension, the dimension sizes must either be equal, one of them is 1, or
  //   one of them does not exist.
  Operation *op = input.getDefiningOp();
  TensorType in_type = input.getType().dyn_cast<TensorType>();

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
          << inDim << ")"
          << "must match the size of tensor b (" << outDim << ")"
          << "at non-singleton dimension " << inPos;
    }
  }
  std::reverse(bcastDims.begin(), bcastDims.end());
  if (!do_bcast) {
    return input;
  }
  auto bcast_attr = rewriter.getDenseI64ArrayAttr(bcastDims);
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
  auto valueTy = value.getType().dyn_cast<RankedTensorType>();
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
  auto valueTy = value.getType().dyn_cast<RankedTensorType>();
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

FailureOr<Value> unsqueezeTensor(PatternRewriter &rewriter, Operation *op,
                                 Value tensor, ArrayRef<int64_t> inputUnsqzDims,
                                 size_t dimSizeIndexBits) {
  // Returns a new tensor with dims of size 1 inserted at the specified
  // position.
  //
  // The position indices (must be high to low dimension number of the returned
  // tensor) are specified with unsqzDims. Indices must be in-order, and in
  // range of tensor rank. Thus, unsqueeze a rank 1 tensor with {0, 2}, {0, 1,
  // 3}, {0, 1, 2} are all valid dimension sets, but {0, 3}, {2} are not.
  auto dimSizesInfo =
      getDimSizesOfTensor(rewriter, op, tensor, dimSizeIndexBits);
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
  auto rankTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto oldShape = rankTy.getShape();
  Type intType = rewriter.getIntegerType(dimSizeIndexBits);
  auto one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));

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
