//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "./MhloLegalizeUtils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::shape;

namespace mlir {
namespace mhlo {

// Create a 32-bit float constant operator from a float
Value getMhloConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                  float val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 64-bit float constant operator from a double
Value getMhloConstTensorSingleF64(PatternRewriter &rewriter, Operation *op,
                                  double val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF64Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
llvm::Optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                     ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for APInt
template <>
llvm::Optional<Value> getConstTensor<APInt>(PatternRewriter &rewriter,
                                            Operation *op, ArrayRef<APInt> vec,
                                            ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }
  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for float
template <>
llvm::Optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                            Operation *op, ArrayRef<float> vec,
                                            ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

template <>
llvm::Optional<Value>
getConstTensor<double>(PatternRewriter &rewriter, Operation *op,
                       ArrayRef<double> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF64Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
template llvm::Optional<Value> getConstTensor<int32_t>(PatternRewriter &,
                                                       Operation *,
                                                       ArrayRef<int32_t> vec,
                                                       ArrayRef<int64_t> shape);

template llvm::Optional<Value> getConstTensor<int64_t>(PatternRewriter &,
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
  auto const_op =
      rewriter.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

LogicalResult torchScalarToMhloTensor(ConversionPatternRewriter &rewriter,
                                      Operation *op, Value torchScalarValue,
                                      Value &mhloTensor, Type dtype,
                                      llvm::ArrayRef<int64_t> dshape,
                                      bool doBroadcast) {
  // Retrieve a const float or int value but create the out Tensor with dtype.
  double doubleValue;
  auto isFloat =
      matchPattern(torchScalarValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(torchScalarValue, m_TorchConstantInt(&intValue));

  if (!isFloat && !isInt)
    return op->emitError("Unable to extract the scalar constant");

  if (dtype.isa<mlir::FloatType>()) {
    if (doBroadcast) {
      mhloTensor = getSplatConstTensor<float>(
          rewriter, op, (isFloat ? doubleValue : intValue), dtype, dshape);
    } else {
      mhloTensor = mhlo::getConstTensor<float>(
                       rewriter, op, (isFloat ? doubleValue : intValue), dshape)
                       .getValue();
    }
  } else if (auto intType = dtype.dyn_cast<mlir::IntegerType>()) {
    auto w = intType.getWidth();
    if (w != 32 && w != 64)
      return op->emitError("Unsupported integer type") << intType;

    if (w == 32) {
      if (!isInValidRange<int32_t>(isFloat, doubleValue, isInt, intValue)) {
        return op->emitError("Supplied value of scalar constant exceeds limits "
                             "of destination type");
      }
      int32_t d = isFloat ? static_cast<int32_t>(doubleValue)
                          : static_cast<int32_t>(intValue);
      if (doBroadcast) {
        mhloTensor =
            getSplatConstTensor<int32_t>(rewriter, op, d, dtype, dshape);
      } else {
        mhloTensor =
            mhlo::getConstTensor<int32_t>(rewriter, op, {d}, dshape).getValue();
      }
    } else if (w == 64) {
      if (!isInValidRange<int64_t>(isFloat, doubleValue, isInt, intValue)) {
        return op->emitError("Supplied value of scalar constant exceeds limits "
                             "of destination type");
      }
      int64_t d = (isFloat ? static_cast<int64_t>(doubleValue) : intValue);
      if (doBroadcast) {
        mhloTensor =
            getSplatConstTensor<int64_t>(rewriter, op, d, dtype, dshape);
      } else {
        mhloTensor =
            mhlo::getConstTensor<int64_t>(rewriter, op, {d}, dshape).getValue();
      }
    }
  } else
    return op->emitError("Usupported element type");

  return success();
}

LogicalResult torchAlphaToMhloTensor(ConversionPatternRewriter &rewriter,
                                     Operation *op, Value alphaScalar,
                                     Value &alphaTensor, Type dtype,
                                     llvm::ArrayRef<int64_t> dshape,
                                     bool checkForUnity) {
  if (succeeded(torchScalarToMhloTensor(rewriter, op, alphaScalar, alphaTensor,
                                        dtype, dshape)))
    return success();

  // `alpha` has not been specified.
  int64_t alphaValue;
  if (!matchPattern(alphaScalar, m_TorchConstantInt(&alphaValue)))
    return op->emitError("Currently only scalar constants are supported for "
                         "alpha in MHLO operation");
  // When no alpha has been specified, this must be 1.
  if (checkForUnity && alphaValue != 1)
    return op->emitError("Unsupported integer value for alpha");

  alphaTensor =
      mlir::mhlo::getMhloConstTensorSingleF32(rewriter, op, alphaValue);

  return success();
}

// Return a tensor filled with the given value of the given elementType, of
// which shape equals the given tensor. Dynamic shape is supported, too.
// This function do not check the compatiblity of storage type T and
// elementType!
template <typename T>
Value getConstTensorLike(PatternRewriter &rewriter, Operation *op, T val,
                         Value tensor, Type elementType) {
  Location loc = op->getLoc();
  // Generate zero rank const tensor.
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  auto constType = RankedTensorType::get({}, elementType);
  auto constAttr = DenseElementsAttr::get(constType, /*ArrayRef<T>*/ {val});
  auto constOp = rewriter.create<mhlo::ConstantOp>(loc, constType, constAttr);

  // Broadcast it
  auto outType = tensorType.cloneWith(tensorType.getShape(), elementType);
  if (tensorType.hasStaticShape()) {
    return broadcastUnaryOperandStatic(rewriter, op, constOp, outType);
  } else {
    auto tensorShape = rewriter.create<shape::ShapeOfOp>(loc, tensor);
    return rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, outType, constOp, tensorShape, rewriter.getI64TensorAttr({}));
  }
}
// template initialization
template Value getConstTensorLike<float>(PatternRewriter &, Operation *, float,
                                         Value, Type);
template Value getConstTensorLike<int32_t>(PatternRewriter &, Operation *,
                                           int32_t, Value, Type);

// Broadcast the input tensor of 0 rank to a tensor with the same shape as the
// given tensor. Supporting dynamic shape.
Value broadcastZeroRankTensorToTensorLike(PatternRewriter &rewriter,
                                          Operation *op, Value input,
                                          Value tensor, Type elementType) {
  Location loc = op->getLoc();
  auto inputType = input.getType().cast<RankedTensorType>();
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  if (inputType.getRank() != 0) {
    op->emitError("Only zero ranked tensor is supported");
  }
  auto outType = tensorType.cloneWith(tensorType.getShape(), elementType);
  input = promoteType(rewriter, input, outType);
  if (tensorType.hasStaticShape()) {
    return broadcastUnaryOperandStatic(rewriter, op, input, outType);
  } else {
    auto tensorShape = rewriter.create<shape::ShapeOfOp>(loc, tensor);
    return rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, outType, input, tensorShape, rewriter.getI64TensorAttr({}));
  }
}

Value promoteAndBroadcast(ConversionPatternRewriter &rewriter, Value input,
                          TensorType outType) {
  Operation *op = input.getDefiningOp();

  input = promoteType(rewriter, input, outType);
  return broadcastUnaryOperandStatic(rewriter, op, input, outType);
}

Value promoteType(PatternRewriter &rewriter, Value input, TensorType outType) {
  Operation *op = input.getDefiningOp();
  auto inputType = input.getType().dyn_cast<mlir::TensorType>();

  if (inputType.getElementType() != outType.getElementType()) {
    TensorType promotedType =
        inputType.cloneWith(inputType.getShape(), outType.getElementType());
    return rewriter.create<mhlo::ConvertOp>(op->getLoc(), promotedType, input);
  }
  return input;
}

Value broadcastUnaryOperandStatic(PatternRewriter &rewriter, Operation *op,
                                  Value input, TensorType outType) {

  auto inputType = input.getType().dyn_cast<TensorType>();
  if (!inputType) {
    op->emitError("Only ranked tensor type input is supported!");
  }
  if (!inputType.hasStaticShape() || !outType.hasStaticShape()) {
    op->emitError("None static shape deteced, please use "
                  "broadcastUnaryOperandDynamic instead");
  }

  // Two tensors are “broadcastable” if the following rules hold:
  //   - Each tensor has at least one dimension.
  //   - When iterating over the dimension sizes, starting at the trailing
  //   dimension, the dimension sizes must either be equal, one of them is 1, or
  //   one of them does not exist.

  ArrayRef<int64_t> inShape = inputType.getShape();
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
  DenseIntElementsAttr bcast_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<long int>(bcastDims.size())},
                            rewriter.getI64Type()),
      bcastDims);
  auto bcast_op = rewriter.create<mhlo::BroadcastInDimOp>(op->getLoc(), outType,
                                                          input, bcast_attr);
  return bcast_op.getResult();
}

void broadcastBinaryOperandsStatic(PatternRewriter &rewriter, Operation *op,
                                   Value lhs, Value rhs,
                                   RankedTensorType outType, Value &bcastLhs,
                                   Value &bcastRhs) {
  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) {
    op->emitError("Only ranked tensor type input is supported!");
  }
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
    op->emitError("None static shape detected, please use "
                  "broadcastBinaryOperandsDynamic instead.");
  }

  bcastLhs = broadcastUnaryOperandStatic(rewriter, op, lhs, outType);
  bcastRhs = broadcastUnaryOperandStatic(rewriter, op, rhs, outType);
}

void broadcastBinaryOperandsDynamic(PatternRewriter &rewriter, Operation *op,
                                    Value lhs, Value rhs,
                                    RankedTensorType outType, Value &bcastLhs,
                                    Value &bcastRhs, Value &bcastShape) {
  Location loc = op->getLoc();

  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) {
    op->emitError("Only ranked tensor type supported!");
  }
  if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
    op->emitError("No dynamic shape involved, please use ...");
  }

  auto lhsRank = lhsType.getRank();
  auto rhsRank = rhsType.getRank();
  auto resultRank = rhsRank > lhsRank ? rhsRank : lhsRank;
  auto lhsShape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
  auto rhsShape = rewriter.create<shape::ShapeOfOp>(loc, rhs);

  auto canBroadcast =
      rewriter.create<shape::CstrBroadcastableOp>(loc, lhsShape, rhsShape);
  auto bcastShapeType = RankedTensorType::get(
      {static_cast<int64_t>(resultRank)}, rewriter.getIndexType());

  auto assumingOp = rewriter.create<shape::AssumingOp>(
      loc, TypeRange{outType, outType, bcastShapeType}, canBroadcast);
  Block &block = assumingOp.getDoRegion().emplaceBlock();
  {
    mlir::IRRewriter::InsertPoint prevIP = rewriter.saveInsertionPoint();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    Value newShape = rewriter.create<shape::BroadcastOp>(
        loc, bcastShapeType, lhsShape, rhsShape, nullptr);

    auto lhsDimensionNumbers = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - lhsRank, resultRank));
    auto rhsDimensionNumbers = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - rhsRank, resultRank));

    Value newLhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, outType, lhs, newShape,
        rewriter.getI64TensorAttr(lhsDimensionNumbers));
    Value newRhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, outType, rhs, newShape,
        rewriter.getI64TensorAttr(rhsDimensionNumbers));

    rewriter.create<shape::AssumingYieldOp>(
        loc, ValueRange{newLhs, newRhs, newShape});

    rewriter.restoreInsertionPoint(prevIP);
  }

  bcastLhs = assumingOp.getResult(0);
  bcastRhs = assumingOp.getResult(1);
  bcastShape = assumingOp.getResult(2);
}

} // namespace mhlo
} // namespace mlir
