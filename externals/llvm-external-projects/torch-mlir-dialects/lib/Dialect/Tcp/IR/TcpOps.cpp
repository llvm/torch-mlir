//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"

#define GET_OP_CLASSES
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.cpp.inc"

namespace mlir::tcp {

LogicalResult ClampOp::verify() {
  auto inputType = getIn().getType().cast<RankedTensorType>();

  if (inputType.getElementType().isa<FloatType>()) {
    if (getMinInt() || getMaxInt())
      return emitOpError("failed to verify that int min / max attributes "
                         "must not be set when input is a float tensor");
    if (!getMinFloat() && !getMaxFloat())
      return emitOpError("failed to verify that at least one of min / max "
                         "attributes must be set");
  }

  if (inputType.getElementType().isa<IntegerType>()) {
    if (getMinFloat() || getMaxFloat())
      return emitOpError("failed to verify that float min / max attributes "
                         "must not be set when input is an int tensor");
    if (!getMinInt() && !getMaxInt())
      return emitOpError("failed to verify that at least one of min / max "
                         "attributes must be set");
  }

  return success();
}

LogicalResult BroadcastOp::verify() {
  auto compareIntAttr = [](Attribute v1, Attribute v2) {
    return v1.cast<IntegerAttr>().getInt() < v2.cast<IntegerAttr>().getInt();
  };

  if (!llvm::is_sorted(getAxes(), compareIntAttr))
    return emitOpError(
        "failed to verify that attribute `axes` must be in increasing order");

  if (std::adjacent_find(std::begin(getAxes()), std::end(getAxes())) !=
      std::end(getAxes()))
    return emitOpError(
        "failed to verify that attribute `axes` must not have any duplicates");

  if (getNewDimSizes().size() != getAxes().size())
    return emitOpError("failed to verify that argument `new_dim_sizes` has the "
                       "same size as the attribute `axes`");

  return success();
}

LogicalResult GroupOp::verify() {
  auto &groupBlock = getBody().front();
  if (groupBlock.empty() ||
      !groupBlock.back().mightHaveTrait<OpTrait::IsTerminator>())
    return emitOpError(
        "failed to verify that op region ends with a terminator");

  auto yieldOp = getBody().front().getTerminator();
  if (yieldOp->getNumOperands() != getNumResults())
    return emitOpError("failed to verify that the number of yielded values is "
                       "same as the number of results");

  for (unsigned i = 0; i < getNumResults(); ++i) {
    if (yieldOp->getOperand(i).getType() != getResult(i).getType())
      return emitOpError()
             << "failed to verify that the type of operand #" << i
             << " of terminator matches the corresponding result type";
  }

  return success();
}

LogicalResult IsolatedGroupOp::verify() {
  auto &groupBlock = getBody().front();
  if (groupBlock.empty() ||
      !groupBlock.back().mightHaveTrait<OpTrait::IsTerminator>())
    return emitOpError(
        "failed to verify that op region ends with a terminator");

  auto yieldOp = getBody().front().getTerminator();
  if (yieldOp->getNumOperands() != getNumResults())
    return emitOpError("failed to verify that the number of yielded values is "
                       "same as the number of results");

  for (unsigned i = 0; i < getNumResults(); ++i) {
    if (yieldOp->getOperand(i).getType() != getResult(i).getType())
      return emitOpError()
             << "failed to verify that the type of operand #" << i
             << " of terminator matches the corresponding result type";
  }

  return success();
}

OpFoldResult ConstOp::fold(FoldAdaptor) { return getValueAttr(); }

LogicalResult ConcatOp::verify() {
  auto outputTensorType = getResult().getType().cast<TensorType>();
  int64_t axis = getAxis();
  int64_t concatDimVal = outputTensorType.getShape()[axis];
  // Accumulate concat dim, ignored if any of the inputs are dynamic.
  int64_t concatDimAcc = 0;
  // All tensors must have the same dtype, rank and all non-concat dims must be
  // the same.
  for (auto type : getInputs().getTypes()) {
    auto inputTensorType = type.cast<TensorType>();
    if (outputTensorType.getRank() != inputTensorType.getRank())
      return emitOpError() << "failed to verify tcp.concat operands and "
                              "results rank mismatched";
    for (int64_t dim = 0; dim < inputTensorType.getRank(); ++dim) {
      if (dim == axis) {
        concatDimAcc += inputTensorType.getShape()[dim];
      } else {
        if (inputTensorType.getShape()[dim] != outputTensorType.getShape()[dim])
          emitOpError() << "failed to verify tcp.concat with non concat dim "
                        << dim << ", having different values "
                        << inputTensorType.getShape()[dim] << " "
                        << outputTensorType.getShape()[dim];
      }
    }
  }

  // If concat dim is dynamic, at least one input must be dynamic.
  if (ShapedType::isDynamic(concatDimVal)) {
    if (!llvm::any_of(getInputs().getTypes(), [axis](Type type) {
          return ShapedType::isDynamic(
              type.cast<TensorType>().getShape()[axis]);
        }))
      return emitOpError() << "failed to verify tcp.concat with dynamic concat "
                              "axis and static inputs";
    else
      return success();
  }

  // Static case, concat dim must be the sum of dim[axis] across all inputs.
  if (concatDimAcc != concatDimVal)
    return emitOpError() << "failed to verify tcp.concat with dim " << axis
                         << " != " << concatDimAcc;
  return success();
}

LogicalResult CastOp::verify() {
  auto inputType = getIn().getType().cast<RankedTensorType>();
  auto outputType = getOut().getType().cast<RankedTensorType>();

  if (inputType.getElementType().isa<FloatType>()) {
    if (getInDtype())
      return emitOpError("in_dtype attr should not set when input is FP");
  }

  if (inputType.getElementType().isa<IntegerType>()) {
    if (!getInDtype())
      return emitOpError("in_dtype attr must be set when input is INT");
    if (inputType.getElementType().isInteger(1) &&
        getInDtype().value() != IntegerType::SignednessSemantics::Signless)
      return emitOpError("in_dtype attr must be set to "
                         "SignednessSemantics::Signless when input is i1");
  }

  if (outputType.getElementType().isa<FloatType>()) {
    if (getOutDtype())
      return emitOpError("out_dtype attr should not set when output is FP");
  }

  if (outputType.getElementType().isa<IntegerType>()) {
    if (!getOutDtype())
      return emitOpError("out_dtype attr must be set when output is INT");
    if (outputType.getElementType().isInteger(1) &&
        getOutDtype().value() != IntegerType::SignednessSemantics::Signless)
      return emitOpError("out_dtype attr must be set to "
                         "SignednessSemantics::Signless when output is i1");
  }

  return success();
}

} // namespace mlir::tcp
