//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

int64_t Torch::toPositiveDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 ? dim : dim + inputRank;
}

bool Torch::isValidDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 && dim < inputRank;
}

std::optional<int64_t>
Torch::matchLegalConstantIndexIntoListOfSize(Value v, int64_t length) {
  int64_t dim;
  if (!matchPattern(v, m_TorchConstantInt(&dim)))
    return std::nullopt;
  dim = toPositiveDim(dim, length);
  if (!isValidDim(dim, length))
    return std::nullopt;
  return dim;
}

bool Torch::getListConstructElements(Value v, SmallVectorImpl<Value> &elems) {
  auto listConstruct = v.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return false;
  elems = llvm::to_vector<4>(listConstruct.getElements());
  return true;
}

torch_upstream::ScalarType Torch::getScalarTypeForType(Type type) {
  if (type.isa<Float32Type>())
    return torch_upstream::ScalarType::Float;
  if (type.isa<Float64Type>())
    return torch_upstream::ScalarType::Double;
  if (type.isSignedInteger(64))
    return torch_upstream::ScalarType::Long;
  if (type.isSignedInteger(32))
    return torch_upstream::ScalarType::Int;
  if (type.isSignlessInteger(1))
    return torch_upstream::ScalarType::Bool;
  if (type.isBF16())
    return torch_upstream::ScalarType::BFloat16;
  if (type.isF16())
    return torch_upstream::ScalarType::Half;
  if (type.isUnsignedInteger(8))
    return torch_upstream::ScalarType::Byte;
  if (type.isSignedInteger(8))
    return torch_upstream::ScalarType::Char;
  if (type.isa<ComplexType>()) {
    mlir::Type complexElemType = type.cast<ComplexType>().getElementType();
    if (complexElemType.isF32())
      return torch_upstream::ScalarType::ComplexHalf;
    if (complexElemType.isF64())
      return torch_upstream::ScalarType::ComplexFloat;
    if (complexElemType.isF128())
      return torch_upstream::ScalarType::ComplexDouble;
  }
  llvm::report_fatal_error("unhandled type for getScalarTypeForType");
}

Type Torch::getTypeForTorchType(
    MLIRContext *context, Type type,
    mlir::IntegerType::SignednessSemantics signedness) {
  if (type.isa<Torch::IntType>())
    return IntegerType::get(context, 64, signedness);
  if (type.isa<Torch::FloatType>())
    return Float64Type::get(context);
  llvm::report_fatal_error("unhandled type for getTypeForTorchType");
}

FailureOr<Type>
Torch::getTypeForScalarType(MLIRContext *context,
                            torch_upstream::ScalarType dtypeInt,
                            mlir::IntegerType::SignednessSemantics signedness) {
  switch (dtypeInt) {
  case torch_upstream::ScalarType::Float:
    return Float32Type::get(context);
  case torch_upstream::ScalarType::Double:
    return Float64Type::get(context);
  case torch_upstream::ScalarType::Long:
    return IntegerType::get(context, 64, signedness);
  case torch_upstream::ScalarType::Int:
    return IntegerType::get(context, 32, signedness);
  case torch_upstream::ScalarType::Bool:
    return IntegerType::get(context, 1);
  case torch_upstream::ScalarType::BFloat16:
    return mlir::FloatType::getBF16(context);
  case torch_upstream::ScalarType::Half:
    return mlir::FloatType::getF16(context);
  case torch_upstream::ScalarType::Byte:
  case torch_upstream::ScalarType::Char:
    return mlir::IntegerType::get(context, 8, signedness);
  case torch_upstream::ScalarType::ComplexHalf:
    return mlir::ComplexType::get(Float32Type::get(context));
  case torch_upstream::ScalarType::ComplexFloat:
    return mlir::ComplexType::get(Float64Type::get(context));
  case torch_upstream::ScalarType::ComplexDouble:
    return mlir::ComplexType::get(Float128Type::get(context));
  case torch_upstream::ScalarType::Undefined:
    return failure();
  default:
    llvm::report_fatal_error("unhandled type for getTypeForScalarType");
  }
}

FailureOr<Type>
Torch::getTorchTypeForScalarType(MLIRContext *context,
                                 torch_upstream::ScalarType dtypeInt) {
  switch (dtypeInt) {
  case torch_upstream::ScalarType::Double:
    return Torch::FloatType::get(context);
  case torch_upstream::ScalarType::Long:
    return Torch::IntType::get(context);
  case torch_upstream::ScalarType::Undefined:
  default:
    return failure();
  }
}

Type Torch::getDefaultDtypeForTorchScalar(Type type) {
  MLIRContext *context = type.getContext();
  if (type.isa<Torch::FloatType>()) {
    // For now, use float32 which is the initial default dtype returned by
    // `torch.get_default_dtype`.
    return Float32Type::get(context);
  }
  if (type.isa<Torch::IntType>())
    return IntegerType::get(context, 64, IntegerType::Signed);
  if (type.isa<Torch::BoolType>())
    return IntegerType::get(context, 1);
  llvm_unreachable(
      "getDefaultDtypeForTorchScalar called on an unsupported type");
}

Type Torch::getBuiltInTypeForTorchScalar(Type type) {
  MLIRContext *context = type.getContext();
  if (type.isa<Torch::FloatType>())
    return Float64Type::get(context);
  if (type.isa<Torch::IntType>())
    return IntegerType::get(context, 64, IntegerType::Signed);
  if (type.isa<Torch::BoolType>())
    return IntegerType::get(context, 1);
  llvm_unreachable(
      "getBuiltInTypeForTorchScalar called on an unsupported type");
}

Value Torch::getDtypeIntValueForType(PatternRewriter &rewriter, Location loc,
                                     Type dtype) {
  int intType = (int)getScalarTypeForType(dtype);
  return rewriter.create<ConstantIntOp>(loc,
                                        rewriter.getI64IntegerAttr(intType));
}

// Helper to convert a tensor to a specific scalar type.
Value Torch::convertTensorToDtype(PatternRewriter &rewriter, Location loc,
                                  Value input, Type dtype) {
  BaseTensorType origType = input.getType().cast<BaseTensorType>();
  Type newType = origType.getWithSizesAndDtype(origType.getSizes(), dtype);
  // `convertIntVal` contains the corresponding integer for the dtype which is
  // used by the aten.to.dtype op.
  Value convertIntVal = getDtypeIntValueForType(rewriter, loc, dtype);
  Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  Value converted = rewriter.create<AtenToDtypeOp>(
      loc, newType, input, convertIntVal, falseVal, falseVal, noneVal);
  return converted;
}

bool Torch::isBuiltInType(Type type) {
  return isa<BuiltinDialect>(type.getDialect());
}

std::optional<unsigned> Torch::getTensorRank(Value tensor) {
  BaseTensorType tensorType = tensor.getType().cast<BaseTensorType>();
  if (!tensorType.hasSizes())
    return std::nullopt;
  return tensorType.getSizes().size();
}

bool Torch::isViewLikeOp(Operation *op) {
  // AtenContiguousOp might return a view, so this is conservatively
  // correct. We could potentially be more precise and identify the cases
  // that it does not return a view and treat those as having value
  // semantics.
  return isa<AtenBroadcastToOp, AtenContiguousOp, AtenDetachOp, AtenExpandAsOp,
             AtenExpandOp, AtenFlattenUsingIntsOp, AtenPermuteOp, AtenReshapeOp,
             Aten_ReshapeAliasOp, AtenSelectIntOp, AtenSliceTensorOp,
             AtenSqueezeDimOp, AtenSqueezeOp, AtenTOp, AtenToDtypeOp,
             AtenTransposeIntOp, AtenUnsqueezeOp, AtenViewOp,
             TensorStaticInfoCastOp, AtenToDtypeLayoutOp, AtenNumpyTOp,
             AtenNarrowOp, AtenToDeviceOp, AtenMovedimIntOp>(op);
}

Value Torch::getConstantWithGivenDtypeAndValue(PatternRewriter &rewriter,
                                               Location loc, float value,
                                               Type dtype) {
  // Creating constants satisfying backend contract.
  if (dtype.isInteger(64) || dtype.isInteger(32) || dtype.isInteger(8) ||
      dtype.isInteger(1))
    return rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr((int64_t)value));
  if (dtype.isF64() || dtype.isF32() || dtype.isF16() || dtype.isBF16())
    return rewriter.create<ConstantFloatOp>(loc,
                                            rewriter.getF64FloatAttr(value));
  llvm::report_fatal_error(
      "unhandled type for getConstantWithGivenDtypeAndValue");
}

// Return the number of elements of a tensor if the shape is static; otherwise,
// return -1.
int64_t Torch::getNumberOfElements(RankedTensorType inputType) {
  if (!inputType.hasStaticShape())
    return -1;
  SmallVector<int64_t> inputShape =
      makeShapeTorchCompatible(inputType.getShape());
  int64_t numel = 1;
  for (int64_t i = 0; i < inputType.getRank(); i++)
    numel *= inputShape[i];
  return numel;
}

SmallVector<int64_t> Torch::makeShapeLLVMCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kUnknownSize);
    if (shape[i] == kUnknownSize)
      updatedShape[i] = kDynamic;
  }
  return updatedShape;
}

SmallVector<int64_t> Torch::makeShapeTorchCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kDynamic);
    if (shape[i] == kDynamic)
      updatedShape[i] = kUnknownSize;
  }
  return updatedShape;
}
