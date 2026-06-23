//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHMLIR_DIALECT_TORCH_UTILS_H
#define TORCHMLIR_DIALECT_TORCH_UTILS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

namespace mlir {
namespace torch {
namespace Torch {

inline int64_t toPositiveDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 ? dim : dim + inputRank;
}

inline bool isValidDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 && dim < inputRank;
}

Value toIntListConstruct(PatternRewriter &rewriter, Location loc,
                         ArrayRef<int64_t> cstInput);

inline bool getListConstructElements(Value v, SmallVectorImpl<Value> &elems) {
  auto listConstruct = v.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return false;
  llvm::append_range(elems, listConstruct.getElements());
  return true;
}

/// Returns the index indicated by `v` for a list of given `length`.
/// If the index is negative, it is adjusted to `length` + `v`.
/// `None` is returned the index is not an integer in the range [0,`length).
inline std::optional<int64_t>
matchLegalConstantIndexIntoListOfSize(Value v, int64_t length) {
  int64_t dim;
  if (!matchPattern(v, m_TorchConstantInt(&dim)))
    return std::nullopt;
  dim = toPositiveDim(dim, length);
  if (!isValidDim(dim, length))
    return std::nullopt;
  return dim;
}

inline torch_upstream::ScalarType getScalarTypeForType(Type type) {
  if (isa<Float32Type>(type))
    return torch_upstream::ScalarType::Float;
  if (isa<Float64Type>(type))
    return torch_upstream::ScalarType::Double;
  if (type.isSignedInteger(64))
    return torch_upstream::ScalarType::Long;
  if (type.isSignedInteger(32))
    return torch_upstream::ScalarType::Int;
  if (type.isSignedInteger(16))
    return torch_upstream::ScalarType::Short;
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
  if (isa<QUInt8Type>(type))
    return torch_upstream::ScalarType::QUInt8;
  if (isa<QInt8Type>(type))
    return torch_upstream::ScalarType::QInt8;
  if (isa<QInt16Type>(type))
    return torch_upstream::ScalarType::QInt16;
  if (isa<QInt32Type>(type))
    return torch_upstream::ScalarType::QInt32;
  if (isa<ComplexType>(type)) {
    mlir::Type complexElemType = cast<ComplexType>(type).getElementType();
    if (complexElemType.isF16())
      return torch_upstream::ScalarType::ComplexHalf;
    if (complexElemType.isF32())
      return torch_upstream::ScalarType::ComplexFloat;
    if (complexElemType.isF64())
      return torch_upstream::ScalarType::ComplexDouble;
  }
  if (isa<Float8E5M2Type>(type))
    return torch_upstream::ScalarType::Float8_e5m2;
  if (isa<Float8E4M3FNType>(type))
    return torch_upstream::ScalarType::Float8_e4m3fn;
  if (isa<Float8E5M2FNUZType>(type))
    return torch_upstream::ScalarType::Float8_e5m2fnuz;
  if (isa<Float8E4M3FNUZType>(type))
    return torch_upstream::ScalarType::Float8_e4m3fnuz;
  if (isa<Float8E8M0FNUType>(type))
    return torch_upstream::ScalarType::Float8_e8m0fnu;
  if (isa<Float4E2M1FNType>(type))
    return torch_upstream::ScalarType::Float4_e2m1fn_x2;
  std::string errorMsg = "Unhandled type in getScalarTypeForType: ";
  llvm::raw_string_ostream os(errorMsg);
  type.print(os);
  // os << "\nType ID: " << type.getTypeID();
  os << "\nType properties:";
  os << "\n  Is integer: " << (type.isInteger() ? "yes" : "no");
  os << "\n  Is float: "
     << (type.isIntOrFloat() && !type.isInteger() ? "yes" : "no");
  os << "\n  Is index: " << (type.isIndex() ? "yes" : "no");
  os << "\n  Bit width: "
     << (type.isIntOrFloat() ? std::to_string(type.getIntOrFloatBitWidth())
                             : "N/A");
  os << "\n  Is signless: " << (type.isSignlessInteger() ? "yes" : "no");
  os << "\n  Is signed: " << (type.isSignedInteger() ? "yes" : "no");
  // special error message for unsigned integer
  if (type.isUnsignedInteger()) {
    os << "\n  Is unsigned: yes";
    os << "\nUnsigned integer support is currently spotty. Please seeheck "
          "https://github.com/llvm/torch-mlir/issues/3720 "
          "for more details.";
  }
  llvm::report_fatal_error(llvm::StringRef(errorMsg));
}

inline FailureOr<Type>
getTypeForScalarType(MLIRContext *context,
                     torch_upstream::ScalarType dtypeInt) {
  switch (dtypeInt) {
  case torch_upstream::ScalarType::Float:
    return Float32Type::get(context);
  case torch_upstream::ScalarType::Double:
    return Float64Type::get(context);
  case torch_upstream::ScalarType::Long:
    return IntegerType::get(context, 64, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::Int:
    return IntegerType::get(context, 32, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::Short:
    return IntegerType::get(context, 16, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::Bool:
    return IntegerType::get(context, 1);
  case torch_upstream::ScalarType::BFloat16:
    return mlir::BFloat16Type::get(context);
  case torch_upstream::ScalarType::Half:
    return mlir::Float16Type::get(context);
  case torch_upstream::ScalarType::Byte:
    return mlir::IntegerType::get(context, 8, mlir::IntegerType::Unsigned);
  case torch_upstream::ScalarType::Char:
    return mlir::IntegerType::get(context, 8, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::QUInt8:
    return QUInt8Type::get(context);
  case torch_upstream::ScalarType::QInt8:
    return QInt8Type::get(context);
  case torch_upstream::ScalarType::QInt16:
    return QInt16Type::get(context);
  case torch_upstream::ScalarType::QInt32:
    return QInt32Type::get(context);
  case torch_upstream::ScalarType::ComplexHalf:
    return mlir::ComplexType::get(Float16Type::get(context));
  case torch_upstream::ScalarType::ComplexFloat:
    return mlir::ComplexType::get(Float32Type::get(context));
  case torch_upstream::ScalarType::ComplexDouble:
    return mlir::ComplexType::get(Float64Type::get(context));
  case torch_upstream::ScalarType::Float8_e5m2:
    return Float8E5M2Type::get(context);
  case torch_upstream::ScalarType::Float8_e4m3fn:
    return Float8E4M3FNType::get(context);
  case torch_upstream::ScalarType::Float8_e5m2fnuz:
    return Float8E5M2FNUZType::get(context);
  case torch_upstream::ScalarType::Float8_e4m3fnuz:
    return Float8E4M3FNUZType::get(context);
  case torch_upstream::ScalarType::Float8_e8m0fnu:
    return Float8E8M0FNUType::get(context);
  case torch_upstream::ScalarType::Float4_e2m1fn_x2:
    return Float4E2M1FNType::get(context);
  case torch_upstream::ScalarType::Undefined:
    return failure();
  default:
    llvm::report_fatal_error("unhandled type for getTypeForScalarType");
  }
}

Type getTypeForTorchType(
    MLIRContext *context, Type type,
    mlir::IntegerType::SignednessSemantics signedness = IntegerType::Signed);

template <typename OpTy>
FailureOr<Value> getDtypeFromOp(PatternRewriter &rewriter, OpTy op);

FailureOr<Type> getTorchTypeForScalarType(MLIRContext *context,
                                          torch_upstream::ScalarType dtypeInt);

// This is the type rule used for deciding dtype for:
// 1. A new tensor created from given data.
// 2. The scalar type for type promotion when a scalar is an operand of a tensor
// operation (such as AtenMulScalarOp, AtenAddScalarOp etc)
// If the data is floating-point, the `dtype` is inferred to be the
// default dtype, see `torch.get_default_dtype`.
Type getDefaultDtypeForTorchScalar(Type type);

// This is the type rule used for deciding builtin type for:
// 1. The dtype of the result tensor when converting a Scalar into a Tensor like
// PrimNumToTensorScalarOp.
// 2. The scalar type for type promotion when a scalar is an operand of scalar
// only operation like AtenAddOp.
Type getBuiltInTypeForTorchScalar(Type type);

Value getDtypeIntValueForType(PatternRewriter &rewriter, Location loc,
                              Type dtype);

// Checks whether the inputs are broadcast compatible or not. If
// yes, then computes the final broadcast shape.
void computeBroadcastShape(PatternRewriter &rewriter, Location loc,
                           ArrayRef<Value> inputs,
                           SmallVector<int64_t> &resultShape,
                           SmallVector<Value> &resultShapeValue);

// Helper to convert a tensor to a specific scalar type.
Value convertTensorToDtype(PatternRewriter &rewriter, Location loc, Value input,
                           Type dtype);

bool isBuiltInType(Type type);

// Helper function to get rank of `Base tensor type`.
// std::nullopt is returned if the tensorRank can't be determined.
inline std::optional<unsigned> getTensorRank(Value tensor) {
  BaseTensorType tensorType = cast<BaseTensorType>(tensor.getType());
  if (!tensorType.hasSizes())
    return std::nullopt;
  return tensorType.getSizes().size();
}

// Helper function to get the number of elements in a tensor.
std::optional<int64_t> getTensorNumel(Value tensor);

bool isViewLikeOp(Operation *op);

Value getConstantWithGivenDtypeAndValue(PatternRewriter &rewriter, Location loc,
                                        float value, Type dtype);

// Return the number of elements of a tensor if the shape is static; otherwise,
// return -1.
int64_t getNumberOfElements(RankedTensorType inputType);

inline SmallVector<int64_t> makeShapeLLVMCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kUnknownSize);
    if (shape[i] == kUnknownSize)
      updatedShape[i] = kDynamic;
  }
  return updatedShape;
}

SmallVector<int64_t> makeShapeTorchCompatible(ArrayRef<int64_t> shape);

ValueTensorType getTensorTypeFromShapeValues(ArrayRef<Value> shapes,
                                             Type dtype);
Value getTensorDimSize(PatternRewriter &rewriter, Value tensor, int64_t dim);

// In Dynamo import paths, we can assume that dynamic dimensions are strictly
// quantities and are not ambiguous with '1' symbols that can be interpreted
// to signal an expansion in various broadcasting scenarios. In the
// torch.compile eager path, this precondition is assured by guards on 0/1
// dimension values, and on the torch.export graph-capture path, the shape
// solver guarantees this.
//
// We let lowerings assume this on a per-scope basis if the
// torch.assume_strict_symbolic_shapes unit attribute is present on any parent
// of the block.
bool isAssumingStrictSymbolicShapes(Block *scope);

// Helper that uses the block from an OpBuilder for determining whether we
// are assuming strict symbolic shapes.
inline bool isAssumingStrictSymbolicShapes(OpBuilder &builder) {
  return isAssumingStrictSymbolicShapes(builder.getBlock());
}

// Helper function for AtenEmptyStrided and friends that checks if the stride
// values are default or not. Throws a runtime assert if not.
LogicalResult checkDefaultStrideHelper(Operation *op, PatternRewriter &rewriter,
                                       Value opSize, Value opStride,
                                       Location loc);

// Helper to create a tensor filled with the given scalar. Scalar would be
// converted the to the element type of the given tensor type.
Value createInitTensor(PatternRewriter &rewriter, Location loc,
                       BaseTensorType resultType, Value scalar, Value sizeList);

// Helper to create a rank 0 tensor filled with the given `scalar`. `scalar`
// would be converted to the element type of the given `inputType`.
Value createRank0Tensor(PatternRewriter &rewriter, Location loc,
                        BaseTensorType inputType, Value scalar);

LogicalResult getTransposedType(BaseTensorType inType, int64_t dimA,
                                int64_t dimB, Type &transposedType);

// Approximates the heuristic in the torch `acc_type` template for kernels
// that are defined in terms of it. For now, this just returns accumulators
// as if for CUDA from that implementation. In the future, this could be
// extended to look at hints on the `forOp` or its container to better
// control the behavior. Such support would be done in coordination with
// the fx_importer and APIs, which could add hints to the IR (based on
// Torch flags, user options, etc).
// Note: The special case of int8 intentionally deviates from the reference, and
// uses int32 instead of int64 accumulation.
Type getDefaultAccType(PatternRewriter &rewriter, Type inputType);

LogicalResult getPermutedType(BaseTensorType inType,
                              SmallVector<int64_t> permuteDims,
                              Type &permutedType);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_UTILS_H
