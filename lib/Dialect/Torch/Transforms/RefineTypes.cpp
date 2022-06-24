//===- RefineTypes.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file implements a dataflow analysis primarily used to infer dtypes
// of tensors in the program. Shapes are handled separately with a
// more involved mechanism (see createTorchShapeRefinementPipeline).
//
// The analysis performed in this file is implemented with MLIR's dataflow
// analysis framework, which was originally developed for SCCP, and so is an
// optimistic framework. It proceeds by assuming that all Value's have a
// maximally optimistic ("bottom") lattice element associated with them, and
// then the `visitOperation` method (and some built-in handling for control
// flow) gradually relaxes that optimism until the lattice elements associated
// with each Value either settle to a (optimistic) fixed-point, or need to fall
// back on a suitable pessimistic lattice element.
//
// A note on dataflow analysis terminology:
// In dataflow analysis (or other contexts where lattices appear), it is
// frequently confusing because meet/join and related aspects of lattices
// (such as what is "up"/"down" or "top"/"bottom" in the lattice) are dual to
// each other and so a convention has to be chosen to ground the terminology.
//
// In the context of this dataflow analysis, we use the terms with the following
// senses (many examples are given to build intuition):
// - "top" means the state of least specific knowledge (i.e. most pessimistic
// possible knowledge)
// - "bottom" is the lattice element with such specific knowledge that "join"ing
// with it is an identity operation. (i.e. most optimistic possible knowledge)
// - "moving down the lattice" means moving towards having more specific
// knowledge
// - "moving up the lattice" means moving towards having less specific knowledge
// - "top" means the state of least specific knowledge (i.e. most pessimistic
// possible knowledge)
// - "meet" means
//   - "move down the lattice" (greatest lower bound)
//   - "constrict"
//   - "refine"
//   - "assume union of information from both lattice elements"
// - "join" means
//   - "move up the lattice" (least upper bound)
//   - "widen"
//   - "relax"
//   - "assume intersection of information from both lattice elements"
//
// Note: This pass is kept completely separate from
// createShapeRefinementPipeline because any interaction between the two would
// usually require a fixed-point iteration to work in generality.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// Analysis.
// -----------------------------------------------------------------------------

static Type getTypeForDTypeInteger(MLIRContext *context, int64_t dtypeInt) {
  return getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
}

static Type getDtypeOrDefault(MLIRContext *context, Value optionalDtype,
                              Type defaultDtype) {
  int64_t dtypeInt;
  if (matchPattern(optionalDtype, m_TorchConstantInt(&dtypeInt)))
    return getTypeForDTypeInteger(context, dtypeInt);
  else if (optionalDtype.getType().isa<Torch::NoneType>())
    return defaultDtype;
  return Type();
}

// Get the kind enum for `ValueKnowledge.kind`.
static torch_upstream::TypeKind getTypeKind(Type type) {
  if (type.isa<NumberType>())
    return torch_upstream::TypeKind::NumberType;
  if (type.isa<IntType>())
    return torch_upstream::TypeKind::IntType;
  if (type.isa<Torch::FloatType>())
    return torch_upstream::TypeKind::FloatType;
  if (type.isa<BaseTensorType>())
    return torch_upstream::TypeKind::TensorType;
  if (type.isa<Torch::NoneType>())
    return torch_upstream::TypeKind::NoneType;
  // Skip the Torch::OptionalType on purpose because optional knowledge is
  // tracked separately. See comments for `ValueKnowledge.kind` field.
  return torch_upstream::TypeKind::AnyType;
}

/// Returns the dtype that assumes information from both `lhs` and `rhs`.
/// Returns `None` if the types are contradictory. Note this can only be used
/// on the `dtype` from tensors and can't be used on other types like scalar
/// types.
static Optional<Type> meetElementTypes(Type lhs, Type rhs) {
  auto isNullOrBuiltIn = [](Type type) { return !type || isBuiltInType(type); };
  (void)isNullOrBuiltIn;
  assert(isNullOrBuiltIn(lhs) && "`lhs` must be a builtin type");
  assert(isNullOrBuiltIn(rhs) && "`rhs` must be a builtin type");

  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  if (lhs == rhs)
    return lhs;
  return None;
}

enum class OptionalKnowledge {
  unKnown,
  isNone,
  notNone,
};

/// Returns the OptionalKnowledge that assumes information from both `lhs` and
/// `rhs`. Returns `None` if the knowledges are contradictory.
static Optional<OptionalKnowledge>
meetOptionalKnowledge(OptionalKnowledge lhs, OptionalKnowledge rhs) {
  if (lhs == OptionalKnowledge::unKnown)
    return rhs;
  if (rhs == OptionalKnowledge::unKnown)
    return lhs;
  if (lhs == rhs)
    return lhs;
  return None;
}

static OptionalKnowledge joinOptionalKnowledge(OptionalKnowledge lhs,
                                               OptionalKnowledge rhs) {
  if (lhs == rhs)
    return lhs;
  return OptionalKnowledge::unKnown;
}

namespace {
// Statically known information for a particular Value.
//
// This struct currently tracks information relevant for tensor/array-like
// shaped types as well as whether an object is None or not, namely
// !torch.optional. It is fine to associate a `ValueKnowledge` with a non-shaped
// type or non OptionalType as long as it is in the default "no knowledge"
// state returned by `getPessimisticValueState`. The important invariant is that
// we cannot claim to know something about a value which is false.
// This class could also be called "dataflow facts", "lattice value", etc.
struct ValueKnowledge {
  ValueKnowledge() = delete;
  ValueKnowledge(Type dtype, Type scalarType,
                 OptionalKnowledge optionalKnowledge,
                 torch_upstream::TypeKind kind)
      : dtype(dtype), scalarType(scalarType), kind(kind),
        optional(optionalKnowledge) {}

  void setScalarType(Type type) {
    bool isValidScalarType = type.isa<NumberType, IntType, Torch::FloatType>();
    (void)isValidScalarType;
    assert(isValidScalarType &&
           "scalarType can only be one of NumberType, IntType and FloatType");
    scalarType = type;
    kind = getTypeKind(type);
  }

  // Get the static knowledge intrinsic to `type`.
  static ValueKnowledge getKnowledgeFromType(Type type) {
    ValueKnowledge result = getPessimisticValueState(type.getContext());
    result.kind = getTypeKind(type);
    switch (result.kind) {
    case torch_upstream::TypeKind::TensorType:
      result.dtype = type.cast<BaseTensorType>().getOptionalDtype();
      result.optional = OptionalKnowledge::notNone;
      return result;
    case torch_upstream::TypeKind::NumberType:
    case torch_upstream::TypeKind::IntType:
    case torch_upstream::TypeKind::FloatType:
      result.scalarType = type;
      result.optional = OptionalKnowledge::notNone;
      return result;
    case torch_upstream::TypeKind::NoneType:
      result.optional = OptionalKnowledge::isNone;
      return result;
    default:
      if (type.isa<OptionalType>())
        return result;
      // All other types that are not optional type.
      result.optional = OptionalKnowledge::notNone;
      return result;
    }
  }

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(Type(), Type(), OptionalKnowledge::unKnown,
                          torch_upstream::TypeKind::AnyType);
  }
  // Return a pessimistic/conservative value state only using knowlege already
  // recorded in the IR.
  static ValueKnowledge getPessimisticValueState(Value value) {
    return getKnowledgeFromType(value.getType());
  }

  static ValueKnowledge getNotNonePessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(Type(), Type(), OptionalKnowledge::notNone,
                          torch_upstream::TypeKind::AnyType);
  }

  static ValueKnowledge getTensorPessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(Type(), Type(), OptionalKnowledge::notNone,
                          torch_upstream::TypeKind::TensorType);
  }

  static ValueKnowledge getScalarPessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(Type(), NumberType::get(context),
                          OptionalKnowledge::notNone,
                          torch_upstream::TypeKind::NumberType);
  }

  bool operator==(const ValueKnowledge &rhs) const {
    return std::make_tuple(dtype, optional) ==
           std::make_tuple(rhs.dtype, rhs.optional);
  }

  // Return true if the `refinedType` has more concrete type info than `type`.
  static bool hasStrictlyMoreRefinedTypeInfo(const ValueKnowledge &refinedType,
                                             const ValueKnowledge &type) {
    if (type.kind == torch_upstream::TypeKind::AnyType &&
        refinedType.kind != torch_upstream::TypeKind::AnyType)
      return true;

    // If both are tensors but `type` doesn't have concrete dtype info.
    if (refinedType.kind == torch_upstream::TypeKind::TensorType &&
        type.kind == torch_upstream::TypeKind::TensorType) {
      return refinedType.dtype && !type.dtype;
    }

    if (refinedType.scalarType && type.scalarType)
      return isValidSubtype(refinedType.scalarType, type.scalarType);

    return false;
  }

  // Given two pieces of static knowledge, intersect the facts that are known in
  // both knowledges. This always produces knowledge that has less (or equal)
  // facts than both the lhs and rhs.
  //
  // This operator is used, for example, at control flow join points: if
  // predecessors A and B forward a block argument to a common successor C, then
  // we need to calculate what can be known for sure about the block argument if
  // the control flow is coming from either A or B. So we can't assume facts
  // just because they are true on one control flow edge. They must be true on
  // both.
  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs) {
    // Mental model: All conditions are checking how to change from the safe "no
    // knowledge" default-initialized state to a state with more knowledge
    // consistent with lhs and rhs.
    ValueKnowledge result = joinTypes(lhs, rhs);
    result.optional = joinOptionalKnowledge(lhs.optional, rhs.optional);
    return result;
  }

  static ValueKnowledge joinTypes(const ValueKnowledge &lhs,
                                  const ValueKnowledge &rhs) {
    if (hasStrictlyMoreRefinedTypeInfo(lhs, rhs))
      return rhs;
    if (hasStrictlyMoreRefinedTypeInfo(rhs, lhs))
      return lhs;
    if (lhs == rhs)
      return lhs;
    return getPessimisticValueState(nullptr);
  }

  // Given two pieces of static knowledge, calculate new knowledge that assumes
  // the facts from both.
  // If the two pieces of knowledge are contradictory, None is returned.
  static Optional<ValueKnowledge> meet(const ValueKnowledge &lhs,
                                       const ValueKnowledge &rhs) {
    Optional<ValueKnowledge> knowledge = meetTypes(lhs, rhs);

    if (!knowledge.hasValue())
      return None;
    ValueKnowledge result = knowledge.getValue();

    Optional<OptionalKnowledge> optional =
        meetOptionalKnowledge(lhs.optional, rhs.optional);
    if (!optional.hasValue())
      return None;
    result.optional = optional.getValue();
    return result;
  }

  static Optional<ValueKnowledge> meetTypes(const ValueKnowledge &lhs,
                                            const ValueKnowledge &rhs) {
    if (hasStrictlyMoreRefinedTypeInfo(lhs, rhs))
      return lhs;
    if (hasStrictlyMoreRefinedTypeInfo(rhs, lhs))
      return rhs;
    if (lhs == rhs)
      return lhs;
    return None;
  }

  // The dtype of a tensor.
  // This is equal to nullptr for the follow cases:
  // 1. it is unknown whether the value is a tensor or not, ie the `kind` field
  // is torch_upstream::TypeKind::AnyType.
  // 2. the value is a tensor type but the dtype is unknown.
  // 3. the value is not a tensor type.
  Type dtype;

  // The type of a scalar.
  // This is equal to nullptr for the follow cases:
  // 1. it is unknown whether the value is a scalar or not, ie the `kind` field
  // is torch_upstream::TypeKind::AnyType.
  // 2. the value is not a scalar type.
  Type scalarType;

  // The type kind. If it's torch_upstream::TypeKind::AnyType,
  // all the type fields are nullptr. Note that the `kind` never equals to
  // torch_upstream::TypeKind::OptionalType because optional knowledge is
  // tracked separately through the `optional` field.
  torch_upstream::TypeKind kind;

  // What is known about an optional value.
  // When equal to OptionalKnowledge::notNone, the type info is kept in type
  // fields like `dtype`, `scalarType`.
  // When equal to OptionalKnowledge::isNone or OptionalKnowledge::unKnown, the
  // other type fields are currently nullptr. It might worth considering
  // tracking wrapped type info when OptionalKnowledge::unKnown in the future.
  OptionalKnowledge optional;
};

// Forward intraprocedural dataflow for type information.
class TypeAnalyzer : public ForwardDataFlowAnalysis<ValueKnowledge> {
public:
  using ForwardDataFlowAnalysis<ValueKnowledge>::ForwardDataFlowAnalysis;

  // Compute the knowledge for the results of an op, based on the knowledge of
  // the operands and any information intrinsic to `op`.
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<ValueKnowledge> *> operands) final;

private:
  /// Incorporates `knowledge` into the lattice state of `v`.
  ///
  /// This method should be used instead of
  /// `getLatticeElement(v).join(knowledge)`, because this method knows how to
  /// correctly handle the case of existing static knowledge from the type
  /// of `v`.
  ChangeResult incorporateKnowledge(Value v, const ValueKnowledge &knowledge);

  ChangeResult
  visitAtenLinearOp(AtenLinearOp op,
                    ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitAtenArangeStartStepOp(AtenArangeStartStepOp op);
  ChangeResult visitAtenArangeStartOp(AtenArangeStartOp op);
  ChangeResult visitAtenArangeOp(AtenArangeOp op);
  ChangeResult visitAtenArangeLikeOpHelper(Operation *op,
                                           llvm::Optional<Value> start,
                                           Value end,
                                           llvm::Optional<Value> step,
                                           Value dtype);
  ChangeResult visitReductionAlongAllDimsOp(
      Operation *op, Type dtype,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitReductionAlongDimIntListOp(
      Operation *op, Value dim, Value keepdim, Type dtype,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitReductionAlongDimIntOp(
      Operation *op, Value dim, Value keepdim, Type dtype,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands, int resNum = 0);
  template <typename OpTy>
  ChangeResult visitScalarToTensorConversionOp(OpTy op);
  ChangeResult visitAtenTensorOp(AtenTensorOp op);
  template <typename OpTy>
  ChangeResult visitConstantTensorAllocOp(OpTy op,
                                          llvm::Optional<Type> dataType);
  template <typename OpTy>
  ChangeResult visitConstantTensorAllocLikeOp(
      OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  template <typename OpTy>
  ChangeResult visitConstantTensorNewLikeOp(
      OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  template <typename OpTy>
  ChangeResult
  visitAtenToDtypeLikeOp(OpTy op,
                         ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  template <typename OpTy>
  ChangeResult
  visitTypeConversionOp(OpTy op,
                        ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenCatOp(AtenCatOp op,
                 ArrayRef<LatticeElement<ValueKnowledge> *> operands);

  template <typename OpTy>
  ChangeResult
  visitAtenSoftmaxLikeOp(OpTy op,
                         ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  template <typename OpTy>
  ChangeResult
  visitAten_SoftmaxLikeOp(OpTy op,
                          ArrayRef<LatticeElement<ValueKnowledge> *> operands);

  ChangeResult visitNumToTensorOp(PrimNumToTensorScalarOp op);
  ChangeResult
  visitBinaryScalarOp(Operation *op,
                      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitAtenScalarImplicitOp(
      AtenScalarImplicitOp op,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
};
} // namespace

// This is the type rule used for deciding dtype for:
// 1. A new tensor created from given data.
// 2. The scalar type for type promotion when a scalar is an operand of a tensor
// operation (such as AtenMulScalarOp, AtenAddScalarOp etc)
// If the data is floating-point, the `dtype` is inferred to be the
// default dtype, see `torch.get_default_dtype`.
static Type getDefaultDtypeForTorchScalar(Type type) {
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

// This is the type rule used for deciding builtin type for:
// 1. The dtype of the result tensor when converting a Scalar into a Tensor like
// PrimNumToTensorScalarOp.
// 2. The scalar type for type promotion when a scalar is an operand of scalar
// only operation like AtenAddOp.
static Type getBuiltInTypeForTorchScalar(Type type) {
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

static torch_upstream::ResultTypeState
updateResultTypeState(Type scalarType,
                      const torch_upstream::ResultTypeState &inState) {
  assert(isBuiltInType(scalarType) && "scalarType must be builtin type");
  torch_upstream::ResultTypeState new_state = inState;
  torch_upstream::ScalarType current = getScalarTypeForType(scalarType);
  new_state.wrappedResult =
      promote_skip_undefined(inState.wrappedResult, current);
  return new_state;
}

// This mostly mirrors the update_result_type_state in
// aten/src/ATen/native/TypeProperties.* except that we don't not support
// is_wrapped_number as it is a runtime property. From perspective of
// torch-mlir, all zero dim tensor are the same priority.
//
// Normally, tensor dimensions need to be known at compile time to do type
// promotion. `skipRankCheck`, when equal to `true`, is used for special cases
// where rank doesn't matter. This could be because that operands can sometimes
// guaranteed to be none zero rank or that the result
// torch_upstream::ResultTypeState is promoted with a scalar which is guaranteed
// to be lower priority.
//
// The `rankIsNonZero` argument indicates whether the rank is nonzero, zero, or
// unknown (None variant of the optional).
static torch_upstream::ResultTypeState
updateResultTypeState(ValueKnowledge *tensor, Optional<bool> rankIsNonZero,
                      const torch_upstream::ResultTypeState &inState,
                      bool skipRankCheck = false) {
  if (!rankIsNonZero.hasValue() && !skipRankCheck)
    return torch_upstream::ResultTypeState{};
  assert(tensor->dtype && "tensor.dtype must be not none");

  torch_upstream::ResultTypeState new_state = inState;
  torch_upstream::ScalarType current = getScalarTypeForType(tensor->dtype);
  if (skipRankCheck || rankIsNonZero.getValue())
    new_state.dimResult = promote_skip_undefined(inState.dimResult, current);
  else
    new_state.zeroResult = promote_skip_undefined(inState.zeroResult, current);

  return new_state;
}

// Type promotion helper for operators where only scalar operands participating
// in type promotion like AtenAddOp.
//
// \return The return type is a TorchType.
static Type getPromotedResultScalarType(ArrayRef<Type> scalarTypes) {
  torch_upstream::ResultTypeState state = {};
  for (const Type &scalarType : scalarTypes) {
    state =
        updateResultTypeState(getBuiltInTypeForTorchScalar(scalarType), state);
  }
  return getTorchTypeForScalarType(scalarTypes[0].getContext(),
                                   result_type(state));
}

// Returns most generic type Type() if the tensor dtype is unknown.
static Type getPromotedResultDType(ValueKnowledge *tensor, Type scalarType) {
  if (!tensor->dtype)
    return Type();
  torch_upstream::ResultTypeState state = {};
  // No need to check if rank is zero for tensor because scalar uses
  // wrappedResult which is a lower priority than both dimResult and zeroResult.
  state = updateResultTypeState(tensor, /*rankIsNonZero=*/None, state,
                                /*skipRankCheck=*/true);
  state =
      updateResultTypeState(getDefaultDtypeForTorchScalar(scalarType), state);
  return getTypeForScalarType(scalarType.getContext(), result_type(state));
}

static SmallVector<Optional<bool>> getRankIsNonZeroArray(ValueRange values) {
  SmallVector<Optional<bool>> rankIsNonZero;
  for (Value v : values) {
    if (auto tensorType = v.getType().dyn_cast<BaseTensorType>()) {
      if (tensorType.hasSizes()) {
        rankIsNonZero.push_back(tensorType.getSizes().size() != 0);
      } else {
        rankIsNonZero.push_back(None);
      }
    }
  }
  return rankIsNonZero;
}

// Normally, tensor dimensions need to be known at compile time to do type
// promotion. `skipRankCheck`, when equal to true, can be used to indicate
// special cases that tensor operands are guaranteed to be not zero dimension
// like operands of `aten.conv2d` or `aten.mm` assuming no runtime error.
//
// Returns most generic type Type() if the tensor dtype is unknown.
static Type getPromotedResultType(MLIRContext *context,
                                  ArrayRef<ValueKnowledge *> tensors,
                                  ArrayRef<Optional<bool>> rankIsNonZero,
                                  bool skipRankCheck = false) {
  torch_upstream::ResultTypeState state = {};
  assert(tensors.size() == rankIsNonZero.size());
  for (auto t : llvm::zip(tensors, rankIsNonZero)) {
    ValueKnowledge *tensor = std::get<0>(t);
    Optional<bool> rankIsNonZero = std::get<1>(t);
    if (!tensor->dtype)
      return Type();
    state = updateResultTypeState(tensor, rankIsNonZero, state, skipRankCheck);
  }
  return getTypeForScalarType(context, result_type(state));
}

static Type
getPromotedResultTypeAssumingNonZeroRank(MLIRContext *context,
                                         ArrayRef<ValueKnowledge *> tensors) {
  SmallVector<Optional<bool>> rankIsNonZero(tensors.size(), true);
  return getPromotedResultType(context, tensors, rankIsNonZero,
                               /*skipRankCheck=*/true);
}

// Get the MLIR type of the tensor dtype given the dtype integer value and the
// input dtype. When DType is None the type is inferred from the input dtype.
static void fillInDTypeGivenDTypeIntAndInputDType(ValueKnowledge &knowledge,
                                                  Value dtype,
                                                  Type inputDType) {
  assert(isBuiltInType(inputDType) && "`inputDType` must be a builtin type");
  int64_t dtypeInt;
  if (dtype.getType().isa<Torch::NoneType>())
    knowledge.dtype = inputDType;
  else if (matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    knowledge.dtype = getTypeForDTypeInteger(dtype.getContext(), dtypeInt);
}

// Get the MLIR type of the tensor dtype given the dtype integer value and data
// type of torch type. When DType is None the type is inferred from the data
// type.
static void fillInDTypeGivenDTypeAndDataType(ValueKnowledge &knowledge,
                                             Value dtype, Type dataType) {
  assert(isa<TorchDialect>(dataType.getDialect()) &&
         "`dataType` must be a torch type");
  Type dtypeForDataType = getDefaultDtypeForTorchScalar(dataType);
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, dtype, dtypeForDataType);
}

ChangeResult TypeAnalyzer::visitOperation(
    Operation *op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {

  // These ops have results that are dynamically the same as their operands.
  if (isa<TensorStaticInfoCastOp, DerefineOp>(op)) {
    return incorporateKnowledge(op->getResult(0), operands[0]->getValue());
  }

  // Take dtype from first operand.
  if (isa<CopyToValueTensorOp, CopyToNonValueTensorOp, AtenBatchNormOp,
          AtenReluOp, AtenGeluOp, AtenCeilOp, AtenGeluBackwardOp,
          AtenBitwiseNotOp, AtenToPrimDeviceOp, AtenCpuOp, AtenContiguousOp,
          AtenFill_ScalarOp, AtenDetachOp, AtenMaskedFill_ScalarOp, AtenCopy_Op,
          AtenCumsumOp, AtenLayerNormOp, AtenClampOp, AtenClampMinOp,
          AtenClampMaxOp, AtenNegOp, AtenFloorOp, Aten_SoftmaxBackwardDataOp,
          AtenDropoutOp, AtenTanhBackwardOp, Aten_LogSoftmaxBackwardDataOp,
          AtenAddIntOp, AtenAbsOp, AtenThresholdOp, AtenSquareOp,
          ValsemVariantAtenUniformOp, AtenBernoulliOp, AtenBernoulli_FloatOp,
          AtenBernoulli_TensorOp, ValsemVariantAtenBernoulliFloatOp,
          ValsemVariantAtenBernoulliTensorOp, ValsemVariantAtenFillScalarOp,
          AtenHardsigmoidOp, AtenCloneOp, AtenHardswishOp, AtenSiluOp,
          AtenHardtanhOp, AtenMaskedSelectOp, AtenMaxPool2dOp, AtenAvgPool2dOp,
          AtenAdaptiveAvgPool2dOp, AtenFlattenUsingIntsOp, AtenSqueezeOp,
          AtenSqueezeDimOp, AtenUnsqueezeOp, AtenViewOp, Aten_UnsafeViewOp,
          AtenReshapeOp, Aten_ReshapeAliasOp, AtenResize_Op, AtenTransposeIntOp,
          AtenTOp, AtenPermuteOp, AtenIndexSelectOp, AtenSelectIntOp,
          AtenSliceTensorOp, AtenGatherOp, AtenExpandOp, AtenExpandAsOp,
          AtenBroadcastToOp, AtenRepeatOp, AtenConstantPadNdOp, AtenPadOp,
          AtenZero_Op, AtenIndexTensorOp, ValsemVariantAtenIndexPutImplOp,
          AtenIndexPutOp, ValsemVariantAtenCopyOp, AtenZeroFunctionalOp,
          AtenIndexPutHackedTwinOp, AtenMaskedFillScalarOp, AtenFlipOp,
          PrimAbsScalarOp, AtenNumpyTOp, AtenTriuOp>(op)) {
    return incorporateKnowledge(op->getResult(0), operands[0]->getValue());
  }

  // Dtype is always float32, except for bfloat16, float64 and nullptr.
  if (isa<AtenTanhOp, AtenExpOp, AtenSinOp, AtenCosOp, AtenSigmoidOp,
          AtenReciprocalOp, AtenLogOp, AtenSqrtOp, AtenLog2Op, AtenRsqrtOp,
          AtenErfOp>(op)) {
    ValueKnowledge knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    Type dtype = operands[0]->getValue().dtype;
    if (dtype) {
      knowledge.dtype = Float32Type::get(op->getContext());
      if (dtype.isa<BFloat16Type, Float64Type>())
        knowledge.dtype = dtype;
    }
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Take dtype from second operand.
  if (isa<AtenNllLossBackwardOp, AtenMaxPool2dWithIndicesBackwardOp>(op)) {
    auto self = operands[1]->getValue();
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = self.dtype;
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Dtype is always i1.
  if (isa<AtenEqScalarOp, AtenGeScalarOp, AtenGtScalarOp, AtenLtScalarOp,
          AtenLeScalarOp, AtenNeScalarOp, AtenAnyOp, AtenAllOp, AtenEqTensorOp,
          AtenGtTensorOp, AtenLtTensorOp, AtenLogicalOrOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = IntegerType::get(op->getContext(), 1);
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Dtype is always si64.
  if (isa<AtenBincountOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Promote the two dtypes assuming non-zero rank.
  if (isa<AtenMmOp, AtenBmmOp, AtenMatmulOp, AtenConv2dOp, AtenConvolutionOp,
          AtenConvolutionOverrideableOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue()});
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Promote the two dtypes assuming possibly-zero rank.
  if (isa<AtenAddTensorOp, AtenSubTensorOp, AtenMulTensorOp, AtenDivTensorOp,
          AtenDivTensorModeOp, Aten__And__TensorOp, AtenMinimumOp,
          AtenMaximumOp, AtenBitwiseAndTensorOp, AtenThresholdBackwardOp,
          AtenFloorDivideOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultType(
        op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue()},
        getRankIsNonZeroArray(op->getOperands()));
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Promote three dtypes.
  if (isa<AtenAddmmOp, AtenLerpTensorOp, AtenAddcmulOp, AtenAddcdivOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue(),
                           &operands[2]->getValue()});
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  if (auto linear = llvm::dyn_cast<AtenLinearOp>(op)) {
    return visitAtenLinearOp(linear, operands);
  }

  // Promote LHS with scalar RHS.
  if (isa<AtenAddScalarOp, AtenSubScalarOp, AtenMulScalarOp, AtenDivScalarOp,
          AtenFmodScalarOp, AtenFloorDivideScalarOp, AtenPowTensorScalarOp,
          AtenRsubScalarOp, AtenLeakyReluOp>(op)) {
    auto lhs = operands[0]->getValue();
    Value scalar = op->getOperand(1);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(getContext());
    knowledge.dtype = getPromotedResultDType(&lhs, scalar.getType());
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereSelfOp, AtenBaddbmmOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(getContext());
    knowledge.dtype = getPromotedResultType(
        getContext(), {&operands[1]->getValue(), &operands[2]->getValue()},
        getRankIsNonZeroArray(op->getOperands().slice(1, 2)));
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereScalarOp>(op)) {
    Value lhsScalar = op->getOperand(1);
    Value rhsScalar = op->getOperand(2);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(getContext());
    knowledge.dtype = getDefaultDtypeForTorchScalar(getPromotedResultScalarType(
        {lhsScalar.getType(), rhsScalar.getType()}));
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereScalarOtherOp>(op)) {
    auto lhs = operands[1]->getValue();
    Value scalar = op->getOperand(2);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(getContext());
    knowledge.dtype = getPromotedResultDType(&lhs, scalar.getType());
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereScalarSelfOp>(op)) {
    auto rhs = operands[2]->getValue();
    Value scalar = op->getOperand(1);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(getContext());
    knowledge.dtype = getPromotedResultDType(&rhs, scalar.getType());
    return incorporateKnowledge(op->getResult(0), knowledge);
  }

  // 2 results take dtype from first operand.
  if (isa<AtenNllLossForwardOp>(op)) {
    auto self = operands[0]->getValue();
    auto result0Knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    result0Knowledge.dtype = self.dtype;
    auto result1Knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    result1Knowledge.dtype = self.dtype;
    auto changed = incorporateKnowledge(op->getResult(0), result0Knowledge);
    changed |= incorporateKnowledge(op->getResult(1), result1Knowledge);
    return changed;
  }

  // 3 results take dtype from first operand.
  if (isa<AtenNativeLayerNormOp, AtenNativeBatchNormOp>(op)) {
    auto self = operands[0]->getValue();
    auto result0Knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    result0Knowledge.dtype = self.dtype;
    auto result1Knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    result1Knowledge.dtype = self.dtype;
    auto result2Knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    result2Knowledge.dtype = self.dtype;
    auto changed = incorporateKnowledge(op->getResult(0), result0Knowledge);
    changed |= incorporateKnowledge(op->getResult(1), result1Knowledge);
    changed |= incorporateKnowledge(op->getResult(2), result1Knowledge);
    return changed;
  }

  if (isa<AtenMaxPool2dWithIndicesOp>(op)) {
    auto self = operands[0]->getValue();
    auto result0Knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    result0Knowledge.dtype = self.dtype;
    auto result1Knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    result1Knowledge.dtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    ;
    auto changed = incorporateKnowledge(op->getResult(0), result0Knowledge);
    changed |= incorporateKnowledge(op->getResult(1), result1Knowledge);
    return changed;
  }

  if (auto arange = dyn_cast<AtenArangeOp>(op)) {
    return visitAtenArangeOp(arange);
  }
  if (auto arangeStart = dyn_cast<AtenArangeStartOp>(op)) {
    return visitAtenArangeStartOp(arangeStart);
  }
  if (auto arangeStartStep = dyn_cast<AtenArangeStartStepOp>(op)) {
    return visitAtenArangeStartStepOp(arangeStartStep);
  }

  if (auto sum = dyn_cast<AtenSumOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype = getDtypeOrDefault(sum.getContext(), sum.dtype(), defaultDtype);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = dtype;
    return incorporateKnowledge(op->getResult(0), knowledge);
  }
  if (auto sumDimIntList = dyn_cast<AtenSumDimIntListOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype = getDtypeOrDefault(sumDimIntList.getContext(),
                                   sumDimIntList.dtype(), defaultDtype);
    return visitReductionAlongDimIntListOp(sumDimIntList, sumDimIntList.dim(),
                                           sumDimIntList.keepdim(), dtype,
                                           operands);
  }
  if (auto meanDim = dyn_cast<AtenMeanDimOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype =
        getDtypeOrDefault(meanDim.getContext(), meanDim.dtype(), defaultDtype);
    return visitReductionAlongDimIntListOp(meanDim, meanDim.dim(),
                                           meanDim.keepdim(), dtype, operands);
  }
  if (auto argmax = dyn_cast<AtenArgmaxOp>(op)) {
    Value dim = argmax.dim();
    Type dtype = IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    if (dim.getType().isa<Torch::NoneType>())
      return visitReductionAlongAllDimsOp(op, dtype, operands);
    if (dim.getType().isa<Torch::IntType>())
      return visitReductionAlongDimIntOp(argmax, argmax.dim(), argmax.keepdim(),
                                         dtype, operands);
  }
  if (auto anyDim = dyn_cast<AtenAnyDimOp>(op)) {
    Type dtype = operands[0]->getValue().dtype;
    return visitReductionAlongDimIntOp(anyDim, anyDim.dim(), anyDim.keepdim(),
                                       dtype, operands);
  }
  if (auto maxDim = dyn_cast<AtenMaxDimOp>(op)) {
    Type firstResDtype = operands[0]->getValue().dtype;
    Type secondResDtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    ChangeResult firstRes = visitReductionAlongDimIntOp(
        maxDim, maxDim.dim(), maxDim.keepdim(), firstResDtype, operands);
    return firstRes |
           visitReductionAlongDimIntOp(maxDim, maxDim.dim(), maxDim.keepdim(),
                                       secondResDtype, operands, /*resNum=*/1);
  }
  if (auto mean = dyn_cast<AtenMeanOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype =
        getDtypeOrDefault(mean.getContext(), mean.dtype(), defaultDtype);
    return visitReductionAlongAllDimsOp(mean, dtype, operands);
  } else if (auto max = dyn_cast<AtenMaxOp>(op)) {
    Type dtype = operands[0]->getValue().dtype;
    return visitReductionAlongAllDimsOp(max, dtype, operands);
  } else if (isa<AtenStdOp, AtenVarOp>(op)) {
    auto input = operands[0]->getValue();
    return visitReductionAlongAllDimsOp(op, input.dtype, operands);
  }

  if (auto tensorFloat = dyn_cast<AtenTensorFloatOp>(op)) {
    return visitScalarToTensorConversionOp<AtenTensorFloatOp>(tensorFloat);
  } else if (auto tensorInt = dyn_cast<AtenTensorIntOp>(op)) {
    return visitScalarToTensorConversionOp<AtenTensorIntOp>(tensorInt);
  } else if (auto tensorBool = dyn_cast<AtenTensorBoolOp>(op)) {
    return visitScalarToTensorConversionOp<AtenTensorBoolOp>(tensorBool);
  }

  if (auto tensor = dyn_cast<AtenTensorOp>(op)) {
    return visitAtenTensorOp(tensor);
  }

  if (auto zeros = dyn_cast<AtenZerosOp>(op)) {
    return visitConstantTensorAllocOp<AtenZerosOp>(zeros, /*dataType=*/{});
  } else if (auto ones = dyn_cast<AtenOnesOp>(op)) {
    return visitConstantTensorAllocOp<AtenOnesOp>(ones, /*dataType=*/{});
  } else if (auto emptyMemoryFormat = dyn_cast<AtenEmptyMemoryFormatOp>(op)) {
    return visitConstantTensorAllocOp<AtenEmptyMemoryFormatOp>(
        emptyMemoryFormat, /*dataType=*/{});
  } else if (auto full = dyn_cast<AtenFullOp>(op)) {
    return visitConstantTensorAllocOp<AtenFullOp>(
        full, /*dataType=*/full.fill_value().getType());
  } else if (auto zerosLike = dyn_cast<AtenZerosLikeOp>(op)) {
    return visitConstantTensorAllocLikeOp<AtenZerosLikeOp>(zerosLike, operands);
  } else if (auto onesLike = dyn_cast<AtenOnesLikeOp>(op)) {
    return visitConstantTensorAllocLikeOp<AtenOnesLikeOp>(onesLike, operands);
  } else if (auto emptyLike = dyn_cast<AtenEmptyLikeOp>(op)) {
    return visitConstantTensorAllocLikeOp<AtenEmptyLikeOp>(emptyLike, operands);
  } else if (auto fullLike = dyn_cast<AtenFullLikeOp>(op)) {
    return visitConstantTensorAllocLikeOp<AtenFullLikeOp>(fullLike, operands);
  } else if (auto newZeros = dyn_cast<AtenNewZerosOp>(op)) {
    return visitConstantTensorNewLikeOp<AtenNewZerosOp>(newZeros, operands);
  } else if (auto newOnes = dyn_cast<AtenNewOnesOp>(op)) {
    return visitConstantTensorNewLikeOp<AtenNewOnesOp>(newOnes, operands);
  } else if (auto newEmpty = dyn_cast<AtenNewEmptyOp>(op)) {
    return visitConstantTensorNewLikeOp<AtenNewEmptyOp>(newEmpty, operands);
  } else if (auto randLike = dyn_cast<AtenRandLikeOp>(op)) {
    return visitConstantTensorAllocLikeOp<AtenRandLikeOp>(randLike, operands);
  } else if (auto toCopy = dyn_cast<Aten_ToCopyOp>(op)) {
    return visitConstantTensorAllocLikeOp<Aten_ToCopyOp>(toCopy, operands);
  }

  if (auto toDtype = dyn_cast<AtenToDtypeOp>(op))
    return visitAtenToDtypeLikeOp<AtenToDtypeOp>(toDtype, operands);

  if (auto toDtypeLayout = dyn_cast<AtenToDtypeLayoutOp>(op))
    return visitAtenToDtypeLikeOp<AtenToDtypeLayoutOp>(toDtypeLayout, operands);

  if (auto toOther = dyn_cast<AtenToOtherOp>(op)) {
    return visitTypeConversionOp<AtenToOtherOp>(toOther, operands);
  } else if (auto typeAs = dyn_cast<AtenTypeAsOp>(op)) {
    return visitTypeConversionOp<AtenTypeAsOp>(typeAs, operands);
  }

  if (auto cat = dyn_cast<AtenCatOp>(op)) {
    return visitAtenCatOp(cat, operands);
  }

  if (auto shapeAsTensor = dyn_cast<Aten_ShapeAsTensorOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    return incorporateKnowledge(shapeAsTensor.getResult(), knowledge);
  }

  if (auto embedding = dyn_cast<AtenEmbeddingOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = Float32Type::get(op->getContext());
    return incorporateKnowledge(embedding.getResult(), knowledge);
  }

  if (auto softmaxIntOp = dyn_cast<AtenSoftmaxIntOp>(op)) {
    return visitAtenSoftmaxLikeOp(softmaxIntOp, operands);
  }
  if (auto _softmaxOp = dyn_cast<Aten_SoftmaxOp>(op)) {
    return visitAten_SoftmaxLikeOp(_softmaxOp, operands);
  } else if (auto _logSoftmaxOp = dyn_cast<Aten_LogSoftmaxOp>(op)) {
    return visitAten_SoftmaxLikeOp(_logSoftmaxOp, operands);
  } else if (auto logSoftmaxIntOp = dyn_cast<AtenLogSoftmaxIntOp>(op)) {
    return visitAtenSoftmaxLikeOp(logSoftmaxIntOp, operands);
  }

  if (auto numToTensorOp = dyn_cast<PrimNumToTensorScalarOp>(op)) {
    return visitNumToTensorOp(numToTensorOp);
  }

  if (isa<AtenAddIntOp, AtenSubIntOp, AtenMulIntOp, AtenAddOp>(op)) {
    return visitBinaryScalarOp(op, operands);
  }

  if (auto scalarImplicit = dyn_cast<AtenScalarImplicitOp>(op))
    return visitAtenScalarImplicitOp(scalarImplicit, operands);

  if (auto vectorNorm = dyn_cast<AtenLinalgVectorNormOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype = getDtypeOrDefault(vectorNorm.getContext(), vectorNorm.dtype(),
                                   defaultDtype);
    return visitReductionAlongDimIntListOp(
        vectorNorm, vectorNorm.dim(), vectorNorm.keepdim(), dtype, operands);
  }

  // Otherwise, this is an unknown operation. Just mark all results as
  // having reached a pessimistic fixpoint.
  return markAllPessimisticFixpoint(op->getResults());
}

ChangeResult
TypeAnalyzer::incorporateKnowledge(Value v, const ValueKnowledge &knowledge) {
  auto updatedKnowledge = ValueKnowledge::meet(
      knowledge, ValueKnowledge::getPessimisticValueState(v));
  assert(updatedKnowledge.hasValue() && "IR has contradictory type!");
  return getLatticeElement(v).join(updatedKnowledge.getValue());
}

ChangeResult TypeAnalyzer::visitAtenLinearOp(
    AtenLinearOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  auto input = operands[0]->getValue();
  auto weight = operands[1]->getValue();
  auto bias = operands[2]->getValue();
  switch (bias.optional) {
  case OptionalKnowledge::isNone:
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&input, &weight});
    break;
  case OptionalKnowledge::notNone:
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&input, &weight, &bias});
    break;
  case OptionalKnowledge::unKnown:
    // When it's unknown, type promotion can't be decided at compile time.
    break;
  }
  return incorporateKnowledge(op->getResult(0), knowledge);
}

// Arange like ops returns a 1-D tensor of size ceil(end - start).
ChangeResult TypeAnalyzer::visitAtenArangeLikeOpHelper(
    Operation *op, llvm::Optional<Value> start, Value end,
    llvm::Optional<Value> step, Value dtype) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  int64_t dtypeInt;
  if (matchPattern(dtype, m_TorchConstantInt(&dtypeInt))) {
    knowledge.dtype = getTypeForDTypeInteger(op->getContext(), dtypeInt);
  } else if (dtype.getType().isa<Torch::NoneType>()) {
    // From torch/_torch_docs.py:
    // If `dtype` is not given, infer the data type from the other input
    // arguments. If any of `start`, `end`, or `step` are floating-point, the
    // `dtype` is inferred to be the default dtype, see
    // `torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
    // be `torch.int64`
    if ((start.hasValue() && (*start).getType().isa<Torch::FloatType>()) ||
        end.getType().isa<Torch::FloatType>() ||
        (step.hasValue() && (*step).getType().isa<Torch::FloatType>())) {
      // TODO: Should get the dtype from torch.get_default_dtype().
      // For now, use float32 which is the initial default dtype.
      knowledge.dtype = Float32Type::get(op->getContext());
    } else
      knowledge.dtype =
          IntegerType::get(op->getContext(), 64, IntegerType::Signed);
  }
  return incorporateKnowledge(op->getResult(0), knowledge);
}

ChangeResult
TypeAnalyzer::visitAtenArangeStartStepOp(AtenArangeStartStepOp op) {
  return visitAtenArangeLikeOpHelper(op, op.start(), op.end(), op.step(),
                                     op.dtype());
}

ChangeResult TypeAnalyzer::visitAtenArangeStartOp(AtenArangeStartOp op) {
  return visitAtenArangeLikeOpHelper(op, op.start(), op.end(), {}, op.dtype());
}

ChangeResult TypeAnalyzer::visitAtenArangeOp(AtenArangeOp op) {
  return visitAtenArangeLikeOpHelper(op, {}, op.end(), {}, op.dtype());
}

ChangeResult TypeAnalyzer::visitReductionAlongAllDimsOp(
    Operation *op, Type dtype,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  return incorporateKnowledge(op->getResult(0), knowledge);
}

// These ops do caculation along the dims given by the integer list and reduce
// each dim to size one. If \p keepdim is false, the dims are squeezed.
ChangeResult TypeAnalyzer::visitReductionAlongDimIntListOp(
    Operation *op, Value dim, Value keepdim, Type dtype,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  return incorporateKnowledge(op->getResult(0), knowledge);
}

ChangeResult TypeAnalyzer::visitReductionAlongDimIntOp(
    Operation *op, Value dim, Value keepdim, Type dtype,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands, int resNum) {
  assert(dim.getType().isa<Torch::IntType>() && "dim must be int type");
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  return incorporateKnowledge(op->getResult(resNum), knowledge);
}

template <typename OpTy>
ChangeResult TypeAnalyzer::visitScalarToTensorConversionOp(OpTy op) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op.getContext());
  Value t = op.t();
  Value dtype = op.dtype();
  fillInDTypeGivenDTypeAndDataType(knowledge, dtype, t.getType());
  return incorporateKnowledge(op.getResult(), knowledge);
}

ChangeResult TypeAnalyzer::visitBinaryScalarOp(
    Operation *op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getScalarPessimisticValueState(op->getContext());
  Type resultType = getPromotedResultScalarType(
      {op->getOperand(0).getType(), op->getOperand(1).getType()});
  knowledge.setScalarType(resultType);
  return incorporateKnowledge(op->getResult(0), knowledge);
}

ChangeResult TypeAnalyzer::visitAtenTensorOp(AtenTensorOp op) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op.getContext());
  Value data = op.data();
  Value dtype = op.dtype();
  Type type = data.getType();
  while (auto listType = type.dyn_cast<ListType>()) {
    type = listType.getContainedType();
  }
  fillInDTypeGivenDTypeAndDataType(knowledge, dtype, type);
  return incorporateKnowledge(op.getResult(), knowledge);
}

template <typename OpTy>
ChangeResult
TypeAnalyzer::visitConstantTensorAllocOp(OpTy op,
                                         llvm::Optional<Type> dataType) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  if (!dataType)
    dataType = Torch::FloatType::get(op->getContext());
  fillInDTypeGivenDTypeAndDataType(knowledge, op.dtype(), dataType.getValue());
  return incorporateKnowledge(op.getResult(), knowledge);
}

template <typename OpTy>
ChangeResult TypeAnalyzer::visitConstantTensorAllocLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, op.dtype(), input.dtype);
  return incorporateKnowledge(op.getResult(), knowledge);
}

template <typename OpTy>
ChangeResult TypeAnalyzer::visitConstantTensorNewLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, op.dtype(), input.dtype);
  return incorporateKnowledge(op.getResult(), knowledge);
}

// Convert input tensor type to the given `dtype`.
template <typename OpTy>
ChangeResult TypeAnalyzer::visitAtenToDtypeLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  Value dtype = op.dtype();
  int64_t dtypeInt;
  if (matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    knowledge.dtype = getTypeForDTypeInteger(op->getContext(), dtypeInt);
  return incorporateKnowledge(op.getResult(), knowledge);
}

// Convert input tensor type to the same as the other tensor.
template <typename OpTy>
ChangeResult TypeAnalyzer::visitTypeConversionOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  Value other = op.other();
  BaseTensorType type = other.getType().cast<BaseTensorType>();
  if (type.hasDtype())
    knowledge.dtype = type.getDtype();
  return incorporateKnowledge(op->getResult(0), knowledge);
}

// `torch.aten.cat` concatenates the given sequence of seq tensors in the given
// dimension. The output has the same sizes as the input for all dimensions
// except the given dimension.
ChangeResult TypeAnalyzer::visitAtenCatOp(
    AtenCatOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto tensorList = op.tensors();
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  auto listConstruct = tensorList.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return incorporateKnowledge(op.getResult(), knowledge);

  auto tensors = llvm::to_vector<4>(
      llvm::map_range(listConstruct.elements(), [&](Value v) -> ValueKnowledge {
        return getLatticeElement(v).getValue();
      }));
  for (auto tensor : tensors) {
    auto newDtype = meetElementTypes(knowledge.dtype, tensor.dtype);
    if (!newDtype.hasValue())
      return incorporateKnowledge(op.getResult(), knowledge);
    knowledge.dtype = newDtype.getValue();
  }
  return incorporateKnowledge(op.getResult(), knowledge);
}

ChangeResult TypeAnalyzer::visitNumToTensorOp(PrimNumToTensorScalarOp op) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  // The resulting type from converting a Scalar into a Tensor is different
  // if the scalar is part of a tensor operation (such as AtenMulScalar) or
  // not. In the former case, the type promotion rules are captured by the
  // `getDefaultDtypeForTorchScalar` helper above. The latter case follows the
  // rules in
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/ScalarOps.h.
  // `NumToTensor` falls in the latter case.
  Type type = op.a().getType();
  knowledge.dtype = getBuiltInTypeForTorchScalar(type);
  return incorporateKnowledge(op.getResult(), knowledge);
}

// Common template for softmax like ops, eg., log_softmax.
template <typename OpTy>
ChangeResult TypeAnalyzer::visitAtenSoftmaxLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto dtype = op.dtype();
  ValueKnowledge knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, dtype, input.dtype);
  return incorporateKnowledge(op.getResult(), knowledge);
}

// Common template for softmax like ops, eg., log_softmax.(underscore variant)
template <typename OpTy>
ChangeResult TypeAnalyzer::visitAten_SoftmaxLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  ValueKnowledge knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  bool halfToFloat;
  if (matchPattern(op.half_to_float(), m_TorchConstantBool(&halfToFloat))) {
    knowledge.dtype =
        halfToFloat ? Float32Type::get(op->getContext()) : input.dtype;
  }
  return incorporateKnowledge(op.getResult(), knowledge);
}

ChangeResult TypeAnalyzer::visitAtenScalarImplicitOp(
    AtenScalarImplicitOp op,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getScalarPessimisticValueState(op.getContext());
  Type dType = operands[0]->getValue().dtype;
  if (dType.isa<mlir::FloatType>())
    knowledge.setScalarType(Torch::FloatType::get(op->getContext()));
  else if (dType.isa<mlir::IntegerType>())
    knowledge.setScalarType(Torch::IntType::get(op->getContext()));
  return incorporateKnowledge(op->getResult(0), knowledge);
}

// -----------------------------------------------------------------------------
// Transforms.
// -----------------------------------------------------------------------------

// Get a the most refined type compatible with ValueKnowledge, or null if that
// is not possible.
static Type getMostRefinedStaticType(Value v, TypeAnalyzer &analyzer) {
  auto getRefinedTensorType = [](BaseTensorType tensorType,
                                 ValueKnowledge const &knowledge) {
    return tensorType
        .getWithSizesAndDtype(tensorType.getOptionalSizes(), knowledge.dtype)
        .cast<BaseTensorType>();
  };

  if (auto tensorType = v.getType().dyn_cast<BaseTensorType>()) {
    LatticeElement<ValueKnowledge> *latticeElement =
        analyzer.lookupLatticeElement(v);
    if (!latticeElement)
      return nullptr;
    const ValueKnowledge &knowledge = latticeElement->getValue();
    return getRefinedTensorType(tensorType, knowledge);
  } else if (auto optionalType = v.getType().dyn_cast<OptionalType>()) {
    LatticeElement<ValueKnowledge> *latticeElement =
        analyzer.lookupLatticeElement(v);
    if (!latticeElement)
      return nullptr;
    const ValueKnowledge &knowledge = latticeElement->getValue();
    if (knowledge.optional == OptionalKnowledge::isNone)
      return Torch::NoneType::get(v.getContext());
    else if (knowledge.optional == OptionalKnowledge::notNone) {
      auto containedType = optionalType.getContainedType();
      if (auto tensorType = containedType.dyn_cast<BaseTensorType>())
        return getRefinedTensorType(tensorType, knowledge);
      else
        return containedType;
    }
  } else if (auto scalarType = v.getType().dyn_cast<NumberType>()) {
    LatticeElement<ValueKnowledge> *latticeElement =
        analyzer.lookupLatticeElement(v);
    if (!latticeElement)
      return nullptr;
    const ValueKnowledge &knowledge = latticeElement->getValue();
    if (knowledge.kind == torch_upstream::TypeKind::IntType)
      return Torch::IntType::get(v.getContext());
    if (knowledge.kind == torch_upstream::TypeKind::FloatType)
      return Torch::FloatType::get(v.getContext());
  }
  return nullptr;
}

// Return true if we can safely change the operands or results of `op`.
//
// The most trivial case is when the op has the AllowsTypeRefinement trait,
// which allows arbitrary refinements. But some other cases are safe too,
// such as when an op has two types that are coupled, but we know that our
// analysis and updating logic will correctly maintain the invariants of the op.
// The `torch.copy.to_tensor` / `torch.copy.to_vtensor` are examples of the
// latter case, since their operand and result types must have the same shape
// and dtype -- we know that our transfer functions and updating logic will do
// the right thing forthose ops.
//
static bool allowsTypeRefinementOrIsSafeToRefine(Operation *op) {
  return op->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>() ||
         isa<CopyToNonValueTensorOp, CopyToValueTensorOp>(op);
}

// Some operations have extra verification logic regarding the relationship
// between the input types and output types. Adding more refined type info to
// the operand might change a valid instruction to be invalid.
static bool operationIsValidWithRefinedType(OpOperand *use, Type newType) {
  Operation *op = use->getOwner();
  if (auto uncheckedCast = llvm::dyn_cast<PrimUncheckedCastOp>(op))
    return uncheckedCast.areCastCompatible(newType, uncheckedCast.getType());
  return true;
}

static bool isSafeToRefineOperandInPlace(OpOperand *use, Type newOperandType) {
  Operation *op = use->getOwner();
  if (!allowsTypeRefinementOrIsSafeToRefine(op))
    return false;
  return operationIsValidWithRefinedType(use, newOperandType);
}

void optimize(func::FuncOp func, TypeAnalyzer &analyzer) {
  func.walk([&](Operation *op) {
    auto convertValuesToMostRefinedType = [&](ValueRange values, OpBuilder &b) {
      for (Value v : values) {
        Type refinedType = getMostRefinedStaticType(v, analyzer);
        Type originalType = v.getType();
        // No type? Nothing to do.
        if (!refinedType)
          continue;
        // Type is same as existing one? Nothing to do.
        if (refinedType == originalType)
          continue;
        // If we have an op that allows adding/removing static information from
        // this type, then we can rewrite. We make sure to always embed the
        // static information in the IR, and insert the minimal number of casts
        // needed to do so.
        using CreateStaticInfoCastFn =
            std::function<Value(Location, Type, Value)>;
        CreateStaticInfoCastFn createStaticInfoDownCast;
        CreateStaticInfoCastFn createStaticInfoUpCast;
        if (originalType.isa<BaseTensorType>()) {
          createStaticInfoDownCast = [&](Location loc, Type newType,
                                         Value v) -> Value {
            return b.create<TensorStaticInfoCastOp>(loc, newType, v);
          };
          createStaticInfoUpCast = createStaticInfoDownCast;
        } else if (originalType.isa<OptionalType, NumberType>()) {
          createStaticInfoDownCast = [&](Location loc, Type newType,
                                         Value v) -> Value {
            return b.create<PrimUncheckedCastOp>(loc, newType, v);
          };
          createStaticInfoUpCast = [&](Location loc, Type newType,
                                       Value v) -> Value {
            return b.create<DerefineOp>(loc, newType, v);
          };
        }

        if (createStaticInfoUpCast) {
          assert(createStaticInfoDownCast &&
                 "createStaticInfoDownCast and createStaticInfoUpCast must be "
                 "defined in pairs");
          // Save off the original uses to avoid iterator invalidation issues
          // or other unexpected behavior since we are creating new ops here
          // that use the value.
          auto originalUses = llvm::to_vector<6>(llvm::map_range(
              v.getUses(), [](OpOperand &use) { return &use; }));
          OpBuilder b(op->getBlock(), std::next(op->getIterator()));
          Value newTypedValue;
          // Always make sure that the new static information is reflected in
          // the IR, either by updating the type in place, or inserting a static
          // info cast.
          if (allowsTypeRefinementOrIsSafeToRefine(op)) {
            newTypedValue = v;
            v.setType(refinedType);
          } else {
            if (auto derefineOp = llvm::dyn_cast<DerefineOp>(op)) {
              newTypedValue = derefineOp.operand();
            } else {
              newTypedValue =
                  createStaticInfoDownCast(op->getLoc(), refinedType, v);
            }
          }

          Value oldTypedValue;
          for (OpOperand *use : originalUses) {
            // If the use can be updated to the new type directly, do it!
            if (isSafeToRefineOperandInPlace(use, refinedType)) {
              use->set(newTypedValue);
              continue;
            } else if (auto overwriteTensorContents =
                           dyn_cast<OverwriteTensorContentsOp>(
                               use->getOwner())) {
              // `OverwriteTensorContentsOp` has special handling here because
              // it requires that both of its operands always have the same
              // shape and dtype.
              //
              // WARNING: In order to simplify the implementation, the type
              // used for both operands is the type of the overwritten tensor.
              // A better way of doing this would be to join the two operand
              // types to create the most specific type possible and use that
              // for both arguments, allowing static sizes to always propagate.
              const unsigned overwriterOperandIndex = 0;
              const unsigned overwrittenOperandIndex = 1;
              unsigned operandNumber = use->getOperandNumber();
              if (operandNumber != overwrittenOperandIndex)
                continue;

              Location loc = overwriteTensorContents.getLoc();
              Value overwriterTensor = overwriteTensorContents.value();
              Type overwriterTensorType = overwriterTensor.getType();
              Type overwrittenTensorType = newTypedValue.getType()
                                               .dyn_cast<NonValueTensorType>()
                                               .getWithValueSemantics();
              if (overwriterTensorType == overwrittenTensorType)
                continue;

              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPoint(overwriteTensorContents);
                Value castedOverwriterTensor = b.create<TensorStaticInfoCastOp>(
                    loc, overwrittenTensorType, overwriterTensor);
                overwriteTensorContents.setOperand(overwriterOperandIndex,
                                                   castedOverwriterTensor);
              }
              continue;
            }

            // If needed, create a value of the original type to appease users
            // that cannot accept the new type.
            if (!oldTypedValue) {
              if (auto derefineOp = llvm::dyn_cast<DerefineOp>(op)) {
                oldTypedValue = derefineOp.result();
              } else {
                oldTypedValue = createStaticInfoUpCast(
                    op->getLoc(), originalType, newTypedValue);
              }
            }
            use->set(oldTypedValue);
          }
        }
      }
    };

    if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      for (auto &region : branch->getRegions()) {
        OpBuilder b(region);
        convertValuesToMostRefinedType(region.front().getArguments(), b);
      }
    }
    OpBuilder b(op->getBlock(), std::next(op->getIterator()));
    convertValuesToMostRefinedType(op->getResults(), b);
  });
}

namespace {
class RefineTypesPass : public RefineTypesBase<RefineTypesPass> {
  void runOnOperation() override {
    auto func = getOperation();
    TypeAnalyzer analyzer(&getContext());
    analyzer.run(func);
    optimize(func, analyzer);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createRefineTypesPass() {
  return std::make_unique<RefineTypesPass>();
}
