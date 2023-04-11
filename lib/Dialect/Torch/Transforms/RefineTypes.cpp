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

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
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
  FailureOr<Type> result =
      getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
  return failed(result) ? Type() : *result;
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


enum class OptionalKnowledge {
  unKnown,
  isNone,
  notNone,
};

/// Returns the OptionalKnowledge that assumes information from both `lhs` and
/// `rhs`. Returns `std::nullopt` if the knowledges are contradictory.
static std::optional<OptionalKnowledge>
meetOptionalKnowledge(OptionalKnowledge lhs, OptionalKnowledge rhs) {
  if (lhs == OptionalKnowledge::unKnown)
    return rhs;
  if (rhs == OptionalKnowledge::unKnown)
    return lhs;
  if (lhs == rhs)
    return lhs;
  return std::nullopt;
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
  ValueKnowledge() = default;
  ValueKnowledge(Type dtype, Type scalarType,
                 OptionalKnowledge optionalKnowledge,
                 torch_upstream::TypeKind kind)
      : isInitialized(true), dtype(dtype), scalarType(scalarType), kind(kind),
        optional(optionalKnowledge) {}

  void print(raw_ostream &os) const {
    os << "ValueKnowledge(";
    if (!isInitialized) {
      os << "uninitialized)";
      return;
    }
    if (dtype)
      os << "dtype=" << dtype;
    if (scalarType)
      os << ", scalarType=" << scalarType;
    if (optional != OptionalKnowledge::unKnown)
      os << ", optional=" << (int)optional;
    os << ", kind=" << (int)kind << ")";
  }
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
    if (!isInitialized && !rhs.isInitialized)
      return true;
    return isInitialized && rhs.isInitialized &&
           std::make_tuple(dtype, optional) ==
               std::make_tuple(rhs.dtype, rhs.optional);
  }

  // Return true if the `refinedType` has more concrete type info than `type`.
  static bool hasStrictlyMoreRefinedTypeInfo(const ValueKnowledge &refinedType,
                                             const ValueKnowledge &type) {
    if (!refinedType.isInitialized)
      return false;
    if (!type.isInitialized)
      return true;

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
    if (!lhs.isInitialized)
      return rhs;
    if (!rhs.isInitialized)
      return lhs;

    // Mental model: All conditions are checking how to change from the safe "no
    // knowledge" default-initialized state to a state with more knowledge
    // consistent with lhs and rhs.
    ValueKnowledge result = joinTypes(lhs, rhs);
    result.optional = joinOptionalKnowledge(lhs.optional, rhs.optional);
    return result;
  }

  static ValueKnowledge joinTypes(const ValueKnowledge &lhs,
                                  const ValueKnowledge &rhs) {
    if (!lhs.isInitialized)
      return rhs;
    if (!rhs.isInitialized)
      return lhs;

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
  // If the two pieces of knowledge are contradictory, std::nullopt is returned.
  static std::optional<ValueKnowledge> meet(const ValueKnowledge &lhs,
                                            const ValueKnowledge &rhs) {
    if (!lhs.isInitialized)
      return lhs;
    if (!rhs.isInitialized)
      return rhs;

    std::optional<ValueKnowledge> knowledge = meetTypes(lhs, rhs);

    if (!knowledge.has_value())
      return std::nullopt;
    ValueKnowledge result = knowledge.value();

    std::optional<OptionalKnowledge> optional =
        meetOptionalKnowledge(lhs.optional, rhs.optional);
    if (!optional.has_value())
      return std::nullopt;
    result.optional = optional.value();
    return result;
  }

  static std::optional<ValueKnowledge> meetTypes(const ValueKnowledge &lhs,
                                                 const ValueKnowledge &rhs) {
    if (!lhs.isInitialized)
      return lhs;
    if (!rhs.isInitialized)
      return rhs;

    if (hasStrictlyMoreRefinedTypeInfo(lhs, rhs))
      return lhs;
    if (hasStrictlyMoreRefinedTypeInfo(rhs, lhs))
      return rhs;
    if (lhs == rhs)
      return lhs;
    return std::nullopt;
  }

  // We start in the uninitialized state by default.
  bool isInitialized = false;

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
} // namespace

using ValueState = dataflow::Lattice<ValueKnowledge>;
// Register TypeID for the dataflow framework.
MLIR_DECLARE_EXPLICIT_TYPE_ID(ValueState)
MLIR_DEFINE_EXPLICIT_TYPE_ID(ValueState)

namespace {
// Forward intraprocedural dataflow for type information.
class TypeAnalysis : public dataflow::SparseDataFlowAnalysis<
                         dataflow::Lattice<ValueKnowledge>> {
public:
  using BaseT =
      dataflow::SparseDataFlowAnalysis<dataflow::Lattice<ValueKnowledge>>;
  using BaseT::SparseDataFlowAnalysis;

  // Compute the knowledge for the results of an op, based on the knowledge of
  // the operands and any information intrinsic to `op`.
  void visitOperation(Operation *op, ArrayRef<const ValueState *> operands,
                      ArrayRef<ValueState *> results) final;

  void setToEntryState(ValueState *lattice) override {
    auto refType = lattice->getPoint().getType();
    auto knowledge = ValueKnowledge::getKnowledgeFromType(refType);
    propagateIfChanged(lattice, lattice->join(knowledge));
  }

private:
  // Get the MLIR type of the tensor dtype given the dtype integer value and the
  // input dtype. When DType is None the type is inferred from the input dtype.
  void fillInDTypeGivenDTypeIntAndInputDType(ValueKnowledge &knowledge,
                                             Value dtype, Type inputDType);

  // Get the MLIR type of the tensor dtype given the dtype integer value and
  // data type of torch type. When DType is None the type is inferred from the
  // data type.
  void fillInDTypeGivenDTypeAndDataType(ValueKnowledge &knowledge, Value dtype,
                                        Type dataType);

  /// Incorporates `knowledge` into the lattice state of `v`.
  ///
  /// This method should be used instead of
  /// `getLatticeElement(v).join(knowledge)`, because this method knows how to
  /// correctly handle the case of existing static knowledge from the type
  /// of `v`.
  void incorporateKnowledge(Value v, const ValueKnowledge &knowledge);

  void visitAtenLinearOp(AtenLinearOp op,
                         ArrayRef<const ValueState *> operands);
  void visitAtenArangeStartStepOp(AtenArangeStartStepOp op);
  void visitAtenArangeStartOp(AtenArangeStartOp op);
  void visitAtenArangeOp(AtenArangeOp op);
  void visitAtenArangeLikeOpHelper(Operation *op, std::optional<Value> start,
                                   Value end, std::optional<Value> step,
                                   Value dtype);
  void visitReductionAlongAllDimsOp(Operation *op, Type dtype,
                                    ArrayRef<const ValueState *> operands);
  void visitReductionAlongDimIntListOp(Operation *op, Value dim, Value keepdim,
                                       Type dtype,
                                       ArrayRef<const ValueState *> operands);
  void visitReductionAlongDimIntOp(Operation *op, Value dim, Value keepdim,
                                   Type dtype,
                                   ArrayRef<const ValueState *> operands,
                                   int resNum = 0);
  template <typename OpTy> void visitScalarToTensorConversionOp(OpTy op);
  void visitAtenTensorOp(AtenTensorOp op);
  template <typename OpTy>
  void visitConstantTensorAllocOp(OpTy op, std::optional<Type> dataType);
  template <typename OpTy>
  void visitConstantTensorAllocLikeOp(OpTy op,
                                      ArrayRef<const ValueState *> operands);
  template <typename OpTy>
  void visitConstantTensorNewLikeOp(OpTy op,
                                    ArrayRef<const ValueState *> operands);
  template <typename OpTy>
  void visitAtenToDtypeLikeOp(OpTy op, ArrayRef<const ValueState *> operands);
  template <typename OpTy>
  void visitTypeConversionOp(OpTy op, ArrayRef<const ValueState *> operands);
  template <typename OpTy>
  void visitAtenCatLikeOp(OpTy op, ArrayRef<const ValueState *> operands);

  template <typename OpTy>
  void visitAtenSoftmaxLikeOp(OpTy op, ArrayRef<const ValueState *> operands);
  template <typename OpTy>
  void visitAten_SoftmaxLikeOp(OpTy op, ArrayRef<const ValueState *> operands);

  void visitNumToTensorOp(PrimNumToTensorScalarOp op);
  void visitBinaryScalarOp(Operation *op,
                           ArrayRef<const ValueState *> operands);
  void visitAtenScalarImplicitOp(AtenScalarImplicitOp op,
                                 ArrayRef<const ValueState *> operands);
  void visitAtenEmbeddingBagOp(Operation *op);
};
} // namespace

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
updateResultTypeState(const ValueKnowledge *tensor,
                      std::optional<bool> rankIsNonZero,
                      const torch_upstream::ResultTypeState &inState,
                      bool skipRankCheck = false) {
  if (!rankIsNonZero.has_value() && !skipRankCheck)
    return torch_upstream::ResultTypeState{};
  assert(tensor->dtype && "tensor.dtype must be not none");

  torch_upstream::ResultTypeState new_state = inState;
  torch_upstream::ScalarType current = getScalarTypeForType(tensor->dtype);
  if (skipRankCheck || rankIsNonZero.value())
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
  FailureOr<Type> result = getTorchTypeForScalarType(
      scalarTypes[0].getContext(), result_type(state));
  if (failed(result))
    return Type();
  return *result;
}

// Returns most generic type Type() if the tensor dtype is unknown.
static Type getPromotedResultDType(ValueKnowledge *tensor, Type scalarType) {
  if (!tensor->dtype)
    return Type();
  torch_upstream::ResultTypeState state = {};
  // No need to check if rank is zero for tensor because scalar uses
  // wrappedResult which is a lower priority than both dimResult and zeroResult.
  state = updateResultTypeState(tensor, /*rankIsNonZero=*/std::nullopt, state,
                                /*skipRankCheck=*/true);
  state =
      updateResultTypeState(getDefaultDtypeForTorchScalar(scalarType), state);
  FailureOr<Type> result =
      getTypeForScalarType(scalarType.getContext(), result_type(state));
  return failed(result) ? Type() : *result;
}

static SmallVector<std::optional<bool>>
getRankIsNonZeroArray(ValueRange values) {
  SmallVector<std::optional<bool>> rankIsNonZero;
  for (Value v : values) {
    if (auto tensorType = v.getType().dyn_cast<BaseTensorType>()) {
      if (tensorType.hasSizes()) {
        rankIsNonZero.push_back(tensorType.getSizes().size() != 0);
      } else {
        rankIsNonZero.push_back(std::nullopt);
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
                                  ArrayRef<const ValueKnowledge *> tensors,
                                  ArrayRef<std::optional<bool>> rankIsNonZero,
                                  bool skipRankCheck = false) {
  torch_upstream::ResultTypeState state = {};
  assert(tensors.size() == rankIsNonZero.size());
  for (auto t : llvm::zip(tensors, rankIsNonZero)) {
    const ValueKnowledge *tensor = std::get<0>(t);
    std::optional<bool> rankIsNonZero = std::get<1>(t);
    if (!tensor->dtype)
      return Type();
    state = updateResultTypeState(tensor, rankIsNonZero, state, skipRankCheck);
  }
  FailureOr<Type> result = getTypeForScalarType(context, result_type(state));
  return failed(result) ? Type() : *result;
}

static Type getPromotedResultTypeAssumingNonZeroRank(
    MLIRContext *context, ArrayRef<const ValueKnowledge *> tensors) {
  SmallVector<std::optional<bool>> rankIsNonZero(tensors.size(), true);
  return getPromotedResultType(context, tensors, rankIsNonZero,
                               /*skipRankCheck=*/true);
}

void TypeAnalysis::fillInDTypeGivenDTypeIntAndInputDType(
    ValueKnowledge &knowledge, Value dtype, Type inputDType) {
  assert(!inputDType ||
         isBuiltInType(inputDType) && "`inputDType` must be a builtin type");
  int64_t dtypeInt;
  if (dtype.getType().isa<Torch::NoneType>())
    knowledge.dtype = inputDType;
  else if (matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    knowledge.dtype = getTypeForDTypeInteger(dtype.getContext(), dtypeInt);
  else if (auto primDtypeOp = dyn_cast<PrimDtypeOp>(dtype.getDefiningOp()))
    knowledge.dtype = getLatticeElement(primDtypeOp.getA())->getValue().dtype;
}

void TypeAnalysis::fillInDTypeGivenDTypeAndDataType(ValueKnowledge &knowledge,
                                                    Value dtype,
                                                    Type dataType) {
  assert(isa<TorchDialect>(dataType.getDialect()) &&
         "`dataType` must be a torch type");
  Type dtypeForDataType = getDefaultDtypeForTorchScalar(dataType);
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, dtype, dtypeForDataType);
}

void TypeAnalysis::visitOperation(Operation *op,
                                  ArrayRef<const ValueState *> operands,
                                  ArrayRef<ValueState *> results) {

  // These ops have results that are dynamically the same as their operands.
  if (isa<TensorStaticInfoCastOp, DerefineOp>(op)) {
    incorporateKnowledge(op->getResult(0), operands[0]->getValue());
    return;
  }

  // Take dtype from first operand.
  if (isa<CopyToValueTensorOp, CopyToNonValueTensorOp, AtenBatchNormOp,
          AtenReluOp, AtenRelu6Op, AtenGeluOp, AtenCeilOp, AtenGeluBackwardOp,
          AtenBitwiseNotOp, AtenToPrimDeviceOp, AtenCpuOp, AtenContiguousOp,
          AtenDetachOp, AtenMaskedFill_ScalarOp, AtenCopyOp, AtenCumsumOp,
          AtenLayerNormOp, AtenClampOp, AtenClampMinOp, AtenClampMaxOp,
          AtenNegOp, AtenFloorOp, Aten_SoftmaxBackwardDataOp, AtenDropoutOp,
          AtenTanhBackwardOp, AtenHardtanhBackwardOp,
          Aten_LogSoftmaxBackwardDataOp, AtenAddIntOp, AtenAbsOp,
          AtenThresholdOp, AtenSquareOp, AtenUniformOp, AtenBernoulliOp,
          AtenBernoulli_FloatOp, AtenBernoulliTensorOp,
          ValsemVariantAtenBernoulliFloatOp, AtenBernoulliTensorOp,
          AtenBernoulliPOp, AtenFillScalarOp, AtenHardsigmoidOp, AtenCloneOp,
          AtenHardswishOp, AtenSiluOp, AtenHardtanhOp, AtenMaskedSelectOp,
          AtenMaxPool2dOp, AtenAvgPool2dOp, AtenAdaptiveAvgPool2dOp,
          AtenFlattenUsingIntsOp, AtenSqueezeOp, AtenSqueezeDimOp,
          AtenUnsqueezeOp, AtenViewOp, Aten_UnsafeViewOp, AtenReshapeOp,
          Aten_ReshapeAliasOp, AtenResize_Op, AtenTransposeIntOp, AtenTOp,
          AtenPermuteOp, AtenIndexSelectOp, AtenSelectIntOp,
          AtenSelectScatterOp, AtenNarrowOp, AtenSliceTensorOp,
          AtenScatterReduceTwoOp, AtenSliceScatterOp, AtenGatherOp,
          AtenExpandOp, AtenExpandAsOp, AtenBroadcastToOp, AtenRepeatOp,
          AtenConstantPadNdOp, AtenPadOp, AtenZero_Op, AtenIndexTensorOp,
          Aten_IndexPutImplOp, AtenIndexPutOp, AtenCopyOp, AtenZeroOp,
          AtenIndexPutHackedTwinOp, AtenPreluOp, AtenMaskedFillScalarOp,
          AtenFlipOp, PrimAbsScalarOp, AtenNumpyTOp, AtenTriuOp,
          AtenMaskedFillTensorOp, AtenRollOp, AtenPowTensorTensorOp,
          AtenLiftFreshCopyOp, AtenIndexTensorHackedTwinOp,
          AtenUpsampleNearest2dOp, AtenMishOp, AtenRoundOp, AtenFillTensorOp,
          AtenUpsampleNearest2dBackwardOp, AtenLeakyReluBackwardOp,
          PrimsSqueezeOp, AtenOneHotOp>(op)) {
    return incorporateKnowledge(op->getResult(0), operands[0]->getValue());
  }

  // Dtype is always float32, except for bfloat16, float16, float64 and nullptr.
  if (isa<AtenTanhOp, AtenExpOp, AtenSinOp, AtenCosOp, AtenSigmoidOp,
          AtenReciprocalOp, AtenLogOp, AtenSqrtOp, AtenLog2Op, AtenLog1pOp,
          AtenRsqrtOp, AtenErfOp, AtenSoftplusOp, AtenFrobeniusNormDimOp,
          PrimsSqrtOp>(op)) {
    ValueKnowledge knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    Type dtype = operands[0]->getValue().dtype;
    if (dtype) {
      knowledge.dtype = Float32Type::get(op->getContext());
      if (dtype.isa<BFloat16Type, Float16Type, Float64Type>())
        knowledge.dtype = dtype;
    }
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Take dtype from second operand.
  if (isa<AtenNllLossBackwardOp, AtenMaxPool2dWithIndicesBackwardOp>(op)) {
    auto self = operands[1]->getValue();
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = self.dtype;
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Dtype is always i1.
  if (isa<AtenEqScalarOp, AtenGeScalarOp, AtenGtScalarOp, AtenLtScalarOp,
          AtenLeScalarOp, AtenNeScalarOp, AtenAnyOp, AtenAllOp, AtenEqTensorOp,
          AtenGtTensorOp, AtenGeTensorOp, AtenLtTensorOp, AtenLeTensorOp,
          AtenLogicalOrOp, AtenLogicalAndOp, AtenLogicalXorOp,
          AtenLogicalNotOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = IntegerType::get(op->getContext(), 1);
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Dtype is always si64.
  if (isa<AtenBincountOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Promote the two dtypes assuming non-zero rank.
  if (isa<AtenMmOp, AtenBmmOp, AtenMatmulOp, AtenConv2dOp, AtenConvolutionOp,
          Aten_ConvolutionOp, AtenMvOp, AtenConvTranspose2dInputOp,
          AtenMseLossOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue()});
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Promote the two dtypes assuming possibly-zero rank.
  if (isa<AtenAddTensorOp, AtenSubTensorOp, AtenMulTensorOp, AtenDivTensorOp,
          AtenDivTensorModeOp, Aten__And__TensorOp, AtenMinimumOp,
          AtenMaximumOp, AtenBitwiseAndTensorOp, AtenBitwiseOrTensorOp,
          AtenBitwiseXorTensorOp, AtenThresholdBackwardOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultType(
        op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue()},
        getRankIsNonZeroArray(op->getOperands()));
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Dtype is always float32, except for bfloat16, float64 and nullptr after
  // promotion and assuming possible-zero rank.
  if (isa<AtenAtan2Op>(op)) {
    ValueKnowledge knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    Type promotedDtype = getPromotedResultType(
        op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue()},
        getRankIsNonZeroArray(op->getOperands()));
    if (promotedDtype) {
      knowledge.dtype = Float32Type::get(op->getContext());
      if (promotedDtype.isa<BFloat16Type, Float64Type>())
        knowledge.dtype = promotedDtype;
    }
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Promote three dtypes.
  if (isa<AtenAddmmOp, AtenLerpTensorOp, AtenAddcmulOp, AtenAddcdivOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue(),
                           &operands[2]->getValue()});
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  if (auto linear = llvm::dyn_cast<AtenLinearOp>(op)) {
    visitAtenLinearOp(linear, operands);
    return;
  }

  // Promote LHS with scalar RHS.
  if (isa<AtenAddScalarOp, AtenSubScalarOp, AtenMulScalarOp, AtenDivScalarOp,
          AtenFmodScalarOp, AtenFloorDivideScalarOp, AtenPowTensorScalarOp,
          AtenLeakyReluOp, AtenRemainderScalarOp>(op)) {
    auto lhs = operands[0]->getValue();
    Value scalar = op->getOperand(1);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultDType(&lhs, scalar.getType());
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereSelfOp, AtenBaddbmmOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultType(
        op->getContext(), {&operands[1]->getValue(), &operands[2]->getValue()},
        getRankIsNonZeroArray(op->getOperands().slice(1, 2)));
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereScalarOp>(op)) {
    Value lhsScalar = op->getOperand(1);
    Value rhsScalar = op->getOperand(2);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getDefaultDtypeForTorchScalar(getPromotedResultScalarType(
        {lhsScalar.getType(), rhsScalar.getType()}));
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereScalarOtherOp>(op)) {
    auto lhs = operands[1]->getValue();
    Value scalar = op->getOperand(2);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultDType(&lhs, scalar.getType());
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }

  // Promote 2nd and 3rd operands.
  if (isa<AtenWhereScalarSelfOp>(op)) {
    auto rhs = operands[2]->getValue();
    Value scalar = op->getOperand(1);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = getPromotedResultDType(&rhs, scalar.getType());
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
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
    incorporateKnowledge(op->getResult(0), result0Knowledge);
    incorporateKnowledge(op->getResult(1), result1Knowledge);
    return;
  }

  // 3 results take dtype from first operand.
  if (isa<AtenNativeLayerNormOp, AtenNativeBatchNormOp,
          AtenConvolutionBackwardOp>(op)) {
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
    incorporateKnowledge(op->getResult(0), result0Knowledge);
    incorporateKnowledge(op->getResult(1), result1Knowledge);
    incorporateKnowledge(op->getResult(2), result1Knowledge);
    return;
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
    incorporateKnowledge(op->getResult(0), result0Knowledge);
    incorporateKnowledge(op->getResult(1), result1Knowledge);
    return;
  }

  if (auto arange = dyn_cast<AtenArangeOp>(op)) {
    visitAtenArangeOp(arange);
    return;
  }
  if (auto arangeStart = dyn_cast<AtenArangeStartOp>(op)) {
    visitAtenArangeStartOp(arangeStart);
    return;
  }
  if (auto arangeStartStep = dyn_cast<AtenArangeStartStepOp>(op)) {
    visitAtenArangeStartStepOp(arangeStartStep);
    return;
  }

  if (auto sum = dyn_cast<AtenSumOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    if (!defaultDtype) {
      incorporateKnowledge(
          sum.getResult(),
          ValueKnowledge::getTensorPessimisticValueState(op->getContext()));
      return;
    }

    // If the input dtype is bool, the result type should be i64.
    if (defaultDtype.isInteger(1))
      defaultDtype =
          IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    Type dtype = getDtypeOrDefault(sum.getContext(), sum.getDtype(), defaultDtype);
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = dtype;
    incorporateKnowledge(op->getResult(0), knowledge);
    return;
  }
  if (auto sumDimIntList = dyn_cast<AtenSumDimIntListOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    if (!defaultDtype) {
      incorporateKnowledge(
          sumDimIntList.getResult(),
          ValueKnowledge::getTensorPessimisticValueState(op->getContext()));
      return;
    }
    // If the input dtype is bool, the result type should be i64.
    if (defaultDtype.isInteger(1))
      defaultDtype =
          IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    Type dtype = getDtypeOrDefault(sumDimIntList.getContext(),
                                   sumDimIntList.getDtype(), defaultDtype);
    visitReductionAlongDimIntListOp(sumDimIntList, sumDimIntList.getDim(),
                                    sumDimIntList.getKeepdim(), dtype, operands);
    return;
  }
  if (auto meanDim = dyn_cast<AtenMeanDimOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype =
        getDtypeOrDefault(meanDim.getContext(), meanDim.getDtype(), defaultDtype);
    visitReductionAlongDimIntListOp(meanDim, meanDim.getDim(), meanDim.getKeepdim(),
                                    dtype, operands);
    return;
  }
  if (auto argmax = dyn_cast<AtenArgmaxOp>(op)) {
    Value dim = argmax.getDim();
    Type dtype = IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    if (dim.getType().isa<Torch::NoneType>()) {
      visitReductionAlongAllDimsOp(op, dtype, operands);
      return;
    }
    if (dim.getType().isa<Torch::IntType>()) {
      visitReductionAlongDimIntOp(argmax, argmax.getDim(), argmax.getKeepdim(), dtype,
                                  operands);
      return;
    }
  }
  if (auto anyDim = dyn_cast<AtenAnyDimOp>(op)) {
    Type dtype = operands[0]->getValue().dtype;
    visitReductionAlongDimIntOp(anyDim, anyDim.getDim(), anyDim.getKeepdim(), dtype,
                                operands);
    return;
  }
  if (auto maxDim = dyn_cast<AtenMaxDimOp>(op)) {
    Type firstResDtype = operands[0]->getValue().dtype;
    Type secondResDtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    visitReductionAlongDimIntOp(maxDim, maxDim.getDim(), maxDim.getKeepdim(),
                                firstResDtype, operands);
    visitReductionAlongDimIntOp(maxDim, maxDim.getDim(), maxDim.getKeepdim(),
                                secondResDtype, operands, /*resNum=*/1);
    return;
  }
  if (auto mean = dyn_cast<AtenMeanOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype =
        getDtypeOrDefault(mean.getContext(), mean.getDtype(), defaultDtype);
    visitReductionAlongAllDimsOp(mean, dtype, operands);
    return;
  } else if (isa<AtenMaxOp, AtenAmaxOp>(op)) {
    Type dtype = operands[0]->getValue().dtype;
    visitReductionAlongAllDimsOp(op, dtype, operands);
    return;
  } else if (isa<AtenStdOp, AtenStdDimOp, AtenStdCorrectionOp, AtenVarOp,
                 AtenVarDimOp, AtenVarCorrectionOp, PrimsVarOp>(op)) {
    auto input = operands[0]->getValue();
    visitReductionAlongAllDimsOp(op, input.dtype, operands);
    return;
  }

  if (auto tensorFloat = dyn_cast<AtenTensorFloatOp>(op)) {
    visitScalarToTensorConversionOp<AtenTensorFloatOp>(tensorFloat);
    return;
  } else if (auto tensorInt = dyn_cast<AtenTensorIntOp>(op)) {
    visitScalarToTensorConversionOp<AtenTensorIntOp>(tensorInt);
    return;
  } else if (auto tensorBool = dyn_cast<AtenTensorBoolOp>(op)) {
    visitScalarToTensorConversionOp<AtenTensorBoolOp>(tensorBool);
    return;
  }

  if (auto tensor = dyn_cast<AtenTensorOp>(op)) {
    visitAtenTensorOp(tensor);
    return;
  }

  if (auto zeros = dyn_cast<AtenZerosOp>(op)) {
    visitConstantTensorAllocOp<AtenZerosOp>(zeros, /*dataType=*/{});
    return;
  } else if (auto ones = dyn_cast<AtenOnesOp>(op)) {
    visitConstantTensorAllocOp<AtenOnesOp>(ones, /*dataType=*/{});
    return;
  } else if (auto emptyMemoryFormat = dyn_cast<AtenEmptyMemoryFormatOp>(op)) {
    visitConstantTensorAllocOp<AtenEmptyMemoryFormatOp>(emptyMemoryFormat,
                                                        /*dataType=*/{});
    return;
  } else if (auto full = dyn_cast<AtenFullOp>(op)) {
    visitConstantTensorAllocOp<AtenFullOp>(
        full, /*dataType=*/full.getFillValue().getType());
    return;
  } else if (auto zerosLike = dyn_cast<AtenZerosLikeOp>(op)) {
    visitConstantTensorAllocLikeOp<AtenZerosLikeOp>(zerosLike, operands);
    return;
  } else if (auto onesLike = dyn_cast<AtenOnesLikeOp>(op)) {
    visitConstantTensorAllocLikeOp<AtenOnesLikeOp>(onesLike, operands);
    return;
  } else if (auto emptyLike = dyn_cast<AtenEmptyLikeOp>(op)) {
    visitConstantTensorAllocLikeOp<AtenEmptyLikeOp>(emptyLike, operands);
    return;
  } else if (auto fullLike = dyn_cast<AtenFullLikeOp>(op)) {
    visitConstantTensorAllocLikeOp<AtenFullLikeOp>(fullLike, operands);
    return;
  } else if (auto newZeros = dyn_cast<AtenNewZerosOp>(op)) {
    visitConstantTensorNewLikeOp<AtenNewZerosOp>(newZeros, operands);
    return;
  } else if (auto newOnes = dyn_cast<AtenNewOnesOp>(op)) {
    visitConstantTensorNewLikeOp<AtenNewOnesOp>(newOnes, operands);
    return;
  } else if (auto newEmpty = dyn_cast<AtenNewEmptyOp>(op)) {
    visitConstantTensorNewLikeOp<AtenNewEmptyOp>(newEmpty, operands);
    return;
  } else if (auto newEmptyStrided = dyn_cast<AtenNewEmptyStridedOp>(op)) {
    visitConstantTensorNewLikeOp<AtenNewEmptyStridedOp>(newEmptyStrided,
                                                        operands);
    return;
  } else if (auto randLike = dyn_cast<AtenRandLikeOp>(op)) {
    visitConstantTensorAllocLikeOp<AtenRandLikeOp>(randLike, operands);
    return;
  } else if (auto randLike = dyn_cast<AtenRandnLikeOp>(op)) {
    visitConstantTensorAllocLikeOp<AtenRandnLikeOp>(randLike, operands);
    return;
  } else if (auto toCopy = dyn_cast<Aten_ToCopyOp>(op)) {
    visitConstantTensorAllocLikeOp<Aten_ToCopyOp>(toCopy, operands);
    return;
  }

  if (auto toDtype = dyn_cast<AtenToDtypeOp>(op)) {
    visitAtenToDtypeLikeOp<AtenToDtypeOp>(toDtype, operands);
    return;
  }

  if (auto primsConvertElementType = dyn_cast<PrimsConvertElementTypeOp>(op)) {
    visitAtenToDtypeLikeOp<PrimsConvertElementTypeOp>(primsConvertElementType,
                                                      operands);
    return;
  }

  if (auto toDtypeLayout = dyn_cast<AtenToDtypeLayoutOp>(op)) {
    visitAtenToDtypeLikeOp<AtenToDtypeLayoutOp>(toDtypeLayout, operands);
    return;
  }

  if (auto toDtype = dyn_cast<AtenToDeviceOp>(op)) {
    visitAtenToDtypeLikeOp<AtenToDeviceOp>(toDtype, operands);
    return;
  }

  if (auto toOther = dyn_cast<AtenToOtherOp>(op)) {
    visitTypeConversionOp<AtenToOtherOp>(toOther, operands);
    return;
  } else if (auto typeAs = dyn_cast<AtenTypeAsOp>(op)) {
    visitTypeConversionOp<AtenTypeAsOp>(typeAs, operands);
    return;
  }

  if (auto cat = dyn_cast<AtenCatOp>(op)) {
    visitAtenCatLikeOp<AtenCatOp>(cat, operands);
    return;
  } else if (auto stack = dyn_cast<AtenStackOp>(op)) {
    visitAtenCatLikeOp<AtenStackOp>(stack, operands);
    return;
  }

  if (auto shapeAsTensor = dyn_cast<Aten_ShapeAsTensorOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    incorporateKnowledge(shapeAsTensor.getResult(), knowledge);
    return;
  }

  if (auto embedding = dyn_cast<AtenEmbeddingOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = operands[0]->getValue().dtype;
    incorporateKnowledge(embedding.getResult(), knowledge);
    return;
  }

  if (isa<Aten_EmbeddingBagOp, AtenEmbeddingBagPaddingIdxOp>(op)) {
    visitAtenEmbeddingBagOp(op);
    return;
  }

  if (auto softmaxIntOp = dyn_cast<AtenSoftmaxIntOp>(op)) {
    visitAtenSoftmaxLikeOp(softmaxIntOp, operands);
    return;
  }
  if (auto _softmaxOp = dyn_cast<Aten_SoftmaxOp>(op)) {
    visitAten_SoftmaxLikeOp(_softmaxOp, operands);
    return;
  } else if (auto _logSoftmaxOp = dyn_cast<Aten_LogSoftmaxOp>(op)) {
    visitAten_SoftmaxLikeOp(_logSoftmaxOp, operands);
    return;
  } else if (auto logSoftmaxIntOp = dyn_cast<AtenLogSoftmaxIntOp>(op)) {
    visitAtenSoftmaxLikeOp(logSoftmaxIntOp, operands);
    return;
  }

  if (auto numToTensorOp = dyn_cast<PrimNumToTensorScalarOp>(op)) {
    visitNumToTensorOp(numToTensorOp);
    return;
  }

  if (isa<AtenAddIntOp, AtenSubIntOp, AtenMulIntOp, AtenAddOp>(op)) {
    visitBinaryScalarOp(op, operands);
    return;
  }

  if (auto scalarImplicit = dyn_cast<AtenScalarImplicitOp>(op)) {
    visitAtenScalarImplicitOp(scalarImplicit, operands);
    return;
  }

  if (auto vectorNorm = dyn_cast<AtenLinalgVectorNormOp>(op)) {
    Type defaultDtype = operands[0]->getValue().dtype;
    Type dtype = getDtypeOrDefault(vectorNorm.getContext(), vectorNorm.getDtype(),
                                   defaultDtype);
    visitReductionAlongDimIntListOp(vectorNorm, vectorNorm.getDim(),
                                    vectorNorm.getKeepdim(), dtype, operands);
    return;
  }

  if (auto randIntLow = dyn_cast<AtenRandintLowOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    Type defaultDtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    knowledge.dtype =
        getDtypeOrDefault(op->getContext(), randIntLow.getDtype(), defaultDtype);
    incorporateKnowledge(randIntLow.getResult(), knowledge);
    return;
  }

  if (auto randInt = dyn_cast<AtenRandintOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    Type defaultDtype =
        IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    knowledge.dtype =
        getDtypeOrDefault(op->getContext(), randInt.getDtype(), defaultDtype);
    incorporateKnowledge(randInt.getResult(), knowledge);
    return;
  }

  if (isa<AtenVarMeanCorrectionOp, AtenVarMeanOp>(op)) {
    auto input = operands[0]->getValue();
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    knowledge.dtype = input.dtype;
    incorporateKnowledge(op->getResult(0), knowledge);
    incorporateKnowledge(op->getResult(1), knowledge);
    return;
  }

  if (auto randn = dyn_cast<AtenRandnOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    Type defaultDtype = Float32Type::get(op->getContext());
    knowledge.dtype =
        getDtypeOrDefault(op->getContext(), randn.getDtype(), defaultDtype);
    incorporateKnowledge(randn.getResult(), knowledge);
    return;
  }

  if (auto randnGenerator = dyn_cast<AtenRandnGeneratorOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    Type defaultDtype = Float32Type::get(op->getContext());
    knowledge.dtype = getDtypeOrDefault(op->getContext(),
                                        randnGenerator.getDtype(), defaultDtype);
    incorporateKnowledge(randnGenerator.getResult(), knowledge);
    return;
  }

  if (auto bucketize = dyn_cast<AtenBucketizeTensorOp>(op)) {
    auto knowledge =
        ValueKnowledge::getTensorPessimisticValueState(op->getContext());
    bool outInt32;
    if (matchPattern(bucketize.getOutInt32(), m_TorchConstantBool(&outInt32)) &&
        outInt32) {
      knowledge.dtype =
          IntegerType::get(op->getContext(), 32, IntegerType::Signed);
    } else {
      knowledge.dtype =
          IntegerType::get(op->getContext(), 64, IntegerType::Signed);
    }
    incorporateKnowledge(bucketize.getResult(), knowledge);
    return;
  }

  // Otherwise, this is an unknown operation, so reset the state.
  setAllToEntryStates(results);
  return;
}

void TypeAnalysis::incorporateKnowledge(Value v,
                                        const ValueKnowledge &knowledge) {
  auto updatedKnowledge = ValueKnowledge::meet(
      knowledge, ValueKnowledge::getPessimisticValueState(v));
  assert(updatedKnowledge.has_value() && "IR has contradictory type!");
  getLatticeElement(v)->join(updatedKnowledge.value());
}

void TypeAnalysis::visitAtenLinearOp(AtenLinearOp op,
                                     ArrayRef<const ValueState *> operands) {
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
  incorporateKnowledge(op->getResult(0), knowledge);
}

void TypeAnalysis::visitAtenEmbeddingBagOp(Operation *op) {
  auto resultFloatKnowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  resultFloatKnowledge.dtype = Float32Type::get(op->getContext());

  incorporateKnowledge(op->getResult(0), resultFloatKnowledge);
  auto resultIntKnowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  resultIntKnowledge.dtype =
      IntegerType::get(op->getContext(), 64, IntegerType::Signed);

  for (int64_t i = 1, e = op->getNumResults(); i < e; i++) {
    incorporateKnowledge(op->getResult(i), resultIntKnowledge);
  }
  return;
}

// Arange like ops returns a 1-D tensor of size ceil(end - start).
void TypeAnalysis::visitAtenArangeLikeOpHelper(Operation *op,
                                               std::optional<Value> start,
                                               Value end,
                                               std::optional<Value> step,
                                               Value dtype) {
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
    if ((start.has_value() && (*start).getType().isa<Torch::FloatType>()) ||
        end.getType().isa<Torch::FloatType>() ||
        (step.has_value() && (*step).getType().isa<Torch::FloatType>())) {
      // TODO: Should get the dtype from torch.get_default_dtype().
      // For now, use float32 which is the initial default dtype.
      knowledge.dtype = Float32Type::get(op->getContext());
    } else
      knowledge.dtype =
          IntegerType::get(op->getContext(), 64, IntegerType::Signed);
  }
  incorporateKnowledge(op->getResult(0), knowledge);
}

void TypeAnalysis::visitAtenArangeStartStepOp(AtenArangeStartStepOp op) {
  visitAtenArangeLikeOpHelper(op, op.getStart(), op.getEnd(), op.getStep(), op.getDtype());
}

void TypeAnalysis::visitAtenArangeStartOp(AtenArangeStartOp op) {
  visitAtenArangeLikeOpHelper(op, op.getStart(), op.getEnd(), {}, op.getDtype());
}

void TypeAnalysis::visitAtenArangeOp(AtenArangeOp op) {
  visitAtenArangeLikeOpHelper(op, {}, op.getEnd(), {}, op.getDtype());
}

void TypeAnalysis::visitReductionAlongAllDimsOp(
    Operation *op, Type dtype, ArrayRef<const ValueState *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  incorporateKnowledge(op->getResult(0), knowledge);
}

// These ops do caculation along the dims given by the integer list and reduce
// each dim to size one. If \p keepdim is false, the dims are squeezed.
void TypeAnalysis::visitReductionAlongDimIntListOp(
    Operation *op, Value dim, Value keepdim, Type dtype,
    ArrayRef<const ValueState *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  incorporateKnowledge(op->getResult(0), knowledge);
}

void TypeAnalysis::visitReductionAlongDimIntOp(
    Operation *op, Value dim, Value keepdim, Type dtype,
    ArrayRef<const ValueState *> operands, int resNum) {
  assert(dim.getType().isa<Torch::IntType>() && "dim must be int type");
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  incorporateKnowledge(op->getResult(resNum), knowledge);
}

template <typename OpTy>
void TypeAnalysis::visitScalarToTensorConversionOp(OpTy op) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op.getContext());
  Value t = op.getT();
  Value dtype = op.getDtype();
  fillInDTypeGivenDTypeAndDataType(knowledge, dtype, t.getType());
  incorporateKnowledge(op.getResult(), knowledge);
}

void TypeAnalysis::visitBinaryScalarOp(Operation *op,
                                       ArrayRef<const ValueState *> operands) {
  auto knowledge =
      ValueKnowledge::getScalarPessimisticValueState(op->getContext());
  Type resultType = getPromotedResultScalarType(
      {op->getOperand(0).getType(), op->getOperand(1).getType()});
  knowledge.setScalarType(resultType);
  incorporateKnowledge(op->getResult(0), knowledge);
}

void TypeAnalysis::visitAtenTensorOp(AtenTensorOp op) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op.getContext());
  Value data = op.getData();
  Value dtype = op.getDtype();
  Type type = data.getType();
  while (auto listType = type.dyn_cast<ListType>()) {
    type = listType.getContainedType();
  }
  // TODO: Support tensor as the contained type of the list.
  // These are the only types handled by fillInDTypeGivenDTypeAndDataType below.
  if (!type.isa<Torch::FloatType, Torch::IntType, Torch::BoolType>()) {
    incorporateKnowledge(op.getResult(), knowledge);
    return;
  }
  fillInDTypeGivenDTypeAndDataType(knowledge, dtype, type);
  incorporateKnowledge(op.getResult(), knowledge);
}

template <typename OpTy>
void TypeAnalysis::visitConstantTensorAllocOp(OpTy op,
                                              std::optional<Type> dataType) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  if (!dataType)
    dataType = Torch::FloatType::get(op->getContext());
  fillInDTypeGivenDTypeAndDataType(knowledge, op.getDtype(), dataType.value());
  incorporateKnowledge(op.getResult(), knowledge);
}

template <typename OpTy>
void TypeAnalysis::visitConstantTensorAllocLikeOp(
    OpTy op, ArrayRef<const ValueState *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, op.getDtype(), input.dtype);
  incorporateKnowledge(op.getResult(), knowledge);
}

template <typename OpTy>
void TypeAnalysis::visitConstantTensorNewLikeOp(
    OpTy op, ArrayRef<const ValueState *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, op.getDtype(), input.dtype);
  incorporateKnowledge(op.getResult(), knowledge);
}

// Convert input tensor type to the given `dtype`.
template <typename OpTy>
void TypeAnalysis::visitAtenToDtypeLikeOp(
    OpTy op, ArrayRef<const ValueState *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  Value dtype = op.getDtype();
  int64_t dtypeInt;
  if (matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    knowledge.dtype = getTypeForDTypeInteger(op->getContext(), dtypeInt);
  incorporateKnowledge(op.getResult(), knowledge);
}

// Convert input tensor type to the same as the other tensor.
template <typename OpTy>
void TypeAnalysis::visitTypeConversionOp(
    OpTy op, ArrayRef<const ValueState *> operands) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  Value other = op.getOther();
  BaseTensorType type = other.getType().cast<BaseTensorType>();
  if (type.hasDtype())
    knowledge.dtype = type.getDtype();
  incorporateKnowledge(op->getResult(0), knowledge);
}

// `torch.aten.cat` concatenates the given sequence of seq tensors in the given
// dimension. The output has the same sizes as the input for all dimensions
// except the given dimension.
template <typename OpTy>
void TypeAnalysis::visitAtenCatLikeOp(OpTy op,
                                      ArrayRef<const ValueState *> operands) {
  auto tensorList = op.getTensors();
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  auto listConstruct = tensorList.template getDefiningOp<PrimListConstructOp>();
  if (!listConstruct) {
    incorporateKnowledge(op.getResult(), knowledge);
    return;
  }

  SmallVector<ValueKnowledge*> tensors = llvm::to_vector(
      llvm::map_range(listConstruct.getElements(), [&](Value v) -> ValueKnowledge* {
        return &getLatticeElement(v)->getValue();
      }));

  knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
      op->getContext(), tensors);
  incorporateKnowledge(op->getResult(0), knowledge);
}

void TypeAnalysis::visitNumToTensorOp(PrimNumToTensorScalarOp op) {
  auto knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  // The resulting type from converting a Scalar into a Tensor is different
  // if the scalar is part of a tensor operation (such as AtenMulScalar) or
  // not. In the former case, the type promotion rules are captured by the
  // `getDefaultDtypeForTorchScalar` helper above. The latter case follows the
  // rules in
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/ScalarOps.h.
  // `NumToTensor` falls in the latter case.
  Type type = op.getA().getType();
  knowledge.dtype = getBuiltInTypeForTorchScalar(type);
  incorporateKnowledge(op.getResult(), knowledge);
}

// Common template for softmax like ops, eg., log_softmax.
template <typename OpTy>
void TypeAnalysis::visitAtenSoftmaxLikeOp(
    OpTy op, ArrayRef<const ValueState *> operands) {
  auto input = operands[0]->getValue();
  auto dtype = op.getDtype();
  ValueKnowledge knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, dtype, input.dtype);
  incorporateKnowledge(op.getResult(), knowledge);
}

// Common template for softmax like ops, eg., log_softmax.(underscore variant)
template <typename OpTy>
void TypeAnalysis::visitAten_SoftmaxLikeOp(
    OpTy op, ArrayRef<const ValueState *> operands) {
  auto input = operands[0]->getValue();
  ValueKnowledge knowledge =
      ValueKnowledge::getTensorPessimisticValueState(op->getContext());
  bool halfToFloat;
  if (matchPattern(op.getHalfToFloat(), m_TorchConstantBool(&halfToFloat))) {
    knowledge.dtype =
        halfToFloat ? Float32Type::get(op->getContext()) : input.dtype;
  }
  incorporateKnowledge(op.getResult(), knowledge);
}

void TypeAnalysis::visitAtenScalarImplicitOp(
    AtenScalarImplicitOp op, ArrayRef<const ValueState *> operands) {
  auto knowledge =
      ValueKnowledge::getScalarPessimisticValueState(op.getContext());
  Type dType = operands[0]->getValue().dtype;
  if (dType.isa<mlir::FloatType>())
    knowledge.setScalarType(Torch::FloatType::get(op->getContext()));
  else if (dType.isa<mlir::IntegerType>())
    knowledge.setScalarType(Torch::IntType::get(op->getContext()));
  incorporateKnowledge(op->getResult(0), knowledge);
}

// -----------------------------------------------------------------------------
// Transforms.
// -----------------------------------------------------------------------------

// Get a the most refined type compatible with ValueKnowledge, or null if that
// is not possible.
static Type getMostRefinedStaticType(Value v, DataFlowSolver &solver) {
  auto getRefinedTensorType = [](BaseTensorType tensorType,
                                 ValueKnowledge const &knowledge) {
    return tensorType
        .getWithSizesAndDtype(tensorType.getOptionalSizes(), knowledge.dtype)
        .cast<BaseTensorType>();
  };
  if (auto tensorType = v.getType().dyn_cast<BaseTensorType>()) {
    const ValueState *latticeElement = solver.lookupState<ValueState>(v);
    if (!latticeElement)
      return nullptr;
    const ValueKnowledge &knowledge = latticeElement->getValue();
    if (!knowledge.isInitialized)
      return nullptr;
    return getRefinedTensorType(tensorType, knowledge);
  } else if (auto optionalType = v.getType().dyn_cast<OptionalType>()) {
    const ValueState *latticeElement = solver.lookupState<ValueState>(v);
    if (!latticeElement)
      return nullptr;
    const ValueKnowledge &knowledge = latticeElement->getValue();
    if (!knowledge.isInitialized)
      return nullptr;
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
    const ValueState *latticeElement = solver.lookupState<ValueState>(v);
    if (!latticeElement)
      return nullptr;
    const ValueKnowledge &knowledge = latticeElement->getValue();
    if (!knowledge.isInitialized)
      return nullptr;
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

void optimize(func::FuncOp func, DataFlowSolver &solver) {
  func.walk([&](Operation *op) {
    auto convertValuesToMostRefinedType = [&](ValueRange values, OpBuilder &b) {
      for (Value v : values) {
        Type refinedType = getMostRefinedStaticType(v, solver);
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
              newTypedValue = derefineOp.getOperand();
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
              Value overwriterTensor = overwriteTensorContents.getValue();
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
                oldTypedValue = derefineOp.getResult();
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
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TypeAnalysis>();
    if (failed(solver.initializeAndRun(func)))
      return signalPassFailure();
    optimize(func, solver);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createRefineTypesPass() {
  return std::make_unique<RefineTypesPass>();
}
