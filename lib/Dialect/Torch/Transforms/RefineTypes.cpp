//===- RefineTypes.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
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
using namespace mlir::torch::torch_upstream; // For ScalarType and type
                                             // promotion related helpers.

// -----------------------------------------------------------------------------
// Analysis.
// -----------------------------------------------------------------------------

static ScalarType getScalarTypeForType(Type type) {
  if (type.isa<Float32Type>())
    return ScalarType::Float;
  if (type.isa<Float64Type>())
    return ScalarType::Double;
  if (type.isSignedInteger(64))
    return ScalarType::Long;
  if (type.isSignedInteger(32))
    return ScalarType::Int;
  if (type.isUnsignedInteger(1))
    return ScalarType::Bool;
  llvm::report_fatal_error("unhandled type for getScalarTypeForType");
}

static Type getTypeForScalarType(MLIRContext *context, ScalarType dtypeInt) {
  switch (dtypeInt) {
  case ScalarType::Float:
    return Float32Type::get(context);
  case ScalarType::Double:
    return Float64Type::get(context);
  case ScalarType::Long:
    return IntegerType::get(context, 64, IntegerType::Signed);
  case ScalarType::Int:
    return IntegerType::get(context, 32, IntegerType::Signed);
  case ScalarType::Bool:
    return IntegerType::get(context, 1);
  default:
    return Type();
  }
}

static Type getTypeForDTypeInteger(MLIRContext *context, int64_t dtypeInt) {
  return getTypeForScalarType(context, (ScalarType)dtypeInt);
}

static Type joinElementTypes(Type lhs, Type rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  if (lhs == rhs)
    return lhs;
  return Type();
}

// This is the type rule used for deciding dtype for:
// 1. A new tensor created from given data.
// 2. The scalar type for type promotion when a scalar is an operand of a tensor
// and scalar binary operation.
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
  return Type();
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
  enum class OptionalKnowledge {
    unKnown,
    isNone,
    notNone,
  };
  ValueKnowledge() = delete;
  ValueKnowledge(bool hasSizes, std::vector<int64_t> sizes, Type dtype,
                 OptionalKnowledge optionalKnowledge)
      : hasSizes(hasSizes), sizes(sizes), dtype(dtype),
        optional(optionalKnowledge) {
    assert(sizes.size() == 0 || hasSizes);
  }

  // Get the static knowledge intrinsic to `type`.
  static ValueKnowledge getKnowledgeFromType(Type type) {
    ValueKnowledge result = getPessimisticValueState(type.getContext());
    if (auto tensorType = type.dyn_cast<BaseTensorType>()) {
      if (tensorType.hasSizes()) {
        result.hasSizes = true;
        result.sizes = tensorType.getSizes().vec();
      }
      result.dtype = tensorType.getOptionalDtype();
      result.optional = OptionalKnowledge::notNone;
    } else if (auto optionalType = type.dyn_cast<Torch::NoneType>()) {
      result.optional = OptionalKnowledge::isNone;
    } else if (!type.isa<OptionalType>()) {
      result.optional = OptionalKnowledge::notNone;
    }
    return result;
  }

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(false, {}, Type(), OptionalKnowledge::unKnown);
  }
  // Return a pessimistic/conservative value state only using knowlege already
  // recorded in the IR.
  static ValueKnowledge getPessimisticValueState(Value value) {
    return getKnowledgeFromType(value.getType());
  }

  static ValueKnowledge getNotNonePessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(false, {}, Type(), OptionalKnowledge::notNone);
  }

  bool operator==(const ValueKnowledge &rhs) const {
    return std::make_tuple(hasSizes, sizes, dtype, optional) ==
           std::make_tuple(rhs.hasSizes, rhs.sizes, rhs.dtype, rhs.optional);
  }

  // Given two pieces of static knowledge, calculate conservatively the
  // information we can be sure about.
  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs) {
    // Mental model: All conditions are checking how to change from the safe "no
    // knowledge" default-initialized state to a state with more knowledge
    // consistent with lhs and rhs.
    ValueKnowledge result = getPessimisticValueState(nullptr);

    // If lhs and rhs are not equal, the knowledge state must be the
    // pessimistic state.
    if (lhs.optional == rhs.optional)
      result.optional = lhs.optional;

    if (lhs.hasSizes && !rhs.hasSizes) {
      result.hasSizes = true;
      result.sizes = lhs.sizes;
    } else if (!lhs.hasSizes && rhs.hasSizes) {
      result.hasSizes = true;
      result.sizes = rhs.sizes;
    } else if (lhs.hasSizes && rhs.hasSizes &&
               lhs.sizes.size() == rhs.sizes.size()) {
      result.hasSizes = true;
      result.sizes.resize(lhs.sizes.size(), kUnknownSize);
      for (int i = 0, e = result.sizes.size(); i != e; i++) {
        int64_t lhsSize = lhs.sizes[i];
        int64_t rhsSize = rhs.sizes[i];
        int64_t &resultSize = result.sizes[i];
        if (lhsSize == kUnknownSize) {
          resultSize = rhsSize;
        } else if (rhsSize == kUnknownSize) {
          resultSize = lhsSize;
        } else if (lhsSize == rhsSize) {
          resultSize = lhsSize;
        }
      }
    }

    result.dtype = joinElementTypes(lhs.dtype, rhs.dtype);
    return result;
  }

  // Whether the Value is known to have a list of sizes.
  bool hasSizes;
  // If `hasSizes`, the sizes along each rank. Unknown sizes are represented as
  // `kUnknownSize`.
  std::vector<int64_t> sizes;
  // The dtype of a tensor.
  // This is equal to nullptr if we don't know that it is a specific concrete
  // type.
  Type dtype;
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
                 ArrayRef<LatticeElement<ValueKnowledge> *> operands) final {
    if (isa<TensorStaticInfoCastOp, CopyToValueTensorOp, CopyToNonValueTensorOp,
            AtenTanhOp, AtenBatchNormOp, AtenReluOp, AtenGeluOp,
            AtenGeluBackwardOp, AtenEqScalarOp, AtenGeScalarOp, AtenGtScalarOp,
            AtenNeScalarOp, AtenBitwiseNotOp, AtenExpOp, AtenSinOp, AtenCosOp,
            AtenSigmoidOp, DerefineOp, AtenToPrimDeviceOp, AtenCpuOp,
            AtenContiguousOp, AtenFill_ScalarOp, AtenDetachOp,
            AtenMaskedFill_ScalarOp, AtenCopy_Op, AtenIndexPut_Op, AtenCumsumOp,
            AtenLayerNormOp, AtenClampOp, AtenLogOp, AtenSqrtOp, AtenFloorOp,
            AtenLog2Op, Aten_SoftmaxBackwardDataOp, AtenRsqrtOp,
            AtenTanhBackwardOp>(op)) {
      return getLatticeElement(op->getResult(0)).join(*operands[0]);
    }

    // Resize to [1, 1] with integer dtype.
    if (isa<AtenAnyOp, AtenAllOp>(op)) {
      auto input = operands[0]->getValue();
      auto knowledge =
          ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
      knowledge.hasSizes = true;
      knowledge.sizes.resize(1, 1);
      knowledge.dtype = IntegerType::get(op->getContext(), 1);
      return getLatticeElement(op->getResult(0)).join(knowledge);
    }
    // `torch.aten.masked_select` returns a new 1-D tensor which indexes the
    // input tensor according to the boolean mask which is a BoolTensor.
    // Resize to [unknown] with same dtype as the input.
    if (auto maskedSelect = dyn_cast<AtenMaskedSelectOp>(op)) {
      auto input = operands[0]->getValue();
      auto knowledge =
          ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
      knowledge.hasSizes = true;
      knowledge.sizes.resize(1, kUnknownSize);
      knowledge.dtype = input.dtype;
      return getLatticeElement(op->getResult(0)).join(knowledge);
    }
    // `torch.aten.index.Tensor` return tensors elements selected by the index
    // tensors. Each index tensor in the list corresponds to each dim in the
    // input tensor.
    // Same number of dims but unknown size along each dim. Same dtype as the
    // input.
    if (auto indexTensor = dyn_cast<AtenIndexTensorOp>(op)) {
      auto input = operands[0]->getValue();
      auto knowledge =
          ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
      if (input.hasSizes) {
        knowledge.hasSizes = true;
        knowledge.sizes.resize(input.sizes.size(), kUnknownSize);
      }
      knowledge.dtype = input.dtype;
      return getLatticeElement(op->getResult(0)).join(knowledge);
    }

    if (auto mm = llvm::dyn_cast<AtenMmOp>(op)) {
      return visitAtenMmOp(mm, operands);
    } else if (auto linear = llvm::dyn_cast<AtenLinearOp>(op)) {
      return visitAtenLinearOp(linear, operands);
    } else if (auto conv2d = llvm::dyn_cast<AtenConv2dOp>(op)) {
      return visitAtenConv2dOp(conv2d, operands);
    } else if (auto maxPool2d = llvm::dyn_cast<AtenMaxPool2dOp>(op)) {
      return visitAtenMaxPool2dOp(maxPool2d, operands);
    } else if (auto avgPool2d = llvm::dyn_cast<AtenAdaptiveAvgPool2dOp>(op)) {
      return visitAtenAdaptiveAvgPool2dOp(avgPool2d, operands);
    } else if (isa<AtenAddScalarOp, AtenSubScalarOp, AtenMulScalarOp,
                   AtenDivScalarOp, AtenFmodScalarOp, AtenFloorDivideScalarOp,
                   AtenPowTensorScalarOp, AtenRsubScalarOp>(op)) {
      return visitBinaryTensorScalarOp(op, operands);
    } else if (isa<AtenAddTensorOp, AtenSubTensorOp, AtenMulTensorOp,
                   AtenDivTensorOp, Aten__And__TensorOp, AtenEqTensorOp,
                   AtenMinimumOp, AtenMaximumOp>(op)) {
      return visitBinaryBroadcastingOp(op, operands);
    } else if (auto lerpTensor = llvm::dyn_cast<AtenLerpTensorOp>(op)) {
      return visitAtenLerpTensorOp(lerpTensor, operands);
    } else if (auto flatten = dyn_cast<AtenFlattenUsingIntsOp>(op)) {
      return visitAtenFlattenUsingIntsOp(flatten, operands);
    } else if (auto unsqueeze = dyn_cast<AtenUnsqueezeOp>(op)) {
      return visitAtenUnsqueezeOp(unsqueeze, operands);
    } else if (auto arange = dyn_cast<AtenArangeOp>(op)) {
      return visitAtenArangeOp(arange);
    } else if (auto arangeStart = dyn_cast<AtenArangeStartOp>(op)) {
      return visitAtenArangeStartOp(arangeStart);
    } else if (auto sum = dyn_cast<AtenSumOp>(op)) {
      Type dtype = operands[0]->getValue().dtype;
      return visitReductionAlongAllDimsOp(sum, dtype, operands);
    } else if (auto sumDimIntList = dyn_cast<AtenSumDimIntListOp>(op)) {
      return visitReductionAlongDimIntListOp(sumDimIntList, sumDimIntList.dim(),
                                             sumDimIntList.keepdim(), operands);
    } else if (auto meanDim = dyn_cast<AtenMeanDimOp>(op)) {
      return visitReductionAlongDimIntListOp(meanDim, meanDim.dim(),
                                             meanDim.keepdim(), operands);
    } else if (auto argmax = dyn_cast<AtenArgmaxOp>(op)) {
      Value dim = argmax.dim();
      Type dtype = IntegerType::get(op->getContext(), 64, IntegerType::Signed);
      if (dim.getType().isa<Torch::NoneType>())
        return visitReductionAlongAllDimsOp(op, dtype, operands);
      if (dim.getType().isa<Torch::IntType>())
        return visitReductionAlongDimIntOp(argmax, argmax.dim(),
                                           argmax.keepdim(), dtype, operands);
    } else if (auto anyDim = dyn_cast<AtenAnyDimOp>(op)) {
      Type dtype = operands[0]->getValue().dtype;
      return visitReductionAlongDimIntOp(anyDim, anyDim.dim(), anyDim.keepdim(),
                                         dtype, operands);
    } else if (auto view = dyn_cast<AtenViewOp>(op)) {
      return visitReshapeLikeOp(view, operands);
    } else if (auto resize = dyn_cast<AtenResize_Op>(op)) {
      return visitReshapeLikeOp(resize, operands);
    } else if (auto transposeInt = dyn_cast<AtenTransposeIntOp>(op)) {
      return visitAtenTransposeIntOp(transposeInt, operands);
    } else if (auto permute = dyn_cast<AtenPermuteOp>(op)) {
      return visitAtenPermuteOp(permute, operands);
    } else if (auto tensorFloat = dyn_cast<AtenTensorFloatOp>(op)) {
      return visitScalarToTensorConversionOp<AtenTensorFloatOp>(tensorFloat);
    } else if (auto tensorInt = dyn_cast<AtenTensorIntOp>(op)) {
      return visitScalarToTensorConversionOp<AtenTensorIntOp>(tensorInt);
    } else if (auto tensorBool = dyn_cast<AtenTensorBoolOp>(op)) {
      return visitScalarToTensorConversionOp<AtenTensorBoolOp>(tensorBool);
    } else if (auto tensor = dyn_cast<AtenTensorOp>(op)) {
      return visitAtenTensorOp(tensor);
    } else if (auto zeros = dyn_cast<AtenZerosOp>(op)) {
      return visitConstantTensorAllocOp<AtenZerosOp>(zeros);
    } else if (auto ones = dyn_cast<AtenOnesOp>(op)) {
      return visitConstantTensorAllocOp<AtenOnesOp>(ones);
    } else if (auto emptyMemoryFormat = dyn_cast<AtenEmptyMemoryFormatOp>(op)) {
      return visitConstantTensorAllocOp<AtenEmptyMemoryFormatOp>(
          emptyMemoryFormat);
    } else if (auto toDtype = dyn_cast<AtenToDtypeOp>(op)) {
      return visitAtenToDtypeOp(toDtype, operands);
    } else if (auto toOther = dyn_cast<AtenToOtherOp>(op)) {
      return visitTypeConversionOp<AtenToOtherOp>(toOther, operands);
    } else if (auto typeAs = dyn_cast<AtenTypeAsOp>(op)) {
      return visitTypeConversionOp<AtenTypeAsOp>(typeAs, operands);
    } else if (auto indexSelect = dyn_cast<AtenIndexSelectOp>(op)) {
      // The index tensor index into the dimension specified by the dim. The dim
      // of output is the same size as the length of index (index must be one
      // dimensional)
      auto setDim = [](int64_t &targetDim, int64_t dim,
                       ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
        auto indexes = operands[2]->getValue();
        targetDim = indexes.hasSizes && indexes.sizes.size() != 0
                        ? indexes.sizes[0]
                        : kUnknownSize;
      };
      return visitSliceLikeOp(indexSelect, operands, setDim);
    } else if (auto selectInt = dyn_cast<AtenSelectIntOp>(op)) {
      // Select one element from the target dim. All the other dims are the same
      // as input.
      auto setDim = [](int64_t &targetDim, int64_t dim,
                       ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
        targetDim = 1;
      };
      return visitSliceLikeOp(selectInt, operands, setDim);
    } else if (auto sliceTensor = dyn_cast<AtenSliceTensorOp>(op)) {
      // Select several elements from the target dim according to the start,
      // end, step. All the other dims are the same as input.
      auto setDim = [](int64_t &targetDim, int64_t dim,
                       ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
        targetDim = kUnknownSize;
      };
      return visitSliceLikeOp(sliceTensor, operands, setDim);
    } else if (auto gather = dyn_cast<AtenGatherOp>(op)) {
      return visitAtenGatherOp(gather, operands);
    } else if (isa<AtenExpandOp, AtenBroadcastToOp>(op)) {
      // Broadcast dimensions of size 1 withs sizes
      // spcecified by the `size` operand. -1 in the `size` list means the
      // dimension is kept unchanged.
      auto setDim = [](int64_t &targetDim, int64_t inputDim, int64_t size) {
        targetDim = size == -1 ? inputDim : size;
      };
      Value size;
      if (auto expand = dyn_cast<AtenExpandOp>(op))
        size = expand.size();
      else if (auto broadcast = dyn_cast<AtenBroadcastToOp>(op))
        size = broadcast.size();
      return visitExpandLikeOp(op, size, operands, setDim);
    } else if (auto repeat = dyn_cast<AtenRepeatOp>(op)) {
      // The repeats list specify the number of times to repeat along each dim
      // of the original tensor.
      auto setDim = [](int64_t &targetDim, int64_t inputDim, int64_t repeat) {
        if (inputDim != kUnknownSize)
          targetDim = inputDim * repeat;
      };
      return visitExpandLikeOp(repeat, repeat.repeats(), operands, setDim);
    } else if (auto cat = dyn_cast<AtenCatOp>(op)) {
      return visitAtenCatOp(cat, operands);
    } else if (auto shapeAsTensor = dyn_cast<Aten_ShapeAsTensorOp>(op)) {
      return visitAtenShapeAsTensorOp(shapeAsTensor, operands);
    } else if (auto embedding = dyn_cast<AtenEmbeddingOp>(op)) {
      return visitAtenEmbeddingOp(embedding, operands);
    } else if (auto bmm = dyn_cast<AtenBmmOp>(op)) {
      return visitAtenBmmOp(bmm, operands);
    } else if (auto matmul = dyn_cast<AtenMatmulOp>(op)) {
      return visitAtenMatmulOp(matmul, operands);
    } else if (auto softmaxIntOp = dyn_cast<AtenSoftmaxIntOp>(op)) {
      return visitAtenSoftmaxLikeOp(softmaxIntOp, operands);
    } else if (auto logSoftmaxIntOp = dyn_cast<AtenLogSoftmaxIntOp>(op)) {
      return visitAtenSoftmaxLikeOp(logSoftmaxIntOp, operands);
    } else if (auto numToTensorOp = dyn_cast<PrimNumToTensorScalarOp>(op)) {
      return visitNumToTensorOp(numToTensorOp);
    }

    // Otherwise, this is an unknown operation. Just mark all results as
    // having reached a pessimistic fixpoint.
    return markAllPessimisticFixpoint(op->getResults());
  }

private:
  ChangeResult
  visitAtenMmOp(AtenMmOp op,
                ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenLinearOp(AtenLinearOp op,
                    ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenConv2dOp(AtenConv2dOp op,
                    ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenMaxPool2dOp(AtenMaxPool2dOp op,
                       ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitAtenAdaptiveAvgPool2dOp(
      AtenAdaptiveAvgPool2dOp op,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitBinaryTensorScalarOp(
      Operation *op, ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitBinaryBroadcastingOp(
      Operation *op, ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenLerpTensorOp(AtenLerpTensorOp op,
                        ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitAtenFlattenUsingIntsOp(
      AtenFlattenUsingIntsOp op,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenUnsqueezeOp(AtenUnsqueezeOp op,
                       ArrayRef<LatticeElement<ValueKnowledge> *> operands);

  ChangeResult visitAtenArangeLikeOpHelper(Operation *op,
                                           llvm::Optional<Value> start,
                                           Value end, Value dtype);
  ChangeResult visitAtenArangeStartOp(AtenArangeStartOp op);
  ChangeResult visitAtenArangeOp(AtenArangeOp op);
  ChangeResult visitReductionAlongAllDimsOp(
      Operation *op, Type dtype,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitReductionAlongDimIntListOp(
      Operation *op, Value dim, Value keepdim,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitReductionAlongDimIntOp(
      Operation *op, Value dim, Value keepdim, Type dtype,
      ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  template <typename OpTy>
  ChangeResult
  visitReshapeLikeOp(OpTy op,
                     ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenTransposeIntOp(AtenTransposeIntOp op,
                          ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenPermuteOp(AtenPermuteOp op,
                     ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult visitNumToTensorOp(PrimNumToTensorScalarOp op);
  template <typename OpTy>
  ChangeResult visitScalarToTensorConversionOp(OpTy op);
  ChangeResult visitAtenTensorOp(AtenTensorOp op);
  template <typename OpTy> ChangeResult visitConstantTensorAllocOp(OpTy op);
  ChangeResult
  visitAtenToDtypeOp(AtenToDtypeOp op,
                     ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  template <typename OpTy>
  ChangeResult
  visitTypeConversionOp(OpTy op,
                        ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  using SetDimSizeFn =
      std::function<void(int64_t &targetDim, int64_t dim,
                         ArrayRef<LatticeElement<ValueKnowledge> *> operands)>;
  template <typename OpTy>
  ChangeResult
  visitSliceLikeOp(OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands,
                   SetDimSizeFn setDim);
  ChangeResult
  visitAtenGatherOp(AtenGatherOp op,
                    ArrayRef<LatticeElement<ValueKnowledge> *> operands);

  using SetDimSizePerListItemFn = std::function<void(
      int64_t &targetDim, int64_t inputDim, int64_t listValue)>;
  ChangeResult
  visitExpandLikeOp(Operation *op, Value list,
                    ArrayRef<LatticeElement<ValueKnowledge> *> operands,
                    SetDimSizePerListItemFn setDim);
  ChangeResult
  visitAtenCatOp(AtenCatOp op,
                 ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenShapeAsTensorOp(Aten_ShapeAsTensorOp op,
                           ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenEmbeddingOp(AtenEmbeddingOp op,
                       ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenBmmOp(AtenBmmOp op,
                 ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  ChangeResult
  visitAtenMatmulOp(AtenMatmulOp op,
                    ArrayRef<LatticeElement<ValueKnowledge> *> operands);
  
  template <typename OpTy>
  ChangeResult
  visitAtenSoftmaxLikeOp(OpTy op,
                        ArrayRef<LatticeElement<ValueKnowledge> *> operands);
};
} // namespace

static ResultTypeState updateResultTypeState(Type scalarType,
                                             const ResultTypeState &inState) {
  ResultTypeState new_state = inState;
  ScalarType current =
      getScalarTypeForType(getDefaultDtypeForTorchScalar(scalarType));
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
// guaranteed to be none zero rank or that the result ResultTypeState is
// promoted with a scalar which is guaranteed to be lower priority.
static ResultTypeState updateResultTypeState(ValueKnowledge *tensor,
                                             const ResultTypeState &inState,
                                             bool skipRankCheck = false) {
  if (!tensor->hasSizes && !skipRankCheck)
    return ResultTypeState{};
  assert(tensor->dtype && "tensor.dtype must be not none");

  ResultTypeState new_state = inState;
  ScalarType current = getScalarTypeForType(tensor->dtype);
  if (skipRankCheck || tensor->sizes.size() > 0)
    new_state.dimResult = promote_skip_undefined(inState.dimResult, current);
  else
    new_state.zeroResult = promote_skip_undefined(inState.zeroResult, current);

  return new_state;
}

// Returns most generic type Type() if the tensor dtype is unknown.
static Type getPromotedResultType(ValueKnowledge *tensor, Type scalarType) {
  if (!tensor->dtype)
    return Type();
  ResultTypeState state = {};
  // No need to check if rank is zero for tensor because scalar uses
  // wrappedResult which is a lower priority than both dimResult and zeroResult.
  state = updateResultTypeState(tensor, state, /*skipRankCheck=*/true);
  state = updateResultTypeState(scalarType, state);
  return getTypeForScalarType(scalarType.getContext(), result_type(state));
}

// Normally, tensor dimensions need to be known at compile time to do type
// promotion. `skipRankCheck`, when equal to true, can be used to indicate
// special cases that tensor operands are guaranteed to be not zero dimension
// like operands of `aten.conv2d` or `aten.mm` assuming no runtime error.
//
// Returns most generic type Type() if the tensor dtype is unknown.
static Type getPromotedResultType(MLIRContext *context,
                                  ArrayRef<ValueKnowledge *> tensors,
                                  bool skipRankCheck = false) {
  ResultTypeState state = {};
  for (ValueKnowledge *tensor : tensors) {
    if (!tensor->dtype)
      return Type();
    state = updateResultTypeState(tensor, state, skipRankCheck);
  }
  return getTypeForScalarType(context, result_type(state));
}

static Type
getPromotedResultTypeAssumingNonZeroRank(MLIRContext *context,
                                         ArrayRef<ValueKnowledge *> tensors) {
  return getPromotedResultType(context, tensors, /*skipRankCheck=*/true);
}
// Get the MLIR type of the tensor dtype given the dtype integer value and the
// input dtype. When DType is None the type is inferred from the input dtype.
static void fillInDTypeGivenDTypeIntAndInputDType(ValueKnowledge &knowledge,
                                                  Value dtype,
                                                  Type inputDType) {
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
  Type dtypeForDataType = getDefaultDtypeForTorchScalar(dataType);
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, dtype, dtypeForDataType);
}

static void fillInSizesGivenSizesList(ValueKnowledge &knowledge, Value sizes) {
  // TODO: This is not safe. Need to check the list users and use aliasing
  // info to safely detect the list is not modified.
  if (auto listConstruct = sizes.getDefiningOp<PrimListConstructOp>()) {
    knowledge.hasSizes = true;
    auto sizes = listConstruct.elements();
    int64_t size;
    for (auto sizeValue : sizes) {
      if (matchPattern(sizeValue, m_TorchConstantInt(&size)))
        knowledge.sizes.push_back(size);
      else
        knowledge.sizes.push_back(kUnknownSize);
    }
  }
}

ChangeResult TypeAnalyzer::visitAtenMmOp(
    AtenMmOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto &lhs = operands[0]->getValue();
  auto &rhs = operands[1]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.hasSizes = true;
  // WARNING: We could be more precise here by calculating the output
  // shape as "(lhs.shape[0], rhs.shape[1])". However, that is really tricky
  // at this stage in the compiler because we don't really have many static
  // guarantees about the input ranks because `aten` ops do dynamic error
  // checking and safely abort the program. There is nothing preventing us
  // from (correctly!) statically inferring the shapes of the operands to
  // shapes that are guaranteed to cause an error at runtime.
  //
  // Example: Suppose a user program calls `aten.mm` with two rank-0
  // operands. The program emits an error when invoked, but when running
  // this pass, we will (correctly!) infer `lhs.hasSizes && lhs.sizes.size()
  // == 0 && rhs.hasSizes && rhs.sizes.size() == 0` -- it's not safe to
  // access `lhs.sizes[0]` / `rhs.sizes[1]`! So when writing this transfer
  // function, it's not as simple as taking `lhs.sizes[0]` and
  // `rhs.sizes[1]`, as both of those might read out of bounds of the array.
  // It would require more complicated logic.
  //
  // Just knowing dtypes and ranks is sufficient at this stage
  // in the compiler. The precise per-dimension size propagation is best
  // done lower in the stack, such as at the linalg level, where we have
  // more static guarantees and more structure.
  knowledge.sizes.resize(2, kUnknownSize);
  // TODO: Investigate promotion rules if element types mismatch.
  // This is conservatively correct, assuming that if both element types are
  // the same, then the result is of that same element type.
  knowledge.dtype =
      getPromotedResultTypeAssumingNonZeroRank(op->getContext(), {&lhs, &rhs});
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenLinearOp(
    AtenLinearOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  // The output shape is the input shape with the last dimension changed
  // to the weight's output dimension.
  auto knowledge = operands[0]->getValue();
  auto weight = operands[1]->getValue();
  auto bias = operands[2]->getValue();
  if (knowledge.hasSizes && knowledge.sizes.size() > 0)
    knowledge.sizes[knowledge.sizes.size() - 1] = kUnknownSize;
  switch (bias.optional) {
  case ValueKnowledge::OptionalKnowledge::isNone:
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&knowledge, &weight});
    break;
  case ValueKnowledge::OptionalKnowledge::notNone:
    knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
        op->getContext(), {&knowledge, &weight, &bias});
    break;
  case ValueKnowledge::OptionalKnowledge::unKnown:
    // When it's unknown, type promotion can't be decided at compile time.
    break;
  }
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenConv2dOp(
    AtenConv2dOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.hasSizes = true;
  knowledge.sizes.resize(4, kUnknownSize);
  // Running some experiments in PyTorch, the bias doesn't seem to
  // contribute to the final element type.
  knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(
      op->getContext(), {&operands[0]->getValue(), &operands[1]->getValue()});
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenMaxPool2dOp(
    AtenMaxPool2dOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.hasSizes = true;
  knowledge.sizes.resize(4, kUnknownSize);
  knowledge.dtype = operands[0]->getValue().dtype;
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenAdaptiveAvgPool2dOp(
    AtenAdaptiveAvgPool2dOp op,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  if (input.hasSizes) {
    knowledge.hasSizes = true;
    knowledge.sizes.resize(input.sizes.size(), kUnknownSize);
  }
  knowledge.dtype = input.dtype;
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitBinaryTensorScalarOp(
    Operation *op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto lhs = operands[0]->getValue();
  Value scalar = op->getOperand(1);
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(getContext());
  if (lhs.hasSizes) {
    knowledge.hasSizes = true;
    knowledge.sizes = lhs.sizes;
  }
  knowledge.dtype = getPromotedResultType(&lhs, scalar.getType());
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitBinaryBroadcastingOp(
    Operation *op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  // This is a general binary broadcasting shape transfer function.
  // We currently don't track "size 1" in our lattice, but we might want to.
  // We could make this more precise as well. But again, as with the other
  // shape transfer functions, handling the statically-invalid case is
  // tricky, so we defer that until we need it.
  auto lhs = operands[0]->getValue();
  auto rhs = operands[1]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(getContext());
  if (lhs.hasSizes && rhs.hasSizes) {
    knowledge.hasSizes = true;
    knowledge.sizes.resize(std::max(lhs.sizes.size(), rhs.sizes.size()),
                           kUnknownSize);
  }

  // The alpha in `aten.add.Tensor` and `aten.sub.Tensor` has to be lower type
  // category than the lhs and rhs and therefore doesn't really contribute to
  // type promotion.
  knowledge.dtype = getPromotedResultType(getContext(), {&lhs, &rhs});
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenLerpTensorOp(
    AtenLerpTensorOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  // This is a general broadcasting shape transfer function.
  // We currently don't track "size 1" in our lattice, but we might want to.
  // We could make this more precise as well. But again, as with the other
  // shape transfer functions, handling the statically-invalid case is
  // tricky, so we defer that until we need it.
  auto self = operands[0]->getValue();
  auto end = operands[1]->getValue();
  auto weight = operands[2]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  if (self.hasSizes && end.hasSizes && weight.hasSizes) {
    knowledge.hasSizes = true;
    knowledge.sizes.resize(
        std::max(std::max(self.sizes.size(), end.sizes.size()),
                 weight.sizes.size()),
        kUnknownSize);
  }
  knowledge.dtype = getPromotedResultType(getContext(), {&self, &end, &weight});
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenFlattenUsingIntsOp(
    AtenFlattenUsingIntsOp op,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  int64_t startDim;
  int64_t endDim;
  auto operand = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op.getContext());
  knowledge.dtype = operand.dtype;
  if (operand.hasSizes && operand.sizes.size() == 0) {
    // Rank 0 is special and flattens to rank 1 with size 1.
    knowledge.hasSizes = true;
    knowledge.sizes.push_back(1);
  } else if (operand.hasSizes &&
             matchPattern(op.start_dim(), m_TorchConstantInt(&startDim)) &&
             matchPattern(op.end_dim(), m_TorchConstantInt(&endDim))) {
    int64_t inputRank = operand.sizes.size();
    if (startDim < 0)
      startDim += inputRank;
    if (endDim < 0)
      endDim += inputRank;
    // Careful: dimension numbers might be out of bounds.
    if (0 <= startDim && startDim <= (inputRank - 1) && 0 <= endDim &&
        endDim <= (inputRank - 1) && startDim <= endDim) {
      knowledge.hasSizes = true;
      for (auto i = 0; i < startDim; i++)
        knowledge.sizes.push_back(operand.sizes[i]);
      knowledge.sizes.push_back(kUnknownSize);
      for (auto i = endDim + 1; i < inputRank; i++)
        knowledge.sizes.push_back(operand.sizes[i]);
    }
  }
  return getLatticeElement(op.getResult()).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenUnsqueezeOp(
    AtenUnsqueezeOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto operand = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op.getContext());
  knowledge.dtype = operand.dtype;
  int64_t dim;
  if (operand.hasSizes && matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    int64_t inputRank = operand.sizes.size();
    // Careful, it's easy to be off by one here for negative values.
    // The dim value is allowed to be in the range
    // `[-inputRank - 1, inputRank]`.
    // And negative values have `inputRank + 1` added to them rather
    // than the more typical `inputRank`.
    if (dim < 0)
      dim += inputRank + 1;
    if (0 <= dim && dim <= inputRank) {
      knowledge.hasSizes = true;
      knowledge.sizes = operand.sizes;
      knowledge.sizes.insert(knowledge.sizes.begin() + dim, 1);
    }
  }
  return getLatticeElement(op.getResult()).join(knowledge);
}

// Arange like ops returns a 1-D tensor of size ceil(end - start).
ChangeResult TypeAnalyzer::visitAtenArangeLikeOpHelper(
    Operation *op, llvm::Optional<Value> start, Value end, Value dtype) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.sizes.resize(1, kUnknownSize);
  knowledge.hasSizes = true;
  int64_t dtypeInt;
  if (matchPattern(dtype, m_TorchConstantInt(&dtypeInt))) {
    knowledge.dtype = getTypeForDTypeInteger(op->getContext(), dtypeInt);
  } else if (dtype.getType().isa<Torch::NoneType>()) {
    // From torch/_torch_docs.py:
    // If `dtype` is not given, infer the data type from the other input
    // arguments. If any of `start`, `end`, or `stop` are floating-point, the
    // `dtype` is inferred to be the default dtype, see
    // `torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
    // be `torch.int64`
    if ((start.hasValue() && (*start).getType().isa<Torch::FloatType>()) ||
        end.getType().isa<Torch::FloatType>()) {
      // TODO: Should get the dtype from torch.get_default_dtype().
      // For now, use float32 which is the initial default dtype.
      knowledge.dtype = Float32Type::get(op->getContext());
    } else
      knowledge.dtype =
          IntegerType::get(op->getContext(), 64, IntegerType::Signed);
  }
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenArangeStartOp(AtenArangeStartOp op) {
  return visitAtenArangeLikeOpHelper(op, op.start(), op.end(), op.dtype());
}

ChangeResult TypeAnalyzer::visitAtenArangeOp(AtenArangeOp op) {
  return visitAtenArangeLikeOpHelper(op, {}, op.end(), op.dtype());
}

ChangeResult TypeAnalyzer::visitReductionAlongAllDimsOp(
    Operation *op, Type dtype,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge = ValueKnowledge::getPessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  // Reduction along all dims always results in a tensor of rank zero,
  // which is represented by the default empty `knowledge.sizes` vector
  knowledge.hasSizes = true;
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

// These ops do caculation along the dims given by the integer list and reduce
// each dim to size one. If \p keepdim is false, the dims are squeezed.
ChangeResult TypeAnalyzer::visitReductionAlongDimIntListOp(
    Operation *op, Value dim, Value keepdim,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.dtype = input.dtype;
  llvm::SmallVector<int64_t> dimList;
  bool keepDim;
  if (matchPattern(keepdim, m_TorchConstantBool(&keepDim))) {
    knowledge.hasSizes = true;
    int64_t inputRank = input.sizes.size();
    // TODO: This is not safe. Need to check the list users and use aliasing
    // info to safely detect the list is not modified.
    if (matchPattern(dim, m_TorchConstantIntList(dimList))) {
      llvm::for_each(
          dimList, [&](int64_t &dim) { dim = toPositiveDim(dim, inputRank); });
      DenseSet<int64_t> dimSet(dimList.begin(), dimList.end());
      for (auto en : llvm::enumerate(input.sizes)) {
        if (dimSet.contains(en.index())) {
          if (keepDim)
            knowledge.sizes.push_back(1);
        } else {
          knowledge.sizes.push_back(en.value());
        }
      }
    } else if (auto listConstruct = dim.getDefiningOp<PrimListConstructOp>()) {
      auto sizes = listConstruct.elements();
      knowledge.sizes.resize(keepDim ? inputRank : inputRank - sizes.size(),
                             kUnknownSize);
    }
  }
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitReductionAlongDimIntOp(
    Operation *op, Value dim, Value keepdim, Type dtype,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  assert(dim.getType().isa<Torch::IntType>() && "dim must be int type");
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.dtype = dtype;
  int64_t dimInt;
  bool keepDim;
  if (matchPattern(keepdim, m_TorchConstantBool(&keepDim))) {
    int64_t inputRank = input.sizes.size();
    knowledge.hasSizes = true;
    if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
      knowledge.sizes = input.sizes;
      dimInt = toPositiveDim(dimInt, inputRank);
      if (isValidDim(dimInt, inputRank)) {
        if (keepDim)
          knowledge.sizes[dimInt] = 1;
        else
          knowledge.sizes.erase(knowledge.sizes.begin() + dimInt);
      }
    } else {
      knowledge.sizes.resize(keepDim ? inputRank : inputRank - 1, kUnknownSize);
    }
  }
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

// Reshape like ops are given a size list which specify the shape of the
// result tensor.
template <typename OpTy>
ChangeResult TypeAnalyzer::visitReshapeLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op.getContext());
  knowledge.dtype = input.dtype;

  fillInSizesGivenSizesList(knowledge, op.size());
  return getLatticeElement(op.getResult()).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenTransposeIntOp(
    AtenTransposeIntOp op,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op.getContext());
  knowledge.dtype = input.dtype;
  knowledge.hasSizes = input.hasSizes;
  auto dim0 = op.dim0();
  auto dim1 = op.dim1();
  int64_t dim0Int;
  int64_t dim1Int;
  if (matchPattern(dim0, m_TorchConstantInt(&dim0Int)) &&
      matchPattern(dim1, m_TorchConstantInt(&dim1Int))) {
    knowledge.sizes = input.sizes;
    int64_t inputRank = input.sizes.size();
    dim0Int = toPositiveDim(dim0Int, inputRank);
    dim1Int = toPositiveDim(dim1Int, inputRank);
    if (isValidDim(dim0Int, inputRank) && isValidDim(dim1Int, inputRank)) {
      std::swap(knowledge.sizes[dim0Int], knowledge.sizes[dim1Int]);
      return getLatticeElement(op.getResult()).join(knowledge);
    }
  }

  knowledge.sizes.resize(input.sizes.size(), kUnknownSize);
  return getLatticeElement(op.getResult()).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenPermuteOp(
    AtenPermuteOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op.getContext());
  knowledge.dtype = input.dtype;
  if (!input.hasSizes)
    return getLatticeElement(op.getResult()).join(knowledge);
  knowledge.hasSizes = input.hasSizes;
  knowledge.sizes.resize(input.sizes.size(), kUnknownSize);
  Value dims = op.dims();
  SmallVector<int64_t> dimensions;
  if (matchPattern(dims, m_TorchConstantIntList(dimensions))) {
    int64_t inputRank = input.sizes.size();
    for (unsigned i = 0; i < inputRank; i++) {
      int64_t dim = toPositiveDim(dimensions[i], inputRank);
      if (!isValidDim(dim, inputRank)) {
        return getLatticeElement(op.getResult()).join(knowledge);
      }
      knowledge.sizes[i] = input.sizes[dim];
    }
  }
  return getLatticeElement(op.getResult()).join(knowledge);
}

template <typename OpTy>
ChangeResult TypeAnalyzer::visitScalarToTensorConversionOp(OpTy op) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op.getContext());
  Value t = op.t();
  Value dtype = op.dtype();
  knowledge.hasSizes = true;
  knowledge.sizes.resize(1, 1);
  fillInDTypeGivenDTypeAndDataType(knowledge, dtype, t.getType());
  return getLatticeElement(op.getResult()).join(knowledge);
}

// `torch.aten.tensor` get a tensor from a list. Each layer of the list
// corresponds to one dim of the tensor.
ChangeResult TypeAnalyzer::visitAtenTensorOp(AtenTensorOp op) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op.getContext());
  Value data = op.data();
  Value dtype = op.dtype();
  Type type = data.getType();
  int64_t rank = 0;
  while (auto listType = type.dyn_cast<ListType>()) {
    type = listType.getContainedType();
    rank++;
  }

  if (rank != 0) {
    knowledge.hasSizes = true;
    knowledge.sizes.resize(rank, kUnknownSize);
  }
  fillInDTypeGivenDTypeAndDataType(knowledge, dtype, type);
  return getLatticeElement(op.getResult()).join(knowledge);
}

template <typename OpTy>
ChangeResult TypeAnalyzer::visitConstantTensorAllocOp(OpTy op) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  fillInSizesGivenSizesList(knowledge, op.size());
  fillInDTypeGivenDTypeAndDataType(knowledge, op.dtype(),
                                   Torch::FloatType::get(op->getContext()));
  return getLatticeElement(op.getResult()).join(knowledge);
}

// Convert input tensor type to the given `dtype`.
ChangeResult TypeAnalyzer::visitAtenToDtypeOp(
    AtenToDtypeOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.hasSizes = input.hasSizes;
  if (input.hasSizes)
    knowledge.sizes = input.sizes;
  Value dtype = op.dtype();
  fillInDTypeGivenDTypeAndDataType(knowledge, dtype, input.dtype);
  return getLatticeElement(op.getResult()).join(knowledge);
}

// Convert input tensor type to the same as the other tensor.
template <typename OpTy>
ChangeResult TypeAnalyzer::visitTypeConversionOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.hasSizes = input.hasSizes;
  knowledge.sizes = input.sizes;
  Value other = op.other();
  BaseTensorType type = other.getType().cast<BaseTensorType>();
  if (type.hasDtype())
    knowledge.dtype = type.getDtype();
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

// The returned tensor has the same number of dimensions as the input tensor.
// The dimension specified by dim has size decided by \p setDim and other
// dimensions have the same size as in the original tensor.
template <typename OpTy>
ChangeResult TypeAnalyzer::visitSliceLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands,
    SetDimSizeFn setDim) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.dtype = input.dtype;
  if (!input.hasSizes)
    return getLatticeElement(op.getResult()).join(knowledge);

  knowledge.hasSizes = true;
  bool dimIsUnknown = false;
  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
    dimIsUnknown = true;
  else {
    int64_t inputRank = input.sizes.size();
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      dimIsUnknown = true;
  }

  if (dimIsUnknown) {
    knowledge.sizes.resize(input.sizes.size(), kUnknownSize);
    return getLatticeElement(op.getResult()).join(knowledge);
  }
  knowledge.sizes = input.sizes;
  setDim(knowledge.sizes[dim], dim, operands);
  return getLatticeElement(op.getResult()).join(knowledge);
}

// For `torch.aten.gather` input and index must have the same number of
// dimensions. Out will have the same shape as index. Note that input and index
// do not broadcast against each other.
ChangeResult TypeAnalyzer::visitAtenGatherOp(
    AtenGatherOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto index = operands[2]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.dtype = input.dtype;
  knowledge.hasSizes = index.hasSizes;
  knowledge.sizes = index.sizes;
  return getLatticeElement(op.getResult()).join(knowledge);
}

// A list is given for setting dims of the output tensor. Each item in the list
// corresponds to each dim and specifies how to transform the dim from input to
// the output.
ChangeResult TypeAnalyzer::visitExpandLikeOp(
    Operation *op, Value list,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands,
    SetDimSizePerListItemFn setDim) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.dtype = input.dtype;
  if (!input.hasSizes)
    return getLatticeElement(op->getResult(0)).join(knowledge);
  int64_t inputRank = input.sizes.size();

  SmallVector<Value, 4> listItems;
  if (!getListConstructElements(list, listItems))
    return getLatticeElement(op->getResult(0)).join(knowledge);
  int64_t listRank = listItems.size();
  knowledge.hasSizes = true;
  knowledge.sizes.resize(listRank, kUnknownSize);

  if (listRank < inputRank)
    return getLatticeElement(op->getResult(0)).join(knowledge);

  for (auto en : llvm::enumerate(listItems)) {
    int64_t listDim = en.index();
    // Given list rank could be larger than the inputRank, subtract the offset
    // to get the inputDim.
    int64_t inputDim = listDim - (listRank - inputRank);
    int64_t size;
    if (!matchPattern(en.value(), m_TorchConstantInt(&size)))
      continue;

    if (inputDim < 0) {
      knowledge.sizes[listDim] = size;
      continue;
    }

    setDim(knowledge.sizes[listDim], input.sizes[inputDim], size);
  }
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

// `torch.aten.cat` concatenates the given sequence of seq tensors in the given
// dimension. The output has the same sizes as the input for all dimensions
// except the given dimension.
ChangeResult TypeAnalyzer::visitAtenCatOp(
    AtenCatOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto tensorList = op.tensors();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  auto listConstruct = tensorList.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return getLatticeElement(op.getResult()).join(knowledge);

  auto tensors = llvm::to_vector<4>(
      llvm::map_range(listConstruct.elements(), [&](Value v) -> ValueKnowledge {
        return getLatticeElement(v).getValue();
      }));
  for (auto tensor : tensors)
    knowledge = ValueKnowledge::join(knowledge, tensor);
  if (!knowledge.hasSizes)
    return getLatticeElement(op.getResult()).join(knowledge);

  int64_t dim;
  bool dimIsUnknown = false;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    dimIsUnknown = true;
  } else {
    int64_t inputRank = knowledge.sizes.size();
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      dimIsUnknown = true;
  }

  if (dimIsUnknown) {
    knowledge.sizes.assign(knowledge.sizes.size(), kUnknownSize);
    return getLatticeElement(op.getResult()).join(knowledge);
  }
  knowledge.sizes[dim] = kUnknownSize;
  return getLatticeElement(op.getResult()).join(knowledge);
}

// Get the shape of the input tensor as a 1-D tensor.
ChangeResult TypeAnalyzer::visitAtenShapeAsTensorOp(
    Aten_ShapeAsTensorOp op,
    ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  if (input.hasSizes)
    knowledge.sizes.resize(1, input.sizes.size());
  else
    knowledge.sizes.push_back(kUnknownSize);
  knowledge.hasSizes = true;
  knowledge.dtype = IntegerType::get(op->getContext(), 64, IntegerType::Signed);
  return getLatticeElement(op.getResult()).join(knowledge);
}

ChangeResult TypeAnalyzer::visitNumToTensorOp(PrimNumToTensorScalarOp op) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.hasSizes = true;
  knowledge.dtype = getDefaultDtypeForTorchScalar(op.a().getType());
  return getLatticeElement(op.getResult()).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenEmbeddingOp(
    AtenEmbeddingOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  auto weight = operands[0]->getValue();
  auto indices = operands[1]->getValue();
  if (indices.hasSizes) {
    knowledge.hasSizes = true;
    knowledge.sizes = indices.sizes;
    // Weight's shape is [num_embedding, embedding_dim] and the last dim of the
    // output should also be embedding_dim.
    if (weight.hasSizes && weight.sizes.size() == 2)
      knowledge.sizes.push_back(weight.sizes[1]);
    else
      knowledge.sizes.push_back(kUnknownSize);
  }
  knowledge.dtype = Float32Type::get(op->getContext());
  return getLatticeElement(op.getResult()).join(knowledge);
}


// Common template for softmax like ops, eg., log_softmax.
template <typename OpTy>
ChangeResult TypeAnalyzer::visitAtenSoftmaxLikeOp(
    OpTy op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto input = operands[0]->getValue();
  auto dtype = op.dtype();
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  knowledge.hasSizes = input.hasSizes;
  knowledge.sizes = input.sizes;
  fillInDTypeGivenDTypeIntAndInputDType(knowledge, dtype, input.dtype);
  return getLatticeElement(op.getResult()).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenBmmOp(
    AtenBmmOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  auto self = operands[0]->getValue();
  auto mat2 = operands[1]->getValue();
  knowledge.sizes.resize(3, kUnknownSize);
  knowledge.dtype = getPromotedResultTypeAssumingNonZeroRank(op->getContext(),
                                                             {&self, &mat2});
  knowledge.hasSizes = true;
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

ChangeResult TypeAnalyzer::visitAtenMatmulOp(
    AtenMatmulOp op, ArrayRef<LatticeElement<ValueKnowledge> *> operands) {
  auto knowledge =
      ValueKnowledge::getNotNonePessimisticValueState(op->getContext());
  auto self = operands[0]->getValue();
  auto other = operands[1]->getValue();
  if (!self.hasSizes || !other.hasSizes)
    return getLatticeElement(op->getResult(0)).join(knowledge);
  unsigned maxRank = self.sizes.size() > other.sizes.size()
                         ? self.sizes.size()
                         : other.sizes.size();
  unsigned lhsDim = self.sizes.size() > 2 ? 2 : self.sizes.size();
  unsigned rhsDim = other.sizes.size() > 2 ? 2 : other.sizes.size();
  unsigned batchDim = maxRank > 2 ? maxRank - 2 : 0;
  unsigned matDim = (lhsDim - 1) + (rhsDim - 1);
  unsigned resultRank = batchDim + matDim;
  knowledge.sizes.resize(resultRank, kUnknownSize);
  knowledge.dtype = joinElementTypes(self.dtype, other.dtype);
  knowledge.hasSizes = true;
  return getLatticeElement(op->getResult(0)).join(knowledge);
}

// -----------------------------------------------------------------------------
// Transforms.
// -----------------------------------------------------------------------------

// Get a the most refined type compatible with ValueKnowledge, or null if that
// is not possible.
static Type getMostRefinedStaticType(Value v, TypeAnalyzer &analyzer) {
  auto getRefinedTensorType = [](BaseTensorType tensorType,
                                 ValueKnowledge const &knowledge) {
    return tensorType.getWithSizesAndDtype(
        knowledge.hasSizes ? llvm::makeArrayRef(knowledge.sizes)
                           : Optional<ArrayRef<int64_t>>(),
        knowledge.dtype);
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
    if (knowledge.optional == ValueKnowledge::OptionalKnowledge::isNone)
      return Torch::NoneType::get(v.getContext());
    else if (knowledge.optional == ValueKnowledge::OptionalKnowledge::notNone) {
      auto containedType = optionalType.getContainedType();
      if (auto tensorType = containedType.dyn_cast<BaseTensorType>())
        return getRefinedTensorType(tensorType, knowledge);
      else
        return containedType;
    }
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

void optimize(FuncOp func, TypeAnalyzer &analyzer) {
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
        } else if (originalType.isa<OptionalType>()) {
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

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::Torch::createRefineTypesPass() {
  return std::make_unique<RefineTypesPass>();
}
