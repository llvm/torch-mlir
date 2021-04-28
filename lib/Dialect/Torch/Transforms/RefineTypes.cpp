//===- RefineTypes.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

// -----------------------------------------------------------------------------
// Analysis.
// -----------------------------------------------------------------------------

constexpr int64_t kUnknownSize = -1;

static Type joinElementTypes(Type lhs, Type rhs) {
  if (lhs.isa<Numpy::AnyDtypeType>())
    return rhs;
  if (rhs.isa<Numpy::AnyDtypeType>())
    return lhs;
  if (lhs == rhs)
    return lhs;
  return Numpy::AnyDtypeType::get(lhs.getContext());
}

namespace {
// Statically known information for a particular Value.
//
// This struct currently tracks only information relevant for tensor/array-like
// shaped types. It is fine to associate a `ValueKnowledge` with a non-shaped
// type as long as it is in the default "no knowledge" state returned by
// `getPessimisticValueState`. The important invariant is that we cannot
// claim to know something about a value which is false.
//
// This class could also be called "dataflow facts", "lattice value", etc.
struct ValueKnowledge {
  ValueKnowledge() = delete;
  // We enforce that `elementType` is always a valid type (possibly
  // !numpy.any_dtype), and `sizes` is empty unless `hasRank`.
  // So default constructing is prohibited.
  ValueKnowledge(bool hasRank, std::vector<int64_t> sizes, Type elementType)
      : hasRank(hasRank), sizes(sizes), elementType(elementType) {
    assert(elementType != nullptr);
    assert(sizes.size() == 0 || hasRank);
  }

  // Get the static knowledge intrinsic to `type`.
  static ValueKnowledge getKnowledgeFromType(Type type) {
    ValueKnowledge result = getPessimisticValueState(type.getContext());
    if (auto tensorType = type.dyn_cast<TensorType>()) {
      if (tensorType.hasRank()) {
        result.hasRank = true;
        result.sizes = tensorType.getShape().vec();
      }
      result.elementType = tensorType.getElementType();
      return result;
    }
    return result;
  }

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(false, {}, Numpy::AnyDtypeType::get(context));
  }
  // Return a pessimistic/conservative value state only using knowlege already
  // recorded in the IR.
  static ValueKnowledge getPessimisticValueState(Value value) {
    return getKnowledgeFromType(value.getType());
  }

  bool operator==(const ValueKnowledge &rhs) const {
    return std::make_tuple(hasRank, sizes, elementType) ==
           std::make_tuple(rhs.hasRank, rhs.sizes, rhs.elementType);
  }

  // Given two pieces of static knowledge, calculate conservatively the
  // information we can be sure about.
  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs) {
    // Mental model: All conditions are checking how to change from the safe "no
    // knowledge" default-initialized state to a state with more knowledge
    // consistent with lhs and rhs.
    ValueKnowledge result =
        getPessimisticValueState(lhs.elementType.getContext());

    if (lhs.hasRank && !rhs.hasRank) {
      result.hasRank = true;
      result.sizes = lhs.sizes;
    } else if (!lhs.hasRank && rhs.hasRank) {
      result.hasRank = true;
      result.sizes = rhs.sizes;
    } else if (lhs.hasRank && rhs.hasRank &&
               lhs.sizes.size() == rhs.sizes.size()) {
      result.hasRank = true;
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

    result.elementType = joinElementTypes(lhs.elementType, rhs.elementType);
    return result;
  }

  // Whether the Value is known to have a rank.
  bool hasRank;
  // If `hasRank` the sizes along each rank. Unknown sizes are represented as
  // `kUnknownSize`.
  std::vector<int64_t> sizes;
  // The element type of a shaped type.
  // This is equal to !numpy.any_dtype if it is not a concrete type.
  Type elementType;
};

// static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ValueKnowledge
// &knowledge) {
//   os << "hasRank = " << knowledge.hasRank << ", sizes = [";
//   llvm::interleaveComma(knowledge.sizes, os);
//   os << "]"
//      << ", elementType = " << knowledge.elementType;
//   return os;
// }

// Forward intraprocedural dataflow for type information.
class TypeAnalyzer : public ForwardDataFlowAnalysis<ValueKnowledge> {
public:
  using ForwardDataFlowAnalysis<ValueKnowledge>::ForwardDataFlowAnalysis;

  // Compute the knowledge for the results of an op, based on the knowledge of
  // the operands and any information intrinsic to `op`.
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<ValueKnowledge> *> operands) final {
    if (isa<Numpy::TensorStaticInfoCastOp, Numpy::CopyToTensorOp,
            Numpy::CreateArrayFromTensorOp, aten::TanhOp, aten::BatchNormOp,
            aten::ReluOp>(op)) {
      return getLatticeElement(op->getResult(0)).join(*operands[0]);
    }
    if (isa<aten::MmOp>(op)) {
      auto &lhs = operands[0]->getValue();
      auto &rhs = operands[1]->getValue();
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      knowledge.hasRank = true;
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
      // this pass, we will (correctly!) infer `lhs.hasRank && lhs.sizes.size()
      // == 0 && rhs.hasRank && rhs.sizes.size() == 0` -- it's not safe to
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
      knowledge.elementType =
          joinElementTypes(lhs.elementType, rhs.elementType);
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<aten::LinearOp>(op)) {
      // The output shape is the input shape with the last dimension changed
      // to the weight's output dimension.
      auto knowledge = operands[0]->getValue();
      if (knowledge.hasRank && knowledge.sizes.size() > 0)
        knowledge.sizes[knowledge.sizes.size() - 1] = kUnknownSize;
      // TODO: Handle case of bias being None gracefully. Requires a lattice
      // that tracks "None" (torch.optional). See also
      // DerefineOp::getCanonicalizationPatterns for more refinement that needs
      // to be done in this pass.
      knowledge.elementType = joinElementTypes(
          knowledge.elementType,
          joinElementTypes(operands[1]->getValue().elementType,
                           operands[2]->getValue().elementType));
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<aten::Conv2dOp>(op)) {
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      knowledge.hasRank = true;
      knowledge.sizes.resize(4, kUnknownSize);
      // Running some experiments in PyTorch, the bias doesn't seem to
      // contribute to the final element type.
      knowledge.elementType =
          joinElementTypes(operands[0]->getValue().elementType,
                           operands[1]->getValue().elementType);
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<aten::MaxPool2dOp>(op)) {
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      knowledge.hasRank = true;
      knowledge.sizes.resize(4, kUnknownSize);
      knowledge.elementType = operands[0]->getValue().elementType;
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<aten::AdaptiveAvgPool2dOp>(op)) {
      auto input = operands[0]->getValue();
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      if (input.hasRank) {
        knowledge.hasRank = true;
        knowledge.sizes.resize(input.sizes.size(), kUnknownSize);
      }
      knowledge.elementType = input.elementType;
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<aten::AddOp>(op)) {
      // This is a general binary broadcasting shape transfer function.
      // We currently don't track "size 1" in our lattice, but we might want to.
      // We could make this more precise as well. But again, as with the other
      // shape transfer functions, handling the statically-invalid case is
      // tricky, so we defer that until we need it.
      auto lhs = operands[0]->getValue();
      auto rhs = operands[1]->getValue();
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      if (lhs.hasRank && rhs.hasRank) {
        knowledge.hasRank = true;
        knowledge.sizes.resize(std::max(lhs.sizes.size(), rhs.sizes.size()),
                               kUnknownSize);
      }
      knowledge.elementType =
          joinElementTypes(lhs.elementType, rhs.elementType);
      return getLatticeElement(op->getResult(0)).join(knowledge);
    }
    // Otherwise, this is an unknown operation. Just mark all results as having
    // reached a pessimistic fixpoint.
    return markAllPessimisticFixpoint(op->getResults());
  }
};
} // namespace

// -----------------------------------------------------------------------------
// Transforms.
// -----------------------------------------------------------------------------

// Get the most refined TensorType compatible with ValueKnowledge.
static TensorType
getTensorTypeFromKnowledge(MLIRContext *context,
                           LatticeElement<ValueKnowledge> *knowledge) {
  if (!knowledge)
    return UnrankedTensorType::get(Numpy::AnyDtypeType::get(context));

  const ValueKnowledge &value = knowledge->getValue();
  if (!value.hasRank)
    return UnrankedTensorType::get(value.elementType);
  return RankedTensorType::get(value.sizes, value.elementType);
}

// Get a the most refined type compatible with ValueKnowledge, or null if that
// is not possible.
static Type getMostRefinedStaticType(Value v, TypeAnalyzer &analyzer) {
  if (v.getType().isa<TensorType>())
    return getTensorTypeFromKnowledge(v.getContext(),
                                      analyzer.lookupLatticeElement(v));
  // TODO: Support !numpy.ndarray type.
  return nullptr;
}

void optimize(FuncOp func, TypeAnalyzer &analyzer) {
  func.walk([&](Operation *op) {
    for (Value v : op->getResults()) {
      Type refinedType = getMostRefinedStaticType(v, analyzer);
      Type originalType = v.getType();
      // No type? Nothing to do.
      if (!refinedType)
        continue;
      // Type is same as existing one? Nothing to do.
      if (refinedType == originalType)
        continue;
      // If the type is a TensorType, then we have numpy.tensor_static_info_cast
      // to add/remove the static information. We make sure to always embed the
      // static information in the IR, and insert the minimal number of casts
      // needed to do so.
      // TODO: This logic should generalize easily to other types. We just
      // need to know which op allows us to add static information and which op
      // allows us to remove static information (in this case, one op allows
      // both).
      if (originalType.isa<TensorType>()) {
        // Save off the original uses to avoid iterator invalidation issues
        // or other unexpected behavior since we are creating new ops here that
        // use the value.
        auto originalUses = llvm::to_vector<6>(
            llvm::map_range(v.getUses(), [](OpOperand &use) { return &use; }));
        OpBuilder b(op->getBlock(), std::next(op->getIterator()));
        Value newTypedValue;
        // Always make sure that the new static information is reflected in the
        // IR, either by updating the type in place, or inserting a
        // numpy.tensor_static_info_cast op.
        if (allowsTypeRefinement(op)) {
          newTypedValue = v;
          v.setType(refinedType);
        } else {
          newTypedValue = b.create<Numpy::TensorStaticInfoCastOp>(
              op->getLoc(), refinedType, v);
        }

        Value oldTypedValue;
        for (OpOperand *use : originalUses) {
          // If the use can be updated to the new type directly, do it!
          if (allowsTypeRefinement(use->getOwner())) {
            use->set(newTypedValue);
            continue;
          }
          // If needed, create a value of the original type to appease users
          // that cannot accept the new type.
          if (!oldTypedValue) {
            oldTypedValue = b.create<Numpy::TensorStaticInfoCastOp>(
                op->getLoc(), originalType, newTypedValue);
          }
          use->set(oldTypedValue);
        }
      }
    }
  });
}

namespace {
class RefineTypesPass : public RefineTypesBase<RefineTypesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Numpy::NumpyDialect, aten::ATenDialect>();
  }
  void runOnOperation() override {
    auto func = getOperation();
    TypeAnalyzer analyzer(&getContext());
    analyzer.run(func);
    optimize(func, analyzer);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Torch::createRefineTypesPass() {
  return std::make_unique<RefineTypesPass>();
}
