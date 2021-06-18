//===- RefineTypes.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

// -----------------------------------------------------------------------------
// Analysis.
// -----------------------------------------------------------------------------

static Type joinElementTypes(Type lhs, Type rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  if (lhs == rhs)
    return lhs;
  return Type();
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
  ValueKnowledge(bool hasSizes, std::vector<int64_t> sizes, Type dtype)
      : hasSizes(hasSizes), sizes(sizes), dtype(dtype) {
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
    }
    return result;
  }

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(false, {}, Type());
  }
  // Return a pessimistic/conservative value state only using knowlege already
  // recorded in the IR.
  static ValueKnowledge getPessimisticValueState(Value value) {
    return getKnowledgeFromType(value.getType());
  }

  bool operator==(const ValueKnowledge &rhs) const {
    return std::make_tuple(hasSizes, sizes, dtype) ==
           std::make_tuple(rhs.hasSizes, rhs.sizes, rhs.dtype);
  }

  // Given two pieces of static knowledge, calculate conservatively the
  // information we can be sure about.
  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs) {
    // Mental model: All conditions are checking how to change from the safe "no
    // knowledge" default-initialized state to a state with more knowledge
    // consistent with lhs and rhs.
    ValueKnowledge result = getPessimisticValueState(nullptr);

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
            AtenTanhOp, AtenBatchNormOp, AtenReluOp>(op)) {
      return getLatticeElement(op->getResult(0)).join(*operands[0]);
    }
    if (isa<AtenMmOp>(op)) {
      auto &lhs = operands[0]->getValue();
      auto &rhs = operands[1]->getValue();
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
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
      knowledge.dtype = joinElementTypes(lhs.dtype, rhs.dtype);
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<AtenLinearOp>(op)) {
      // The output shape is the input shape with the last dimension changed
      // to the weight's output dimension.
      auto knowledge = operands[0]->getValue();
      if (knowledge.hasSizes && knowledge.sizes.size() > 0)
        knowledge.sizes[knowledge.sizes.size() - 1] = kUnknownSize;
      // TODO: Handle case of bias being None gracefully. Requires a lattice
      // that tracks "None" (torch.optional). See also
      // DerefineOp::getCanonicalizationPatterns for more refinement that needs
      // to be done in this pass.
      knowledge.dtype = joinElementTypes(
          knowledge.dtype, joinElementTypes(operands[1]->getValue().dtype,
                                            operands[2]->getValue().dtype));
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<AtenConv2dOp>(op)) {
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      knowledge.hasSizes = true;
      knowledge.sizes.resize(4, kUnknownSize);
      // Running some experiments in PyTorch, the bias doesn't seem to
      // contribute to the final element type.
      knowledge.dtype = joinElementTypes(operands[0]->getValue().dtype,
                                         operands[1]->getValue().dtype);
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<AtenMaxPool2dOp>(op)) {
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      knowledge.hasSizes = true;
      knowledge.sizes.resize(4, kUnknownSize);
      knowledge.dtype = operands[0]->getValue().dtype;
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<AtenAdaptiveAvgPool2dOp>(op)) {
      auto input = operands[0]->getValue();
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      if (input.hasSizes) {
        knowledge.hasSizes = true;
        knowledge.sizes.resize(input.sizes.size(), kUnknownSize);
      }
      knowledge.dtype = input.dtype;
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (isa<AtenAddTensorOp>(op)) {
      // This is a general binary broadcasting shape transfer function.
      // We currently don't track "size 1" in our lattice, but we might want to.
      // We could make this more precise as well. But again, as with the other
      // shape transfer functions, handling the statically-invalid case is
      // tricky, so we defer that until we need it.
      auto lhs = operands[0]->getValue();
      auto rhs = operands[1]->getValue();
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      if (lhs.hasSizes && rhs.hasSizes) {
        knowledge.hasSizes = true;
        knowledge.sizes.resize(std::max(lhs.sizes.size(), rhs.sizes.size()),
                               kUnknownSize);
      }
      knowledge.dtype = joinElementTypes(lhs.dtype, rhs.dtype);
      return getLatticeElement(op->getResult(0)).join(knowledge);
    } else if (auto flatten = dyn_cast<AtenFlattenUsingIntsOp>(op)) {
      int64_t startDim;
      int64_t endDim;
      auto operand = operands[0]->getValue();
      auto knowledge =
          ValueKnowledge::getPessimisticValueState(op->getContext());
      knowledge.dtype = operand.dtype;
      if (operand.hasSizes && operand.sizes.size() == 0) {
        // Rank 0 is special and flattens to rank 1.
        knowledge.hasSizes = true;
        knowledge.sizes.push_back(kUnknownSize);
      } else if (operand.hasSizes &&
                 matchPattern(flatten.start_dim(),
                              m_TorchConstantInt(&startDim)) &&
                 matchPattern(flatten.end_dim(), m_TorchConstantInt(&endDim))) {
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

// Get a the most refined type compatible with ValueKnowledge, or null if that
// is not possible.
static Type getMostRefinedStaticType(Value v, TypeAnalyzer &analyzer) {
  if (auto tensorType = v.getType().dyn_cast<BaseTensorType>()) {
    LatticeElement<ValueKnowledge> *latticeElement =
        analyzer.lookupLatticeElement(v);
    if (!latticeElement)
      return nullptr;
    const ValueKnowledge &knowledge = latticeElement->getValue();
    return tensorType.getWithSizesAndDtype(
        knowledge.hasSizes ? llvm::makeArrayRef(knowledge.sizes)
                           : Optional<ArrayRef<int64_t>>(),
        knowledge.dtype);
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
static bool allowsTypeRefinementOrWillBeOtherwiseSafelyRefined(Operation *op) {
  return allowsTypeRefinement(op) ||
         isa<CopyToNonValueTensorOp, CopyToValueTensorOp>(op);
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
      // If we have an op that allows adding/removing static information from
      // this type, then we can rewrite. We make sure to always embed the static
      // information in the IR, and insert the minimal number of casts needed to
      // do so.
      // TODO: For some types, we will need 2 ops here: one to add static
      // information, and the other to remove static information.
      // (for example, torch.unchecked_cast / torch.derefine for torch.optional
      // types).
      std::function<Value(Location, Type, Value)> createStaticInfoCast;
      OpBuilder b(op->getBlock(), std::next(op->getIterator()));
      if (originalType.isa<BaseTensorType>()) {
        createStaticInfoCast = [&](Location loc, Type newType,
                                   Value v) -> Value {
          return b.create<TensorStaticInfoCastOp>(loc, newType, v);
        };
      }
      if (createStaticInfoCast) {
        // Save off the original uses to avoid iterator invalidation issues
        // or other unexpected behavior since we are creating new ops here that
        // use the value.
        auto originalUses = llvm::to_vector<6>(
            llvm::map_range(v.getUses(), [](OpOperand &use) { return &use; }));
        OpBuilder b(op->getBlock(), std::next(op->getIterator()));
        Value newTypedValue;
        // Always make sure that the new static information is reflected in the
        // IR, either by updating the type in place, or inserting a static info
        // cast.
        if (allowsTypeRefinementOrWillBeOtherwiseSafelyRefined(op)) {
          newTypedValue = v;
          v.setType(refinedType);
        } else {
          newTypedValue = createStaticInfoCast(op->getLoc(), refinedType, v);
        }

        Value oldTypedValue;
        for (OpOperand *use : originalUses) {
          // If the use can be updated to the new type directly, do it!
          if (allowsTypeRefinementOrWillBeOtherwiseSafelyRefined(
                  use->getOwner())) {
            use->set(newTypedValue);
            continue;
          }
          // If needed, create a value of the original type to appease users
          // that cannot accept the new type.
          if (!oldTypedValue) {
            oldTypedValue =
                createStaticInfoCast(op->getLoc(), originalType, newTypedValue);
          }
          use->set(oldTypedValue);
        }
      }
    }
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
mlir::NPCOMP::Torch::createRefineTypesPass() {
  return std::make_unique<RefineTypesPass>();
}
