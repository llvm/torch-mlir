//===- RefineTypes.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

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

namespace {
// Statically known information for a particular Value.
//
// This struct currently tracks only information relevant for tensor/array-like
// shaped types. It is fine to associate a `ValueKnowledge` with a non-shaped
// type as long as it is in the default "no knowledge" state returned by
// `getMostConservativeKnowledge`. The important invariant is that we cannot
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

  // Get a safe "most conservative knowledge" default state.
  static ValueKnowledge getMostConservativeKnowledge(MLIRContext *context) {
    return ValueKnowledge(false, {}, Numpy::AnyDtypeType::get(context));
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
} // namespace

bool operator==(const ValueKnowledge &lhs, const ValueKnowledge &rhs) {
  return std::make_tuple(lhs.hasRank, lhs.sizes, lhs.elementType) ==
         std::make_tuple(rhs.hasRank, rhs.sizes, rhs.elementType);
}

// static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ValueKnowledge &knowledge) {
//   os << "hasRank = " << knowledge.hasRank << ", sizes = [";
//   llvm::interleaveComma(knowledge.sizes, os);
//   os << "]"
//      << ", elementType = " << knowledge.elementType;
//   return os;
// }

// Given two pieces of static knowledge, calculate conservatively the
// information we can be sure about.
ValueKnowledge join(const ValueKnowledge &lhs, const ValueKnowledge &rhs) {
  // Mental model: All conditions are checking how to change from the safe "no
  // knowledge" default-initialized state to a state with more knowledge
  // consistent with lhs and rhs.
  ValueKnowledge result = ValueKnowledge::getMostConservativeKnowledge(
      lhs.elementType.getContext());

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

  if (!lhs.elementType || lhs.elementType.isa<Numpy::AnyDtypeType>()) {
    result.elementType = rhs.elementType;
  } else if (!rhs.elementType || rhs.elementType.isa<Numpy::AnyDtypeType>()) {
    result.elementType = lhs.elementType;
  } else if (lhs.elementType == rhs.elementType) {
    result.elementType = lhs.elementType;
  }
  return result;
}

// Get the static knowledge intrinsic to `type`.
ValueKnowledge getKnowledgeFromType(Type type) {
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    ValueKnowledge result =
        ValueKnowledge::getMostConservativeKnowledge(type.getContext());
    if (tensorType.hasRank()) {
      result.hasRank = true;
      result.sizes = tensorType.getShape().vec();
    }
    result.elementType = tensorType.getElementType();
    return result;
  }
  return ValueKnowledge::getMostConservativeKnowledge(type.getContext());
}

// Simple forward intraprocedural dataflow for type information.
class TypeAnalyzer {
public:
  TypeAnalyzer(MLIRContext *context) : context(context) {}
  void propagate(Region &region);
  // Get the knowledge that is known about `v`.
  ValueKnowledge &getKnowledge(Value v);

private:
  // Incorporate `knowledge` into what is known about `v`.
  // Return true if new knowledge was obtained about `v`.
  bool incorporateKnowledge(Value v, ValueKnowledge knowledge);
  MLIRContext *context;
  DenseMap<Value, ValueKnowledge> facts;
};

void TypeAnalyzer::propagate(Region &region) {
  bool changed;
  do {
    changed = false;
    // TODO: Find out why region.walk doesn't walk the blocks.
    for (Block &block : region) {
      for (Value v : block.getArguments())
        changed |= incorporateKnowledge(v, getKnowledgeFromType(v.getType()));
      for (Operation &op : block.getOperations()) {
        for (Value v : op.getResults())
          changed |= incorporateKnowledge(v, getKnowledgeFromType(v.getType()));
        if (isa<Numpy::TensorStaticInfoCastOp, aten::TanhOp>(op)) {
          changed |= incorporateKnowledge(op.getResult(0),
                                          getKnowledge(op.getOperand(0)));
        }
      }
    };
  } while (changed);
}

ValueKnowledge &TypeAnalyzer::getKnowledge(Value v) {
  auto p =
      facts.insert({v, ValueKnowledge::getMostConservativeKnowledge(context)});
  return p.first->second;
}

bool TypeAnalyzer::incorporateKnowledge(Value v, ValueKnowledge knowledge) {
  ValueKnowledge &currentKnowledge = getKnowledge(v);
  ValueKnowledge updatedKnowledge = join(currentKnowledge, knowledge);
  assert(join(updatedKnowledge, knowledge) == updatedKnowledge &&
         "nonmonotonic!");
  assert(join(updatedKnowledge, currentKnowledge) == updatedKnowledge &&
         "nonmonotonic!");
  if (currentKnowledge == updatedKnowledge)
    return false;
  currentKnowledge = updatedKnowledge;
  return true;
}

// -----------------------------------------------------------------------------
// Transforms.
// -----------------------------------------------------------------------------

// Get the most refined TensorType compatible with ValueKnowledge.
static TensorType getTensorTypeFromKnowledge(MLIRContext *context,
                                             ValueKnowledge &knowledge) {
  Type elementType = knowledge.elementType ? knowledge.elementType
                                           : Numpy::AnyDtypeType::get(context);
  if (!knowledge.hasRank)
    return UnrankedTensorType::get(elementType);
  return RankedTensorType::get(knowledge.sizes, elementType);
}

// Get a the most refined type compatible with ValueKnowledge, or null if that
// is not possible.
static Type getMostRefinedStaticType(Value v, TypeAnalyzer &analyzer) {
  if (v.getType().isa<TensorType>())
    return getTensorTypeFromKnowledge(v.getContext(), analyzer.getKnowledge(v));
  // TODO: Support !numpy.ndarray type.
  return nullptr;
}

// Return true whether a type `v` can have its type updated in place.
// This is a function of the value itself and also its users.
static bool canUpdateTypeInPlace(Value v) {
  // TODO: There are really two different predicates here, which need to be
  // properly interface-ized or otherwise make pluggable.
  // 1. Whether an operation allows its result to be refined to a certain type.
  // 2. Whether an operand of an operation can be refined to a certain
  // type.
  //
  // A simple first step that probably is enough in practice is a simple trait
  // AllowsTypeRefinement which answers yes to both questions. In general, an op
  // might allow refinement of some operands/results but not others, but that
  // seems unlikely.
  //
  // Currently, we answer both with the same logic, which is just enough for our
  // e2e bringup.
  Dialect *atenDialect = v.getContext()->getOrLoadDialect<aten::ATenDialect>();
  auto canValueIntrinsicallyBeUpdated = [&](Value v) {
    // TODO: Update block arguments.
    if (v.isa<BlockArgument>())
      return false;
    Operation *op = v.cast<OpResult>().getOwner();
    if (op->getDialect() == atenDialect)
      return true;
    if (isa<Numpy::TensorStaticInfoCastOp>(op))
      return true;
    return false;
  };
  // TODO: Handle BranchOpInterface and RegionBranchOpInterface ops.
  return canValueIntrinsicallyBeUpdated(v) &&
         llvm::all_of(v.getUses(), [&](OpOperand &use) {
           Operation *user = use.getOwner();
           if (user->getDialect() == atenDialect)
             return true;
           if (isa<Numpy::TensorStaticInfoCastOp>(user))
             return true;
           return false;
         });
}

void optimize(FuncOp func, TypeAnalyzer &analyzer) {
  func.walk([&](Operation *op) {
    for (Value v : op->getResults()) {
      Type refinedType = getMostRefinedStaticType(v, analyzer);
      // No type? Nothing to do.
      if (!refinedType)
        return;
      // Type is same as existing one? Nothing to do.
      if (refinedType == v.getType())
        return;
      if (canUpdateTypeInPlace(v)) {
        // Update type in place if possible.
        v.setType(refinedType);
      } else if (v.getType().isa<TensorType>() && v.isa<OpResult>()) {
        // Update the type in place, and cast the information way explicitly so
        // that users observe the original type.
        // TODO: Support updating !numpy.ndarray type too.
        // TODO: Support updating block arguments.
        // TODO: This update could be more fine-grained (RAUW with per-use
        // canUpdateTypeInPlace information), or to get we could get the same
        // effect by implementing a canonicalization that uses the fine-grained
        // information used by canUpdateTypeInPlace to bypass
        // numpy.tensor_static_info_cast when the consuming op is ok with that.
        Operation *op = v.cast<OpResult>().getOwner();
        OpBuilder builder(op->getBlock(), std::next(op->getIterator()));
        auto staticInfoCast = builder.create<Numpy::TensorStaticInfoCastOp>(
            op->getLoc(), v.getType(), v);
        SmallPtrSet<Operation *, 1> exceptions;
        exceptions.insert(staticInfoCast);
        v.replaceAllUsesExcept(staticInfoCast, exceptions);
        v.setType(refinedType);
      }
    }
  });
}

namespace {
class RefineTypesPass : public RefineTypesBase<RefineTypesPass> {
  void runOnOperation() override {
    auto func = getOperation();
    TypeAnalyzer analyzer(&getContext());
    analyzer.propagate(func.getRegion());
    optimize(func, analyzer);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Torch::createRefineTypesPass() {
  return std::make_unique<RefineTypesPass>();
}
