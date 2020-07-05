//===- Algorith.cpp - Main algorithm --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Typing/Analysis/CPA/Algorithm.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cpa-type-inference"

using namespace mlir;
using namespace mlir::NPCOMP::Typing::CPA;

//------------------------------------------------------------------------------
// PropagationWorklist
//------------------------------------------------------------------------------

PropagationWorklist::PropagationWorklist(Environment &env) : env(env) {
  auto &contents = env.getConstraints();
  currentConstraints.reserve(contents.size() * 2);
  for (auto *c : contents) {
    currentConstraints.insert(c);
  }
}

bool PropagationWorklist::commit() {
  bool hadNew = newConstraintCount > 0;
  newConstraintCount = 0;
  return hadNew;
}

void PropagationWorklist::propagateTransitivity() {
  // Prepare for join.
  constexpr size_t N = 8;
  llvm::DenseMap<TypeVar *, llvm::SmallVector<ValueType *, N>> varToValueType;
  llvm::DenseMap<TypeVar *, llvm::SmallVector<TypeNode *, N>> varToAny;
  for (auto *c : currentConstraints) {
    auto *lhsVar = llvm::dyn_cast<TypeVar>(c->getFrom());
    auto *rhsVar = llvm::dyn_cast<TypeVar>(c->getTo());

    if (lhsVar) {
      varToAny[lhsVar].push_back(c->getTo());
    }
    if (rhsVar) {
      if (auto *vt = llvm::dyn_cast<ValueType>(c->getFrom())) {
        varToValueType[rhsVar].push_back(vt);
      }
    }
  }

  // Expand join.
  for (auto vtIt : varToValueType) {
    auto &lhsSet = vtIt.second;
    auto anyIt = varToAny.find(vtIt.first);
    if (anyIt == varToAny.end())
      continue;
    auto &rhsSet = anyIt->second;

    for (ValueType *lhsItem : lhsSet) {
      for (TypeNode *rhsItem : rhsSet) {
        Constraint *newC = env.getContext().getConstraint(lhsItem, rhsItem);
        if (currentConstraints.insert(newC).second) {
          LLVM_DEBUG(llvm::dbgs() << "-->ADD TRANS CONSTRAINT: ";
                     newC->print(env.getContext(), llvm::dbgs());
                     llvm::dbgs() << "\n";);
          newConstraintCount += 1;
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// GreedyTypeNodeVarResolver
//------------------------------------------------------------------------------

ValueType *
GreedyTypeNodeVarResolver::unionCandidateTypes(const ValueTypeSet &candidates) {
  if (candidates.empty()) {
    mlir::emitOptionalError(loc, "no candidate types were identified");
    return nullptr;
  }
  if (candidates.size() != 1) {
    mlir::emitOptionalError(loc, "ambiguous candidate types were identified");
    return nullptr;
  }

  return *candidates.begin();
}

LogicalResult GreedyTypeNodeVarResolver::analyzeTypeNode(TypeNode *tn) {
  TypeVarSet newVars;
  tn->collectDependentTypeVars(context, newVars);
  if (newVars.empty())
    return success();

  // Breadth-first resolution of vars (that do not depend on other vars).
  ValueTypeSet pendingValueTypes;
  for (TypeVar *newTv : newVars) {
    if (!allVars.insert(newTv).second)
      continue;
    if (mappings.count(newTv) == 1)
      continue;

    // Known mappings to this TypeVar.
    auto &existingMembers = context.getMembers(newTv);
    ValueTypeSet members(existingMembers.begin(), existingMembers.end());

    ValueType *concreteVt = unionCandidateTypes(members);
    if (!concreteVt)
      return failure();
    mappings[newTv] = concreteVt;
    pendingValueTypes.insert(concreteVt);
  }

  // Recursively analyze any newly discovered concrete types.
  for (ValueType *nextValueType : pendingValueTypes) {
    if (failed(analyzeTypeNode(nextValueType)))
      return failure();
  }

  return success();
}
