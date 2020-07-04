//===- Algorith.cpp - Main algorithm --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Typing/Analysis/CPA/Algorithm.h"

using namespace mlir::NPCOMP::Typing::CPA;

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
          llvm::errs() << "-->ADD TRANS CONSTRAINT: ";
          newC->print(env.getContext(), llvm::errs());
          llvm::errs() << "\n";
          newConstraintCount += 1;
        }
      }
    }
  }
}
