//===- Algorithm.h - Main algorithm ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support types and utilities for the Cartesian Product Algorithm for
// Type Inference.
//
// See:
//   http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.30.8177
//   http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.129.2756
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TYPING_ANALYSIS_CPA_ALGORITHM_H
#define NPCOMP_TYPING_ANALYSIS_CPA_ALGORITHM_H

#include "npcomp/Typing/Analysis/CPA/Types.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace NPCOMP {
namespace Typing {
namespace CPA {

/// Propagates constraints in an environment.
class PropagationWorklist {
public:
  PropagationWorklist(Environment &env);

  /// Propagates any current constraints that match the transitivity rule:
  ///   τv <: t, t <: τ   (τv=ValueType, t=TypeVar, τ=TypeBase)
  /// Expanding to:
  ///   τv <: τ
  /// (τv=ValueType, t=TypeVar, τ=TypeBase)
  void propagateTransitivity();

  /// Commits the current round, returning true if any new constraints were
  /// added.
  bool commit();

private:
  Environment &env;
  llvm::DenseSet<Constraint *> currentConstraints;
  int newConstraintCount = 0;
};

/// Resolves all variables associated with a type node in a greedy fashion.
///
/// Given a TypeNode that may or may not have dependent variables, this
/// will recursively resolve all variables to concrete types by applying
/// a hook that reduces members of the type variable set to a singular
/// ValueType. The default hook will only allow memberships of 1 and performs
/// no union widening.
///
/// This greedy algorithm is fairly limited in what it can resolve. For example,
/// it cannot disambiguate two candidates like Array<?> and Array<i32> in a
/// robust way.
class GreedyTypeNodeVarResolver {
public:
  GreedyTypeNodeVarResolver(Context &context, MLIRContext &mlirContext,
                            llvm::Optional<Location> loc)
      : context(context), loc(loc) {}

  /// Analyzes TypeNode, adding any necessary variable mappings. On failure,
  /// an error will be emitted.
  LogicalResult analyzeTypeNode(TypeNode *tn);

  /// The mappings for all TypeVars. If all calls to analyzeTypeNode()
  /// succeeded, this should be sufficient to construct a concrete IR Type.
  TypeVarMap &getMappings() { return mappings; }

private:
  ValueType *unionCandidateTypes(const ValueTypeSet &candidates);

  // Initialization.
  Context &context;
  llvm::Optional<Location> loc;

  // Runtime state.
  TypeVarSet allVars;
  TypeVarMap mappings;
};

} // namespace CPA
} // namespace Typing
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_TYPING_ANALYSIS_CPA_ALGORITHM_H
