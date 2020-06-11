//===- TypeInference.cpp - Type inference passes -----------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Basicpy/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "basicpy-type-inference"

using namespace llvm;
using namespace mlir;
using namespace mlir::NPCOMP::Basicpy;

namespace {

/// Value type wrapping a type node.
class TypeNode : public ilist_node<TypeNode> {
public:
  enum class Discrim {
    CONST_TYPE,
    VAR_ORDINAL,
  };

  TypeNode(Value def, Type constType)
      : def(def), select(constType), discrim(Discrim::CONST_TYPE) {}
  TypeNode(Value def, unsigned varOrdinal)
      : def(def), select(varOrdinal), discrim(Discrim::VAR_ORDINAL) {}

  bool operator==(const TypeNode &other) const {
    if (discrim != other.discrim)
      return false;
    switch (discrim) {
    case Discrim::CONST_TYPE:
      return select.constType == other.select.constType;
    case Discrim::VAR_ORDINAL:
      return select.varOrdinal == other.select.varOrdinal;
    }
    return false;
  }

  Value getDef() const { return def; }
  Discrim getDiscrim() const { return discrim; }

  Type getConstType() const {
    assert(discrim == Discrim::CONST_TYPE);
    return select.constType;
  }

  unsigned getVarOrdinal() const {
    assert(discrim == Discrim::VAR_ORDINAL);
    return select.varOrdinal;
  }

private:
  Value def;
  union Select {
    Select(Type constType) : constType(constType) {}
    Select(unsigned varOrdinal) : varOrdinal(varOrdinal) {}
    Type constType;
    unsigned varOrdinal;
  } select;
  Discrim discrim;
};

/// A type equation, representing expected equality of types.
/// If the equation is derived from an operation, it is preserved for debugging
/// and messaging.
class TypeEquation : public ilist_node<TypeEquation> {
public:
  TypeEquation(TypeNode *left, TypeNode *right, Operation *context)
      : left(left), right(right), context(context) {}

  TypeNode *getLeft() const { return left; }
  TypeNode *getRight() const { return right; }
  Operation *getContext() const { return context; }

private:
  TypeNode *left;
  TypeNode *right;
  Operation *context;
};

raw_ostream &operator<<(raw_ostream &os, const TypeNode &tn) {
  switch (tn.getDiscrim()) {
  case TypeNode::Discrim::CONST_TYPE:
    os << "CONST(" << tn.getConstType() << ")";
    break;
  case TypeNode::Discrim::VAR_ORDINAL:
    os << "VAR(" << tn.getVarOrdinal() << ")";
    break;
  }
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const TypeEquation &eq) {
  os << "[TypeEq left=<" << *eq.getLeft() << ">, right=<" << *eq.getRight()
     << ">: " << *eq.getContext() << "]";
  return os;
}

/// Container for constructing type equations in an HM-like type inference
/// setup.
///
/// As a first pass, every eligible Value (def) is assigned either a Type or
/// a TypeVar (placeholder).
class TypeEquations {
public:
  TypeEquations() = default;

  // Gets a type node for the given def, creating it if necessary.
  TypeNode *getTypeNode(Value def) {
    TypeNode *&typeNode = defToNodeMap[def];
    if (typeNode)
      return typeNode;

    if (def.getType().isa<UnknownType>()) {
      // Type variable.
      typeNode = createTypeVar(def);
    } else {
      // Constant type.
      typeNode = createConstType(def, def.getType());
    }
    return typeNode;
  }

  template <typename... Args> TypeEquation *addEquation(Args &&... args) {
    TypeEquation *eq = allocator.Allocate<TypeEquation>(1);
    new (eq) TypeEquation(std::forward<Args>(args)...);
    equations.push_back(*eq);
    return eq;
  }

  /// Adds an equality equation for two defs, creating type nodes if necessary.
  void addTypeEqualityEquation(Value def1, Value def2, Operation *context) {
    addEquation(getTypeNode(def1), getTypeNode(def2), context);
  }

  /// Print a report of the equations for debugging.
  void report(raw_ostream &os) {
    os << "Type equations:\n";
    os << "---------------\n";
    for (auto &eq : equations) {
      os << "  : " << eq << "\n";
    }
  }

  simple_ilist<TypeEquation> &getEquations() { return equations; }

  void applySubst(unsigned ordinal, TypeNode *resolved) {
    assert(ordinal >= 0 && ordinal < ordinalToVarNode.size());
    Type constType = resolved->getConstType();
    TypeNode *varNode = ordinalToVarNode[ordinal];
    varNode->getDef().setType(constType);
  }

private:
  TypeNode *createConstType(Value def, Type constType) {
    TypeNode *n = allocator.Allocate<TypeNode>(1);
    new (n) TypeNode(def, constType);
    nodes.push_back(*n);
    return n;
  }

  TypeNode *createTypeVar(Value def) {
    TypeNode *n = allocator.Allocate<TypeNode>(1);
    new (n) TypeNode(def, nextOrdinal++);
    nodes.push_back(*n);
    ordinalToVarNode.push_back(n);
    assert(ordinalToVarNode.size() == nextOrdinal);
    return n;
  }

  BumpPtrAllocator allocator;
  simple_ilist<TypeNode> nodes;
  simple_ilist<TypeEquation> equations;
  llvm::DenseMap<Value, TypeNode *> defToNodeMap;
  llvm::SmallVector<TypeNode *, 16> ordinalToVarNode;
  unsigned nextOrdinal = 0;
};

/// (Very) simple type unification. This really isn't advanced enough for
/// anything beyond simple, unambiguous programs.
/// It is also terribly inefficient.
class TypeUnifier {
public:
  using SubstMap = llvm::DenseMap<unsigned, TypeNode *>;

  Optional<SubstMap> unifyEquations(TypeEquations &equations) {
    Optional<SubstMap> subst;
    subst.emplace();

    for (auto &eq : equations.getEquations()) {
      subst = unify(eq.getLeft(), eq.getRight(), std::move(subst));
      if (!subst) {
        break;
      }
    }

    return subst;
  }

  Optional<SubstMap> unify(TypeNode *typeX, TypeNode *typeY,
                           Optional<SubstMap> subst) {
    LLVM_DEBUG(llvm::dbgs() << "+ UNIFY: " << *typeX << ", " << *typeY << "\n");
    if (!subst) {
      emitError(typeX->getDef().getLoc()) << "cannot unify type";
      emitRemark(typeY->getDef().getLoc()) << "conflicting expression here";
      return None;
    } else if (*typeX == *typeY) {
      return subst;
    } else if (typeX->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      return unifyVariable(typeX, typeY, std::move(*subst));
    } else if (typeY->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      return unifyVariable(typeY, typeX, std::move(*subst));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  Unify fallthrough\n");
      return None;
    }
  }

  Optional<SubstMap> unifyVariable(TypeNode *varNode, TypeNode *typeNode,
                                   SubstMap subst) {
    assert(varNode->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL);
    // Var node in subst?
    auto it = subst.find(varNode->getVarOrdinal());
    if (it != subst.end()) {
      TypeNode *found = it->second;
      return unify(found, typeNode, std::move(subst));
    }

    // Type node in subst?
    if (typeNode->getDiscrim() == TypeNode::Discrim::VAR_ORDINAL) {
      it = subst.find(typeNode->getVarOrdinal());
      if (it != subst.end()) {
        TypeNode *found = it->second;
        return unify(varNode, found, std::move(subst));
      }
    }

    // varNode is not yet in subst and cannot simplify typeNode. Extend.
    subst[varNode->getVarOrdinal()] = typeNode;
    return std::move(subst);
  }
};

class TypeEquationPopulator {
public:
  TypeEquationPopulator(TypeEquations &equations) : equations(equations) {}

  /// If a return op was visited, this will be one of them.
  Operation *getLastReturnOp() { return funcReturnOp; }

  LogicalResult runOnFunction(FuncOp funcOp) {
    // Iterate and create type nodes for entry block arguments, as these
    // must be resolved no matter what.
    if (funcOp.getBody().empty())
      return success();
    auto &entryBlock = funcOp.getBody().front();
    for (auto blockArg : entryBlock.getArguments()) {
      equations.getTypeNode(blockArg);
    }

    // Then walk ops, creating equations.
    auto result = funcOp.walk([&](Operation *childOp) -> WalkResult {
      if (childOp == funcOp)
        return WalkResult::advance();

      // Trait based equations.
      // ----------------------
      // Function returns must all have the same types.
      if (childOp->hasTrait<OpTrait::ReturnLike>() &&
          childOp->getParentOp() == funcOp) {
        if (funcReturnOp) {
          if (funcReturnOp->getNumOperands() != childOp->getNumOperands()) {
            childOp->emitOpError() << "different arity of function returns";
            return WalkResult::interrupt();
          }
          for (auto it :
               llvm::zip(funcReturnOp->getOperands(), childOp->getOperands())) {
            equations.addTypeEqualityEquation(std::get<0>(it), std::get<1>(it),
                                              childOp);
          }
        }
        funcReturnOp = childOp;
        return WalkResult::advance();
      }

      // Ensure that constant nodes get assigned a constant type.
      if (childOp->hasTrait<OpTrait::ConstantLike>()) {
        equations.getTypeNode(childOp->getResult(0));
        return WalkResult::advance();
      }

      // Special op handling.
      // Many of these (that are not standard ops) should become op interfaces.
      // --------------------
      if (auto op = dyn_cast<UnknownCastOp>(childOp)) {
        equations.addTypeEqualityEquation(op.operand(), op.result(), op);
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<BinaryExprOp>(childOp)) {
        // TODO: This should really be applying arithmetic promotion, not
        // strict equality.
        equations.addTypeEqualityEquation(op.left(), op.right(), op);
        equations.addTypeEqualityEquation(op.left(), op.result(), op);
        return WalkResult::advance();
      }

      childOp->emitWarning() << "unhandled op in type inference";

      return WalkResult::advance();
    });

    return success(result.wasInterrupted());
  }

private:
  // The last encountered ReturnLike op.
  Operation *funcReturnOp = nullptr;
  TypeEquations &equations;
};

class FunctionTypeInferencePass
    : public FunctionTypeInferenceBase<FunctionTypeInferencePass> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    if (func.getBody().empty())
      return;

    TypeEquations equations;
    TypeEquationPopulator p(equations);
    p.runOnFunction(func);
    LLVM_DEBUG(equations.report(llvm::dbgs()));

    TypeUnifier unifier;
    auto substMap = unifier.unifyEquations(equations);
    if (!substMap) {
      func.emitError() << "type inference failed";
      return signalPassFailure();
    }

    // Apply substitutions.
    LLVM_DEBUG(llvm::dbgs() << "Unification subst:\n");
    LLVM_DEBUG(for (auto it
                    : *substMap) {
      llvm::dbgs() << "  " << it.first << " -> " << *it.second << "\n";
    });
    for (auto it : *substMap) {
      equations.applySubst(it.first, it.second);
    }

    // Now rewrite the function type based on actual types of entry block
    // args and the final return op operands.
    auto entryBlockTypes = func.getBody().front().getArgumentTypes();
    SmallVector<Type, 4> inputTypes(entryBlockTypes.begin(),
                                    entryBlockTypes.end());
    SmallVector<Type, 4> resultTypes;
    if (p.getLastReturnOp()) {
      auto resultRange = p.getLastReturnOp()->getOperandTypes();
      resultTypes.append(resultRange.begin(), resultRange.end());
    }
    auto funcType = FunctionType::get(inputTypes, resultTypes, &getContext());
    func.setType(funcType);
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Basicpy::createFunctionTypeInferencePass() {
  return std::make_unique<FunctionTypeInferencePass>();
}
