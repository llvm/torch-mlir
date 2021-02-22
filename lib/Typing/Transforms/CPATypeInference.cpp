//===- CPATypeInference.cpp - Type inference passes -----------------*-
// C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Typing/Analysis/CPA/Algorithm.h"
#include "npcomp/Typing/Analysis/CPA/Interfaces.h"
#include "npcomp/Typing/Analysis/CPA/Types.h"
#include "npcomp/Typing/Support/CPAIrHelpers.h"
#include "npcomp/Typing/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cpa-type-inference"

using namespace llvm;
using namespace mlir;
using namespace mlir::NPCOMP::Basicpy;
using namespace mlir::NPCOMP::Typing;

static void printReport(CPA::Environment &env, MLIRContext &mlirContext,
                        llvm::raw_ostream &os) {
  auto &cpaContext = env.getContext();
  os << "CONSTRAINTS:\n";
  os << "------------\n";
  env.getConstraints().print(cpaContext, os);

  os << "\nTYPEVARS:\n";
  os << "---------\n";
  env.getTypeVars().print(cpaContext, os);

  os << "\nVALUE->TYPE NODE MAPPING:";
  os << "\n-------------------------\n";
  for (auto &it : env.getValueTypeMap()) {
    auto irValue = it.first;
    auto typeNode = it.second;
    CPA::GreedyTypeNodeVarResolver resolver(cpaContext, mlirContext,
                                            irValue.getLoc());
    if (failed(resolver.analyzeTypeNode(typeNode))) {
      os << "! ";
      typeNode->print(cpaContext, os, false);
      os << " -> " << irValue;
      os << "\n";
      continue;
    }

    if (resolver.getMappings().empty()) {
      // Not generic.
      os << "= ";
      typeNode->print(cpaContext, os, false);
      os << " -> " << irValue;
      os << "\n";
      continue;
    }

    // Generic.
    os << "*";
    auto newIrType = typeNode->constructIrType(
        cpaContext, resolver.getMappings(), &mlirContext, irValue.getLoc());
    if (!newIrType) {
      os << "!";
    } else {
      os << " " << newIrType << ":";
    }

    os << " ";
    typeNode->print(cpaContext, os, false);
    os << " -> " << irValue;
    os << "\n";
  }
}

namespace {

class InitialConstraintGenerator {
public:
  InitialConstraintGenerator(CPA::Environment &env) : env(env) {}

  /// If a return op was visited, this will be one of them.
  Operation *getLastReturnOp() { return funcReturnOp; }

  /// Gets any ReturnLike ops that do not return from the outer function.
  /// This is used to fixup parent SCF ops and the like.
  llvm::SmallVectorImpl<Operation *> &getInnerReturnLikeOps() {
    return innerReturnLikeOps;
  }

  CPA::TypeNode *resolveValueType(Value value) {
    return env.mapValueToType(value);
  }

  void addSubtypeConstraint(Value superValue, Value subValue,
                            Operation *contextOp) {
    auto superVt = resolveValueType(superValue);
    auto subVt = resolveValueType(subValue);
    env.getContext().getConstraint(superVt, subVt);
  }

  LogicalResult runOnFunction(FuncOp funcOp) {
    // Iterate and create type nodes for entry block arguments, as these
    // must be resolved no matter what.
    if (funcOp.getBody().empty())
      return success();

    auto &entryBlock = funcOp.getBody().front();
    for (auto blockArg : entryBlock.getArguments()) {
      resolveValueType(blockArg);
    }

    // Then walk ops, creating equations.
    LLVM_DEBUG(llvm::dbgs() << "POPULATE CHILD OPS:\n");
    auto result = funcOp.walk([&](Operation *childOp) -> WalkResult {
      if (childOp == funcOp)
        return WalkResult::advance();
      LLVM_DEBUG(llvm::dbgs() << "  + POPULATE: " << *childOp << "\n");
      // Use the op interface.
      if (auto opInt =
              dyn_cast<NPCOMPTypingCPATypeInferenceOpInterface>(childOp)) {
        opInt.addCPAConstraints(env.getContext());
        return WalkResult::advance();
      }
      // Special op handling.
      // Many of these (that are not standard ops) should become op
      // interfaces.
      // --------------------
      if (auto op = dyn_cast<SelectOp>(childOp)) {
        // Note that the condition is always i1 and not subject to type
        // inference.
        // addSubtypeConstraint(op.true_value(), op.false_value(), op);
        // addSubtypeConstraint(op.false_value(), op.true_value(), op);
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<AsI1Op>(childOp)) {
        // Note that the result is always i1 and not subject to type
        // inference.
        resolveValueType(op.operand());
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<scf::IfOp>(childOp)) {
        // Note that the condition is always i1 and not subject to type
        // inference.
        for (auto result : op.getResults()) {
          resolveValueType(result);
        }
        return WalkResult::advance();
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(childOp)) {
        auto scfParentOp = yieldOp->getParentOp();
        if (scfParentOp->getNumResults() != yieldOp.getNumOperands()) {
          yieldOp.emitWarning()
              << "cannot run type inference on yield due to arity mismatch";
          return WalkResult::advance();
        }
        for (auto it :
             llvm::zip(scfParentOp->getResults(), yieldOp.getOperands())) {
          addSubtypeConstraint(std::get<1>(it), std::get<0>(it), yieldOp);
        }
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<UnknownCastOp>(childOp)) {
        addSubtypeConstraint(op.operand(), op.result(), op);
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<BinaryExprOp>(childOp)) {
        // TODO: This should really be applying arithmetic promotion, not
        // strict equality.
        addSubtypeConstraint(op.left(), op.result(), op);
        addSubtypeConstraint(op.right(), op.result(), op);
        return WalkResult::advance();
      }
      if (auto op = dyn_cast<BinaryCompareOp>(childOp)) {
        // TODO: This should really be applying arithmetic promotion, not
        // strict equality.
        addSubtypeConstraint(op.left(), op.right(), op);
        addSubtypeConstraint(op.right(), op.left(), op);
        return WalkResult::advance();
      }

      // Fallback trait based equations.
      // ----------------------
      // Ensure that constant nodes get assigned a constant type.
      if (childOp->hasTrait<OpTrait::ConstantLike>()) {
        resolveValueType(childOp->getResult(0));
        return WalkResult::advance();
      }
      // Function returns must all have the same types.
      if (childOp->hasTrait<OpTrait::ReturnLike>()) {
        if (childOp->getParentOp() == funcOp) {
          if (funcReturnOp) {
            if (funcReturnOp->getNumOperands() != childOp->getNumOperands()) {
              childOp->emitOpError() << "different arity of function returns";
              return WalkResult::interrupt();
            }
            for (auto it : llvm::zip(funcReturnOp->getOperands(),
                                     childOp->getOperands())) {
              // addSubtypeConstraint(std::get<0>(it), std::get<1>(it),
              // childOp);
              addSubtypeConstraint(std::get<1>(it), std::get<0>(it), childOp);
            }
          }
          funcReturnOp = childOp;
          return WalkResult::advance();
        } else {
          innerReturnLikeOps.push_back(childOp);
        }
      }

      childOp->emitRemark() << "unhandled op in type inference";

      return WalkResult::advance();
    });

    return success(result.wasInterrupted());
  }

private:
  // The last encountered ReturnLike op.
  Operation *funcReturnOp = nullptr;
  llvm::SmallVector<Operation *, 4> innerReturnLikeOps;
  CPA::Environment &env;
};

class CPAFunctionTypeInferencePass
    : public CPAFunctionTypeInferenceBase<CPAFunctionTypeInferencePass> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    if (func.getBody().empty())
      return;

    CPA::Context cpaContext(CPA::createDefaultTypeMapHook());
    auto &env = cpaContext.getCurrentEnvironment();

    InitialConstraintGenerator p(env);
    (void)p.runOnFunction(func);

    CPA::PropagationWorklist prop(env);
    do {
      prop.propagateTransitivity();
    } while (prop.commit());

    LLVM_DEBUG(printReport(env, getContext(), llvm::dbgs()));

    // Apply updates.
    // TODO: This is far too naive and is basically only valid for single-block
    // functions that are not called. Generalize it.
    for (auto &it : env.getValueTypeMap()) {
      auto irValue = it.first;
      auto typeNode = it.second;
      auto loc = irValue.getLoc();
      CPA::GreedyTypeNodeVarResolver resolver(cpaContext, getContext(),
                                              irValue.getLoc());
      if (failed(resolver.analyzeTypeNode(typeNode))) {
        mlir::emitRemark(loc)
            << "type inference did not converge to an "
            << "unambiguous type (this is a terribly unacceptable level of "
            << "detail in an error message)";
        return signalPassFailure();
      }

      if (resolver.getMappings().empty()) {
        // The type is not generic/unknown, so it does not need to be updated.
        continue;
      }

      auto newType = typeNode->constructIrType(
          cpaContext, resolver.getMappings(), &getContext(), loc);
      if (!newType) {
        auto diag = mlir::emitRemark(loc);
        diag << "type inference converged but a concrete IR "
             << "type could not be constructed";
        return signalPassFailure();
      }
      irValue.setType(newType);
    }

    // Now rewrite the function type based on actual types of entry block
    // args and the final return op operands.
    // Again, this is just a toy that will work for very simple, global
    // functions.
    auto entryBlockTypes = func.getBody().front().getArgumentTypes();
    SmallVector<Type, 4> inputTypes(entryBlockTypes.begin(),
                                    entryBlockTypes.end());
    SmallVector<Type, 4> resultTypes;
    if (p.getLastReturnOp()) {
      auto resultRange = p.getLastReturnOp()->getOperandTypes();
      resultTypes.append(resultRange.begin(), resultRange.end());
    }
    auto funcType = FunctionType::get(&getContext(), inputTypes, resultTypes);
    func.setType(funcType);
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Typing::createCPAFunctionTypeInferencePass() {
  return std::make_unique<CPAFunctionTypeInferencePass>();
}
