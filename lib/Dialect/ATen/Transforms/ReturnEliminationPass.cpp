//===- ReturnEliminationPass.cpp --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

#include <set>
#include <vector>

#define DEBUG_TYPE "return-elimination"

using namespace mlir;
using namespace mlir::NPCOMP::aten;

namespace {

/// In the process of lowering the ATenLoweringPass generates function calls
/// that take and return memrefs. However, this makes memory managment
/// somewhat awkward, since we need to take care of allocating and
/// deallocating memory.  It also forces a copy to pytorch, which wants to
/// pass in a pre-allocated buffer for the return type.  To simplify the
/// library, we convert the signature of function calls (particularly the
/// toplevel) to pass return values by reference.
class ReturnEliminationPass
    : public ATenReturnEliminationBase<ReturnEliminationPass> {
public:
  ReturnEliminationPass() {}

  void runOn(Operation *op) {
    auto module = getOperation();

    if (visitedOps.count(op))
      return;
    visitedOps.insert(op);

    if (auto callOp = dyn_cast<CallOp>(op)) {

      auto builder = std::make_unique<mlir::OpBuilder>(op);

      std::vector<Type> tys;
      for (auto t : callOp.getCalleeType().getInputs())
        tys.push_back(t);
      for (auto t : callOp.getCalleeType().getResults())
        tys.push_back(t);

      auto newFnTy = FunctionType::get(tys, {}, op->getContext());
      // FIXME: possible name collision
      std::string newFnName = callOp.callee().str() + "_out";

      if (!module.lookupSymbol<FuncOp>(newFnName)) {
        auto fn = FuncOp::create(op->getLoc(), newFnName, newFnTy);
        module.push_back(fn);
      }

      std::vector<Value> newCallArgs{callOp.arg_operand_begin(),
                                     callOp.arg_operand_end()};

      for (auto v : callOp.getResults()) {
        if (!v.getType().isa<MemRefType>())
          llvm_unreachable("function returns non-memref");
        if (!valueMap.count(v)) {
          valueMap[v] = builder->create<AllocOp>(
              op->getLoc(), v.getType().cast<MemRefType>());
        }
        v.replaceAllUsesWith(valueMap[v]);
        newCallArgs.push_back(valueMap[v]);
      }

      builder->create<CallOp>(op->getLoc(), newFnName, ArrayRef<Type>{},
                              newCallArgs);
      erasedOps.insert(op);
      auto fn = module.lookupSymbol<FuncOp>(callOp.callee());
      if (fn && fn.use_empty())
        erasedOps.insert(fn);
    } else if (isa<AllocOp>(op)) {
      Value v = op->getResult(0);
      if (valueMap.count(v)) {
        v.replaceAllUsesWith(valueMap[v]);
        erasedOps.insert(op);
      }
    }

    for (Value v : op->getOperands()) {
      if (!v.getType().isa<MemRefType>())
        continue;
      if (v.isa<BlockArgument>())
        continue;
      if (v.getDefiningOp())
        runOn(v.getDefiningOp());
    }
  }

  void runOnOperation() override {

    auto module = getOperation();

    // check that a function called "graph" exists
    auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    if (!graph) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a graph function\n");
      signalPassFailure();
      return;
    }

    // assume a single bb with a single return statement
    Block &BB = graph.front();

    FunctionType funcTy = graph.getType();
    std::vector<Type> newFuncInputTys;

    for (auto ty : funcTy.getInputs())
      newFuncInputTys.push_back(ty);

    for (auto ty : funcTy.getResults())
      newFuncInputTys.push_back(ty);

    FunctionType newFuncTy =
        FunctionType::get(newFuncInputTys, {}, module.getContext());
    graph.setType(newFuncTy);

    Operation *retOp = BB.getTerminator();
    auto builder = std::make_unique<mlir::OpBuilder>(retOp);

    builder->create<ReturnOp>(retOp->getLoc());

    std::vector<Value> operands{retOp->getOperands().begin(),
                                retOp->getOperands().end()};

    retOp->dropAllReferences();
    erasedOps.insert(retOp);

    for (Value v : operands)
      valueMap[v] = BB.addArgument(v.getType());

    for (Value v : operands) {
      if (!v.getType().isa<MemRefType>())
        llvm_unreachable("graph function returns non-memref");
      if (v.getDefiningOp())
        runOn(v.getDefiningOp());
    }

    for (auto oi = BB.rbegin(), oe = BB.rend(); oi != oe; ++oi) {
      Operation *o = &*oi;
      for (Value v : o->getResults()) {
        if (v.getType().isa<MemRefType>()) {
          runOn(o);
          break;
        }
      }
    }

    for (Operation *o : erasedOps)
      o->erase();
  }

private:
  llvm::DenseMap<Value, Value> valueMap;
  std::set<Operation *> visitedOps;
  std::set<Operation *> erasedOps;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::aten::createReturnEliminationPass() {
  return std::make_unique<ReturnEliminationPass>();
}
