//===- RefinePublicReturn.cpp ------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch::Torch {

#define GEN_PASS_DEF_REFINEPUBLICRETURN
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h.inc"

namespace {

class RefinePublicReturnPass
    : public impl::RefinePublicReturnBase<RefinePublicReturnPass> {
  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](func::FuncOp func) {
      if (func.getVisibility() != SymbolTable::Visibility::Public)
        return;
      if (func.isExternal())
        return;
      auto uses = SymbolTable::getSymbolUses(func, module);
      if (!uses || uses->begin() != uses->end()) {
        func.emitError() << "unimplemented: cannot refine public return for "
                         << "for public function with uses";
        return signalPassFailure();
      }
      rewriteSignature(func);
    });
  }

  void rewriteSignature(func::FuncOp func) {
    // Find the unique return op.
    func::ReturnOp returnOp;
    WalkResult walkResult = func.walk([&](func::ReturnOp op) {
      if (returnOp)
        return WalkResult::interrupt();
      returnOp = op;
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      func.emitError() << "unimplemented: refining returns for function with "
                          "more than one return op";
      return signalPassFailure();
    }

    // Get the new operands. Either the original operand, or for tensors,
    // looking through TensorStaticInfoCastOp/CopyToNonValueTensorOp which are
    // presumed to have a more precise type.
    SmallVector<Value> newOperands;
    OpBuilder builder(returnOp);
    for (auto operand : returnOp.getOperands()) {
      Value newOperand = operand;
      // Look through TensorStaticInfoCastOp's, CopyToNonValueTensorOp's, and
      // DerefineOp's.
      for (;;) {
        if (auto cast = newOperand.getDefiningOp<TensorStaticInfoCastOp>()) {
          newOperand = cast.getOperand();
        } else if (auto copy =
                       newOperand.getDefiningOp<CopyToNonValueTensorOp>()) {
          // If the return (or transitively other ops) are not the only users,
          // then we can't be sure that the tensor hasn't been mutated, so stop
          // here.
          SetVector<Operation *> users(copy->getUsers().begin(),
                                       copy->getUsers().end());
          if (users.size() != 1)
            break;
          newOperand = copy.getOperand();
        } else if (auto derefine = newOperand.getDefiningOp<DerefineOp>()) {
          newOperand = derefine.getOperand();
        } else {
          break;
        }
      }

      if (auto tensorType = dyn_cast<BaseTensorType>(newOperand.getType())) {
        newOperands.push_back(
            copyTensorToType(builder, returnOp->getLoc(),
                             tensorType.getWithValueSemantics(), newOperand));
      } else {
        newOperands.push_back(newOperand);
      }
    }
    returnOp->setOperands(newOperands);

    // Update the function type.
    auto funcType = func.getFunctionType();
    func.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   ValueRange(newOperands).getTypes()));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRefinePublicReturnPass() {
  return std::make_unique<RefinePublicReturnPass>();
}

} // namespace mlir::torch::Torch
