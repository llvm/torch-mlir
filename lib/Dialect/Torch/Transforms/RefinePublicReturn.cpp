//===- RefinePublicReturn.cpp ------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

namespace {

class RefinePublicReturnPass
    : public RefinePublicReturnBase<RefinePublicReturnPass> {
  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](FuncOp func) {
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

  void rewriteSignature(FuncOp func) {
    // Find the unique return op.
    ReturnOp returnOp;
    WalkResult walkResult = func.walk([&](ReturnOp op) {
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

    // Get the new operands. Either the original operand, or if there is a
    // TensorStaticInfoCastOp then the pre-casted operand, which is presumed to
    // have a more precise type.
    SmallVector<Value> newOperands;
    OpBuilder builder(returnOp);
    for (auto operand : returnOp.getOperands()) {
      Value newOperand;
      if (auto cast = operand.getDefiningOp<TensorStaticInfoCastOp>()) {
        newOperand = cast.getOperand();
      } else {
        newOperand = operand;
      }
      if (auto tensorType = newOperand.getType().dyn_cast<BaseTensorType>()) {
        newOperands.push_back(
            copyTensorToType(builder, returnOp->getLoc(),
                             tensorType.getWithValueSemantics(), newOperand));
      } else {
        newOperands.push_back(newOperand);
      }
    }
    returnOp->setOperands(newOperands);

    // Update the function type.
    auto funcType = func.getType();
    func.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   ValueRange(newOperands).getTypes()));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::Torch::createRefinePublicReturnPass() {
  return std::make_unique<RefinePublicReturnPass>();
}
