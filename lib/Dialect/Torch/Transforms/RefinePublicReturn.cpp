//===- RefinePublicReturn.cpp ------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class RefinePublicReturnPass
    : public RefinePublicReturnBase<RefinePublicReturnPass> {
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

  Value newValueOperand(OpBuilder &builder, Location loc, Value operand) {
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
      } else if (auto listCon =
                     newOperand.getDefiningOp<PrimListConstructOp>()) {
        // clang-format off
        // Convert:
        //   %4 = torch.copy.to_tensor %3 : !torch.tensor
        //   %5 = torch.prim.ListConstruct %4 : (!torch.tensor) -> !torch.list<tensor>
        //   return %5 : !torch.list<tensor>
        // to:
        //   %4 = torch.prim.ListConstruct %3 : (!torch.vtensor) -> !torch.list<vtensor>
        //   return %4 : !torch.list<vtensor>
        // clang-format on
        auto listType = listCon.getType().dyn_cast<ListType>();
        if (listCon.getElements().empty() ||
            !(listType.getContainedType().isa<NonValueTensorType>() ||
              listType.getContainedType().isa<OptionalType>())) {
          break;
        }
        auto newListElements = llvm::to_vector(
            llvm::map_range(listCon.getElements(), [&](Value tensor) -> Value {
              return newValueOperand(builder, loc, tensor);
            }));
        assert(!newListElements.empty());
        newOperand = builder.create<PrimListConstructOp>(
            loc, ListType::get(newListElements[0].getType()), newListElements);
        break;
      } else {
        break;
      }
    }
    if (auto tensorType = newOperand.getType().dyn_cast<BaseTensorType>()) {
      return copyTensorToType(builder, loc, tensorType.getWithValueSemantics(),
                              newOperand);
    }
    return newOperand;
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
    llvm::transform(returnOp.getOperands(), std::back_inserter(newOperands),
                    [this, &builder, &returnOp](Value operand) {
                      return newValueOperand(builder, returnOp.getLoc(),
                                             operand);
                    });
    returnOp->setOperands(newOperands);

    // Update the function type.
    auto funcType = func.getFunctionType();
    func.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   ValueRange(newOperands).getTypes()));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createRefinePublicReturnPass() {
  return std::make_unique<RefinePublicReturnPass>();
}
