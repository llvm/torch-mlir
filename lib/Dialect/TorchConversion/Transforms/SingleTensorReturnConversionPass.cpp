//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

//===----------------------------------------------------------------------===//
// SingleTensorReturnConversionPass
//===----------------------------------------------------------------------===//

namespace {
struct SingleTensorReturnConversionPass
    : public SingleTensorReturnConversionBase<SingleTensorReturnConversionPass> {
  using SingleTensorReturnConversionBase<
      SingleTensorReturnConversionPass>::SingleTensorReturnConversionBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TorchConversion::TorchConversionDialect>();
  }
  SingleTensorReturnConversionPass() = default;
  SingleTensorReturnConversionPass(unsigned valueIndex) {
    this->valueIndex = valueIndex;
  }
  void runOnOperation() override {
    auto module = getOperation();
    // We deal with ModuleOp containing ONLY one FuncOp.
    module.walk([&](func::FuncOp funcOp) {
      rewriteSignature(funcOp);
      return WalkResult::interrupt();
    });
    return;
  }
  void rewriteSignature(func::FuncOp func) {
    // Get the ReturnOp within function.
    func::ReturnOp returnOp;
    WalkResult walkResult = func.walk([&](func::ReturnOp op) {
      returnOp = op;
      return WalkResult::interrupt();
    });

    // Add an assertion to check whether the value index picked for
    // returning is within the number of operands that the return op has.
    assert(valueIndex<returnOp.getNumOperands() && "return value index incorrect");

    // Update the ReturnOp to only have the value at `valueIndex` remain.
    Value valueToReturn = returnOp.getOperand(valueIndex);
    returnOp->setOperands({valueToReturn});

    // Since we've changed the ReturnOp's return value set, we need to update
    // the containing function's return type as well.
    auto funcType = func.getFunctionType();
    func.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   ValueRange({valueToReturn}).getTypes()));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::TorchConversion::createSingleTensorReturnConversionPass(unsigned valueIndex) {
  return std::make_unique<SingleTensorReturnConversionPass>(valueIndex);
}
