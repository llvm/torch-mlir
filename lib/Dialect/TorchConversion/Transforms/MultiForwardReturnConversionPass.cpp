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

#include "llvm/ADT/PriorityQueue.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

//===----------------------------------------------------------------------===//
// MultiForwardReturnConversionPass
//===----------------------------------------------------------------------===//

namespace {

class CompareOperation {
public:
  bool operator()(Operation* opA,
                  Operation* opB) {
    return opB->isBeforeInBlock(opA);
  }
};

class CompareValue {
public:
  bool operator()(Value valA,
                  Value valB) {
    return valB.getDefiningOp()->isBeforeInBlock(valA.getDefiningOp());
  }
};

struct MultiForwardReturnConversionPass
    : public MultiForwardReturnConversionBase<MultiForwardReturnConversionPass> {
  using MultiForwardReturnConversionBase<
      MultiForwardReturnConversionPass>::MultiForwardReturnConversionBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TorchConversion::TorchConversionDialect>();
  }
  MultiForwardReturnConversionPass() = default;
  MultiForwardReturnConversionPass(unsigned valueIndex) {
    this->valueIndex = valueIndex;
  }
  void runOnOperation() override {
    auto module = getOperation();
    // We deal with ModuleOp containing ONLY one FuncOp.
    func::FuncOp funcOpToDelete;
    module.walk([&](func::FuncOp funcOp) {
      rewriteSignature(funcOp, module);
      funcOpToDelete = funcOp;
      return WalkResult::interrupt();
    });
    // Remove any references to the FuncOp from other operations
    funcOpToDelete.walk([](mlir::Operation *op) {
      op->dropAllDefinedValueUses();
      op->dropAllUses();
    });

    // Remove the FuncOp from the parent region
    mlir::Region *parentRegion = funcOpToDelete->getParentRegion();
    parentRegion->front().getOperations().erase(funcOpToDelete);

    return;
  }
  void rewriteSignature(func::FuncOp func , ModuleOp moduleOp) {
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

    Operation* defOp = valueToReturn.getDefiningOp();
    assert(defOp && "the value initially being returned seems to be a BlockArgument");
    
    llvm::PriorityQueue<Operation *, std::vector<Operation *>, CompareOperation> defOpQueue;
    DenseSet<Operation *> visitedOps;
    DenseSet<Value> visitedValues;
    llvm::PriorityQueue<Value, std::vector<Value>, CompareValue> valuesToReturn;

    defOpQueue.push(defOp);
    visitedOps.insert(defOp);
    visitedValues.insert(valueToReturn);
    valuesToReturn.push(valueToReturn);
    while (!defOpQueue.empty()) {
      Operation *defOp = defOpQueue.top();
      defOpQueue.pop();
      for (Value val : defOp->getOperands()) {
        Operation *defOpVal = val.getDefiningOp();
        if (!defOpVal)
          continue;
        if (visitedValues.count(val))
          continue;
        visitedValues.insert(val);
        valuesToReturn.push(val);
        if (visitedOps.count(defOpVal))
          continue;
        visitedOps.insert(defOpVal);
        defOpQueue.push(defOpVal);
      }
    }

    OpBuilder builder(moduleOp);
    ConversionPatternRewriter rewriter(moduleOp->getContext());
    unsigned funcNumber = 0;
    auto funcOp = static_cast<Operation*>(func);
    auto funcType = func.getFunctionType();
    while (!valuesToReturn.empty()) {
      // Create new function.
      Value val = valuesToReturn.top();
      valuesToReturn.pop();
      if (!val.getType().isa<RankedTensorType>())
        continue;
      std::string funcName = "forward" + std::to_string(funcNumber);
      builder.setInsertionPoint(funcOp);
      auto newFuncOp = builder.create<mlir::func::FuncOp>(moduleOp.getLoc(), funcName,
                                                           funcType);
      returnOp->setOperands({val});
      Operation* funcOpClone = funcOp->clone();
      rewriter.inlineRegionBefore(funcOpClone->getRegion(0), newFuncOp.getBody(), newFuncOp.end());
      rewriter.eraseOp(funcOpClone);
      newFuncOp.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   ValueRange({val}).getTypes()));
      funcNumber++;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::TorchConversion::createMultiForwardReturnConversionPass(unsigned valueIndex) {
  return std::make_unique<MultiForwardReturnConversionPass>(valueIndex);
}

