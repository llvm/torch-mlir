//===- PublicFunctionToTensor.cpp - Type inference passes --------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Dialect/Numpy/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP::Numpy;

namespace {

class PublicFunctionsToTensorPass
    : public NumpyPublicFunctionsToTensorBase<PublicFunctionsToTensorPass> {
  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](FuncOp func) {
      if (func.getVisibility() != SymbolTable::Visibility::Public)
        return;
      if (func.isExternal())
        return;
      auto uses = SymbolTable::getSymbolUses(func, module);
      if (!uses || uses->begin() != uses->end()) {
        func.emitWarning() << "unimplemented: cannot convert ndarray->tensor "
                           << "signature for public function with uses";
        return;
      }
      rewriteSignature(func);
    });
  }

  void rewriteSignature(FuncOp func) {
    auto &entryBlock = func.getBlocks().front();
    auto funcType = func.getType();
    auto loc = func.getLoc();

    // Rewrite inputs.
    auto builder = OpBuilder::atBlockBegin(&entryBlock);
    auto inputTypes = llvm::to_vector<4>(funcType.getInputs());
    for (unsigned i = 0; i < inputTypes.size(); ++i) {
      auto arrayType = inputTypes[i].dyn_cast<NdArrayType>();
      if (!arrayType)
        continue;
      Type tensorType = arrayType.toTensorType();
      BlockArgument argument = entryBlock.getArgument(i);
      argument.setType(tensorType);
      auto createOp =
          builder.create<CreateArrayFromTensorOp>(loc, arrayType, argument);
      argument.replaceAllUsesExcept(createOp,
                                    SmallPtrSet<Operation *, 1>{createOp});
      inputTypes[i] = tensorType;
    }

    // Rewrite result signature.
    auto resultTypes = llvm::to_vector<4>(funcType.getResults());
    for (auto &resultType : resultTypes) {
      auto arrayType = resultType.dyn_cast<NdArrayType>();
      if (arrayType)
        resultType = arrayType.toTensorType();
    }

    // Update signature.
    funcType =
        FunctionType::get(inputTypes, resultTypes, funcType.getContext());
    func.setType(funcType);

    // Rewrite all return terminators.
    func.walk([&](ReturnOp term) {
      OpBuilder builder(term);
      for (unsigned i = 0; i < term.getNumOperands(); ++i) {
        Value operand = term.getOperand(i);
        auto arrayType = operand.getType().dyn_cast<NdArrayType>();
        if (!arrayType)
          continue;
        Type tensorType = arrayType.toTensorType();
        auto copyOp = builder.create<CopyToTensorOp>(loc, tensorType, operand);
        term.setOperand(i, copyOp);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::Numpy::createPublicFunctionsToTensorPass() {
  return std::make_unique<PublicFunctionsToTensorPass>();
}
