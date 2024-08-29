//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "ReifyAbstractInterpCalculationsUtils.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Massage the op operands to match the dtype function signature.
// The dtype function generally takes the same operands as the op, with a few
// systematic modifications, such as replacing each tensor with a tuple of
// its rank and dtype.
static FailureOr<SmallVector<Value>>
dtypeFunctionArgsBuilder(OpBuilder &b, Location loc,
                         ValueRange originalOperands, func::FuncOp dtypeFunc) {
  // Turn every tensor into a tuple of (tensor_rank, tensor_dtype)
  auto dtypeArgAdjuster = [](OpBuilder &b, Location loc, Value operand,
                             Type desiredType) -> Value {
    if (isa<Torch::TupleType>(desiredType) &&
        isa<Torch::BaseTensorType>(operand.getType())) {
      Type intType = Torch::IntType::get(b.getContext());
      Type sizeListType = Torch::ListType::get(intType);
      Value size = b.create<AtenSizeOp>(loc, sizeListType, operand);
      Value rank = b.create<AtenLenTOp>(loc, intType, size);
      Value dtype = b.create<PrimDtypeOp>(loc, intType, operand);
      return b.create<PrimTupleConstructOp>(loc, desiredType,
                                            ArrayRef{rank, dtype});
    }
    return operand;
  };

  SmallVector<Value> dtypeFuncArgs;
  ArrayRef<Type> desiredTypes = dtypeFunc.getArgumentTypes();
  for (auto operand : originalOperands) {
    assert(!desiredTypes.empty() &&
           "`dtypeFunc` should have at least one argument for each argument in "
           "`originalOperands`");
    Type desiredType = desiredTypes.front();
    FailureOr<Value> otherArg;
    if (failed(otherArg = adjustFunctionArg(b, loc, operand, desiredType,
                                            dtypeArgAdjuster)))
      return failure();
    dtypeFuncArgs.push_back(*otherArg);
    desiredTypes = desiredTypes.drop_front();
  }

  return dtypeFuncArgs;
}

namespace {
struct ReifyDtypeCalculationsPass
    : public ReifyDtypeCalculationsBase<ReifyDtypeCalculationsPass> {
  ReifyDtypeCalculationsPass() = default;
  ReifyDtypeCalculationsPass(StringRef extraLibrary) {
    this->extraLibrary = extraLibrary.str();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    OwningOpRef<ModuleOp> library =
        parseSourceString<ModuleOp>(getAbstractInterpLibrary(), context);
    if (!extraLibrary.empty())
      if (failed(mlir::torch::Torch::loadExtraLibrary(extraLibrary, library))) {
        emitError(module->getLoc(),
                  "Failed to load extra-library file at " + extraLibrary);
        return signalPassFailure();
      }

    // Walk all the operations, and if we have a dtype function, wrap the op
    // in a `torch.dtype.calculate` op.
    SmallVector<std::string> functionsNeeded;
    WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
      return wrapWithCalculateOpIfLibraryFunctionAvailable(
          op, *library, LibraryFunctionKind::DtypeFunction, functionsNeeded,
          dtypeFunctionArgsBuilder);
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
    importLibraryFunctions(module, *library, std::move(functionsNeeded));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
Torch::createReifyDtypeCalculationsPass(StringRef extraLibrary) {
  return std::make_unique<ReifyDtypeCalculationsPass>(extraLibrary);
}
