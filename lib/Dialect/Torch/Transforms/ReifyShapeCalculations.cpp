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
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static FailureOr<SmallVector<Value>>
shapeFunctionArgsBuilder(OpBuilder &b, Location loc,
                         ValueRange originalOperands, func::FuncOp shapeFunc) {
  // Massage the op operands to match the shape function signature.
  // The shape function generally takes the same operands as the op, with a few
  // systematic modifications, such as replacing tensors with their shapes.
  SmallVector<Value> shapeFuncArgs;
  for (auto operandAndDesiredType :
       llvm::zip(originalOperands, shapeFunc.getArgumentTypes())) {
    Value operand;
    Type desiredType;
    std::tie(operand, desiredType) = operandAndDesiredType;
    FailureOr<Value> shapeFuncArg = adjustFunctionArg(
        b, loc, operand, desiredType,
        [](OpBuilder &b, Location loc, Value operand,
           Type desiredType) -> Value {
          // The shape library functions have tensor operands replaced with
          // `!torch.list<int>` types for the shape. Get the sizes.
          auto desiredListType = dyn_cast<Torch::ListType>(desiredType);
          if (!desiredListType)
            return operand;
          if (isa<Torch::BaseTensorType>(operand.getType()) &&
              isa<Torch::IntType>(desiredListType.getContainedType())) {
            return b.create<AtenSizeOp>(loc, desiredType, operand);
          }
          return operand;
        });
    if (failed(shapeFuncArg))
      return failure();
    shapeFuncArgs.push_back(*shapeFuncArg);
  }

  return shapeFuncArgs;
}

namespace {
struct ReifyShapeCalculationsPass
    : public ReifyShapeCalculationsBase<ReifyShapeCalculationsPass> {
  ReifyShapeCalculationsPass() = default;
  ReifyShapeCalculationsPass(StringRef extraLibrary) {
    this->extraLibrary = extraLibrary.str();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // TODO: Find a way to not have to parse this every time.
    // The library is O(#ops we know about), and this pass should be
    // O(#ops in the program) ideally.
    OwningOpRef<ModuleOp> library =
        parseSourceString<ModuleOp>(getAbstractInterpLibrary(), context);
    if (!extraLibrary.empty())
      if (failed(mlir::torch::Torch::loadExtraLibrary(extraLibrary, library))) {
        emitError(module->getLoc(),
                  "Failed to load extra-library file at " + extraLibrary);
        return signalPassFailure();
      }

    // Walk all the operations, and if we have a shape function, wrap the op
    // in a `torch.shape.calculate` op.
    SmallVector<std::string> functionsNeeded;
    WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
      return wrapWithCalculateOpIfLibraryFunctionAvailable(
          op, *library, LibraryFunctionKind::ShapeFunction, functionsNeeded,
          shapeFunctionArgsBuilder);
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
    importLibraryFunctions(module, *library, std::move(functionsNeeded));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createReifyShapeCalculationsPass(StringRef extraLibrary) {
  return std::make_unique<ReifyShapeCalculationsPass>(extraLibrary);
}
