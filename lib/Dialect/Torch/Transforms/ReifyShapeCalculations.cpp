//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static Value adjustShapeFunctionArg(Value operand, Type desiredType,
                                    OpBuilder &b, Location loc);

static Value adjustListArg(Value operand, Torch::ListType desiredType,
                           OpBuilder &b, Location loc) {
  auto providedType = operand.getType().cast<Torch::ListType>();

  // Pseudocode:
  //
  // operand = ...
  // adjusted_list = []
  // for i in range(len(operand)):
  //     adjusted_list.append(adjust(operand[i]))
  // return adjusted_list
  Value adjustedList =
      b.create<PrimListConstructOp>(loc, desiredType, ValueRange({}));
  // Create a for-like PrimLoopOp.
  Value maxTripCount = b.create<AtenLenTOp>(loc, operand);
  Value cTrue = b.create<Torch::ConstantBoolOp>(loc, true);
  auto loop = b.create<PrimLoopOp>(loc, TypeRange({}), maxTripCount,
                                   /*initialCondition=*/cTrue,
                                   /*iterArgsInit=*/ValueRange({}));
  OpBuilder::InsertionGuard guard(b);
  Block *body = b.createBlock(&loop.region(), loop.region().begin(),
                              TypeRange({b.getType<Torch::IntType>()}), {loc});
  // Create the loop body.
  {
    Value iterationNumber = body->getArgument(0);
    Value element = b.create<Aten__Getitem__TOp>(
        loc, providedType.getContainedType(), operand, iterationNumber);
    Value adjustedElement =
        adjustShapeFunctionArg(element, desiredType.getContainedType(), b, loc);
    b.create<AtenAppendTOp>(loc, adjustedList.getType(), adjustedList,
                            adjustedElement);
    b.create<PrimLoopConditionOp>(loc, /*shouldContinue=*/cTrue,
                                  /*iterArgs=*/ValueRange({}));
  }

  return adjustedList;
}

static Value adjustShapeFunctionArg(Value operand, Type desiredType,
                                    OpBuilder &b, Location loc) {
  auto operandType = operand.getType();

  // No need for adjustment if they already match.
  if (operandType == desiredType)
    return operand;

  // If the operand is NoneType, then we just need to derefine it to the
  // optional type in the shape function signature.
  if (operandType.isa<Torch::NoneType>()) {
    assert(desiredType.isa<Torch::OptionalType>() &&
           "Don't expect shape functions to have NoneType parameters");
    return b.create<DerefineOp>(loc, desiredType, operand);
  }

  // If the operand type is statically !torch.optional, then we need to do
  // different things for the None and non-None cases.
  // For the None case, we just need to derefine it to the desired type.
  // For the non-None case, we need to unwrap the optional type and then adjust
  // it recursively (which also takes care of derefining it to ultimate desired
  // type).
  // A case where this happens is `!torch.optional<!torch.vtensor>` ->
  // `!torch.optional<!torch.list<!torch.int>>>`.
  if (auto operandOptionalType = operandType.dyn_cast<Torch::OptionalType>()) {
    if (desiredType.isa<Torch::OptionalType>()) {
      // if optional is None:
      //     return derefine(None)
      // else:
      //     return adjust(unchecked_cast(optional))
      auto none = b.create<ConstantNoneOp>(loc);
      auto isNone = b.create<Aten__Is__Op>(loc, operand, none);
      auto primIf = b.create<PrimIfOp>(loc, desiredType, isNone);
      {
        Region &thenRegion = primIf.thenRegion();
        b.createBlock(&thenRegion, thenRegion.end());
        auto derefineNone = b.create<DerefineOp>(loc, desiredType, none);
        b.create<PrimIfYieldOp>(loc, ValueRange{derefineNone});
      }
      {
        Region &elseRegion = primIf.elseRegion();
        b.createBlock(&elseRegion, elseRegion.end());
        auto downcasted = b.create<PrimUncheckedCastOp>(
            loc, operandOptionalType.getContainedType(), operand);
        auto adjusted = adjustShapeFunctionArg(downcasted, desiredType, b, loc);
        b.create<PrimIfYieldOp>(loc, adjusted);
      }
      b.setInsertionPointAfter(primIf);
      return primIf.getResult(0);
    }
  }

  // If the desired type is OptionalType, then recursively adjust the operand to
  // the contained type, then derefine it to `!torch.optional`. For example,
  // `!torch.vtensor -> !torch.optional<!torch.list<!torch.int>>>`.
  if (auto desiredOptionalType = desiredType.dyn_cast<Torch::OptionalType>()) {
    auto adjusted = adjustShapeFunctionArg(
        operand, desiredOptionalType.getContainedType(), b, loc);
    return b.create<DerefineOp>(loc, desiredType, adjusted);
  }

  // The shape library functions have tensor operands replaced with
  // `!torch.list<!torch.int>` types for the shape. Get the sizes.
  if (operand.getType().isa<Torch::BaseTensorType>()) {
    assert(desiredType.isa<Torch::ListType>() &&
           "Don't expect shape functions to have tensor parameters");
    return b.create<AtenSizeOp>(loc, desiredType, operand);
  }

  // Run this after `operand.getType().isa<Torch::BaseTensorType>()` so that
  // `!torch.vtensor` -> `!torch.list<!torch.int>` is handled there specially
  // first.
  if (auto desiredListType = desiredType.dyn_cast<Torch::ListType>()) {
    return adjustListArg(operand, desiredListType, b, loc);
  }

  // The shape library functions use `float` where the operator
  // signature uses `Scalar` (see comments in torch_ods_gen.py for
  // explanation).
  if (desiredType.isa<Torch::FloatType>() &&
      operand.getType().isa<Torch::IntType>()) {
    return b.create<AtenFloatScalarOp>(loc, desiredType, operand);
  }

  // Pass the operand as-is.
  return operand;
}

// Populates the shape calculation region with a call to the shape function
// from the shape library.
static LogicalResult
populateShapeCalculationRegion(ShapeCalculateOp op, ValueRange originalOperands,
                               mlir::FuncOp shapeFunction) {
  // Create a call to the shape function in the `shapeCalculation` region.
  // We will import the callee from the shape library later.
  OpBuilder b(op.getContext());
  Location loc = op->getLoc();
  b.createBlock(&op.shapeCalculation());
  // Massage the op operands to match the shape function signature.
  // The shape function generally takes the same operands as the op, with a few
  // systematic modifications, such as replacing tensors with their shapes.
  SmallVector<Value> shapeFunctionArgs;
  for (auto operandAndDesiredType :
       llvm::zip(originalOperands, shapeFunction.getArgumentTypes())) {
    Value operand;
    Type desiredType;
    std::tie(operand, desiredType) = operandAndDesiredType;
    Value shapeFunctionArg =
        adjustShapeFunctionArg(operand, desiredType, b, loc);
    if (!shapeFunctionArg)
      return failure();
    shapeFunctionArgs.push_back(shapeFunctionArg);
  }

  // Create the call to the shape function!
  auto call = b.create<mlir::CallOp>(loc, shapeFunction, shapeFunctionArgs);

  // Python models multiple results with a tuple, so we need to unpack it
  // if the op has multiple results.
  SmallVector<Value> unpackedResults;
  assert(call.getNumResults() == 1 &&
         "Multiple results are packed in a tuple in Python!");
  Value result = call.getResult(0);
  if (auto tupleType = result.getType().dyn_cast<Torch::TupleType>()) {
    auto unpack =
        b.create<PrimTupleUnpackOp>(loc, tupleType.getContainedTypes(), result);
    llvm::append_range(unpackedResults, unpack.getResults());
  } else {
    unpackedResults.push_back(result);
  }

  // Terminate the region.
  b.create<ShapeCalculateYieldShapesOp>(loc, unpackedResults);
  return success();
}

namespace {
class ReifyShapeCalculationsPass
    : public ReifyShapeCalculationsBase<ReifyShapeCalculationsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // TODO: Find a way to not have to parse this every time.
    // The shape library is O(#ops we know about), and this pass should be
    // O(#ops in the program) ideally.
    auto shapeLibrary = parseSourceString(getShapeLibrary(), context);

    // Walk all the operations, and if we have a shape function, wrap the op
    // in a `torch.shape.calculate` op.
    SmallVector<std::string> neededShapeFunctions;
    bool hadError = false;
    module.walk([&](Operation *op) {
      Location loc = op->getLoc();
      auto name = op->getName().stripDialect();
      auto shapeFunctionName = ("__torch_mlir_shape_fn." + Twine(name)).str();
      auto shapeFunction =
          shapeLibrary->lookupSymbol<FuncOp>(shapeFunctionName);
      if (!shapeFunction)
        return;
      neededShapeFunctions.push_back(shapeFunctionName);
      auto shapeCalculate =
          OpBuilder(op).create<ShapeCalculateOp>(loc, op->getResultTypes());
      op->replaceAllUsesWith(shapeCalculate);
      {
        // Move the op into the body of the `torch.shape.calculate` op and yield
        // its results.
        OpBuilder b(context);
        Block *block = b.createBlock(&shapeCalculate.body());
        op->moveBefore(block, block->end());
        b.setInsertionPointAfter(op);
        b.create<ShapeCalculateYieldOp>(loc, op->getResults());
      }
      if (failed(populateShapeCalculationRegion(
              shapeCalculate, op->getOperands(), shapeFunction))) {
        hadError = true;
        return;
      }
    });

    if (hadError)
      return signalPassFailure();

    // Import just the functions we need. This includes transitive callees,
    // so we use a worklist algorithm.
    llvm::StringSet<> importedFunctions;
    SmallVector<std::string> worklist;
    llvm::append_range(worklist, neededShapeFunctions);
    while (!worklist.empty()) {
      auto symName = worklist.pop_back_val();
      if (importedFunctions.count(symName))
        continue;
      auto func = shapeLibrary->lookupSymbol<mlir::FuncOp>(symName);
      assert(func && "broken shape library");
      // Move the shape function from the library to the module this pass
      // is running on. (this mutates the library, but we re-parse it each time
      // so this is safe to do).
      func->moveBefore(&module.getBody()->front());
      // Set the visibility to private so that the shape functions go away
      // nicely after we are done with them.
      func.setVisibility(SymbolTable::Visibility::Private);
      // Continue the DFS.
      importedFunctions.insert(symName);
      func.walk([&](CallOp op) { worklist.push_back(op.getCallee().str()); });
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createReifyShapeCalculationsPass() {
  return std::make_unique<ReifyShapeCalculationsPass>();
}
