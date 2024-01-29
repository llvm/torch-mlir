//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "ReifyAbstractInterpCalculationsUtils.h"
#include "mlir/Parser/Parser.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

std::string
mlir::torch::Torch::getLibraryFunctionPrefix(LibraryFunctionKind libFuncKind) {
  if (libFuncKind == LibraryFunctionKind::ShapeFunction)
    return "__torch_mlir_shape_fn.";
  else if (libFuncKind == LibraryFunctionKind::DtypeFunction)
    return "__torch_mlir_dtype_fn.";
  else if (libFuncKind == LibraryFunctionKind::HasValueSemantics)
    return "__torch_mlir_has_value_semantics_fn.";
  llvm_unreachable(
      "`getLibraryFunctionPrefix` called with an unsupported `CalculateOp`");
}

static Operation *createCalculateOp(OpBuilder &b, Location loc,
                                    TypeRange resultTypes,
                                    LibraryFunctionKind libFuncKind) {
  if (libFuncKind == LibraryFunctionKind::ShapeFunction)
    return b.create<ShapeCalculateOp>(loc, resultTypes);
  else if (libFuncKind == LibraryFunctionKind::DtypeFunction)
    return b.create<DtypeCalculateOp>(loc, resultTypes);
  llvm_unreachable(
      "`createCalculateOp` called with an unsupported `LibraryFunctionKind`");
}

static Operation *createCalculateYieldOp(OpBuilder &b, Location loc,
                                         ValueRange results,
                                         LibraryFunctionKind libFuncKind) {
  if (libFuncKind == LibraryFunctionKind::ShapeFunction)
    return b.create<ShapeCalculateYieldOp>(loc, results);
  else if (libFuncKind == LibraryFunctionKind::DtypeFunction)
    return b.create<DtypeCalculateYieldOp>(loc, results);
  llvm_unreachable("`createCalculateYieldOp` called with an unsupported "
                   "`LibraryFunctionKind`");
}

static Operation *
createCalculateYieldCalculationOp(OpBuilder &b, Location loc,
                                  ValueRange results,
                                  LibraryFunctionKind libFuncKind) {
  if (libFuncKind == LibraryFunctionKind::ShapeFunction)
    return b.create<ShapeCalculateYieldShapesOp>(loc, results);
  else if (libFuncKind == LibraryFunctionKind::DtypeFunction)
    return b.create<DtypeCalculateYieldDtypesOp>(loc, results);
  llvm_unreachable("`createCalculateYieldCalculationOp` called with an "
                   "unsupported `LibraryFunctionKind`");
}

LogicalResult Torch::wrapWithCalculateOpIfLibraryFunctionAvailable(
    Operation *op, ModuleOp library, LibraryFunctionKind libFuncKind,
    SmallVector<std::string> &libFuncNamesUsed,
    function_ref<FailureOr<SmallVector<Value>>(OpBuilder &, Location,
                                               ValueRange, func::FuncOp)>
        libFuncArgsBuilder) {
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  auto name = op->getName().stripDialect();
  // For value-semantic variant ops, i.e. valsem-ops (ops that are
  // mechanically consistent with existing torch conventions of in-place vs.
  //  out-of-place (value-semantic) variants), remove the prefix when
  // looking them up in the library.
  if (name.starts_with("valsem."))
    name = name.drop_front(strlen("valsem."));
  if (isa<OperatorOp>(op))
    name = cast<OperatorOp>(op)->getAttr("name").cast<StringAttr>().getValue();
  std::string libFuncName =
      (getLibraryFunctionPrefix(libFuncKind) + Twine(name)).str();
  auto libFunc = library.lookupSymbol<func::FuncOp>(libFuncName);
  if (!libFunc)
    return success();
  libFuncNamesUsed.push_back(libFuncName);
  OpBuilder b(op);
  Operation *calculate =
      createCalculateOp(b, loc, op->getResultTypes(), libFuncKind);
  op->replaceAllUsesWith(calculate);
  {
    // Move the op into the body of the `torch.{libFuncType}.calculate` op
    // and yield its results.
    OpBuilder b(context);
    Block *bodyBlock = b.createBlock(&calculate->getRegion(0));
    op->moveBefore(bodyBlock, bodyBlock->end());
    b.setInsertionPointAfter(op);
    createCalculateYieldOp(b, loc, op->getResults(), libFuncKind);
  }

  {
    OpBuilder b(context);
    b.createBlock(&calculate->getRegion(1));
    // Create the call to the library function!
    FailureOr<SmallVector<Value>> libFuncArgs =
        libFuncArgsBuilder(b, loc, op->getOperands(), libFunc);
    if (failed(libFuncArgs))
      return failure();
    auto call = b.create<mlir::func::CallOp>(loc, libFunc, *libFuncArgs);

    // Python models multiple results with a tuple, so we need to unpack it
    // if the op has multiple results.
    SmallVector<Value> unpackedResults;
    assert(call.getNumResults() == 1 &&
           "Multiple results are packed in a tuple in Python!");
    Value result = call.getResult(0);
    if (auto tupleType = result.getType().dyn_cast<Torch::TupleType>()) {
      auto unpack = b.create<PrimTupleUnpackOp>(
          loc, tupleType.getContainedTypes(), result);
      llvm::append_range(unpackedResults, unpack.getResults());
    } else {
      unpackedResults.push_back(result);
    }

    // Terminate the region.
    createCalculateYieldCalculationOp(b, loc, unpackedResults, libFuncKind);
  }
  return success();
}

void Torch::importLibraryFunctions(ModuleOp module, ModuleOp library,
                                   SmallVector<std::string> functionsNeeded) {
  // Import just the functions we need. This includes transitive callees,
  // so we use a worklist algorithm.
  llvm::StringSet<> importedFunctions;
  while (!functionsNeeded.empty()) {
    std::string symName = functionsNeeded.pop_back_val();
    if (importedFunctions.contains(symName))
      continue;
    auto func = library.lookupSymbol<func::FuncOp>(symName);
    assert(func && "broken library");
    // Move the function from the library to the module this pass
    // is running on. (this mutates the library, but we re-parse it each time
    // so this is safe to do).
    func->moveBefore(&module.getBody()->front());
    // Set the visibility to private so that the functions go away
    // nicely after we are done with them.
    func.setVisibility(SymbolTable::Visibility::Private);
    // Continue the DFS.
    importedFunctions.insert(symName);
    func.walk([&](func::CallOp op) {
      functionsNeeded.push_back(op.getCallee().str());
    });
  }
}

FailureOr<Value>
Torch::adjustFunctionArg(OpBuilder &b, Location loc, Value operand,
                         Type desiredType,
                         function_ref<Value(OpBuilder &, Location, Value, Type)>
                             baseTransformation) {
  operand = baseTransformation(b, loc, operand, desiredType);

  // No need for adjustment if they already match.
  auto operandType = operand.getType();
  if (operandType == desiredType)
    return operand;

  if (desiredType.isa<Torch::AnyType>()) {
    // Generator's are currently passed as Any because TorchScript cannot
    // compile a function with Generator type arguments.
    // Ignoring that hack, this is a correct handling of Any type should we need
    // to actually support it in the future.
    return b.create<DerefineOp>(loc, desiredType, operand).getResult();
  }

  // The type `!torch.number` can be an `int`, `float`, or `complex`.
  // TODO: Add a new type `Torch::ComplexType` to handle the complex case.
  if (desiredType.isa<Torch::NumberType>() &&
      operandType.isa<Torch::IntType, Torch::FloatType>()) {
    return b.create<DerefineOp>(loc, desiredType, operand).getResult();
  }

  // !torch.union<int, float, none> is the type used for optional
  // `Scalar` inputs. At compile time, such inputs will usually be
  // resolved to an `int`, `float`, or `None` so we need to derefine
  // to match the library function signature.
  if (auto unionType = desiredType.dyn_cast<Torch::UnionType>()) {
    if (llvm::all_of(unionType.getContainedTypes(), [](Type containedType) {
          return containedType
              .isa<Torch::IntType, Torch::FloatType, Torch::NoneType>();
        }))
      return b.create<DerefineOp>(loc, desiredType, operand).getResult();
  }

  // Operands with type `!torch.none` correspond to library function inputs with
  // types like `!torch.optional<...>` or `!torch.union<..., none>`, so here the
  // type is derefined to match the expected type of the library function.
  if (operandType.isa<Torch::NoneType>()) {
    assert(!desiredType.isa<Torch::NoneType>() &&
           "Don't expect library functions to have NoneType parameters");
    return b.create<DerefineOp>(loc, desiredType, operand).getResult();
  }

  // To keep things simple in shape functions, `Scalar` inputs are considered
  // `float`s. This is safe since output shape of torch ops never depends on the
  // dtype of input scalars. However, this also means we sometimes have to
  // manually turn `Scalar`s into `float`s when inserting the shape functions
  // into the IR.
  if (operandType.isa<Torch::NumberType>() &&
      desiredType.isa<Torch::FloatType>()) {
    return b.create<AtenFloatScalarOp>(loc, desiredType, operand).getResult();
  }

  // If the operand type is statically !torch.optional, then we need to do
  // different things for the None and non-None cases.
  // For the None case, we just need to derefine it to the desired type.
  // For the non-None case, we need to unwrap the optional type and then adjust
  // it recursively (which also takes care of derefining it to ultimate desired
  // type).
  // A case where this happens is `!torch.optional<vtensor>` ->
  // `!torch.optional<list<int>>>`.
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
        Region &thenRegion = primIf.getThenRegion();
        b.createBlock(&thenRegion, thenRegion.end());
        auto derefineNone = b.create<DerefineOp>(loc, desiredType, none);
        b.create<PrimIfYieldOp>(loc, ValueRange{derefineNone});
      }
      {
        Region &elseRegion = primIf.getElseRegion();
        b.createBlock(&elseRegion, elseRegion.end());
        auto downcasted = b.create<PrimUncheckedCastOp>(
            loc, operandOptionalType.getContainedType(), operand);
        FailureOr<Value> adjusted = adjustFunctionArg(
            b, loc, downcasted, desiredType, baseTransformation);
        if (failed(adjusted))
          return failure();
        b.create<PrimIfYieldOp>(loc, *adjusted);
      }
      b.setInsertionPointAfter(primIf);
      return primIf.getResult(0);
    }
  }

  // If the desired type is OptionalType, then recursively adjust the operand to
  // the contained type, then derefine it to `!torch.optional`. For example,
  // `!torch.vtensor -> !torch.optional<list<int>>>`.
  if (auto desiredOptionalType = desiredType.dyn_cast<Torch::OptionalType>()) {
    FailureOr<Value> adjusted = adjustFunctionArg(
        b, loc, operand, desiredOptionalType.getContainedType(),
        baseTransformation);
    if (failed(adjusted))
      return failure();
    return b.create<DerefineOp>(loc, desiredType, *adjusted).getResult();
  }

  if (auto desiredListType = desiredType.dyn_cast<Torch::ListType>()) {
    // Pseudocode:
    //
    // operand = ...
    // adjusted_list = []
    // for i in range(len(operand)):
    //     adjusted_list.append(adjust(operand[i]))
    // return adjusted_list
    auto providedType = operand.getType().cast<Torch::ListType>();
    Value adjustedList =
        b.create<PrimListConstructOp>(loc, desiredListType, ValueRange({}));
    // Create a for-like PrimLoopOp.
    Value maxTripCount = b.create<AtenLenTOp>(loc, operand);
    Value cTrue = b.create<Torch::ConstantBoolOp>(loc, true);
    auto loop = b.create<PrimLoopOp>(loc, TypeRange({}), maxTripCount,
                                     /*initialCondition=*/cTrue,
                                     /*iterArgsInit=*/ValueRange({}));

    // Create the loop body.
    {
      OpBuilder::InsertionGuard guard(b);
      Block *body =
          b.createBlock(&loop.getRegion(), loop.getRegion().begin(),
                        TypeRange({b.getType<Torch::IntType>()}), {loc});
      Value iterationNumber = body->getArgument(0);
      Value element = b.create<Aten__Getitem__TOp>(
          loc, providedType.getContainedType(), operand, iterationNumber);
      FailureOr<Value> adjustedElement =
          adjustFunctionArg(b, loc, element, desiredListType.getContainedType(),
                            baseTransformation);
      if (failed(adjustedElement))
        return failure();
      b.create<AtenAppendTOp>(loc, adjustedList.getType(), adjustedList,
                              *adjustedElement);
      b.create<PrimLoopConditionOp>(loc, /*shouldContinue=*/cTrue,
                                    /*iterArgs=*/ValueRange({}));
    }

    return adjustedList;
  }

  // The library functions use `float` where the operator
  // signature uses `Scalar` (see comments in torch_ods_gen.py for
  // explanation).
  if (desiredType.isa<Torch::FloatType>() &&
      operand.getType().isa<Torch::IntType>()) {
    return b.create<AtenFloatScalarOp>(loc, desiredType, operand).getResult();
  }

  // Pass the operand as-is.
  return operand;
}

LogicalResult
mlir::torch::Torch::loadExtraLibrary(const std::string &filename,
                                     OwningOpRef<ModuleOp> &moduleToAppendTo) {
  auto ctx = moduleToAppendTo->getContext();
  assert(ctx && "Module should be fully initialized.");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  OwningOpRef<ModuleOp> module_ =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, ctx);
  if (!module_) {
    llvm::errs() << "Error can't load file " << filename << "\n";
    return failure();
  }

  assert((moduleToAppendTo->getBodyRegion().empty() ||
          moduleToAppendTo->getBodyRegion().hasOneBlock()) &&
         "Module should have at most one block.");
  if (moduleToAppendTo->getBodyRegion().empty()) {
    moduleToAppendTo = std::move(module_);
  } else {
    Block *block = moduleToAppendTo->getBody(0);
    block->getOperations().splice(block->end(),
                                  module_->getBody(0)->getOperations());
  }

  return success();
}
