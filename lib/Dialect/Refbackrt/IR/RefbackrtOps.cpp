//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Refbackrt/IR/RefbackrtOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP::refbackrt;

//===----------------------------------------------------------------------===//
// ModuleMetadataOp
//===----------------------------------------------------------------------===//

static void printModuleMetadataOp(OpAsmPrinter &p, ModuleMetadataOp &op) {
  p.printOptionalAttrDictWithKeyword(op->getAttrs());
  p.printRegion(op.metadatas(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseModuleMetadataOp(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None))
    return failure();
  ModuleMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// FuncMetadataOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(FuncMetadataOp op) {
  auto *module = op->getParentOp()->getParentOp();
  auto func = dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupSymbolIn(module, op.funcName()));
  if (!func)
    return op.emitError() << "must reference a valid func";

  if (op.numInputs() != func.getNumArguments())
    return op.emitError() << "must agree on number of inputs";
  if (op.numOutputs() != func.getNumResults())
    return op.emitError() << "must agree on number of outputs";

  if (op.numInputs() > 0) {
    if (op.numInputs() != op.inputArgTypes()->size()) {
      return op.emitError() << "number of inputTypes must match number of inputs";
    }
  }
  if (op.numOutputs() > 0) {
    if (op.numOutputs() != op.outputArgTypes()->size())
      return op.emitError() << "number of outputTypes must match number of outputs";
  }
  return success();
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtOps.cpp.inc"
