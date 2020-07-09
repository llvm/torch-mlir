//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Npcomprt/IR/NpcomprtOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/SymbolTable.h"
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP::npcomprt;

//===----------------------------------------------------------------------===//
// ModuleMetadataOp
//===----------------------------------------------------------------------===//

static void printModuleMetadataOp(OpAsmPrinter &p, ModuleMetadataOp &op) {
  p << "npcomprt.module_metadata";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
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
  auto *module = op.getParentOp()->getParentOp();
  auto func = dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupSymbolIn(module, op.funcName()));
  if (!func)
    return op.emitError() << "must reference a valid func";

  if (op.numInputs().getLimitedValue() != func.getNumArguments())
    return op.emitError() << "must agree on number of inputs";
  if (op.numOutputs().getLimitedValue() != func.getNumResults())
    return op.emitError() << "must agree on number of outputs";
  return success();
}

namespace mlir {
namespace NPCOMP {
namespace npcomprt {
#define GET_OP_CLASSES
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtOps.cpp.inc"
} // namespace npcomprt
} // namespace NPCOMP
} // namespace mlir
