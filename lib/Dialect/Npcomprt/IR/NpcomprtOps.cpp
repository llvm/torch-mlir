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
#include "mlir/IR/TypeUtilities.h"
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP::npcomprt;

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static void printGlobalOp(OpAsmPrinter &p, GlobalOp &op) {
  p << "npcomprt.global ";
  p.printSymbolName(op.sym_name());
  p << ' ';
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     /*elidedAttrs=*/{"sym_name", "value"});
  p.printAttribute(op.valueAttr());
}

static ParseResult parseGlobalOp(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  Attribute valueAttr;
  if (parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyGetGlobalOp(GetGlobalOp op) {
  auto global = SymbolTable::lookupNearestSymbolFrom<GlobalOp>(op, op.global());
  if (!global)
    return op.emitError() << "must reference a valid npcomprt.global";
  auto globalType = global.value().getType();
  auto resultType = op.getType().cast<ShapedType>();
  if (globalType.getElementType() != resultType.getElementType())
    return op.emitError() << "inconsistent with element type of global";
  return success();
}

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
