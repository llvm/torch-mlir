//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Refback/IR/RefbackOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::refback;

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static void printGlobalOp(OpAsmPrinter &p, GlobalOp &op) {
  p << "refback.global ";
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
// GetGlobalMemrefOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyGetGlobalMemrefOp(GetGlobalMemrefOp op) {
  auto global = SymbolTable::lookupNearestSymbolFrom<GlobalOp>(op, op.global());
  if (!global)
    return op.emitError() << "must reference a valid symbol";
  auto globalType = global.value().getType();
  auto resultType = op.getType().cast<ShapedType>();
  if (failed(
          verifyCompatibleShape(globalType.getShape(), resultType.getShape())))
    return op.emitError() << "inconsistent with shape of global";
  if (globalType.getElementType() != resultType.getElementType())
    return op.emitError() << "inconsistent with element type of global";
  return success();
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/Refback/IR/RefbackOps.cpp.inc"
