//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/TCP/IR/TCPOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::tcp;

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static void printGlobalOp(OpAsmPrinter &p, GlobalOp &op) {
  p << "tcp.global ";
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

//===----------------------------------------------------------------------===//
// ShapeObserveErrorOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeObserveErrorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(NoneType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// GetExtentOp
//===----------------------------------------------------------------------===//

LogicalResult
GetExtentOp::inferReturnTypes(MLIRContext *context, Optional<Location> location,
                              ValueRange operands, DictionaryAttr attributes,
                              RegionRange regions,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(IndexType::get(context));
  return success();
}

namespace mlir {
namespace NPCOMP {
namespace tcp {
#define GET_OP_CLASSES
#include "npcomp/Dialect/TCP/IR/TCPOps.cpp.inc"
} // namespace tcp
} // namespace NPCOMP
} // namespace mlir
