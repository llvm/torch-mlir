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
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::tcp;

//===----------------------------------------------------------------------===//
// TensorToMemrefOp
//===----------------------------------------------------------------------===//

OpFoldResult TensorToMemrefOp::fold(ArrayRef<Attribute> operands) {
  if (auto memrefToTensor = tensor().getDefiningOp<tcp::MemrefToTensorOp>())
    return memrefToTensor.memref();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ShapedResultsOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyShapedResultsOp(ShapedResultsOp op) {
  if (op.getNumOperands() != op.getNumResults())
    return op.emitError() << "number of operands must equal number of results";
  if (op.getNumOperands() == 0)
    return op.emitError() << "must have at least one operand/result";
  return RegionBranchOpInterface::verifyTypes(op);
}

static void printShapedResultsOp(OpAsmPrinter &p, ShapedResultsOp &op) {
  p << "tcp.shaped_results ";
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
  p.printOperands(op.getOperands());
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
  p << " : ";
  interleaveComma(op.getOperandTypes(), p);
  p << " -> ";
  interleaveComma(op.getResultTypes(), p);
}

static ParseResult parseShapedResultsOp(OpAsmParser &parser,
                                        OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  SmallVector<OpAsmParser::OperandType, 6> operands;
  if (parser.parseOperandList(operands))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None))
    return failure();
  SmallVector<Type, 6> inputTypes;
  if (parser.parseColonTypeList(inputTypes))
    return failure();
  if (parser.resolveOperands(operands, inputTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  if (parser.parseArrowTypeList(result.types))
    return failure();
  return success();
}

void ShapedResultsOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (index.hasValue())
    regions.push_back(RegionSuccessor(getResults()));
  else
    regions.push_back(RegionSuccessor(&body()));
}

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

#define GET_OP_CLASSES
#include "npcomp/Dialect/TCP/IR/TCPOps.cpp.inc"
