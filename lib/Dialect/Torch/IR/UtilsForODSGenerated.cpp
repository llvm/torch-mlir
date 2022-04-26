//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities referenced by ODS generated code.
//
// The utilities defined here are only meant for use by ODS generated code.
// If something is of wider use, then it should be moved elsewhere.
//
//===----------------------------------------------------------------------===//

#include "UtilsForODSGenerated.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

ParseResult Torch::parseDefaultTorchOp(OpAsmParser &parser,
                                       OperationState &result, int numOperands,
                                       int numResults) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/numOperands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();
  if (numOperands > 0) {
    SmallVector<Type> operandTypes;
    if (parser.parseTypeList(operandTypes))
      return failure();
    if (parser.resolveOperands(operands, operandTypes, loc, result.operands))
      return failure();
  }
  if (numOperands > 0 && numResults > 0) {
    if (parser.parseArrow())
      return failure();
  }
  if (numResults > 0) {
    if (parser.parseTypeList(result.types))
      return failure();
  }
  return success();
}

void Torch::printDefaultTorchOp(OpAsmPrinter &p, Operation *op, int numOperands,
                                int numResults) {
  if (numOperands > 0) {
    p << ' ';
    llvm::interleaveComma(op->getOperands(), p);
  }
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << " : ";
  if (numOperands > 0)
    llvm::interleaveComma(op->getOperandTypes(), p);
  if (numOperands > 0 && numResults > 0)
    p << " -> ";
  if (numResults > 0)
    llvm::interleaveComma(op->getResultTypes(), p);
}
