//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

static SmallVector<StringRef, 4> strArrayAttrToVector(ArrayAttr array) {
  SmallVector<StringRef, 4> strings;
  strings.reserve(array.size());
  for (auto stringAttr : array) {
    strings.push_back(stringAttr.cast<StringAttr>().getValue());
  }
  return strings;
}

//===----------------------------------------------------------------------===//
// KernelCallOp
//===----------------------------------------------------------------------===//

KernelMetadata KernelCallOp::getTorchKernelMetadata() {
  return KernelMetadata{
      .kernelName = kernelName(),
      .isVararg = sigIsVararg(),
      .isVarret = sigIsVarret(),
      .argTypes = strArrayAttrToVector(sigArgTypes()),
      .returnTypes = strArrayAttrToVector(sigRetTypes()),
  };
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

LogicalResult MethodOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto func = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, function());
  if (!func)
    return emitError() << "'" << function()
                       << "' does not reference a valid function";
  return success();
}

//===----------------------------------------------------------------------===//
// NnModuleOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(NnModuleOp op) {
  for (Operation &child : *op.getBody())
    if (!isa<AttrOp, MethodOp, NnModuleTerminatorOp>(&child))
      return child.emitOpError() << "is not allowed inside `torch.nn_module`";
  return success();
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchOps.cpp.inc"
