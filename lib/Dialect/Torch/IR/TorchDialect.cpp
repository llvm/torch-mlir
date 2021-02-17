//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::NPCOMP::Torch;

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchTypes.cpp.inc"

void TorchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Torch/IR/TorchOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "npcomp/Dialect/Torch/IR/TorchTypes.cpp.inc"
      >();
}

Type TorchDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  if (Type type = generatedTypeParser(getContext(), parser, keyword))
    return type;

  parser.emitError(parser.getNameLoc(), "invalid 'torch' type: `")
      << keyword << "'";
  return Type();
}

void TorchDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (failed(generatedTypePrinter(type, printer)))
    llvm_unreachable("unknown 'torch' type");
}

#include "npcomp/Dialect/Torch/IR/OpInterfaces.h"
