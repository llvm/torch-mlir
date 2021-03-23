//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::NPCOMP::Torch;

//===----------------------------------------------------------------------===//
// Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TorchInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

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
  addInterfaces<TorchInlinerInterface>();
}

Type TorchDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  Type type;
  if (generatedTypeParser(getContext(), parser, keyword, type).hasValue())
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
