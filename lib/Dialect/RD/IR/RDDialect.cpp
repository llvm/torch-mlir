//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::NPCOMP::rd;

//===----------------------------------------------------------------------===//
// TCPDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct RDInlinerInterface : public DialectInlinerInterface {
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

void RDDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/RD/IR/RDOps.cpp.inc"
      >();
  addTypes<DatasetType, IteratorType>();
  // addInterfaces<RDInlinerInterface>();
}

::mlir::Type RDDialect::parseType(::mlir::DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "Dataset")
    return DatasetType::get(getContext());
  if (keyword == "Iterator")
    return IteratorType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown rd type");
  return Type();
}

void RDDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
    .Case<DatasetType>([&](Type) { os << "Dataset"; })
    .Case<IteratorType>([&](Type) { os << "Iterator"; })
    .Default(
      [&](Type) { llvm_unreachable("unexpected 'rd' type kind"); });
}
