//===- NumpyDialect.cpp - Core numpy dialect --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"

using namespace mlir;
using namespace mlir::NPCOMP::Numpy;

NumpyDialect::NumpyDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Numpy/IR/NumpyOps.cpp.inc"
      >();
  addTypes<AnyDtypeType>();
}

Type NumpyDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "any_dtype")
    return AnyDtypeType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown numpy type: ") << keyword;
  return Type();
}

void NumpyDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  case NumpyTypes::AnyDtypeType:
    os << "any_dtype";
    return;
  default:
    llvm_unreachable("unexpected 'numpy' type kind");
  }
}
