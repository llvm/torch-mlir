//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Npcomprt/IR/NpcomprtDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtOps.h"

using namespace mlir;
using namespace mlir::NPCOMP::npcomprt;

NpcomprtDialect::NpcomprtDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtOps.cpp.inc"
      >();
  addTypes<TensorType>();
}

Type NpcomprtDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "tensor")
    return TensorType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type in 'npcomprt' dialect: ")
      << keyword;
  return Type();
}

void NpcomprtDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  case NpcomprtTypes::Kind::TensorType:
    os << "tensor";
    return;
  default:
    llvm_unreachable("unexpected 'npcomprt' type kind");
  }
}
