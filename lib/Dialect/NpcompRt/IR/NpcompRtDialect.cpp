//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/NpcompRt/IR/NpcompRtDialect.h"
#include "npcomp/Dialect/NpcompRt/IR/NpcompRtOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::NPCOMP::npcomp_rt;

NpcompRtDialect::NpcompRtDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/NpcompRt/IR/NpcompRtOps.cpp.inc"
      >();
  addTypes<BufferViewType>();
}

Type NpcompRtDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "buffer_view")
    return BufferViewType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type in 'npcomp_rt' dialect: ")
      << keyword;
  return Type();
}

void NpcompRtDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  case NpcompRtTypes::Kind::BufferViewType:
    os << "buffer_view";
    return;
  default:
    llvm_unreachable("unexpected 'npcomp_rt' type kind");
  }
}

