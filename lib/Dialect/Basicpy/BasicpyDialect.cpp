//===- BasicpyDialect.cpp - Basic python dialect ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Basicpy/BasicpyDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "npcomp/Dialect/Basicpy/BasicpyOps.h"

using namespace mlir;
using namespace mlir::NPCOMP::Basicpy;

BasicpyDialect::BasicpyDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Basicpy/BasicpyOps.cpp.inc"
      >();
  // addTypes<AnyDtypeType>();
}

// Type BasicpyDialect::parseType(DialectAsmParser &parser) const {
//   parser.emitError(parser.getNameLoc(), "unknown numpy type");
//   return Type();
// }

// void BasicpyDialect::printType(Type type, DialectAsmPrinter &os) const {
//   switch (type.getKind()) {
//   default:
//     llvm_unreachable("unexpected 'basicpy' type kind");
//   }
// }
