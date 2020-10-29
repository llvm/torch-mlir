//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "mlir/Transforms/InliningUtils.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP::tcp;

//===----------------------------------------------------------------------===//
// TCPDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TCPInlinerInterface : public DialectInlinerInterface {
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

void TCPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/TCP/IR/TCPOps.cpp.inc"
      >();
  addInterfaces<TCPInlinerInterface>();
}
