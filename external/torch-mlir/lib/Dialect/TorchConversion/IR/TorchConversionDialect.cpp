//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TorchConversionInlinerInterface : public DialectInlinerInterface {
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
// Dialect initialize method.
//===----------------------------------------------------------------------===//

void TorchConversionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.cpp.inc"
      >();
  addInterfaces<TorchConversionInlinerInterface>();
}
