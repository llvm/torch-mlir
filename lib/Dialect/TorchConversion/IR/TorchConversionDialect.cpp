//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
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
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
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

//===----------------------------------------------------------------------===//
// Constant materializer.
//===----------------------------------------------------------------------===//

Operation *TorchConversionDialect::materializeConstant(OpBuilder &builder,
                                                       Attribute value,
                                                       Type type,
                                                       Location loc) {
  if (auto integerType = dyn_cast<Torch::IntType>(type))
    return builder.create<Torch::ConstantIntOp>(loc, cast<IntegerAttr>(value));

  if (auto floatType = dyn_cast<Torch::FloatType>(type))
    return builder.create<Torch::ConstantFloatOp>(loc, cast<FloatAttr>(value));

  if (isa<Torch::BoolType>(type)) {
    return builder.create<Torch::ConstantBoolOp>(loc, cast<IntegerAttr>(value));
  }

  return arith::ConstantOp::materialize(builder, value, type, loc);
}
