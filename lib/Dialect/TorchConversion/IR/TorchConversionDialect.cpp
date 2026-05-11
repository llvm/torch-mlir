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
#include "mlir/IR/DialectResourceBlobManager.h"
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
  assert(!isa<Torch::NonValueTensorType>(type));
  if (auto integerType = dyn_cast<Torch::IntType>(type))
    return Torch::ConstantIntOp::create(builder, loc, cast<IntegerAttr>(value));

  if (auto floatType = dyn_cast<Torch::FloatType>(type))
    return Torch::ConstantFloatOp::create(builder, loc, cast<FloatAttr>(value));

  if (auto numberType = dyn_cast<Torch::NumberType>(type)) {
    if (auto floatValue = dyn_cast<mlir::FloatAttr>(value)) {
      return Torch::ConstantNumberOp::create(builder, loc, floatValue);
    } else if (auto intValue = dyn_cast<mlir::IntegerAttr>(value)) {
      return Torch::ConstantNumberOp::create(builder, loc, intValue);
    }
  }

  if (isa<Torch::BoolType>(type)) {
    return Torch::ConstantBoolOp::create(builder, loc,
                                         cast<IntegerAttr>(value));
  }

  if (isa<Torch::NoneType>(type))
    return Torch::ConstantNoneOp::create(builder, loc);

  if (auto stringAttr = dyn_cast<StringAttr>(value))
    return Torch::ConstantStrOp::create(builder, loc, stringAttr);

  if (auto valueTensorType = dyn_cast<Torch::ValueTensorType>(type)) {
    if (auto elementsAttr = dyn_cast<ElementsAttr>(value)) {
      // `torch_c.from_builtin_tensor` allows resource-backed integer tensors to
      // use signless storage while the destination torch dtype carries explicit
      // signedness. Preserve that requested signedness on the literal attr so
      // `torch.vtensor.literal` infers the same type as its declared result.
      auto literalType = cast<ShapedType>(elementsAttr.getType());
      if (auto resourceAttr =
              dyn_cast<DenseResourceElementsAttr>(elementsAttr)) {
        auto literalIntType =
            dyn_cast<IntegerType>(literalType.getElementType());
        auto resultIntType =
            dyn_cast_or_null<IntegerType>(valueTensorType.getDtype());
        if (literalIntType && resultIntType && literalIntType.isSignless() &&
            !resultIntType.isSignless() &&
            literalIntType.getWidth() == resultIntType.getWidth()) {
          elementsAttr = DenseResourceElementsAttr::get(
              literalType.clone(resultIntType), resourceAttr.getRawHandle());
        }
      }
      return Torch::ValueTensorLiteralOp::create(builder, loc, type,
                                                 elementsAttr);
    }
    return nullptr;
  }

  return arith::ConstantOp::materialize(builder, value, type, loc);
}
