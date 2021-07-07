//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

#include "npcomp/Dialect/Torch/IR/TorchDialect.cpp.inc"

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

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Type-related Dialect methods.
//===----------------------------------------------------------------------===//

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


//===----------------------------------------------------------------------===//
// Dialect-level verifiers.
//===----------------------------------------------------------------------===//

LogicalResult TorchDialect::verifyRegionArgAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute namedAttr) {
  if (namedAttr.first == "torch.type_bound") {
    auto func = dyn_cast<FuncOp>(op);
    if (!func)
      return op->emitError() << "'torch.type_bound' must be attached to a func";
    TypeAttr attr = namedAttr.second.dyn_cast<TypeAttr>();
    if (!attr)
      return op->emitError() << "'torch.type_bound' must be TypeAttr";
    auto type = attr.getValue().dyn_cast<BaseTensorType>();
    if (!type)
      return op->emitError() << "'torch.type_bound' must be of "
                                "!torch.tensor/!torch.vtensor type";
    if (!func.getType().getInput(argIndex).isa<BaseTensorType>())
      return op->emitError() << "'torch.type_bound' must be attached to an "
                                "argument of !torch.tensor/!torch.vtensor type";
    return success();
  }

  return op->emitError() << "unknown region arg attribute '" << namedAttr.first
                         << "'";
}

//===----------------------------------------------------------------------===//
// Constant materializer.
//===----------------------------------------------------------------------===//

Operation *TorchDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto integerType = type.dyn_cast<Torch::IntType>())
    return builder.create<Torch::ConstantIntOp>(loc, value.cast<IntegerAttr>());

  if (auto floatType = type.dyn_cast<Torch::FloatType>())
    return builder.create<Torch::ConstantFloatOp>(loc, value.cast<FloatAttr>());

  if (type.isa<Torch::BoolType>()) {
    return builder.create<Torch::ConstantBoolOp>(loc,
                                                 value.cast<IntegerAttr>());
  }

  if (type.isa<Torch::NoneType>())
    return builder.create<ConstantNoneOp>(loc);

  if (auto stringAttr = value.dyn_cast<StringAttr>())
    return builder.create<ConstantStrOp>(loc, stringAttr);

  if (auto elementsAttr = value.dyn_cast<ElementsAttr>()) {
    // Only !torch.vtensor can be constant folded. !torch.tensor has
    // non-trivial aliasing semantics which prevent deduplicating it.
    assert(type.isa<ValueTensorType>() && "should be a vtensor type!");
    return builder.create<ValueTensorLiteralOp>(loc, elementsAttr);
  }

  return nullptr;
}
