//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TorchInlinerInterface : public DialectInlinerInterface {
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
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Top-level parsing/printing of types for TorchDialect.
//===----------------------------------------------------------------------===//
//
// Unfortunately, TorchDialect::parseType/printType are non-static member
// functions, even though they don't depend on any instance state of the
// dialect. This is problematic, for example, when wanting to call these
// functions directly from type printers/parsers.
//
// So define some helpers that are free functions.

/// Parse a type registered to this dialect.
Type Torch::parseTorchDialectType(AsmParser &parser) {
  SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  parser.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `"
                            << TorchDialect::getDialectNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect.
void Torch::printTorchDialectType(Type type, AsmPrinter &printer) {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
}

//===----------------------------------------------------------------------===//
// Torch dialect parseType/printType methods.
//===----------------------------------------------------------------------===//

/// Parse a type registered to this dialect.
Type TorchDialect::parseType(DialectAsmParser &parser) const {
  return parseTorchDialectType(parser);
}
/// Print a type registered to this dialect.
void TorchDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printTorchDialectType(type, printer);
}

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//

void TorchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "torch-mlir/Dialect/Torch/IR/TorchOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.cpp.inc"
      >();
  addInterfaces<TorchInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Dialect-level verifiers.
//===----------------------------------------------------------------------===//

LogicalResult TorchDialect::verifyRegionArgAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute namedAttr) {
  if (namedAttr.getName().getValue() == "torch.type_bound") {
    auto func = dyn_cast<func::FuncOp>(op);
    if (!func)
      return op->emitError() << "'torch.type_bound' must be attached to a func";
    TypeAttr attr = namedAttr.getValue().dyn_cast<TypeAttr>();
    if (!attr)
      return op->emitError() << "'torch.type_bound' must be TypeAttr";
    auto type = attr.getValue().dyn_cast<BaseTensorType>();
    if (!type)
      return op->emitError() << "'torch.type_bound' must be of "
                                "!torch.tensor/!torch.vtensor type";
    if (!func.getFunctionType().getInput(argIndex).isa<BaseTensorType>())
      return op->emitError() << "'torch.type_bound' must be attached to an "
                                "argument of !torch.tensor/!torch.vtensor type";
    return success();
  }

  return op->emitError() << "unknown region arg attribute '"
                         << namedAttr.getName().getValue() << "'";
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

  if (auto numberType = type.dyn_cast<Torch::NumberType>()) {
    if (auto floatValue = value.dyn_cast<mlir::FloatAttr>()) {
      return builder.create<Torch::ConstantNumberOp>(loc, floatValue);
    } else if (auto intValue = value.dyn_cast<mlir::IntegerAttr>()) {
      return builder.create<Torch::ConstantNumberOp>(loc, intValue);
    }
  }

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

//===----------------------------------------------------------------------===//
// OptionalType and ListType
//===----------------------------------------------------------------------===//

void OptionalType::print(AsmPrinter &printer) const {
  printer << "<";
  // Print the contained type without the `!torch.` prefix.
  printTorchDialectType(getImpl()->containedType, printer);
  printer << ">";
}

void ListType::print(AsmPrinter &printer) const {
  printer << "<";
  // Print the contained type without the `!torch.` prefix.
  printTorchDialectType(getImpl()->containedType, printer);
  printer << ">";
}

Type OptionalType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();

  // Parse the contained type, but forward directly to our internal parsing
  // of `torch` dialect types, so that we can parse nested types without
  // the `!torch.` prefix.
  Type containedType = parseTorchDialectType(odsParser);
  if (!containedType)
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), containedType);
}

Type ListType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();

  // Parse the contained type, but forward directly to our internal parsing
  // of `torch` dialect types, so that we can parse nested types without
  // the `!torch.` prefix.
  Type containedType = parseTorchDialectType(odsParser);
  if (!containedType)
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), containedType);
}

//===----------------------------------------------------------------------===//
// DictType
//===----------------------------------------------------------------------===//

void DictType::print(AsmPrinter &printer) const {
  printer << "<";
  printTorchDialectType(getImpl()->keyType, printer);
  printer << ", ";
  printTorchDialectType(getImpl()->valueType, printer);
  printer << ">";
}

Type DictType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();
  Type keyType = parseTorchDialectType(odsParser);
  if (!keyType)
    return Type();
  if (odsParser.parseComma())
    return Type();
  Type valueType = parseTorchDialectType(odsParser);
  if (!valueType)
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), keyType, valueType);
}

//===----------------------------------------------------------------------===//
// NnModuleType
//===----------------------------------------------------------------------===//

void NnModuleType::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getImpl()->className, printer.getStream());
  printer << "\">";
}

Type NnModuleType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();
  std::string className;
  if (odsParser.parseOptionalString(&className))
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), className);
}
