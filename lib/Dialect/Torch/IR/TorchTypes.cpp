//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// isValidSubtype
//===----------------------------------------------------------------------===//

bool Torch::isValidSubtype(Type subtype, Type type) {
  if (subtype == type)
    return true;

  if (auto any = type.dyn_cast<AnyType>())
    return true;

  if (auto number = type.dyn_cast<NumberType>())
    return subtype.isa<IntType>() || subtype.isa<Torch::FloatType>();

  if (auto optional = type.dyn_cast<OptionalType>())
    return isValidSubtype(subtype, optional.getContainedType()) ||
           subtype.isa<Torch::NoneType>();

  if (auto tuple = type.dyn_cast<Torch::TupleType>()) {
    if (!subtype.isa<Torch::TupleType>())
      return false;
    auto subtypes = subtype.cast<Torch::TupleType>().getContainedTypes();
    auto types = tuple.getContainedTypes();
    if (subtypes.size() != types.size())
      return false;
    for (auto t : llvm::zip(subtypes, types)) {
      if (!isValidSubtype(std::get<0>(t), std::get<1>(t)))
        return false;
    }
    return true;
  }

  // TODO: This is not subtyping according to PEP 483. See description
  // of NonValueTensorType.
  if (subtype.isa<NonValueTensorType>() && type.isa<NonValueTensorType>() &&
      type ==
          NonValueTensorType::getWithLeastStaticInformation(type.getContext()))
    return true;

  if (subtype.isa<ValueTensorType>() && type.isa<ValueTensorType>() &&
      type == ValueTensorType::getWithLeastStaticInformation(type.getContext()))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// NnModuleType
//===----------------------------------------------------------------------===//
void Torch::NnModuleType::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getClassName(), printer.getStream());
  printer << "\">";
}

Type Torch::NnModuleType::parse(AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  if (parser.parseLess())
    return Type();
  std::string className;
  if (parser.parseOptionalString(&className))
    return Type();
  if (parser.parseGreater())
    return Type();
  return get(ctxt, className);
}

//===----------------------------------------------------------------------===//
// NnModuleType
//===----------------------------------------------------------------------===//
template <typename TorchTy>
static void printTorchTypeWithContainedType(TorchTy type,
                                            ::mlir::AsmPrinter &printer) {
  printer << "<";
  printTorchDialectType(type.getContainedType(), printer);
  printer << ">";
}

template <typename TorchTy>
static Type parseTorchTypeWithContainedType(AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  if (parser.parseLess())
    return Type();

  // Parse the contained type, but forward directly to our internal parsing
  // of `torch` dialect types, so that we can parse nested types without
  // the `!torch.` prefix.
  Type containedType = parseTorchDialectType(parser);
  if (!containedType)
    return Type();
  if (parser.parseGreater())
    return Type();
  return TorchTy::get(ctxt, containedType);
}

#define DEFINE_OP_PRINTER_AND_PARSER(TorchTy)                                  \
  void Torch::TorchTy::print(AsmPrinter &printer) const {                      \
    printTorchTypeWithContainedType<Torch::TorchTy>(*this, printer);           \
  }                                                                            \
                                                                               \
  Type Torch::TorchTy::parse(AsmParser &parser) {                              \
    return parseTorchTypeWithContainedType<Torch::TorchTy>(parser);            \
  }

DEFINE_OP_PRINTER_AND_PARSER(ListType)
DEFINE_OP_PRINTER_AND_PARSER(OptionalType)
#undef DEFINE_OP_PRINTER_AND_PARSER

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

Type Torch::TupleType::parse(AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  if (parser.parseLess())
    return Type();
  if (!parser.parseOptionalGreater())
    return Torch::TupleType::get(context, {});

  SmallVector<Type> containedTypes;
  do {
    Type containedType = parseTorchDialectType(parser);
    if (!containedType)
      return Type();
    containedTypes.push_back(containedType);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return Type();
  return Torch::TupleType::get(context, containedTypes);
}

void Torch::TupleType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getContainedTypes(), printer, [&](Type type) {
    printTorchDialectType(type, printer);
  });
  printer << ">";
}

//===----------------------------------------------------------------------===//
// DictType
//===----------------------------------------------------------------------===//

Type Torch::DictType::parse(AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  if (parser.parseLess())
    return Type();
  Type keyType = parseTorchDialectType(parser);
  if (!keyType)
    return Type();
  if (parser.parseComma())
    return Type();
  Type valueType = parseTorchDialectType(parser);
  if (!valueType)
    return Type();
  if (parser.parseGreater())
    return Type();
  return get(ctxt, keyType, valueType);
}

void Torch::DictType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printTorchDialectType(getKeyType(), printer);
  printer << ", ";
  printTorchDialectType(getValueType(), printer);
  printer << ">";
}

//===----------------------------------------------------------------------===//
// BaseTensorType
//===----------------------------------------------------------------------===//

static bool isValidTorchDtype(Type dtype) {
  // Torch quantized types.
  if (dtype.isa<Torch::QInt8Type>())
    return true;
  // Builtin floating point types.
  if (dtype.isa<Float16Type, BFloat16Type, Float32Type, Float64Type>())
    return true;
  // Builtin integer types.
  if (IntegerType type = dtype.dyn_cast<IntegerType>()) {
    if (type.isSignless() && type.getWidth() == 1)
      return true;
    if (type.isSigned()) {
      for (unsigned width : {8, 16, 32, 64}) {
        if (type.getWidth() == width)
          return true;
      }
    }
    if (type.isUnsigned()) {
      return type.getWidth() == 8;
    }
  }
  return false;
}

bool BaseTensorType::hasSameSizesAndDtype(BaseTensorType other) const {
  return getOptionalSizes() == other.getOptionalSizes() &&
         getOptionalDtype() == other.getOptionalDtype();
}

Type BaseTensorType::getWithSizesAndDtypeFrom(BaseTensorType other) const {
  return getWithSizesAndDtype(other.getOptionalSizes(),
                              other.getOptionalDtype());
}

Type BaseTensorType::getWithSizesAndDtype(
    Optional<ArrayRef<int64_t>> optionalSizes, Type optionalDtype) const {
  if (isa<NonValueTensorType>())
    return NonValueTensorType::get(getContext(), optionalSizes, optionalDtype);
  if (isa<ValueTensorType>())
    return ValueTensorType::get(getContext(), optionalSizes, optionalDtype);
  llvm_unreachable("not a BaseTensorType!");
}

ValueTensorType BaseTensorType::getWithValueSemantics() const {
  if (auto tensor = dyn_cast<NonValueTensorType>())
    return tensor.getWithValueSemantics();
  if (auto tensor = dyn_cast<ValueTensorType>())
    return tensor;
  llvm_unreachable("not a BaseTensorType!");
}

static LogicalResult
verifyTensorType(function_ref<InFlightDiagnostic()> emitError,
                 Optional<ArrayRef<int64_t>> optionalSizes,
                 Type optionalDtype) {
  if (optionalDtype && !isValidTorchDtype(optionalDtype)) {
    emitError() << "invalid dtype " << optionalDtype
                << " for !torch.tensor type";
    return failure();
  }
  return success();
}

Type parseTensorType(MLIRContext *context, AsmParser &parser,
                     GetTensorTypeFn getTensorType) {
  llvm::SMLoc startLoc = parser.getCurrentLocation();
  if (parser.parseOptionalLess())
    return getTensorType(context,
                         /*optionalSizes=*/None, /*optionalDtype=*/Type());
  bool hasSizes;
  SmallVector<int64_t> sizes;
  if (succeeded(parser.parseOptionalStar())) {
    // Unranked.
    hasSizes = false;
  } else {
    // Parse list of sizes.
    hasSizes = true;
    if (parser.parseLSquare())
      return Type();
    for (bool first = true;; first = false) {
      if (!first) {
        if (failed(parser.parseOptionalComma())) {
          break;
        }
      }
      if (succeeded(parser.parseOptionalQuestion())) {
        sizes.push_back(-1);
        continue;
      }
      int64_t size;
      auto optionalInt = parser.parseOptionalInteger(size);
      if (optionalInt.hasValue()) {
        if (failed(*optionalInt))
          return Type();
        sizes.push_back(size);
        continue;
      }
      break;
    }
    if (parser.parseRSquare()) {
      return Type();
    }
  }
  if (parser.parseComma())
    return Type();
  Type optionalDtype;
  if (succeeded(parser.parseOptionalKeyword("unk"))) {
    // Unknown dtype.
  } else {
    // Known dtype.
    if (parser.parseType(optionalDtype))
      return Type();
  }
  if (parser.parseGreater())
    return Type();
  Optional<ArrayRef<int64_t>> optionalSizes;
  if (hasSizes)
    optionalSizes.emplace(sizes);

  if (failed(verifyTensorType([&]() { return parser.emitError(startLoc); },
                              optionalSizes, optionalDtype)))
    return Type();

  return getTensorType(context, optionalSizes, optionalDtype);
}

static void printTensorType(AsmPrinter &printer,
                            Optional<ArrayRef<int64_t>> optionalSizes,
                            Type optionalDtype) {
  if (!optionalSizes && !optionalDtype)
    return;
  printer << "<";
  if (optionalSizes) {
    printer << "[";
    for (auto it : llvm::enumerate(*optionalSizes)) {
      if (it.index() > 0)
        printer << ",";
      if (it.value() < 0)
        printer << "?";
      else
        printer << it.value();
    }
    printer << "]";
  } else {
    printer << "*";
  }
  printer << ",";
  if (optionalDtype)
    printer.printType(optionalDtype);
  else
    printer << "unk";
  printer << ">";
}

//===----------------------------------------------------------------------===//
// NonValueTensorType
//===----------------------------------------------------------------------===//

ValueTensorType NonValueTensorType::getWithValueSemantics() const {
  return ValueTensorType::get(getContext(), getOptionalSizes(),
                              getOptionalDtype());
}

NonValueTensorType
NonValueTensorType::getWithLeastStaticInformation(MLIRContext *context) {
  return NonValueTensorType::get(context,
                                 /*optionalSizes=*/None,
                                 /*optionalDtype=*/Type());
}

LogicalResult
NonValueTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                           Optional<ArrayRef<int64_t>> optionalSizes,
                           Type optionalDtype) {
  return verifyTensorType(emitError, optionalSizes, optionalDtype);
}

Type NonValueTensorType::parse(AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  return parseTensorType(
      context, parser,
      [](MLIRContext *context, Optional<ArrayRef<int64_t>> optionalSizes,
         Type optionalType) {
        return NonValueTensorType::get(context, optionalSizes, optionalType);
      });
}

void NonValueTensorType::print(AsmPrinter &printer) const {
  printTensorType(printer, getOptionalSizes(), getOptionalDtype());
}

//===----------------------------------------------------------------------===//
// ValueTensorType
//===----------------------------------------------------------------------===//

NonValueTensorType ValueTensorType::getWithoutValueSemantics() const {
  return NonValueTensorType::get(getContext(), getOptionalSizes(),
                                 getOptionalDtype());
}

ValueTensorType
ValueTensorType::getWithLeastStaticInformation(MLIRContext *context) {
  return ValueTensorType::get(context,
                              /*optionalSizes=*/None,
                              /*optionalDtype=*/Type());
}

static Type convertDtypeToBuiltinElementType(MLIRContext *context, Type dtype) {
  if (auto floatType = dtype.dyn_cast<mlir::FloatType>()) {
    return dtype;
  } else if (auto integerType = dtype.dyn_cast<IntegerType>()) {
    return IntegerType::get(context, integerType.getWidth(),
                            IntegerType::Signless);
  }
  emitError(UnknownLoc::get(context))
      << "unimplemented: conversion of dtype " << dtype
      << " to builtin tensor element type";
  return nullptr;
}

TensorType ValueTensorType::toBuiltinTensor() const {
  if (!hasDtype())
    return nullptr;
  if (!hasSizes())
    return UnrankedTensorType::get(getDtype());
  Type elementType = convertDtypeToBuiltinElementType(getContext(), getDtype());
  if (!elementType)
    return nullptr;
  return RankedTensorType::get(getSizes(), elementType);
}

LogicalResult
ValueTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                        Optional<ArrayRef<int64_t>> optionalSizes,
                        Type optionalDtype) {
  return verifyTensorType(emitError, optionalSizes, optionalDtype);
}

Type ValueTensorType::parse(AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  return parseTensorType(
      context, parser,
      [](MLIRContext *context, Optional<ArrayRef<int64_t>> optionalSizes,
         Type optionalType) {
        return ValueTensorType::get(context, optionalSizes, optionalType);
      });
}

void ValueTensorType::print(AsmPrinter &printer) const {
  printTensorType(printer, getOptionalSizes(), getOptionalDtype());
}

Type Torch::meetTensorTypes(BaseTensorType lhs, BaseTensorType rhs) {
  assert(((lhs.isa<ValueTensorType>() && rhs.isa<ValueTensorType>()) ||
          (lhs.isa<NonValueTensorType>() && rhs.isa<NonValueTensorType>())) &&
         "expected lhs and rhs to have same sense of value semantics");

  // First, calculate the dtype.

  // If the dtypes are contradictory, return null.
  if (lhs.hasDtype() && rhs.hasDtype() && lhs.getDtype() != rhs.getDtype())
    return nullptr;
  Type dtype;
  // If we have a dtype, use it. If not, then the dtype Type remains in its
  // default null state, which the constructor of ValueTensorType treats as
  // "unknown".
  if (lhs.hasDtype() || rhs.hasDtype()) {
    dtype = lhs.hasDtype() ? lhs.getDtype() : rhs.getDtype();
  }

  // Then, calculate the sizes and return the new Type.

  // If neither has sizes, we have nothing left to do.
  if (!lhs.hasSizes() && !rhs.hasSizes()) {
    return ValueTensorType::get(lhs.getContext(), /*optionalSizes=*/None,
                                dtype);
  }

  // If the number of sizes is different, the two types are contradictory.
  if (lhs.hasSizes() && rhs.hasSizes() &&
      lhs.getSizes().size() != rhs.getSizes().size()) {
    return nullptr;
  }

  // Either lhs or rhs has sizes. If either one doesn't have sizes, we can
  // replace it with the other one's sizes, since the meet logic below is
  // idempotent.
  ArrayRef<int64_t> lhsSizes = lhs.hasSizes() ? lhs.getSizes() : rhs.getSizes();
  ArrayRef<int64_t> rhsSizes = rhs.hasSizes() ? rhs.getSizes() : lhs.getSizes();
  // Meet the sizes.
  SmallVector<int64_t> newSizes;
  for (int i = 0, e = lhsSizes.size(); i < e; i++) {
    if (lhsSizes[i] == rhsSizes[i]) {
      newSizes.push_back(lhsSizes[i]);
    } else if (lhsSizes[i] == kUnknownSize) {
      newSizes.push_back(rhsSizes[i]);
    } else if (rhsSizes[i] == kUnknownSize) {
      newSizes.push_back(lhsSizes[i]);
    } else {
      // The two sizes are contradictory.
      return nullptr;
    }
  }

  return lhs.getWithSizesAndDtype(makeArrayRef(newSizes), dtype);
}
