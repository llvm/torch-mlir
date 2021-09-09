//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

Type Torch::TupleType::parse(MLIRContext *context, DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();
  if (!parser.parseOptionalGreater())
    return Torch::TupleType::get(context, {});

  SmallVector<Type> containedTypes;
  do {
    Type containedType;
    if (parser.parseType(containedType))
      return Type();
    containedTypes.push_back(containedType);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return Type();
  return Torch::TupleType::get(context, containedTypes);
}

void Torch::TupleType::print(::mlir::DialectAsmPrinter &printer) const {
  printer << "tuple<";
  llvm::interleaveComma(getContainedTypes(), printer);
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

Type parseTensorType(MLIRContext *context, DialectAsmParser &parser,
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

static void printTensorType(DialectAsmPrinter &printer,
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

NonValueTensorType NonValueTensorType::getFromShaped(ShapedType type) {
  return NonValueTensorType::get(type.getContext(),
                                 type.hasRank() ? type.getShape()
                                                : Optional<ArrayRef<int64_t>>(),
                                 type.getElementType());
}

LogicalResult
NonValueTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                           Optional<ArrayRef<int64_t>> optionalSizes,
                           Type optionalDtype) {
  return verifyTensorType(emitError, optionalSizes, optionalDtype);
}

Type NonValueTensorType::parse(MLIRContext *context, DialectAsmParser &parser) {
  return parseTensorType(
      context, parser,
      [](MLIRContext *context, Optional<ArrayRef<int64_t>> optionalSizes,
         Type optionalType) {
        return NonValueTensorType::get(context, optionalSizes, optionalType);
      });
}

void NonValueTensorType::print(DialectAsmPrinter &printer) const {
  printer << "tensor";
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

ValueTensorType ValueTensorType::getFromShaped(ShapedType type) {
  return ValueTensorType::get(type.getContext(),
                              type.hasRank() ? type.getShape()
                                             : Optional<ArrayRef<int64_t>>(),
                              type.getElementType());
}

TensorType ValueTensorType::toBuiltinTensor() const {
  if (!hasDtype())
    return nullptr;
  if (!hasSizes())
    return UnrankedTensorType::get(getDtype());
  return RankedTensorType::get(getSizes(), getDtype());
}

LogicalResult
ValueTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                        Optional<ArrayRef<int64_t>> optionalSizes,
                        Type optionalDtype) {
  return verifyTensorType(emitError, optionalSizes, optionalDtype);
}

Type ValueTensorType::parse(MLIRContext *context, DialectAsmParser &parser) {
  return parseTensorType(
      context, parser,
      [](MLIRContext *context, Optional<ArrayRef<int64_t>> optionalSizes,
         Type optionalType) {
        return ValueTensorType::get(context, optionalSizes, optionalType);
      });
}

void ValueTensorType::print(DialectAsmPrinter &printer) const {
  printer << "vtensor";
  printTensorType(printer, getOptionalSizes(), getOptionalDtype());
}
