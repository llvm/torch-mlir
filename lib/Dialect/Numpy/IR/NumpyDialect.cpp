//===- NumpyDialect.cpp - Core numpy dialect --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Typing/Support/CPAIrHelpers.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Numpy;

NumpyDialect::NumpyDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Numpy/IR/NumpyOps.cpp.inc"
      >();
  addTypes<AnyDtypeType, NdArrayType>();
}

Type NumpyDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "any_dtype")
    return AnyDtypeType::get(getContext());
  if (keyword == "ndarray") {
    // Parse:
    //   ndarray<*:?>
    //   ndarray<*:i32>
    //   ndarary<[1,2,3]:i32>
    // Note that this is a different syntax than the built-ins as the dialect
    // parser is not general enough to parse a dimension list with an optional
    // element type (?). The built-in form is also remarkably ambiguous when
    // considering extending it.
    Type dtype = Basicpy::UnknownType::get(getContext());
    bool hasShape = false;
    llvm::SmallVector<int64_t, 4> shape;
    if (parser.parseLess())
      return Type();
    if (succeeded(parser.parseOptionalStar())) {
      // Unranked.
    } else {
      // Parse dimension list.
      hasShape = true;
      if (parser.parseLSquare())
        return Type();
      for (bool first = true;; first = false) {
        if (!first) {
          if (failed(parser.parseOptionalComma())) {
            break;
          }
        }
        if (succeeded(parser.parseOptionalQuestion())) {
          shape.push_back(-1);
          continue;
        }

        int64_t dim;
        auto optionalPr = parser.parseOptionalInteger(dim);
        if (optionalPr.hasValue()) {
          if (failed(*optionalPr))
            return Type();
          shape.push_back(dim);
          continue;
        }
        break;
      }
      if (parser.parseRSquare()) {
        return Type();
      }
    }

    // Parse colon dtype.
    if (parser.parseColon()) {
      return Type();
    }

    if (failed(parser.parseOptionalQuestion())) {
      // Specified dtype.
      if (parser.parseType(dtype)) {
        return Type();
      }
    }
    if (parser.parseGreater()) {
      return Type();
    }

    llvm::Optional<ArrayRef<int64_t>> optionalShape;
    if (hasShape)
      optionalShape = shape;
    auto ndarray = NdArrayType::get(dtype, optionalShape);
    return ndarray;
  }

  parser.emitError(parser.getNameLoc(), "unknown numpy type: ") << keyword;
  return Type();
}

void NumpyDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  case NumpyTypes::AnyDtypeType:
    os << "any_dtype";
    return;
  case NumpyTypes::NdArray: {
    auto unknownType = Basicpy::UnknownType::get(getContext());
    auto ndarray = type.cast<NdArrayType>();
    auto shape = ndarray.getOptionalShape();
    auto dtype = ndarray.getDtype();
    os << "ndarray<";
    if (!shape) {
      os << "*:";
    } else {
      os << "[";
      for (auto it : llvm::enumerate(*shape)) {
        if (it.index() > 0)
          os << ",";
        if (it.value() < 0)
          os << "?";
        else
          os << it.value();
      }
      os << "]:";
    }
    if (dtype != unknownType)
      os.printType(dtype);
    else
      os << "?";
    os << ">";
    return;
  }
  default:
    llvm_unreachable("unexpected 'numpy' type kind");
  }
}

//----------------------------------------------------------------------------//
// Type and attribute detail
//----------------------------------------------------------------------------//
namespace mlir {
namespace NPCOMP {
namespace Numpy {
namespace detail {

struct NdArrayTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, llvm::Optional<ArrayRef<int64_t>>>;
  NdArrayTypeStorage(Type dtype, int rank, const int64_t *shapeElements)
      : dtype(dtype), rank(rank), shapeElements(shapeElements) {}
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dtype, getOptionalShape());
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.second) {
      return llvm::hash_combine(key.first, *key.second);
    } else {
      return llvm::hash_combine(key.first, -1);
    }
  }
  static NdArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    int rank = -1;
    const int64_t *shapeElements = nullptr;
    if (key.second.hasValue()) {
      auto allocElements = allocator.copyInto(*key.second);
      rank = key.second->size();
      shapeElements = allocElements.data();
    }
    return new (allocator.allocate<NdArrayTypeStorage>())
        NdArrayTypeStorage(key.first, rank, shapeElements);
  }

  llvm::Optional<ArrayRef<int64_t>> getOptionalShape() const {
    if (rank < 0)
      return llvm::None;
    return ArrayRef<int64_t>(shapeElements, rank);
  }

  Type dtype;
  int rank;
  const int64_t *shapeElements;
};

} // namespace detail
} // namespace Numpy
} // namespace NPCOMP
} // namespace mlir

NdArrayType NdArrayType::get(Type dtype,
                             llvm::Optional<ArrayRef<int64_t>> shape) {
  assert(dtype && "dtype cannot be null");
  return Base::get(dtype.getContext(), NumpyTypes::NdArray, dtype, shape);
}

bool NdArrayType::hasKnownDtype() {
  return getDtype() != Basicpy::UnknownType::get(getContext());
}

Type NdArrayType::getDtype() { return getImpl()->dtype; }

llvm::Optional<ArrayRef<int64_t>> NdArrayType::getOptionalShape() {
  return getImpl()->getOptionalShape();
}

Typing::CPA::TypeNode *
NdArrayType::mapToCPAType(Typing::CPA::Context &context) {
  llvm::Optional<Typing::CPA::TypeNode *> dtype;
  if (hasKnownDtype()) {
    // TODO: This should be using a general mechanism for resolving the dtype,
    // but we don't have that yet, and for NdArray, these must be primitives
    // anyway.
    dtype = context.getIRValueType(getDtype());
  }
  // Safe to capture an ArrayRef backed by type storage since it is uniqued.
  auto optionalShape = getOptionalShape();
  auto irCtor = [optionalShape](Typing::CPA::ObjectValueType *ovt,
                                llvm::ArrayRef<mlir::Type> fieldTypes,
                                MLIRContext *mlirContext,
                                llvm::Optional<Location>) {
    assert(fieldTypes.size() == 1);
    return NdArrayType::get(fieldTypes.front(), optionalShape);
  };
  return Typing::CPA::newArrayType(context, irCtor,
                                   context.getIdentifier("!NdArray"), dtype);
}
