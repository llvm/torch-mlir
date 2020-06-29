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

using namespace mlir;
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
    //   ndarray<?>
    //   ndarray<i32>
    Type dtype = Basicpy::UnknownType::get(getContext());
    if (parser.parseLess())
      return Type();
    if (failed(parser.parseOptionalQuestion())) {
      // Specified dtype.
      if (parser.parseType(dtype))
        return Type();
    }
    if (parser.parseGreater())
      return Type();
    return NdArrayType::get(dtype);
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
    auto dtype = ndarray.getDtype();
    os << "ndarray<";
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
  using KeyTy = Type;
  NdArrayTypeStorage(Type optionalDtype) : optionalDtype(optionalDtype) {}
  bool operator==(const KeyTy &other) const { return optionalDtype == other; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }
  static NdArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<NdArrayTypeStorage>())
        NdArrayTypeStorage(key);
  }

  Type optionalDtype;
};

} // namespace detail
} // namespace Numpy
} // namespace NPCOMP
} // namespace mlir

NdArrayType NdArrayType::get(Type dtype) {
  assert(dtype && "dtype cannot be null");
  return Base::get(dtype.getContext(), NumpyTypes::NdArray, dtype);
}

Type NdArrayType::getDtype() { return getImpl()->optionalDtype; }
