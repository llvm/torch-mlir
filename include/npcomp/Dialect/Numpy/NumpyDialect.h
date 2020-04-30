//===- NumpyDialect.h - Core numpy dialect ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT_H
#define NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace NPCOMP {
namespace Numpy {

namespace NumpyTypes {
enum Kind {
  AnyDtypeType = Type::FIRST_PRIVATE_EXPERIMENTAL_9_TYPE,
  LAST_NUMPY_TYPE = AnyDtypeType
};
} // namespace NumpyTypes

// The singleton type representing an unknown dtype.
class AnyDtypeType : public Type::TypeBase<AnyDtypeType, Type> {
public:
  using Base::Base;

  static AnyDtypeType get(MLIRContext *context) {
    return Base::get(context, NumpyTypes::Kind::AnyDtypeType);
  }

  static bool kindof(unsigned kind) {
    return kind == NumpyTypes::Kind::AnyDtypeType;
  }
};

#include "npcomp/Dialect/Numpy/NumpyOpsDialect.h.inc"

} // namespace Numpy
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT_H
