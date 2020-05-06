//===- BasicPyDialect.h - Basic Python --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT_H
#define NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "npcomp/Dialect/Common.h"

namespace mlir {
namespace NPCOMP {
namespace Basicpy {

namespace BasicpyTypes {
enum Kind {
  // Dialect types.
  NoneType = TypeRanges::Basicpy,
  EllipsisType,
  SlotObjectType,

  // Dialect attributes.
  SingletonAttr,
  LAST_BASICPY_TYPE = SingletonAttr,
};
} // namespace BasicpyTypes

namespace detail {
struct SlotObjectTypeStorage;
} // namespace detail

/// The type of the Python `None` value.
class NoneType : public Type::TypeBase<NoneType, Type> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == BasicpyTypes::NoneType; }
  static NoneType get(MLIRContext *context) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type.
    return Base::get(context, BasicpyTypes::NoneType);
  }
};

/// The type of the Python `Ellipsis` value.
class EllipsisType : public Type::TypeBase<EllipsisType, Type> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) {
    return kind == BasicpyTypes::EllipsisType;
  }
  static EllipsisType get(MLIRContext *context) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type.
    return Base::get(context, BasicpyTypes::EllipsisType);
  }
};

class SlotObjectType : public Type::TypeBase<SlotObjectType, Type,
                                             detail::SlotObjectTypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) {
    return kind == BasicpyTypes::SlotObjectType;
  }
  static SlotObjectType get(StringAttr className, ArrayRef<Type> slotTypes);
  StringAttr getClassName();
  unsigned getSlotCount();
  ArrayRef<Type> getSlotTypes();
};

#include "npcomp/Dialect/Basicpy/BasicpyOpsDialect.h.inc"

} // namespace Basicpy
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT_H
