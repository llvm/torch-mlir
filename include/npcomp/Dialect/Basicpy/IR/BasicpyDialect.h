//===- BasicPyDialect.h - Basic Python --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_IR_BASICPY_DIALECT_H
#define NPCOMP_DIALECT_BASICPY_IR_BASICPY_DIALECT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "npcomp/Dialect/Common.h"
#include "npcomp/Typing/CPA/Interfaces.h"

namespace mlir {
namespace NPCOMP {
namespace Basicpy {

namespace BasicpyTypes {
enum Kind {
  // Dialect types.
  BoolType = TypeRanges::Basicpy,
  BytesType,
  EllipsisType,
  NoneType,
  SlotObjectType,
  StrType,
  UnknownType,

  // Dialect attributes.
  SingletonAttr,
  LAST_BASICPY_TYPE = SingletonAttr,
};
} // namespace BasicpyTypes

namespace detail {
struct SlotObjectTypeStorage;
} // namespace detail

/// Python 'bool' type (can contain values True or False, corresponding to
/// i1 constants of 0 or 1).
class BoolType : public Type::TypeBase<BoolType, Type, TypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == BasicpyTypes::BoolType; }
  static BoolType get(MLIRContext *context) {
    return Base::get(context, BasicpyTypes::BoolType);
  }
};

/// The type of the Python `bytes` values.
class BytesType : public Type::TypeBase<BytesType, Type, TypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == BasicpyTypes::BytesType; }
  static BytesType get(MLIRContext *context) {
    return Base::get(context, BasicpyTypes::BytesType);
  }
};

/// The type of the Python `Ellipsis` value.
class EllipsisType : public Type::TypeBase<EllipsisType, Type, TypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) {
    return kind == BasicpyTypes::EllipsisType;
  }
  static EllipsisType get(MLIRContext *context) {
    return Base::get(context, BasicpyTypes::EllipsisType);
  }
};

/// The type of the Python `None` value.
class NoneType : public Type::TypeBase<NoneType, Type, TypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == BasicpyTypes::NoneType; }
  static NoneType get(MLIRContext *context) {
    return Base::get(context, BasicpyTypes::NoneType);
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

  // Shorthand to check whether the SlotObject is of a given className and
  // arity.
  bool isOfClassArity(StringRef className, unsigned arity) {
    return getClassName().getValue() == className && getSlotCount() == arity;
  }
};

/// The type of the Python `str` values.
class StrType : public Type::TypeBase<StrType, Type, TypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == BasicpyTypes::StrType; }
  static StrType get(MLIRContext *context) {
    return Base::get(context, BasicpyTypes::StrType);
  }
};

/// An unknown type that could be any supported python type.
class UnknownType
    : public Type::TypeBase<UnknownType, Type, TypeStorage,
                            Typing::CPA::TypeMapInterface::Trait> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) {
    return kind == BasicpyTypes::UnknownType;
  }
  static UnknownType get(MLIRContext *context) {
    return Base::get(context, BasicpyTypes::UnknownType);
  }

  Typing::CPA::TypeNode *mapToCPAType(Typing::CPA::Context &context);
};

#include "npcomp/Dialect/Basicpy/IR/BasicpyOpsDialect.h.inc"

} // namespace Basicpy
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_BASICPY_IR_BASICPY_DIALECT_H
