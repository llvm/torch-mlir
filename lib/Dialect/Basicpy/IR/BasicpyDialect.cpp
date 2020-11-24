//===- BasicpyDialect.cpp - Basic python dialect ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Basicpy;

void BasicpyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.cpp.inc"
      >();
  addTypes<BoolType, BytesType, DictType, EllipsisType, ListType, NoneType,
           SlotObjectType, StrType, TupleType, UnknownType>();

  // TODO: Make real ops for everything we need.
  allowUnknownOperations();
}

Operation *BasicpyDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  // NumericConstantOp.
  // Supports IntegerType (any signedness), FloatType and ComplexType.
  if (type.isa<IntegerType>() || type.isa<FloatType>() ||
      type.isa<ComplexType>())
    return builder.create<NumericConstantOp>(loc, type, value);

  // Bool (i1 -> !basicpy.BoolType).
  if (type.isa<Basicpy::BoolType>()) {
    auto i1Value = value.dyn_cast<IntegerAttr>();
    if (i1Value && i1Value.getType().getIntOrFloatBitWidth() == 1)
      return builder.create<BoolConstantOp>(loc, type, i1Value);
  }

  // Bytes.
  if (type.isa<Basicpy::BytesType>()) {
    if (auto strValue = value.dyn_cast<StringAttr>())
      return builder.create<BytesConstantOp>(loc, type, strValue);
  }

  // Str.
  if (type.isa<Basicpy::StrType>()) {
    if (auto strValue = value.dyn_cast<StringAttr>())
      return builder.create<StrConstantOp>(loc, type, strValue);
  }

  return nullptr;
}

Type BasicpyDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "BoolType")
    return BoolType::get(getContext());
  if (keyword == "BytesType")
    return BytesType::get(getContext());
  if (keyword == "DictType")
    return DictType::get(getContext());
  if (keyword == "EllipsisType")
    return EllipsisType::get(getContext());
  if (keyword == "ListType")
    return ListType::get(getContext());
  if (keyword == "NoneType")
    return NoneType::get(getContext());
  if (keyword == "SlotObject") {
    StringRef className;
    if (parser.parseLess() || parser.parseKeyword(&className)) {
      return Type();
    }

    llvm::SmallVector<Type, 4> slotTypes;
    while (succeeded(parser.parseOptionalComma())) {
      Type slotType;
      if (parser.parseType(slotType))
        return Type();
      slotTypes.push_back(slotType);
    }
    if (parser.parseGreater())
      return Type();
    return SlotObjectType::get(StringAttr::get(className, getContext()),
                               slotTypes);
  }
  if (keyword == "StrType")
    return StrType::get(getContext());
  if (keyword == "TupleType")
    return TupleType::get(getContext());
  if (keyword == "UnknownType")
    return UnknownType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown basicpy type");
  return Type();
}

void BasicpyDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<BoolType>([&](Type) { os << "BoolType"; })
      .Case<BytesType>([&](Type) { os << "BytesType"; })
      .Case<DictType>([&](Type) { os << "DictType"; })
      .Case<EllipsisType>([&](Type) { os << "EllipsisType"; })
      .Case<ListType>([&](Type) { os << "ListType"; })
      .Case<NoneType>([&](Type) { os << "NoneType"; })
      .Case<SlotObjectType>([&](SlotObjectType slotObject) {
        auto slotTypes = slotObject.getSlotTypes();
        os << "SlotObject<" << slotObject.getClassName().getValue();
        if (!slotTypes.empty()) {
          os << ", ";
          llvm::interleaveComma(slotTypes, os,
                                [&](Type t) { os.printType(t); });
        }
        os << ">";
      })
      .Case<StrType>([&](Type) { os << "StrType"; })
      .Case<TupleType>([&](Type) { os << "TupleType"; })
      .Case<UnknownType>([&](Type) { os << "UnknownType"; })
      .Default(
          [&](Type) { llvm_unreachable("unexpected 'basicpy' type kind"); });
}

//----------------------------------------------------------------------------//
// Type and attribute detail
//----------------------------------------------------------------------------//
namespace mlir {
namespace NPCOMP {
namespace Basicpy {
namespace detail {

struct SlotObjectTypeStorage : public TypeStorage {
  using KeyTy = std::pair<StringAttr, ArrayRef<Type>>;
  SlotObjectTypeStorage(StringAttr className, ArrayRef<Type> slotTypes)
      : className(className), slotTypes(slotTypes) {}
  bool operator==(const KeyTy &other) const {
    return className == other.first && slotTypes == other.second;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }
  static SlotObjectTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    ArrayRef<Type> slotTypes = allocator.copyInto(key.second);
    return new (allocator.allocate<SlotObjectTypeStorage>())
        SlotObjectTypeStorage(key.first, slotTypes);
  }

  StringAttr className;
  ArrayRef<Type> slotTypes;
};
} // namespace detail
} // namespace Basicpy
} // namespace NPCOMP
} // namespace mlir

StringAttr SlotObjectType::getClassName() { return getImpl()->className; }
ArrayRef<Type> SlotObjectType::getSlotTypes() { return getImpl()->slotTypes; }
unsigned SlotObjectType::getSlotCount() { return getImpl()->slotTypes.size(); }

SlotObjectType SlotObjectType::get(StringAttr className,
                                   ArrayRef<Type> slotTypes) {
  return Base::get(className.getContext(), className, slotTypes);
}

//----------------------------------------------------------------------------//
// CPA Interface Implementations
//----------------------------------------------------------------------------//

Typing::CPA::TypeNode *
UnknownType::mapToCPAType(Typing::CPA::Context &context) {
  return context.newTypeVar();
}
