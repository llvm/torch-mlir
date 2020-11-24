//===- BasicPyDialect.td - Basic python dialect ------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_IR_BASICPY_DIALECT
#define NPCOMP_DIALECT_BASICPY_IR_BASICPY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Basicpy_Dialect : Dialect {
  let name = "basicpy";
  let summary = "Basic Python dialect";
  let description = [{
    Core types and ops
  }];
  let cppNamespace = "::mlir::NPCOMP::Basicpy";
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Op templates
//===----------------------------------------------------------------------===//

class Basicpy_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Basicpy_Dialect, mnemonic, traits> {
  let parser = [{ return ::parse$cppClass(parser, &result); }];
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Dialect types
//===----------------------------------------------------------------------===//

def Basicpy_BoolType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::BoolType>()">, "Bool type">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Basicpy::BoolType>()"> {
  let typeDescription = [{
    Type for 'True' and 'False' values.
  }];
}

def Basicpy_BytesType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::BytesType>()">, "Bytes type">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Basicpy::BytesType>()"> {
  let typeDescription = [{
    Represents Python 'bytes' values.
  }];
}

def Basicpy_EllipsisType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::EllipsisType>()">, "Ellipsis type">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Basicpy::EllipsisType>()"> {
  let typeDescription = [{
    Type of the Python 'Ellipsis' value.
  }];
}

def Basicpy_NoneType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::NoneType>()">, "None type">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Basicpy::NoneType>()"> {
  let typeDescription = [{
    Type of the Python 'None' value.
  }];
}

def Basicpy_SlotObjectType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::SlotObjectType>()">,
    "Slot object"> {
  let typeDescription = [{
    Type for built-in objects which have a fixed number of slots and a type
    name in the system catalog of types. In some ways, this resembles a
    namedtuple, but it is used for built-in custom objects.
  }];
}

def Basicpy_StrType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::StrType>()">,"String type">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Basicpy::StrType>()"> {
  let typeDescription = [{
    Represents values of the python 'str' type.
  }];
}

def Basicpy_UnknownType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::UnknownType>()">,
    "Unknown type"> {
  let typeDescription = [{
    An unknown type (for the current phase of processing).
  }];
}

def Basicpy_ListType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::ListType>()">,
    "List type"> {
  let typeDescription = [{
    A Python list type. In the non-parameterized case, there are limited
    constraints on the element type or length; however, it can be refined to
    include such constraints.

    As in Python, this list type represents a mutable, reference counted
    object in a corresponding runtime layer.
  }];
}

def Basicpy_TupleType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::TupleType>()">,
    "Tuple type"> {
  let typeDescription = [{
    A Python tuple type. In the non-parameterized case, there are limited
    constraints on the element type or length; however, it can be refined to
    include such constraints.

    As in Python, post-construction tuple's are immutable, reference counted
    objects in a corresponding runtime layer. However, since they are
    immutable, they can also serve as value-typed entities if their elements
    are immutable.
  }];
}

def Basicpy_DictType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::DictType>()">,
    "Dict type"> {
  let typeDescription = [{
    A Python dict type. In the non-parameterized case, there are limited
    constraints on the key or value types; however, it can be refined to
    include such constraints.

    As in Python, this list type represents a mutable, reference counted
    object in a corresponding runtime layer.
  }];
}

//===----------------------------------------------------------------------===//
// Type/attribute predicates
//===----------------------------------------------------------------------===//

def Basicpy_SingletonType : AnyTypeOf<[
  Basicpy_NoneType,
  Basicpy_EllipsisType
]>;

// A predicate to determine whether a Type is a SlotObject of a given
// className and arity. Does no checking of slot types.
class Basicpy_SlotObjectOfClassArity<string className, int arity> :
  And<[
    Basicpy_SlotObjectType.predicate,
    CPred<
      "$_self.cast<::mlir::NPCOMP::Basicpy::SlotObjectType>().isOfClassArity(\""
      # className # "\", " # arity # ")">
  ]>;

// Type representing a 'slice' object, which mirrors the Python built-in
// slice class.
def Basicpy_SliceSlotObjectType :
  Type<Basicpy_SlotObjectOfClassArity<"slice", 3>>;

#endif // NPCOMP_DIALECT_BASICPY_IR_BASICPY_DIALECT
