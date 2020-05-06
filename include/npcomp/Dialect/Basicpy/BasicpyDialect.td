//===- BasicPyDialect.td - Basic python dialect ------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT
#define NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT

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
  let cppNamespace = "Basicpy";
}

//===----------------------------------------------------------------------===//
// Op templates
//===----------------------------------------------------------------------===//

class Basicpy_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Basicpy_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];  
}

//===----------------------------------------------------------------------===//
// Dialect types
//===----------------------------------------------------------------------===//

def Basicpy_SlotObjectType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::SlotObjectType>()">, 
    "Slot object"> {
  let typeDescription = [{
    Type for built-in objects which have a fixed number of slots and a type
    name in the system catalog of types. In some ways, this resembles a
    namedtuple, but it is used for built-in custom objects.
  }];
}

def Basicpy_NoneType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::NoneType>()">, "None type">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Basicpy::NoneType>()"> {
  let typeDescription = [{
    Type of the Python 'None' value.
  }];
}

def Basicpy_EllipsisType : DialectType<Basicpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Basicpy::EllipsisType>()">, "Ellipsis type">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Basicpy::EllipsisType>()"> {
  let typeDescription = [{
    Type of the Python 'Ellipsis' value.
  }];
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

def Basicpy_SingletonType : AnyTypeOf<[
  Basicpy_NoneType,
  Basicpy_EllipsisType
]>;

#endif // NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT
