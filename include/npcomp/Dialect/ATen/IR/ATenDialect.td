//===- ATenDialect.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_IR_ATEN_DIALECT
#define NPCOMP_DIALECT_ATEN_IR_ATEN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

/// The ATenDialect models 'A Tensor library' from Pytorch.  The intention
/// is to provide an abstraction which is isomorphic with datastructures
/// returned from the pytorch jit, enabling integration with Pytorch models.
/// Most of the actual operation definitions in tablegen are themselves
/// generated from C APIs exported by Pytorch.
def ATen_Dialect : Dialect {
  let name = "aten";
  let cppNamespace = "::mlir::NPCOMP::aten";
}

//===----------------------------------------------------------------------===//
// Dialect types
//===----------------------------------------------------------------------===//

def ATen_ListType : DialectType<ATen_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::aten::ATenListType>()">, "ATen List">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::aten::ATenListType()"> {
  let typeDescription = [{
    A variadic list of arguments in ATen.
  }];
}

#endif // NPCOMP_DIALECT_ATEN_IR_ATEN_DIALECT
