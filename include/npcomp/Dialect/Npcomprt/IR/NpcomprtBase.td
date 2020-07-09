//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMPRT_BASE
#define NPCOMPRT_BASE

include "mlir/IR/OpBase.td"

def Npcomprt_Dialect : Dialect {
  let name = "npcomprt";
  let cppNamespace = "::mlir::NPCOMP::npcomprt";
  let description = [{
The `npcomprt` dialect is the IR manifestation for interaction with the
npcomp runtime. It primarily serves as a layer that enapsulates the data
structures and functions available in the runtime, and faciliates
conversion to those conventions, such as by providing utilities for being
lowered to the llvm dialect.
  }];
}

def Npcomprt_Tensor
    : DialectType<
          Npcomprt_Dialect,
          CPred<"$_self.isa<::mlir::NPCOMP::npcomprt::TensorType>()">,
          "buffer view">,
      BuildableType<
          "$_builder.getType<::mlir::NPCOMP::npcomprt::TensorType>()"> {
  let typeDescription = [{The runtime type that represents a buffer.}];
}

#endif // #ifndef NPCOMPRT_BASE
