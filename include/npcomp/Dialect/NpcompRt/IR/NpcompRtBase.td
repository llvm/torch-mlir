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

def NpcompRt_Dialect : Dialect {
  let name = "npcomp_rt";
  let cppNamespace = "::mlir::NPCOMP::npcomp_rt";
  let description = [{
The `npcomp_rt` dialect is the IR manifestation for interaction with the
npcomp runtime. It primarily serves as a layer that enapsulates the data
structures and functions available in the runtime, and faciliates
conversion to those conventions, such as by providing utilities for being
lowered to the llvm dialect.
  }];
}

def NpcompRt_BufferView
    : DialectType<
          NpcompRt_Dialect,
          CPred<"$_self.isa<::mlir::NPCOMP::npcomp_rt::BufferViewType>()">,
          "buffer view">,
      BuildableType<
          "$_builder.getType<::mlir::NPCOMP::npcomp_rt::BufferViewType>()"> {
  let typeDescription = [{The runtime type that represents a buffer.}];
}

#endif // #ifndef NPCOMPRT_BASE
