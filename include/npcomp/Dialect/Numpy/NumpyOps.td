//===- NumpyOps.td - Core numpy dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_NUMPY_OPS
#define NPCOMP_DIALECT_NUMPY_NUMPY_OPS

include "NumpyDialect.td"
include "mlir/Interfaces/SideEffects.td"

def Numpy_FooOp : Numpy_Op<"foo", [NoSideEffect,
                                   SameOperandsAndResultType]> {
  let summary = "Temp op";
  let description = [{
    Temp op
  }];

  let arguments = (ins I32:$input);
  let results = (outs I32:$res);

  let assemblyFormat = [{
      $input attr-dict `:` type($input)
  }];
}

#endif // NPCOMP_DIALECT_NUMPY_NUMPY_OPS
