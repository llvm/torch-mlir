//===- BasicPyOps.td - Basic Python ops --------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_BASICPY_OPS
#define NPCOMP_DIALECT_BASICPY_BASICPY_OPS

include "BasicpyDialect.td"
include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/SymbolInterfaces.td"

def Basicpy_ExampleOp : Basicpy_Op<"example", []> {
  let summary = "Move along, nothing to see here.";
  let description = [{
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{
    attr-dict
  }];  
}


#endif // NPCOMP_DIALECT_BASICPY_BASICPY_OPS
