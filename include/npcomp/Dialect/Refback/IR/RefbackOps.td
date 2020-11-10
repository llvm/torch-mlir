//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef REFBACK_OPS
#define REFBACK_OPS

include "npcomp/Dialect/Refback/IR/RefbackBase.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class Refback_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Refback_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Ops related to bufferization.
//===----------------------------------------------------------------------===//

def Refback_AllocMemRefOp : Refback_Op<"alloc_memref", []> {
  let summary = "Allocates a memref of the given shape.";
  let description = [{
    Allocates a memref of the given shape.

    This op is a convenience for creating a bunch of
    shape.get_extent + std.alloc ops.
  }];
  let arguments = (ins Shape_ExtentTensorType:$shape);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "$shape attr-dict `:`  type($memref)";
}

#endif // REFBACK_OPS
