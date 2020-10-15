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

def Refback_GlobalOp : Refback_Op<"global", [Symbol]> {
  let summary = "Represents a global variable";
  let description = [{
    Represents a global variable.

    Currently, only constant tensors are supported, and they are not
    considered to be exported.
  }];
  let arguments = (ins StrAttr:$sym_name, ElementsAttr:$value);
  let results = (outs);

  let printer = [{ return ::print$cppClass(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
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

def Refback_GetGlobalMemrefOp : Refback_Op<"get_global_memref"> {
  let summary = "Obtain a memref pointing at the given global";
  let description = [{
    Obtain a memref pointing at the given global.
  }];
  let arguments = (ins FlatSymbolRefAttr:$global);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "$global attr-dict `:` type($memref)";
  let verifier = "return ::verify$cppClass(*this);";
}

#endif // REFBACK_OPS
