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
    tensor.extract ops + std.alloc.
  }];
  let arguments = (ins Shape_ExtentTensorType:$shape);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "$shape attr-dict `:`  type($memref)";
}

//===----------------------------------------------------------------------===//
// Ops related to Torch.
//===----------------------------------------------------------------------===//

def Refback_TorchFallbackOp : Refback_Op<"torch_fallback",
                           [SingleBlockImplicitTerminator<"TorchFallbackYieldOp">,
                            DeclareOpInterfaceMethods<RegionBranchOpInterface>,
                            RecursiveSideEffects]> {
  let summary = "Execute region through the Torch fallback.";
  let description = [{
    Execute region through the Torch fallback.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let regions = (region SizedRegion<1>:$doRegion);
  let results = (outs Variadic<AnyType>:$results);
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let hasCanonicalizer = 1;
}

def Refback_TorchFallbackYieldOp : Refback_Op<"torch_fallback_yield",
       [NoSideEffect, ReturnLike, Terminator, HasParent<"TorchFallbackOp">]> {
  let summary = "Yield operation";
  let description = [{
    This yield operation represents a return operation within the
    `refback.torch_fallback` operation region. The operation takes variable number of
    operands and produces no results. The operand number and types must match
    the number and types of parent `refback.torch_fallback` results.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // REFBACK_OPS
