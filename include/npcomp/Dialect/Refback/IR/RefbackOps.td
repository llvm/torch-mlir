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

//===----------------------------------------------------------------------===//
// Ops related to shapes.
//===----------------------------------------------------------------------===//
// TODO: These belong in a shape-related dialect.

def Refback_ShapedResultsOp : Refback_Op<"shaped_results", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">,
  RecursiveSideEffects,
  NoRegionArguments
]> {
  let summary = "Result shape annotation";
  let description = [{
    Represents a computation whose outputs have a precomputed shape.
    The i-th result has the shape described by the i-th operand.

    This op is not isolated from above, so if the region needs any inputs,
    they can simply be captured. Hence, this op is a
    "this tensor has this shape" annotation with a slightly different set of
    tradeoffs than the so-called "tie shape" kinds of operations.
    In particular, this region-based formulation has the opportunity to
    capture structural invariants.

    Example:
    ```mlir
    // sincos is an elementwise operation, so it doesn't change the shape.
    %x = ...
    %xShape = ...
    %sin, %cos = refback.shaped_results %xShape, %xShape {
      %sin, cos = "some.sincos"(%x)
          : tensor<?xf32> -> (tensor<?xf32>, tensor<?xf32>)
      refback.yield %sin, %cos : tensor<?xf32>, tensor<?xf32>
    }
    ```
  }];
  let arguments = (ins
    Variadic<Shape_ExtentTensorType>:$resultShapes
  );
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let printer = [{ return ::print$cppClass(p, *this); }];
  let verifier = [{ return ::verify$cppClass(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def Refback_YieldOp : Refback_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["ShapedResultsOp"]>]> {
  let summary = "Yield-like terminator for Refback dialect";
  let description = "See scf.yield";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // REFBACK_OPS
