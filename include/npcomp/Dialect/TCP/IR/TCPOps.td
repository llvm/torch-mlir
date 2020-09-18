//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TCP_OPS
#define TCP_OPS

include "npcomp/Dialect/TCP/IR/TCPBase.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class TCP_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<TCP_Dialect, mnemonic, traits> {
}

// TODO: Clarify allowed tensor element types.
def TCP_AddOp
    : TCP_Op<"add", []> {
  let summary = "Adds two tensors.";
  let description = [{
Adds two tensors.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

// TODO: Generalize this op appropriately and add more verification.
// For example, should we have a single primitive that does multidimensional
// contractions? + batching as well in the same op? In fact, if we want to
// get really general, we can include convolution as well; matmul is the 1x1
// image and 1x1 kernel special case.
// It still lowers trivially into linalg.generic even with such generalization
// -- the main question is what transforms we want to do at the TCP level that
// would be affected by those design choices.
def TCP_MatmulOp : TCP_Op<"matmul"> {
  let summary = "Performs a matrix multiplication";
  let description = [{
    Performs a matrix multiplication.

    The tensors have dimensions:
    - lhs: [M, K]
    - rhs: [K, N]
    - result: [M, N]

    If the `K` dimension mismatches between operands, this op has
    undefined behavior.
  }];
  let arguments = (ins 2DTensorOf<[F32]>:$lhs, 2DTensorOf<[F32]>:$rhs);
  let results = (outs 2DTensorOf<[F32]>:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def TCP_BroadcastToOp : TCP_Op<"broadcast_to"> {
  let summary = "Broadcasts an operand to a given shape.";
  let description = [{
Broadcasts `operand` to the shape `shape`.

It is undefined behavior if such a broadcast is not legal.
  }];
  let arguments = (ins AnyRankedTensor:$operand, Shape_ExtentTensorType:$shape);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$operand `,` $shape attr-dict `:` functional-type(operands, results)";
}

def TCP_GlobalOp : TCP_Op<"global", [Symbol]> {
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
// Ops related to tensor->memref conversion.
//===----------------------------------------------------------------------===//
// TODO: These ops probably belong in a "TCP on memrefs" dialect analogous
// to `lmhlo`

// TODO: Use TypesMatchWith to verify this better.
def TCP_TensorToMemrefOp : TCP_Op<"tensor_to_memref", [NoSideEffect]> {
  let summary = "Converts a tensor to a memref";
  let description = [{
    This op is used to materialize conversions to allow incremental lowering of
    tensors to memrefs.
  }];
  let arguments = (ins AnyRankedTensor:$tensor);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "attr-dict $tensor `:` type($tensor) `->` type($memref)";
  let hasFolder = 1;
}

// TODO: Use TypesMatchWith to verify this better.
def TCP_MemrefToTensorOp : TCP_Op<"memref_to_tensor", [NoSideEffect]> {
  let summary = "Converts a memref to a tensor";
  let description = [{
    This op is used to materialize conversions to allow incremental lowering of
    tensors to memrefs.
  }];
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs AnyRankedTensor:$tensor);
  let assemblyFormat = "attr-dict $memref `:` type($memref) `->` type($tensor)";
}

def TCP_AllocMemRefOp : TCP_Op<"alloc_memref", []> {
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

def TCP_GetGlobalMemrefOp : TCP_Op<"get_global_memref"> {
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

def TCP_ShapedResultsOp : TCP_Op<"shaped_results", [
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
    %sin, %cos = tcp.shaped_results %xShape, %xShape {
      %sin, cos = "some.sincos"(%x)
          : tensor<?xf32> -> (tensor<?xf32>, tensor<?xf32>)
      tcp.yield %sin, %cos : tensor<?xf32>, tensor<?xf32>
    }
    ```
  }];
  let arguments = (ins
    Variadic<Shape_ExtentTensorType>:$resultShapes
  );
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<
        "OpBuilder &builder, OperationState &result, TypeRange resultTypes, "
        "ValueRange resultShapes">
  ];

  let printer = [{ return ::print$cppClass(p, *this); }];
  let verifier = [{ return ::verify$cppClass(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def TCP_YieldOp : TCP_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["ShapedResultsOp"]>]> {
  let summary = "Yield-like terminator for TCP dialect";
  let description = "See scf.yield";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // TCP_OPS
