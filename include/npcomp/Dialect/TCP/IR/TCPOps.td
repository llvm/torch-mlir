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
include "mlir/Interfaces/SideEffects.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class TCP_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<TCP_Dialect, mnemonic, traits> {
}

// TODO: Do islands belong inside this dialect?
// It almost feels like they should be in an `error` dialect (containing
// the witness type as well).
// There would be no "shape.abort_if_error" because the aborting happens
// inside the witness ops, with the island operating as a witness sink.
def TCP_IslandOp : TCP_Op<"island"> {
  let summary = "Island of no-side-effect ops.";
  let description = [{
Most ops in the `tcp` dialect have complex preconditions on their tensor 
arguments (usually their shapes) so that they can be a good starting point
for code generation compilation flows.
We want an efficient way to understand which ops are related to which
preconditions without cluttering the TCP ops themselves.
To do this, we have this `tcp.island` op which takes as operands
witness values establishing use-def edges between precondition assertions
and this island op, and then restrict most other `tcp` ops to reside inside
these islands. This makes code motion to rearrange `tcp` ops simpler
by having the witness use-def edges, without needing for every `tcp` op
to have extra operands.
// TODO: Still unclear if this is really that useful. This mainly affects the
// ability to hoist tcp ops. It should always be safe to sink TCP ops.
  }];
  let arguments = (ins Variadic<NoneType>:$witnesses);
  let regions = (region AnyRegion:$body);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  // TODO: verify return types match internal tcp.yield return ValueRange's.
}

def TCP_YieldOp
    : TCP_Op<"yield", [NoSideEffect, HasParent<"IslandOp">, Terminator]> {
  let summary = "yield-like terminator for tcp.island op.";
  let description = [{
Returns control and a variadic list of values to the parent tcp.island op.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$operands);
  let results = (outs);
}

// TODO: clarify allowed tensor element types.
// TODO: HasParent is too restrictive? can't have an island with loop.for with
// further ops inside it?
def TCP_AddOp
    : TCP_Op<"add", []> {
  let summary = "Adds two tensors.";
  let description = [{
Adds two tensors.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
}

def TCP_BroadcastToOp : TCP_Op<"broadcast_to"> {
  let summary = "Broadcasts an operand to a given shape.";
  let description = [{
Broadcasts `operand` to the shape `shape`.

It is undefined behavior if such a broadcast is not legal.
  }];
  let arguments = (ins AnyRankedTensor:$operand, Shape_ShapeType:$shape);
  let results = (outs AnyRankedTensor:$result);
}

// TODO: This probably doesn't belong in the tcp dialect.
def TCP_AllocMemRefOp : TCP_Op<"alloc_memref", []> {
  let summary = "Allocates a memref of the given shape.";
  let description = [{
Allocates a memref of the given shape.
  }];
  let arguments = (ins Shape_ShapeType:$shape);
  let results = (outs AnyMemRef:$memref);
}

// TODO: Change to a more principled witness-based error handling mechanism.
// This op probably doesn't need to exist eventually.
def TCP_AbortIfErrorOp : TCP_Op<"abort_if_error",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Aborts the program if the argument is an error shape.";
  let description = [{
Aborts the program if its `shape` argument is an error shape.
TODO: can we do something better designed here then just abort?

Returns `none`, which can be used as a witness value to establish a use-def
relationship between this op and an op that requires the precondition
established by this op.
  }];
  let arguments = (ins Shape_ShapeType:$shape);
  let results = (outs NoneType:$result);
}

// TODO: This probably belongs in the shape dialect.
def TCP_GetExtentOp : TCP_Op<"get_extent",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Gets the specified extent from a shape.";
  let description = [{
Gets the specified extent from a shape.

This op has undefined behavior if the shape is an error.
  }];
  let arguments = (ins Shape_ShapeType:$shape, I64Attr:$dim);
  let results = (outs Index:$extent);
}

#endif // TCP_OPS
