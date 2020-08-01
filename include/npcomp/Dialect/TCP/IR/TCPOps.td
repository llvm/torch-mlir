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
include "mlir/IR/SymbolInterfaces.td"

class TCP_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<TCP_Dialect, mnemonic, traits> {
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

//===----------------------------------------------------------------------===//
// Ops that need to be factored to a proper home.
//===----------------------------------------------------------------------===//
// TODO: Find a home for these.

// TODO: This probably doesn't belong in the tcp dialect.
def TCP_AllocMemRefOp : TCP_Op<"alloc_memref", []> {
  let summary = "Allocates a memref of the given shape.";
  let description = [{
Allocates a memref of the given shape.
  }];
  let arguments = (ins Shape_ShapeType:$shape);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "$shape attr-dict `:`  type($memref)";
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

// TODO: Change to a more principled error handling mechanism.
// This op probably doesn't need to exist eventually.
// This op is also not correctly modeled right now, since it itself doesn't
// produce the error in practice. The ops like shape.broadcast itself, when
// lowered, immediately produce errors.
// TODO: This should eventually be moved to a shape dialect.
def TCP_ShapeObserveErrorOp : TCP_Op<"shape_observe_error", []> {
  let summary = "Observes the fact that a shape might be an error.";
  let description = [{
    This op is a structural placeholder that captures a shape such that it
    is not erased. This will keep around shape computations that are later
    lowered into eager error handling code.

    The interaction of this op, especially with control flow and side
    effecting ops, is not very well-defined, and needs to be worked
    on/redesigned.
  }];
  let arguments = (ins Shape_ShapeType:$shape);
  // TODO: ODS seems to create redeclared class members if we remove this,
  // resulting in C++ compilation errors.
  let results = (outs NoneType:$dummy);
}

// TODO: This probably belongs in the shape dialect.
def TCP_GetExtentOp : TCP_Op<"get_extent", [NoSideEffect]> {
  let summary = "Gets the specified extent from a shape.";
  let description = [{
Gets the specified extent from a shape.

This op has undefined behavior if the shape is an error.
  }];
  let arguments = (ins Shape_ShapeType:$shape, I64Attr:$dim);
  let results = (outs Index:$extent);
  let assemblyFormat = "$shape `,` $dim attr-dict";

  let builders = [
    // Helper to pass a simple integer instead of an integer attr.
    OpBuilder<
      [{
        OpBuilder &builder, OperationState &result,
        Value shape, int64_t dim
      }],
      [{
        build(builder, result, shape, builder.getI64IntegerAttr(dim));
      }]
    >
  ];
}

#endif // TCP_OPS
