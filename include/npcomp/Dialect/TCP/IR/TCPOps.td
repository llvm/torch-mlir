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
class BinaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
  TCP_Op<mnemonic, traits> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def TCP_AddOp : BinaryArithmeticOp<"add"> {
  let summary = "Addition of two tensors";
  let description = [{
    Addition of two tensors.
  }];
}

def TCP_MaxOp : BinaryArithmeticOp<"max"> {
  let summary = "Maximum of two tensors";
  let description = [{
    Maximum of two tensors.
  }];
}

def TCP_MulOp : BinaryArithmeticOp<"mul"> {
  let summary = "Multiply an input tensor by a scalar tensor.";
  let description = [{
    Multiplies each element of the input `input` with the scalar `other` and returns a new resulting tensor. The tensor types must match and shapes must be broadcastable.
  }];
}

class UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
  TCP_Op<mnemonic,
        !listconcat(traits, [AllTypesMatch<["operand", "result"]>])>,
  AllTypesMatch<["operand", "result"]> {
  let arguments = (ins AnyTensor:$operand);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def TCP_ExpOp : UnaryArithmeticOp<"exp"> {
  let summary = "base-e exponential";
  let description = [{
    See std.exp for more details.
  }];
}

def TCP_TanhOp : UnaryArithmeticOp<"tanh"> {
  let summary = "hyperbolic tangent";
  let description = [{
    See std.tanh for more details.
  }];
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

#endif // TCP_OPS
