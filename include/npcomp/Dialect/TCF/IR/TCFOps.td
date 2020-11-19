//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TCF_OPS
#define TCF_OPS

include "npcomp/Dialect/TCF/IR/TCFBase.td"

class TCF_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<TCF_Dialect, mnemonic, traits> {
}

// TODO: investigate effects framework for defining error semantics
// TODO: define in a general way across the dialect what "encounters an error" means.

class BinaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
  TCF_Op<mnemonic, traits> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def TCF_AddOp : BinaryArithmeticOp<"add"> {
  let summary = "Addition of two tensors.";
  let description = [{
    Addition of two tensors.

    Numpy-style broadcasting is allowed.
  }];
}

def TCF_MaxOp : BinaryArithmeticOp<"max"> {
  let summary = "Maximum of two tensors.";
  let description = [{
    Maximum of two tensors.

    Numpy-style broadcasting is allowed.
  }];
}

def TCF_MulOp : BinaryArithmeticOp<"mul"> {
  let summary = "Multiply an input tensor by a scalar tensor.";
  let description = [{
    Multiplies each element of the input `input` with the scalar `other` and returns a new resulting tensor. The tensor types must match and shapes must be broadcastable.
  }];
}

class UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
  TCF_Op<mnemonic,
        !listconcat(traits, [AllTypesMatch<["operand", "result"]>])>,
  AllTypesMatch<["operand", "result"]> {
  let arguments = (ins AnyTensor:$operand);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def TCF_ExpOp : UnaryArithmeticOp<"exp"> {
  let summary = "base-e exponential";
  let description = [{
    See std.exp for more details.
  }];
}

def TCF_TanhOp : UnaryArithmeticOp<"tanh"> {
  let summary = "hyperbolic tangent";
  let description = [{
    See std.tanh for more details.
  }];
}

// TODO: Generalize this op appropriately and add more verification.
// For example, an unranked operand probably should be allowed and verified
// dynamically in TCF->TCP lowering if needed.
def TCF_MatmulOp : TCF_Op<"matmul"> {
  let summary = "Performs a matrix multiplication";
  let description = [{
    Performs a matrix multiplication.

    The tensors have dimensions:
    - lhs: [M, K]
    - rhs: [K, N]
    - result: [M, N]

    If the `K` dimension mismatches between the operands, this op aborts the
    program.
  }];
  let arguments = (ins 2DTensorOf<[F32]>:$lhs, 2DTensorOf<[F32]>:$rhs);
  let results = (outs 2DTensorOf<[F32]>:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def TCF_ConvNCHWBiasOp : TCF_Op<"conv_2d_nchw_bias"> {
  let summary = "2-D convolution with bias";
  let description = [{
    Performs 2-D convolution plus bias addition.

    The tensors have dimensions:
    - in:     [N, C, H, W]
    - filter: [F, C, KH, KW]
    - bias:   [N, F, H, W]
    - result: [N, F, H, W]

    If the `K` dimension mismatches between the operands, this op aborts the
    program.
  }];
  let arguments = (ins 4DTensorOf<[F32]>:$in, 4DTensorOf<[F32]>:$filter, 4DTensorOf<[F32]>:$bias);
  let results = (outs 4DTensorOf<[F32]>:$result);

  let assemblyFormat = "$in `,` $filter `,` $bias attr-dict `:` functional-type(operands, results)";
}

#endif // #ifndef TCF_OPS
