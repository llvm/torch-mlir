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

// TODO: verify same dtype?
// TODO: what are the allowable dtypes?
def TCF_AddOp : TCF_Op<"add"> {
  let summary = "Add two tensors.";
  let description = [{
Add two tensors.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
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

#endif // #ifndef TCF_OPS
