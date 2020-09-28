//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_OPS
#define TORCH_OPS

include "npcomp/Dialect/Torch/IR/TorchBase.td"

class Torch_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Torch_Dialect, mnemonic, traits> {
}

def Torch_DummyOp : Torch_Op<"dummy"> {
  let summary = "Dummy placeholder op until more is defined";
}

#endif // TORCH_OPS
