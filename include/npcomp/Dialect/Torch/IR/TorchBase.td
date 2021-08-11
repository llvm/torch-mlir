//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_BASE
#define TORCH_BASE

include "mlir/IR/OpBase.td"

def Torch_Dialect : Dialect {
  let name = "torch";
  let cppNamespace = "::mlir::NPCOMP::Torch";
  let description = [{
    Top-level dialect for interfacing PyTorch and MLIR.

    This dialect maintains a fairly isomorphic representation with TorchScript.

    This dialect also provides transforms that lower it to the
    "Torch backend contract", which is an IR form that we present to
    later conversions, such as conversion to the npcomp backend contract.
    The Torch backend contract significantly simplifies the IR representation
    and puts it in a form easier for later lowering to work on. Specifically:
    - The TorchScript object graph has been flattened to a list of globals (see
      the GlobalizeObjectGraph tranformation).
    - Most of the operations have been changed to operate on value-semantic
      tensors (see MaximizeValueSemantics)
    - The number of op variants have been reduced (see ReduceOpVariants)
    - Tensor sizes have been analyzed and static ranks inferred where possible
      and propagated throughout the program.
  }];

  let hasRegionArgAttrVerify = 1;
  let hasConstantMaterializer = 1;
}

class TorchOpTrait<string name> : OpTrait, NativeTrait<"", ""> {
  let trait = name;
  let cppNamespace = "::mlir::NPCOMP::Torch::OpTrait";
}

def HasValueSemantics : TorchOpTrait<"HasValueSemantics">;
def IsTrailingUnderscoreInplaceVariant
  : TorchOpTrait<"IsTrailingUnderscoreInplaceVariant">;

#endif // TORCH_BASE
