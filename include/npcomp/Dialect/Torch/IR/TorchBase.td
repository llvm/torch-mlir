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
include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.td"

def Torch_Dialect : Dialect {
  let name = "torch";
  let cppNamespace = "::mlir::NPCOMP::Torch";
  let description = [{
    Top-level dialect for interfacing PyTorch and MLIR.

    This dialect maintains a fairly isomorphic representation with TorchScript.

    TODO: Add more detail here.
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
