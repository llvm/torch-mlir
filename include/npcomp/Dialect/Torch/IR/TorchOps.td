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

def Torch_KernelCall : Torch_Op<"kernel_call"> {
  let summary = "Calls a Torch custom kernel";
  let description = [{
    Torch kernel calls are matched by the runtime based on signature, including
    the fully qualified kernel name (i.e. "namespace::name") and the tuple of
    argument types. This op models such an invocation.

    Caveat: When interacting with the PyTorch boxed interpreter stack, the
    arguments follow a convention that the top of stack is the first argument.
  }];

  let arguments = (ins
    StrAttr:$kernel_name,
    Variadic<AnyTorchType>:$args
  );
  let results = (outs
    Variadic<AnyTorchType>:$results
  );

  let assemblyFormat = [{
    $kernel_name $args attr-dict `:` functional-type($args, results)
  }];
}

#endif // TORCH_OPS
