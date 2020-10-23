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
include "npcomp/Dialect/Torch/IR/OpInterfaces.td"

class Torch_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Torch_Dialect, mnemonic, traits> {
}

// TODO: Add alias mapping from the signature and use it to implement the
// effects interface (since whether the kernel_call has side effects is
// dependent on its metadata).
def Torch_KernelCallOp : Torch_Op<"kernel_call", [
    DeclareOpInterfaceMethods<TorchKernelOpInterface>]> {
  let summary = "Calls a Torch custom kernel";
  let description = [{
    Torch kernel calls are matched by the runtime based on signature, including
    the fully qualified kernel name (i.e. "namespace::name") and the tuple of
    argument types. This op models such an invocation.
  }];

  let arguments = (ins
    StrAttr:$kernelName,
    Variadic<AnyTorchType>:$args,
    StrArrayAttr:$sigArgTypes,
    StrArrayAttr:$sigRetTypes,
    BoolAttr:$sigIsVararg,
    BoolAttr:$sigIsVarret,
    BoolAttr:$sigIsMutable
    // TODO: Add alias mapping.
  );
  let results = (outs
    Variadic<AnyTorchType>:$results
  );

  let assemblyFormat = [{
    $kernelName $args `:` functional-type($args, results) attr-dict
  }];
}

#endif // TORCH_OPS
