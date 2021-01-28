//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_OPS
#define TORCH_OPS

include "npcomp/Dialect/Torch/IR/TorchTypes.td"
include "npcomp/Dialect/Torch/IR/OpInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

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

//===----------------------------------------------------------------------===//
// TorchScript modeling ops.
//===----------------------------------------------------------------------===//

def Torch_NnModuleOp : Torch_Op<"nn_module", [
    SingleBlockImplicitTerminator<"::mlir::NPCOMP::Torch::NnModuleTerminatorOp">]> {
  let summary = "Constructs a torch.nn.Module";
  let description = [{
    This op is used to represent a torch.nn.Module when importing a
    graph of Python objects.

    This op returns a new torch.nn.Module as an SSA value, with a set of
    declaratively specified properties.

    Example:

    ```mlir
    %2 = torch.nn_module  {
      torch.attr "b", %bool_true : !basicpy.BoolType
      torch.attr "i", %num3_i64 : i64
      torch.attr "f", %num : f64
      torch.attr "t", %0 : !numpy.ndarray<*:!numpy.any_dtype>
      torch.attr "submodule", %1 : !torch.nn.Module
      torch.method "method", @f
    }
    ```
  }];

  let arguments = (ins);
  let results = (outs Torch_NnModuleType:$result);
  let regions = (region SizedRegion<1>:$region);
  let verifier = "return ::verify(*this);";

  let assemblyFormat = "$region attr-dict";
}

def Torch_NnModuleTerminatorOp : Torch_Op<"nn_module_terminator", [Terminator,
    HasParent<"::mlir::NPCOMP::Torch::NnModuleOp">]> {
  let summary = "Implicit terminator for torch.nn_module";

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Torch_AttrOp : Torch_Op<"attr", [
    HasParent<"::mlir::NPCOMP::Torch::NnModuleOp">]> {
  let summary = "Define an attribute of a torch.nn.Module";
  let description = [{
    This op declaratively specifies that the parent torch.nn_module has an
    attribute `name` with value `value`, which is allowed to be an arbitrary
    Torch-compatible SSA value, including other torch.nn.Module's.
  }];

  let arguments = (ins StrAttr:$name, AnyTorchType:$value);
  let results = (outs);

  let assemblyFormat = [{
    $name `,` $value attr-dict `:` type($value)
  }];
}

def Torch_MethodOp : Torch_Op<"method", [
    HasParent<"::mlir::NPCOMP::Torch::NnModuleOp">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Define a method of a torch.nn.Module";
  let description = [{
    This op declaratively specifies that the parent torch.nn_module has a
    method `name` which calls `function`. `function` is an unbound function.
    That is, it explicitly takes the torch.nn.Module as a parameter (no implicit
    "self" object).
  }];

  let arguments = (ins StrAttr:$name, FlatSymbolRefAttr:$function);
  let results = (outs);

  let assemblyFormat = [{
    $name `,` $function attr-dict
  }];
}

#endif // TORCH_OPS
