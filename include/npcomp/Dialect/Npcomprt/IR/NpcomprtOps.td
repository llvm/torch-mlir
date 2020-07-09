//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMPRT_OPS
#define NPCOMPRT_OPS

include "npcomp/Dialect/Npcomprt/IR/NpcomprtBase.td"

class Npcomprt_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Npcomprt_Dialect, mnemonic, traits> {
}

def Npcomprt_ToMemrefOp : Npcomprt_Op<"to_memref"> {
  let summary = "Gets a memref descriptor from a tensor";
  let description = [{
    Gets a memref descriptor from a tensor.
  }];
  let arguments = (ins Npcomprt_Tensor:$tensor);
  let results = (outs AnyUnrankedMemRef:$memref);
  let assemblyFormat = "$tensor attr-dict `:` type($memref)";
}

def Npcomprt_FromMemrefOp : Npcomprt_Op<"from_memref"> {
  let summary = "Converts a memref descriptor to a tensor";
  let description = [{
    Copies the data from a memref into a new tensor.
  }];
  let arguments = (ins AnyUnrankedMemRef:$memref);
  let results = (outs Npcomprt_Tensor:$tensor);
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

def Npcomprt_GetExtentOp : Npcomprt_Op<"get_extent"> {
  let summary = "Gets the specified extent of the tensor";
  let description = [{
    Gets the `dim`'th extent of the tensor.
  }];
  let arguments = (ins Npcomprt_Tensor:$tensor, I32:$dim);
  // TODO: Use i32 instead of index so the runtime function
  // can return std::int32_t.
  let results = (outs Index:$extent);
  let assemblyFormat = "$tensor `,` $dim attr-dict";
}

def Npcomprt_AbortIfOp : Npcomprt_Op<"abort_if"> {
  let summary = "Aborts if the predicate is true";
  let description = [{
    Aborts if the predicate is true.
  }];
  let arguments = (ins I1:$pred);
  let results = (outs);
  let assemblyFormat = "$pred attr-dict";
}

def Npcomprt_ModuleMetadataOp : Npcomprt_Op<"module_metadata", [
  SingleBlockImplicitTerminator<"ModuleMetadataTerminatorOp">
]> {
  let summary = "Global metadata for the module";
  let description = [{
    This op contains a region containing npcomprt.func_metadata ops,
    which give information about the functions in the module. This allows
    the module to be introspected when it is loaded, such as looking up
    functions.
    Future uses are checking how many results functions should have, or
    what their argument types are expected to be to provide clean and safe
    errors when invocations fail.

    TODO: Verify that there should be no more than one of these ops in a
    module.

    This op is designed to hold a region, which makes it easy to convert to
    a single LLVM global with a single conversion pattern.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$metadatas);

  let printer = [{ return ::print$cppClass(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def Npcomprt_ModuleMetadataTerminatorOp
    : Npcomprt_Op<"module_metadata_terminator",
                  [Terminator, HasParent<"ModuleMetadataOp">]> {
  let summary = "Implicit terminator for ModuleMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Npcomprt_FuncMetadataOp
    : Npcomprt_Op<"func_metadata", [HasParent<"ModuleMetadataOp">]> {
  let summary = "Runtime metadata for a single func";
  let description = [{
    Runtime metadata for a single func.

    TODO: Augment this with information for type/shape checking of arguments.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$funcName,
    I32Attr:$numInputs,
    I32Attr:$numOutputs
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";

  let verifier = [{ return ::verify(*this); }];
}

#endif // #ifndef NPCOMPRT_OPS
