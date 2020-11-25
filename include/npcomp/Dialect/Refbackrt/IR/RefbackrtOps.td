//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef REFBACKRT_OPS
#define REFBACKRT_OPS

include "npcomp/Dialect/Refbackrt/IR/RefbackrtBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Refbackrt_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Refbackrt_Dialect, mnemonic, traits> {
}

def Refbackrt_AbortIfOp : Refbackrt_Op<"abort_if"> {
  let summary = "Aborts if the predicate is true";
  let description = [{
    Aborts if the predicate is true.
  }];
  let arguments = (ins I1:$pred, StrAttr:$msg);
  let results = (outs);
  let assemblyFormat = "$pred `,` $msg attr-dict";
}

def Refbackrt_ModuleMetadataOp : Refbackrt_Op<"module_metadata", [
  SingleBlockImplicitTerminator<"ModuleMetadataTerminatorOp">
]> {
  let summary = "Global metadata for the module";
  let description = [{
    This op contains a region containing refbackrt.func_metadata ops,
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

def Refbackrt_ModuleMetadataTerminatorOp
    : Refbackrt_Op<"module_metadata_terminator",
                  [Terminator, HasParent<"ModuleMetadataOp">]> {
  let summary = "Implicit terminator for ModuleMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def Refbackrt_FuncMetadataOp
    : Refbackrt_Op<"func_metadata", [HasParent<"ModuleMetadataOp">]> {
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

#endif // #ifndef REFBACKRT_OPS
