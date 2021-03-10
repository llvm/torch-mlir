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

    Contains type / shape information for arguments as described below:

    * ArgType(s): 
        Integer value from `CompilerDataStructures.h` for each argument
        indicating what type it is (e.g. Float, Int, Tensor, Dict, etc.)
    * ElementType(s): 
        Certain input ArgType's also have an element type (e.g. Tensor<float>,
        List<int>, etc.)
        TODO(brycearden): Support nested types (e.g. List<Tensor<float>>)
    * Rank(s):
        Integer value indicating the rank for each argument.
    * Shape(s):
        Flattened hyper-rectangular representation of the shapes for each argument.
        Since each shape's size varies based on the Rank, we pad out the shapes
        to size kMaxRank to make ABI lowering easier. See `LowerToRefbackrtABI.cpp`
        for details.

      Shapes Example: 
        constexpr int kMaxRank = 6;
        // func @f(%arg0: f32, %arg1: tensor<5xf32>) would result in...
        inputShapes = dense<...> : tensor<12xi32>
        // 2 shapes with 6 elements each so that the LowerToLLVM pass
        // where only the first `rank` values in each shape are valid.
        // 
        // can update the struct(s) by just grabbing a pointer at 
        // %shape_ptr = %base + (kMaxRank * argIndex)
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$funcName,
    I32Attr:$numInputs,
    I32Attr:$numOutputs,
    OptionalAttr<I32ElementsAttr>:$inputArgTypes,
    OptionalAttr<I32ElementsAttr>:$inputElementTypes,
    OptionalAttr<I32ElementsAttr>:$inputRanks,
    OptionalAttr<I32ElementsAttr>:$inputShapes,
    // I32ElementsAttr:$inputIsStatic,
    OptionalAttr<I32ElementsAttr>:$outputArgTypes,
    OptionalAttr<I32ElementsAttr>:$outputElementTypes,
    OptionalAttr<I32ElementsAttr>:$outputRanks,
    OptionalAttr<I32ElementsAttr>:$outputShapes
    //I32ElementsAttr:$outputIsStatic
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";

  let verifier = [{ return ::verify(*this); }];
}

#endif // #ifndef REFBACKRT_OPS
