//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RD_OPS
#define RD_OPS

include "npcomp/Dialect/RD/IR/RDBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class RD_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<RD_Dialect, mnemonic, traits> {
}

// TODO: Consider defining a region for representing Dataset construction
// instead of relying on FuncOp's to enforce additional constraints.

def RD_MakeIteratorOp : RD_Op<"make_iterator"> {
    let summary = "Builds an iterator for a given dataset.";
    let description = [{
        Builds an iterator for a given dataset.
    }];

    let arguments = (ins FlatSymbolRefAttr:$ds, Variadic<AnyType>:$operands);
    let results = (outs RD_Iterator:$itr);

    // TODO: add verification to ensure that types match up!!

    let assemblyFormat = [{
        $ds `(` $operands `)` attr-dict `:` functional-type(operands, results)
    }];
}

def RD_IteratorNextOp : RD_Op<"iterator_next"> {
    let summary = "Advances the iterator by one and returns the current values.";
    let description = [{
        Returns a tuple of values:
         1. A Boolean indicating if there was a valid value.
         2. The value (only valid if the boolean is true).
    }];

    let arguments = (ins RD_Iterator:$iterator);
    let results = (outs I1:$valid, I64:$value);  // TODO: make generic over values.

    let assemblyFormat = "$iterator attr-dict `:` functional-type(operands, results)";  // TODO: just write the element type in future?
}

def RD_Print : RD_Op<"print"> {
    let summary = "Prints to the console.";
    let description = [{
        Example:

        rd.print %value : i64
    }];

    let arguments = (ins I64:$value);

    let assemblyFormat = "$value attr-dict `:` type($value)";
}

// Dataset ops allow users to build up compositions of computations.

class RD_DatasetOp<string mnemonic>
    : RD_Op<mnemonic, [NoSideEffect]> {
}

def RD_RangeOp : RD_DatasetOp<"range"> {
    let summary = "Range of values.";
    let description = [{
        A sequence of consecutive integers.
    }];

    let arguments = (ins I64:$start, I64:$end, I64:$step);
    let results = (outs RD_Dataset:$result);

    let assemblyFormat = "$start `to` $end `by` $step attr-dict `:` functional-type(operands, results)";
}

def RD_InlineMapOp : RD_DatasetOp<"inline_map"> {
    let summary = "Transform element values by applying function `f` to each of them.";
    let description = [{
        A serial, sequential execution implementation of the elementwise application of
        f to each of the elements in this dataset.
    }];

    let arguments = (ins RD_Dataset:$src, FlatSymbolRefAttr:$f);
    let results = (outs RD_Dataset:$result);

    let assemblyFormat = "$f `(` $src `)` attr-dict `:` functional-type(operands, results)";
}

def RD_FilterOp : RD_DatasetOp<"filter"> {
    let summary = "Removes all values where `f` returns true.";
    let description = [{
        A new dataset containing every element in `src` except for those
        that `f` of the element returns true.
    }];

    let arguments = (ins RD_Dataset:$src, FlatSymbolRefAttr:$f);
    let results = (outs RD_Dataset:$result);

    let assemblyFormat = "$src `excluding` $f attr-dict `:` functional-type(operands, results)";
}

#endif // #ifndef RD_OPS
