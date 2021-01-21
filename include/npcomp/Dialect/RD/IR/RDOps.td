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
include "npcomp/Dialect/RD/IR/RDDatasetInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

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

    // TODO: add an extra argument for the pointer to the memory.
    let arguments = (ins FlatSymbolRefAttr:$ds, Variadic<AnyType>:$operands);
    let results = (outs RD_Iterator:$itr);  // TODO: Refactor to have the op take this as a pointer.

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

def RD_IteratorDeleteOp : RD_Op<"iterator_deinit"> {
    let summary = "Releases dynamically allocated resources associated with the iterator.";

    let arguments = (ins RD_Iterator:$iterator);

    let assemblyFormat = "$iterator attr-dict `:` type($iterator)";
}

def RD_IteratorAlloca : RD_Op<"iterator_alloca"> {
    let summary = "Stack-allocate memory that can be used for the iterator.";
    let description = [{
        Lowers to llvm.alloca and returns a pointer to an uninitialized iterator.
    }];

    let arguments = (ins FlatSymbolRefAttr:$ds);
    // let results = TODO!!!
}

def RD_IteratorMallocOp : RD_Op<"iterator_malloc"> {
    let summary = "Heap allocate memory that can be used for the iterator.";
    let description = [{
        Allocates (with malloc) the memory required to execute an iterator.

        The returned iterator memory is unitialized, and it is the responsibility of
        the caller to eventually `rd.iterator_free` the iterator memory.
    }];

    let arguments = (ins FlatSymbolRefAttr:$ds);
    // TODO: results!
}

def RD_IteratorDeallocOp : RD_Op<"iterator_free"> {
    let summary = "Deallocates heap allocated memory allocated with iterator_malloc.";
    // TODO: FILL ME IN!
    let arguments = (ins FlatSymbolRefAttr:$ds);  // TODO: Also add memory pointer.
}

def RD_IteratorCheckpointOp : RD_Op<"iterator_checkpoint"> {
    let summary = "Persist iterator state to a buffer.";

    let description = [{
        To handle faults, or to quickly duplicate the state of an iterator,
        the iterator_checkpoint op serializes state into a buffer which can later
        be used to initialize a new iterator using the `rd.iterator_restore` op.

        The input iterator is not modified.

        As with all rd operations, they may not be called concurrently with each
        other on the same iterator.
    }];
    // TODO: Implement me!
}

def RD_IteratorRestoreOp : RD_Op<"iterator_restore"> {
    let summary = "Restore iterator state from a buffer.";
    // TODO: Implement me!
}

def RD_PipelineDefinitionOp : RD_Op<"pipeline_def", [IsolatedFromAbove, Symbol, NoRegionArguments, SingleBlockImplicitTerminator<"PipelineDefinitionTerminatorOp">]> {
    let summary = "[Internal] A region to collect ops associated with a single pipeline.";
    let description = [{
        The RD compilation flow translates the input functions and operations
        into a collection of coroutines. All ops associated with a single dataset are
        contained within the region of this op.
    }];

    let regions = (region AnyRegion:$body);
}

def RD_IteratorIndexOp : RD_Op<"iterator_index"> {
    let summary = "[Internal] Extracts the state for a nested iterator within this iterator.";

    let arguments = (ins RD_Iterator:$parent, I64ArrayAttr:$childIndex);
    let results = (outs RD_Iterator:$result);

    let assemblyFormat = "$parent $childIndex attr-dict `:` type($result)";

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$parent, "int64_t":$childIndex)>
    ];
}

def RD_PipelineDefinitionTerminatorOp : RD_Op<"pipeline_def_terminator", [Terminator, HasParent<"PipelineDefinitionOp">]> {
    let summary = "A pseudo op that marks the end of a pipeline definition";
    let description = [{
        `pipeline_def_terminator` is a special terminator operation for the body of a `pipeline_def`,
        it has no semantic meaning beyond keeping the body of a `pipeline_def` well-formed.
    }];

    let assemblyFormat = "attr-dict";
}

def RD_PrintOp : RD_Op<"print"> {
    let summary = "[Debug] Prints to the console.";
    let description = [{
        Example:

        rd.print %value : i64

        Used for debugging only!
    }];

    let arguments = (ins I64:$value);

    let assemblyFormat = "$value attr-dict `:` type($value)";
}

// Dataset ops allow users to build up compositions of computations.

multiclass RD_DatasetOp<
        string mnemonic,
        string summaryStr,
        string descriptionStr,
        dag datasetInputs,
        dag initInputs,
        string assemblyFormatStr,
        string typeStr = "return {};"> {
    // The main op used to describe the input pipeline.
    def Op : RD_Op<mnemonic, [NoSideEffect, DeclareOpInterfaceMethods<DatasetTransformOpInterface>]> {
        let summary = summaryStr;
        let description = descriptionStr;
        let arguments = !con(datasetInputs, initInputs);
        let results = (outs RD_Dataset:$result);
        let assemblyFormat = assemblyFormatStr;
    }

    // A related op to initialize iterator state.
    def InitOp : RD_Op<!strconcat(mnemonic, ".init"), [DatasetInitOpInterface]> {
        let summary = !strconcat("Initialize state for ", mnemonic);
        let arguments = !con(initInputs, (ins RD_Iterator:$state));
        // No results.

        let extraClassDeclaration = !strconcat([{
            llvm::Optional<LLVM::LLVMType> buildStateLLVMType() {
        }], typeStr, [{
            }
        }]);
    }

    // A related op to advance the iterator state and return a result.
    def NextOp : RD_Op<!strconcat(mnemonic, ".next"), [DatasetNextOpInterface]> {
        let summary = !strconcat("Next func for ", mnemonic);
        let arguments = !con(datasetInputs, (ins RD_Iterator:$state));
        let results = (outs RD_Dataset:$result);

        let extraClassDeclaration = !strconcat([{    llvm::Optional<LLVM::LLVMType> buildStateLLVMType() {
        }], typeStr, [{    } }]);
    }
}

defm RD_Range : RD_DatasetOp<"range", "Range of values.", [{
        A sequence of consecutive integers.
    }], (ins ), (ins I64:$start, I64:$end, I64:$step),
    "$start `to` $end `by` $step attr-dict `:` functional-type(operands, results)", [{
        auto int64Ty = LLVM::LLVMType::getInt64Ty(getContext());
        return LLVM::LLVMType::getStructTy(getContext(), {int64Ty, int64Ty, int64Ty});
    }]>;

defm RD_InlineMap : RD_DatasetOp<"inline_map", "Transform element values by applying function `f` to each of them.", [{
        A serial, sequential execution implementation of the elementwise application of
        f to each of the elements in this dataset.
    }], (ins RD_Dataset:$src), (ins FlatSymbolRefAttr:$f),
    "$f `(` $src `)` attr-dict `:` functional-type(operands, results)">;

defm RD_Filter : RD_DatasetOp<"filter", "Removes all values where `f` returns true.", [{
        A new dataset containing every element in `src` except for those
        that `f` of the element returns true.
    }], (ins RD_Dataset:$src), (ins FlatSymbolRefAttr:$f),
    "$src `excluding` $f attr-dict `:` functional-type(operands, results)">;

#endif // #ifndef RD_OPS
