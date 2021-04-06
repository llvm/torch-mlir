//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_NUMPY_PASSES
#define NPCOMP_NUMPY_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def NumpyPublicFunctionsToTensor : Pass<"numpy-public-functions-to-tensor", "ModuleOp"> {
  let summary = "Converts public functions to operate on tensors (instead of ndarray)";
  let constructor = "mlir::NPCOMP::Numpy::createPublicFunctionsToTensorPass()";
}

def NumpyArrayToTensor : Pass<"numpy-array-to-tensor", "FuncOp"> {
  let summary = "Replace arrays with tensors where possible (optimization only).";
  let description = [{
    This pass is analogous to an SSA-formation pass in a
    traditional compiler, with the added complication that arrays can alias
    each other in interesting ways.

    The current code doesn't implement any fancy algorithm, and is intended
    to be just sufficient for a first e2e spike. An algorithm inspired by the
    SSA formation literature will need to be implemented.

    Also, this pass doesn't currently handle interprocedural rewriting
    (of private functions), which is even more complex.
  }];
  let constructor = "mlir::NPCOMP::Numpy::createArrayToTensorPass()";
}


def NumpyRefinePublicReturn : Pass<"numpy-refine-public-return", "ModuleOp"> {
  let summary = "Refine public return";
  let constructor = "mlir::NPCOMP::Numpy::createRefinePublicReturnPass()";
  let description = [{
    Refines types of values return from public functions based on
    intraprocedural information.

    This pass effectively encodes an assumption by the pass pipeline author that
    the public calling convention of the module can have its types refined,
    without causing ABI mismatches. This is frequently true -- for example, in
    many systems, `tensor<?x?xf32>`, `tensor<3x3xf32>` and
    `tensor<*x!numpy.any_dtype>` are all the same data structure on calling
    convention boundaries.

    This pass is expected to run after shape refinement has occurred to
    otherwise resolve shapes, and is currently mainly useful to convert
    rank/dtype-erased function boundaries to ranked, dtyped code for
    compiler backends.
  }];
}


#endif // NPCOMP_NUMPY_PASSES
