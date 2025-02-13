//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_PASSES
#define TORCHMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Torch conversions
//===----------------------------------------------------------------------===//

def ConvertTorchToArith : Pass<"convert-torch-to-arith", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to Std ops";
  let constructor = "mlir::torch::createConvertTorchToArithPass()";
}

def ConvertTorchToSCF: Pass<"convert-torch-to-scf", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to SCF ops";
  let constructor = "mlir::torch::createConvertTorchToSCFPass()";
}

def ConvertTorchToLinalg : Pass<"convert-torch-to-linalg", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to Linalg ops";
  let description = [{
    Convert ATen ops to linalg ops.

    This pass's main responsibility is to bridge the world between ops
    that safely terminate the program in case of operand shape mismatches
    (ATen) and ops where such mismatches are undefined behavior (linalg).

    To model the termination of the program for implementing error guards,
    we use the `cf.assert` op.
    This is a design decision that is at variance from other passes in the
    ecosystem, which use the
    `shape` dialect's witness system (`shape.cstr_*` family of ops feeding into
    `shape.assuming` regions). This is a change in design decisions
    from those passes (which the authors of this pass have contributed to).
    The reasons for this change are heuristic, but boil down to:
    1. The modeling of `shape.assuming` is odd, as it uses a region, which is
       not a good fit for modeling error guards. Regions mark a "start" and an
       "end" (which is their nesting property). But
       modeling assertions in the program doesn't fit into that. For assertions,
       only the "start" matters (once tested, a predicate remains true "forever"
       -- it doesn't end at the "yield" of the region).
       Thus, having regions places arbitrary "end"s that just add IR structure
       that has no semantic value for modeling this problem! (and to make things
       worse the "end"s, which we don't need, are what require "yielding"
       values, which interrupts use-def chains). Consider the different
       structural properties of regions:
       a. IsolatedFromAbove region:
          - "start" interrupts use-def chains,
          - "end" interrupts use-def chains
          - structurally protects from intra-block upward and downward
            code motion
       b. Capturing region (like `shape.assuming`):
          - "start" does not interrupt use-def chains,
          - "end" interrupts use-def chains
          - structurally protects from intra-block upward and downward
            code motion
       c. What we "ideally" want:
          - "start" interrupts use-def chains (can be pruned though)
          - no "end" IR structure!
          - structurally protects from intra-block upward code motion
            (but not downward code motion!)
          - Observation: We probably can't get all of this, but overall this
            problem is much better suited for a "MemorySSA"-like
            abstraction, call it "EffectSSA" which is constructed on-demand
            based on MLIR's effect modeling system (rather than
            `shape.assuming`, which only covers the effects the IR creator
            encoded -- with witnesses/`shape.assuming` -- it is easy to forget
            to handle effects other than those encoded in the
            witness structure).
    2. The presence of `shape.assuming` regions tends to create highly nested
       IR structures, which don't interoperate well with any other IR
       structures, and creates very bulky IR (and IR creation code). In general
       if we are going to do anything with anything (e.g. canonicalize) we
       end up needing need to either:
       a. Flatten the `shape.assuming` IR (defeating the purpose of having
          it).
       b. Do some sort of shape.assuming "region merging".
       c. Have special patterns that handle a subset of special cases (looking
          through "yields" and such) and don't generalize.
    3. Witnesses tend to encourage non-scalable peephole transformations, which
       tend to make analyses/transformations non-robust to the presence of
       control flow and side effecting ops (easy to forget to handle side
       effects other than those modeled by the witness system).
    4. All this code operates on ranked tensors, for which using individual
       SSA values for sizes (rather than a "shape type") seems to
       work really well at this level of abstraction based on prior experience
       in other projects. (unranked code tends to benefit from having a discrete
       "shape type" to model shapes).

    We will see if we end up needing something like `shape.assuming`, but for
    now, it seems likely we can do something simpler and just bypass it. The
    design of having an EffectSSA that is constructed on-demand seems very
    compelling for modeling effects more broadly.
  }];
  let constructor = "mlir::torch::createConvertTorchToLinalgPass()";
}

def ConvertTorchToTensor : Pass<"convert-torch-to-tensor", "func::FuncOp"> {
  let summary = "Convert Torch ops to the Tensor dialect";
  let description = [{
    Converts any `Torch` operators that were expressible as `Tensor` dialect
    operations.
  }];
  let constructor = "mlir::torch::createConvertTorchToTensorPass()";
}

#ifdef TORCH_MLIR_ENABLE_TOSA
def ConvertTorchToTosa : Pass<"convert-torch-to-tosa", "func::FuncOp"> {
  let summary = "Convert Torch ops to TOSA ops";
  let description = [{
    This pass assumes that TOSA ops are responsible for emitting error
    guards in case of shape mismatches.
  }];
  let constructor = "mlir::torch::createConvertTorchToTosaPass()";
}
#endif

def ConvertTorchToTMTensor : Pass<"convert-torch-to-tmtensor", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to TMTensor/Linalg ops";
  let description = [{
    Convert ATen ops to tmtensor/linalg ops.

    This pass is similar to the TorchToLinalg pass; the difference is that this
    pass also makes use of TMTensor Dialect, which the former one doesn't.
  }];
  let constructor = "mlir::torch::createConvertTorchToTMTensorPass()";
}

def ConvertTorchConversionToMLProgram : Pass<"convert-torch-conversion-to-mlprogram", "ModuleOp"> {
  let summary = "Convert recognized TorchConversion ops to MLProgram ops";
  let description = [{
    Convert TorchConversion ops to mlprogram ops.
  }];
  let constructor = "mlir::torch::createConvertTorchConversionToMLProgramPass()";
}

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
def ConvertTorchToStablehlo : Pass<"convert-torch-to-stablehlo", "func::FuncOp"> {
  let summary = "Convert Torch ops to Stablehlo ops";
  let description = [{
    Convert Torch ops to Stablehlo ops.
  }];
  let constructor = "mlir::torch::createConvertTorchToStablehloPass()";

  // Specify any options.
  let options = [
    Option<"enableStaticShape", "enable-static-shape", "bool", /*default=*/"false",
           "Enable static shape conversion">,
    // The i64 calculation is much slower than i32 on some devices, such as
    // Nvidia GPU. One can truncate from i64 to i32 since dimension sizes
    // are unlikely to exceed the range of i32(4GiB)
    Option<"enableI32Index", "enable-i32-index", "bool", /*default=*/"false",
           "Enable truncate index from i64 to i32(unsafely)">,
  ];
}
#endif

#endif // TORCHMLIR_CONVERSION_PASSES
