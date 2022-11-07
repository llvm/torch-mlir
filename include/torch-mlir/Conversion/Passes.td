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

def ConvertTorchBackendLegalToTosaCustom
    : Pass<"convert-torch-backend-legal-to-tosa-custom", "func::FuncOp"> {
  let summary = "Convert Torch backend-legal operations to TOSA CustomOps.";
  let description = [{
    This pass extends the selective decomposition patterns to TOSA backend,
    such that it will convert operations marked to custom-ops by the user
    for the TOSA backend to CustomOps in TOSA dialect.
    The backend-custom operations will not be decomposed or lowered through
    the existing Torch to TOSA conversion patterns, even if such patterns
    exist in the subsequent Torch to TOSA conversion pass; instead a TOSA
    CustomOp will be created with the specifications below:
      1. The 'identifier' attribute in the CustomOp will be the operation
         name of the corresponding ATen operation.
      2. The 'config' attribute in the CustomOp will be the specification
         of the compilation pipeline, i.e. "torch_mlir"
      3. All inputs to the ATen operation will be converted into legal TOSA
         operations. There will be no distiction between inputs and constant
         attributes; all will be treated as inputs to the CustomOp. It is up
         to the consuming backend to deduce them based on the ATen operation
         semantics.
      4. Since TOSA conversion pattern of Torch operations are responsible
         to consume and convet the constant attributes (such as 'axis' for a
         reduction operation), the TOSA CustomOp conversion will also match
         and rewrite these attributes as TOSA ConstOps as well. Specifically:
           - !torch.bool                -> tosa.ConstOp (of tensor type i64)
           - !torch.int                 -> tosa.ConstOp (of tensor type i64)
           - !torch.float               -> tosa.ConstOp (of tensor type f32)
           - !torch.list<int>           -> tosa.ConstOp (of tensor type i64)
           - !torch.none                -> tosa.CustomOp (of tensor type i1)
                                           The 'identifier' attribute of this
                                           CustomOp is 'torch.constant.none'
           - !torch.str                 -> tosa.CustomOp (of tensor type i1)
                                           The 'identifier' attribute of this
                                           CustomOp is 'torch.constant.str'
         All other Torch ATen operations will be lowered to TOSA by the Torch
         to TOSA conversion pass after this one.
        TODO: Extend the contract for other Torch constant operations such as:
          - torch.constant.device
          - torch.constant.number
      5. The input operands of the backend-legal Torch operation and the TOSA
         CustomOp will have the same order.
        TODO: Update this pass to populate the 'config' attribute of the TOSA
              CustomOp to establish a proper operand mapping scheme between
              the backend-legal Torch operation and the TOSA CustomOp.
              (See the commit at https://github.com/llvm/llvm-project/commit/d94ee70f4f01e4d9eec49e02eff57a5655618401)
  }];
  let options = [ListOption<
      "customOps", "custom-ops", "std::string",
      "List of operations considered backend-legal that should be converted to"
      " CustomOps in TOSA dialect.",
      "llvm::cl::ZeroOrMore">];
  let constructor =
      "mlir::torch::createConvertTorchBackendLegalToTosaCustomPass(/*customOps=*/{})";
}

def ConvertTorchToTosa : Pass<"convert-torch-to-tosa", "func::FuncOp"> {
  let summary = "Convert Torch ops to TOSA ops";
  let description = [{
    This pass assumes that TOSA ops are responsible for emitting error
    guards in case of shape mismatches.
  }];
  let constructor = "mlir::torch::createConvertTorchToTosaPass()";
}

def ConvertTorchToTMTensor : Pass<"convert-torch-to-tmtensor", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to TMTensor/Linalg ops";
  let description = [{
    Convert ATen ops to tmtensor/linalg ops.

    This pass is similar to the TorchToLinalg pass; the difference is that this
    pass also makes use of TMTensor Dialect, which the former one doesn't.
  }];
  let constructor = "mlir::torch::createConvertTorchToTMTensorPass()";
}

#ifdef TORCH_MLIR_ENABLE_MHLO
def ConvertTorchToMhlo : Pass<"convert-torch-to-mhlo", "func::FuncOp"> {
  let summary = "Convert Torch ops to MHLO ops";
  let description = [{
    Convert Torch ops to mhlo ops.
  }];
  let constructor = "mlir::torch::createConvertTorchToMhloPass()";

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
