//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TYPING_TRANSFORMS_PASSES
#define NPCOMP_TYPING_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TypeInference
//===----------------------------------------------------------------------===//

def CPAFunctionTypeInference : Pass<"npcomp-cpa-type-inference", "FuncOp"> {
  let summary = "Performs CPA function level type inference";
  let constructor = "mlir::NPCOMP::Typing::createCPAFunctionTypeInferencePass()";
}

#endif // NPCOMP_TYPING_TRANSFORMS_PASSES
