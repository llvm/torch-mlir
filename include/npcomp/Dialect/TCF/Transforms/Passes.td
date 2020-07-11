//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TCF_PASSES
#define NPCOMP_TCF_PASSES

include "mlir/Pass/PassBase.td"

def TCFShapeRefinement : Pass<"tcf-shape-refinement", "FuncOp"> {
  let summary = "Refines shapes of tensors";
  let constructor = "mlir::NPCOMP::tcf::createShapeRefinementPass()";
}

#endif // NPCOMP_TCF_PASSES
