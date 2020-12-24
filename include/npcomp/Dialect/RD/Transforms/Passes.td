//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_RD_PASSES
#define NPCOMP_RD_PASSES

include "mlir/Pass/PassBase.td"

def RDMergeFuncs : Pass<"rd-mergefuncs", "FuncOp"> {
  let summary = "Merges functions in the rd dialect";
  let constructor = "mlir::NPCOMP::createRDMergeFuncsPass()";
}

#endif // NPCOMP_RD_PASSES
