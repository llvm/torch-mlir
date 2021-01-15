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

def RDExtractPipelineDefs : Pass<"rd-extract-pipeline-def", "ModuleOp"> {
  let summary = "Extracts a pipeline def op from a user's program.";
  let constructor = "mlir::NPCOMP::createExtractPipelineDefPass()";
}

def RDBuildInitFunc : Pass<"rd-build-init-func", "::mlir::NPCOMP::rd::PipelineDefinitionOp"> {
  let summary = "Builds init funcs for a pipeline.";
  let constructor = "mlir::NPCOMP::createBuildInitFuncPass()";
}

def RDBuildNextFunc : Pass<"rd-build-next-func", "::mlir::NPCOMP::rd::PipelineDefinitionOp"> {
  let summary = "Builds next funcs for a pipeline.";
  let constructor = "mlir::NPCOMP::createBuildNextFuncPass()";
}

def RDMergeFuncs : Pass<"rd-mergefuncs", "FuncOp"> {
  let summary = "Merges functions in the rd dialect";
  let constructor = "mlir::NPCOMP::createRDMergeFuncsPass()";
}

#endif // NPCOMP_RD_PASSES
