//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_BASICPY_PASSES
#define NPCOMP_BASICPY_PASSES

include "mlir/Pass/PassBase.td"

def ATenRecognizeKernels : Pass<"aten-recognize-kernels", "FuncOp"> {
  let summary = "Recognizes ATen named kernels from torch.kernel_call ops.";
  let constructor = "mlir::NPCOMP::aten::createRecognizeKernelsPass()";
}

def ATenRecognizeTorchFallback : Pass<"aten-recognize-torch-fallback", "FuncOp"> {
  let summary = "Recognizes and assumes unsupported ATen kernels from torch.kernel_call ops.";
  let constructor = "mlir::NPCOMP::aten::createRecognizeTorchFallbackPass()";
}

def ATenLayerName : Pass<"aten-layer-name", "ModuleOp"> {
  let summary = "Generate layer names for ATen Dialect.";
  let constructor = "mlir::NPCOMP::aten::createATenLayerNamePass()";
}

def ATenLowering : Pass<"aten-to-std", "ModuleOp"> {
  let summary = "ATen dialect lowering to function calls.";
  let constructor = "mlir::NPCOMP::aten::createATenLoweringPass()";
}

def ATenOpReport : Pass<"aten-op-report", "ModuleOp"> {
  let summary = "Generate ATen operation report.";
  let constructor = "mlir::NPCOMP::aten::createATenOpReportPass()";
}

// TODO: Prefix pass with "aten-" and better document what this does.
def ATenReturnElimination : Pass<"return-elimination", "ModuleOp"> {
  let summary = "eliminate returns.";
  let constructor = "mlir::NPCOMP::aten::createReturnEliminationPass()";
}

#endif // NPCOMP_BASICPY_PASSES
