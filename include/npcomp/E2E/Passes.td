//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_E2E_PASSES
#define NPCOMP_E2E_PASSES

include "mlir/Pass/PassBase.td"

def LowerLinalgOnTensorToLinalgOnMemref :
    Pass<"lower-linalg-tensor-to-memref", "FuncOp"> {
  let summary = "Lowers linalg on tensors to linalg on memrefs";
  let constructor = "mlir::NPCOMP::createLowerLinalgOnTensorToLinalgOnMemrefPass()";
}

def LowerBroadcastToToLoops :
    Pass<"lower-broadcast-to-to-loops", "FuncOp"> {
  let summary = "Lower tcp::BroadcastTo to loops.";
  let constructor = "mlir::NPCOMP::createLowerBroadcastToToLoopsPass()";
}

def ResolveShapeOfOps : Pass<"resolve-shape-of-ops", "FuncOp"> {
  let summary = "Resolve shape.shape_of ops to other shapes.";
  let constructor = "mlir::NPCOMP::createResolveShapeOfOpsPass()";
}

def ResolveTensorLoadStoreOps : Pass<"resolve-tensor-load-store-ops", "FuncOp"> {
  let summary = "Resolve tensor_load/tensor_store ops";
  let constructor = "mlir::NPCOMP::createResolveTensorLoadStoreOpsPass()";
}

def LowerLinalgLoopDimOps : Pass<"lower-linalg-loop-dim-ops", "FuncOp"> {
  let summary = "Lower dim ops introduced by linalg to loops lowering";
  let constructor = "mlir::NPCOMP::createLowerLinalgLoopDimOpsPass();";
}

def LowerRankedShapes : Pass<"lower-ranked-shapes", "FuncOp"> {
  let summary = "Lower ranked !shape.shape types to SSA values";
  let constructor = "mlir::NPCOMP::createLowerRankedShapesPass()";
}

#endif // NPCOMP_E2E_PASSES
