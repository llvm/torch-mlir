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

def BypassShapes : Pass<"bypass-shapes", "FuncOp"> {
  let summary = "Bypass shape calculations around ops";
  let constructor = "mlir::NPCOMP::createBypassShapesPass()";
}

def LowerShapeConstraints : Pass<"lower-shape-constraints", "FuncOp"> {
  let summary = "Lower shape dialect constructs related to constraints";
  let constructor = "mlir::NPCOMP::createLowerShapeConstraintsPass()";
}

def LowerShapedResultsToMemref : Pass<"lower-shaped-results-to-memref", "FuncOp"> {
  let summary = "Lower tcp.shaped_results regions";
  let constructor = "mlir::NPCOMP::createLowerShapedResultsToMemrefPass()";
}

def LowerStdToMemref : Pass<"lower-std-to-memref", "FuncOp"> {
  let summary = "Lower std ops to memref";
  let constructor = "mlir::NPCOMP::createLowerStdToMemrefPass()";
}

def LowerConstantTensorsToMemref :
    Pass<"lower-constant-tensors-to-memref", "ModuleOp"> {
  let summary = "Lower std.constant of tensor type to memref";
  let description = [{
    This must be a module pass since it involves creating tcp.global ops.
  }];
  let constructor = "mlir::NPCOMP::createLowerConstantTensorsToMemrefPass()";
}

def LowerStructuralToMemref :
    Pass<"lower-structural-to-memref", "FuncOp"> {
  let summary = "Lower structural IR constructs to memref";
  let description = [{
    Structural constructs include:
      - control flow ops (both CFG and SCF)
      - function signatures
      - TODO: calls
    An op is "structural" if it doesn't really care about the types it operates
    on, but the types just have to converted to be consistent.

    This pass also cleans up any previous memref<->tensor materializations,
    finalizing the conversion from tensor to memref.
  }];
  let constructor = "mlir::NPCOMP::createLowerStructuralToMemrefPass()";
}

def LowerToNpcomprtABI : Pass<"lower-to-npcomprt-abi", "ModuleOp"> {
  let summary = "Lower constructs requiring runtime support to `npcomprt`";
  let description = [{
    We have a specialized dialect `npcomprt` which models our runtime's data
    structures, and function signatures (and presumably eventually, other
    ABI boundaries like external calls if we ever support it) will be
    converted.

    The constructs requiring runtime support are:
    - function signatures / module metadata
    - globals
    - error handling
  }];
  let constructor = "mlir::NPCOMP::createLowerToNpcomprtABIPass()";
}

def LowerAllocMemRefOps : Pass<"lower-alloc-memref-ops", "FuncOp"> {
  let summary = "Lower AllocMemRefOp's";
  let constructor = "mlir::NPCOMP::createLowerAllocMemRefOpsPass()";
}

def LowerToLLVM : Pass<"e2e-lower-to-llvm", "ModuleOp"> {
  let summary = "Lower everything to LLVM";
  let constructor = "mlir::NPCOMP::createLowerToLLVMPass();";
}

#endif // NPCOMP_E2E_PASSES
