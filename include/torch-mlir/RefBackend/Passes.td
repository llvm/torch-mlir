//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_REFBACKEND_PASSES
#define TORCHMLIR_REFBACKEND_PASSES

include "mlir/Pass/PassBase.td"

def MungeCallingConventions : Pass<"refback-munge-calling-conventions", "ModuleOp"> {
  let summary = "Munge calling conventions for calling via ExecutionEngine";
  let constructor = "mlir::torch::RefBackend::createMungeCallingConventionsPass();";
  let dependentDialects = ["memref::MemRefDialect"];
}

def InsertRngGlobals: Pass<"refback-insert-rng-globals", "ModuleOp"> {
  let summary = "Insert global variables and sequence to get the next global seed for RNG ops";
  let constructor = "mlir::torch::RefBackend::createInsertRngGlobalsPass();";
  let dependentDialects = ["memref::MemRefDialect"];
}

def ExpandOpsForLLVM : Pass<"refback-expand-ops-for-llvm", "func::FuncOp"> {
  let summary = "Expand ops into more primitive ops before LLVM lowering.";
  let constructor = "mlir::torch::RefBackend::createExpandOpsForLLVMPass();";
}

def MungeMemrefCopy : Pass<"refback-munge-memref-copy", "func::FuncOp"> {
  let summary = "Munge memref.copy to linalg.copy";
  let constructor = "mlir::torch::RefBackend::createMungeMemrefCopyPass();";
  let dependentDialects = ["memref::MemRefDialect"];
}

def GeneralizeTensorPad : Pass<"refback-generalize-tensor-pad", "func::FuncOp"> {
  let summary = "Convert tensor.pad to linalg ops";
  let constructor = "mlir::torch::RefBackend::createGeneralizeTensorPadPass()";
}

#endif // TORCHMLIR_REFBACKEND_PASSES
