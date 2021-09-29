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

def ExpandOpsForLLVM : Pass<"refback-expand-ops-for-llvm", "FuncOp"> {
  let summary = "Expand ops into more primitive ops before LLVM lowering.";
  let constructor = "mlir::torch::RefBackend::createExpandOpsForLLVMPass();";
}

#endif // TORCHMLIR_REFBACKEND_PASSES
