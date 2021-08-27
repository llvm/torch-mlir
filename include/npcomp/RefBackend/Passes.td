//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_REFBACKEND_PASSES
#define NPCOMP_REFBACKEND_PASSES

include "mlir/Pass/PassBase.td"

def LowerToRefbackrtABI : Pass<"lower-to-refbackrt-abi", "ModuleOp"> {
  let summary = "Lower constructs requiring runtime support to `refbackrt`";
  let description = [{
    We have a specialized dialect `refbackrt` which models our runtime's data
    structures, and function signatures (and presumably eventually, other
    ABI boundaries like external calls if we ever support it) will be
    converted.

    The constructs requiring runtime support are:
    - function signatures / module metadata
    - error handling
  }];
  let constructor = "mlir::NPCOMP::createLowerToRefbackrtABIPass()";
}

def LowerAllocMemRefOps : Pass<"lower-alloc-memref-ops", "FuncOp"> {
  let summary = "Lower AllocMemRefOp's";
  let constructor = "mlir::NPCOMP::createLowerAllocMemRefOpsPass()";
  let dependentDialects = ["tensor::TensorDialect", "memref::MemRefDialect"];
}

def LowerToLLVM : Pass<"refback-lower-to-llvm", "ModuleOp"> {
  let summary = "Lower everything to LLVM";
  let constructor = "mlir::NPCOMP::createLowerToLLVMPass();";
}

#endif // NPCOMP_REFBACKEND_PASSES
