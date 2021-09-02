//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_BACKEND_COMMON_PASSES
#define NPCOMP_BACKEND_COMMON_PASSES

include "mlir/Pass/PassBase.td"

def VerifyBackendContract : Pass<"npcomp-verify-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the backend contract that npcomp targets";
  let constructor = "mlir::NPCOMP::CommonBackend::createVerifyBackendContractPass()";
}

def DeleteDeadIREELists : Pass<"delete-dead-iree-lists", "FuncOp"> {
  let summary = "Deletes dead `!iree.list` values";
  let description = [{
    Some backends cannot handle `!iree.list` values which might be incidentally
    created during type conversion. This pass deletes them so those backends
    can still run programs that don't use lists in an essential way.

    For example, list arguments to convolution ops (such as strides) turn into
    dead lists with our current lowerings.
  }];
  let constructor = "mlir::NPCOMP::CommonBackend::createDeleteDeadIREEListsPass()";
}

#endif // NPCOMP_BACKEND_COMMON_PASSES
