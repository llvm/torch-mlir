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

#endif // NPCOMP_BACKEND_COMMON_PASSES
