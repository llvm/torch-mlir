//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TORCHCONVERSION_PASSES
#define NPCOMP_TORCHCONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def VerifyInvariantsBeforeBackendLowering
    : Pass<"torch-verify-invariants-before-backend-lowering", "ModuleOp"> {
  let summary = "Verify invariants required by backend lowering";
  let constructor =
    "mlir::NPCOMP::TorchConversion::createVerifyInvariantsBeforeBackendLoweringPass()";
  let description = [{
    This pass checks any invariants needed by the process of lowering the
    `torch` dialect to the npcomp backend contract.

    The most important invariant is that all tensors should be ranked and have
    a known dtype. It is useful to catch this early because it usually
    represents a simple bug in RefineTypes, but can manifest as many different
    kinds of obscure symptoms during lowering.

    TODO: This pass should probably be phrased as checking the
    "torch backend contract" and moved to that dialect once we have more
    substantial definition definition around what that layer is from an
    "allowlist" perspective.
  }];
}

def FuncBackendTypeConversion : Pass<"torch-func-backend-type-conversion", "ModuleOp"> {
  let summary = "Convert functions to operate on builtin tensors";
  let constructor = "mlir::NPCOMP::TorchConversion::createFuncBackendTypeConversionPass()";
  let description = [{
    Partial type conversion pass analogous in scope to the upstream
    `func-bufferize` pass. See details there.
  }];
}

def FinalizingBackendTypeConversion
    : Pass<"torch-finalizing-backend-type-conversion", "FuncOp"> {
  let summary = "Finalizes a partial conversion to builtin tensors";
  let constructor =
    "mlir::NPCOMP::TorchConversion::createFinalizingBackendTypeConversionPass()";
  let description = [{
    Analogous in scope to the upstream `finalizing-bufferize` pass.
    See details there.
  }];
}


def TmpDeleteDeadIREELists
    : Pass<"torch-tmp-delete-dead-lists", "FuncOp"> {
  let summary = "Delete dead !iree.list ops";
  let constructor =
    "mlir::NPCOMP::TorchConversion::createTmpDeleteDeadIREEListsPass()";
  let description = [{
    Runs a few patterns to delete dead !iree.list ops until IREE can support
    running them. Currently, these will get materialized as part of conversions
    for ops like AtenConv2dOp that have list operands, even though they are dead
    (for those ops, we pattern match a specific case of static constant lists).
    Currently, this will break execution of those tests because the IREE
    side of these ops still doesn't work (nor is IREE able to delete them
    itself).

    TODO: Add support to IREE to run these ops E2E.
    TODO: Remove this pass once IREE can run them e2e.
  }];
}


#endif // NPCOMP_TORCHCONVERSION_PASSES
