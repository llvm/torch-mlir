//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TOSA_PASSES
#define TORCHMLIR_CONVERSION_TOSA_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Torch conversions
//===----------------------------------------------------------------------===//

def ConvertTorchToTosa : Pass<"convert-torch-to-tosa", "func::FuncOp"> {
  let summary = "Convert Torch ops to TOSA ops";
  let description = [{
    This pass assumes that TOSA ops are responsible for emitting error
    guards in case of shape mismatches.
  }];
  let constructor = "mlir::torch::createConvertTorchToTosaPass()";
}

def VerifyTosaBackendContract : Pass<"torch-verify-tosa-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the linalg-on-tensors backend contract";
  let constructor = "mlir::torch::createVerifyTosaBackendContractPass()";
}

#endif // TORCHMLIR_CONVERSION_TOSA_PASSES
