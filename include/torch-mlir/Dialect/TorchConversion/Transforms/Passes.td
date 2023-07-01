//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_TORCHCONVERSION_PASSES
#define TORCHMLIR_TORCHCONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FuncBackendTypeConversion : Pass<"torch-func-backend-type-conversion", "ModuleOp"> {
  let summary = "Convert functions to operate on builtin tensors";
  let constructor = "mlir::torch::TorchConversion::createFuncBackendTypeConversionPass()";
  let description = [{
    Partial type conversion pass analogous in scope to the upstream
    `func-bufferize` pass. See details there.
  }];
}

def SingleTensorReturnConversion : Pass<"torch-single-tensor-return-conversion", "ModuleOp"> {
  let summary = "Convert functions to return single tensor";
  let constructor = "mlir::torch::TorchConversion::createSingleTensorReturnConversionPass(/*valueIndex=*/0)";
  let options = [
    Option<"valueIndex", "value-index", "unsigned", /*default=*/"0",
           "Index of the value in original return op to return instead">
  ];
  let description = [{
    Convert functions to return single tensor
  }];
}

def FinalizingBackendTypeConversion
    : Pass<"torch-finalizing-backend-type-conversion", "func::FuncOp"> {
  let summary = "Finalizes a partial conversion to builtin tensors";
  let constructor =
    "mlir::torch::TorchConversion::createFinalizingBackendTypeConversionPass()";
  let description = [{
    Analogous in scope to the upstream `finalizing-bufferize` pass.
    See details there.
  }];
}

def VerifyLinalgOnTensorsBackendContract : Pass<"torch-verify-linalg-on-tensors-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the linalg-on-tensors backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyLinalgOnTensorsBackendContractPass()";
}

def VerifyTosaBackendContract : Pass<"torch-verify-tosa-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the linalg-on-tensors backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyTosaBackendContractPass()";
}

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
def VerifyStablehloBackendContract : Pass<"torch-verify-stablehlo-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the stablehlo backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyStablehloBackendContractPass()";
}
#endif // TORCH_MLIR_ENABLE_STABLEHLO
#endif // TORCHMLIR_TORCHCONVERSION_PASSES
