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

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
def FuncBackendTypeConversionForStablehlo : Pass<"torch-func-backend-type-conversion-for-stablehlo", "ModuleOp"> {
  let summary = "Convert functions to operate on builtin tensors for stablehlo backend";
  let constructor = "mlir::torch::TorchConversion::createFuncBackendTypeConversionForStablehloPass()";
  let description = [{
    Partial type conversion pass analogous in scope to the upstream
    `func-bufferize` pass. See details there.
  }];
}
#endif // TORCH_MLIR_ENABLE_STABLEHLO

def FinalizingBackendTypeConversion
    : InterfacePass<"torch-finalizing-backend-type-conversion", "mlir::FunctionOpInterface"> {
  let summary = "Finalizes a partial conversion to builtin tensors";
  let constructor =
    "mlir::torch::TorchConversion::createFinalizingBackendTypeConversionPass()";
  let description = [{
    Analogous in scope to the upstream `finalizing-bufferize` pass.
    See details there.
  }];
}

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
def FinalizingBackendTypeConversionForStablehlo
    : InterfacePass<"torch-finalizing-backend-type-conversion-for-stablehlo", "mlir::FunctionOpInterface"> {
  let summary = "Finalizes a partial conversion to builtin tensors for stablehlo";
  let constructor =
    "mlir::torch::TorchConversion::createFinalizingBackendTypeConversionForStablehloPass()";
  let description = [{
    Analogous in scope to the upstream `finalizing-bufferize` pass.
    See details there.
  }];
}
#endif // TORCH_MLIR_ENABLE_STABLEHLO

def VerifyLinalgOnTensorsBackendContract : Pass<"torch-verify-linalg-on-tensors-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the linalg-on-tensors backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyLinalgOnTensorsBackendContractPass()";
}

#ifdef TORCH_MLIR_ENABLE_TOSA
def VerifyTosaBackendContract : Pass<"torch-verify-tosa-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the linalg-on-tensors backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyTosaBackendContractPass()";
}
#endif

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
def VerifyStablehloBackendContract : Pass<"torch-verify-stablehlo-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the stablehlo backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyStablehloBackendContractPass()";
}
#endif // TORCH_MLIR_ENABLE_STABLEHLO

// The following passes are for a one-off conversion of a specific kind of quantized group matmul.
// They should not be included in default lowering flows until further along.
def UnpackQuantTensor : InterfacePass<"torch-unpack-quant-tensor", "mlir::FunctionOpInterface"> {
  let summary = "Unpack quantized int4 tensor from int8 containter";
  let constructor = "mlir::torch::TorchConversion::createUnpackQuantTensorPass()";
}

def ConvertCustomQuantOp : InterfacePass<"torch-convert-custom-quant-op", "mlir::FunctionOpInterface"> {
  let summary = "Convert torch custom quant op to linalg";
  let constructor = "mlir::torch::TorchConversion::createConvertCustomQuantOpPass()";
}

def LiftUserAttrs : Pass<"torch-lift-user-attrs", "ModuleOp"> {
  let summary = "Strip the `mlir.user.` prefix off user-annotation attrs";
  let constructor = "mlir::torch::TorchConversion::createLiftUserAttrsPass()";
  let description = [{
    Frontends emit user-supplied annotations (e.g. those produced from
    `node.meta["mlir.attrs"]` / `mlir.arg_attrs` by the FX importer) with a
    `mlir.user.` prefix on each attribute name, so they can be forwarded
    selectively through lowerings without leaking dialect-internal flags. This
    pass runs at the end of the backend pipelines and, for every op discardable
    attribute and every `func.func` per-argument attribute whose name starts
    with `mlir.user.`, re-attaches the value under the un-prefixed name and
    removes the prefixed entry. Non-prefixed attrs are left untouched.
    Downstream consumers see plain attrs like
    `{attr1=...}` as if they had been set directly.
  }];
}
#endif // TORCHMLIR_TORCHCONVERSION_PASSES
