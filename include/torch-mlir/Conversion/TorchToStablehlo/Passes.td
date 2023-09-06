//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_STABLEHLO_PASSES
#define TORCHMLIR_CONVERSION_STABLEHLO_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Torch conversions
//===----------------------------------------------------------------------===//

def ConvertTorchToStablehlo : Pass<"convert-torch-to-stablehlo", "func::FuncOp"> {
  let summary = "Convert Torch ops to Stablehlo ops";
  let description = [{
    Convert Torch ops to Stablehlo ops.
  }];
  let constructor = "mlir::torch::createConvertTorchToStablehloPass()";

  // Specify any options.
  let options = [
    Option<"enableStaticShape", "enable-static-shape", "bool", /*default=*/"false",
           "Enable static shape conversion">,
    // The i64 calculation is much slower than i32 on some devices, such as
    // Nvidia GPU. One can truncate from i64 to i32 since dimension sizes
    // are unlikely to exceed the range of i32(4GiB)
    Option<"enableI32Index", "enable-i32-index", "bool", /*default=*/"false",
           "Enable truncate index from i64 to i32(unsafely)">,
  ];
}

def VerifyStablehloBackendContract : Pass<"torch-verify-stablehlo-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the stablehlo backend contract";
  let constructor = "mlir::torch::createVerifyStablehloBackendContractPass()";
}

#endif // TORCHMLIR_CONVERSION_STABLEHLO_PASSES
