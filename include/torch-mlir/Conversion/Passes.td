//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_PASSES
#define TORCHMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Common conversions
//===----------------------------------------------------------------------===//

def ConvertTorchToArith : Pass<"convert-torch-to-arith", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to Std ops";
  let constructor = "mlir::torch::createConvertTorchToArithPass()";
}

def ConvertTorchToSCF: Pass<"convert-torch-to-scf", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to SCF ops";
  let constructor = "mlir::torch::createConvertTorchToSCFPass()";
}

#endif // TORCHMLIR_CONVERSION_PASSES
