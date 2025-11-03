//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_PASSES
#define TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTorchOnnxToTorch : Pass<"convert-torch-onnx-to-torch", "func::FuncOp"> {
    let summary = "Converts ONNX custom ops in the torch dialect to native torch ops";
    let description = [{
      Converts equivalent ONNX custom ops to built-in equivalents.

      See the README for a detailed description of how this operates.
    }];

    let constructor = "mlir::torch::onnx_c::createTorchOnnxToTorchPass()";
    let options = [
    Option<"supportsNonFinites", "supports-non-finites",
            "bool", /*default=*/"true",
            "When enabled emits non-finites, such as, inf. Otherwise emits representative float value, for example, realmax instead of inf.">,
  ];
}

#endif  // TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_PASSES
