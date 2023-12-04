//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_PASSDETAIL_H
#define TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::torch::onnx_c {

#define GEN_PASS_CLASSES
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h.inc"

} // namespace mlir::torch::onnx_c

#endif // TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_PASSDETAIL_H
