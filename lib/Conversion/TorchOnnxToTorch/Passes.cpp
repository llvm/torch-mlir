//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h.inc"
} // end namespace

void mlir::torch::onnx_c::registerTorchOnnxToTorchPasses() {
  ::registerPasses();
}
