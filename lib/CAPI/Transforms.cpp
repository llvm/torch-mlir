//===- CAPIPasses.cpp - C API for Transformations Passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Pass.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

// Must include the declarations as they carry important visibility attributes.
#include "torch-mlir/Dialect/Torch/Transforms/Transforms.capi.h.inc"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

#ifdef __cplusplus
extern "C" {
#endif

#include "torch-mlir/Dialect/Torch/Transforms/Transforms.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
