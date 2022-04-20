//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace torch {
namespace TMTensor {

namespace detail {
#define GEN_PASS_REGISTRATION
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace detail

} // namespace TMTensor
} // namespace torch
} // namespace mlir

void torch::TMTensor::registerPasses() {
  torch::TMTensor::detail::registerPasses();
}
