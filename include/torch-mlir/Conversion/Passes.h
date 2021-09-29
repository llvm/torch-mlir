//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_PASSES_H
#define TORCHMLIR_CONVERSION_PASSES_H

namespace mlir {
namespace torch {

/// Registers all torch-mlir conversion passes.
void registerConversionPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_PASSES_H
