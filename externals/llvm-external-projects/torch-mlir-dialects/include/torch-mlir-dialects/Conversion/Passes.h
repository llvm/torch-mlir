//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_CONVERSION_PASSES_H
#define TORCH_MLIR_DIALECTS_CONVERSION_PASSES_H

namespace mlir {
namespace torch_mlir_dialects {

void registerConversionPasses();

} // namespace torch_mlir_dialects
} // namespace mlir

#endif // TORCH_MLIR_DIALECTS_CONVERSION_PASSES_H
