//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHCONVERSIONTOMLPROGRAM_TORCHCONVERSIONTOMLPROGRAM_H
#define TORCHMLIR_CONVERSION_TORCHCONVERSIONTOMLPROGRAM_TORCHCONVERSIONTOMLPROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace torch {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTorchConversionToMLProgramPass();
}
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHCONVERSIONTOMLPROGRAM_TORCHCONVERSIONTOMLPROGRAM_H
