//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_BACKENDTYPECONVERSION_H
#define TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_BACKENDTYPECONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace torch {
namespace TorchConversion {

/// Get the dependent dialects which might be involved in a backend type
/// conversion.
void getBackendTypeConversionDependentDialects(DialectRegistry &registry);

/// Set up the provided ConversionTarget and TypeConverter for converting
/// from `torch` dialect types to the types along the linalg-on-tensors backend
/// boundary (which currently consist only of builtin types).
void setupBackendTypeConversion(ConversionTarget &target,
                                TypeConverter &typeConverter);

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
void setupBackendTypeConversionForStablehlo(ConversionTarget &target,
                                            TypeConverter &typeConverter);
#endif
} // namespace TorchConversion
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_BACKENDTYPECONVERSION_H
