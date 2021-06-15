//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCH_TRANSFORMS_BACKENDTYPECONVERSION_H
#define NPCOMP_DIALECT_TORCH_TRANSFORMS_BACKENDTYPECONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace NPCOMP {
namespace Torch {
/// Set up the provided ConversionTarget and TypeConverter for converting
/// from `torch` dialect types to the types along the npcomp backend boundary
/// (which currently consist only of builtin types).
void setupBackendTypeConversion(ConversionTarget &target,
                                TypeConverter &typeConverter);
} // namespace Torch
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TORCH_TRANSFORMS_BACKENDTYPECONVERSION_H
