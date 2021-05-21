//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCH_IR_TORCHUTILS_H
#define NPCOMP_DIALECT_TORCH_IR_TORCHUTILS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace NPCOMP {
namespace Torch {
void setupValueTensorToBuiltinTensorConversion(ConversionTarget &target,
                                               TypeConverter &typeConverter);
}
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TORCH_IR_TORCHUTILS_H
