//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_LIB_CONVERSION_TORCHTOMHLO_POPULATEPATTERNS_H
#define TORCHMLIR_LIB_CONVERSION_TORCHTOMHLO_POPULATEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace torch {
namespace torch_to_mhlo {

void populateBasicOpPatternsAndLegality(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        ConversionTarget &target);
void populateViewLikeOpPatternsAndLegality(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           ConversionTarget &target);
void populateGatherOpPatternsAndLegality(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target);
} // namespace torch_to_mhlo
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_LIB_CONVERSION_TORCHTOMHLO_POPULATEPATTERNS_H
