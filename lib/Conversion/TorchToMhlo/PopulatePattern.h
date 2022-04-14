//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace torch {
namespace torch_to_mhlo {

void populateBasicOpPatternsAndLegality(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        ConversionTarget &target);

void populateCustomOpPatternsAndLegality(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target);

void populateReductionOpPatternsAndLegality(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target);
void populateLinearOpPatternsAndLegality(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target);
void populatePoolingOpPatternsAndLegality(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target);

} // namespace torch_to_mhlo
} // namespace torch
} // namespace mlir
