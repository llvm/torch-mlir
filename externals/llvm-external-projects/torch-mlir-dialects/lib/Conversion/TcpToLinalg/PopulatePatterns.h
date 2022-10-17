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
namespace tcp_to_linalg {

void populateElementwisePatternsAndLegality(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            ConversionTarget &target);
void populateMiscPatternsAndLegality(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target);

} // namespace tcp_to_linalg
} // namespace mlir