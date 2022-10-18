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
namespace torch_to_tcp {

Value broadcastRankInLeadingDims(ConversionPatternRewriter &rewriter,
                                 Value input, int64_t rankIncrease);

Value broadcastShapeInLeadingDims(ConversionPatternRewriter &rewriter,
                                  Value input, Value target,
                                  int64_t numLeadingAxes);

} // namespace torch_to_tcp
} // namespace mlir
