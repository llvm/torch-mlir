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
namespace torch_to_linalg {

// -----------------------------------------------------------------------------
// TorchToLinalg conversion patterns
// -----------------------------------------------------------------------------
//
// Most of these patterns consist of:
// 1. Checking that the operand/result types and other static properties are
//    good-enough to create a valid linalg op (such as operands being of
//    ranks/dtypes acceptable to the linalg op).
// 2. Creating dynamic error guards, usually checking a predicate on the
//    compatibility of operand shapes.
// 3. Creating init tensors for the computation op. Usually this involves
//    reifying IR for a shape transfer function based on the operand shapes.
// 4. Creating a named linalg op to replace the original op.
//
// TODO: This should be more automated, and ideally written as fully-executable
// Python code that is tested for correctness at the Python level and then
// just serialized (e.g. as a TorchScript function) and inlined to provide
// the implementation of the op.
// TODO: Linalg is not a stable format. As much as possible of this should
// lower through a more stable format, such as TOSA. However, TOSA currently
// lacks enough support for dynamic shapes and error assertions to be used
// for this purpose.

void populateTensorScalarInteropPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target);
void populateLinearPatternsAndLegality(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       ConversionTarget &target);
void populatePoolingPatternsAndLegality(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        ConversionTarget &target);
void populateRandomPatternsAndLegality(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       ConversionTarget &target);
void populateUncategorizedPatternsAndLegality(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns,
                                              ConversionTarget &target);
void populateReductionPatternsAndLegality(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target);
void populateDataMovementPatternsAndLegality(TypeConverter &typeConverter,
                                             RewritePatternSet &patterns,
                                             ConversionTarget &target);
void populateIndirectDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target);
void populateTensorConstructorsPatternsAndLegality(TypeConverter &typeConverter,
                                                   RewritePatternSet &patterns,
                                                   ConversionTarget &target);
void populateTorchvisionPatternsAndLegality(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            ConversionTarget &target);

} // namespace torch_to_linalg
} // namespace torch
} // namespace mlir
