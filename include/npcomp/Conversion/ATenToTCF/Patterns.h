//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_CONVERSION_ATENTOTCF_PATTERNS_H
#define NPCOMP_CONVERSION_ATENTOTCF_PATTERNS_H

#include <memory>

namespace mlir {

class MLIRContext;
class OwningRewritePatternList;

namespace NPCOMP {

/// Populates patterns for converting core ATen ops to TCF. These patterns
/// cover core arithmetic ops that are on the order of 1:1 representationally.
/// More advanced patterns are managed elsewhere.
void populateCoreATenToTCFPatterns(MLIRContext *context,
                                   OwningRewritePatternList &patterns);

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_CONVERSION_ATENTOTCF_PATTERNS_H
