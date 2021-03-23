//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_CONVERSION_BASICPYTOSTD_PATTERNS_H
#define NPCOMP_CONVERSION_BASICPYTOSTD_PATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir {
namespace NPCOMP {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

void populateBasicpyToStdPrimitiveOpPatterns(RewritePatternSet &patterns);

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_CONVERSION_BASICPYTOSTD_PATTERNS_H
