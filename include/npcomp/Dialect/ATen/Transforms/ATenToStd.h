//===- ATenToStd.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_TO_STD_H
#define NPCOMP_DIALECT_ATEN_TO_STD_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace NPCOMP {
namespace aten {

class ATenDialect;

} // namespace aten
} // namespace NPCOMP
} // namespace mlir

namespace mlir {

void populateATenToStdPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // NPCOMP_DIALECT_ATEN_TO_STD_H
