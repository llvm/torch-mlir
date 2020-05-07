//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TCF_IR_TCFDIALECT_H
#define NPCOMP_DIALECT_TCF_IR_TCFDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace NPCOMP {
namespace tcf {

#include "npcomp/Dialect/TCF/IR/TCFOpsDialect.h.inc"

} // namespace tcf
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TCF_IR_TCFDIALECT_H
