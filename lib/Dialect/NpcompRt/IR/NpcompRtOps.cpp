//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/NpcompRt/IR/NpcompRtOps.h"
#include "mlir/IR/Builders.h"
#include "npcomp/Dialect/NpcompRt/IR/NpcompRtDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP::npcomp_rt;

namespace mlir {
namespace NPCOMP {
namespace npcomp_rt {
#define GET_OP_CLASSES
#include "npcomp/Dialect/NpcompRt/IR/NpcompRtOps.cpp.inc"
} // namespace npcomp_rt
} // namespace NPCOMP
} // namespace mlir
