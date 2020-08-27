//===- ATenToStd.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/ATenToStd.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "npcomp/Dialect/ATen/ATenDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
// import patterns
#include "npcomp/Dialect/ATen/ATenToStd.cpp.inc"
} // namespace

namespace mlir {
void populateATenToStdPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns) {
  populateWithGenerated(context, &patterns);
}
} // namespace mlir
