//===- ATenToStd.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
// import patterns
#include "npcomp/Dialect/ATen/Transforms/ATenToStd.cpp.inc"
} // namespace

namespace mlir {
void populateATenToStdPatterns(RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}
} // namespace mlir
