//===- PassDetail.h - RefBackend Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef REFBACKEND_PASSDETAIL_H
#define REFBACKEND_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace torch {

#define GEN_PASS_CLASSES
#include "torch-mlir/RefBackend/Passes.h.inc"

} // namespace torch
} // end namespace mlir

#endif // REFBACKEND_PASSDETAIL_H
