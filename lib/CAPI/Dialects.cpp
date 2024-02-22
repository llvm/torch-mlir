//===- Dialects.cpp - C Interface for Dialects ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-c/Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Torch, torch,
                                      mlir::torch::Torch::TorchDialect)
