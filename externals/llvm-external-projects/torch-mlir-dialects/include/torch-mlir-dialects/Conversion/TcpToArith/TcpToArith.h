//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_CONVERSION_TCPTOARITH_TCPTOARITH_H_
#define TORCH_MLIR_DIALECTS_CONVERSION_TCPTOARITH_TCPTOARITH_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTTCPTOARITH
#include "torch-mlir-dialects/Conversion/Passes.h.inc"

namespace tcp {

std::unique_ptr<Pass> createConvertTcpToArithPass();

} // namespace tcp
} // namespace mlir

#endif // TORCH_MLIR_DIALECTS_CONVERSION_TCPTOARITH_TCPTOARITH_H_
