//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/Passes.h"

#ifdef TORCH_MLIR_DIALECTS_ENABLE_TCP
#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "torch-mlir-dialects/Conversion/StablehloToTcp/StablehloToTcp.h"
#endif // TORCH_MLIR_ENABLE_STABLEHLO
#include "torch-mlir-dialects/Conversion/TcpToArith/TcpToArith.h"
#include "torch-mlir-dialects/Conversion/TcpToLinalg/TcpToLinalg.h"
#endif // TORCH_MLIR_DIALECTS_ENABLE_TCP

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

#ifdef TORCH_MLIR_DIALECTS_ENABLE_TCP
namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir-dialects/Conversion/Passes.h.inc"
} // end namespace
#endif // TORCH_MLIR_DIALECTS_ENABLE_TCP

void mlir::torch_mlir_dialects::registerConversionPasses() {
#ifdef TORCH_MLIR_DIALECTS_ENABLE_TCP
  ::registerPasses();
#endif // TORCH_MLIR_DIALECTS_ENABLE_TCP
}
