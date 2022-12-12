//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/Passes.h"

#ifdef TORCH_MLIR_ENABLE_MHLO
#include "mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"
#endif // TORCH_MLIR_ENABLE_MHLO
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"
#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"
#ifdef TORCH_MLIR_ENABLE_TCP
#include "torch-mlir-dialects/Conversion/TcpToLinalg/TcpToLinalg.h"
#include "torch-mlir/Conversion/TorchToTcp/TorchToTcp.h"
#endif // TORCH_MLIR_ENABLE_TCP

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/Passes.h.inc"
} // end namespace

void mlir::torch::registerConversionPasses() {
  ::registerPasses();
#ifdef TORCH_MLIR_ENABLE_MHLO
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::mhlo::createLegalizeHloToLinalgPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createSymbolicShapeOptimizationPass();
  });
#endif // TORCH_MLIR_ENABLE_MHLO
#if TORCH_MLIR_ENABLE_TCP
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::tcp::createConvertTcpToLinalgPass();
  });
#endif // TORCH_MLIR_ENABLE_TCP
}
