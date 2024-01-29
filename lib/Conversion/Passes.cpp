//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#endif // TORCH_MLIR_ENABLE_STABLEHLO

#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/Passes.h.inc"
} // end namespace

void mlir::torch::registerConversionPasses() { ::registerPasses(); }
