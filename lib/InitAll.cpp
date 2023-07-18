//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/InitAll.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/RefBackend/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "mhlo/transforms/passes.h"
#endif

void mlir::torch::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();
  registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
  registry.insert<mlir::torch::TMTensor::TMTensorDialect>();
  mlir::func::registerInlinerExtension(registry);
}

void mlir::torch::registerAllPasses() {
  mlir::torch::registerTorchPasses();
  mlir::torch::registerTorchConversionPasses();

  mlir::torch::registerConversionPasses();
  mlir::torch::RefBackend::registerRefBackendPasses();
  mlir::torch::TMTensor::registerPasses();

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  mlir::mhlo::registerSymbolicShapeOptimizationPass();
  mlir::mhlo::registerStablehloLegalizeToHloPass();
  mlir::mhlo::registerChloLegalizeToHloPass();
  mlir::mhlo::registerHloLegalizeToLinalgPass();
  mlir::mhlo::registerTestUnfuseBatchNormPass();
#endif // TORCH_MLIR_ENABLE_STABLEHLO
}
