//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "transforms/passes.h"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/TorchToStablehlo/Passes.h.inc"
} // end namespace

void mlir::torch::registerStablehloConversionPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<StablehloBackendPipelineOptions>(
      "torch-backend-to-stablehlo-backend-pipeline",
      "Pipeline lowering torch backend contract to StableHLO backend "
      "contract.",
      createTorchBackendToStablehloBackendPipeline);
}

void mlir::torch::createTorchBackendToStablehloBackendPipeline(
    OpPassManager &pm, const StablehloBackendPipelineOptions &options) {
  // Generate Stablehlo ops.
  pm.addNestedPass<func::FuncOp>(createConvertTorchToStablehloPass(
      options.enableStaticShape, options.enableI32Index));
  // Lowering remained ops to Arith
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // StableHLO backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to Stablehlo and Chlo ops.
  pm.addPass(createVerifyStablehloBackendContractPass());
}
