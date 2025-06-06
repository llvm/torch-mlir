//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "stablehlo/transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#endif

#ifdef TORCH_MLIR_ENABLE_TOSA
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
using namespace mlir::tosa;
#endif

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace reg {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"
} // end namespace reg

void mlir::torch::registerTorchConversionPasses() {
  reg::registerPasses();
  mlir::PassPipelineRegistration<>(
      "torch-backend-to-linalg-on-tensors-backend-pipeline",
      "Pipeline lowering torch backend contract to linalg-on-tensors backend "
      "contract.",
      TorchConversion::createTorchBackendToLinalgOnTensorsBackendPipeline);
#ifdef TORCH_MLIR_ENABLE_TOSA
  mlir::PassPipelineRegistration<TorchConversion::TosaBackendPipelineOptions>(
      "torch-backend-to-tosa-backend-pipeline",
      "Pipeline lowering torch backend contract to TOSA backend "
      "contract.",
      TorchConversion::createTorchBackendToTosaBackendPipeline);
#endif
#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  mlir::PassPipelineRegistration<
      TorchConversion::StablehloBackendPipelineOptions>(
      "torch-backend-to-stablehlo-backend-pipeline",
      "Pipeline lowering torch backend contract to StableHLO backend "
      "contract.",
      TorchConversion::createTorchBackendToStablehloBackendPipeline);
#endif
}

void TorchConversion::createTorchBackendToLinalgOnTensorsBackendPipeline(
    OpPassManager &pm) {
  // Fix non constant dims passed to reduction ops
  pm.addNestedPass<func::FuncOp>(
      torch::Torch::createRestructureNonConstantAxesPass());

  // We want to fuse quantized operations together before lowering to linalg.
  pm.addNestedPass<func::FuncOp>(Torch::createFuseQuantizedOpsPass());

  // Lower to linalg + guards which is the input to codegen backends.
  // We do this first as it tends to involve pattern-matching against constants,
  // (e.g. dimensions which must be constant in a ranked programming model)
  // and those constants get somewhat obscured by TorchToArith.
  pm.addNestedPass<func::FuncOp>(createConvertTorchToTMTensorPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToTensorPass());
  pm.addPass(createConvertTorchConversionToMLProgramPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Resolve `dim` ops on tensors (which currently live in the `memref`
  // dialect for some reason -- we don't have memrefs at this level).
  pm.addNestedPass<func::FuncOp>(
      memref::createResolveShapedTypeResultDimsPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // linalg-on-tensors backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that linalg on tensors backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(TorchConversion::createVerifyLinalgOnTensorsBackendContractPass());
}

#ifdef TORCH_MLIR_ENABLE_TOSA
void TorchConversion::createTorchBackendToTosaBackendPipeline(
    OpPassManager &pm,
    const TorchConversion::TosaBackendPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(
      createConvertTorchToTosaPass(options.requireFullTosaConversion));
  // Fold full-layer operations on TOSA constants
  pm.addNestedPass<func::FuncOp>(createTosaLayerwiseConstantFoldPass());

  // Perform transpose reductions for avoidable data movements
  pm.addNestedPass<func::FuncOp>(createTosaReduceTransposes());

  // Perform rank broadcasting so TosaToLinalg pass works
  pm.addNestedPass<func::FuncOp>(createTosaMakeBroadcastablePass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // TOSA backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that TOSA backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(TorchConversion::createVerifyTosaBackendContractPass());
}
#endif

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
void TorchConversion::createTorchBackendToStablehloBackendPipeline(
    OpPassManager &pm,
    const TorchConversion::StablehloBackendPipelineOptions &options) {
  // Generate Stablehlo & Chlo ops.
  pm.addNestedPass<func::FuncOp>(createConvertTorchToStablehloPass(
      options.enableStaticShape, options.enableI32Index));
  // Lowering Chlo ops to Stablehlo
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createChloLegalizeToStablehloPass());
  // Lowering remained ops to Arith
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // StableHLO backend contract.
  pm.addPass(
      TorchConversion::createFuncBackendTypeConversionForStablehloPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionForStablehloPass());

  // Verify that we have lowered to Stablehlo ops.
  pm.addPass(TorchConversion::createVerifyStablehloBackendContractPass());

  // Canonicalize Stablehlo dynamic ops to static ops
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(stablehlo::createStablehloRefineShapesPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Legalize deprecated ops to Stablehlo ops
  stablehlo::StablehloLegalizeDeprecatedOpsPassOptions stablehloOptions;
  stablehloOptions.failOnUnusedOps = false;
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloLegalizeDeprecatedOpsPass(stablehloOptions));
  pm.addPass(createCanonicalizerPass());
}
#endif
