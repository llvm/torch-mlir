//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToStd/TorchToStd.h"
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"
} // end namespace

void mlir::torch::registerTorchConversionPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-backend-to-linalg-on-tensors-backend-pipeline",
      "Pipeline lowering torch backend contract to linalg-on-tensors backend "
      "contract.",
      TorchConversion::createTorchBackendToLinalgOnTensorsBackendPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-backend-to-tosa-backend-pipeline",
      "Pipeline lowering torch backend contract to TOSA backend "
      "contract.",
      TorchConversion::createTorchBackendToTosaBackendPipeline);
}

void TorchConversion::createTorchBackendToLinalgOnTensorsBackendPipeline(
    OpPassManager &pm, const Torch::TorchLoweringPipelineOptions &options) {
  // Check some invariants to catch errors in a clear way.
  pm.addPass(
      TorchConversion::createVerifyInvariantsBeforeBackendLoweringPass());

  // Lower to linalg + guards which is the input to codegen backends.
  // We do this first as it tends to involve pattern-matching against constants,
  // (e.g. dimensions which must be constant in a ranked programming model)
  // and those constants get somewhat obscured by TorchToStd.
  pm.addNestedPass<FuncOp>(createConvertTorchToLinalgPass());
  pm.addNestedPass<FuncOp>(createConvertTorchToStdPass());
  pm.addNestedPass<FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<FuncOp>(createStdExpandOpsPass());

  if (options.optimize) {
    // Clean up any non-canonical code introduced above..
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    // Resolve `dim` ops on tensors (which currently live in the `memref`
    // dialect for some reason -- we don't have memrefs at this level).
    pm.addNestedPass<FuncOp>(memref::createResolveShapedTypeResultDimsPass());
    // The resolution of `dim` ops tends to create identical ops. CSE them.
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // Finish the type conversion from `torch` types to the types of the
  // linalg-on-tensors backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that linalg on tensors backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(TorchConversion::createVerifyLinalgOnTensorsBackendContractPass());
}

void TorchConversion::createTorchBackendToTosaBackendPipeline(
    OpPassManager &pm, const Torch::TorchLoweringPipelineOptions &options) {
  // Check some invariants to catch errors in a clear way.
  pm.addPass(
      TorchConversion::createVerifyInvariantsBeforeBackendLoweringPass());

  pm.addNestedPass<FuncOp>(createConvertTorchToTosaPass());

  if (options.optimize) {
    // Clean up any non-canonical code introduced above..
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    // The resolution of `dim` ops tends to create identical ops. CSE them.
    pm.addNestedPass<FuncOp>(createCSEPass());
  }

  // Finish the type conversion from `torch` types to the types of the
  // TOSA backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that TOSA backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(TorchConversion::createVerifyTosaBackendContractPass());
}
