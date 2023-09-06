//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/Passes.h"

#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/TorchToLinalg/Passes.h.inc"
} // end namespace

void mlir::torch::registerLinalgConversionPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<>(
      "torch-backend-to-linalg-on-tensors-backend-pipeline",
      "Pipeline lowering torch backend contract to linalg-on-tensors backend "
      "contract.",
      createTorchBackendToLinalgOnTensorsBackendPipeline);
}

void mlir::torch::createTorchBackendToLinalgOnTensorsBackendPipeline(
    OpPassManager &pm) {
  // Lower to linalg + guards which is the input to codegen backends.
  // We do this first as it tends to involve pattern-matching against constants,
  // (e.g. dimensions which must be constant in a ranked programming model)
  // and those constants get somewhat obscured by TorchToArith.
  pm.addNestedPass<func::FuncOp>(createConvertTorchToTMTensorPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());
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
  pm.addPass(createVerifyLinalgOnTensorsBackendContractPass());
}
