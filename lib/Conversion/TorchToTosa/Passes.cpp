//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/Passes.h"

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::tosa;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/TorchToTosa/Passes.h.inc"
} // end namespace

void mlir::torch::registerTosaConversionPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<>(
      "torch-backend-to-tosa-backend-pipeline",
      "Pipeline lowering torch backend contract to TOSA backend "
      "contract.",
      createTorchBackendToTosaBackendPipeline);
}

void mlir::torch::createTorchBackendToTosaBackendPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvertTorchToTosaPass());
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
  pm.addPass(createVerifyTosaBackendContractPass());
}
