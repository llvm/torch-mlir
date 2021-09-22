//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/TorchConversion/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "npcomp/Backend/Common/Passes.h"
#include "npcomp/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "npcomp/Conversion/TorchToSCF/TorchToSCF.h"
#include "npcomp/Conversion/TorchToStd/TorchToStd.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "npcomp/Dialect/TorchConversion/Transforms/Passes.h.inc"
} // end namespace

void mlir::NPCOMP::registerTorchConversionPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torchscript-to-npcomp-backend-pipeline",
      "Pipeline lowering torch object graph to npcomp backend format.",
      mlir::NPCOMP::TorchConversion::createTorchScriptToNpcompBackendPipeline);
}

void mlir::NPCOMP::TorchConversion::createTorchScriptToNpcompBackendPipeline(
    OpPassManager &pm, const Torch::TorchLoweringPipelineOptions &options) {

  // Conversion to the npcomp backend contract starts from the Torch backend
  // contract.
  Torch::createTorchScriptToTorchBackendPipeline(pm, options);

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

  // Finish the type conversion from `torch` types to the types of the npcomp
  // backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that npcomp backends expect.
  // This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(CommonBackend::createVerifyBackendContractPass());
}
