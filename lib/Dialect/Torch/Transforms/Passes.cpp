//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "npcomp/Backend/Common/Passes.h"
#include "npcomp/Conversion/ATenToLinalg/ATenToLinalg.h"
#include "npcomp/Conversion/ATenToTCF/Passes.h"
#include "npcomp/Conversion/TCFToStd/TCFToStd.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"
#include "npcomp/Dialect/Numpy/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "npcomp/Dialect/Torch/Transforms/Passes.h.inc"
} // end namespace

void mlir::NPCOMP::registerTorchPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<>(
      "torchscript-to-npcomp-backend-pipeline",
      "Pipeline lowering torch object graph to npcomp backend format.",
      mlir::NPCOMP::Torch::createLowerObjectGraphPipeline);
  mlir::PassPipelineRegistration<>(
      "torch-globalized-module-to-npcomp-backend-pipeline",
      "Pipeline lowering to npcomp backend form.",
      mlir::NPCOMP::Torch::createLowerToNpcompBackendPipeline);
}

void mlir::NPCOMP::Torch::createLowerObjectGraphPipeline(OpPassManager &pm) {
  // When we import TorchScript IR, we import their entire "compilation unit",
  // which can contain numerous functions unrelated to the current program,
  // which breaks torch-globalization-pipeline; for example, there can be
  // random functions referencing types that haven't been imported
  // as part of the root `torch.nn.Module` we imported. Those will
  // be unreferenced private functions which symbol-dce will clean up nicely.
  pm.addPass(createSymbolDCEPass());
  // Globalize the program. The rest of the compiler assumes a globalized
  // program, which makes all analyses and transforms significantly easier
  // to write.
  pm.addPass(createPrepareForGlobalizeObjectGraphPass());
  pm.addPass(createGlobalizeObjectGraphPass());
  // "lower" `torch.global_slot` ops by deleting them if unused, which we
  // currently require because we don't have a lowering path for backends to
  // handle them.
  // Torch usually inserts a few unused global slots so this ends up hitting
  // every single module even if it doesn't have any explicit slots.
  // TODO: Support global slots in backends.
  pm.addPass(createSymbolDCEPass());
  // Currently, our shape inference is not powerful enough to deal with
  // calls, so inline everything.
  // TODO: Improve shape inference.
  pm.addPass(createInlinerPass());
  // Incorporate user annotations and remove signature Python-isms.
  pm.addPass(createAdjustCallingConventionsPass());

  createLowerToNpcompBackendPipeline(pm);
}

void mlir::NPCOMP::Torch::createLowerToNpcompBackendPipeline(
    OpPassManager &pm) {
  // Recognize ATen kernels.
  pm.addNestedPass<FuncOp>(aten::createRecognizeKernelsPass());

  // Convert the bulk of the program to ranked tensors with known dtype.
  // This is the input to the backend layer that we are aiming for.

  // First, unilaterally convert public functions to tensor.
  // The way this pass is currently written, this implies that
  // as pipeline authors, we are restricting our users to not be able to see
  // updates to "out params" on their public functions.
  // This is deemed ok for now.
  pm.addPass(Numpy::createPublicFunctionsToTensorPass());
  // Inline global slots, which for most inference scenarios deletes them.
  // This also exposes more information to intraprocedural transformations
  // below like ArrayToTensor and RefineTypes.
  // TODO: Don't rely on this pass to "lower" global slots by deleting.
  // This pass should eventually be "just an optimization".
  pm.addPass(createInlineGlobalSlotsPass());
  // Convert the bulk of non-ABI-visible arrays to tensors.
  pm.addNestedPass<FuncOp>(Numpy::createArrayToTensorPass());
  // Do shape and dtype refinement.
  // We could do it sooner, but the pass currently doesn't have transfer
  // functions for array ops.
  pm.addNestedPass<FuncOp>(Torch::createRefineTypesPass());
  // Propagate to ABI return types the shape/dtype information discovered by
  // the previous pass. Doing this is ABI-compatible for our backends.
  pm.addPass(Numpy::createRefinePublicReturnPass());
  // Clean up a few stray array/tensor conversion remnants.
  pm.addNestedPass<FuncOp>(Numpy::createArrayToTensorPass());

  // Lower to TCP (+ guards) which is the input to codegen backends.
  // Most of this should be subsumed by aten->linalg+guards conversions.
  // (the guard generation will be automated from the linalg Op DSL).
  pm.addNestedPass<FuncOp>(createConvertATenToLinalgPass());
  pm.addNestedPass<FuncOp>(createConvertATenToTCFPass());
  pm.addNestedPass<FuncOp>(createConvertTCFToStdPass());
  pm.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());

  // Verify that we have lowered to the form that backends expect.
  // This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(CommonBackend::createVerifyBackendContractPass());
}
