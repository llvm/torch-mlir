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
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torchscript-to-npcomp-backend-pipeline",
      "Pipeline lowering torch object graph to npcomp backend format.",
      mlir::NPCOMP::Torch::createLowerObjectGraphPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-globalized-module-to-npcomp-backend-pipeline",
      "Pipeline lowering to npcomp backend form.",
      mlir::NPCOMP::Torch::createLowerToNpcompBackendPipeline);
}

void mlir::NPCOMP::Torch::createLowerObjectGraphPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
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

  createLowerToNpcompBackendPipeline(pm, options);
}

void mlir::NPCOMP::Torch::createLowerToNpcompBackendPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  // General considerations: As a matter of bring-up, we are simultaneously
  // building out the frontend pipeline and also co-developing the backend
  // support story as well. This means that sometimes the most expedient way to
  // support a given program is to "optimize hard enough" that the parts of the
  // program that touch unimplemented backend support go away (constant folded,
  // dead-code-eliminated, etc.). In the fullness of time, most of that
  // optimization should not be necessary, and we should have an "O0" pipeline
  // that runs practically no optimizations.
  // However, as a matter of expediency, at the moment we do run those
  // optimizations. We guard those passes under the `options.optimize` option
  // (which default to true, currently). We leave notes with the `OPT-ONLY` tag
  // why we currently need that pass for correctness.
  // We should eventually remove those passes from the default pipeline once
  // backends have enough support.
  // In particular the following features are needed in some form from backends:
  // - Error handling (RaiseException + error string formatting)
  // - First-class list type
  // - torch.global_slot lowering
  // - ...
  // Please try to keep this list somewhat up to date when adding
  // "optimize hard enough that it works" transformations.

  if (options.optimize) {
    // Inline global slots, which for most inference scenarios deletes them.
    // This also exposes more information to intraprocedural transformations
    // below like ArrayToTensor and RefineTypes.
    // OPT-ONLY: Don't rely on this pass to "lower" global slots by deleting.
    // Also don't rely on this pass to expose constants into the program to
    // simplify handling of "optional".
    pm.addPass(createInlineGlobalSlotsPass());
  }

  // Recognize ATen kernels. This is a totally local transformation that
  // we want to run as soon as possible.
  pm.addNestedPass<FuncOp>(aten::createRecognizeKernelsPass());

  if (options.optimize) {
    // OPT-ONLY: Right now we rely on this to eliminate certain branches that
    // guard unreachable code that backends can't handle yet, such as lists,
    // RaiseException, unimplemented aten ops, and only-used-in-training
    // operations on `torch.global_slot`'s.
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    // OPT-ONLY: We may have deleted some `torch.global_slot.get` /
    // `torch.global_slot.get` ops, which may have left more
    // `torch.global_slot`'s unused.
    pm.addPass(createSymbolDCEPass());
  }

  // Convert the bulk of the program to ranked tensors with known dtype.
  // This is the input to the backend layer that we are aiming for.

  // First, unilaterally convert public functions to tensor.
  // The way this pass is currently written, this implies that
  // as pipeline authors, we are restricting our users to not be able to see
  // updates to "out params" on their public functions.
  // This is deemed ok for now.
  pm.addPass(Numpy::createPublicFunctionsToTensorPass());
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

  if (options.optimize) {
    // RefineTypes has exposed new type information that allows folding away
    // more stuff. OPT-ONLY: Right now we rely on this to eliminate certain
    // branches that guard unreachable code that backends can't handle yet, such
    // as lists, RaiseException, unimplemented aten ops, and
    // only-used-in-training operations on `torch.global_slot`'s.
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  }

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
