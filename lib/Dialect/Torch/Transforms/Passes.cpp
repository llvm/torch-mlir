//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h.inc"
} // end namespace

void mlir::torch::registerTorchPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torchscript-module-to-torch-backend-pipeline",
      "Pipeline lowering TorchScript object graph IR to Torch backend form.",
      mlir::torch::Torch::createTorchScriptModuleToTorchBackendPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-function-to-torch-backend-pipeline",
      "Pipeline lowering a Torch function to Torch backend form.",
      mlir::torch::Torch::createTorchFunctionToTorchBackendPipeline);
}

void mlir::torch::Torch::createTorchScriptModuleToTorchBackendPipeline(
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

  createTorchFunctionToTorchBackendPipeline(pm, options);
}

void mlir::torch::Torch::createTorchFunctionToTorchBackendPipeline(
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

  // Incorporate user annotations and remove signature Python-isms.
  pm.addPass(createAdjustCallingConventionsPass());

  if (options.optimize) {
    // Eliminate the PrimTupleIndexOp generated from the
    // adjustCallingConventions
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    // Inline global slots, which for most inference scenarios deletes them.
    // This also exposes more information to intraprocedural transformations
    // below like MaximizeValueSemantics and RefineTypes.
    // OPT-ONLY: Don't rely on this pass to "lower" global slots by deleting.
    // Also don't rely on this pass to expose constants into the program to
    // simplify handling of "optional".
    pm.addPass(createInlineGlobalSlotsPass());
  }

  // Reduce variants of ops to a smaller set of primitives.
  pm.addNestedPass<FuncOp>(createReduceOpVariantsPass());

  if (options.optimize) {
    // OPT-ONLY: Right now we rely on this to eliminate certain branches that
    // guard unreachable code that backends can't handle yet, such as lists,
    // RaiseException, unimplemented tensor ops, and only-used-in-training
    // operations on `torch.global_slot`'s.
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    // OPT-ONLY: We may have deleted some `torch.global_slot.get` /
    // `torch.global_slot.get` ops, which may have left more
    // `torch.global_slot`'s unused.
    pm.addPass(createSymbolDCEPass());
  }

  //===--------------------------------------------------------------------===//
  // Lowering to ranked !torch.vtensors of known dtype.
  //===--------------------------------------------------------------------===//

  // Do shape and dtype refinement.
  pm.addNestedPass<FuncOp>(Torch::createRefineTypesPass());

  // Propagate to ABI return types the shape/dtype information discovered by
  // the previous pass. Doing this is ABI-compatible for our backends.
  pm.addPass(Torch::createRefinePublicReturnPass());

  if (options.optimize) {
    // This can fold away some branches given the information got from
    // RefineTypes before doing maximize value sematics which only works with
    // basic blocks.
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  }
  // Convert the bulk of non-ABI-visible !torch.tensor's to !torch.vtensor's.
  pm.addNestedPass<FuncOp>(Torch::createMaximizeValueSemanticsPass());

  if (options.optimize) {
    // All the type refinement we've done above has exposed new information
    // that allows folding away more stuff.
    // OPT-ONLY: Right now we rely on this to eliminate certain
    // branches that guard unreachable code that backends can't handle yet, such
    // as lists, RaiseException, unimplemented aten ops, and
    // only-used-in-training operations on `torch.global_slot`'s.
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  }
  pm.addNestedPass<FuncOp>(Torch::createDecomposeComplexOpsPass());

  // TODO: VerifyTorchBackendContractPass.
}
