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
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"

void mlir::torch::registerTorchPasses() {
  mlir::torch::registerPasses();
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torchscript-module-to-torch-backend-pipeline",
      "Pipeline lowering TorchScript object graph IR to Torch backend form.",
      mlir::torch::Torch::createTorchScriptModuleToTorchBackendPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torchdynamo-export-to-torch-backend-pipeline",
      "Pipeline lowering TorchDynamo exported graph IR to Torch backend form.",
      mlir::torch::Torch::createTorchDynamoExportToTorchBackendPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-function-to-torch-backend-pipeline",
      "Pipeline lowering a Torch function to Torch backend form.",
      mlir::torch::Torch::createTorchFunctionToTorchBackendPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-onnx-to-torch-backend-pipeline",
      "Pipeline lowering Torch Onnx IR to Torch backend form.",
      mlir::torch::Torch::createTorchOnnxToTorchBackendPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-simplification-pipeline",
      "Pipeline simplifying computations in the program.",
      mlir::torch::Torch::createTorchSimplificationPipeline);
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-shape-refinement-pipeline", "Pipeline refining shapes of tensors.",
      mlir::torch::Torch::createTorchShapeRefinementPipeline);
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

void mlir::torch::Torch::createTorchDynamoExportToTorchBackendPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  // Inline func.call operations created by higher-order ops like while_loop
  // to conform to the linalg-on-tensors backend contract.
  pm.addPass(createInlinerPass());
  pm.addNestedPass<func::FuncOp>(
      createReduceOpVariantsPass(options.extraLibrary));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  if (options.decompose) {
    pm.addNestedPass<func::FuncOp>(
        Torch::createDecomposeComplexOpsPass(options.backendLegalOps));
    pm.addNestedPass<func::FuncOp>(Torch::createRecomposeComplexOpsPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
}

void mlir::torch::Torch::createTorchFunctionToTorchBackendPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  // Incorporate user annotations and remove signature Python-isms.
  pm.addPass(createAdjustCallingConventionsPass());
  // Perform the bulk of lowering to the backend contract.
  // See the pass documentation for more information.
  pm.addPass(createLowerToBackendContractPass(
      options.maxIterations, options.decompose, options.shapeDtypeRefine,
      options.backendLegalOps, options.extraLibrary));
}

void mlir::torch::Torch::createTorchOnnxToTorchBackendPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(
      onnx_c::createTorchOnnxToTorchPass(options.allowNonFinites));
  // The above pass just converts the torch onnx IR to torch, hence the given
  // pipeline will make sure that the IR is transformed such that it satisfies
  // the backend contract.
  if (options.decompose) {
    pm.addNestedPass<func::FuncOp>(
        Torch::createDecomposeComplexOpsPass(options.backendLegalOps));
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
  // TODO: Move the combination of two passes i.e., ScalarizeShapes and
  // TorchShapeRefinementPipeline out of here and create an onnx shape
  // refinement pipeline which runs iteratively over the IR.
  createTorchShapeRefinementPipeline(pm, options);
  // This pass scalarizes the tensor shape computations.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::torch::Torch::createScalarizeShapesPass());
  createTorchShapeRefinementPipeline(pm, options);
  pm.addPass(Torch::createRefinePublicReturnPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The decompose pass is run again here since the scalarize shapes pass and
  // shape refinement pipeline might create some ops for which decomposition
  // exists.
  if (options.decompose) {
    pm.addNestedPass<func::FuncOp>(
        Torch::createDecomposeComplexOpsPass(options.backendLegalOps));
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
}

// A simplification pipeline to establish the invariants of the backend
// contract (see `satisfiedBackendContract` in `LowerToBackendContract`).
//
// We structure this so that a single run of this pipeline is enough for
// most models, but it is possible for it to take multiple runs to fully
// clean things up when there are cyclic dependencies between certain
// simplifications, such as a decomposition relying on shape refinement which
// depends on another decomposition.
//
// Although technically this pipeline is an implementation detail of
// LowerToBackendContract, we expose it here to help debugging.
//
// LowerToBackendContract will run this pipeline as many times as necessary, but
// in general, it is costly to re-run this pipeline, since all the passes do
// O(module size) work. We want the number of iterations of this pipeline
// to be bounded by meaningful "always in practice small" program properties,
// such as loop nesting depth, number of sequentially dependent steps of
// constant global slots proving that other global slots are dead, etc.
//
// It is generally always possible to construct a pathological input that will
// exceed the number of iterations. If we do find practical cases with
// O(module size) number of iterations of this simplification pipeline, then
// we may need to adjust the approach, such as to do some of the transformations
// together at finer granularity.
void mlir::torch::Torch::createTorchSimplificationPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  // General cleanup.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Inline global slots to expose a bunch of simplification opportunities
  // from constant hyperparameters, weights, etc.
  pm.addPass(createInlineGlobalSlotsPass());
  // Erase the module initializer if we have proven that all the global slots
  // are gone.
  pm.addPass(createEraseModuleInitializerPass());
  // Clean up again to avoid needing to to back around the fixed-point
  // iteration.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createRecomposeComplexOpsPass());
  // Reduce variants of ops to a smaller set of primitives.
  pm.addNestedPass<func::FuncOp>(
      createReduceOpVariantsPass(options.extraLibrary));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Remove dead global slots.
  pm.addPass(createSymbolDCEPass());
  // Convert the bulk of non-ABI-visible !torch.tensor's to !torch.vtensor's.
  pm.addNestedPass<func::FuncOp>(Torch::createMaximizeValueSemanticsPass());
  // Update the return op to return value tensors.
  pm.addPass(Torch::createRefinePublicReturnPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  if (options.shapeDtypeRefine) {
    // Do shape and dtype refinement.
    // Shape refinement should be run before dtype refinement because Torch type
    // promotion rules actually depend on the shape of the operand.
    createTorchShapeRefinementPipeline(pm, options);
    createTorchDtypeRefinementPipeline(pm, options);
  }
  // Propagate to ABI return types the shape/dtype information discovered by
  // the previous pass. Doing this is ABI-compatible for our backends.
  pm.addPass(Torch::createRefinePublicReturnPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  if (options.decompose) {
    pm.addNestedPass<func::FuncOp>(
        Torch::createDecomposeComplexOpsPass(options.backendLegalOps));
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
}

static void createRefinementPipeline(
    mlir::OpPassManager &pm,
    llvm::function_ref<
        std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>(llvm::StringRef)>
        reifyCalculationsPass,
    llvm::function_ref<
        std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>()>
        simplifyCalculationsPass,
    const mlir::torch::Torch::TorchLoweringPipelineOptions &options) {
  // Reify the library functions for each op that is present in the library.
  pm.addPass(reifyCalculationsPass(options.extraLibrary));

  // Inline the library functions to enable analysis and transformation.
  // TODO: Only inline library functions (this will currently inline
  // everything).
  pm.addPass(mlir::createInlinerPass());

  // Now, try to simplify calculations. This is unfortunately a "optimize
  // as hard as possible" kind of thing, so it's inherently somewhat brittle.
  // The idea is to keep strengthening what we do here to support the
  // library functions. We don't need to support arbitrary programs, thankfully.
  pm.addNestedPass<mlir::func::FuncOp>(simplifyCalculationsPass());
  // Run CSE, then see if we can simplify further.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(simplifyCalculationsPass());

  // Drop calculations, leaving behind the-refined program.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::torch::Torch::createDropAbstractInterpCalculationsPass());
}

void mlir::torch::Torch::createTorchShapeRefinementPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  createRefinementPipeline(pm, Torch::createReifyShapeCalculationsPass,
                           Torch::createSimplifyShapeCalculationsPass, options);
}

void mlir::torch::Torch::createTorchDtypeRefinementPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options) {
  createRefinementPipeline(pm, Torch::createReifyDtypeCalculationsPass,
                           Torch::createSimplifyDtypeCalculationsPass, options);
}
