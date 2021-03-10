//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

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
      "torch-globalize-pipeline", "Globalization pipeline.",
      mlir::NPCOMP::Torch::createGlobalizePipeline);
}

void mlir::NPCOMP::Torch::createGlobalizePipeline(OpPassManager &pm) {
  pm.addPass(createPrepareForGlobalizeObjectGraphPass());
  pm.addPass(createGlobalizeObjectGraphPass());
}
