//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/Backend/IREE/Passes.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::IREEBackend;

namespace {
#define GEN_PASS_REGISTRATION
#include "npcomp/Backend/IREE/Passes.h.inc"
} // end namespace

void mlir::NPCOMP::IREEBackend::createNpcompBackendToIreeFrontendPipeline(
    OpPassManager &pm) {
  pm.addPass(createLowerLinkagePass());
}

void mlir::NPCOMP::IREEBackend::registerIREEBackendPasses() {
  ::registerPasses();

  mlir::PassPipelineRegistration<>(
      "npcomp-backend-to-iree-frontend-pipeline",
      "Pipeline lowering the npcomp backend contract IR to IREE's frontend "
      "contract.",
      mlir::NPCOMP::IREEBackend::createNpcompBackendToIreeFrontendPipeline);
}
