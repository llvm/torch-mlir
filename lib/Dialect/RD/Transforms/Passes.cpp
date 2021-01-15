//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/RD/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "npcomp/Dialect/RD/Transforms/Passes.h.inc"
} // end namespace

void mlir::NPCOMP::registerRDPasses() { ::registerPasses(); }

#include "PassDetail.h"

namespace mlir {
namespace NPCOMP {

llvm::Optional<FuncOp> findDefinitionFunc(rd::PipelineDefinitionOp definition) {
  llvm::Optional<FuncOp> def;
  definition.walk([&](FuncOp op) {
    if (op.getName() == "definition") {
      if (def) {
        op.emitError("Multiple definition functions.").attachNote(def->getLoc())
            << "Previous definition here.";
        return;
      }
      def = op;
    }
  });
  if (!def) {
    definition.emitError("Missing definition function.");
  }
  return def;
}

} // namespace NPCOMP
} // namespace mlir
