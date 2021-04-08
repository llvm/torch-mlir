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
// This pass lowers the public ABI of the module to the primitives exposed by
// the refbackrt dialect.
class LowerLinkagePass : public LowerLinkageBase<LowerLinkagePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto func : module.getOps<FuncOp>()) {
      if (func.getVisibility() == SymbolTable::Visibility::Public)
        func->setAttr("iree.module.export", UnitAttr::get(&getContext()));
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::IREEBackend::createLowerLinkagePass() {
  return std::make_unique<LowerLinkagePass>();
}
