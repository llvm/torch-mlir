//===- TmpDeleteDeadIREELists.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "npcomp/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::TorchConversion;

namespace {
class TmpDeleteDeadIREEListsPass
    : public TmpDeleteDeadIREEListsBase<TmpDeleteDeadIREEListsPass> {
  void runOnOperation() override {
    SmallVector<Operation *> toErase;
    // Delete lists that are only set (but not read from).
    // This is created by our lowering for torch.prim.ListConstruct.
    // Until IREE can run such ops e2e (or delete them itself), we need to
    // do this cleanup.
    // TODO: Add support to IREE to run these ops E2E.
    getOperation().walk([&](iree::ListCreateOp op) {
      SmallVector<Operation *> deadOps;
      deadOps.push_back(op);
      for (auto &use : op.getResult().getUses()) {
        if (isa<iree::ListSetOp, iree::ListResizeOp>(use.getOwner())) {
          deadOps.push_back(use.getOwner());
        } else {
          // We can't analyze the list op if it is used by something else.
          return;
        }
      }
      llvm::append_range(toErase, deadOps);
    });
    for (auto *op : toErase) {
      op->dropAllDefinedValueUses();
      op->erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::TorchConversion::createTmpDeleteDeadIREEListsPass() {
  return std::make_unique<TmpDeleteDeadIREEListsPass>();
}
