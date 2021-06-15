//===- InlineGlobalSlots.cpp -------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

namespace {
class InlineGlobalSlotsPass
    : public InlineGlobalSlotsBase<InlineGlobalSlotsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    auto uses = SymbolTable::getSymbolUses(&module.getBodyRegion());
    if (!uses) {
      module.emitError() << "cannot analyze symbol uses";
      return signalPassFailure();
    }
    // Find all the global slots potentially written from within the module.
    // (we handle the case of non-private symbols later).
    DenseSet<Torch::GlobalSlotOp> potentiallyWrittenGlobalSlots;
    for (const SymbolTable::SymbolUse &use : *uses) {
      auto flatSymbolRef = use.getSymbolRef().dyn_cast<FlatSymbolRefAttr>();
      if (!flatSymbolRef) {
        use.getUser()->emitError() << "unimplemented: nested SymbolRef's";
        return signalPassFailure();
      }
      auto globalSlot =
          symbolTable.lookup<Torch::GlobalSlotOp>(flatSymbolRef.getValue());

      if (!globalSlot)
        continue;
      if (isa<Torch::GlobalSlotGetOp>(use.getUser()))
        continue;

      potentiallyWrittenGlobalSlots.insert(globalSlot);
    }

    DenseSet<Operation *> toErase;
    // Inline all the global slots that are not potentially written.
    for (const SymbolTable::SymbolUse &use : *uses) {
      auto flatSymbolRef = use.getSymbolRef().cast<FlatSymbolRefAttr>();
      auto globalSlot =
          symbolTable.lookup<Torch::GlobalSlotOp>(flatSymbolRef.getValue());
      if (!globalSlot)
        continue;
      // And external user might write to the global slot.
      if (!globalSlot.isPrivate())
        continue;
      // An internal user exists which might write to the global slot.
      if (potentiallyWrittenGlobalSlots.contains(globalSlot))
        continue;
      auto globalSlotGet = cast<Torch::GlobalSlotGetOp>(use.getUser());
      OpBuilder builder(globalSlotGet);
      BlockAndValueMapping mapper;
      for (Operation &op : globalSlot.getBody()->without_terminator())
        builder.clone(op, mapper);
      Value cloned = mapper.lookup(
          cast<GlobalSlotInitOp>(globalSlot.getBody()->getTerminator())
              .getOperand());
      globalSlotGet.replaceAllUsesWith(cloned);
      toErase.insert(globalSlotGet);
      toErase.insert(globalSlot);
    }

    for (Operation *op : toErase)
      op->erase();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::Torch::createInlineGlobalSlotsPass() {
  return std::make_unique<InlineGlobalSlotsPass>();
}
