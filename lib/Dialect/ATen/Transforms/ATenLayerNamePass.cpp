//===- ATenLayerNamePass.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Pass/Pass.h"

#include <iostream>
#include <vector>

#define DEBUG_TYPE "aten-layer-name"

using namespace mlir;
using namespace mlir::NPCOMP::aten;

namespace {

struct ATenLayerNamePass : public ATenLayerNameBase<ATenLayerNamePass> {
private:
  std::map<Operation *, std::string> opToName;

public:
  ATenLayerNamePass() {}

  void runOnOperation() override {

    markAllAnalysesPreserved();

    auto module = getOperation();

    // find the function called 'graph'
    auto graph = module.lookupSymbol<mlir::FuncOp>("graph");
    if (!graph) {
      emitError(mlir::UnknownLoc::get(module.getContext()),
                "OpReportPass failed: can't find a graph function\n");
      signalPassFailure();
      return;
    }

    // Construct a name for each aten operation
    std::map<std::string, uint64_t> layerIDmap;
    unsigned currentLayer = 0;

    graph.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();

      // if it's not an aten operation, continue
      // TODO: need an interface for this rather than just
      // doing string compare.
      if (!name.startswith("aten."))
        return;

      // strip the aten prefix to get the operation type
      auto type = name.split("aten.").second;

      // if it's an aten constant op, continue
      if (type.equals("constant"))
        return;

      unsigned ID = 0;
      if (layerIDmap.count(type.str()) == 0)
        layerIDmap[type.str()] = 0;
      else
        ID = ++layerIDmap[type.str()];

      std::string layerName = "L" + std::to_string(currentLayer++) + "-" +
                              type.str() + "-" + std::to_string(ID);

      LLVM_DEBUG(llvm::dbgs()
                 << "generated layer_name: '" << layerName << "'\n");

      auto attr = StringAttr::get(module.getContext(), layerName);
      op->setAttr(StringRef("layer_name"), attr);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::aten::createATenLayerNamePass() {
  return std::make_unique<ATenLayerNamePass>();
}
