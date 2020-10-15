//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "npcomp/RefBackend/RefBackend.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Refback/IR/RefbackDialect.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// LowerConstantTensorsToMemref
//===----------------------------------------------------------------------===//

namespace {
// This class creates global ops for all tensor-valued constants in the program.
// It creates them with pretty names and makes sure that duplicate globals
// aren't created.
class GlobalCreator {
public:
  explicit GlobalCreator(ModuleOp module);
  refback::GlobalOp getGlobalFor(Attribute attr) {
    assert(globals.find(attr) != globals.end() && "unknown constant attr");
    return globals[attr];
  }

private:
  DenseMap<Attribute, refback::GlobalOp> globals;
};

GlobalCreator::GlobalCreator(ModuleOp module) {
  // Create a builder without an insertion point. We will insert using the
  // symbol table to guarantee unique names.
  OpBuilder globalBuilder(module.getContext());
  SymbolTable symbolTable(module);
  module.walk([&](ConstantOp op) {
    // We only want tensor constants for now.
    auto type = op.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return;
    // If we already have a global for this constant value, no need to do
    // anything else.
    auto it = globals.find(op.getValue());
    if (it != globals.end())
      return;

    // Create a pretty name.
    SmallString<64> buf;
    llvm::raw_svector_ostream os(buf);
    interleave(type.getShape(), os, "x");
    os << "x" << type.getElementType();

    auto global = globalBuilder.create<refback::GlobalOp>(
        op.getLoc(), (Twine("__constant_") + os.str()).str(),
        op.getValue().cast<ElementsAttr>());
    symbolTable.insert(global);
    // The symbol table inserts at the end of the module, but globals are a bit
    // nicer if they are at the beginning.
    global.getOperation()->moveBefore(&module.front());
    globals[op.getValue()] = global;
  });
}
} // namespace

namespace {
class LowerConstantTensorsToMemref
    : public LowerConstantTensorsToMemrefBase<LowerConstantTensorsToMemref> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<refback::RefbackDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    GlobalCreator globals(module);

    // With the global traversal factored into GlobalCreator, this could in
    // principle be done with a pattern.
    module.walk([&](ConstantOp op) {
      auto type = op.getType().dyn_cast<RankedTensorType>();
      if (!type)
        return;
      auto global = globals.getGlobalFor(op.getValue());
      OpBuilder builder(op);
      auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
      auto memref = builder.create<refback::GetGlobalMemrefOp>(
          op.getLoc(), memrefType, global.getName());
      Value tensor = builder.create<TensorLoadOp>(op.getLoc(), type, memref);
      op.replaceAllUsesWith(tensor);
      op.erase();
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::createLowerConstantTensorsToMemrefPass() {
  return std::make_unique<LowerConstantTensorsToMemref>();
}
