//===- InlineGlobalSlots.cpp -------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file implements an optimistic dataflow analysis that proves that values
// used in global slot initializers are "safe" (see definition below). This
// analysis allows us to inline global slot initializers.
//
// One thing to note is that this inlining (as with all inlining) can create
// duplicate ops. That is usually not a problem, except for certain large
// tensor literals. We rely on later CSE passes to deduplicate those literals.
//
// For debugging this pass an effort has been made for
// `-debug-only=dataflow` and `-debug-only=torch-inline-global-slots` to give a
// good experience. When debugging this pass, it is recommended to start with
// `-debug-only=torch-inline-global-slots` to find values that are marked
// unsafe unexpectedly and then `-debug-only=dataflow` to find why.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torch-inline-global-slots"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/// A program point representing a symbol.
///
/// In principle we could use the `Operation *` program point of the Symbol op,
/// but that just adds a layer of indirection through a symbol table for the
/// purpose of this analysis.
///
/// This is easier because we only support FlatSymbolRefAttr's in Torch-MLIR in
/// a single module. If we had to support complex nested symbol references, we
/// would probably want to go through the effort to indirect through the symbol
/// tables to make things clearer.
class FlatSymbolRefProgramPoint
    : public GenericProgramPointBase<FlatSymbolRefProgramPoint,
                                     FlatSymbolRefAttr> {
public:
  using Base::Base;
  void print(raw_ostream &os) const override {
    os << "FlatSymbolRefProgramPoint(" << getValue() << ")";
  }
  Location getLoc() const override {
    return UnknownLoc::get(getValue().getContext());
  }
};

static bool isTypeTriviallySafe(Type type) {
  return type.isa<Torch::IntType, Torch::FloatType, Torch::BoolType,
                  Torch::StringType, Torch::NoneType, Torch::ValueTensorType>();
}

static bool isUseTreatedWithValueSemantics(OpOperand &use) {
  Operation *op = use.getOwner();
  // If the op unconditionally has value semantics, then the use has value
  // semantics.
  if (op->hasTrait<Torch::OpTrait::HasValueSemantics>())
    return true;
  // The condition of the torch.prim.if op is treated with value semantics.
  if (isa<PrimIfOp>(op) && use.getOperandNumber() == 0)
    return true;
  // TODO: Generalize the HasValueSemantics trait to support
  // operand/result-granularity.
  return false;
}

/// State tracking if an IR construct is "safe".
///
/// This state is tracked on Value's and also on global slots (via a
/// FlatSymbolRefProgramPoint).
///
/// In this context, "safe" means that the object is safe to inline.
/// This covers a few concepts
/// - the value cannot be mutated by the program
/// - the value cannot be potentially aliased, with that alias itself being
///   unsafe
class InlineGlobalSlotsAnalysisState : public AnalysisState {
public:
  InlineGlobalSlotsAnalysisState(ProgramPoint point) : AnalysisState(point) {
    (void)setSafe();
  }

  void print(raw_ostream &os) const override {
    os << "InlineGlobalSlotsAnalysisState(" << (isSafe ? "safe" : "unsafe")
       << ")";
  }

  /// Helper for setting the state with the correct ChangeResult.
  ChangeResult setSafe(bool newIsSafe = true) {
    // As an optimistic analysis, once we prove that a value is unsafe, nothing
    // can prove that it is safe again. This is the monotonicity property of
    // the dataflow analysis that guarantees that we reach a fixed-point.
    // If that property doesn't hold, then there is a bug in the analysis.
    assert(!(isSafe == false && newIsSafe == true) && "non-monotonic update");
    if (isSafe == newIsSafe)
      return ChangeResult::NoChange;
    isSafe = newIsSafe;
    return ChangeResult::Change;
  }

  /// Helper for updatating the state with the correct ChangeResult based on the
  /// safety of a use.
  ChangeResult
  incorporateSafetyOfUse(const InlineGlobalSlotsAnalysisState *useState) {
    // The use is safe, so no need to change anything.
    if (useState->isSafe)
      return ChangeResult::NoChange;
    return setSafe(false);
  }

  /// This is an optimistic analysis. We start assuming everything is safe.
  bool isSafe = true;
};

class InlineGlobalSlotsAnalysis : public DataFlowAnalysis {
public:
  InlineGlobalSlotsAnalysis(DataFlowSolver &solver);
  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

private:
  /// The local transfer function determining the safety of `value`.
  bool isValueSafeTransferFunction(Value value);
  /// The InitializeGlobalSlotsOp of the current module we are analyzing.
  ///
  /// This is used to propagate the analysis from globals into to the module
  /// initializer.
  InitializeGlobalSlotsOp initializeGlobalSlotsOp;
};

InlineGlobalSlotsAnalysis::InlineGlobalSlotsAnalysis(DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerPointKind<FlatSymbolRefProgramPoint>();
}

LogicalResult InlineGlobalSlotsAnalysis::initialize(Operation *top) {
  auto walkResult = top->walk([this](Operation *op) {
    if (auto globalSlot = dyn_cast<Torch::GlobalSlotOp>(op)) {
      auto *state = getOrCreate<InlineGlobalSlotsAnalysisState>(
          getProgramPoint<FlatSymbolRefProgramPoint>(
              FlatSymbolRefAttr::get(globalSlot.getSymNameAttr())));
      propagateIfChanged(state,
                         state->setSafe(globalSlot.getVisibility() !=
                                        SymbolTable::Visibility::Public));
    }
    if (auto globalSlotSet = dyn_cast<Torch::GlobalSlotSetOp>(op)) {
      auto *state = getOrCreate<InlineGlobalSlotsAnalysisState>(
          getProgramPoint<FlatSymbolRefProgramPoint>(
              globalSlotSet.getSlotAttr()));
      propagateIfChanged(state, state->setSafe(false));
    }
    // Save the InitializeGlobalSlotsOp for later referencee
    if (auto initialize = dyn_cast<Torch::InitializeGlobalSlotsOp>(op)) {
      initializeGlobalSlotsOp = initialize;
    }
    for (Value result : op->getResults()) {
      if (failed(visit(result)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

LogicalResult InlineGlobalSlotsAnalysis::visit(ProgramPoint point) {
  if (Value value = point.dyn_cast<Value>()) {
    bool isSafe = isValueSafeTransferFunction(value);
    auto *state = getOrCreate<InlineGlobalSlotsAnalysisState>(value);
    propagateIfChanged(state, state->setSafe(isSafe));

    // Handle GlobalSlotGetOp's.
    if (auto opResult = value.dyn_cast<OpResult>()) {
      if (auto globalSlotGet =
              dyn_cast<Torch::GlobalSlotGetOp>(opResult.getOwner())) {
        auto *flatSymbolRefPoint = getProgramPoint<FlatSymbolRefProgramPoint>(
            globalSlotGet.getSlotAttr());
        auto *valueState = getOrCreateFor<InlineGlobalSlotsAnalysisState>(
            flatSymbolRefPoint, globalSlotGet.getResult());
        auto *globalState =
            getOrCreate<InlineGlobalSlotsAnalysisState>(flatSymbolRefPoint);
        propagateIfChanged(globalState,
                           globalState->incorporateSafetyOfUse(valueState));
      }
    }

    return success();
  }
  if (auto *genericProgramPoint = point.dyn_cast<GenericProgramPoint *>()) {
    if (auto *flatSymbolRefPoint =
            dyn_cast<FlatSymbolRefProgramPoint>(genericProgramPoint)) {
      if (initializeGlobalSlotsOp) {
        auto it =
            llvm::find(initializeGlobalSlotsOp.getSlotSymNames(),
                       static_cast<Attribute>(flatSymbolRefPoint->getValue()));
        Value value = initializeGlobalSlotsOp->getOperand(std::distance(
            initializeGlobalSlotsOp.getSlotSymNames().begin(), it));
        auto *flatSymbolRefState =
            getOrCreateFor<InlineGlobalSlotsAnalysisState>(value,
                                                           flatSymbolRefPoint);
        auto *valueState = getOrCreate<InlineGlobalSlotsAnalysisState>(value);
        propagateIfChanged(valueState,
                           valueState->setSafe(flatSymbolRefState->isSafe));
      }
      return success();
    }
  }
  LLVM_DEBUG(
      { llvm::dbgs() << "visit failing because of: " << point << "\n"; });
  return failure();
}

// This is only a member function to access protected get* functions.
bool InlineGlobalSlotsAnalysis::isValueSafeTransferFunction(Value value) {
  if (isTypeTriviallySafe(value.getType()))
    return true;
  for (OpOperand &use : value.getUses()) {
    Operation *op = use.getOwner();

    if (isUseTreatedWithValueSemantics(use))
      continue;
    // If the op is read-only and all results are safe, then this value is
    // safe. This covers, for example, view-like ops that create aliases.
    if ((op->hasTrait<Torch::OpTrait::ReadOnly>() || isMemoryEffectFree(op)) &&
        llvm::all_of(op->getResults(), [&](Value result) {
          auto *state =
              getOrCreateFor<InlineGlobalSlotsAnalysisState>(value, result);
          return state->isSafe;
        }))
      continue;
    if (auto initialize = dyn_cast<Torch::InitializeGlobalSlotsOp>(op)) {
      auto symName = initialize.getSlotSymNames()[use.getOperandNumber()]
                         .cast<FlatSymbolRefAttr>();
      auto *state = getOrCreateFor<InlineGlobalSlotsAnalysisState>(
          value, getProgramPoint<FlatSymbolRefProgramPoint>(symName));
      if (state->isSafe)
        continue;
    }
    // We may not create all the dependency edges, but that is ok since at
    // this point we have already reached the fixed-point.
    return false;
  }
  return true;
}

SmallVector<Operation *> getBackwardSliceIncludingRoot(Value initialValue) {
  SetVector<Operation *> sliceSet;
  getBackwardSlice(initialValue, &sliceSet);
  SmallVector<Operation *> slice;
  llvm::append_range(slice, sliceSet);
  slice.push_back(initialValue.getDefiningOp());
  return slice;
}

static bool isInitialValueTransitivelySafeToInline(Value initialValue,
                                                   DataFlowSolver &solver) {
  SmallVector<Operation *> slice = getBackwardSliceIncludingRoot(initialValue);
  for (Operation *op : slice) {
    for (auto result : op->getResults()) {
      auto *state = solver.lookupState<InlineGlobalSlotsAnalysisState>(result);
      if (!state->isSafe) {
        return false;
      }
    }
  }
  return true;
}

namespace {
class InlineGlobalSlotsPass
    : public InlineGlobalSlotsBase<InlineGlobalSlotsPass> {
  void runOnOperation() override {

    ModuleOp module = getOperation();
    DataFlowSolver solver;
    solver.load<InlineGlobalSlotsAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();

    LLVM_DEBUG({
      module->walk([&](Operation *op) {
        if (auto globalSlot = dyn_cast<Torch::GlobalSlotOp>(op)) {
          auto *state = solver.lookupState<InlineGlobalSlotsAnalysisState>(
              solver.getProgramPoint<FlatSymbolRefProgramPoint>(
                  FlatSymbolRefAttr::get(globalSlot.getSymNameAttr())));
          state->print(llvm::dbgs());
          llvm::dbgs() << ": "
                       << FlatSymbolRefAttr::get(globalSlot.getSymNameAttr())
                       << "\n";
          return;
        }
        if (op->getNumResults() != 1)
          return;
        auto *state = solver.lookupState<InlineGlobalSlotsAnalysisState>(
            op->getResult(0));
        state->print(llvm::dbgs());
        llvm::dbgs() << ": ";
        op->dump();
      });
    });

    Torch::InitializeGlobalSlotsOp initialize;
    // TODO: Have a torch.module with an optional initializer region to make
    // this tighter.
    for (auto moduleInitializer :
         module.getOps<Torch::GlobalSlotModuleInitializerOp>()) {
      initialize = cast<Torch::InitializeGlobalSlotsOp>(
          moduleInitializer.getBody()->getTerminator());
    }
    if (!initialize) {
      return;
    }

    DenseSet</*FlatSymbolRefAttr*/ Attribute> safeToInline;
    for (int i = 0, e = initialize->getNumOperands(); i != e; i++) {
      auto slotSymName =
          initialize.getSlotSymNames()[i].cast<FlatSymbolRefAttr>();
      Value operand = initialize.getOperand(i);
      auto symbolRefPoint = solver.getProgramPoint<FlatSymbolRefProgramPoint>(
          initialize.getSlotSymNames()[i].cast<FlatSymbolRefAttr>());
      auto *state =
          solver.lookupState<InlineGlobalSlotsAnalysisState>(symbolRefPoint);
      // We roll the analysis of whether a slot is set or public into the
      // main dataflow analysis, so we need to check the slot's
      // FlatSymbolRefProgramPoint itself to see if it is safe to inline.
      // For example, a public !torch.int is not safe to inline, even though
      // it is a value-semantic type and so the actual initializer value
      // itself is conceptually safe to inline.
      if (!state->isSafe) {
        continue;
      }
      // Check to see if the initializing value is safe to inline.
      // This requires a transitive check of all subobjects.
      // TODO: This would really be more logical to do as a forward dataflow
      // analyis on the whole module initializer rather than doing the
      // transitive check backward for each initial value. But it is just
      // too much boilerplate to write that with the dataflow framework and we
      // generally don't expect long transitive chains of values here -- most
      // initial values are just single tensor literals.
      if (isInitialValueTransitivelySafeToInline(operand, solver)) {
        safeToInline.insert(slotSymName);
      }
    }

    SymbolTable symbolTable(module);
    DenseSet<Operation *> toErase;
    module.walk([&](Torch::GlobalSlotGetOp op) {
      if (!safeToInline.count(op.getSlotAttr()))
        return;
      // TODO: Make this more ergonomic.
      auto it = llvm::find(initialize.getSlotSymNames(), op.getSlotAttr());
      Value initialValue = initialize.getOperand(
          std::distance(initialize.getSlotSymNames().begin(), it));
      // It seems inefficient to get a backward slice again here, but we are
      // going to be cloning the whole slice anyway, so it doesn't seem like a
      // big deal.
      SmallVector<Operation *> slice =
          getBackwardSliceIncludingRoot(initialValue);
      IRMapping mapping;
      OpBuilder builder(op);
      for (Operation *opInSlice : slice)
        builder.clone(*opInSlice, mapping);
      auto inlinedInitialValue = mapping.lookup(initialValue);
      inlinedInitialValue = Torch::adjustStaticInformation(
          builder, op.getLoc(), inlinedInitialValue, op.getType(),
          /*userAllowsRefinement=*/false);
      op.replaceAllUsesWith(inlinedInitialValue);
      toErase.insert(op);
    });

    // Clean up after the transform.

    // Erase any pending ops.
    for (Operation *op : toErase)
      op->erase();
    // Erase any global slots that we inlined.
    // This could be left to SymbolDCE but it's not hard to do here.
    for (FlatSymbolRefAttr symName :
         llvm::map_range(safeToInline, [](Attribute attr) {
           return attr.cast<FlatSymbolRefAttr>();
         })) {
      auto globalSlot =
          symbolTable.lookup<Torch::GlobalSlotOp>(symName.getValue());
      globalSlot.erase();
    }

    // Update the initializer.
    SmallVector<Attribute> newSlotSymNames;
    SmallVector<Value> newInitialValues;
    for (int i = 0, e = initialize.getNumOperands(); i != e; i++) {
      auto slotSymName =
          initialize.getSlotSymNames()[i].cast<FlatSymbolRefAttr>();
      if (!safeToInline.count(slotSymName)) {
        newSlotSymNames.push_back(slotSymName);
        newInitialValues.push_back(initialize.getOperand(i));
      }
    }
    {
      OpBuilder builder(initialize);
      builder.create<Torch::InitializeGlobalSlotsOp>(
          initialize.getLoc(),
          ArrayAttr::get(module.getContext(), newSlotSymNames),
          newInitialValues);
    }
    initialize.erase();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createInlineGlobalSlotsPass() {
  return std::make_unique<InlineGlobalSlotsPass>();
}
