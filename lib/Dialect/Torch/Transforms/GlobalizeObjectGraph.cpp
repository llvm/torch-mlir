//===- GlobalizeObjectGraph.cpp ----------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static FailureOr<NnModuleOp> findRootNnModule(ModuleOp module) {
  NnModuleOp rootNnModule;
  for (NnModuleOp op : module.getOps<NnModuleOp>()) {
    if (!op.use_empty())
      continue;
    if (rootNnModule) {
      op.emitError()
          .append("found more than one root module (module that is not a "
                  "child of any other module)")
          .attachNote(rootNnModule.getLoc())
          .append("see other root module here");
      return failure();
    }
    rootNnModule = op;
  }
  if (!rootNnModule) {
    module.emitError() << "module does not contain a root torch.nn_module";
    return failure();
  }
  return rootNnModule;
}

//===----------------------------------------------------------------------===//
// Object graph recursive traversal.
//===----------------------------------------------------------------------===//

namespace {
struct LinkageInfo {
  std::string linkageName;
  bool isPrivate;
};
} // namespace
namespace {
/// Calculates the linkage names of all the potentially exported objects in the
/// module and also creates GlobalSlotOp's for each SlotOp and tracks their
/// associations.
///
/// The mechanics of both of these tasks involve the same object graph
/// traversal, so it's useful to roll them together.
class ObjectGraphInfo {
public:
  ObjectGraphInfo(ModuleOp module)
      : globalSlotBuilder(module.getBodyRegion()), symbolTable(module) {}

  LogicalResult initialize(NnModuleOp rootNnModule) {
    if (failed(collectUsedSlots()))
      return failure();
    return recursivelyTraverse(rootNnModule);
  }

  LinkageInfo getSlotLinkageInfo(SlotOp op) {
    auto it = slotLinkageInfo.find(op);
    assert(it != slotLinkageInfo.end());
    return it->second;
  }
  std::optional<LinkageInfo> getFuncLinkageInfo(NnModuleOp instance,
                                                func::FuncOp methodFunc) {
    auto it = funcLinkageInfo.find({instance, methodFunc});
    if (it == funcLinkageInfo.end())
      return std::nullopt;
    return it->second;
  }

  GlobalSlotOp getGlobalSlotFor(SlotOp slot) {
    auto it = slotToGlobalSlot.find(slot);
    assert(it != slotToGlobalSlot.end() && "didn't create global slot");
    return it->second;
  }
  llvm::MapVector<StringAttr, Value> &getGlobalSlotInitialValues() {
    return globalSlotInitialValues;
  }

private:
  LogicalResult collectUsedSlots() {
    // Collect all the slots in each module.
    // moduleClassNameToSlots tracks, for each class, for each attribute, the
    // set of slot instances that belong to that attribute. E.g. if there are
    // two instances of a class "Foo" with an attribute "a", then there will be
    // two SlotOps in the inner vector of moduleClassNameToSlots["Foo"]["a"].
    // This is not precise -- in the code below it effectively results in the
    // conservative assumption that all instances of a class might reach all
    // GetAttr ops on that type.
    llvm::StringMap<llvm::StringMap<std::vector<SlotOp>>>
        moduleClassNameToSlots;
    symbolTable.getOp()->walk([&](NnModuleOp moduleOp) {
      llvm::StringMap<SlotOp> nameToSlot;
      auto &slotNameToSlots = moduleClassNameToSlots[moduleOp.getClassName()];
      for (auto slotOp : moduleOp.getOps<SlotOp>())
        slotNameToSlots[slotOp.getName()].push_back(slotOp);
    });

    // Find all the module slots that are accessed through `PrimGetAttrOp` or
    // `PrimSetAttrOp`.
    symbolTable.getOp()->walk([&](Operation *op) {
      if (!isa<PrimGetAttrOp, PrimSetAttrOp>(op))
        return;

      Value module;
      StringRef slotName;
      if (auto getAttrOp = llvm::dyn_cast<PrimGetAttrOp>(op)) {
        module = getAttrOp.getReceiver();
        slotName = getAttrOp.getName();
      } else {
        auto setAttrOp = cast<PrimSetAttrOp>(op);
        module = setAttrOp.getReceiver();
        slotName = setAttrOp.getName();
      }

      auto moduleType = module.getType().cast<NnModuleType>();
      auto slots = moduleClassNameToSlots.find(moduleType.getClassName());
      // TODO: Improve verifier so that this can never happen
      if (slots == moduleClassNameToSlots.end())
        op->emitError() << "Reference to non-existing module type "
                        << moduleType.getClassName();

      auto &slotNameToSlots = slots->getValue();
      auto slotIt = slotNameToSlots.find(slotName);
      // TODO: Improve verifier so that this can never happen
      if (slotIt == slotNameToSlots.end())
        op->emitError() << "Reference to non-existing module slot " << slotName
                        << "in " << moduleType.getClassName();
      for (SlotOp slotOp : slotIt->getValue())
        usedSlots.insert(slotOp);
    });
    return success();
  }

  LogicalResult recursivelyTraverse(NnModuleOp nnModule) {
    std::string pathToClassFromRoot = llvm::join(nameStack, ".");
    if (!seenNnModules.insert({nnModule, pathToClassFromRoot}).second) {
      return nnModule.emitError()
             << "reachable by multiple paths from root object: '<root>."
             << seenNnModules[nnModule] << "' and '<root>."
             << pathToClassFromRoot << "'";
    }

    auto classType = symbolTable.lookup<ClassTypeOp>(
        nnModule.getType().cast<NnModuleType>().getClassName());
    for (auto t :
         llvm::zip(nnModule.getOps<SlotOp>(), classType.getOps<AttrOp>())) {
      auto slot = std::get<0>(t);
      auto attr = std::get<1>(t);
      nameStack.push_back(attr.getName().str());
      if (attr.getType().isa<NnModuleType>()) {
        if (failed(recursivelyTraverse(
                slot.getValue().getDefiningOp<NnModuleOp>())))
          return failure();
      } else if (usedSlots.find(slot) != usedSlots.end()) {
        // Only create the GlobalSlotOp if the slot is used at all.
        std::string linkageName = llvm::join(nameStack, ".");
        auto globalSlot = globalSlotBuilder.create<GlobalSlotOp>(
            slot.getLoc(), linkageName,
            /*sym_visibility=*/nullptr, attr.getType());
        if (attr.getIsPrivate())
          globalSlot.setVisibility(SymbolTable::Visibility::Private);
        assert(slotToGlobalSlot.find(slot) == slotToGlobalSlot.end());
        slotToGlobalSlot[slot] = globalSlot;
        slotLinkageInfo[slot] = LinkageInfo{linkageName, attr.getIsPrivate()};
        globalSlotInitialValues[globalSlot.getSymNameAttr()] = slot.getValue();
      }
      nameStack.pop_back();
    }
    for (auto method : classType.getOps<MethodOp>()) {
      nameStack.push_back(method.getName().str());
      funcLinkageInfo[{
          nnModule, symbolTable.lookup<func::FuncOp>(method.getFunction())}] =
          LinkageInfo{llvm::join(nameStack, "."), method.getIsPrivate()};
      nameStack.pop_back();
    }
    return success();
  }
  // Builder for creating GlobalSlotOp's in the module.
  OpBuilder globalSlotBuilder;
  // Symbol table for the module.
  SymbolTable symbolTable;
  // The set of NnModuleOp's that have already been processed.
  // Used for diagnostics.
  // The map value is the original path from the root that we found it at.
  DenseMap<NnModuleOp, std::string> seenNnModules;

  // The stack of attribute names we have traversed during our recursive
  // traversal of the class/object hierarchy.
  //
  // Linkage names are calculated based on the set of attribute names traversed
  // from the root class/module in the program.
  std::vector<std::string> nameStack;
  // Linkage info for each SlotOp in the program.
  DenseMap<SlotOp, LinkageInfo> slotLinkageInfo;
  // Linkage info for each method in the program. Since we are going to be
  // monomorphizing all the functions, we also need to key this off of the
  // instance (NnModuleOp) that the func is monomorphized for.
  DenseMap<std::pair<NnModuleOp, func::FuncOp>, LinkageInfo> funcLinkageInfo;
  // The corresponding GlobalSlotOp for each SlotOp in the program.
  DenseMap<SlotOp, GlobalSlotOp> slotToGlobalSlot;
  // The initializing value for each GlobalSlotOp.
  // This is a MapVector to keep the order deterministic.
  llvm::MapVector<StringAttr, Value> globalSlotInitialValues;
  // Used to keep track of all the used torch slots so that the restrictions can
  // be applied to those slots only.
  DenseSet<SlotOp> usedSlots;
};
} // namespace

LogicalResult
createGlobalSlotModuleInitializer(ModuleOp module, SymbolTable &symbolTable,
                                  ObjectGraphInfo &objectGraphInfo) {
  auto builder = OpBuilder::atBlockBegin(module.getBody());
  auto moduleInitializer =
      builder.create<GlobalSlotModuleInitializerOp>(module.getLoc());
  Block *body = builder.createBlock(&moduleInitializer.getInitializer());
  builder.setInsertionPointToEnd(body);
  SmallVector<Operation *> opsToMove;
  for (Operation &op : *module.getBody()) {
    if (isa<ClassTypeOp, NnModuleOp, GlobalSlotOp, func::FuncOp,
            GlobalSlotModuleInitializerOp>(op))
      continue;
    opsToMove.push_back(&op);
  }
  IRMapping mapping;
  for (Operation *op : opsToMove) {
    // The ops are used by `torch.slot` ops in the enclosing module.
    // Cloning avoids needing to handle those uses specially.
    builder.clone(*op, mapping);
  }
  SmallVector<Attribute> slotSymNames;
  SmallVector<Value> initialValues;
  for (auto &kv : objectGraphInfo.getGlobalSlotInitialValues()) {
    StringAttr symName = kv.first;
    Value initializer = kv.second;
    slotSymNames.push_back(FlatSymbolRefAttr::get(symName));
    initialValues.push_back(mapping.lookup(initializer));
  }
  builder.create<InitializeGlobalSlotsOp>(
      moduleInitializer.getLoc(),
      ArrayAttr::get(module.getContext(), slotSymNames), initialValues);
  return success();
}

//===----------------------------------------------------------------------===//
// Monomorphization.
//===----------------------------------------------------------------------===//

namespace {
// When used in an Monomorphization, indicates that the arg at `argIndex` will
// correspond to instance `instance.
struct ArgInstance {
  int argIndex;
  Value instance; // Result of an NnModuleOp.
};
static llvm::hash_code hash_value(const ArgInstance &argInstance) {
  return llvm::hash_combine(argInstance.argIndex, argInstance.instance);
}
static bool operator==(const ArgInstance &lhs, const ArgInstance &rhs) {
  return std::make_tuple(lhs.argIndex, lhs.instance) ==
         std::make_tuple(rhs.argIndex, rhs.instance);
}
} // namespace

namespace {
// Record indicating that a particular function must be monomorphized for the
// given ArgInstance's, which involves deleting those arguments and specializing
// all their uses to operate on GlobalSlotOp's that we have created for the
// SlotOp's of the NnModuleOp instances.
//
// NOTE: Unlike the more traditional use of monomorphization to mean a single
// *type* is being specialized for, here we are specializing for a specific
// *instance*. This still fits the definition of monomorphization though, albeit
// with each instance being considered to have a maximally refined type which is
// a set with a single element (just this instance). This does not correspond to
// any notion of "type" that we have in the IR, but still fits the formal
// definition.
struct Monomorphization {
  func::FuncOp func;
  std::vector<ArgInstance> argInstances;
};
} // namespace

template <> struct llvm::DenseMapInfo<Monomorphization> {
  static Monomorphization getEmptyKey() {
    return Monomorphization{nullptr, {ArgInstance{-1, nullptr}}};
  }
  static Monomorphization getTombstoneKey() {
    return Monomorphization{nullptr, {ArgInstance{-2, nullptr}}};
  }
  static unsigned getHashValue(Monomorphization val) {
    return llvm::hash_combine(val.func.getAsOpaquePointer(),
                              llvm::hash_combine_range(val.argInstances.begin(),
                                                       val.argInstances.end()));
  }
  static bool isEqual(Monomorphization lhs, Monomorphization rhs) {
    return lhs.func == rhs.func &&
           std::equal(lhs.argInstances.begin(), lhs.argInstances.end(),
                      rhs.argInstances.begin(), rhs.argInstances.end());
  }
};

// Populate `mapping` such that values of NnModuleType in the function are
// mapped to appropriate global objects of NnModuleType.
//
// This generalizes to a full abstract interpretation of the function, but
// currently only analyzes a subset of ops.
static LogicalResult analyzeInstances(func::FuncOp func,
                                      ArrayRef<ArgInstance> argInstances,
                                      IRMapping &mapping) {
  for (auto &argInstance : argInstances)
    mapping.map(func.getArgument(argInstance.argIndex), argInstance.instance);
  auto walkResult = func.walk([&](PrimGetAttrOp op) {
    if (!op.getType().isa<NnModuleType>())
      return WalkResult::advance();
    auto instance = mapping.lookupOrNull(op.getReceiver());
    assert(instance && "verifyFuncConformsToSubset should ensure this");
    for (auto slot : instance.getDefiningOp<NnModuleOp>().getOps<SlotOp>()) {
      if (slot.getName() == op.getName()) {
        mapping.map(op, slot.getValue());
        break;
      }
    }
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

static FailureOr<Monomorphization>
createMonomorphizationForCall(func::CallOp op, IRMapping &mapping,
                              SymbolTable &symbolTable) {
  auto func = symbolTable.lookup<func::FuncOp>(op.getCallee());
  Monomorphization monomorphization;
  monomorphization.func = func;
  for (auto operand : llvm::enumerate(op->getOperands())) {
    if (!operand.value().getType().isa<NnModuleType>())
      continue;
    Value instance = mapping.lookupOrNull(operand.value());
    assert(instance && "verifyFuncConformsToSubset should ensure this");
    monomorphization.argInstances.push_back(
        ArgInstance{static_cast<int>(operand.index()), instance});
  }
  return monomorphization;
}

namespace {
class MonomorphizationTracker {
public:
  MonomorphizationTracker(ModuleOp module)
      : module(module), symbolTable(module) {}
  LogicalResult
  initialize(DenseMap<ClassTypeOp, std::vector<NnModuleOp>> &instances) {
    for (auto func : module.getOps<func::FuncOp>()) {
      Monomorphization monomorphization;
      monomorphization.func = func;
      bool canTriviallyMonomorphize = true;
      for (auto arg : llvm::enumerate(func.getArguments())) {
        auto type = arg.value().getType().dyn_cast<NnModuleType>();
        if (!type)
          continue;
        auto classType = symbolTable.lookup<ClassTypeOp>(type.getClassName());
        auto &classTypeInstances = instances[classType];
        if (classTypeInstances.size() != 1) {
          canTriviallyMonomorphize = false;
          break;
        }
        monomorphization.argInstances.push_back(
            {static_cast<int>(arg.index()), classTypeInstances[0]});
      }

      if (canTriviallyMonomorphize) {
        dirtyMonomorphizations.push_back(monomorphization);
        monomorphizations.insert(monomorphization);
      }
    }
    while (!dirtyMonomorphizations.empty()) {
      Monomorphization dirty = dirtyMonomorphizations.pop_back_val();
      if (failed(generateNewMonomorphizations(dirty)))
        return failure();
    }
    return success();
  }

  llvm::SetVector<Monomorphization> &getMonomorphizations() {
    return monomorphizations;
  }

private:
  LogicalResult generateNewMonomorphizations(const Monomorphization &m) {
    auto func = m.func;
    IRMapping mapping;
    if (failed(analyzeInstances(func, m.argInstances, mapping)))
      return failure();
    auto walkResult = func.walk([&](func::CallOp op) {
      FailureOr<Monomorphization> maybeMonomorphization =
          createMonomorphizationForCall(op, mapping, symbolTable);
      if (failed(maybeMonomorphization))
        return WalkResult::interrupt();
      if (monomorphizations.insert(*maybeMonomorphization))
        dirtyMonomorphizations.push_back(*maybeMonomorphization);
      return WalkResult::advance();
    });
    return success(!walkResult.wasInterrupted());
  }

  ModuleOp module;
  SymbolTable symbolTable;
  SmallVector<Monomorphization> dirtyMonomorphizations;
  llvm::SetVector<Monomorphization> monomorphizations;
};
} // namespace

// Verify that a value conforms to the subset of allowed uses for
// !torch.nn.Module<"..."> types.
static LogicalResult verifyNnModuleValueUses(Value value) {
  // Trivially succeed for non-module types.
  if (!value.getType().isa<NnModuleType>())
    return success();
  for (Operation *op : value.getUsers()) {
    if (isa<func::CallOp, PrimGetAttrOp>(op))
      continue;
    // Only allow `value` as the receiver.
    if (isa<PrimSetAttrOp>(op) && cast<PrimSetAttrOp>(op).getValue() != value)
      continue;
    // TODO: Improve this based on real user use cases.
    // This is a diagnostic that users will hit if they do not conform to
    // the supported subset of TorchScript.
    return op->emitError() << "unsupported use of a torch.nn.Module. Expected "
                              "only method calls or attribute get/set";
  }
  return success();
}

// Verify that `func` conforms to the subset of allowable method bodies
// that we can convert.
static LogicalResult verifyFuncConformsToSubset(func::FuncOp func) {
  auto walkResult = func.walk([&](Block *block) {
    for (Value arg : block->getArguments())
      if (failed(verifyNnModuleValueUses(arg)))
        return WalkResult::interrupt();
    for (Operation &op : *block)
      for (Value result : op.getResults())
        if (failed(verifyNnModuleValueUses(result)))
          return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

static LogicalResult
verifyPublicMonomorphizations(ModuleOp module, SymbolTable &symbolTable,
                              MonomorphizationTracker &tracker) {
  DenseMap<func::FuncOp, int> numMonomorphizations;
  for (auto &monomorphization : tracker.getMonomorphizations()) {
    numMonomorphizations[monomorphization.func] += 1;
  }
  bool sawError = false;
  for (auto classType : module.getOps<ClassTypeOp>()) {
    for (auto method : classType.getOps<MethodOp>()) {
      if (!method.getIsPrivate()) {
        if (numMonomorphizations[symbolTable.lookup<func::FuncOp>(
                method.getFunction())] > 1) {
          method.emitError()
              << "public function with multiple monomorphizations";
          sawError = true;
        }
      }
    }
  }
  return success(!sawError);
}

// Rewrite `func`, given that all values of `NnModuleType` have been mapped in
// `mapping` to corresponding global instances.
static LogicalResult rewriteMonomorphizedFuncClone(
    func::FuncOp func, IRMapping mapping, SymbolTable &symbolTable,
    DenseMap<Monomorphization, func::FuncOp> &newFuncs,
    ObjectGraphInfo &objectGraphInfo) {

  SmallVector<Operation *> toErase;
  auto handlePrimSetAttr = [&](PrimSetAttrOp op) {
    auto instance =
        mapping.lookup(op.getReceiver()).getDefiningOp<NnModuleOp>();
    SlotOp affectedSlot;
    for (auto slot : instance.getOps<SlotOp>()) {
      if (slot.getName() == op.getName())
        affectedSlot = slot;
    }
    OpBuilder(op).create<GlobalSlotSetOp>(
        op.getLoc(),
        objectGraphInfo.getGlobalSlotFor(affectedSlot).getSymName(),
        op.getValue());
    toErase.push_back(op);
    return WalkResult::advance();
  };
  auto handlePrimGetAttr = [&](PrimGetAttrOp op) {
    if (!op.getType().isa<NnModuleType>()) {
      auto instance =
          mapping.lookup(op.getReceiver()).getDefiningOp<NnModuleOp>();
      SlotOp affectedSlot;
      for (auto slot : instance.getOps<SlotOp>()) {
        if (slot.getName() == op.getName())
          affectedSlot = slot;
      }
      auto newOp = OpBuilder(op).create<GlobalSlotGetOp>(
          op.getLoc(), op.getType(),
          objectGraphInfo.getGlobalSlotFor(affectedSlot).getSymName());
      op.replaceAllUsesWith(&*newOp);
    }
    toErase.push_back(op);
    return WalkResult::advance();
  };
  auto handleCall = [&](func::CallOp op) {
    FailureOr<Monomorphization> maybeMonomorphization =
        createMonomorphizationForCall(op, mapping, symbolTable);
    if (failed(maybeMonomorphization))
      return WalkResult::interrupt();
    Monomorphization monomorphization = std::move(*maybeMonomorphization);
    auto newArguments = llvm::to_vector<6>(
        llvm::make_filter_range(op->getOperands(), [](Value v) {
          return !v.getType().isa<NnModuleType>();
        }));
    assert(newFuncs.find(monomorphization) != newFuncs.end());
    auto newOp = OpBuilder(op).create<func::CallOp>(
        op.getLoc(), newFuncs[monomorphization], newArguments);
    op.replaceAllUsesWith(newOp);
    toErase.push_back(op);
    return WalkResult::advance();
  };
  auto walkResult = func.walk([&](Operation *op) {
    if (auto primSetAttr = dyn_cast<PrimSetAttrOp>(op))
      return handlePrimSetAttr(primSetAttr);
    if (auto primGetAttr = dyn_cast<PrimGetAttrOp>(op))
      return handlePrimGetAttr(primGetAttr);
    if (auto call = dyn_cast<func::CallOp>(op))
      return handleCall(call);
    return WalkResult::advance();
  });
  for (auto op : toErase) {
    op->dropAllDefinedValueUses();
    op->erase();
  }
  llvm::BitVector argsToErase(func.getNumArguments());
  for (auto type : llvm::enumerate(func.getArgumentTypes())) {
    if (type.value().isa<NnModuleType>()) {
      argsToErase.set(type.index());
    }
  }
  func.eraseArguments(argsToErase);
  return success(!walkResult.wasInterrupted());
}

static LogicalResult globalizeObjectGraph(ModuleOp module) {

  // Step 1: Traverse object graph and collect information.

  FailureOr<NnModuleOp> maybeRootNnModule = findRootNnModule(module);
  if (failed(maybeRootNnModule))
    return failure();
  NnModuleOp rootNnModule = *maybeRootNnModule;
  ObjectGraphInfo objectGraphInfo(module);
  if (failed(objectGraphInfo.initialize(rootNnModule)))
    return failure();

  DenseMap<ClassTypeOp, std::vector<NnModuleOp>> instances;
  SymbolTable symbolTable(module);
  for (auto nnModule : module.getOps<NnModuleOp>()) {
    auto classType = nnModule.getClassType(symbolTable);
    instances[classType].push_back(nnModule);
  }

  // Step 2: Create the torch.global_slot.module_initializer op.

  if (failed(createGlobalSlotModuleInitializer(module, symbolTable,
                                               objectGraphInfo)))
    return failure();

  // Step 3: Verify all functions are suitable to be analyzed by our later code.
  // This eliminates special handling / error code later.
  //
  // This is important, because in principle, we can perform arbitrarily complex
  // static analysis to discover how to monomorphize th eprogram, including
  // tracking instances through control flow, through get/set attr, etc. We
  // implement a very simple subset of cases.
  for (auto func : module.getOps<func::FuncOp>()) {
    if (failed(verifyFuncConformsToSubset(func)))
      return failure();
  }

  // Step 4: Calculate the set of monomorphized functions that need to be
  // created. For each call that passes !torch.nn.Module to a function, we need
  // to create a specialized version of that function just for that instance (or
  // combination of instances in the case of multiple arguments).
  //
  // At this stage, we only analyze which monomorphizations are needed and
  // whether it is possible to monomorphize the program. The actual
  // cloning/rewriting mechanics happen later.
  //
  // This lets us know which GlobalSlotOp we need to reference when we replace
  // PrimSetAttrOp/PrimGetAttrOp.
  //
  // Note that in general there can be mutually recursive functions that
  // re-enter themselves with a different set of instances -- the process of
  // calculating these monomorphizations is a fixed-point iteration that
  // discovers all needed monomorphizations. In practice this yields a
  // controllable number.
  MonomorphizationTracker tracker(module);
  if (failed(tracker.initialize(instances)))
    return failure();

  if (failed(verifyPublicMonomorphizations(module, symbolTable, tracker))) {
    return failure();
  }

  // Step 5: Clone/rewrite functions to implement the necessary
  // monomorphizations.
  DenseMap<Monomorphization, func::FuncOp> newFuncs;
  int uniquifier = 0;
  for (auto &monomorphization : tracker.getMonomorphizations()) {
    auto newFunc = cast<func::FuncOp>(monomorphization.func->clone());
    newFuncs[monomorphization] = newFunc;
    std::optional<LinkageInfo> linkageInfo = std::nullopt;
    // If it is potentially a method, check its linkage info.
    if (monomorphization.argInstances.size() != 0 &&
        monomorphization.argInstances[0].argIndex == 0) {
      linkageInfo = objectGraphInfo.getFuncLinkageInfo(
          monomorphization.argInstances[0].instance.getDefiningOp<NnModuleOp>(),
          monomorphization.func);
    }
    if (linkageInfo.has_value()) {
      // It's a method.
      newFunc.setVisibility(linkageInfo->isPrivate
                                ? SymbolTable::Visibility::Private
                                : SymbolTable::Visibility::Public);
      newFunc.setName(linkageInfo->linkageName);
    } else {
      // It's a free function.
      // TODO: Make the name nicer (no suffix in typical case).
      newFunc.setName(
          (Twine(newFunc.getName()) + "$" + Twine(uniquifier++)).str());
    }
    module.push_back(newFunc);
  }

  for (auto &kv : newFuncs) {
    IRMapping mapping;
    if (failed(analyzeInstances(kv.second, kv.first.argInstances, mapping)))
      return failure();
    if (failed(rewriteMonomorphizedFuncClone(kv.second, mapping, symbolTable,
                                             newFuncs, objectGraphInfo)))
      return failure();
  }

  // Step 6: Clean up object graph.
  DenseSet<func::FuncOp> liveFuncs;
  for (auto &kv : newFuncs) {
    liveFuncs.insert(kv.second);
  }
  for (auto &op : llvm::make_early_inc_range(module.getOps())) {
    if (isa<GlobalSlotOp, GlobalSlotModuleInitializerOp>(&op))
      continue;
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      if (liveFuncs.contains(func))
        continue;
    }
    op.dropAllDefinedValueUses();
    op.dropAllReferences();
    op.erase();
  }

  return success();
}

namespace {
class GlobalizeObjectGraphPass
    : public GlobalizeObjectGraphBase<GlobalizeObjectGraphPass> {
  void runOnOperation() override {
    if (failed(globalizeObjectGraph(getOperation())))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createGlobalizeObjectGraphPass() {
  return std::make_unique<GlobalizeObjectGraphPass>();
}
