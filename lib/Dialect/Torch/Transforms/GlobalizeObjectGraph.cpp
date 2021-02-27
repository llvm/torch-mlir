//===- GlobalizeObjectGraph.cpp ----------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

namespace {
// See the pass documentation for `torch-globalize-object-graph`.
class ObjectGraphGlobalizer {
public:
  ObjectGraphGlobalizer(ModuleOp module);
  LogicalResult globalizeObjectGraph();

private:
  FailureOr<NnModuleOp> findRootNnModule();
  LogicalResult checkSingleInstanceOfEachClass();
  LogicalResult recursivelyTraverseClassType(ClassTypeOp classType);
  LogicalResult populateGlobalSlotInitializer(GlobalSlotOp op,
                                              Value initialValue);
  LogicalResult rewriteMethods();
  void removeObjectGraph();

  ModuleOp module;
  SymbolTable symbolTable;
  OpBuilder globalBuilder;
  // The stack of attribute names we have traversed during our recursive
  // traversal of the class/object hierarchy.
  //
  // Linkage names are calculated based on the set of attribute names traversed
  // from the root class/module in the program.
  SmallVector<std::string> nameStack;

  // Sometimes it is natural to want a map keyed on torch.attr ops or torch.slot
  // ops. However, usually it is better to keep a map keyed on an ClassTypeOp
  // + attr name since frequently that is all one has access to and it
  // would be tedious to scan the body of the ClassTypeOp for the torch.attr
  // with the corresponding name.
  using AttrOfClass =
      std::pair</*ClassTypeOp*/ Operation *, /*attr name*/ StringRef>;
  // The initial value associated with an attribute of a class.
  // Since we only allow a single instance of a class, this is equivalent to
  // the initial value of the unique slot corresponding to that attr.
  DenseMap<AttrOfClass, Value> slotInitialValues;
  // The inverse map of `slotInitialValues`.
  // Many attributes can have the same initial value, so the value type
  // is a vector.
  DenseMap<Value, std::vector<AttrOfClass>> slotInitialValuesInverseMap;

  // The torch.global_slot corresponding to each torch.attr/torch.slot.
  DenseMap<AttrOfClass, GlobalSlotOp> globalSlotForAttr;
  // The linkage name (value) for the function with symbol name (key).
  DenseMap<StringRef, std::string> methodLinkageNames;

  // The set of class types that have already been processed.
  // Used for diagnostics.
  // The map value is the original path from the root that we found it at.
  DenseMap</*ClassTypeOp*/ Operation *, std::string> seenClassTypes;

  // A set of values that we have copied into torch.global_slot initializers,
  // which cannot be used in multiple initializers because their object
  // identity is important.
  DenseSet<Value> objectsWithIdentityAlreadyCopiedIntoInitializers;
};
} // namespace

ObjectGraphGlobalizer::ObjectGraphGlobalizer(ModuleOp module)
    : module(module), symbolTable(module),
      globalBuilder(module.getBodyRegion()) {}

LogicalResult ObjectGraphGlobalizer::globalizeObjectGraph() {
  // We require there to be a unique root !torch.nn.Module.
  FailureOr<NnModuleOp> maybeRootNnModule = findRootNnModule();
  if (failed(maybeRootNnModule))
    return failure();
  NnModuleOp rootNnModule = *maybeRootNnModule;
  if (!rootNnModule)
    return module.emitError()
           << "module does not contain a root torch.nn_module";

  // We require one instance of each class. That is, there is a single
  // torch.nn_module for each torch.class_type.
  if (failed(checkSingleInstanceOfEachClass()))
    return failure();

  for (NnModuleOp nnModule : module.getOps<NnModuleOp>()) {
    auto classType = symbolTable.lookup<ClassTypeOp>(nnModule.getClassName());
    for (auto slot : nnModule.getOps<SlotOp>()) {
      AttrOfClass attrOfClass = {classType, slot.name()};
      slotInitialValues[attrOfClass] = slot.value();
      slotInitialValuesInverseMap[slot.value()].push_back(attrOfClass);
    }
  }

  // Recursively traverse the class hierarchy, globalizing slots and
  // tracking linkage names for methods.
  auto rootClassType =
      symbolTable.lookup<ClassTypeOp>(rootNnModule.getClassName());
  if (failed(recursivelyTraverseClassType(rootClassType)))
    return failure();

  // Rewrite torch.prim.GetAttr/torch.prim.SetAttr/torch.prim.CallMethod.
  if (failed(rewriteMethods()))
    return failure();

  // Now that all we have finished converting to the new form, remove
  // the original object graph.
  removeObjectGraph();

  return success();
}

FailureOr<NnModuleOp> ObjectGraphGlobalizer::findRootNnModule() {
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
  return rootNnModule;
}

LogicalResult ObjectGraphGlobalizer::checkSingleInstanceOfEachClass() {
  llvm::MapVector</*ClassTypeOp*/ Operation *, std::vector<NnModuleOp>>
      classInstances;
  for (NnModuleOp op : module.getOps<NnModuleOp>()) {
    auto classType = symbolTable.lookup<ClassTypeOp>(op.getClassName());
    classInstances[classType].push_back(op);
  }
  for (auto &p : classInstances) {
    ClassTypeOp classType = cast<ClassTypeOp>(p.first);
    ArrayRef<NnModuleOp> instances = p.second;
    if (instances.size() > 1) {
      // TODO: Improve this diagnostic based on user use cases.
      // This is a user-facing diagnostic that enforces key invariants to
      // our TorchScript subset.
      auto diag = classType.emitError(
          "class type has more than one instance: the current TorchScript "
          "supported subset only allows single instances");
      for (NnModuleOp instance : instances) {
        diag.attachNote(instance.getLoc()) << "see instance here";
      }
      return failure();
    }
  }
  return success();
}

LogicalResult
ObjectGraphGlobalizer::recursivelyTraverseClassType(ClassTypeOp classType) {
  std::string pathToClassFromRoot = llvm::join(nameStack, ".");
  if (!seenClassTypes.insert({classType, pathToClassFromRoot}).second) {
    return classType.emitError()
           << "reachable by multiple paths from root object: '<root>."
           << seenClassTypes[classType] << "' and '<root>."
           << pathToClassFromRoot << "'";
  }

  // For each attr, create a global slot for it.
  for (auto attr : classType.getOps<AttrOp>()) {
    nameStack.push_back(attr.name().str());
    if (auto type = attr.type().dyn_cast<NnModuleType>()) {
      if (failed(recursivelyTraverseClassType(
              symbolTable.lookup<ClassTypeOp>(type.getClassName()))))
        return failure();
    } else {
      auto linkageName = llvm::join(nameStack, ".");
      auto globalSlot = globalBuilder.create<GlobalSlotOp>(
          attr->getLoc(), linkageName, /*sym_visibility=*/nullptr,
          TypeAttr::get(attr.type()));
      if (attr.isPrivate())
        globalSlot.setVisibility(SymbolTable::Visibility::Private);
      AttrOfClass attrOfClass = {classType, attr.name()};
      assert(globalSlotForAttr.find(attrOfClass) == globalSlotForAttr.end());
      globalSlotForAttr[attrOfClass] = globalSlot;
      if (failed(populateGlobalSlotInitializer(globalSlot,
                                               slotInitialValues[attrOfClass])))
        return failure();
    }
    nameStack.pop_back();
  }
  // For each method, track the linkage name it will eventually have.
  for (auto method : classType.getOps<MethodOp>()) {
    nameStack.push_back(method.name().str());
    auto linkageName = llvm::join(nameStack, ".");
    nameStack.pop_back();
    if (!methodLinkageNames.insert({method.function(), linkageName}).second)
      return method.emitError()
             << "unbound function shared by multiple methods";
  }
  return success();
}

static bool hasMeaningfulObjectIdentity(Type type) {
  return !type.isa<IntegerType, FloatType, Basicpy::BoolType,
                   Basicpy::BytesType, TensorType>();
}

LogicalResult
ObjectGraphGlobalizer::populateGlobalSlotInitializer(GlobalSlotOp globalSlot,
                                                     Value initialValue) {
  OpBuilder builder(globalSlot.getContext());
  builder.createBlock(&globalSlot.getRegion());

  SmallPtrSet<Operation *, 6> needToClone;
  SmallVector<Operation *> worklist = {initialValue.getDefiningOp()};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!needToClone.insert(op).second)
      continue;
    for (Value operand : op->getOperands()) {
      if (auto def = operand.getDefiningOp())
        worklist.push_back(def);
    }
  }
  worklist.assign(needToClone.begin(), needToClone.end());
  llvm::sort(worklist, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  BlockAndValueMapping mapping;
  for (Operation *op : worklist) {
    builder.clone(*op, mapping);
    for (Value result : op->getResults()) {
      if (!hasMeaningfulObjectIdentity(result.getType()))
        continue;
      if (!objectsWithIdentityAlreadyCopiedIntoInitializers.insert(result)
               .second) {
        return op->emitError()
               << "potentially-aliased value used to initialize multiple slots";
      }
    }
  }
  builder.create<GlobalSlotInitOp>(globalSlot->getLoc(),
                                   mapping.lookup(initialValue));
  return success();
}

// Verify that a value conforms to the subset of allowed uses for
// !torch.nn.Module<"..."> types.
static LogicalResult verifyNnModuleValueUses(Value value) {
  // Trivially true for non-module types.
  if (!value.getType().isa<NnModuleType>())
    return success();
  for (Operation *op : value.getUsers()) {
    if (isa<CallOp, PrimGetAttrOp, PrimSetAttrOp, PrimCallMethodOp>(op))
      continue;
    // TODO: Improve this based on real user use cases.
    // This is a diagnostic that users will hit if they do not conform to
    // the supported subset of TorchScript.
    return op->emitError() << "unsupported use of a torch.nn.Module. Expected "
                              "only method calls or attribute get/set";
  }
  return success();
}

static std::string getNonMethodMangledFunctionName(StringRef originalName) {
  return "__npcomp_priv_fn$" + originalName.str();
}

// Verify that `func` conforms to the subset of allowable method bodies
// that we can convert.
static LogicalResult verifyFuncConformsToSubset(FuncOp func) {
  auto walkResult = func.walk([](Block *block) {
    for (Value arg : block->getArguments()) {
      if (failed(verifyNnModuleValueUses(arg)))
        return WalkResult::interrupt();
    }
    for (Operation &op : *block) {
      for (Value result : op.getResults()) {
        if (failed(verifyNnModuleValueUses(result)))
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

LogicalResult ObjectGraphGlobalizer::rewriteMethods() {
  DenseMap<AttrOfClass, StringRef> linkageNames;
  for (auto classType : module.getOps<ClassTypeOp>()) {
    for (auto method : classType.getOps<MethodOp>()) {
      auto it = methodLinkageNames.find(method.function());
      if (it == methodLinkageNames.end())
        continue;
      linkageNames[{classType, method.name()}] = it->second;
    }
  }
  // We only handle a small subset of ops that conform with the set of
  // assumptions that allow us to globalize the object graph. Anything that
  // tries to treat modules as bona-fide objects and not just namespaces
  // of methods with a single instance of the corresponding type just gets
  // arbitrarily tricky to rewrite. E.g. what if the user creates a list
  // of modules, or there is an scf.if selecting between modules, etc.
  SmallVector<Operation *> toErase;
  auto rewriteOpWithNnModuleTypeOperand = [&](Operation *op) {
    if (auto primSetAttr = dyn_cast<PrimSetAttrOp>(op)) {
      auto classType = symbolTable.lookup<ClassTypeOp>(
          primSetAttr.receiver().getType().cast<NnModuleType>().getClassName());
      auto globalSlot = globalSlotForAttr[{classType, primSetAttr.name()}];
      OpBuilder(primSetAttr)
          .create<GlobalSlotSetOp>(primSetAttr.getLoc(), globalSlot.sym_name(),
                                   primSetAttr.value());
      toErase.push_back(primSetAttr);
    } else if (auto primGetAttr = dyn_cast<PrimGetAttrOp>(op)) {
      // If the return value is NnModuleType, then we don't need to do anything.
      // Our verification earlier ensured that there are no uses that
      // we won't properly rewrite.
      if (!primGetAttr.getType().isa<NnModuleType>()) {
        auto classType =
            symbolTable.lookup<ClassTypeOp>(primGetAttr.receiver()
                                                .getType()
                                                .cast<NnModuleType>()
                                                .getClassName());
        auto globalSlot = globalSlotForAttr[{classType, primGetAttr.name()}];
        auto globalSlotGet =
            OpBuilder(primGetAttr)
                .create<GlobalSlotGetOp>(primGetAttr.getLoc(),
                                         primGetAttr.getType(),
                                         globalSlot.sym_name());
        primGetAttr.replaceAllUsesWith(globalSlotGet.getOperation());
      }
      toErase.push_back(primGetAttr);
    } else if (auto primCallMethod = dyn_cast<PrimCallMethodOp>(op)) {
      auto classType = symbolTable.lookup<ClassTypeOp>(primCallMethod.receiver()
                                                           .getType()
                                                           .cast<NnModuleType>()
                                                           .getClassName());
      StringRef linkageName = linkageNames[{classType, primCallMethod.name()}];

      auto newOperands = llvm::to_vector<6>(
          llvm::make_filter_range(primCallMethod.operands(), [](Value v) {
            return !v.getType().isa<NnModuleType>();
          }));
      auto call = OpBuilder(primCallMethod)
                      .create<CallOp>(primCallMethod.getLoc(), linkageName,
                                      primCallMethod.getType(), newOperands);
      primCallMethod.replaceAllUsesWith(call);
      toErase.push_back(primCallMethod);
    } else if (auto callOp = dyn_cast<CallOp>(op)) {
      auto newOperands = llvm::to_vector<6>(
          llvm::make_filter_range(callOp.operands(), [](Value v) {
            return !v.getType().isa<NnModuleType>();
          }));
      auto newCallOp = OpBuilder(callOp).create<CallOp>(
          callOp.getLoc(), getNonMethodMangledFunctionName(callOp.callee()),
          callOp.getResultTypes(), newOperands);
      callOp.replaceAllUsesWith(newCallOp);
      toErase.push_back(callOp);
    }
  };
  struct MethodFuncRewrite {
    bool isPrivate;
    std::string linkageName;
  };

  DenseMap<FuncOp, MethodFuncRewrite> methodFuncRewrites;
  for (auto classType : module.getOps<ClassTypeOp>()) {
    for (auto method : classType.getOps<MethodOp>()) {
      auto it = methodLinkageNames.find(method.function());
      if (it == methodLinkageNames.end())
        continue;
      FuncOp func = symbolTable.lookup<FuncOp>(method.function());
      methodFuncRewrites[func] =
          MethodFuncRewrite{method.isPrivate(), it->second};
    }
  }
  for (auto func : module.getOps<FuncOp>()) {
    if (failed(verifyFuncConformsToSubset(func)))
      return failure();
    func.walk(rewriteOpWithNnModuleTypeOperand);
    for (Operation *op : toErase) {
      op->dropAllDefinedValueUses();
      op->erase();
    }
    toErase.clear();
    SmallVector<unsigned> argsToErase;
    for (auto arg : llvm::enumerate(func.getArguments())) {
      if (!arg.value().getType().isa<NnModuleType>())
        continue;
      assert(arg.value().use_empty() && "all uses should have been removed");
      argsToErase.push_back(arg.index());
    }
    func.eraseArguments(argsToErase);
    // No need to handle the results, since we currently don't allow ReturnOp
    // as a user of module types.

    // Adjust the linkage names to adopt the linkage convention of this pass,
    // namely that only objects accessible from a root !torch.nn.Module are
    // possible external linkage candidates, and their linkage name is the
    // dotted path from the root.
    //
    // Any other function gets a prefix to avoid collisions. These other
    // functions correspond to free functions somewhere outside the module
    // hierarchy.
    auto it = methodFuncRewrites.find(func);
    if (it != methodFuncRewrites.end()) {
      if (!it->second.isPrivate)
        func.setVisibility(SymbolTable::Visibility::Public);
      func.setName(it->second.linkageName);
    } else {
      func.setName(getNonMethodMangledFunctionName(func.getName()));
    }
  }

  return success();
}

void ObjectGraphGlobalizer::removeObjectGraph() {
  for (Operation &op : llvm::make_early_inc_range(*module.getBody())) {
    if (!isa<FuncOp, GlobalSlotOp, ModuleTerminatorOp>(op)) {
      op.dropAllDefinedValueUses();
      op.erase();
    }
  }
}

namespace {
class GlobalizeObjectGraphPass
    : public GlobalizeObjectGraphBase<GlobalizeObjectGraphPass> {
  void runOnOperation() override {
    if (failed(ObjectGraphGlobalizer(getOperation()).globalizeObjectGraph()))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::Torch::createGlobalizeObjectGraphPass() {
  return std::make_unique<GlobalizeObjectGraphPass>();
}
