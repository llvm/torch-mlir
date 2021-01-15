//===- BuildInitFunc.cpp - Extracts a pipeline definition ---*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>

#include "PassDetail.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "npcomp/Dialect/RD/IR/RDDatasetInterface.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "npcomp/Dialect/RD/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::NPCOMP;

#define DEBUG_TYPE "rd-build-init-func"

namespace {
/// The ordered list of dataset operations that define a dataset.
using PipelineDefinition = llvm::SmallVector<mlir::Operation*, 6>;

/// The ordered list of dataset ops within the @definition func op.
PipelineDefinition extractDatasetOps(FuncOp definition) {
  PipelineDefinition ops;
  definition.walk([&](mlir::Operation *op) {
    if (op == definition.getOperation() || op->getDialect()->getNamespace() == "std") {
      return;  // Skip these.
    }
    // Note: we include all other ops to allow other dialects to define dataset ops.
    ops.push_back(op);
  });
  return ops;
}

/// The state types corresponding to each dataset transformation's iterator state.
///
/// Note: because these types are used in a struct, we use a 0-sized int8 array to
/// represent an absence of state (as opposed to LLVM's Void Type).
SmallVector<LLVM::LLVMType, 6> computeStateTypeEntries(FuncOp definitionFunc, PipelineDefinition datasetOps) {
  SmallVector<LLVM::LLVMType, 6> entries;

  for (mlir::Operation *op : datasetOps) {
    if (rd::DatasetTransform datasetOp = dyn_cast<rd::DatasetTransform>(op)) {
      if (auto elem = datasetOp.buildStateLLVMType()) {
        entries.push_back(*elem);
      } else {
        // Use [u_int8; 0] arrays as padding.
        auto charTy = LLVM::LLVMType::getInt8Ty(definitionFunc.getContext());
        auto emptyCharArray = LLVM::LLVMType::getArrayTy(charTy, 0);
        entries.push_back(emptyCharArray);
      }
    } else {
      op->emitError(
          "Unexpected op; please add the DatasetTransformOpInterface to the op.");
    }
  }
  return entries;
}

/// Builds function to initialize iterator state.
///
/// Arguments:
///  - initFuncOp: the function to build into.
///  - datasetOps: the ordered list of ops that constitutes the pipeline definition.
///  - statePtr: a value that points to the struct for the state of the iterator.
///  - argMap: a mapping between values in the definition function and the init function.
void buildIteratorInit(LLVM::LLVMFuncOp initFuncOp, FuncOp definitionOp,
                       PipelineDefinition datasetOps) {
  Block *initBody = initFuncOp.addEntryBlock();
  auto builder = OpBuilder::atBlockBegin(initBody);

  rd::InitArgMap argMap;
  assert(definitionOp.getNumArguments() + 1 == initFuncOp.getNumArguments()
    && "expected one extra argument for return pointer");
  assert(initBody->getNumArguments() == initFuncOp.getNumArguments()
    && "expected argument counts to correspond.");
  unsigned arg_count = initBody->getNumArguments();
  argMap.reserve(arg_count);
  for (unsigned int i = 0; i < arg_count - 1; ++i) {
    argMap[definitionOp.getArgument(i)] = initBody->getArgument(i);
  }
  auto statePtr =
      initBody->getArgument(initBody->getNumArguments() - 1); // Last arg.
  auto int64Ty = LLVM::LLVMType::getInt64Ty(definitionOp.getContext());
  auto zeroValue = builder.create<LLVM::ConstantOp>(definitionOp.getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));


  for (unsigned int i = 0; i < datasetOps.size(); ++i) {
    auto* op = datasetOps[i];
    if (rd::DatasetTransform datasetOp = dyn_cast<rd::DatasetTransform>(op)) {
      if (auto stateTy = datasetOp.buildStateLLVMType()) {
        auto iValue = builder.create<LLVM::ConstantOp>(op->getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), i));
        auto ptr = builder.create<LLVM::GEPOp>(op->getLoc(), stateTy->getPointerTo(), statePtr, ValueRange({zeroValue, iValue}));  // TODO: FIX ME!
        datasetOp.buildInitState(builder, ptr, argMap);
      } else {
        // Nothing to do here.
      }
    } else {
      op->emitError("Unexpected op; please add the DatasetTransformOpInterface to the op.");
    }
  }
  builder.create<LLVM::ReturnOp>(initFuncOp.getLoc(), ValueRange({}));
}

class BuildInitFunc : public RDBuildInitFuncBase<BuildInitFunc> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto* context = &getContext();
    auto pipelineDefOp = getOperation();
    auto defFuncOpt = findDefinitionFunc(pipelineDefOp);
    if (!defFuncOpt) {
      return signalPassFailure();
    }
    auto defFunc = *defFuncOpt;
    auto datasetOps = extractDatasetOps(defFunc);
    auto iteratorTypeEntries = computeStateTypeEntries(defFunc, datasetOps);
    auto stateTy = LLVM::LLVMType::getStructTy(&getContext(), iteratorTypeEntries);

    LLVMTypeConverter typeConverter(context);
    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    LLVMConversionTarget target(getContext());

    llvm::SmallVector<Type, 4> initFuncArgs;
    for (auto& arg : defFunc.getArguments()) {
      initFuncArgs.push_back(arg.getType());
    }
    // Do "return value to argument pointer" conversion by appending the stateTy pointer to the argumnt list.
    initFuncArgs.push_back(stateTy.getPointerTo());
    // TODO: add context pointers & allow for a (nullable) error pointer to be returned.
    auto initFuncStdTy =
        FunctionType::get(TypeRange(initFuncArgs), TypeRange(), context);
    TypeConverter::SignatureConversion init_conversion_result(initFuncArgs.size());
    auto initFuncTy = typeConverter.convertFunctionSignature(
        initFuncStdTy, false, init_conversion_result);

    OpBuilder builder(context);
    builder.setInsertionPointAfter(defFunc);
    auto initFuncOp = builder.create<LLVM::LLVMFuncOp>(
      pipelineDefOp.getLoc(), "init", initFuncTy);
    buildIteratorInit(initFuncOp, defFunc, datasetOps);
  }
};
} // namespace

std::unique_ptr<OperationPass<rd::PipelineDefinitionOp>> mlir::NPCOMP::createBuildInitFuncPass() {
  return std::make_unique<BuildInitFunc>();
}
