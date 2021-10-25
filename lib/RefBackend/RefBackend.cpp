//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// The torch-mlir "reference backend" requires a few passes to glue things
// together so that the final IR will work with ExecutionEngine.
//
// There is no actual "backend".
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/RefBackend/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::RefBackend;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/RefBackend/Passes.h.inc"
} // end namespace

void mlir::torch::RefBackend::registerRefBackendPasses() { ::registerPasses(); }

//===----------------------------------------------------------------------===//
// MungeCallingConventions
//===----------------------------------------------------------------------===//

static bool isArgMemRefTypeValid(Type type) {
  if (auto memRefType = type.dyn_cast<MemRefType>()) {
    Type elemTy = memRefType.getElementType();
    if (elemTy.isa<Float32Type>()) {
      return true;
    } else if (elemTy.isa<Float64Type>()) {
      return true;
    } else if (auto integerTy = elemTy.dyn_cast<IntegerType>()) {
      if (integerTy.isSignlessInteger(64))
        return true;
    }
  }
  return false;
}

static void addEmitCInterfaceAttr(FuncOp func) {
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(func.getContext()));
}

static Type getAbiTypeForMemRef(Type type) {
  return UnrankedMemRefType::get(type.cast<MemRefType>().getElementType(), 0);
}

static LogicalResult mungeFunction(
    FuncOp func,
    DenseMap</*returnElementType*/ Type, FuncOp> consumeFuncReturnFuncs) {
  // Add `llvm.emit_c_interface`.
  // This allows ExecutionEngine to resolve the symbol properly.
  addEmitCInterfaceAttr(func);

  // Rewrite the function as follows:
  // - replace all memref arguments with unranked memref
  // - replace all returns with a call to a function, which is going to be
  //   supplied by the code setting up the ExecutionEngine to process the
  //   result. Additionally, ensure that all results are passed as unranked
  //   memrefs.
  // - replace the function signature accordingly (unranked inputs, no returns).
  OpBuilder b(func.getBody());

  SmallVector<Type> newArgTypes;
  for (auto arg : func.getArguments()) {
    auto type = arg.getType();
    if (!isArgMemRefTypeValid(type))
      return emitError(arg.getLoc(),
                       "argument must be a memref of f32, f64, i64");
    auto cast = b.create<memref::CastOp>(arg.getLoc(), arg, type);
    arg.replaceAllUsesExcept(cast, cast);
    arg.setType(getAbiTypeForMemRef(type));
    newArgTypes.push_back(arg.getType());
  }

  SmallVector<Operation *> toErase;
  bool hadError = false;
  func.walk([&](ReturnOp op) {
    auto memRefType = op.getOperandTypes()[0].dyn_cast<MemRefType>();
    if (!memRefType) {
      hadError = true;
      op.emitError("return value must be memref type");
      return;
    }
    auto returnType = memRefType.getElementType();
    auto it = consumeFuncReturnFuncs.find(returnType);
    if (op.getNumOperands() != 1 || it == consumeFuncReturnFuncs.end()) {
      hadError = true;
      op.emitError("must have one return value: a memref of f32, i64 or f64");
      return;
    }

    b.setInsertionPoint(op);
    auto cast =
        b.create<memref::CastOp>(op.getLoc(), op.getOperand(0),
                                 getAbiTypeForMemRef(op.getOperandTypes()[0]));
    b.create<mlir::CallOp>(op.getLoc(), consumeFuncReturnFuncs[returnType],
                           cast.getResult());
    b.create<mlir::ReturnOp>(op.getLoc());
    toErase.push_back(op);
  });
  if (hadError)
    return failure();

  func.setType(FunctionType::get(func.getContext(), newArgTypes, {}));

  for (Operation *op : toErase)
    op->erase();

  return success();
}

namespace {
class MungeCallingConventions
    : public MungeCallingConventionsBase<MungeCallingConventions> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getBodyRegion());
    DenseMap</*returnElementType*/ Type, FuncOp> consumeFuncReturnFuncs;
    DenseSet<FuncOp> consumeFuncReturnFuncsSet;
    auto createConsumeFuncReturnFunc = [&](Type elemTy, std::string funcName) {
      auto consumeFuncReturnFunc = b.create<FuncOp>(
          module.getLoc(), funcName,
          FunctionType::get(module.getContext(),
                            UnrankedMemRefType::get(elemTy, /*memorySpace=*/0),
                            {}),
          b.getStringAttr("private"));
      addEmitCInterfaceAttr(consumeFuncReturnFunc);
      consumeFuncReturnFuncs[elemTy] = consumeFuncReturnFunc;
      consumeFuncReturnFuncsSet.insert(consumeFuncReturnFunc);
    };
    createConsumeFuncReturnFunc(b.getI64Type(),
                                "refbackend_consume_int64_func_return");
    createConsumeFuncReturnFunc(b.getF32Type(),
                                "refbackend_consume_float32_func_return");
    createConsumeFuncReturnFunc(b.getF64Type(),
                                "refbackend_consume_float64_func_return");
    for (auto func : module.getOps<FuncOp>()) {
      if (consumeFuncReturnFuncsSet.contains(func))
        continue;
      if (failed(mungeFunction(func, consumeFuncReturnFuncs)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::RefBackend::createMungeCallingConventionsPass() {
  return std::make_unique<MungeCallingConventions>();
}

//===----------------------------------------------------------------------===//
// ExpandOpsForLLVM
//===----------------------------------------------------------------------===//

namespace {
class ExpandOpsForLLVM : public ExpandOpsForLLVMBase<ExpandOpsForLLVM> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    populateExpandTanhPattern(patterns);
    patterns.add<math::ErfPolynomialApproximation>(patterns.getContext());
    ConversionTarget target(*context);
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addIllegalOp<math::TanhOp>();
    target.addIllegalOp<math::ErfOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::RefBackend::createExpandOpsForLLVMPass() {
  return std::make_unique<ExpandOpsForLLVM>();
}
