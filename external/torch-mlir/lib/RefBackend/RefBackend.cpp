//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    } else if (auto integerTy = elemTy.dyn_cast<IntegerType>()) {
      if (integerTy.isSignlessInteger(64))
        return true;
    }
  }
  return false;
}

static bool isReturnMemRefTypeValid(Type type) {
  if (auto memRefType = type.dyn_cast<MemRefType>()) {
    if (memRefType.getElementType().isa<Float32Type>()) {
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

static LogicalResult mungeFunction(FuncOp func, FuncOp consumeFuncReturnFunc) {
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
      return emitError(arg.getLoc(), "argument must be a memref of f32 or i64");
    auto cast = b.create<memref::CastOp>(arg.getLoc(), arg, type);
    arg.replaceAllUsesExcept(cast, cast);
    arg.setType(getAbiTypeForMemRef(type));
    newArgTypes.push_back(arg.getType());
  }

  SmallVector<Operation *> toErase;
  bool hadError = false;
  func.walk([&](ReturnOp op) {
    if (op.getNumOperands() != 1 ||
        !isReturnMemRefTypeValid(op.getOperandTypes()[0])) {
      hadError = true;
      op.emitError("must have one return value and it must be a memref of f32");
      return;
    }
    b.setInsertionPoint(op);
    auto cast =
        b.create<memref::CastOp>(op.getLoc(), op.getOperand(0),
                                 getAbiTypeForMemRef(op.getOperandTypes()[0]));
    b.create<mlir::CallOp>(op.getLoc(), consumeFuncReturnFunc,
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

    auto consumeFuncReturnFunc = b.create<FuncOp>(
        module.getLoc(), "refbackend_consume_func_return",
        FunctionType::get(
            module.getContext(),
            UnrankedMemRefType::get(b.getF32Type(), /*memorySpace=*/0), {}),
        b.getStringAttr("private"));
    addEmitCInterfaceAttr(consumeFuncReturnFunc);
    for (auto func : module.getOps<FuncOp>()) {
      if (func == consumeFuncReturnFunc)
        continue;
      if (failed(mungeFunction(func, consumeFuncReturnFunc)))
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
    ConversionTarget target(*context);
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addIllegalOp<math::TanhOp>();
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
