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
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "set"
#include "torch-mlir/RefBackend/Passes.h"
#include <numeric>

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
      if (integerTy.isSignlessInteger(32))
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

// Helper function to get the type string for one return value like i32, f64,
// mri32 etc. The strings from multiple return values are concatenated to get
// the consumeFuncReturnFunc name.
static std::string getTypeToken(Type type) {
  if (type.isSignlessInteger())
    return ("i" + Twine(type.getIntOrFloatBitWidth())).str();
  else if (type.isa<mlir::FloatType>())
    return ("f" + Twine(type.getIntOrFloatBitWidth())).str();
  else if (auto memRefType = type.dyn_cast<UnrankedMemRefType>())
    return "mr" + getTypeToken(memRefType.getElementType());

  llvm_unreachable(
      "Type token should handle all types: memref, float and int type");
}

// Systematically derive the consumeFuncReturnFunc name from return value types.
static std::string getConsumeReturnFunctionNameForReturnTypes(TypeRange types) {
  SmallVector<std::string> tokens = {"refbackend_consume_func_return"};
  for (auto type : types)
    tokens.push_back(getTypeToken(type));

  return std::accumulate(tokens.begin(), tokens.end(), std::string(),
                         [](std::string &a, std::string &b) {
                           return a.empty() ? b : (a + "_" + b);
                         });
}

// Replace the original returnOp with a call to consumeFuncReturnFunc and add
// the op to the `toErase` vector.
static void replaceReturnWithCall(OpBuilder b, ReturnOp op, StringRef funcName,
                                  TypeRange retTypes,
                                  SmallVectorImpl<Value> &vals,
                                  SmallVectorImpl<Operation *> &toErase) {
  b.create<mlir::CallOp>(op.getLoc(), funcName, TypeRange({}), vals);
  b.create<mlir::ReturnOp>(op.getLoc());
  toErase.push_back(op);
}

static LogicalResult mungeFunction(
    FuncOp func, std::set<std::string> &supportedConsumeFuncReturnFuncs,
    std::map<std::string, std::vector<Type>> &invokedConsumeFuncReturnFuncs) {
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
                       "argument must be a memref of f32, f64, i32, i64");
    auto cast = b.create<memref::CastOp>(arg.getLoc(), arg, type);
    arg.replaceAllUsesExcept(cast, cast);
    arg.setType(getAbiTypeForMemRef(type));
    newArgTypes.push_back(arg.getType());
  }

  SmallVector<Operation *> toErase;
  bool isSupported = true;
  func.walk([&](ReturnOp op) {
    auto types = op.getOperandTypes();
    b.setInsertionPoint(op);
    // Memref Types.
    std::vector<Type> retTypes;
    SmallVector<Value> retVals;
    for (auto en : llvm::enumerate(types)) {
      Type retType = en.value();
      Value retVal = op.getOperand(en.index());
      if (auto memrefReturnType = retType.dyn_cast<MemRefType>()) {
        auto elemType = memrefReturnType.getElementType();
        retType = UnrankedMemRefType::get(elemType, 0);
        // Cast to unranked memref type before sending it as a function
        // argument.
        retVal = b.create<memref::CastOp>(
            op.getLoc(), retVal, getAbiTypeForMemRef(types[en.index()]));
      }
      retTypes.push_back(retType);
      retVals.push_back(retVal);
    }

    auto supportedFuncsEnd = supportedConsumeFuncReturnFuncs.end();
    std::string funcName = getConsumeReturnFunctionNameForReturnTypes(retTypes);
    if (supportedConsumeFuncReturnFuncs.find(funcName) == supportedFuncsEnd) {
      op.emitError(
          "must have one return value of memref types or scalar types "
          "of i32, i64, f32, f64 or three return values of memref f32");
      isSupported = false;
    }

    auto invokedFuncsEnd = invokedConsumeFuncReturnFuncs.end();
    if (invokedConsumeFuncReturnFuncs.find(funcName) == invokedFuncsEnd)
      invokedConsumeFuncReturnFuncs.insert({funcName, retTypes});
    replaceReturnWithCall(b, op, funcName, retTypes, retVals, toErase);
  });
  if (!isSupported)
    return failure();
  func.setType(FunctionType::get(func.getContext(), newArgTypes, {}));
  for (Operation *op : toErase)
    op->erase();
  return success();
}

static std::set<std::string> getSupportedConsumeFuncReturnFuncs(OpBuilder &b) {
  std::set<std::string> funcNames;
  Type mri32 = UnrankedMemRefType::get(b.getI32Type(), 0);
  Type mri64 = UnrankedMemRefType::get(b.getI64Type(), 0);
  Type mrf32 = UnrankedMemRefType::get(b.getF32Type(), 0);
  Type mrf64 = UnrankedMemRefType::get(b.getF64Type(), 0);
  Type i64 = b.getI64Type();
  Type f32 = b.getF32Type();
  Type f64 = b.getF64Type();

  SmallVector<TypeRange> supportedReturnTypes = {
      mri32, mri64, mrf32, mrf64, i64, f32, f64, {mrf32, mrf32, mrf32}};

  llvm::for_each(supportedReturnTypes, [&](TypeRange &types) {
    funcNames.insert(getConsumeReturnFunctionNameForReturnTypes(types));
  });
  return funcNames;
}

namespace {
class MungeCallingConventions
    : public MungeCallingConventionsBase<MungeCallingConventions> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getBodyRegion());
    static std::set<std::string> supported =
        getSupportedConsumeFuncReturnFuncs(b);
    std::map<std::string, std::vector<Type>> invokedConsumeFuncReturnFuncs;
    for (auto func : module.getOps<FuncOp>()) {
      if (failed(mungeFunction(func, supported, invokedConsumeFuncReturnFuncs)))
        return signalPassFailure();
    }

    // Create FuncOp for consumeFuncReturnFuncs that are used.
    for (auto &p : invokedConsumeFuncReturnFuncs) {
      auto consumeFuncReturnFunc =
          b.create<FuncOp>(module.getLoc(), p.first,
                           FunctionType::get(module.getContext(), p.second, {}),
                           b.getStringAttr("private"));
      addEmitCInterfaceAttr(consumeFuncReturnFunc);
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
