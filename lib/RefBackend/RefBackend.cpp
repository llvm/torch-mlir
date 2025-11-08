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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/RefBackend/Passes.h"
#include <numeric>
#include <set>

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
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    Type elemTy = memRefType.getElementType();
    if (isa<Float16Type, Float32Type, Float64Type>(elemTy)) {
      return true;
    } else if (auto integerTy = dyn_cast<IntegerType>(elemTy)) {
      if (integerTy.isSignlessInteger(64))
        return true;
      if (integerTy.isSignlessInteger(32))
        return true;
      if (integerTy.isSignlessInteger(8))
        return true;
      if (integerTy.isSignedInteger(8))
        return true;
      if (integerTy.isSignlessInteger(1))
        return true;
    } else if (auto complexTy = dyn_cast<ComplexType>(elemTy)) {
      return isa<Float32Type, Float64Type>(complexTy.getElementType());
    }
  }
  return false;
}

static void addEmitCInterfaceAttr(func::FuncOp func) {
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(func.getContext()));
}

static Type getAbiTypeForMemRef(Type type) {
  return UnrankedMemRefType::get(cast<MemRefType>(type).getElementType(), 0);
}

// Helper function to get the type string for one return value like i32, f64,
// mri32 etc. The strings from multiple return values are concatenated to get
// the consumeFuncReturnFunc name.
static std::string getTypeToken(Type type) {
  if (type.isSignlessInteger())
    return ("i" + Twine(type.getIntOrFloatBitWidth())).str();
  else if (isa<mlir::FloatType>(type))
    return ("f" + Twine(type.getIntOrFloatBitWidth())).str();
  else if (auto complexTy = dyn_cast<mlir::ComplexType>(type))
    return ("c" + Twine(complexTy.getElementType().getIntOrFloatBitWidth()))
        .str();
  else if (auto memRefType = dyn_cast<UnrankedMemRefType>(type))
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
                         [](std::string a, std::string b) {
                           return a.empty() ? b : (a + "_" + b);
                         });
}

// Replace the original returnOp with a call to consumeFuncReturnFunc and add
// the op to the `toErase` vector.
static void replaceReturnWithCall(OpBuilder b, func::ReturnOp op,
                                  StringRef funcName, TypeRange retTypes,
                                  SmallVectorImpl<Value> &vals,
                                  SmallVectorImpl<Operation *> &toErase) {
  mlir::func::CallOp::create(b, op.getLoc(), funcName, TypeRange({}), vals);
  mlir::func::ReturnOp::create(b, op.getLoc());
  toErase.push_back(op);
}

static LogicalResult mungeFunction(
    func::FuncOp func,
    std::map<std::string, std::vector<Type>> &invokedConsumeFuncReturnFuncs) {
  // Only need to call mungeFunction for functions callable from outside of the
  // module.
  if (func.isPrivate())
    return success();
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
    if (!isArgMemRefTypeValid(type)) {
      return emitError(arg.getLoc())
          .append("argument must be a memref of f32, f64, i32, i64, i8, i1, "
                  "c32, c64, but "
                  "got ",
                  type);
    }
    auto cast = memref::CastOp::create(b, arg.getLoc(), type, arg);
    arg.replaceAllUsesExcept(cast, cast);
    arg.setType(getAbiTypeForMemRef(type));
    newArgTypes.push_back(arg.getType());
  }

  SmallVector<Operation *> toErase;
  func.walk([&](func::ReturnOp op) {
    auto types = op.getOperandTypes();
    b.setInsertionPoint(op);
    // Memref Types.
    std::vector<Type> retTypes;
    SmallVector<Value> retVals;
    for (auto en : llvm::enumerate(types)) {
      Type retType = en.value();
      Value retVal = op.getOperand(en.index());
      if (auto memrefReturnType = dyn_cast<MemRefType>(retType)) {
        auto elemType = memrefReturnType.getElementType();
        retType = UnrankedMemRefType::get(elemType, 0);
        // Cast to unranked memref type before sending it as a function
        // argument.
        retVal = memref::CastOp::create(
            b, op.getLoc(), getAbiTypeForMemRef(types[en.index()]), retVal);
      }
      retTypes.push_back(retType);
      retVals.push_back(retVal);
    }

    std::string funcName = getConsumeReturnFunctionNameForReturnTypes(retTypes);

    auto invokedFuncsEnd = invokedConsumeFuncReturnFuncs.end();
    if (invokedConsumeFuncReturnFuncs.find(funcName) == invokedFuncsEnd)
      invokedConsumeFuncReturnFuncs.insert({funcName, retTypes});
    replaceReturnWithCall(b, op, funcName, retTypes, retVals, toErase);
  });
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
    std::map<std::string, std::vector<Type>> invokedConsumeFuncReturnFuncs;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (failed(mungeFunction(func, invokedConsumeFuncReturnFuncs)))
        return signalPassFailure();
    }

    // Create FuncOp for consumeFuncReturnFuncs that are used.
    for (auto &p : invokedConsumeFuncReturnFuncs) {
      auto consumeFuncReturnFunc = func::FuncOp::create(
          b, module.getLoc(), p.first,
          FunctionType::get(module.getContext(), p.second, {}));
      consumeFuncReturnFunc.setPrivate();
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
// MLProgramBufferize
//===----------------------------------------------------------------------===//

static LogicalResult bufferizeMLProgramGlobalOp(ml_program::GlobalOp globalOp,
                                                OpBuilder &b) {
  if (!globalOp.getValue().has_value())
    return globalOp.emitError("global op must have a value");

  RankedTensorType tensorType = cast<RankedTensorType>(globalOp.getType());
  MemRefType memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  b.setInsertionPointToStart(globalOp->getParentOfType<ModuleOp>().getBody());
  memref::GlobalOp::create(b, UnknownLoc::get(b.getContext()),
                           globalOp.getSymName(),
                           /*sym_visibility=*/globalOp.getSymVisibilityAttr(),
                           /*type=*/memrefType,
                           /*initial_value=*/globalOp.getValue().value(),
                           /*constant=*/globalOp.getIsMutable() ? false : true,
                           /*alignment=*/nullptr);
  return success();
}

static LogicalResult
bufferizeMLProgramGlobaLoadOp(ml_program::GlobalLoadOp globalLoadOp,
                              OpBuilder &b, SmallVector<Operation *> &toErase) {
  RankedTensorType tensorType = cast<RankedTensorType>(globalLoadOp.getType());
  MemRefType memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  b.setInsertionPoint(globalLoadOp);
  Value globalVal = memref::GetGlobalOp::create(
      b, globalLoadOp.getLoc(), memrefType,
      globalLoadOp.getGlobalAttr().getLeafReference());
  globalVal = bufferization::ToTensorOp::create(b, globalLoadOp->getLoc(),
                                                tensorType, globalVal);
  globalLoadOp->getResult(0).replaceAllUsesWith(globalVal);
  return success();
}

static LogicalResult
bufferizeMLProgramGlobaStoreOp(ml_program::GlobalStoreOp globalStoreOp,
                               OpBuilder &b,
                               SmallVector<Operation *> &toErase) {
  RankedTensorType tensorType =
      cast<RankedTensorType>(globalStoreOp.getValue().getType());
  MemRefType memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  b.setInsertionPoint(globalStoreOp);
  Value memref = memref::GetGlobalOp::create(
      b, globalStoreOp.getLoc(), memrefType,
      globalStoreOp.getGlobalAttr().getLeafReference());
  Value copyValue = bufferization::ToBufferOp::create(
      b, globalStoreOp->getLoc(), memrefType, globalStoreOp.getValue());
  memref::CopyOp::create(b, globalStoreOp->getLoc(), copyValue, memref);
  return success();
}

namespace {
/// Converts MLProgram operations that work on tensor-type operands or results
/// to work on buffers.
class MLProgramBufferize : public MLProgramBufferizeBase<MLProgramBufferize> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getBodyRegion());
    SmallVector<Operation *> toErase;

    auto walkResult = module.walk([&](ml_program::GlobalOp op) {
      if (auto type = dyn_cast<RankedTensorType>(op.getType())) {
        if (!type.hasStaticShape()) {
          // If the ml_program.global has dynamically shaped tensor.
          op.emitError(
              "unimplemented: global op bufferization with dynamic shape");
          return WalkResult::interrupt();
        }
      } else {
        // If the ml_program.global is of non-tensor type.
        op.emitError("unsupported global op type");
        return WalkResult::interrupt();
      }

      if (failed(bufferizeMLProgramGlobalOp(op, b))) {
        op.emitError("bufferization for this op failed");
        return WalkResult::interrupt();
      }
      toErase.push_back(op);
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();

    module.walk([&](ml_program::GlobalLoadOp op) {
      if (failed(bufferizeMLProgramGlobaLoadOp(op, b, toErase))) {
        op.emitError("bufferization for this op failed");
        return;
      }
      toErase.push_back(op);
    });

    module.walk([&](ml_program::GlobalStoreOp op) {
      if (failed(bufferizeMLProgramGlobaStoreOp(op, b, toErase))) {
        op.emitError("bufferization for this op failed");
        return;
      }
      toErase.push_back(op);
    });

    for (auto op : llvm::reverse(toErase))
      op->erase();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::RefBackend::createMLProgramBufferizePass() {
  return std::make_unique<MLProgramBufferize>();
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
    math::populateExpansionPatterns(patterns, {"tanh"});
    patterns.add<math::ErfPolynomialApproximation>(patterns.getContext());
    ConversionTarget target(*context);
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<math::TanhOp>();
    target.addIllegalOp<math::ErfOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::RefBackend::createExpandOpsForLLVMPass() {
  return std::make_unique<ExpandOpsForLLVM>();
}

//===----------------------------------------------------------------------===//
// MungeMemrefCopy
//===----------------------------------------------------------------------===//

Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from,
                              Value to) {
  auto memrefTypeFrom = cast<MemRefType>(from.getType());
  auto memrefTypeTo = cast<MemRefType>(to.getType());
  (void)memrefTypeFrom;
  assert(memrefTypeFrom && memrefTypeTo &&
         memrefTypeFrom.getRank() == memrefTypeTo.getRank());
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return linalg::GenericOp::create(
      b, loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        linalg::YieldOp::create(b, loc, args.front());
      });
}

namespace {
class MemrefCopyOpToLinalg : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Operation *linalgCopy = createLinalgCopyOp(
        rewriter, copyOp.getLoc(), copyOp.getSource(), copyOp.getTarget());
    rewriter.replaceOp(copyOp, linalgCopy->getResults());
    return success();
  }
};

class MungeMemrefCopy : public MungeMemrefCopyBase<MungeMemrefCopy> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<MemrefCopyOpToLinalg>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::RefBackend::createMungeMemrefCopyPass() {
  return std::make_unique<MungeMemrefCopy>();
}

namespace {
class GeneralizeTensorConcat
    : public GeneralizeTensorConcatBase<GeneralizeTensorConcat> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::RefBackend::createGeneralizeTensorConcatPass() {
  return std::make_unique<GeneralizeTensorConcat>();
}

namespace {
class GeneralizeTensorPad
    : public GeneralizeTensorPadBase<GeneralizeTensorPad> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<linalg::DecomposePadOpPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::RefBackend::createGeneralizeTensorPadPass() {
  return std::make_unique<GeneralizeTensorPad>();
}
