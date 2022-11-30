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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
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
      if (integerTy.isSignlessInteger(8))
        return true;
      if (integerTy.isSignedInteger(8))
        return true;
      if (integerTy.isSignlessInteger(1))
        return true;

    }
  }
  return false;
}

static void addEmitCInterfaceAttr(func::FuncOp func) {
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
static void replaceReturnWithCall(OpBuilder b, func::ReturnOp op,
                                  StringRef funcName, TypeRange retTypes,
                                  SmallVectorImpl<Value> &vals,
                                  SmallVectorImpl<Operation *> &toErase) {
  b.create<mlir::func::CallOp>(op.getLoc(), funcName, TypeRange({}), vals);
  b.create<mlir::func::ReturnOp>(op.getLoc());
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
          .append("argument must be a memref of f32, f64, i32, i64, i8, i1 but "
                  "got ",
                  type);
    }
    auto cast = b.create<memref::CastOp>(arg.getLoc(), type, arg);
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
      if (auto memrefReturnType = retType.dyn_cast<MemRefType>()) {
        auto elemType = memrefReturnType.getElementType();
        retType = UnrankedMemRefType::get(elemType, 0);
        // Cast to unranked memref type before sending it as a function
        // argument.
        retVal = b.create<memref::CastOp>(
            op.getLoc(), getAbiTypeForMemRef(types[en.index()]), retVal);
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
      auto consumeFuncReturnFunc = b.create<func::FuncOp>(
          module.getLoc(), p.first,
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
// InsertRngGlobals
//===----------------------------------------------------------------------===//

static constexpr StringRef getSeedGobalVarName() { return "global_seed"; }

// Declare a memref<i64> global variable for the seed.
static void createGlobalVariableForSeed(OpBuilder &b, ModuleOp module) {
  b.setInsertionPointToStart(module.getBody());
  Type elemTy = b.getI64Type();
  auto memref0D = MemRefType::get({}, elemTy);
  auto tensor0D = RankedTensorType::get({}, elemTy);
  b.create<memref::GlobalOp>(
      UnknownLoc::get(b.getContext()), getSeedGobalVarName(),
      /*sym_visibility=*/b.getStringAttr("private"),
      /*type=*/memref0D,
      /*initial_value=*/DenseIntElementsAttr::get(tensor0D, {APInt(64, 0)}),
      /*constant=*/false,
      /*alignment=*/nullptr);
}

// Generate sequence for getting the next seed with LCG step:
//    nextSeed = (multiplier * currentSeed + incrementStep) mod 64.
// Refer to https://en.wikipedia.org/wiki/Linear_congruential_generator.
static Value lowerGetNextSeed(OpBuilder &b, Location loc) {
  // Get the current seed value.
  auto memref1DType = MemRefType::get({}, b.getI64Type());
  Value globalVar =
      b.create<memref::GetGlobalOp>(loc, memref1DType, getSeedGobalVarName());
  Value currentSeed = b.create<memref::LoadOp>(loc, globalVar);

  // The value of multiplier and incrementStep are referenced from
  // https://en.wikipedia.org/wiki/Linear_congruential_generator for 2^64.
  Value multiplier = b.create<arith::ConstantOp>(
      loc, b.getI64IntegerAttr(6364136223846793005));
  Value incrementStep = b.create<arith::ConstantOp>(
      loc, b.getI64IntegerAttr(1442695040888963407));
  // temp = multiplier * currentSeed + incrementStep
  Value mul = b.create<arith::MulIOp>(loc, currentSeed, multiplier);
  Value nextSeed = b.create<arith::AddIOp>(loc, mul, incrementStep);
  b.create<memref::StoreOp>(loc, nextSeed, globalVar);
  return nextSeed;
}

// The global seed is stored into a memref<i64> global variable as the only
// element.
namespace {
class InsertRngGlobals : public InsertRngGlobalsBase<InsertRngGlobals> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getBodyRegion());
    createGlobalVariableForSeed(b, module);
    SmallVector<Operation *> toErase;
    module.walk([&](TorchConversion::GetNextSeedOp op) {
      b.setInsertionPoint(op);
      Value seed = lowerGetNextSeed(b, op.getLoc());
      op.replaceAllUsesWith(seed);
      toErase.push_back(op);
    });

    for (auto op : toErase)
      op->erase();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::RefBackend::createInsertRngGlobalsPass() {
  return std::make_unique<InsertRngGlobals>();
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
  auto memrefTypeFrom = from.getType().cast<MemRefType>();
  auto memrefTypeTo = to.getType().cast<MemRefType>();
  (void)memrefTypeFrom;
  assert(memrefTypeFrom && memrefTypeTo &&
         memrefTypeFrom.getRank() == memrefTypeTo.getRank());
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::makeArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
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
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
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
class GeneralizeTensorPad
    : public GeneralizeTensorPadBase<GeneralizeTensorPad> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<linalg::GeneralizePadOpPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::RefBackend::createGeneralizeTensorPadPass() {
  return std::make_unique<GeneralizeTensorPad>();
}
