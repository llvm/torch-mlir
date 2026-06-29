//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;
namespace mlir::torch::TorchConversion {

#define GEN_PASS_DEF_FUNCBACKENDTYPECONVERSION
#define GEN_PASS_DEF_FUNCBACKENDTYPECONVERSIONFORSTABLEHLO
#define GEN_PASS_DEF_FINALIZINGBACKENDTYPECONVERSION
#define GEN_PASS_DEF_FINALIZINGBACKENDTYPECONVERSIONFORSTABLEHLO
#define GEN_PASS_DEF_LIFTUSERATTRS
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// FuncBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {

// TODO: Consider upstreaming this to an `arith::ExtFOp` folder:
struct ExtFTruncFPattern : public OpRewritePattern<arith::TruncFOp> {
  ExtFTruncFPattern(MLIRContext *context) : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(arith::TruncFOp truncf,
                                PatternRewriter &rewriter) const override {
    Value operand = truncf.getOperand();
    auto extf = operand.getDefiningOp<arith::ExtFOp>();
    if (!extf)
      return failure();

    auto parentOperand = extf.getOperand();
    if (truncf.getType() != parentOperand.getType())
      return failure();

    rewriter.replaceOp(truncf, parentOperand);
    return success();
  }
};

void populateFuncBackendTypeConversionPatterns(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  cf::populateCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                     target);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addLegalOp<ModuleOp>();

  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                            typeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });
}

struct FuncBackendTypeConversionPass
    : public impl::FuncBackendTypeConversionBase<
          FuncBackendTypeConversionPass> {
  using FuncBackendTypeConversionBase<
      FuncBackendTypeConversionPass>::FuncBackendTypeConversionBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TorchConversion::TorchConversionDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    populateFuncBackendTypeConversionPatterns(typeConverter, patterns, target);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
struct FuncBackendTypeConversionForStablehloPass
    : public impl::FuncBackendTypeConversionForStablehloBase<
          FuncBackendTypeConversionForStablehloPass> {
  using FuncBackendTypeConversionForStablehloBase<
      FuncBackendTypeConversionForStablehloPass>::
      FuncBackendTypeConversionForStablehloBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TorchConversion::TorchConversionDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversionForStablehlo(target,
                                                            typeConverter);

    populateFuncBackendTypeConversionPatterns(typeConverter, patterns, target);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
#endif // TORCH_MLIR_ENABLE_STABLEHLO
} // namespace

// Create functions for passes
std::unique_ptr<OperationPass<ModuleOp>> createFuncBackendTypeConversionPass() {
  return std::make_unique<FuncBackendTypeConversionPass>();
}

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
std::unique_ptr<OperationPass<ModuleOp>>
createFuncBackendTypeConversionForStablehloPass() {
  return std::make_unique<FuncBackendTypeConversionForStablehloPass>();
}
#endif // TORCH_MLIR_ENABLE_STABLEHLO

//===----------------------------------------------------------------------===//
// FinalizingBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {
// In a finalizing conversion, we know that all of the source types have been
// converted to the destination types, so the materialization becomes an
// identity.
template <typename OpTy>
class FinalizeMaterialization : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};
} // namespace

template <typename OpTy>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  target.addIllegalOp<OpTy>();
  patterns.add<FinalizeMaterialization<OpTy>>(typeConverter,
                                              patterns.getContext());
}

template <typename OpTy, typename OpTy2, typename... OpTys>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  setupFinalization<OpTy>(target, patterns, typeConverter);
  setupFinalization<OpTy2, OpTys...>(target, patterns, typeConverter);
}

static void stripTorchAttrs(FunctionOpInterface func) {
  bool modified = false;
  SmallVector<NamedAttribute> newAttrs;
  for (auto attr : func->getDialectAttrs()) {
    if (attr.getName().getValue().starts_with("torch."))
      modified = true;
    else
      newAttrs.push_back(attr);
  }
  if (modified)
    func->setDialectAttrs(newAttrs);

  // Note: this could also strip "arg" and "result" attrs if they were used.
}

namespace {
struct FinalizingBackendTypeConversionPass
    : public impl::FinalizingBackendTypeConversionBase<
          FinalizingBackendTypeConversionPass> {
  using FinalizingBackendTypeConversionBase<
      FinalizingBackendTypeConversionPass>::FinalizingBackendTypeConversionBase;

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    // Mark materializations as illegal in this pass (since we are finalizing)
    // and add patterns that eliminate them.
    setupFinalization<ToBuiltinTensorOp, FromBuiltinTensorOp, FromI1Op, ToI1Op,
                      FromI64Op, ToI64Op, FromF64Op, ToF64Op, I64ToGeneratorOp,
                      GeneratorToI64Op>(target, patterns, typeConverter);

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet greedyPatterns(context);
    greedyPatterns.insert<ExtFTruncFPattern>(context);
    if (failed(applyPatternsGreedily(func, std::move(greedyPatterns))))
      signalPassFailure();

    // Drop attributes that are no longer used after conversion out of Torch.
    stripTorchAttrs(func);
  }
};

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
struct FinalizingBackendTypeConversionForStablehloPass
    : public impl::FinalizingBackendTypeConversionForStablehloBase<
          FinalizingBackendTypeConversionForStablehloPass> {
  using FinalizingBackendTypeConversionForStablehloBase<
      FinalizingBackendTypeConversionForStablehloPass>::
      FinalizingBackendTypeConversionForStablehloBase;

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversionForStablehlo(target,
                                                            typeConverter);

    // Mark materializations as illegal in this pass (since we are finalizing)
    // and add patterns that eliminate them.
    setupFinalization<ToBuiltinTensorOp, FromBuiltinTensorOp, FromI1Op, ToI1Op,
                      FromI64Op, ToI64Op, FromF64Op, ToF64Op, I64ToGeneratorOp,
                      GeneratorToI64Op>(target, patterns, typeConverter);

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();

    // Drop attributes that are no longer used after conversion out of Torch.
    stripTorchAttrs(func);
  }
};
#endif // TORCH_MLIR_ENABLE_STABLEHLO
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createFinalizingBackendTypeConversionPass() {
  return std::make_unique<FinalizingBackendTypeConversionPass>();
}

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createFinalizingBackendTypeConversionForStablehloPass() {
  return std::make_unique<FinalizingBackendTypeConversionForStablehloPass>();
}
#endif // TORCH_MLIR_ENABLE_STABLEHLO

//===----------------------------------------------------------------------===//
// LiftUserAttrsPass
//===----------------------------------------------------------------------===//

namespace {
constexpr StringLiteral kUserAttrPrefix("mlir.user.");

// Strip the `mlir.user.` prefix from every prefixed discardable attr on `op`,
// re-attaching the value under its un-prefixed name.
static void liftPrefixedAttrs(Operation *op) {
  SmallVector<NamedAttribute> toLift;
  for (NamedAttribute named : op->getDiscardableAttrs()) {
    if (named.getName().getValue().starts_with(kUserAttrPrefix))
      toLift.push_back(named);
  }
  if (toLift.empty())
    return;
  for (NamedAttribute named : toLift) {
    StringRef stripped =
        named.getName().getValue().drop_front(kUserAttrPrefix.size());
    op->setAttr(stripped, named.getValue());
    op->removeAttr(named.getName());
  }
}

// Same for every per-arg dict in a function's `arg_attrs` array.
static void liftFuncArgAttrs(func::FuncOp func) {
  ArrayAttr argAttrs = func.getAllArgAttrs();
  if (!argAttrs)
    return;
  bool changed = false;
  SmallVector<Attribute> newArgAttrs(argAttrs.begin(), argAttrs.end());
  for (auto &attr : newArgAttrs) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      continue;
    NamedAttrList rewritten;
    bool localChanged = false;
    for (NamedAttribute named : dict.getValue()) {
      if (named.getName().getValue().starts_with(kUserAttrPrefix)) {
        StringRef stripped =
            named.getName().getValue().drop_front(kUserAttrPrefix.size());
        rewritten.set(stripped, named.getValue());
        localChanged = true;
      } else {
        rewritten.set(named.getName(), named.getValue());
      }
    }
    if (localChanged) {
      attr = rewritten.getDictionary(func.getContext());
      changed = true;
    }
  }
  if (changed)
    func.setAllArgAttrs(newArgAttrs);
}

struct LiftUserAttrsPass
    : public impl::LiftUserAttrsBase<LiftUserAttrsPass> {
  using LiftUserAttrsBase::LiftUserAttrsBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([](Operation *op) { liftPrefixedAttrs(op); });
    module.walk([](func::FuncOp func) { liftFuncArgAttrs(func); });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLiftUserAttrsPass() {
  return std::make_unique<LiftUserAttrsPass>();
}

} // namespace mlir::torch::TorchConversion
