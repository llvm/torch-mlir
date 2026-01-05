//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;
namespace mlir::torch::TorchConversion {

#define GEN_PASS_DEF_FUNCBACKENDTYPECONVERSION
#define GEN_PASS_DEF_FUNCBACKENDTYPECONVERSIONFORSTABLEHLO
#define GEN_PASS_DEF_FINALIZINGBACKENDTYPECONVERSION
#define GEN_PASS_DEF_FINALIZINGBACKENDTYPECONVERSIONFORSTABLEHLO
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// FuncBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {

static FailureOr<Block *>
convertBlockSignatureIfNeeded(Block *block, const TypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter) {
  std::optional<TypeConverter::SignatureConversion> conversion =
      typeConverter->convertBlockSignature(block);
  if (!conversion)
    return failure();
  Block *newBlock =
      rewriter.applySignatureConversion(block, *conversion, typeConverter);
  return newBlock;
}

static FailureOr<SmallVector<Value>>
convertSuccessorOperands(Location loc, ValueRange operands, Block *dest,
                         const TypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter) {
  if (operands.size() != dest->getNumArguments())
    return failure();
  SmallVector<Value> converted;
  converted.reserve(operands.size());
  for (auto it : llvm::zip_equal(operands, dest->getArguments())) {
    Value operand = std::get<0>(it);
    Value arg = std::get<1>(it);
    if (operand.getType() == arg.getType()) {
      converted.push_back(operand);
      continue;
    }
    SmallVector<Value, 1> inputs{operand};
    Value newOperand = typeConverter->materializeTargetConversion(
        rewriter, loc, arg.getType(), inputs, operand.getType());
    if (!newOperand)
      return failure();
    converted.push_back(newOperand);
  }
  return converted;
}

static bool hasTorchTensor(Value value) {
  return isa<Torch::BaseTensorType>(value.getType());
}

static bool hasTorchTensor(ValueRange values) {
  return llvm::any_of(values, [](Value v) { return hasTorchTensor(v); });
}

static bool blockHasTorchTensor(Block *block) {
  return llvm::any_of(block->getArgumentTypes(), [](Type type) {
    return isa<Torch::BaseTensorType>(type);
  });
}

class ConvertCFBrOp : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  ConvertCFBrOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}
  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Block *> dest = convertBlockSignatureIfNeeded(
        op.getDest(), this->getTypeConverter(), rewriter);
    if (failed(dest))
      return failure();
    FailureOr<SmallVector<Value>> newOperands =
        convertSuccessorOperands(op.getLoc(), adaptor.getDestOperands(), *dest,
                                 this->getTypeConverter(), rewriter);
    if (failed(newOperands))
      return failure();
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, *dest, *newOperands);
    return success();
  }
};

class ConvertCFCondBranchOp : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  ConvertCFCondBranchOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}
  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Block *> trueDest = convertBlockSignatureIfNeeded(
        op.getTrueDest(), this->getTypeConverter(), rewriter);
    if (failed(trueDest))
      return failure();
    FailureOr<Block *> falseDest = convertBlockSignatureIfNeeded(
        op.getFalseDest(), this->getTypeConverter(), rewriter);
    if (failed(falseDest))
      return failure();
    FailureOr<SmallVector<Value>> trueOperands =
        convertSuccessorOperands(op.getLoc(), adaptor.getTrueDestOperands(),
                                 *trueDest, this->getTypeConverter(), rewriter);
    if (failed(trueOperands))
      return failure();
    FailureOr<SmallVector<Value>> falseOperands = convertSuccessorOperands(
        op.getLoc(), adaptor.getFalseDestOperands(), *falseDest,
        this->getTypeConverter(), rewriter);
    if (failed(falseOperands))
      return failure();
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(op, adaptor.getCondition(),
                                                  *trueDest, *trueOperands,
                                                  *falseDest, *falseOperands);
    return success();
  }
};

class ConvertCFSwitchOp : public OpConversionPattern<cf::SwitchOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  ConvertCFSwitchOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}
  LogicalResult
  matchAndRewrite(cf::SwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Block *> defaultDest = convertBlockSignatureIfNeeded(
        op.getDefaultDestination(), this->getTypeConverter(), rewriter);
    if (failed(defaultDest))
      return failure();
    FailureOr<SmallVector<Value>> defaultOperands = convertSuccessorOperands(
        op.getLoc(), adaptor.getDefaultOperands(), *defaultDest,
        this->getTypeConverter(), rewriter);
    if (failed(defaultOperands))
      return failure();

    SmallVector<Block *> caseDests;
    SmallVector<SmallVector<Value>> caseOperandsStorage;
    SmallVector<ValueRange> caseOperandRanges;
    auto adaptorCaseOperands = adaptor.getCaseOperands();
    for (auto it : llvm::enumerate(op.getCaseDestinations())) {
      FailureOr<Block *> newDest = convertBlockSignatureIfNeeded(
          it.value(), this->getTypeConverter(), rewriter);
      if (failed(newDest))
        return failure();
      FailureOr<SmallVector<Value>> convertedOperands =
          convertSuccessorOperands(op.getLoc(), adaptorCaseOperands[it.index()],
                                   *newDest, this->getTypeConverter(),
                                   rewriter);
      if (failed(convertedOperands))
        return failure();
      caseDests.push_back(*newDest);
      caseOperandsStorage.push_back(std::move(*convertedOperands));
    }
    caseOperandRanges.reserve(caseOperandsStorage.size());
    for (auto &storage : caseOperandsStorage)
      caseOperandRanges.push_back(storage);

    rewriter.replaceOpWithNewOp<cf::SwitchOp>(
        op, adaptor.getFlag(), *defaultDest, *defaultOperands,
        adaptor.getCaseValuesAttr(), caseDests, caseOperandRanges);
    return success();
  }
};

static void
populateCFStructuralTypeConversionPatterns(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  patterns.add<ConvertCFBrOp, ConvertCFCondBranchOp, ConvertCFSwitchOp>(
      typeConverter, patterns.getContext());
}

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

  target.addDynamicallyLegalOp<cf::BranchOp>([&](cf::BranchOp op) {
    return !hasTorchTensor(op.getDestOperands()) &&
           !blockHasTorchTensor(op.getDest());
  });
  target.addDynamicallyLegalOp<cf::CondBranchOp>([&](cf::CondBranchOp op) {
    return !hasTorchTensor(op.getTrueDestOperands()) &&
           !hasTorchTensor(op.getFalseDestOperands()) &&
           !blockHasTorchTensor(op.getTrueDest()) &&
           !blockHasTorchTensor(op.getFalseDest());
  });
  target.addDynamicallyLegalOp<cf::SwitchOp>([&](cf::SwitchOp op) {
    if (hasTorchTensor(op.getDefaultOperands()) ||
        blockHasTorchTensor(op.getDefaultDestination()))
      return false;
    auto caseOperands = op.getCaseOperands();
    for (auto [index, dest] : llvm::enumerate(op.getCaseDestinations()))
      if (hasTorchTensor(caseOperands[index]) || blockHasTorchTensor(dest))
        return false;
    return true;
  });

  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateCFStructuralTypeConversionPatterns(typeConverter, patterns);
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

} // namespace mlir::torch::TorchConversion
