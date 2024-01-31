//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

static constexpr StringRef getSeedGobalVarName() { return "global_seed"; }

// Declare a tensor<i64> global variable for the seed.
static LogicalResult getOrCreateGlobalVariableForSeed(OpBuilder &b,
                                                      ModuleOp module) {
  auto globalSeedSymbol =
      SymbolTable::lookupSymbolIn(module, getSeedGobalVarName());

  Type elemTy = b.getI64Type();
  auto tensorType = RankedTensorType::get({}, elemTy);

  if (globalSeedSymbol) {
    auto globalSeed = dyn_cast<ml_program::GlobalOp>(globalSeedSymbol);
    if (!globalSeed || globalSeed.getType() != tensorType)
      return module.emitError("Unexpected type for global seed.");
    return success();
  }

  b.setInsertionPointToStart(module.getBody());
  b.create<ml_program::GlobalOp>(
      UnknownLoc::get(b.getContext()),
      /*sym_name=*/getSeedGobalVarName(),
      /*type=*/tensorType,
      /*is_mutable=*/true,
      /*value=*/DenseIntElementsAttr::get(tensorType, {APInt(64, 0)}),
      /*sym_visibility=*/b.getStringAttr("private"));

  return success();
}

namespace {
class ConvertGetNextSeedOp : public OpConversionPattern<GetNextSeedOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetNextSeedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Generate sequence for getting the next seed with LCG step:
    //    nextSeed = (multiplier * currentSeed + incrementStep) mod 2^64.
    // Refer to https://en.wikipedia.org/wiki/Linear_congruential_generator.
    // Get the current seed value.
    auto tensorType = RankedTensorType::get({}, rewriter.getI64Type());
    Value globalVar = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType,
        SymbolRefAttr::get(op->getContext(), getSeedGobalVarName()));
    Value currentSeed = rewriter.create<tensor::ExtractOp>(loc, globalVar);

    // The value of multiplier and incrementStep are referenced from
    // https://en.wikipedia.org/wiki/Linear_congruential_generator for 2^64.
    Value multiplier = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(6364136223846793005));
    Value incrementStep = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(1442695040888963407));
    // temp = multiplier * currentSeed + incrementStep
    Value mul = rewriter.create<arith::MulIOp>(loc, currentSeed, multiplier);
    Value seed = rewriter.create<arith::AddIOp>(loc, mul, incrementStep);
    globalVar =
        rewriter.create<tensor::InsertOp>(loc, seed, globalVar, ValueRange());
    rewriter.create<ml_program::GlobalStoreOp>(
        loc, SymbolRefAttr::get(op->getContext(), getSeedGobalVarName()),
        globalVar);
    rewriter.replaceOp(op, seed);
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchConversionToMLProgram
    : public ConvertTorchConversionToMLProgramBase<
          ConvertTorchConversionToMLProgram> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<ml_program::MLProgramDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tensor::TensorDialect, arith::ArithDialect,
                           ml_program::MLProgramDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    auto module = getOperation();
    OpBuilder b(module.getBodyRegion());
    if (failed(getOrCreateGlobalVariableForSeed(b, module)))
      signalPassFailure();

    RewritePatternSet patterns(context);
    target.addIllegalOp<GetNextSeedOp>();
    patterns.add<ConvertGetNextSeedOp>(typeConverter, context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    getOperation()->walk(
        [this, &target, &frozenPatterns](func::FuncOp function) {
          if (failed(applyPartialConversion(function, target, frozenPatterns)))
            return signalPassFailure();
        });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::createConvertTorchConversionToMLProgramPass() {
  return std::make_unique<ConvertTorchConversionToMLProgram>();
}
