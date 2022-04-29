//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertTorchPrimIfYieldOp : public OpConversionPattern<PrimIfYieldOp> {
public:
  using OpConversionPattern<PrimIfYieldOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimIfYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchPrimIfOp : public OpConversionPattern<PrimIfOp> {
public:
  using OpConversionPattern<PrimIfOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimIfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                newResultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "could not convert PrimIfOp outputs");
    auto scfIf = rewriter.create<scf::IfOp>(op->getLoc(), newResultTypes,
                                            adaptor.condition(),
                                            /*withElseRegion=*/true);
    auto inlineIfCase = [&](Region &srcRegion, Region &dstRegion) {
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
      rewriter.eraseBlock(&dstRegion.back());
    };
    inlineIfCase(op.thenRegion(), scfIf.getThenRegion());
    inlineIfCase(op.elseRegion(), scfIf.getElseRegion());
    rewriter.replaceOp(op, scfIf.getResults());
    return success();
  }
};
} // namespace

namespace {

// Converts the Torch::PrimLoopOp which is ``While-like`` into scf::WhileOp.
class ConvertTorchPrimLoopWhileLikeOp : public OpConversionPattern<PrimLoopOp> {
public:
  using OpConversionPattern<PrimLoopOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimLoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Return failure on for-like loops.
    if (op.isForLike())
      return failure();

    TypeConverter *typeConverter = getTypeConverter();
    SmallVector<Type, 1> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return rewriter.notifyMatchFailure(
          op, "could not convert PrimLoopOp outputs");

    // Create scf.while operation using the operands of torch::primloop. The
    // first argument of the primloop correspond to `maxTripCount`  which
    // can be omitted in the `scf.while` operation.
    Value condition = adaptor.initialCondition();
    ValueRange iterArgsInit = adaptor.iterArgsInit();
    SmallVector<Value> scfWhileOpOperands{condition};
    scfWhileOpOperands.append(iterArgsInit.begin(), iterArgsInit.end());
    auto scfWhileOp = rewriter.create<scf::WhileOp>(
        op->getLoc(), newResultTypes, scfWhileOpOperands);

    // Populate the before region of the scf.while operation. The `before`
    // region will have only one block and the arguments of the block must match
    // the arguments of `scf.while` operation.
    SmallVector<Type> beforeRegionArgTypes;
    SmallVector<Location> beforeRegionArgLocs;
    for (Value value : scfWhileOp->getOperands()) {
      beforeRegionArgTypes.push_back(value.getType());
      beforeRegionArgLocs.push_back(value.getLoc());
    }
    auto *beforeBlock = rewriter.createBlock(
        &scfWhileOp.getBefore(), scfWhileOp.getBefore().begin(),
        beforeRegionArgTypes, beforeRegionArgLocs);

    rewriter.setInsertionPointToEnd(beforeBlock);
    // Fetch the condition passed as the iter argument. Pass rest of the
    // arguments to the after block.
    auto scfConditionOp = rewriter.create<scf::ConditionOp>(
        op.getLoc(), beforeBlock->getArgument(0),
        beforeBlock->getArguments().drop_front());

    // Populate the after region.
    if (!scfWhileOp.getAfter().empty())
      rewriter.eraseBlock(&scfWhileOp.getAfter().back());

    SmallVector<Type> afterRegionArgTypes;
    SmallVector<Location> afterRegionArgLocs;
    for (Value value : scfConditionOp.getArgs()) {
      afterRegionArgTypes.push_back(value.getType());
      afterRegionArgLocs.push_back(value.getLoc());
    }
    auto *afterBlock = rewriter.createBlock(
        &scfWhileOp.getAfter(), scfWhileOp.getAfter().begin(),
        afterRegionArgTypes, afterRegionArgLocs);

    // Rewrite uses of the torch loop block arguments to the new while-loop
    // "after" arguments. Leave the induction variable of prim loop(first
    // argument) because while like prim loops does not use the induction
    // variable.
    for (const auto &barg :
         enumerate(op.region().front().getArguments().drop_front())) {
      Value to = afterBlock->getArgument(barg.index());
      Type targetType = to.getType();
      Value torchArg = to;

      // If the target type is non-torch type, then use TypeConverter to convert
      // the type of the source.
      if (targetType.isa<mlir::FloatType>()) {
        targetType = Torch::FloatType::get(op->getContext());
        torchArg = typeConverter->materializeSourceConversion(
            rewriter, scfWhileOp.getLoc(), targetType, {to});
      } else if (targetType.isa<mlir::IntegerType>()) {
        unsigned bitWidth = targetType.getIntOrFloatBitWidth();
        if (bitWidth == 1)
          targetType = Torch::BoolType::get(op->getContext());
        else
          targetType = Torch::IntType::get(op->getContext());
        torchArg = typeConverter->materializeSourceConversion(
            rewriter, scfWhileOp.getLoc(), targetType, {to});
      }
      if (!torchArg)
        return rewriter.notifyMatchFailure(op,
                                           "unsupported type of the operand");
      barg.value().replaceAllUsesWith(torchArg);
    }
    // Inline torch loop body operations into 'after' region.
    PatternRewriter::InsertionGuard guard(rewriter);
    for (auto &operation :
         llvm::make_early_inc_range(op.region().front().getOperations())) {
      if (auto primLoopConditionOp = dyn_cast<PrimLoopConditionOp>(operation)) {
        // Fix up the terminator.
        SmallVector<Value> loopConditionIterArgs;
        Value torchShouldContinue = primLoopConditionOp.shouldContinue();
        Value shouldContinue = typeConverter->materializeTargetConversion(
            rewriter, scfWhileOp->getLoc(),
            typeConverter->convertType(torchShouldContinue.getType()),
            {torchShouldContinue});
        if (!shouldContinue)
          return rewriter.notifyMatchFailure(op,
                                             "unsupported type of the operand");
        loopConditionIterArgs.push_back(shouldContinue);
        for (auto torchArg : primLoopConditionOp.iterArgs()) {
          Type torchType = torchArg.getType();

          // If the argument is a torch tensor, directly add it in the list of
          // iter args.
          if (torchType.isa<Torch::BaseTensorType>()) {
            loopConditionIterArgs.push_back(torchArg);
            continue;
          }
          Value arg = typeConverter->materializeTargetConversion(
              rewriter, scfWhileOp->getLoc(),
              typeConverter->convertType(torchArg.getType()), {torchArg});
          if (!arg)
            return rewriter.notifyMatchFailure(
                op, "unsupported type of the operand");
          loopConditionIterArgs.push_back(arg);
        }
        rewriter.create<scf::YieldOp>(scfWhileOp.getLoc(),
                                      loopConditionIterArgs);

      } else {
        operation.moveBefore(afterBlock, afterBlock->end());
      }
    }
    rewriter.replaceOp(op, scfWhileOp->getResults());
    return success();
  }
};
} // namespace

namespace {
// Converts the Torch::PrimLoopOp which is ``For-like`` into scf::ForOp.
class ConvertTorchPrimLoopForLikeOp : public OpConversionPattern<PrimLoopOp> {
public:
  using OpConversionPattern<PrimLoopOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimLoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Return failure on while-like loops.
    if (!op.isForLike())
      return failure();

    TypeConverter *typeConverter = getTypeConverter();
    SmallVector<Type, 1> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return rewriter.notifyMatchFailure(
          op, "could not convert PrimLoopOp outputs");

    // Calculate the lower bound, upper bound and step indices. Currently only
    // lower-bound = 0 and step = 1 is supported.
    Location loc = op.getLoc();
    Value lowerBoundIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value stepIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value upperBoundIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), adaptor.maxTripCount());
    auto scfForOp =
        rewriter.create<scf::ForOp>(loc, lowerBoundIndex, upperBoundIndex,
                                    stepIndex, adaptor.iterArgsInit());

    SmallVector<Type> regionArgTypes;
    SmallVector<Location> regionArgLocs;
    for (Value value : scfForOp.getLoopBody().front().getArguments()) {
      regionArgTypes.push_back(value.getType());
      regionArgLocs.push_back(value.getLoc());
    }

    // Populate the loop body region.
    if (!scfForOp.getLoopBody().empty())
      rewriter.eraseBlock(&scfForOp.getLoopBody().back());

    auto *block = rewriter.createBlock(&scfForOp.getLoopBody(),
                                       scfForOp.getLoopBody().begin(),
                                       regionArgTypes, regionArgLocs);

    // Rewrite uses of the torch loop block arguments to the new for-loop
    // "block" arguments
    for (const auto &barg : enumerate(op.region().front().getArguments())) {
      Value to = block->getArgument(barg.index());
      if (to.getType().isa<mlir::IndexType>())
        to =
            rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), to);
      Type targetType = to.getType();
      Value torchArg = to;

      // If the target type is non-torch type, then use TypeConverter to convert
      // the type of the source.
      if (targetType.isa<mlir::FloatType>()) {
        targetType = Torch::FloatType::get(op->getContext());
        torchArg = typeConverter->materializeSourceConversion(
            rewriter, scfForOp.getLoc(), targetType, {to});
      } else if (targetType.isa<mlir::IntegerType>()) {
        unsigned bitWidth = targetType.getIntOrFloatBitWidth();
        if (bitWidth == 1)
          targetType = Torch::BoolType::get(op->getContext());
        else
          targetType = Torch::IntType::get(op->getContext());
        torchArg = typeConverter->materializeSourceConversion(
            rewriter, scfForOp.getLoc(), targetType, {to});
      }
      if (!torchArg)
        return rewriter.notifyMatchFailure(op,
                                           "unsupported type of the operand");
      barg.value().replaceAllUsesWith(torchArg);
    }

    // Inline torch loop body operations into 'after' region.
    PatternRewriter::InsertionGuard guard(rewriter);
    for (auto &operation :
         llvm::make_early_inc_range(op.region().front().getOperations())) {
      if (auto primLoopConditionOp = dyn_cast<PrimLoopConditionOp>(operation)) {
        // Fix up the terminator.
        SmallVector<Value> loopConditionIterArgs;
        for (auto torchArg : primLoopConditionOp.iterArgs()) {
          Type torchType = torchArg.getType();

          // If the argument is a torch tensor, directly add it in the list of
          // iter args.
          if (torchType.isa<Torch::BaseTensorType>()) {
            loopConditionIterArgs.push_back(torchArg);
            continue;
          }
          Value arg = typeConverter->materializeTargetConversion(
              rewriter, scfForOp.getLoc(),
              typeConverter->convertType(torchArg.getType()), {torchArg});
          if (!arg)
            return rewriter.notifyMatchFailure(
                op, "unsupported type of the operand");
          loopConditionIterArgs.push_back(arg);
        }
        rewriter.create<scf::YieldOp>(scfForOp.getLoc(), loopConditionIterArgs);
      } else {
        operation.moveBefore(block, block->end());
      }
    }

    rewriter.replaceOp(op, scfForOp->getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchToSCF : public ConvertTorchToSCFBase<ConvertTorchToSCF> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, arith::ArithmeticDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, scf::SCFDialect,
                           arith::ArithmeticDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<PrimIfOp>();
    patterns.add<ConvertTorchPrimIfOp>(typeConverter, context);
    target.addIllegalOp<PrimIfYieldOp>();
    patterns.add<ConvertTorchPrimIfYieldOp>(typeConverter, context);
    target.addIllegalOp<PrimLoopOp>();
    patterns.add<ConvertTorchPrimLoopWhileLikeOp>(typeConverter, context);
    patterns.add<ConvertTorchPrimLoopForLikeOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToSCFPass() {
  return std::make_unique<ConvertTorchToSCF>();
}
