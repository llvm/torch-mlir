//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/BackendTypeConversion.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

static void
setupValueTensorToBuiltinTensorConversion(ConversionTarget &target,
                                          TypeConverter &typeConverter) {
  target.addLegalOp<Torch::ToBuiltinTensorOp, Torch::FromBuiltinTensorOp>();
  typeConverter.addConversion(
      [](Torch::ValueTensorType type) -> Optional<Type> {
        return type.toBuiltinTensor();
      });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, TensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<BaseTensorType>());
    return builder.create<ToBuiltinTensorOp>(loc, inputs[0]);
  });
  auto sourceMaterialization = [](OpBuilder &builder, ValueTensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<FromBuiltinTensorOp>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

static void setupTorchBoolToI1Conversion(ConversionTarget &target,
                                         TypeConverter &typeConverter) {
  target.addLegalOp<Torch::ToI1Op, Torch::FromI1Op>();
  typeConverter.addConversion([](Torch::BoolType type) -> Optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            IntegerType type, ValueRange inputs,
                                            Location loc) -> Optional<Value> {
    // Other builtin integer types could be handled by other materializers.
    if (!(type.getWidth() == 1 && type.isSignless()))
      return None;
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BoolType>());
    return builder.create<ToI1Op>(loc, inputs[0]).getResult();
  });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::BoolType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IntegerType>());
    return builder.create<FromI1Op>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

void mlir::NPCOMP::Torch::setupBackendTypeConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  setupValueTensorToBuiltinTensorConversion(target, typeConverter);
  setupTorchBoolToI1Conversion(target, typeConverter);
}

//===----------------------------------------------------------------------===//
// FuncBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {
struct FuncBackendTypeConversionPass
    : public FuncBackendTypeConversionBase<FuncBackendTypeConversionPass> {
  using FuncBackendTypeConversionBase<
      FuncBackendTypeConversionPass>::FuncBackendTypeConversionBase;
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    typeConverter.addConversion([](Type type) { return type; });
    setupBackendTypeConversion(target, typeConverter);

    populateFuncOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<CallOp>(
        [&](CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::Torch::createFuncBackendTypeConversionPass() {
  return std::make_unique<FuncBackendTypeConversionPass>();
}

//===----------------------------------------------------------------------===//
// FinalizingBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {
// In a finalizing conversion, we know that all `!torch.vtensor` have been
// converted to `tensor`, thus, this op becomes an identity.
class FinalizeToBuiltinTensorOp
    : public OpConversionPattern<ToBuiltinTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToBuiltinTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

namespace {
// In a finalizing conversion, we know that all `!torch.vtensor` have been
// converted to `tensor`, thus, this op becomes an identity.
class FinalizeFromBuiltinTensorOp
    : public OpConversionPattern<FromBuiltinTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FromBuiltinTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

namespace {
// In a finalizing conversion, we know that all `!torch.bool` have been
// converted to `i1`, thus, this op becomes an identity.
class FinalizeToI1Op : public OpConversionPattern<ToI1Op> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToI1Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

namespace {
// In a finalizing conversion, we know that all `!torch.bool` have been
// converted to `i1`, thus, this op becomes an identity.
class FinalizeFromI1Op : public OpConversionPattern<FromI1Op> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FromI1Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

namespace {
struct FinalizingBackendTypeConversionPass
    : public FinalizingBackendTypeConversionBase<
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
    setupBackendTypeConversion(target, typeConverter);

    // Mark materializations as illegal in this pass (since we are finalizing)
    // and add patterns that eliminate them.
    target.addIllegalOp<ToBuiltinTensorOp, FromBuiltinTensorOp, FromI1Op,
                        ToI1Op>();
    patterns.add<FinalizeFromBuiltinTensorOp, FinalizeToBuiltinTensorOp,
                 FinalizeFromI1Op, FinalizeToI1Op>(typeConverter, context);

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
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Torch::createFinalizingBackendTypeConversionPass() {
  return std::make_unique<FinalizingBackendTypeConversionPass>();
}
