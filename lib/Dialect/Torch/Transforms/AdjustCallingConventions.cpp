//===- AdjustCallingConventions.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Map from func name and arg index to the type bound for that arg.
// This is needed because to rewrite calls, we need the non-local information
// from the func definition.
// We also benefit from populating this all at once, which avoids ordering
// issues between rewriting of func ops vs call ops.
using TypeBoundMap = DenseMap<std::pair<StringRef, int>, Type>;

namespace {
class AdjustCallingConventionForFunc
    : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp func, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = func.getContext();
    auto typeBoundIdent = StringAttr::get(context, "torch.type_bound");
    TypeConverter::SignatureConversion conversion(func.getNumArguments());

    // The TypeConverter hooks for type conversion are "context free", so we
    // cannot use the usual helpers here for populating SignatureConversion and
    // new result types.
    //
    // The incoporation of the torch.type_bound arg attr is context-dependent.

    for (auto type : llvm::enumerate(func.getArgumentTypes())) {
      if (type.value().isa<NonValueTensorType>()) {
        auto typeBoundAttr =
            func.getArgAttrOfType<TypeAttr>(type.index(), typeBoundIdent);
        Type bound = typeBoundAttr ? typeBoundAttr.getValue() : Type();
        if (!bound.isa<ValueTensorType>())
          return rewriter.notifyMatchFailure(
              func, "unimplemented: preserving aliasing for non-value-semantic "
                    "type bounds");
        conversion.addInputs(type.index(), typeBoundAttr
                                               ? typeBoundAttr.getValue()
                                               : type.value());
        continue;
      } else if (auto none = type.value().dyn_cast<Torch::NoneType>()) {
        continue;
      }
      // TODO: add tuple type.
      conversion.addInputs(type.index(), type.value());
    }
    rewriter.applySignatureConversion(&func.getBody(), conversion,
                                      typeConverter);

    SmallVector<Type> newResultTypes;
    for (auto type : func.getFunctionType().getResults()) {
      if (auto none = type.dyn_cast<Torch::NoneType>()) {
        continue;
      }
      if (auto tuple = type.dyn_cast<Torch::TupleType>()) {
        llvm::append_range(newResultTypes, tuple.getContainedTypes());
        continue;
      }
      newResultTypes.push_back(type);
    }
    rewriter.modifyOpInPlace(func, [&] {
      func.setType(FunctionType::get(
          getContext(), conversion.getConvertedTypes(), newResultTypes));
      // Clear out the type bounds, now that the type incorporates them.
      for (int i = 0, e = func.getNumArguments(); i != e; i++)
        func.removeArgAttr(i, typeBoundIdent);
    });
    return success();
  }
};
} // namespace

namespace {
class AdjustCallingConventionForCall
    : public OpConversionPattern<func::CallOp> {
public:
  AdjustCallingConventionForCall(TypeConverter &converter, MLIRContext *context,
                                 TypeBoundMap &typeBoundMap)
      : OpConversionPattern<func::CallOp>(converter, context),
        typeBoundMap(typeBoundMap) {}
  LogicalResult
  matchAndRewrite(func::CallOp call, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convertedResults;
    if (failed(typeConverter->convertTypes(call.getResultTypes(),
                                           convertedResults)))
      return failure();

    SmallVector<Value> newOperands;
    for (auto operand : llvm::enumerate(adaptor.getOperands())) {
      if (operand.value().getType().isa<Torch::NoneType>())
        continue;
      auto it = typeBoundMap.find({call.getCallee(), operand.index()});
      if (it != typeBoundMap.end()) {
        if (auto valueTensorType = it->second.dyn_cast<ValueTensorType>()) {
          newOperands.push_back(copyTensorToType(
              rewriter, call->getLoc(), valueTensorType, operand.value()));
          continue;
        } else {
          return rewriter.notifyMatchFailure(
              call, "unimplemented: preserving aliasing for non-value-semantic "
                    "type bounds");
        }
      }
      newOperands.push_back(operand.value());
    }

    func::CallOp newCall = rewriter.create<func::CallOp>(
        call.getLoc(), call.getCallee(), convertedResults, newOperands);
    int newOpResultIdx = 0;
    SmallVector<Value> newResults;
    for (auto type : call.getResultTypes()) {
      if (type.isa<Torch::NoneType>()) {
        newResults.push_back(
            rewriter.create<ConstantNoneOp>(call.getLoc(), type));
        continue;
      }
      if (type.isa<Torch::TupleType>()) {
        newResults.push_back(rewriter.create<PrimTupleConstructOp>(
            call.getLoc(), type, newCall.getResults()));
        continue;
      }
      newResults.push_back(newCall.getResult(newOpResultIdx++));
    }
    rewriter.replaceOp(call, newResults);
    return success();
  }

private:
  TypeBoundMap &typeBoundMap;
};
} // namespace

namespace {
class AdjustCallingConventionForReturn
    : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value> newOperands;
    for (auto operand : adaptor.getOperands()) {
      if (!operand)
        continue;
      if (operand.getType().isa<Torch::NoneType>())
        continue;
      if (auto tuple = operand.getType().dyn_cast<Torch::TupleType>()) {
        Location loc = op.getLoc();
        for (auto en : llvm::enumerate(tuple.getContainedTypes())) {
          auto i = rewriter.create<ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(en.index()));
          newOperands.push_back(
              rewriter.create<PrimTupleIndexOp>(loc, en.value(), operand, i));
        }
        continue;
      }
      newOperands.push_back(operand);
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, newOperands);
    return success();
  }
};
} // namespace

static LogicalResult adjustCallingConventions(func::FuncOp func,
                                              TypeBoundMap &typeBoundMap) {
  MLIRContext *context = func.getContext();
  RewritePatternSet patterns(context);
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion(
      [](Torch::TupleType type, SmallVectorImpl<Type> &types) -> LogicalResult {
        llvm::append_range(types, type.getContainedTypes());
        return success();
      });
  typeConverter.addConversion(
      [](Torch::NoneType type, SmallVectorImpl<Type> &types) -> LogicalResult {
        return success();
      });

  typeConverter.addArgumentMaterialization(
      [](OpBuilder &builder, Torch::BaseTensorType type, ValueRange inputs,
         Location loc) -> Value {
        assert(inputs.size() == 1);
        assert(inputs[0].getType().isa<BaseTensorType>());
        return copyTensorToType(builder, loc, type, inputs[0]);
      });
  patterns.add<AdjustCallingConventionForFunc>(typeConverter, context);
  patterns.add<AdjustCallingConventionForCall>(typeConverter, context,
                                               typeBoundMap);
  patterns.add<AdjustCallingConventionForReturn>(typeConverter, context);

  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp func) {
    for (int i = 0, e = func.getNumArguments(); i != e; i++) {
      if (func.getArgAttr(i, "torch.type_bound"))
        return false;
      if (func.getArgumentTypes()[i].isa<Torch::NoneType>())
        return false;
    }
    for (int i = 0, e = func.getNumResults(); i != e; i++) {
      if (func.getFunctionType().getResults()[i].isa<Torch::NoneType>())
        return false;
    }
    return true;
  });
  // The dynamic legality conditions for call and return are a pain to write...
  // Just run the patterns once and call it a day.
  //
  // Bug for doing this better https://bugs.llvm.org/show_bug.cgi?id=49812
  DenseSet<Operation *> opsInOriginalProgram;
  func.walk(
      [&](func::CallOp op) { opsInOriginalProgram.insert(op.getOperation()); });
  func.walk([&](func::ReturnOp op) {
    opsInOriginalProgram.insert(op.getOperation());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return !opsInOriginalProgram.contains(op.getOperation());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return !opsInOriginalProgram.contains(op.getOperation());
  });
  target.addLegalOp<CopyToNonValueTensorOp, CopyToValueTensorOp>();
  target.addLegalOp<TensorStaticInfoCastOp>();
  target.addLegalOp<ConstantNoneOp>();
  target.addLegalOp<ConstantIntOp>();
  target.addLegalOp<PrimTupleIndexOp>();
  target.addLegalOp<PrimTupleConstructOp>();
  // We don't know how to rewrite it, so mark it as illegal.
  target.addIllegalOp<func::CallIndirectOp>();
  if (failed(applyPartialConversion(func.getOperation(), target,
                                    std::move(patterns))))
    return failure();
  return success();
}

namespace {
class AdjustCallingConventionsPass
    : public AdjustCallingConventionsBase<AdjustCallingConventionsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    TypeBoundMap typeBoundMap;
    for (auto func : module.getOps<func::FuncOp>()) {
      for (int i = 0, e = func.getNumArguments(); i != e; i++) {
        auto typeBoundAttr =
            func.getArgAttrOfType<TypeAttr>(i, "torch.type_bound");
        if (!typeBoundAttr)
          continue;
        typeBoundMap[{func.getName(), i}] = typeBoundAttr.getValue();
      }
    }
    for (auto func : module.getOps<func::FuncOp>()) {
      if (failed(adjustCallingConventions(func, typeBoundMap)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createAdjustCallingConventionsPass() {
  return std::make_unique<AdjustCallingConventionsPass>();
}
