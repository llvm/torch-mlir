//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include <unordered_map>
#include <iostream>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertSelectiveOpToTosaCustom : public ConversionPattern {
public:
  ArrayRef<std::string>  customOps;
  ConvertSelectiveOpToTosaCustom(TypeConverter &typeConverter, MLIRContext *context,
                                 ArrayRef<std::string> customOps)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context){
    this->customOps = customOps;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    std::string op_name = op->getName().getStringRef().str();
    for (auto customOpName : customOps) {
      if (customOpName == op_name) {
        int num_operands = operands.size();
        std::vector<mlir::Value> inputs_vec;
        for (int i = 0; i < num_operands; i++) {
          auto operand = op->getOperands()[i];
          auto operand_type = operand.getType();
          // type convert for operands
          if (operand_type.template isa<torch::Torch::IntType>()) {
            int64_t operand_tosa;
            if (!matchPattern(operand, m_TorchConstantInt(&operand_tosa)))
              return rewriter.notifyMatchFailure(
                  op, "unimplemented: operand should be a torch.constant.int");
            auto operand_tensor_int =
                tosa::getConstTensor<int64_t>(rewriter, op, operand_tosa, {1});
            inputs_vec.push_back(operand_tensor_int.value());
          } else if (operand_type.template isa<torch::Torch::FloatType>()) {
            double operand_tosa;
            if (!matchPattern(operand, m_TorchConstantFloat(&operand_tosa)))
              return rewriter.notifyMatchFailure(
                  op,
                  "unimplemented: operand should be a torch.constant.float");
            auto operand_tensor_float =
                tosa::getConstTensor<float>(rewriter, op, operand_tosa, {1});
            inputs_vec.push_back(operand_tensor_float.value());
          } else if (operand_type
                         .template isa<torch::Torch::ValueTensorType>()) {
            inputs_vec.push_back(operands[i]);
          } else {
            // TODO Handle more types like !torch.list<...>, !torch.device,
            // !torch.string, !torch.none, !torch.generator.
            return rewriter.notifyMatchFailure(
                op, "unimplemented: inputs type. The input has to be int/float/tensor ");
          }
        }
        // Create operands for tosa::CustomOp
        llvm::ArrayRef<mlir::Value> ref(inputs_vec.data(), inputs_vec.size());
        ValueRange custom_inputs(ref);
        // Create output type for tosa::CustomOp
        auto outType =
            getTypeConverter()->convertType(op->getResult(0).getType());

        rewriter.replaceOpWithNewOp<tosa::CustomOp>(
            op, outType, op->getName().getStringRef(), custom_inputs);
        return success();
      }
    }
    return failure();
  }
};

} // namespace

// -----------------------------------------------------------------------------
// TorchToTosaCustom Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTosaCustom
    : public ConvertTorchToTosaCustomBase<ConvertTorchToTosaCustom> {
public:
  ConvertTorchToTosaCustom() = default;
  ConvertTorchToTosaCustom(ArrayRef<std::string> customOps) {
    this->customOps = customOps;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tosa::TosaDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    patterns.add<ConvertSelectiveOpToTosaCustom>(typeConverter,
                                                 context, customOps);

    for (std::string opName : customOps) {
      target.addIllegalOp(OperationName(opName, context));
    }

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTosaCustomPass(
    ArrayRef<std::string> customOps) {
  return std::make_unique<ConvertTorchToTosaCustom>(customOps);
}
