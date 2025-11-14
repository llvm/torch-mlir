//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using llvm::dbgs;
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;
namespace mlir::torch::onnx_c {

#define GEN_PASS_DEF_CONVERTTORCHONNXTOTORCH
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h.inc"

#define DEBUG_TYPE "torch-onnx"

namespace {

int64_t getDefaultOpsetVersion(Operation *containerOp) {
  auto attr =
      containerOp->getAttrOfType<IntegerAttr>("torch.onnx_meta.opset_version");
  if (!attr)
    return 0;
  if (auto type = dyn_cast<IntegerType>(attr.getType())) {
    if (!type || !type.isSigned())
      return 0;
  }
  return attr.getSInt();
}

class ConvertTorchOnnxToTorch
    : public impl::ConvertTorchOnnxToTorchBase<ConvertTorchOnnxToTorch> {
public:
  ConvertTorchOnnxToTorch() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // Populate our patterns for each handled domain.
    int64_t defaultOpsetVersion = getDefaultOpsetVersion(getOperation());
    if (defaultOpsetVersion == 0) {
      emitError(getOperation().getLoc())
          << "function is missing onnx opset version attribute "
             "(torch.onnx_meta.opset_version)";
      return signalPassFailure();
    }

    auto defaultDomainPatterns =
        std::make_unique<OnnxCustomOpConversionPattern>(
            context, "onnx.",
            /*domainVersion=*/defaultOpsetVersion);
    populateComMicrosoftDomain(*defaultDomainPatterns);
    populateDefaultDomainAtoF(*defaultDomainPatterns);
    populateDefaultDomainGtoP(*defaultDomainPatterns);
    populateDefaultDomainQtoZ(*defaultDomainPatterns);

    // Ask each domain for its handled names and configure the
    // conversion target.
    ConversionTarget target(*context);
    DenseSet<StringAttr> legalizedNames;
    defaultDomainPatterns->populateLegalizedNames(legalizedNames);
    target.addLegalDialect<Torch::TorchDialect>();
    target.addDynamicallyLegalOp<Torch::OperatorOp>([&](Torch::OperatorOp op) {
      return !legalizedNames.contains(op.getNameAttr());
    });

    RewritePatternSet patterns(context);
    patterns.insert(std::move(defaultDomainPatterns));

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTorchOnnxToTorchPass() {
  return std::make_unique<ConvertTorchOnnxToTorch>();
}

} // namespace mlir::torch::onnx_c
