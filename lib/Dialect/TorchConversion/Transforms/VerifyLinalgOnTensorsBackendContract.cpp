//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;
using namespace TMTensor;

namespace {
class VerifyLinalgOnTensorsBackendContractPass
    : public VerifyLinalgOnTensorsBackendContractBase<
          VerifyLinalgOnTensorsBackendContractPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto module = getOperation();
    TypeConverter converter;
    converter.addConversion([](RankedTensorType type) -> Type {
      if (BaseMemRefType::isValidElementType(type.getElementType()))
        return type;
      return nullptr;
    });
    TypeConverter scalarConverter;
    for (TypeConverter *c : {&converter, &scalarConverter}) {
      c->addConversion([](FloatType type) { return type; });
      c->addConversion([](IntegerType type) { return type; });
      c->addConversion([](IndexType type) { return type; });
      c->addConversion([](ComplexType type) { return type; });
    }

    auto opHasLegalTypes = [&](Operation *op) { return converter.isLegal(op); };
    auto isLegalScalarOp = [&](Operation *op) {
      // We recognize basic scalar ops by them having the trait "Elementwise",
      // even though we don't expect them to operate on tensors.
      return scalarConverter.isLegal(op) &&
             op->hasTrait<OpTrait::Elementwise>();
    };

    ConversionTarget target(*context);

    // Structural operations.
    target.addDynamicallyLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>(
        opHasLegalTypes);

    target.addDynamicallyLegalOp<GetNextSeedOp>(opHasLegalTypes);

    // Basic scalar operations.
    target.addDynamicallyLegalDialect<func::FuncDialect>(isLegalScalarOp);
    target.addDynamicallyLegalDialect<math::MathDialect>(isLegalScalarOp);
    target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegalScalarOp);
    target.addDynamicallyLegalDialect<complex::ComplexDialect>(isLegalScalarOp);

    // Tensor operations should go through linalg and the tensor dialect.
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<tensor::TensorDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<affine::AffineDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<cf::ControlFlowDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<TMTensorDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<scf::SCFDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<ml_program::MLProgramDialect>(
        opHasLegalTypes);

    // ConstantOp is used for tensors and for scalars.
    target.addDynamicallyLegalOp<arith::ConstantOp>(opHasLegalTypes);
    target.addDynamicallyLegalOp<complex::CreateOp>(opHasLegalTypes);

    RewritePatternSet patterns(context);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      // We avoid `module.emitError()` so that mlir-print-op-on-diagnostics
      // doesn't unnecessarily spew out the entire module.
      emitError(module.getLoc())
          << "Module does not conform to the linalg-on-tensors backend "
             "contract. "
             "See dialect conversion legality information above.";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::TorchConversion::createVerifyLinalgOnTensorsBackendContractPass() {
  return std::make_unique<VerifyLinalgOnTensorsBackendContractPass>();
}
