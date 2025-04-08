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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

namespace {
class VerifyTosaLinalgBackendContractPass
    : public VerifyTosaLinalgBackendContractBase<
          VerifyTosaLinalgBackendContractPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto module = getOperation();
    TypeConverter converter;
    converter.addConversion([](Type type) -> Type {
      auto elemTy = type;
      if (isa<TensorType>(type))
        elemTy = cast<TensorType>(type).getElementType();
      if (isa<MemRefType>(type))
        elemTy = cast<MemRefType>(type).getElementType();
      if (BaseMemRefType::isValidElementType(elemTy))
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

    target.addDynamicallyLegalDialect<func::FuncDialect>(isLegalScalarOp);
    target.addDynamicallyLegalDialect<math::MathDialect>(isLegalScalarOp);
    target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegalScalarOp);
    target.addDynamicallyLegalDialect<bufferization::BufferizationDialect>(
        opHasLegalTypes);
    target.addDynamicallyLegalDialect<complex::ComplexDialect>(isLegalScalarOp);

    // Tensor operations should go through linalg and the tensor dialect.
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<sparse_tensor::SparseTensorDialect>(
        opHasLegalTypes);
    target.addDynamicallyLegalDialect<tensor::TensorDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<affine::AffineDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<cf::ControlFlowDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<scf::SCFDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<memref::MemRefDialect>(opHasLegalTypes);
    target.addDynamicallyLegalDialect<ml_program::MLProgramDialect>(
        opHasLegalTypes);

    // ConstantOp is used for tensors and for scalars.
    target.addDynamicallyLegalOp<arith::ConstantOp>(opHasLegalTypes);
    target.addDynamicallyLegalOp<complex::CreateOp>(opHasLegalTypes);

    // Any TosaOp is valid
    target.addLegalDialect<tosa::TosaDialect>();

    RewritePatternSet patterns(context);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      // We avoid `module.emitError()` so that mlir-print-op-on-diagnostics
      // doesn't unnecessarily spew out the entire module.
      emitError(module.getLoc())
          << "Module does not conform to the "
             "torch-backend-to-tosa-linalg-backend "
             "contract. "
             "See dialect conversion legality information above.";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::TorchConversion::createVerifyTosaLinalgBackendContractPass() {
  return std::make_unique<VerifyTosaLinalgBackendContractPass>();
}
