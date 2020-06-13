//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/E2E/E2E.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using mlir::LLVM::LLVMType;

namespace {
class LowerAbortIf : public OpConversionPattern<tcp::AbortIfOp> {
public:
  LowerAbortIf(LLVM::LLVMFuncOp abortIfFunc)
      : OpConversionPattern(abortIfFunc.getContext()),
        abortIfFunc(abortIfFunc) {}
  LogicalResult
  matchAndRewrite(tcp::AbortIfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    tcp::AbortIfOp::OperandAdaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, abortIfFunc, adaptor.pred());
    return success();
  }
  LLVM::LLVMFuncOp abortIfFunc;
};
} // namespace

// Create the LLVM function declaration for our runtime function
// that backs the tcp.abort_if op.
LLVM::LLVMFuncOp createAbortIfFuncDecl(ModuleOp module) {
  auto *llvmDialect =
      module.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  auto abortIfFuncTy = LLVMType::getFunctionTy(
      LLVMType::getVoidTy(llvmDialect), {LLVMType::getInt1Ty(llvmDialect)},
      /*isVarArg=*/false);
  OpBuilder builder(module.getBodyRegion());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "__npcomp_abort_if",
                                          abortIfFuncTy,
                                          LLVM::Linkage::External);
}

namespace {
class LowerToLLVM : public LowerToLLVMBase<LowerToLLVM> {
  void runOnOperation() {
    auto module = getOperation();
    auto *context = &getContext();

    LLVM::LLVMFuncOp abortIfFunc = createAbortIfFuncDecl(module);

    LLVMTypeConverter converter(context);
    OwningRewritePatternList patterns;
    LLVMConversionTarget target(*context);
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.insert<LowerAbortIf>(abortIfFunc);

    if (failed(applyFullConversion(module, target, patterns, &converter))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::NPCOMP::createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVM>();
}
