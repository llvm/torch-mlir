//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/NumpyToTCF/Passes.h"

#include "../PassDetail.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
template <typename TargetTcfOp>
class ConvertBinaryBuiltinUfuncCallOp
    : public OpRewritePattern<Numpy::BuiltinUfuncCallOp> {
public:
  ConvertBinaryBuiltinUfuncCallOp(MLIRContext *context, StringRef qualifiedName,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), qualifiedName(qualifiedName) {}
  LogicalResult matchAndRewrite(Numpy::BuiltinUfuncCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.qualified_name() != qualifiedName)
      return failure();
    if (op.inputs().size() != 2)
      return failure();

    rewriter.replaceOpWithNewOp<TargetTcfOp>(op, op.getResult().getType(),
                                             op.inputs()[0], op.inputs()[1]);
    return success();
  }

private:
  StringRef qualifiedName;
};
} // namespace

namespace {
class ConvertNumpyToTCF : public ConvertNumpyToTCFBase<ConvertNumpyToTCF> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NPCOMP::tcf::TCFDialect>();
  }

  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<ConvertBinaryBuiltinUfuncCallOp<tcf::AddOp>>(context,
                                                                 "numpy.add");
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertNumpyToTCFPass() {
  return std::make_unique<ConvertNumpyToTCF>();
}
