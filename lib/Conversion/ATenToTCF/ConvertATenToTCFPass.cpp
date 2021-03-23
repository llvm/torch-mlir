//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/ATenToTCF/Passes.h"

#include "../PassDetail.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Conversion/ATenToTCF/Patterns.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {

class ConvertATenToTCF : public ConvertATenToTCFBase<ConvertATenToTCF> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tcf::TCFDialect>();
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateCoreATenToTCFPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertATenToTCFPass() {
  return std::make_unique<ConvertATenToTCF>();
}
