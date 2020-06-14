//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/BasicpyToStd/Passes.h"
#include "npcomp/Conversion/BasicpyToStd/Patterns.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {

class ConvertBasicpyToStd
    : public ConvertBasicpyToStdBase<ConvertBasicpyToStd> {
public:
  void runOnOperation() {
    FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    OwningRewritePatternList patterns;
    populateBasicpyToStdPrimitiveOpPatterns(context, patterns);
    (void)applyPatternsAndFoldGreedily(func, patterns);
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertBasicpyToStdPass() {
  return std::make_unique<ConvertBasicpyToStd>();
}
