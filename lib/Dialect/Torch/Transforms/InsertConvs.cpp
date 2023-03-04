//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include <cstdlib>
#include <ctime>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static void insertConvs(MLIRContext *context, Operation *f) {
  // insert a invariant convolution

  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (!isa<mlir::func::ReturnOp, mlir::func::FuncOp>(op)) {
      if (auto t = op->getResult(0).getType().dyn_cast<ValueTensorType>()) {
        if (t.getSizes().size() == 4 && op->getNumResults() == 1) {
          opWorklist.insert(op);
          // llvm::outs() << op->getName() << "\n";
        }
      }
    }
  });

  // select a random place to insert
  auto it = opWorklist.begin();
  std::srand(std::time(0));
  Operation* originOp = *(std::next(it, std::rand() % opWorklist.size()));
  // llvm::outs() << "=========select random op\n";
  // llvm::outs() << originOp->getName() << "\n";
  // llvm::outs() << originOp->getResult(0).getType() << "\n";
  IRRewriter rewriter(context);
  rewriter.setInsertionPointAfter(originOp);
  // copy origin op
  Operation* op = rewriter.clone(*originOp);
  Location loc = op->getLoc();

  // create unit tensor as convolution kernel
  auto rst = op->getResult(0);
  auto shape = rst.getType().cast<ValueTensorType>().getSizes().vec();
  int sz = shape[1];
  // new kernel size is: sz x sz x 1 x 1
  shape[0] = sz;
  shape[2] = shape[3] = 1;
  std::vector<float> unitWeightVec(sz * sz, 0);
  for (int i = 0; i < sz; i++) {
    unitWeightVec[i * sz + i] = 1;
  }
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                               rewriter.getF32Type());
  auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(unitWeightVec));
  Value unitWeight =
      rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  // create zero bias
  shape.erase(shape.begin() + 1, shape.end());
  std::vector<float> zeroBiasVec(shape[0], 0);
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                          rewriter.getF32Type());
  dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(zeroBiasVec));
  Value zeroBias =
      rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  // create other oprands for conv
  Value int0 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value int1 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);
  Value listInt0_0 = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
  Value listInt1_1 = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
  Value listInt = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({}));
  // create conv
  Value conv = rewriter.create<AtenConvolutionOp>(
      loc, rst.getType(), rst, unitWeight, zeroBias, listInt1_1, listInt0_0,
      listInt1_1, constFalse, listInt, int1);
  originOp->replaceAllUsesWith(ValueRange({conv}));
  originOp->erase();
}

namespace {
class InsertConvsPass : public InsertConvsBase<InsertConvsPass> {
public:
  InsertConvsPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    insertConvs(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertConvsPass() {
  return std::make_unique<InsertConvsPass>();
}