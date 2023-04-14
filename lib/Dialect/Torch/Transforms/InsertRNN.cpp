//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include <iostream>
#include <random>

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

static void insertRNN(MLIRContext *context, Operation *f, int number) {
  // this demo insert a insert an RNN after Relu
  // 此RNN保证输入等于输出

  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenMaxPool2dOp>(op)) {
      opWorklist.insert(op);
    }
  });
  auto it = opWorklist.begin();
  AtenMaxPool2dOp maxpool2dOp = llvm::dyn_cast<AtenMaxPool2dOp>(*it);
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(maxpool2dOp);
  Location loc = maxpool2dOp.getLoc();
  Value rst = maxpool2dOp.getOperand(0);
  auto shape = maxpool2dOp.getOperand(0)
                   .getType()
                   .cast<ValueTensorType>()
                   .getSizes()
                   .vec();
  int cycles = number; // RNN层数
  // create a RNN to make sure output is the same as input
  Value int0 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value int1 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));

  // hidden
  auto shape_hidden = shape;
  shape_hidden.erase(shape_hidden.begin(), shape_hidden.begin() + 1);
  shape_hidden[0] = shape[0] * shape[1];
  int size_hidden = shape_hidden[0] * shape_hidden[1] * shape_hidden[2];
  std::vector<float> zeroHiddenVec(size_hidden, 0);
  Value zeroHidden =
      Torch::createTensor(rewriter, loc, context, shape_hidden, zeroHiddenVec);

  // slice_type
  auto shape_slice = shape;
  shape_slice[0] = 1;
  int size_slice =
      shape_slice[0] * shape_slice[1] * shape_slice[2] * shape_slice[3];
  std::vector<float> zeroSliceVec(size_slice, 0);
  Value zeroSlice =
      Torch::createTensor(rewriter, loc, context, shape_slice, zeroSliceVec);

  // view
  auto shape_view = shape;
  shape_view[0] = 1;
  shape_view[1] = shape[0] * shape[1];
  Value view = Torch::createReshape(rewriter, loc, context, shape_view, rst);

  Value cat_extra;
  std::vector<Value> values(cycles);
  values[0] = view;
  if (cycles > 1) {
    for (int i = 0; i < cycles - 1; i++) {
      values[i + 1] = view;
    }
    // PrimListConstructOp
    Value list_extra = rewriter.create<PrimListConstructOp>(
        loc,
        ListType::get(
            ValueTensorType::get(context, shape_view, rewriter.getF32Type())),
        ValueRange(values));
    // cat
    auto shape_cat = shape;
    shape_cat[0] = shape_view[0] * cycles;
    int size_cat = shape_cat[0] * shape_cat[1] * shape_cat[2] * shape_cat[3];
    std::vector<float> zeroCatVec(size_cat);
    Value zeroCat =
        Torch::createTensor(rewriter, loc, context, shape_cat, zeroCatVec);
    cat_extra =
        rewriter.create<AtenCatOp>(loc, zeroCat.getType(), list_extra, int0);
  }
  // cat_type
  auto shape_cat = shape;
  shape_cat.erase(shape_cat.begin(), shape_cat.begin() + 1);
  shape_cat[1] = shape[2] * 2;
  int size_cat = shape_cat[0] * shape_cat[1] * shape_cat[2];
  std::vector<float> zeroCatVec(size_cat);
  Value zeroCat =
      Torch::createTensor(rewriter, loc, context, shape_cat, zeroCatVec);

  // transpose_hidden
  auto shape_transpose = shape;
  shape_transpose.erase(shape_transpose.begin(), shape_transpose.begin() + 2);
  shape_transpose[0] = shape_transpose[1] = shape[2];
  int size_transpose = shape_transpose[0] * shape_transpose[1];
  std::vector<float> valueTransposeVec(size_transpose);
  for (int i = 0; i < shape_transpose[0]; i++) {
    for (int j = 0; j < shape_transpose[1]; j++) {
      if (i == j) {
        valueTransposeVec[i * shape_transpose[1] + j] = 1.0f;
      } else {
        valueTransposeVec[i * shape_transpose[1] + j] = 0.0f;
      }
    }
  }
  Value valueTranspose = Torch::createTensor(
      rewriter, loc, context, shape_transpose, valueTransposeVec);

  // add_hidden
  Value float1 =
      rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
  auto shape_add = shape;
  shape_add.erase(shape_add.begin() + 1, shape_add.end());
  shape_add[0] = shape[3];
  std::vector<float> valueAddVec(shape_add[0], 0);

  Value valueAdd =
      Torch::createTensor(rewriter, loc, context, shape_add, valueAddVec);

  Value int_start =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

  Value int_end =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[2]));

  Value int_dim =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));

  Value slice;
  Value slice_relu;
  Value input;

  for (int i = 0; i < cycles; i++) {
    if (cycles > 1) {
      // slice
      Value int_num1 =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i));
      Value int_num2 = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i + 1));
      input = rewriter.create<AtenSliceTensorOp>(
          loc, zeroSlice.getType(), cat_extra, int0, int_num1, int_num2, int1);
    } else {
      input = rst;
    }

    // squeeze
    Value squeeze_dim = rewriter.create<AtenSqueezeDimOp>(
        loc, zeroHidden.getType(), input, int0);
    // PrimListconstruct
    Value listTensor;
    if (i == 0) {
      listTensor = rewriter.create<PrimListConstructOp>(
          loc,
          ListType::get(ValueTensorType::get(
              context, llvm::ArrayRef(shape_hidden), rewriter.getF32Type())),
          ValueRange({squeeze_dim, zeroHidden}));
    } else {
      listTensor = rewriter.create<PrimListConstructOp>(
          loc,
          ListType::get(ValueTensorType::get(
              context, llvm::ArrayRef(shape_hidden), rewriter.getF32Type())),
          ValueRange({squeeze_dim, slice_relu}));
    }
    // cat
    Value cat =
        rewriter.create<AtenCatOp>(loc, zeroCat.getType(), listTensor, int1);

    // transpose
    Value intTranspose_1 = rewriter.create<AtenTransposeIntOp>(
        loc, valueTranspose.getType(), valueTranspose, int0, int1);

    // matmul
    Value matmul_1 =
        rewriter.create<AtenMatmulOp>(loc, cat.getType(), cat, intTranspose_1);

    // add
    Value add_1 = rewriter.create<AtenAddTensorOp>(loc, matmul_1.getType(),
                                                   matmul_1, valueAdd, float1);
    // relu
    Value relu_hidden =
        rewriter.create<AtenReluOp>(loc, add_1.getType(), add_1);

    // slice_relu
    slice_relu = rewriter.create<AtenSliceTensorOp>(loc, zeroHidden.getType(),
                                                    relu_hidden, int_dim,
                                                    int_start, int_end, int1);

    if (i == cycles - 1) { // 最后一次for循环对output进行修改
      // transpose
      auto shape_transpose_2 = shape;
      shape_transpose_2.erase(shape_transpose_2.begin(),
                              shape_transpose_2.begin() + 2);
      shape_transpose_2[0] = shape_transpose_2[1] = shape[2];
      int size_transpose_2 = shape_transpose_2[0] * shape_transpose_2[1];
      std::vector<float> valueTranspose_2Vec(size_transpose_2);
      for (int i = 0; i < shape_transpose_2[0]; i++) {
        for (int j = 0; j < shape_transpose_2[1]; j++) {
          if (i == j) {
            valueTranspose_2Vec[i * shape_transpose_2[1] + j] = 1.0f;
          } else {
            valueTranspose_2Vec[i * shape_transpose_2[1] + j] = 0.0f;
          }
        }
      }
      Value valueTranspose_2 = Torch::createTensor(
          rewriter, loc, context, shape_transpose_2, valueTranspose_2Vec);
      Value intTranspose_2 = rewriter.create<AtenTransposeIntOp>(
          loc, valueTranspose.getType(), valueTranspose_2, int0, int1);
      // matmul
      Value matmul_2 = rewriter.create<AtenMatmulOp>(loc, cat.getType(), cat,
                                                     intTranspose_2);
      // add
      auto shape_add_2 = shape;
      shape_add_2.erase(shape_add_2.begin() + 1, shape_add_2.end());
      shape_add_2[0] = shape[3];
      std::vector<float> valueAdd_2Vec(shape_add_2[0], 0);

      Value valueAdd_2 = Torch::createTensor(rewriter, loc, context,
                                             shape_add_2, valueAdd_2Vec);
      Value add_2 = rewriter.create<AtenAddTensorOp>(
          loc, matmul_2.getType(), matmul_2, valueAdd_2, float1);
      // relu
      Value relu_2 = rewriter.create<AtenReluOp>(loc, add_2.getType(), add_2);
      // 将结果的形状还原为经过rnn之前
      // unsqueeze
      auto shape_unsqueeze = shape;
      shape_unsqueeze[0] = shape[0];
      shape_unsqueeze[1] = shape_cat[0];
      shape_unsqueeze[2] = shape_cat[1];
      shape_unsqueeze[3] = shape_cat[2];
      Value unsqueeze = rewriter.create<AtenUnsqueezeOp>(
          loc,
          ValueTensorType::get(context, llvm::ArrayRef(shape_unsqueeze),
                               rewriter.getF32Type()),
          relu_2, int0);

      // slice
      Value int_start =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

      Value int_end = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(shape[2]));

      Value int_dim =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(2));

      slice = rewriter.create<AtenSliceTensorOp>(
          loc, maxpool2dOp.getOperand(0).getType(), unsqueeze, int_dim,
          int_start, int_end, int1);
    }
  }

  rewriter.replaceOpWithNewOp<AtenMaxPool2dOp>(
      maxpool2dOp, maxpool2dOp.getType(), slice, maxpool2dOp.getOperand(1),
      maxpool2dOp.getOperand(2), maxpool2dOp.getOperand(3),
      maxpool2dOp.getOperand(4), maxpool2dOp.getOperand(5));
}

namespace {
class InsertRNNPass : public InsertRNNBase<InsertRNNPass> {
public:
  InsertRNNPass() = default;
  InsertRNNPass(int number) { this->number = number; }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    insertRNN(context, f, number);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertRNNPass(int number) {
  return std::make_unique<InsertRNNPass>(number);
}
