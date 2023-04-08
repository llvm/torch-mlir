//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include <random>
#include <iostream>


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

static void insertInception(MLIRContext *context, Operation *f){
    // this demo insert a Inception into the network

    llvm::SmallPtrSet<Operation *, 16> opWorklist;
    int i = 0;
    f->walk([&](Operation *op){
        if(isa<AtenConvolutionOp>(op)){
            i++;
            opWorklist.insert(op);
        }
        else if(i == 2){
            opWorklist.insert(op);
            i++;
        }
    });

    auto it = opWorklist.begin();
    it++;
    AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
    IRRewriter rewriter(context);
    rewriter.setInsertionPoint(convOp);
    Location loc = convOp.getLoc();
    auto shape = convOp.getOperand(0).getType().cast<ValueTensorType>().getSizes().vec();
    Value int0 = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value int1 = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    
    // parameters of maxPool2dOp
    // kernel_size
    Value list_kernel = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
    // stride
    Value list_stride = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
    // padding
    Value list_padding = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
    // dilation
    Value list_dilation = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
    // ceil_mode
    Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);

    Value maxPool2dOp = rewriter.create<AtenMaxPool2dOp>(
        loc, convOp.getOperand(0).getType(), convOp.getOperand(0), list_kernel, 
        list_stride, list_padding, list_dilation, constFalse
    );

    // parameters of convOp
    // weight
    auto shape_weight = convOp.getOperand(1).getType().cast<ValueTensorType>().getSizes().vec();
    int weightSize = shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
    std::vector<float> WeightVec(weightSize);
    // 随机数引擎对象
    std::mt19937 generator(std::random_device{}());

    // 创建随机数分布器，范围为[-1, 1]
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);

    // 循环对 zeroWeightVec 中的每个元素进行随机化
    for (int i = 0; i < weightSize; ++i) {
        WeightVec[i] = distribution(generator);
    }
    auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_weight),
                                                rewriter.getF32Type());
    auto dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_weight), rewriter.getF32Type()),
        llvm::ArrayRef(WeightVec));
    Value Weight =
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    // bias
    auto shape_bias = convOp.getOperand(2).getType().cast<ValueTensorType>().getSizes().vec();;
    std::vector<float> BiasVec(shape_bias[0]);
    for (int i = 0; i < shape_bias[0]; ++i) {
        BiasVec[i] = distribution(generator);
    }
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_bias),
                                            rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_bias), rewriter.getF32Type()),
        llvm::ArrayRef(BiasVec));
    Value Bias =
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    // conv
    Value list = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange());
    Value randomConv = rewriter.create<AtenConvolutionOp>(
        loc, convOp.getType(), maxPool2dOp, Weight, Bias, 
        list_stride, list_padding, list_dilation, constFalse, list, int1);
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        convOp, convOp.getType(), convOp.getOperand(0), convOp.getOperand(1), convOp.getOperand(2), 
        convOp.getOperand(3),convOp.getOperand(4), convOp.getOperand(5), 
        convOp.getOperand(6),convOp.getOperand(7), convOp.getOperand(8));
    it++;
    AtenReluOp reluOp = llvm::dyn_cast<AtenReluOp>(*it);
    rewriter.setInsertionPoint(reluOp);
    loc = reluOp.getLoc();
    // cat
    shape = reluOp.getOperand().getType().cast<ValueTensorType>().getSizes().vec();
    auto shape_cat = shape;
    shape_cat[1] = shape[1] * 2;
    int size_cat = shape_cat[0] * shape_cat[1] * shape_cat[2] * shape_cat[3];
    std::vector<float> valueCatVec(size_cat);
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_cat),
                                                rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_cat), rewriter.getF32Type()),
        llvm::ArrayRef(valueCatVec));
    Value valueCat = 
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    Value listTensor = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(ValueTensorType::get(context, llvm::ArrayRef(shape), 
        rewriter.getF32Type())), ValueRange({reluOp.getOperand(), randomConv}));
    Value cat = rewriter.create<AtenCatOp>(
        loc, valueCat.getType(), listTensor, int1);
    // slice
    Value int_start = 
    rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

    Value int_end = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[1]));

    Value int_dim = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));

    Value slice = rewriter.create<AtenSliceTensorOp>(
        loc, reluOp.getOperand().getType(), cat, int_dim, int_start, int_end, int1);
    
    // relu
    rewriter.replaceOpWithNewOp<AtenReluOp>(
        reluOp, reluOp.getType(), slice);
}

namespace {
class InsertInceptionPass : public InsertInceptionBase<InsertInceptionPass> {
public:
    InsertInceptionPass() = default;
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        auto f = getOperation();
        insertInception(context, f);
    }
};
} //namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertInceptionPass() {
  return std::make_unique<InsertInceptionPass>();
}