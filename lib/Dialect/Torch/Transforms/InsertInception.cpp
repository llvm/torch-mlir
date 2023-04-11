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

static void insertInception(MLIRContext *context, Operation *f, int number){
    // this demo insert a Inception into the network

    llvm::SmallPtrSet<Operation *, 16> opWorklist;
    f->walk([&](Operation *op){
        if(isa<AtenConvolutionOp>(op)){
            opWorklist.insert(op);
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
    int number_branch = number;
    std::random_device rd;  // 用于获取真随机数种子
    std::mt19937 gen(rd());  // 以真随机数种子生成随机数生成器
    std::uniform_int_distribution<> dis(1, 3);  // 定义随机数分布，范围为 1 到 3

    // common parameters
    Value list_stride = rewriter.create<PrimListConstructOp>(
         loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
    Value list_dilation = rewriter.create<PrimListConstructOp>(
         loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
    Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);
    Value list = rewriter.create<PrimListConstructOp>(
         loc, ListType::get(IntType::get(context)), ValueRange());
    
    std::vector<Value> values(number_branch + 1);
    values[0] = convOp.getOperand(0);
    for(int i = 0; i < number_branch; i++){
        int branch_structure = dis(gen);
        if(branch_structure == 1){
            // 添加一个1 * 1的卷积
            auto shape_weight = convOp.getOperand(1).getType().cast<ValueTensorType>().getSizes().vec();
            shape_weight[0] = shape_weight[1];
            shape_weight[2] = shape_weight[3] = 1;
            int weightSize = shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
            std::vector<float> WeightVec(weightSize);
            // 随机数引擎对象
            std::mt19937 generator(std::random_device{}());

            // 创建随机数分布器，范围为[-1, 1]
            std::uniform_real_distribution<float> distribution(-1.f, 1.f);

            // 循环对 WeightVec 中的每个元素进行随机化
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
            auto shape_bias = convOp.getOperand(2).getType().cast<ValueTensorType>().getSizes().vec();
            shape_bias[0] = shape_weight[1];
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
            Value list_padding = rewriter.create<PrimListConstructOp>(
                loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
            values[i + 1] = rewriter.create<AtenConvolutionOp>(
                loc, convOp.getOperand(0).getType(), convOp.getOperand(0), Weight, Bias, 
                 list_stride, list_padding, list_dilation, constFalse, list, int1);
            // list_cat = rewriter.create<PrimListConstructOp>(
            //     loc, ListType::get(ValueTensorType::get(context, shape_list_cat, 
            //     rewriter.getF32Type())), ValueRange(randomConv));
        }
        else if(branch_structure == 2){
             // 添加一个1 * 1的卷积
            auto shape_weight = convOp.getOperand(1).getType().cast<ValueTensorType>().getSizes().vec();
            shape_weight[0] = shape_weight[1];
            shape_weight[2] = shape_weight[3] = 1;
            int weightSize = shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
            std::vector<float> WeightVec(weightSize);
            // 随机数引擎对象
            std::mt19937 generator(std::random_device{}());

            // 创建随机数分布器，范围为[-1, 1]
            std::uniform_real_distribution<float> distribution(-1.f, 1.f);

            // 循环对 WeightVec 中的每个元素进行随机化
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
            auto shape_bias = convOp.getOperand(2).getType().cast<ValueTensorType>().getSizes().vec();
            shape_bias[0] = shape_weight[1];
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
            Value list_padding = rewriter.create<PrimListConstructOp>(
                loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
            Value randomConv_1 = rewriter.create<AtenConvolutionOp>(
                loc, convOp.getOperand(0).getType(), convOp.getOperand(0), Weight, Bias, 
                 list_stride, list_padding, list_dilation, constFalse, list, int1);
            Value relu_1 = rewriter.create<AtenReluOp>(
                loc, convOp.getOperand(0).getType(), randomConv_1);
            
            // 改变kernel_size的大小,进行第二次卷积
            int padNum = dis(gen);
            Value intPad =
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(padNum));
            list_padding = rewriter.create<PrimListConstructOp>(
                loc, ListType::get(IntType::get(context)), ValueRange({intPad, intPad}));
            int kernel = padNum * 2 + 1;
            shape_weight[2] = shape_weight[3] = kernel;
            weightSize = shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
            std::vector<float> WeightVec_2(weightSize);
            for (int i = 0; i < weightSize; ++i) {
                WeightVec_2[i] = distribution(generator);
            }
            resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_weight),
                                                        rewriter.getF32Type());
            dense = DenseElementsAttr::get(
                RankedTensorType::get(llvm::ArrayRef(shape_weight), rewriter.getF32Type()),
                llvm::ArrayRef(WeightVec_2));
            Weight =
                rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
            // bias
            for (int i = 0; i < shape_bias[0]; ++i) {
                BiasVec[i] = distribution(generator);
            }
            resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_bias),
                                                    rewriter.getF32Type());
            dense = DenseElementsAttr::get(
                RankedTensorType::get(llvm::ArrayRef(shape_bias), rewriter.getF32Type()),
                llvm::ArrayRef(BiasVec));
            Bias =
                rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
            Value randomConv_2 = rewriter.create<AtenConvolutionOp>(
                loc, convOp.getOperand(0).getType(), relu_1, Weight, Bias, 
                 list_stride, list_padding, list_dilation, constFalse, list, int1);
            values[i + 1] = rewriter.create<AtenReluOp>(
                loc, convOp.getOperand(0).getType(), randomConv_2);
            // list_cat = rewriter.create<PrimListConstructOp>(
            //     loc, ListType::get(ValueTensorType::get(context, shape_list_cat, 
            //     rewriter.getF32Type())), ValueRange(relu_2));
        }
        else if(branch_structure == 3){
            // maxpool2dOp
            int padNum = dis(gen);
            Value intPad =
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(padNum));
            Value list_padding = rewriter.create<PrimListConstructOp>(
                loc, ListType::get(IntType::get(context)), ValueRange({intPad, intPad}));
            int kernel = padNum * 2 + 1;
            Value intKernel =
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(kernel));
            Value list_kernel = rewriter.create<PrimListConstructOp>(
                loc, ListType::get(IntType::get(context)), ValueRange({intKernel, intKernel}));
            Value maxPool2dOp = rewriter.create<AtenMaxPool2dOp>(
                loc, convOp.getOperand(0).getType(), convOp.getOperand(0), list_kernel, 
                list_stride, list_padding, list_dilation, constFalse);
            
            // convOp
            padNum = dis(gen);
            intPad =
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(padNum));
            list_padding = rewriter.create<PrimListConstructOp>(
                loc, ListType::get(IntType::get(context)), ValueRange({intPad, intPad}));
            kernel = padNum * 2 + 1;
            auto shape_weight = convOp.getOperand(1).getType().cast<ValueTensorType>().getSizes().vec();
            shape_weight[0] = shape_weight[1];
            shape_weight[2] = shape_weight[3] = kernel;
            int weightSize = shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
            std::vector<float> WeightVec(weightSize);
            // 随机数引擎对象
            std::mt19937 generator(std::random_device{}());

            // 创建随机数分布器，范围为[-1, 1]
            std::uniform_real_distribution<float> distribution(-1.f, 1.f);

            // 循环对 WeightVec 中的每个元素进行随机化
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
            auto shape_bias = convOp.getOperand(2).getType().cast<ValueTensorType>().getSizes().vec();
             shape_bias[0] = shape_weight[1];
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
            Value randomConv = rewriter.create<AtenConvolutionOp>(
                loc, convOp.getOperand(0).getType(), maxPool2dOp, Weight, Bias, 
                 list_stride, list_padding, list_dilation, constFalse, list, int1);
            values[i + 1] = rewriter.create<AtenReluOp>(
                loc, convOp.getOperand(0).getType(), randomConv);
            // list_cat = rewriter.create<PrimListConstructOp>(
            //     loc, ListType::get(ValueTensorType::get(context, shape_list_cat, 
            //     rewriter.getF32Type())), ValueRange(relu));
        }
    }

    // list
    auto shape_list_cat = convOp.getOperand(0).getType().cast<ValueTensorType>().getSizes().vec();
    Value list_cat = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(ValueTensorType::get(context, shape_list_cat, 
        rewriter.getF32Type())), ValueRange(values));
    
    // cat
    auto shape_cat = shape;
    shape_cat[0] = shape[0] * (number_branch + 1);
    int size_cat = shape_cat[0] * shape_cat[1] * shape_cat[2] * shape[3];
    std::vector<float> catVec(size_cat);
    auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_cat),
                                                  rewriter.getF32Type());
    auto dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_cat), rewriter.getF32Type()),
        llvm::ArrayRef(catVec));
    Value zeroCat =
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    Value cat = rewriter.create<AtenCatOp>(
        loc, zeroCat.getType(), list_cat, int0);
    
    // slice
    Value int_start = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

    Value int_end = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[0]));

    Value int_dim = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

    Value slice = rewriter.create<AtenSliceTensorOp>(
        loc, convOp.getOperand(0).getType(), cat, int_dim, int_start, int_end, int1);
    
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        convOp, convOp.getType(), slice, convOp.getOperand(1), convOp.getOperand(2), 
        convOp.getOperand(3),convOp.getOperand(4), convOp.getOperand(5), 
        convOp.getOperand(6),convOp.getOperand(7), convOp.getOperand(8));
}

namespace {
class InsertInceptionPass : public InsertInceptionBase<InsertInceptionPass> {
public:
    InsertInceptionPass() = default;
    InsertInceptionPass(int number){
        this->number = number;
    }
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        auto f = getOperation();
        insertInception(context, f, number);
    }
};
} //namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertInceptionPass(int number) {
  return std::make_unique<InsertInceptionPass>(number);
}