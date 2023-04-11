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

static void insertRNNWithZeros(MLIRContext *context, Operation *f, std::string activationFunc, int number){
    // this demo insert a RNN into the network
    // 纯粹增加了一个RNN循环层,最后通过残差连接达到传播原tensor的目的
    llvm::SmallPtrSet<Operation *, 16> opWorklist;
    f->walk([&](Operation *op) {
    if (isa<AtenReluOp>(op)) {
      if (op->getResult(0).getType().isa<ValueTensorType>()) {
        opWorklist.insert(op);
      }
    }
    });

    auto it = opWorklist.begin();
    AtenReluOp reluOp = llvm::dyn_cast<AtenReluOp>(*it);
    IRRewriter rewriter(context);
    rewriter.setInsertionPoint(reluOp);
    Location loc = reluOp.getLoc();


    // create other oprands for RNN
    Value int0 =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value int1 =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value float0 = 
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0));
    Value float1 = 
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
    Value rst = reluOp.getOperand();
    auto shape = reluOp.getOperand().getType().cast<ValueTensorType>().getSizes().vec();
    // create related parameters
    int cycles = number; // RNN层数

    // hidden
    auto shape_hidden = shape;
    shape_hidden.erase(shape_hidden.begin(), shape_hidden.begin()+1);
    int size_hidden = shape_hidden[0] * shape_hidden[1] * shape_hidden[2];
    std::vector<float> zeroHiddenVec(size_hidden, 0);
    auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_hidden),
                                                 rewriter.getF32Type());
    auto dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_hidden), rewriter.getF32Type()),
        llvm::ArrayRef(zeroHiddenVec));
    Value zeroHidden = 
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    
    // valueTranspose
    auto shape_transpose_a = shape;
    shape_transpose_a.erase(shape_transpose_a.begin(), shape_transpose_a.begin()+2);
    shape_transpose_a[0] = shape_transpose_a[1] = shape[2];
    int size_transpose_a = shape_transpose_a[0] * shape_transpose_a[1];
    std::vector<float> zeroTranspose_aVec(size_transpose_a, 0);
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_transpose_a),
                                                 rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_transpose_a), rewriter.getF32Type()),
        llvm::ArrayRef(zeroTranspose_aVec));
    Value valueTranspose_a = 
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

    // add
    auto shape_add = shape;
    shape_add.erase(shape_add.begin() + 1 , shape_add.end());
    std::vector<float> valueAddVec(shape_add[0], 0);
    shape_add[0] = shape[3];
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_add),
                                                 rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_add), rewriter.getF32Type()),
        llvm::ArrayRef(valueAddVec));
    Value valueAdd = 
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);  
    
    // create result_type
    // slice_type
    auto shape_slice = shape;
    shape_slice[0] = 1;
    int size_slice = shape_slice[0] * shape_slice[1] * shape_slice[2] * shape_slice[3];
    std::vector<float> zeroSliceVec(size_slice, 0);
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_slice),
                                                 rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_slice), rewriter.getF32Type()),
        llvm::ArrayRef(zeroSliceVec));
    Value zeroSlice = 
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

    // cat_type
    auto shape_cat = shape;
    shape_cat.erase(shape_cat.begin(), shape_cat.begin()+1);
    shape_cat[1] = shape_hidden[1] + shape_hidden[1];
    int size_cat = shape_cat[0] * shape_cat[1] * shape_cat[2];
    std::vector<float> zeroCatVec(size_cat, 0);
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_cat),
                                                 rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_cat), rewriter.getF32Type()),
        llvm::ArrayRef(zeroCatVec));
    Value zeroCat = 
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    
    // parameters of slice
    Value int_start = 
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

    Value int_end = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[2]));

    Value int_dim = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    
    // reshape
    Value int_shape_0 = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value int_shape_1 = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[0] * shape[1]));
    Value int_shape_2 = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[2]));
    Value int_shape_3 = 
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[3]));
    Value list_reshape = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), 
        ValueRange({int_shape_0, int_shape_1, int_shape_2, int_shape_3}));
    
    // view
    auto shape_view = shape;
    shape_view[0] = 1;
    shape_view[1] = shape[0] * shape[1];
    int size_view = shape_view[0] * shape_view[1] * shape_view[2] * shape_view[3];
    std::vector<float> zeroViewVec(size_view, 0);
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_view),
                                                 rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape_view), rewriter.getF32Type()),
        llvm::ArrayRef(zeroViewVec));
    Value zeroView = 
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    Value view = rewriter.create<AtenViewOp>(
        loc, zeroView.getType(), rst, list_reshape);
    
    Value cat_extra;
    std::vector<Value> values(cycles);
    values[0] = view;
    if(cycles > 1){
        for(int i = 0; i < cycles - 1; i++){
            values[i + 1] = view;
        }
        // PrimListConstructOp
        Value list_extra = rewriter.create<PrimListConstructOp>(
            loc, ListType::get(ValueTensorType::get(context, shape_view, 
            rewriter.getF32Type())), ValueRange(values));
        // cat
        auto shape_cat = shape;
        shape_cat[0] = shape_view[0] * cycles;
        int size_cat = shape_cat[0] * shape_cat[1] * shape_cat[2] * shape_cat[3];
        std::vector<float> zeroCatVec(size_cat);
        resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape_cat),
                                                    rewriter.getF32Type());
        dense = DenseElementsAttr::get(
            RankedTensorType::get(llvm::ArrayRef(shape_cat), rewriter.getF32Type()),
            llvm::ArrayRef(zeroCatVec));
        Value zeroCat = 
            rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
        cat_extra = rewriter.create<AtenCatOp>(
            loc, zeroCat.getType(), list_extra, int0);
    }
    
    Value slice_relu;
    Value slice;
    Value input;
    
    for(int i = 0;i < cycles;i++){
        if(cycles > 1){
            // slice
            Value int_num1 =
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i));
            Value int_num2 =
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i+1));
            input = rewriter.create<AtenSliceTensorOp>(
                loc, zeroSlice.getType(), cat_extra, int0, int_num1, int_num2, int1);
        }
        else{
            input = rst;
        }

        // squeeze_dim
        Value squeeze_dim = rewriter.create<AtenSqueezeDimOp>(
            loc, zeroHidden.getType(), rst, int0);

        // PrimListconstruct
        Value listTensor;
        if(i == 0){
            listTensor = 
            rewriter.create<PrimListConstructOp>(
                loc, ListType::get(ValueTensorType::get(context, llvm::ArrayRef(shape_hidden), 
                rewriter.getF32Type())), ValueRange({squeeze_dim, zeroHidden})
            );
        }
        else{
            listTensor = 
            rewriter.create<PrimListConstructOp>(
                loc, ListType::get(ValueTensorType::get(context, llvm::ArrayRef(shape_hidden), 
                rewriter.getF32Type())), ValueRange({squeeze_dim, slice_relu})
            );
        }

        // cat
        Value cat = rewriter.create<AtenCatOp>(
            loc, zeroCat.getType(), listTensor, int1);
        
        // 以下四步对hidden进行linear和relu操作
        // transpose
        Value transposeInt_hidden = rewriter.create<AtenTransposeIntOp>(
            loc, valueTranspose_a.getType(), valueTranspose_a, int0, int1);

        // matmul
        Value matmul_hidden = rewriter.create<AtenMatmulOp>(
            loc, cat.getType(), cat, transposeInt_hidden);
        
        // add
        Value add_hidden = rewriter.create<AtenAddTensorOp>(
            loc, matmul_hidden.getType(), matmul_hidden, valueAdd, float1);

        // random activation
        Value random_hidden;
        if(activationFunc == "relu" || activationFunc == ""){
            random_hidden = rewriter.create<AtenReluOp>(
                loc, add_hidden.getType(), add_hidden);
        }
        else if(activationFunc == "sigmoid"){
            random_hidden = rewriter.create<AtenSigmoidOp>(
                loc, add_hidden.getType(), add_hidden);
        }
        else if(activationFunc == "tanh"){
            random_hidden = rewriter.create<AtenTanhOp>(
                loc, add_hidden.getType(), add_hidden);
        }
        
        // slice_relu
        slice_relu = rewriter.create<AtenSliceTensorOp>(
            loc, zeroHidden.getType(), random_hidden, int_dim, int_start, int_end, int1);
        
        if(i == cycles - 1){ // 最后一次for循环对output进行修改
            // 以下三步对output进行linear操作
            // transpose
            Value transposeInt_output = rewriter.create<AtenTransposeIntOp>(
                loc, valueTranspose_a.getType(), valueTranspose_a, int0, int1);
            
            // matmul
            Value matmul_output = rewriter.create<AtenMatmulOp>(
                loc, cat.getType(), cat, transposeInt_output);
            
            // add
            Value add_output = rewriter.create<AtenAddTensorOp>(
                loc, matmul_output.getType(), matmul_output, valueAdd, float1);
            
            // 将output的形状修改为和rst一致
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
                add_output, int0);

            Value int_start = 
                 rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

            Value int_end = 
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(shape[2]));

            Value int_dim = 
                rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(2));

            slice = rewriter.create<AtenSliceTensorOp>(
                loc, reluOp.getOperand().getType(), unsqueeze, int_dim, int_start, int_end, int1);

        }
    }

    // 利用残差进行相加
    Value add_rst = rewriter.create<AtenAddTensorOp>(
        loc, reluOp.getOperand().getType(), rst, slice, float0);
   
    //relu
    rewriter.replaceOpWithNewOp<AtenReluOp>(
        reluOp, reluOp.getType(), add_rst);
}

namespace {
class InsertRNNWithZerosPass : public InsertRNNWithZerosBase<InsertRNNWithZerosPass> {
public:
    InsertRNNWithZerosPass() = default;
    InsertRNNWithZerosPass(std::string activationFunc, int number){
        this->activationFunc = activationFunc;
        this->number = number;
    }
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        auto f = getOperation();
        insertRNNWithZeros(context, f, activationFunc, number);
    }
};
} //namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertRNNWithZerosPass(std::string activationFunc, int number) {
  return std::make_unique<InsertRNNWithZerosPass>(activationFunc, number);
}

