//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/MLIRContext.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;
using namespace mlir::torch::TMTensor;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#aten ops), which is in the 100s.
//
// Most of these patterns consist of:
// 1. Checking that the operand/result types and other static properties are
//    good-enough to create a valid linalg op (such as operands being of
//    ranks/dtypes acceptable to the linalg op).
// 2. Creating dynamic error guards, usually checking a predicate on the
//    compatibility of operand shapes.
// 3. Creating init tensors for the computation op. Usually this involves
//    reifying IR for a shape transfer function based on the operand shapes.
// 4. Creating a named linalg op to replace the original op.
//
// TODO: Use linalg OpDSL to autogenerate at least 1)/2)/3) such
// that these patterns become mostly mechanical associations of
// "aten.foo -> linalg.foo".

namespace {
// aten::bincount op counts the frequency of each value in a 1-d input tensor of
// non-negative ints.
class ConvertAtenBincountOp : public OpConversionPattern<AtenBincountOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBincountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();
    TypeConverter *typeConverter = getTypeConverter();
    Value input = adaptor.self();
    Value torchTypeInput = op.self();
    Value minlength = adaptor.minlength();
    Value weights = adaptor.weights();

    // TODO: Add a check to verify that the input tensor elements are all
    // non-negative.
    // Check whether the input is a 1-d tensor of integer type or not.
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    if (inputType.getRank() != 1 ||
        !inputType.getElementType().isa<mlir::IntegerType>())
      return rewriter.notifyMatchFailure(
          op,
          "Input tensor has to be a one-dimensional tensor of integer type.");

    // Check whether the input tensor element type is i64 or not.
    IntegerType inputIntegerType =
        inputType.getElementType().cast<IntegerType>();
    if (inputIntegerType.getWidth() != 64)
      return rewriter.notifyMatchFailure(
          op,
          "Unimplemented: Integer width not equal to 64 are not supported.");

    // TODO: Incorporate the weight argument.
    if (!weights.getType().isa<mlir::torch::Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weights operand is not incorporated.");

    // Finding the maximum value in the input tensor.
    SmallVector<int64_t> maxTensorSizes;
    ValueTensorType maxTensorType = ValueTensorType::get(
        context, llvm::makeArrayRef(maxTensorSizes),
        torchTypeInput.getType().cast<ValueTensorType>().getDtype());
    Value maxTensor =
        rewriter.create<AtenMaxOp>(loc, maxTensorType, torchTypeInput);
    maxTensor = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(maxTensor.getType()),
        maxTensor);

    // `maxTensor` is a 0-d tensor, extracting its only element and
    // storing it in `maxInput`.
    Value maxInput = rewriter.create<tensor::ExtractOp>(loc, maxTensor);

    // Creating a tm_tensor.scatter op with the following mapping:
    // 1.) `input` tensor maps to the indices in scatter op. `input` is
    // expanded from 1-d to 2-d, and its element type is set to i32 as required
    // for the scatter op.
    // 2.) `updates` is a 1-d dummy tensor with the size equivalent to the
    // `input`.
    // 3.) `bincount` a 1-d tensor maps to the original in scatter op
    // with size equal to the max(max(input) + 1, minlength).
    SmallVector<int64_t> expandedInputSizes{inputType.getShape()[0], 1};
    ValueTensorType expandInputType = ValueTensorType::get(
        context, llvm::makeArrayRef(expandedInputSizes),
        torchTypeInput.getType().cast<ValueTensorType>().getDtype());
    Value torchCstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value expandedInputTensor = rewriter.create<AtenUnsqueezeOp>(
        loc, expandInputType, torchTypeInput, torchCstOne);

    // Converting the input element type to i32.
    Value indices = convertTensorToDtype(
        rewriter, loc, expandedInputTensor,
        mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed));
    indices = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(indices.getType()), indices);

    Type resultElemType = typeConverter->convertType(op->getResult(0).getType())
                              .cast<RankedTensorType>()
                              .getElementType();

    SmallVector<Value, 1> inputSizeDynamic =
        getTensorSizesUntilDim(rewriter, loc, input, 0);
    Value updatesTensor = rewriter.create<linalg::InitTensorOp>(
        loc, getAsOpFoldResult(inputSizeDynamic), resultElemType);

    Value constantZero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultElemType));
    Value constantOne = rewriter.create<arith::ConstantIntOp>(
        loc, 1, resultElemType.getIntOrFloatBitWidth());

    // Bincount size = max(max(input) + 1, minlength)
    Value maxInputPlusOne =
        rewriter.create<arith::AddIOp>(loc, maxInput, constantOne);
    Value bincountSize =
        rewriter.create<arith::MaxSIOp>(loc, maxInputPlusOne, minlength);
    bincountSize = castIntToIndex(rewriter, loc, bincountSize);
    Value bincountTensor = createInitTensor(rewriter, loc, {bincountSize},
                                            resultElemType, constantZero);

    auto scatterOp = rewriter.create<TMTensor::ScatterOp>(
        loc, bincountTensor.getType(), ValueRange{updatesTensor, indices},
        ValueRange{bincountTensor},
        /*unique_indices=*/false);

    Region &scatterOpRegion = scatterOp.region();
    auto &scatterOpBlock = scatterOpRegion.emplaceBlock();
    scatterOpBlock.addArguments(TypeRange{resultElemType, resultElemType},
                                {loc, loc});
    auto blockArgs = scatterOpBlock.getArguments();

    // Creating an add instruction inside the scatter op region to increment the
    // frequency counter with one.
    OpBuilder regionBuilder(scatterOpRegion);
    Value add = regionBuilder.create<arith::AddIOp>(loc,
                                                    /*bincount=*/blockArgs[1],
                                                    constantOne);
    regionBuilder.create<TMTensor::YieldOp>(loc, add);
    rewriter.replaceOp(op, scatterOp->getResult(0));
    return success();
  }
};
} // namespace

namespace {
class ConvertValsemVariantAtenIndexPutImplOp
    : public OpConversionPattern<ValsemVariantAtenIndexPutImplOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ValsemVariantAtenIndexPutImplOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.self();
    Value values = adaptor.values();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType valuesType = values.getType().cast<RankedTensorType>();
    Type resultElemType = typeConverter->convertType(op->getResult(0).getType())
                              .cast<RankedTensorType>()
                              .getElementType();

    // TODO: Add support for the input with rank other than one.
    if (inputType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: input rank other than one is not supported");

    // The unsafe should be either `False` or `none`.
    if (!op.unsafe().getType().isa<Torch::NoneType>()) {
      bool unsafe;
      if (!matchPattern(op.unsafe(), m_TorchConstantBool(&unsafe)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: unsafe must be a constant");
      else if (unsafe)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: unsafe is expected to be false");
    }

    // The accumulate should be a torch constant of boolean type.
    bool accumulate;
    if (!matchPattern(op.accumulate(), m_TorchConstantBool(&accumulate)))
      return rewriter.notifyMatchFailure(
          op, "Expected accumulate to be constant bool.");

    // The element type of the `input` and `values` should be same.
    if (inputType.getElementType() != valuesType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "Input element type should be same as the values element type.");

    SmallVector<Value> indicesList;
    getListConstructElements(adaptor.indices(), indicesList);
    // The size of the list of the index tensors should not be greater than the
    // input rank.
    if ((int64_t)indicesList.size() > inputType.getRank())
      return rewriter.notifyMatchFailure(
          op, "Indices list size should not be greater than the input rank.");

    // TODO: Add support for cases with indices list size smaller than the input
    // rank.
    if ((int64_t)indicesList.size() < inputType.getRank())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, Indices list size smaller than input rank");

    if (indicesList[0].getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op,
                                         "Indices tensor must not be none.");

    // TODO: Add support for the index with rank other than one.
    int64_t indexRank = typeConverter->convertType(indicesList[0].getType())
                            .cast<RankedTensorType>()
                            .getRank();
    if (indexRank != 1)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: index rank other than one is not supported");

    // Creating a tm_tensor.scatter op with the following mapping:
    // 1.) Index tensor from the `indicesList` maps to the indices in scatter
    // op. Index tensor is expanded from 1-d to 2-d, and its element type is set
    // to i32 as required for the scatter op.
    // 2.) `values` is mapped to `updates` in scatter op.
    // 3.) `input` is mapped to `original` in scatter op.
    ValueTensorType indexType =
        indicesList[0].getType().cast<ValueTensorType>();
    SmallVector<int64_t> expandedIndexSizes{indexType.getSizes()[0], 1};
    ValueTensorType expandedIndexType = ValueTensorType::get(
        context, llvm::makeArrayRef(expandedIndexSizes), indexType.getDtype());
    Value torchCstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value expandedIndexTensor = rewriter.create<AtenUnsqueezeOp>(
        loc, expandedIndexType, indicesList[0], torchCstOne);

    // Converting the index element type to i32.
    Value indices = convertTensorToDtype(
        rewriter, loc, expandedIndexTensor,
        mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed));
    indices = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(indices.getType()), indices);

    auto scatterOp = rewriter.create<TMTensor::ScatterOp>(
        loc, input.getType(), ValueRange{values, indices}, ValueRange{input},
        /*unique_indices=*/false);

    Region &scatterOpRegion = scatterOp.region();
    auto &scatterOpBlock = scatterOpRegion.emplaceBlock();
    scatterOpBlock.addArguments(TypeRange{resultElemType, resultElemType},
                                {loc, loc});
    auto blockArgs = scatterOpBlock.getArguments();

    OpBuilder regionBuilder(scatterOpRegion);
    Value update = blockArgs[0];
    Value original = blockArgs[1];
    Value yieldValue = update;
    // Create an add instruction inside the scatter op region to increment the
    // `original` value with the value from `updates` if the accumulate flag is
    // true.
    if (accumulate) {
      if (inputType.getElementType().isa<mlir::IntegerType>())
        yieldValue = regionBuilder.create<arith::AddIOp>(loc, original, update);
      else if (inputType.getElementType().isa<mlir::FloatType>())
        yieldValue = regionBuilder.create<arith::AddFOp>(loc, original, update);
    }
    regionBuilder.create<TMTensor::YieldOp>(loc, yieldValue);
    rewriter.replaceOp(op, scatterOp->getResult(0));
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenMaxPool2dWithIndicesBackwardOp : public OpConversionPattern<AtenMaxPool2dWithIndicesBackwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dWithIndicesBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify types
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    // For debugging

    op->getParentOfType<ModuleOp>().dump();

    Value grad_output = adaptor.grad_output();
    Value input = adaptor.self();
    MLIRContext *context = op->getContext();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType gradType = grad_output.getType().cast<RankedTensorType>();
    SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
    Type inputEType = inputType.getElementType();

    // Get values
    SmallVector<Value> kernelSize;
    if (!getListConstructElements(op.kernel_size(), kernelSize))
      return rewriter.notifyMatchFailure(op, "kernel size not constructed from ListConstruct");
    SmallVector<Value> stride;
    if (!getListConstructElements(op.stride(), stride))
      return rewriter.notifyMatchFailure(op, "stride not constructed from ListConstruct");
    SmallVector<Value> padding;
    if (!getListConstructElements(op.padding(), padding))
      return rewriter.notifyMatchFailure(op, "padding not constructed from ListConstruct");
    SmallVector<Value> dilation;
    if (!getListConstructElements(op.dilation(), dilation))
      return rewriter.notifyMatchFailure(op, "dilation not constructed from ListConstruct");

    // The element type of the `self` and `values` should be same.
    if (inputType.getElementType() != gradType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "Input element type should be same as the grad_output element type.");

    // Add asserts/compile-time checks for values
    // TODO

    // Create empty output tensor
    // Value output = rewriter.create<linalg::InitTensorOp>(loc, selfShape, selfEType);

    // Add in indices
    Attribute attrs[2];
    if (inputEType.isa<mlir::FloatType>()) {
      attrs[0] = rewriter.getFloatAttr(inputEType, 0);
      attrs[1] = rewriter.getFloatAttr(inputEType, 1);
    } else if (inputEType.isa<mlir::IntegerType>()) {
      attrs[0] = rewriter.getIntegerAttr(inputEType, 0);
      attrs[1] = rewriter.getIntegerAttr(inputEType, 1);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported dtype");
    }

    Value output = createZeroInitTensor(rewriter, loc, inputShape, inputEType);
    //Value numTensorElements = rewriter.create<AtenNumelOp>(loc, input);
    //rewriter.create<AtenZerosOp>(loc, inputType, numTensorElements, inputEType);

    //Value indicesList;
    //if (!getListConstructElements(adaptor.indices(), indicesList))
     // return rewriter.notifyMatchFailure(op, "indices not constructed from ListConstruct");
    

    ValueTensorType indexType =
        adaptor.indices().getType().cast<ValueTensorType>();

    Value expandedIndexSizes{indexType.getSizes(), 1};
    ValueTensorType expandedIndexType = ValueTensorType::get(
        context, llvm::makeArrayRef(expandedIndexSizes), indexType.getDtype());
    Value torchCstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value expandedIndexTensor = rewriter.create<AtenUnsqueezeOp>(
        loc, expandedIndexType, adaptor.indices(), torchCstOne);

    // Converting the index element type to i32.
    Value indices = convertTensorToDtype(
        rewriter, loc, expandedIndexTensor,
        mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed));
    indices = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(indices.getType()), indices);

    // Scatter
    auto scatterOp = rewriter.create<TMTensor::ScatterOp>(
        loc, input.getType(), ValueRange{grad_output, indices}, ValueRange{output},
        /*unique_indices=*/false);
    indices.dump();

    // For debugging
    op->getParentOfType<ModuleOp>().dump();

    rewriter.replaceOp(op, scatterOp->getResult(0));
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTMTensor
    : public ConvertTorchToTMTensorBase<ConvertTorchToTMTensor> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<TMTensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           tensor::TensorDialect, arith::ArithmeticDialect,
                           Torch::TorchDialect, TMTensorDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenBincountOp>();
    patterns.add<ConvertAtenBincountOp>(typeConverter, context);
    target.addIllegalOp<ValsemVariantAtenIndexPutImplOp>();
    patterns.add<ConvertValsemVariantAtenIndexPutImplOp>(typeConverter,
                                                         context);
    target.addIllegalOp<AtenMaxPool2dWithIndicesBackwardOp>();
    patterns.add<ConvertAtenMaxPool2dWithIndicesBackwardOp>(typeConverter,
                                                         context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertTorchToTMTensorPass() {
  return std::make_unique<ConvertTorchToTMTensor>();
}
