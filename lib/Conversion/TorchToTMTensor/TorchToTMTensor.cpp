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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include <iostream>

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

static Value createTMTensorScatterOp(
    OpBuilder &b, Location loc, Value updates, Value indices, Value original,
    bool uniqueIndices,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuild) {
  auto originalTensorType = original.getType().cast<RankedTensorType>();
  Type originalElementType = originalTensorType.getElementType();
  auto scatterOp = b.create<TMTensor::ScatterOp>(
      loc, originalTensorType, ValueRange{updates, indices},
      ValueRange{original}, uniqueIndices);

  Region &scatterOpRegion = scatterOp.region();
  auto &scatterOpBlock = scatterOpRegion.emplaceBlock();
  scatterOpBlock.addArguments({originalElementType, originalElementType},
                              {loc, loc});
  OpBuilder regionBuilder(scatterOpRegion);
  auto blockArgs = scatterOpBlock.getArguments();
  Value updatesElement = blockArgs[0];
  Value originalElement = blockArgs[1];
  bodyBuild(regionBuilder, loc, updatesElement, originalElement);
  return scatterOp->getResult(0);
}

static Value createTMTensorScanOp(
    OpBuilder &b, Location loc, Value input, Value output, Value accumulator,
    int64_t dim, bool inclusive,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuild) {
  auto inputType = input.getType().cast<RankedTensorType>();
  auto accType = accumulator.getType().cast<RankedTensorType>();
  Type elementType = inputType.getElementType();
  auto scanOp = b.create<TMTensor::ScanOp>(
      loc, TypeRange{inputType, accType}, input,
      ValueRange{output, accumulator}, b.getI64IntegerAttr(dim),
      b.getBoolAttr(inclusive));

  Region &scanOpRegion = scanOp.region();
  auto &scanOpBlock = scanOpRegion.emplaceBlock();
  scanOpBlock.addArguments({elementType, elementType}, {loc, loc});
  OpBuilder regionBuilder(scanOpRegion);
  auto blockArgs = scanOpBlock.getArguments();
  Value inputElement = blockArgs[0];
  Value accElement = blockArgs[1];
  bodyBuild(regionBuilder, loc, inputElement, accElement);
  return scanOp->getResult(0);
}

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
          op, "Unimplemented: the weights operand is not incorporated.");

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

    auto resultType = typeConverter->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    Type resultElemType = resultType.getElementType();

    SmallVector<Value, 1> inputSizeDynamic =
        getTensorSizesUntilDim(rewriter, loc, input, 0);
    Value updatesTensor = rewriter.create<tensor::EmptyOp>(
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

    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, updatesTensor, indices, bincountTensor,
        /*uniqueIndices=*/false,
        [&](OpBuilder &b, Location loc, Value _, Value bincountElem) {
          Value add = b.create<arith::AddIOp>(loc, bincountElem, constantOne);
          b.create<TMTensor::YieldOp>(loc, add);
        });
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, scatterOp);
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
    auto resultType = typeConverter->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();

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

    // TODO: Add support for cases with indices list size not equal to 1.
    if (indicesList.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: Indices list size != 1");
    Value indexTensor = indicesList[0];

    if (indexTensor.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op, "Index tensor must not be None.");

    // Creating a tm_tensor.scatter op with the following mapping:
    // 1.) Index tensor from the `indicesList` maps to the indices in scatter
    // op. Index tensor is expanded from 1-d to 2-d, and its element type is set
    // to i32 as required for the scatter op.
    // 2.) `values` is mapped to `updates` in scatter op.
    // 3.) `input` is mapped to `original` in scatter op.
    if (getTensorRank(indexTensor) != 1)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: index tensor with rank != 1 is not supported");
    auto indexTensorType = indexTensor.getType().cast<BaseTensorType>();
    int64_t indexTensorSize = indexTensorType.getSizes()[0];
    SmallVector<int64_t> expandedIndexTensorSizes{indexTensorSize, 1};
    ValueTensorType expandedIndexTensorType = ValueTensorType::get(
        context, llvm::makeArrayRef(expandedIndexTensorSizes),
        indexTensorType.getDtype());
    Value torchCstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value expandedIndexTensor = rewriter.create<AtenUnsqueezeOp>(
        loc, expandedIndexTensorType, indexTensor, torchCstOne);

    // `TMTensor::ScatterOp` expects indices of element type i32.
    Value indices = convertTensorToDtype(
        rewriter, loc, expandedIndexTensor,
        mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed));
    indices = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(indices.getType()), indices);

    bool invalidInputTypeFound = false;
    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, values, indices, input, /*uniqueIndices=*/false,
        [&](OpBuilder &b, Location loc, Value valuesElement,
            Value inputElement) {
          Value yieldValue = valuesElement;
          if (accumulate) {
            if (inputElement.getType().isa<mlir::IntegerType>()) {
              yieldValue =
                  b.create<arith::AddIOp>(loc, inputElement, valuesElement);
            } else if (inputElement.getType().isa<mlir::FloatType>()) {
              yieldValue =
                  b.create<arith::AddFOp>(loc, inputElement, valuesElement);
            } else {
              invalidInputTypeFound = true;
              return;
            }
          }
          b.create<TMTensor::YieldOp>(loc, yieldValue);
        });

    if (invalidInputTypeFound) {
      return rewriter.notifyMatchFailure(
          op,
          "unimplemented: input tensor must be of integer type or float type");
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, scatterOp);
    return success();
  }
};
} // namespace

namespace {
// The original implementation of the op is as follows:
//
// Indices and GradOutput Layout: [N, C, H, W] or [C, H, W]
// Input Layout: [N, C, Hin, Win] or [C, Hin, Win]
//
// for i in range(N):
//   for j in range(C):
//       for k in range(H):
//           for l in range(W):
//               index = indices[i, j, k, l]
//               result[i, j, index/Win, index%Win] += gradOutput[i, j, k, l]
//
//                    OR
//
// for i in range(C):
//   for j in range(H):
//       for k in range(W):
//           index = indices[i, j, k]
//           result[i, index/Win, index%Win] += gradOutput[i, j, k]
//
class ConvertAtenMaxPool2dWithIndicesBackwardOp
    : public OpConversionPattern<AtenMaxPool2dWithIndicesBackwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dWithIndicesBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();
    Value gradOutput = adaptor.grad_output();
    Value input = adaptor.self();
    RankedTensorType gradOutputType =
        gradOutput.getType().cast<RankedTensorType>();
    Type gradOutputElemType = gradOutputType.getElementType();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type inputElemType = inputType.getElementType();
    int64_t tensorOperandRank = inputType.getRank();

    // `TMTensor::ScatterOp` expects indices of element type i32.
    Value indices = convertTensorToDtype(
        rewriter, loc, op.indices(),
        mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed));
    indices = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(indices.getType()), indices);
    RankedTensorType indicesType = indices.getType().cast<RankedTensorType>();
    Type indicesElemType = indicesType.getElementType();

    // The element type of the `input` and `grad_output` should be same.
    if (inputElemType != gradOutputElemType)
      return rewriter.notifyMatchFailure(
          op,
          "Input element type should be same as the grad_output element type.");

    // Since the scatter op requires indices to be a 2-d tensor, we create a new
    // 5-d/4-d tensor (depending on the original indices layout) comprising the
    // index values. We will collapse this tensor into a 2-d tensor. The
    // algorithm for the creation of updated indices tensor is as follows:
    //
    // for i in range(N):
    //   for j in range(C):
    //     for k in range(H):
    //       for l in range(W):
    //         for m in range(4):
    //           if m == 0:
    //             updatedIndices[N][C][H][W][0] = i
    //           if m == 1:
    //             updatedIndices[N][C][H][W][1] = j
    //           if m == 2:
    //             updatedIndices[N][C][H][W][2] =
    //                                      originalIndices[i, j, k, l] / Win
    //           if m == 3:
    //             updatedIndices[N][C][H][W][3] =
    //                                      originalIndices[i, j, k, l] % Win
    //
    //                  OR
    //
    //  for j in range(C):
    //   for k in range(H):
    //     for l in range(W):
    //       for m in range(3):
    //         if m == 0:
    //           updatedIndices[C][H][W][0] = i
    //         if m == 1:
    //          updatedIndices[C][H][W][1] = originalIndices[i, j, k, l] / Win
    //         if m == 2:
    //           updatedIndices[C][H][W][2] = originalIndices[i, j, k, l] % Win

    SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);

    SmallVector<AffineExpr> originalIndicesDimExprs, updatedIndicesDimExprs;
    for (int64_t i = 0; i < tensorOperandRank; i++) {
      originalIndicesDimExprs.push_back(rewriter.getAffineDimExpr(i));
      updatedIndicesDimExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    updatedIndicesDimExprs.push_back(
        rewriter.getAffineDimExpr(tensorOperandRank));

    SmallVector<AffineMap> indexingMaps = AffineMap::inferFromExprList(
        {originalIndicesDimExprs, updatedIndicesDimExprs});
    SmallVector<StringRef> iteratorTypes(tensorOperandRank + 1, "parallel");

    SmallVector<OpFoldResult> updatedIndicesShape =
        getAsOpFoldResult(getTensorSizes(rewriter, loc, indices));
    updatedIndicesShape.push_back(rewriter.getIndexAttr(tensorOperandRank));

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, updatedIndicesShape, indicesElemType);

    Value wIn = inputShape[tensorOperandRank - 1];
    SmallVector<Value> cstValues;
    for (int64_t i = 0; i < tensorOperandRank; i++)
      cstValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));

    Value updatedIndices =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), indices, initTensor, indexingMaps,
                iteratorTypes,
                [tensorOperandRank, wIn, cstValues,
                 indicesElemType](OpBuilder &b, Location loc, ValueRange args) {
                  Value index = castIntToIndex(b, loc, args[0]);
                  Value updatedIndex = cstValues[0];
                  Value lastDim =
                      b.create<linalg::IndexOp>(loc, tensorOperandRank);

                  for (int64_t i = tensorOperandRank - 1; i >= 0; i--) {
                    Value result;
                    if (i == tensorOperandRank - 1)
                      result = b.create<arith::RemSIOp>(loc, index, wIn);
                    if (i == tensorOperandRank - 2)
                      result = b.create<arith::FloorDivSIOp>(loc, index, wIn);
                    if (i == tensorOperandRank - 3 ||
                        i == tensorOperandRank - 4)
                      result = b.create<linalg::IndexOp>(loc, i);

                    Value pred = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::eq, lastDim, cstValues[i]);
                    Value addAmount = b.create<arith::SelectOp>(
                        loc, pred, result, cstValues[0]);
                    updatedIndex =
                        b.create<arith::AddIOp>(loc, updatedIndex, addAmount);
                  }

                  updatedIndex = b.create<arith::IndexCastOp>(
                      loc, indicesElemType, updatedIndex);
                  b.create<linalg::YieldOp>(loc, updatedIndex);
                })
            .getResult(0);

    // Creating a new tensor initialized with zeros and size same as the input
    // tensor.
    Value outputTensor =
        createZeroInitTensor(rewriter, loc, inputShape, inputElemType);

    // Collapsing `gradOutput` into a 1-d tensor.
    SmallVector<ReassociationIndices> reassociationCollapse(1);
    for (auto i = 0; i < gradOutputType.getRank(); i++)
      reassociationCollapse[0].push_back(i);
    RankedTensorType gradOutputFlattenedType;
    int64_t numelGradOutput = getNumberOfElements(gradOutputType);
    gradOutputFlattenedType =
        RankedTensorType::get({numelGradOutput}, gradOutputElemType);
    Value gradOutputFlattened = rewriter.create<tensor::CollapseShapeOp>(
        loc, gradOutputFlattenedType, gradOutput, reassociationCollapse);

    // Collapsing updated indices into a 2-d tensor.
    SmallVector<ReassociationIndices> reassociationCollapseIndices(2);
    for (auto i = 0; i < tensorOperandRank; i++)
      reassociationCollapseIndices[0].push_back(i);
    reassociationCollapseIndices[1].push_back(tensorOperandRank);
    int64_t numelIndices = getNumberOfElements(indicesType);
    Value indicesCollapsed = rewriter.create<tensor::CollapseShapeOp>(
        loc,
        RankedTensorType::get({numelIndices, tensorOperandRank},
                              indicesElemType),
        updatedIndices, reassociationCollapseIndices);

    bool invalidInputTypeFound = false;
    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, /*updates=*/gradOutputFlattened,
        /*indices=*/indicesCollapsed, /*original=*/outputTensor,
        /*uniqueIndices=*/false,
        [&](OpBuilder &b, Location loc, Value valuesElement,
            Value inputElement) {
          Value yieldValue = valuesElement;
          if (inputElement.getType().isa<mlir::IntegerType>()) {
            yieldValue =
                b.create<arith::AddIOp>(loc, inputElement, valuesElement);
          } else if (inputElement.getType().isa<mlir::FloatType>()) {
            yieldValue =
                b.create<arith::AddFOp>(loc, inputElement, valuesElement);
          } else {
            invalidInputTypeFound = true;
            return;
          }
          b.create<TMTensor::YieldOp>(loc, yieldValue);
        });

    if (invalidInputTypeFound) {
      return rewriter.notifyMatchFailure(
          op,
          "unimplemented: input tensor must be of integer type or float type");
    }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, scatterOp);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenCumsumOp : public OpConversionPattern<AtenCumsumOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenCumsumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.self();
    auto resultType = input.getType().cast<RankedTensorType>();
    Type elementType = resultType.getElementType();
    int64_t inputRank = resultType.getRank();
    Location loc = op->getLoc();
    Value dtype = op.dtype();
    if (!dtype.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "unsupported: dtype argument not supported");

    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant dim value is supported");
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      return rewriter.notifyMatchFailure(op, "invalid dim");

    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, input);
    Value output = createZeroInitTensor(rewriter, loc, sizes, elementType);
    output = rewriter.create<tensor::CastOp>(loc, resultType, output);

    SmallVector<Value> accSizes(sizes);
    accSizes.erase(accSizes.begin() + dim);
    SmallVector<int64_t> accStatic(resultType.getShape());
    accStatic.erase(accStatic.begin() + dim);
    Value acc = createZeroInitTensor(rewriter, loc, accSizes, elementType);
    Type accType = RankedTensorType::get(accStatic, elementType);
    acc = rewriter.create<tensor::CastOp>(loc, accType, acc);

    Value result = createTMTensorScanOp(
        rewriter, loc, input, output, acc, dim, /*inclusive=*/true,
        [](OpBuilder &b, Location loc, Value input, Value acc) {
          Value sum = (input.getType().isa<mlir::FloatType>()
                           ? b.create<arith::AddFOp>(loc, input, acc)
                           : b.create<arith::AddIOp>(loc, input, acc))
                          ->getResult(0);
          b.create<TMTensor::YieldOp>(loc, sum);
        });

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
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
    registry.insert<arith::ArithDialect>();
    registry.insert<TMTensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           tensor::TensorDialect, arith::ArithDialect,
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
    target.addIllegalOp<AtenCumsumOp>();
    patterns.add<ConvertAtenCumsumOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTMTensorPass() {
  return std::make_unique<ConvertTorchToTMTensor>();
}
