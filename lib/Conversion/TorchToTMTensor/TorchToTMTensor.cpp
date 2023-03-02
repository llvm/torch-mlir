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
#include "mlir/IR/ValueRange.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"

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

static Attribute getNumericLimit(PatternRewriter &rewriter, Type elementType,
                                 bool getMin = true) {
  auto bitWidth = elementType.getIntOrFloatBitWidth();
  if (llvm::isa<mlir::IntegerType>(elementType)) {
    if (getMin) {
      return rewriter.getIntegerAttr(elementType,
                                     APInt::getSignedMinValue(bitWidth));
    } else {
      return rewriter.getIntegerAttr(elementType,
                                     APInt::getSignedMaxValue(bitWidth));
    }
  } else if (mlir::FloatType floatType =
                 llvm::dyn_cast<mlir::FloatType>(elementType)) {
    return rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(floatType.getFloatSemantics(), getMin));
  } else {
    llvm_unreachable("Only float/integer types are supported!");
  }
}

// This function will reformat the `index` and `src` from torch operations
// like `torch.scatter` or `torch.scatter_reduce` to match the expected
// input for the TMScatterOp. It will return the reformated `index` and `src`
// as a pair of mlir::Value that can be used as inputs for the TMScatterOp.
static std::pair<Value, Value>
convertTorchScatterIndexAndSrcToTMScatterIndexAndSrc(PatternRewriter &rewriter,
                                                     Value indices, Value src,
                                                     int64_t dim) {
  // Get information on types for inputs
  RankedTensorType indexType = indices.getType().cast<RankedTensorType>();
  RankedTensorType srcSelf = src.getType().cast<RankedTensorType>();

  // Store location for insertions
  Location loc = src.getLoc();

  Value indexSize = getTensorSize(rewriter, loc, indices);
  indexSize = castIntToIndex(rewriter, loc, indexSize);
  SmallVector<Value> indexShape = getTensorSizes(rewriter, loc, indices);
  Value cstOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // We flatten the `src` values from (i, j, k, ...) -> (i * j * k * ...)
  SmallVector<Value> indSliceShape({indexSize, cstOne});
  Value indSlice =
      createZeroInitTensor(rewriter, loc, indSliceShape, rewriter.getI32Type());

  // New output shape will be equal to the product of the dimensions of the
  // updates
  SmallVector<Value> outputs(indexType.getRank(), indSlice);
  outputs.push_back(createZeroInitTensor(rewriter, loc, {indexSize},
                                         srcSelf.getElementType()));
  SmallVector<Type> outputsType(indexType.getRank(), indSlice.getType());
  outputsType.push_back(outputs[indexType.getRank()].getType());

  // Create mapping over flattened iteration space
  SmallVector<AffineExpr> indSliceExpr = {rewriter.getAffineDimExpr(0),
                                          rewriter.getAffineConstantExpr(0)};
  SmallVector<AffineMap> mapping(
      indexType.getRank(), AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                          indSliceExpr, src.getContext()));
  // Mapping for updates
  mapping.push_back(rewriter.getDimIdentityMap());
  SmallVector<utils::IteratorType> iteratorTypes(
      {utils::IteratorType::parallel});

  // This function goes over the flattened iteration space of the `indices`
  // and `src`. It will reconstruct the original induction variables based
  // on the current flattened index. The flattened iteration space is required
  // because TMTensorScatterOp expects a list of single element updates.
  auto flattenedUpdates =
      rewriter
          .create<linalg::GenericOp>(
              loc, outputsType, ValueRange(), outputs, mapping, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                SmallVector<Value> indexValues(indexType.getRank());
                Value ind = b.create<linalg::IndexOp>(loc, 0);
                for (int i = indexType.getRank() - 1; i >= 0; i--) {
                  indexValues[i] =
                      b.create<arith::RemSIOp>(loc, ind, indexShape[i]);
                  ind = b.create<arith::DivSIOp>(loc, ind, indexShape[i]);
                }
                // Extract the scatter index and update value
                Value extractIndexValue =
                    b.create<tensor::ExtractOp>(loc, indices, indexValues);
                Value extractSrcValue =
                    b.create<tensor::ExtractOp>(loc, src, indexValues);
                SmallVector<Value> yieldVals;
                for (Value v : indexValues) {
                  Value scalar = castIndexToInt64(b, loc, v);
                  yieldVals.push_back(b.create<arith::TruncIOp>(
                      loc, rewriter.getI32Type(), scalar));
                }
                // Replace the original index with the index specified
                // by the scatter.
                yieldVals[dim] = b.create<arith::TruncIOp>(
                    loc, rewriter.getI32Type(), extractIndexValue);
                yieldVals.push_back(extractSrcValue);
                b.create<linalg::YieldOp>(loc, yieldVals);
              })
          .getResultTensors();

  auto toOpFoldResult = [](Value v) -> OpFoldResult {
    auto op = v.getDefiningOp<arith::ConstantIndexOp>();
    if (!op)
      return v;
    return op.getValue();
  };

  // The result of the linalg::Generic operation gives us (rank(`src`) + 1)
  // 1D-tensors where each contains a number of elements equal to the total
  // number of elements in the `src` tensor. The indices must now be
  // constructed by concatanating the first rank(`src`) tensors together. The
  // new `src` tensor is the last tensor returned from the linalg::Generic
  // operation.
  SmallVector<Value> offsets = {
      rewriter.create<arith::ConstantIndexOp>(loc, 0),
      rewriter.create<arith::ConstantIndexOp>(loc, 0)};
  SmallVector<Value> strides = {
      rewriter.create<arith::ConstantIndexOp>(loc, 1),
      rewriter.create<arith::ConstantIndexOp>(loc, 1)};
  Value indicesRank =
      rewriter.create<arith::ConstantIndexOp>(loc, indexType.getRank());
  Value flattenedIndices = createZeroInitTensor(
      rewriter, loc, SmallVector<Value>({indexSize, indicesRank}),
      rewriter.getI32Type());
  SmallVector<Value> scatterInputsVector(flattenedUpdates);
  for (auto const slice : ArrayRef(scatterInputsVector).drop_back()) {
    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, slice);
    flattenedIndices = rewriter.createOrFold<tensor::InsertSliceOp>(
        loc, slice, flattenedIndices,
        llvm::to_vector(llvm::map_range(offsets, toOpFoldResult)),
        llvm::to_vector(llvm::map_range(sizes, toOpFoldResult)),
        llvm::to_vector(llvm::map_range(strides, toOpFoldResult)));
    // Increment offset to insert into next column
    offsets[1] = rewriter.createOrFold<arith::AddIOp>(loc, offsets[1], cstOne);
  }

  return std::make_pair(flattenedIndices,
                        scatterInputsVector[indexType.getRank()]);
}

static Value createTMTensorScatterOp(
    OpBuilder &b, Location loc, Value updates, Value indices, Value original,
    bool uniqueIndices,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuild) {
  auto originalTensorType = original.getType().cast<RankedTensorType>();
  Type originalElementType = originalTensorType.getElementType();
  auto scatterOp = b.create<TMTensor::ScatterOp>(
      loc, originalTensorType, ValueRange{updates, indices},
      ValueRange{original}, uniqueIndices);

  Region &scatterOpRegion = scatterOp.getRegion();
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

  Region &scanOpRegion = scanOp.getRegion();
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
    Value input = adaptor.getSelf();
    Value torchTypeInput = op.getSelf();
    Value minlength = adaptor.getMinlength();
    Value weights = adaptor.getWeights();

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
        context, llvm::ArrayRef(maxTensorSizes),
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
    SmallVector<int64_t> expandedInputSizes{
        makeShapeTorchCompatible(inputType.getShape())[0], 1};
    ValueTensorType expandInputType = ValueTensorType::get(
        context, llvm::ArrayRef(expandedInputSizes),
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
class ConvertAten_IndexPutImplOp
    : public OpConversionPattern<Aten_IndexPutImplOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Aten_IndexPutImplOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.getSelf();
    Value values = adaptor.getValues();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType valuesType = values.getType().cast<RankedTensorType>();
    auto resultType = typeConverter->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();

    // The unsafe should be either `False` or `none`.
    if (!op.getUnsafe().getType().isa<Torch::NoneType>()) {
      bool unsafe;
      if (!matchPattern(op.getUnsafe(), m_TorchConstantBool(&unsafe)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: unsafe must be a constant");
      else if (unsafe)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: unsafe is expected to be false");
    }

    // The accumulate should be a torch constant of boolean type.
    bool accumulate;
    if (!matchPattern(op.getAccumulate(), m_TorchConstantBool(&accumulate)))
      return rewriter.notifyMatchFailure(
          op, "Expected accumulate to be constant bool.");

    // The element type of the `input` and `values` should be same.
    if (inputType.getElementType() != valuesType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "Input element type should be same as the values element type.");

    SmallVector<Value> indicesList;
    getListConstructElements(adaptor.getIndices(), indicesList);
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
    std::optional<unsigned> indexTensorRank = getTensorRank(indexTensor);
    if (!indexTensorRank || *indexTensorRank != 1)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: index tensor with rank != 1 is not supported");
    auto indexTensorType = indexTensor.getType().cast<BaseTensorType>();
    int64_t indexTensorSize = indexTensorType.getSizes()[0];
    SmallVector<int64_t> expandedIndexTensorSizes{indexTensorSize, 1};
    ValueTensorType expandedIndexTensorType =
        ValueTensorType::get(context, llvm::ArrayRef(expandedIndexTensorSizes),
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
    Value gradOutput = adaptor.getGradOutput();
    Value input = adaptor.getSelf();
    RankedTensorType gradOutputType =
        gradOutput.getType().cast<RankedTensorType>();
    Type gradOutputElemType = gradOutputType.getElementType();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type inputElemType = inputType.getElementType();
    int64_t tensorOperandRank = inputType.getRank();

    // `TMTensor::ScatterOp` expects indices of element type i32.
    Value indices = convertTensorToDtype(
        rewriter, loc, op.getIndices(),
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
    SmallVector<utils::IteratorType> iteratorTypes(
        tensorOperandRank + 1, utils::IteratorType::parallel);

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
    gradOutputFlattenedType = RankedTensorType::get(
        makeShapeLLVMCompatible({numelGradOutput}), gradOutputElemType);
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
        RankedTensorType::get(
            makeShapeLLVMCompatible({numelIndices, tensorOperandRank}),
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
class ConvertAtenScatterReduceTwoOp
    : public OpConversionPattern<AtenScatterReduceTwoOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenScatterReduceTwoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();

    RankedTensorType selfType =
        adaptor.getSelf().getType().cast<RankedTensorType>();
    RankedTensorType indexType =
        adaptor.getIndex().getType().cast<RankedTensorType>();
    RankedTensorType srcType =
        adaptor.getSrc().getType().cast<RankedTensorType>();

    Value self = adaptor.getSelf();

    if (selfType.getRank() != indexType.getRank() ||
        indexType.getRank() != srcType.getRank())
      return rewriter.notifyMatchFailure(op,
                                         "'self', 'index' and 'src' should all "
                                         "have the same number of dimensions.");

    std::string reduceType;
    if (!matchPattern(op.getReduce(), m_TorchConstantStr(reduceType)))
      return rewriter.notifyMatchFailure(op,
                                         "'reduce' must be a costant string");

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "'dim' is not constant");

    bool includeSelf;
    if (!matchPattern(op.getIncludeSelf(), m_TorchConstantBool(&includeSelf)))
      return rewriter.notifyMatchFailure(op, "'include_self' is not constant");

    // Get reduce string as the equivalent enum
    auto reduceEnum = torch_upstream::get_reduction_enum(reduceType);

    // Get the inputs reformatted for the TMScatterOp
    auto [indices, updates] =
        convertTorchScatterIndexAndSrcToTMScatterIndexAndSrc(
            rewriter, adaptor.getIndex(), adaptor.getSrc(), dim);

    // Value 'counts' will be used to tally the number of reductions into
    // each unique index. The tally is used to calculate the average of the
    // values scattered per index.
    Value counts = nullptr;
    if (reduceEnum == torch_upstream::ReductionType::MEAN) {
      SmallVector<Value> selfShape =
          getTensorSizes(rewriter, loc, adaptor.getSelf());
      Attribute initAttr;
      if (llvm::isa<mlir::FloatType>(srcType.getElementType())) {
        initAttr = rewriter.getFloatAttr(srcType.getElementType(), 1);
      } else if (llvm::isa<mlir::IntegerType>(srcType.getElementType())) {
        initAttr = rewriter.getIntegerAttr(srcType.getElementType(), 1);
      } else {
        llvm_unreachable("Only integer/float types supported!");
      }
      Value initElement = rewriter.create<arith::ConstantOp>(loc, initAttr);
      counts = createInitTensor(rewriter, loc, selfShape,
                                selfType.getElementType(), initElement);
    }

    // If the original values shouldn't be included, normalize the
    // input tensor where the scatters take place.
    if (!includeSelf) {
      Value normalizationValue;
      if (reduceEnum == torch_upstream::ReductionType::SUM ||
          reduceEnum == torch_upstream::ReductionType::MEAN) {
        // Set the values in the input tensor to '0' so they are not included
        normalizationValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(srcType.getElementType()));
      } else if (reduceEnum == torch_upstream::ReductionType::PROD) {
        // Set the values in the input tensor to '1' (multiplication identity)
        if (llvm::isa<mlir::FloatType>(srcType.getElementType())) {
          normalizationValue = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getFloatAttr(srcType.getElementType(), 1.0));
        } else if (llvm::isa<mlir::IntegerType>(srcType.getElementType())) {
          normalizationValue = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIntegerAttr(srcType.getElementType(), 1));
        } else {
          llvm_unreachable("Only integer/float types supported!");
        }
      } else if (reduceEnum == torch_upstream::ReductionType::MAX) {
        // Set the values in the input tensor to the smallest element of that
        // type
        auto minAttr = getNumericLimit(rewriter, srcType.getElementType(),
                                       /*getMin=*/true);
        normalizationValue = rewriter.create<arith::ConstantOp>(loc, minAttr);
      } else if (reduceEnum == torch_upstream::ReductionType::MIN) {
        // Set the values in the input tensor to the largest element of that
        // type
        auto maxAttr = getNumericLimit(rewriter, srcType.getElementType(),
                                       /*getMin=*/false);
        normalizationValue = rewriter.create<arith::ConstantOp>(loc, maxAttr);
      }

      // Scatter the normalizations into the input tensor
      Value indexSize = getTensorSize(rewriter, loc, adaptor.getIndex());
      indexSize = castIntToIndex(rewriter, loc, indexSize);
      Value normalizations = createInitTensor(
          rewriter, loc, SmallVector<Value>({indexSize}),
          srcType.getElementType(), /*init_element=*/normalizationValue);
      self = createTMTensorScatterOp(
          rewriter, loc, normalizations, indices, self,
          /*uniqueIndices=*/false,
          [&](OpBuilder &b, Location loc, Value update, Value current) {
            b.create<TMTensor::YieldOp>(loc, update);
          });
      if (reduceEnum == torch_upstream::ReductionType::MEAN) {
        counts = createTMTensorScatterOp(
            rewriter, loc, normalizations, indices, counts,
            /*uniqueIndices=*/false,
            [&](OpBuilder &b, Location loc, Value update, Value current) {
              b.create<TMTensor::YieldOp>(loc, update);
            });
      }
    }

    // Create final operation
    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, updates, indices, self,
        /*uniqueIndices=*/false,
        [&](OpBuilder &b, Location loc, Value update, Value current) {
          Value result;
          if (reduceEnum == torch_upstream::ReductionType::SUM ||
              reduceEnum == torch_upstream::ReductionType::MEAN) {
            if (update.getType().isa<mlir::IntegerType>()) {
              result = b.create<arith::AddIOp>(loc, update, current);
            } else if (update.getType().isa<mlir::FloatType>()) {
              result = b.create<arith::AddFOp>(loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          } else if (reduceEnum == torch_upstream::ReductionType::PROD) {
            if (update.getType().isa<mlir::IntegerType>()) {
              result = b.create<arith::MulIOp>(loc, update, current);
            } else if (update.getType().isa<mlir::FloatType>()) {
              result = b.create<arith::MulFOp>(loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          } else if (reduceEnum == torch_upstream::ReductionType::MAX) {
            if (update.getType().isa<mlir::IntegerType>()) {
              result = b.create<arith::MaxSIOp>(loc, update, current);
            } else if (update.getType().isa<mlir::FloatType>()) {
              result = b.create<arith::MaxFOp>(loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          } else if (reduceEnum == torch_upstream::ReductionType::MIN) {
            if (update.getType().isa<mlir::IntegerType>()) {
              result = b.create<arith::MinSIOp>(loc, update, current);
            } else if (update.getType().isa<mlir::FloatType>()) {
              result = b.create<arith::MinFOp>(loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          }
          b.create<TMTensor::YieldOp>(loc, result);
        });

    // Special case for the mean
    if (reduceEnum == torch_upstream::ReductionType::MEAN) {
      counts = createTMTensorScatterOp(
          rewriter, loc, updates, indices, counts,
          /*uniqueIndices=*/false,
          [&](OpBuilder &b, Location loc, Value update, Value current) {
            Value result;
            if (mlir::IntegerType intType =
                    llvm::dyn_cast<mlir::IntegerType>(current.getType())) {
              Value constantUpdate = b.create<arith::ConstantOp>(
                  loc, b.getIntegerAttr(intType, 1));
              result = b.create<arith::AddIOp>(loc, constantUpdate, current);
            } else if (mlir::FloatType floatType =
                           llvm::dyn_cast<mlir::FloatType>(current.getType())) {
              Value constantUpdate = b.create<arith::ConstantOp>(
                  loc, b.getFloatAttr(floatType, 1.0));
              result = b.create<arith::AddFOp>(loc, constantUpdate, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
            b.create<TMTensor::YieldOp>(loc, result);
          });

      Value output = rewriter.create<tensor::EmptyOp>(
          loc, tensor::getMixedSizes(rewriter, loc, self),
          selfType.getElementType());

      // Finally divide the result
      scatterOp =
          rewriter
              .create<linalg::MapOp>(
                  loc, ValueRange{scatterOp, counts}, output,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value result;
                    if (llvm::isa<mlir::IntegerType>(args[0].getType())) {
                      result = b.create<arith::DivSIOp>(loc, args[0], args[1]);
                    } else if (llvm::isa<mlir::FloatType>(args[0].getType())) {
                      result = b.create<arith::DivFOp>(loc, args[0], args[1]);
                    } else {
                      llvm_unreachable("Only integer/float types supported!");
                    }
                    b.create<linalg::YieldOp>(loc, result);
                  })
              .getResult()[0];
    }
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, scatterOp);

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

    Value input = adaptor.getSelf();
    auto resultType = input.getType().cast<RankedTensorType>();
    Type elementType = resultType.getElementType();
    int64_t inputRank = resultType.getRank();
    Location loc = op->getLoc();
    Value dtype = op.getDtype();
    if (!dtype.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "unsupported: dtype argument not supported");

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
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
    SmallVector<int64_t> accStatic(
        makeShapeTorchCompatible(resultType.getShape()));
    accStatic.erase(accStatic.begin() + dim);
    Value acc = createZeroInitTensor(rewriter, loc, accSizes, elementType);
    Type accType =
        RankedTensorType::get(makeShapeLLVMCompatible(accStatic), elementType);
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
    target.addIllegalOp<Aten_IndexPutImplOp>();
    patterns.add<ConvertAten_IndexPutImplOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxPool2dWithIndicesBackwardOp>();
    patterns.add<ConvertAtenMaxPool2dWithIndicesBackwardOp>(typeConverter,
                                                            context);
    target.addIllegalOp<AtenScatterReduceTwoOp>();
    patterns.add<ConvertAtenScatterReduceTwoOp>(typeConverter, context);
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
