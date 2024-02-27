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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
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

static TypedAttr getNumericLimit(PatternRewriter &rewriter, Type elementType,
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

static llvm::SmallVector<int64_t> createDefaultDimMap(Value indices) {
  llvm::SmallVector<int64_t> dmap;
  if (auto iTy = dyn_cast<BaseTensorType>(indices.getType()))
    dmap.resize(iTy.getSizes()[1]);

  if (auto iTy = dyn_cast<RankedTensorType>(indices.getType()))
    dmap.resize(iTy.getDimSize(1));

  for (int i = 0, s = dmap.size(); i < s; ++i)
    dmap[i] = i;

  return dmap;
}

static Value createTMTensorScatterOp(
    OpBuilder &b, Location loc, Value updates, Value indices, Value original,
    llvm::ArrayRef<int64_t> dimensionsMap, bool uniqueIndices,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuild) {
  auto dimensionsMapAttr = b.getDenseI64ArrayAttr(dimensionsMap);
  auto originalTensorType = original.getType().cast<RankedTensorType>();
  Type originalElementType = originalTensorType.getElementType();
  auto scatterOp = b.create<TMTensor::ScatterOp>(
      loc, originalTensorType, ValueRange{updates, indices},
      ValueRange{original}, dimensionsMapAttr, uniqueIndices);

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

// Utility function to create a TMTensor::SortOp.
static FailureOr<SmallVector<Value>>
createTMTensorSortOp(PatternRewriter &rewriter, Location sortOpLoc,
                     llvm::ArrayRef<Value> operands,
                     llvm::ArrayRef<Type> elementTypes, int64_t dimension,
                     bool isStable, bool isDescending) {
  // Step 1. Create TMTensor::SortOp structure.
  SmallVector<Type> sortResultTypes;
  for (Value val : operands) {
    sortResultTypes.push_back(val.getType());
  }
  ValueRange inputs;
  auto sortOp = rewriter.create<TMTensor::SortOp>(
      sortOpLoc, sortResultTypes, inputs, operands,
      rewriter.getI64IntegerAttr(dimension));

  // Step 2. Add two arguments for each element type in the SortOp's block.
  Region *body = &sortOp.getRegion();
  Block *block = rewriter.createBlock(body);
  Location loc = body->getLoc();
  for (Type elementType : elementTypes) {
    block->addArguments({elementType, elementType},
                        SmallVector<Location, 2>(2, loc));
  }

  // Step 3. Create comparison op which will be used as the sorting predicate.
  Value compareOp;
  if (auto intType = elementTypes[0].dyn_cast<mlir::IntegerType>()) {
    // Case for using arith::CmpIOp.
    arith::CmpIPredicate ge = arith::CmpIPredicate::sge;
    arith::CmpIPredicate le = arith::CmpIPredicate::sle;
    if (intType.isUnsignedInteger()) {
      ge = arith::CmpIPredicate::uge;
      le = arith::CmpIPredicate::ule;
    }
    arith::CmpIPredicate predicate = isDescending ? ge : le;
    compareOp = rewriter.create<arith::CmpIOp>(
        loc, predicate, block->getArgument(0), block->getArgument(1));
  } else if (elementTypes[0].isa<mlir::FloatType>()) {
    // Case for using arith::CmpFOp.
    arith::CmpFPredicate predicate =
        isDescending ? arith::CmpFPredicate::OGE : arith::CmpFPredicate::OLE;
    compareOp = rewriter.create<arith::CmpFOp>(
        loc, predicate, block->getArgument(0), block->getArgument(1));
  } else {
    return rewriter.notifyMatchFailure(
        sortOpLoc, "Only Integer and Floating element type expected.");
  }

  // Step 4. Create yield op for yielding the sorting predicate.
  rewriter.create<TMTensor::YieldOp>(loc, compareOp);
  return SmallVector<Value>(sortOp.getResults());
}

namespace {
class ConvertAtenScatterSrcOp : public OpConversionPattern<AtenScatterSrcOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenScatterSrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();
    Value self = adaptor.getSelf();
    Value index = adaptor.getIndex();
    Value src = adaptor.getSrc();

    RankedTensorType selfType = self.getType().cast<RankedTensorType>();
    RankedTensorType indexType = index.getType().cast<RankedTensorType>();
    RankedTensorType srcType = src.getType().cast<RankedTensorType>();
    if (selfType.getRank() != indexType.getRank() ||
        indexType.getRank() != srcType.getRank())
      return rewriter.notifyMatchFailure(op,
                                         "'self', 'index' and 'src' should all"
                                         "have the same number of dimensions.");

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: dim is not constant");

    // Get the inputs reformatted for the TMScatterOp
    auto [indices, updates] =
        convertTorchScatterIndexAndSrcToTMScatterIndexAndSrc(rewriter, index,
                                                             src, dim);
    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, updates, indices, self,
        /*dimensionsMap=*/createDefaultDimMap(indices), /*uniqueIndices=*/false,
        [&](OpBuilder &b, Location loc, Value updatesElement,
            Value inputElement) {
          b.create<TMTensor::YieldOp>(loc, updatesElement);
        });

    auto resultType = typeConverter->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, scatterOp);
    return success();
  }
};
} // namespace

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
    const TypeConverter *typeConverter = getTypeConverter();
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
        /*dimensionsMap=*/createDefaultDimMap(indices), /*uniqueIndices=*/false,
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

Value combinePutIndices(Location loc, llvm::ArrayRef<Value> indicesRef,
                        OpBuilder b) {
  llvm::SmallVector<Value> indices(indicesRef);
  // Declare commonly used constants up front:
  Value torchCstZero =
      b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(0));
  Value torchCstOne =
      b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(1));
  Value torchCstNegOne =
      b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(-1));

  // Determine the broadcast sizes and materialize missing implicit end
  // dimensions:
  int64_t indicesRank = 0;
  for (auto index : indices) {
    auto indexTy = cast<Torch::ValueTensorType>(index.getType());
    int64_t rank = indexTy.getSizes().size();
    indicesRank = std::max(rank, indicesRank);
  }

  auto maxDim = [](int64_t dim0, int64_t dim1) {
    if (dim0 == Torch::kUnknownSize || dim1 == Torch::kUnknownSize)
      return Torch::kUnknownSize;
    return std::max(dim0, dim1);
  };

  llvm::SmallVector<Value> broadcastSizes(indicesRank, torchCstOne);
  llvm::SmallVector<int64_t> broadcastShape(indicesRank, 0);
  for (auto index : indices) {
    auto indexTy = cast<Torch::ValueTensorType>(index.getType());
    auto shape = indexTy.getSizes();
    int32_t rank = shape.size();

    for (int32_t j = 0; j < rank; ++j) {
      Value dim = b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(j));
      auto sizeOp = b.create<Torch::AtenSizeIntOp>(loc, index, dim);
      auto size = shape[j];

      int32_t idx = broadcastShape.size() - rank + j;
      broadcastSizes[idx] =
          b.create<Torch::PrimMaxIntOp>(loc, sizeOp, broadcastSizes[idx]);
      broadcastShape[idx] = maxDim(size, broadcastShape[idx]);
    }
  }

  auto mulDim = [](int64_t dim0, int64_t dim1) {
    if (dim0 == Torch::kUnknownSize || dim1 == Torch::kUnknownSize)
      return Torch::kUnknownSize;
    return dim0 * dim1;
  };

  int64_t scatterBatchCount = 1;
  for (auto dim : broadcastShape) {
    scatterBatchCount = mulDim(scatterBatchCount, dim);
  }

  // Broadcast together and flatten to batch values:
  Value broadcastSizeList = b.create<PrimListConstructOp>(
      loc, Torch::ListType::get(b.getType<Torch::IntType>()), broadcastSizes);
  for (Value &index : indices) {
    auto indexTy = cast<Torch::ValueTensorType>(index.getType());
    auto expandTy = b.getType<Torch::ValueTensorType>(
        broadcastShape, indexTy.getOptionalDtype());
    index = b.create<Torch::AtenBroadcastToOp>(loc, expandTy, index,
                                               broadcastSizeList);

    auto flattenTy = b.getType<Torch::ValueTensorType>(
        scatterBatchCount, indexTy.getOptionalDtype());
    index = b.create<Torch::AtenFlattenUsingIntsOp>(
        loc, flattenTy, index, torchCstZero, torchCstNegOne);
  }

  // Unsqueeze so we have a 1 dim to concat along:
  for (Value &tensor : indices) {
    auto btt = cast<Torch::BaseTensorType>(tensor.getType());
    if (!btt.hasSizes())
      return nullptr;

    llvm::SmallVector<int64_t> shape(btt.getSizes());
    shape.push_back(1);

    auto unsqueezeTy = b.getType<Torch::ValueTensorType>(shape, btt.getDtype());
    Value unsqueezed =
        b.create<AtenUnsqueezeOp>(loc, unsqueezeTy, tensor, torchCstOne);
    tensor = unsqueezed;
  }

  BaseTensorType unsqueezedTensorType =
      indices[0].getType().cast<BaseTensorType>();
  Value indicesTorchList = b.create<PrimListConstructOp>(
      loc, Torch::ListType::get(unsqueezedTensorType), indices);
  llvm::SmallVector<int64_t, 2> concatShape{
      unsqueezedTensorType.getSizes()[0], static_cast<int64_t>(indices.size())};
  ValueTensorType concatIndicesType = b.getType<ValueTensorType>(
      llvm::ArrayRef(concatShape), unsqueezedTensorType.getDtype());
  return b.create<AtenCatOp>(loc, concatIndicesType, indicesTorchList,
                             torchCstOne);
}

// Helper that collapses the batch dimensions together and moves it to the front
// of the array.
static Value collapseAndMoveBatchDims(Location loc, Value values, int64_t batch,
                                      int64_t count, OpBuilder b) {
  if (batch == 0 && count == 1)
    return values;

  auto valuesTy = cast<Torch::ValueTensorType>(values.getType());
  auto inShape = valuesTy.getSizes();

  llvm::SmallVector<int64_t> outShape;
  llvm::SmallVector<Value> outDims;

  // We need a length-1 dim at the start to transpose the batch to:
  if (batch != 0) {
    outDims.push_back(b.create<Torch::ConstantIntOp>(loc, 1));
    outShape.push_back(1);
  }

  // Dimensions before the batch stay the same:
  for (int i = 0; i <= batch; i++) {
    auto k = b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(i));
    auto dim = b.create<Torch::AtenSizeIntOp>(loc, values, k);
    outDims.push_back(dim);
    outShape.push_back(inShape[i]);
  }

  auto mulI = [](int64_t dim0, int64_t dim1) {
    if (dim0 == Torch::kUnknownSize || dim1 == Torch::kUnknownSize)
      return Torch::kUnknownSize;
    return dim0 * dim1;
  };

  // Determine the collapse size of the batch dimension:
  for (int i = 1; i < count; i++) {
    outShape.back() = mulI(outShape.back(), inShape[batch + i]);

    auto k =
        b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(batch + i));
    auto dim = b.create<Torch::AtenSizeIntOp>(loc, values, k);
    outDims.back() = b.create<Torch::AtenMulIntOp>(loc, dim, outDims.back());
  }

  // Add the dimensions after the batch dims:
  for (int i = batch + count, s = inShape.size(); i < s; ++i) {
    auto k = b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(i));
    auto dim = b.create<Torch::AtenSizeIntOp>(loc, values, k);
    outDims.push_back(dim);
    outShape.push_back(inShape[i]);
  }

  Value outDimsList = b.create<PrimListConstructOp>(
      loc, Torch::ListType::get(b.getType<Torch::IntType>()), outDims);

  valuesTy =
      b.getType<Torch::ValueTensorType>(outShape, valuesTy.getOptionalDtype());
  values = b.create<AtenViewOp>(loc, valuesTy, values, outDimsList);

  if (batch == 0)
    return values;

  // Batch is already at the front, no need to transpose:
  std::swap(outDims[0], outDims[batch + 1]);
  std::swap(outShape[0], outShape[batch + 1]);

  Value dim0 = b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(0));
  Value dimB =
      b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(batch + 1));

  valuesTy =
      b.getType<Torch::ValueTensorType>(outShape, valuesTy.getOptionalDtype());
  values =
      b.create<Torch::AtenTransposeIntOp>(loc, valuesTy, values, dim0, dimB);

  outDims.clear();
  outShape.clear();
  auto transposeShape = valuesTy.getSizes();
  int64_t transposeRank = transposeShape.size();
  for (int i = 0; i < transposeRank; ++i) {
    if (i == batch + 1)
      continue;
    Value k = b.create<Torch::ConstantIntOp>(loc, b.getI64IntegerAttr(i));
    outDims.push_back(b.create<AtenSizeIntOp>(loc, values, k));
    outShape.push_back(transposeShape[i]);
  }

  valuesTy =
      b.getType<Torch::ValueTensorType>(outShape, valuesTy.getOptionalDtype());
  outDimsList = b.create<PrimListConstructOp>(
      loc, Torch::ListType::get(b.getType<Torch::IntType>()), outDims);
  return b.create<AtenViewOp>(loc, valuesTy, values, outDimsList);
}

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
    Value input = op.getSelf();
    Value values = op.getValues();
    auto inputType = cast<ValueTensorType>(input.getType());
    auto valuesType = cast<ValueTensorType>(values.getType());
    int64_t inputRank = inputType.getSizes().size();
    auto valuesTensorType = op.getValues().getType().cast<BaseTensorType>();
    auto resultType = typeConverter->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();

    if (!valuesTensorType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the values tensor type must have sizes.");

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
    if (inputType.getDtype() != valuesType.getDtype())
      return rewriter.notifyMatchFailure(
          op, "Input element type should be same as the values element type.");

    SmallVector<Value> optionalIndicesList;
    getListConstructElements(op.getIndices(), optionalIndicesList);
    int64_t optionalIndicesCount = optionalIndicesList.size();
    // The size of the list of the index tensors should not be greater than the
    // input rank.
    if (optionalIndicesCount > inputRank)
      return rewriter.notifyMatchFailure(
          op, "Indices list size should not be greater than the input rank.");

    if (optionalIndicesCount == 0)
      return rewriter.notifyMatchFailure(op, "Indices list must not be empty.");

    // Filter to available indices and get the indicesMap:
    SmallVector<Value> indicesList;
    SmallVector<int64_t> indicesMap;
    int64_t numBatchDims = 0;
    for (int i = 0, s = optionalIndicesList.size(); i < s; ++i) {
      if (isa<Torch::NoneType>(optionalIndicesList[i].getType()))
        continue;
      indicesList.push_back(optionalIndicesList[i]);
      indicesMap.push_back(i);

      auto indexTy = cast<ValueTensorType>(indicesList.back().getType());
      numBatchDims = std::max(static_cast<int64_t>(indexTy.getSizes().size()),
                              numBatchDims);
    }

    // Value broadcasting semantics require batch dimensions to be up front if
    // the indices are not sequential, otherwise they are sequentially at their
    // location:
    int64_t batchDim = 0;
    for (int s = optionalIndicesList.size(); batchDim < s; ++batchDim)
      if (!isa<Torch::NoneType>(optionalIndicesList[batchDim].getType()))
        break;

    int64_t nextNone = batchDim;
    for (int s = optionalIndicesList.size(); nextNone < s; ++nextNone)
      if (isa<Torch::NoneType>(optionalIndicesList[nextNone].getType()))
        break;

    for (int s = optionalIndicesList.size(); nextNone < s; ++nextNone)
      if (!isa<Torch::NoneType>(optionalIndicesList[nextNone].getType()))
        batchDim = 0;

    // Indices are extended, catted, and collapsed into a [batch, depth] tensor:
    Value indices = combinePutIndices(loc, indicesList, rewriter);

    // Bove batch dimensions to the front and collapse into a single dim:
    values =
        collapseAndMoveBatchDims(loc, values, batchDim, numBatchDims, rewriter);
    valuesType = cast<Torch::ValueTensorType>(values.getType());

    // Materialize out the length-1 dimensions:
    Value zero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    llvm::SmallVector<int64_t> valuesShape{valuesType.getSizes().front()};
    llvm::SmallVector<Value> valuesDims;
    valuesDims.push_back(
        rewriter.create<Torch::AtenSizeIntOp>(loc, values, zero));

    int vDim = 1;
    for (int i = 0, s = inputType.getSizes().size(); i < s; ++i) {
      if (i < optionalIndicesCount &&
          !isa<Torch::NoneType>(optionalIndicesList[i].getType())) {
        valuesDims.push_back(one);
        valuesShape.push_back(1);
        continue;
      }

      Value k = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(vDim));
      valuesDims.push_back(
          rewriter.create<Torch::AtenSizeIntOp>(loc, values, k));
      valuesShape.push_back(inputType.getSizes()[i]);
      vDim++;
    }

    Value valuesDimsList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        valuesDims);

    valuesType = rewriter.getType<Torch::ValueTensorType>(
        valuesShape, valuesType.getOptionalDtype());
    values =
        rewriter.create<AtenViewOp>(loc, valuesType, values, valuesDimsList);

    // `TMTensor::ScatterOp` expects indices of element type i32.
    indices = convertTensorToDtype(
        rewriter, loc, indices,
        mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed));

    input = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(input.getType()), input);
    values = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(values.getType()), values);
    indices = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(indices.getType()), indices);

    // Creating a tm_tensor.scatter op with the following mapping:
    // 1.) Index tensor from the `indicesList` maps to the indices in scatter
    // op.
    // 2.) `values` is mapped to `updates` in scatter op.
    // 3.) `input` is mapped to `original` in scatter op.
    bool invalidInputTypeFound = false;
    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, values, indices, input, indicesMap,
        /*uniqueIndices=*/false,
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
        {originalIndicesDimExprs, updatedIndicesDimExprs},
        rewriter.getContext());
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
        /*dimensionsMap=*/createDefaultDimMap(indicesCollapsed),
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
      TypedAttr initAttr;
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
        TypedAttr minAttr = getNumericLimit(rewriter, srcType.getElementType(),
                                            /*getMin=*/true);
        normalizationValue = rewriter.create<arith::ConstantOp>(loc, minAttr);
      } else if (reduceEnum == torch_upstream::ReductionType::MIN) {
        // Set the values in the input tensor to the largest element of that
        // type
        TypedAttr maxAttr = getNumericLimit(rewriter, srcType.getElementType(),
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
          /*dimensionsMap=*/createDefaultDimMap(indices),
          /*uniqueIndices=*/false,
          [&](OpBuilder &b, Location loc, Value update, Value current) {
            b.create<TMTensor::YieldOp>(loc, update);
          });
      if (reduceEnum == torch_upstream::ReductionType::MEAN) {
        counts = createTMTensorScatterOp(
            rewriter, loc, normalizations, indices, counts,
            /*dimensionsMap=*/createDefaultDimMap(indices),
            /*uniqueIndices=*/false,
            [&](OpBuilder &b, Location loc, Value update, Value current) {
              b.create<TMTensor::YieldOp>(loc, update);
            });
      }
    }

    // Create final operation
    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, updates, indices, self,
        /*dimensionsMap=*/createDefaultDimMap(indices), /*uniqueIndices=*/false,
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
              result = b.create<arith::MaximumFOp>(loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          } else if (reduceEnum == torch_upstream::ReductionType::MIN) {
            if (update.getType().isa<mlir::IntegerType>()) {
              result = b.create<arith::MinSIOp>(loc, update, current);
            } else if (update.getType().isa<mlir::FloatType>()) {
              result = b.create<arith::MinimumFOp>(loc, update, current);
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
          /*dimensionsMap=*/createDefaultDimMap(indices),
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
class ConvertAtenSortOp : public OpConversionPattern<AtenSortOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    // Step 1. Fetch Input to sort.
    Value inputTensor = adaptor.getSelf();
    auto inputType = inputTensor.getType().cast<RankedTensorType>();
    unsigned inputRank = inputType.getRank();

    // Step 2. Fetch dimension to perform sort in.
    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant dim value is supported");
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank)) {
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");
    }

    // Step 3. Fetch the order of sorting.
    bool descending;
    if (!matchPattern(op.getDescending(), m_TorchConstantBool(&descending)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant descending value is supported");

    // Step 4. Form a RankedTensorType with same shape as that of the input's
    //         but with elemental type i64.
    RankedTensorType indicesType =
        RankedTensorType::get(inputType.getShape(), rewriter.getI64Type());

    // Step 5. Generate indices tensor.
    SmallVector<Value> dynDims;
    for (unsigned i = 0; i < inputType.getRank(); i++) {
      if (inputType.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, inputTensor, i));
      }
    }
    Value initEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), rewriter.getI64Type(), dynDims);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::getMultiDimIdentityMap(inputRank, op.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
    Value indicesTensor =
        rewriter
            .create<linalg::GenericOp>(
                loc, initEmptyTensor.getType(), ValueRange{}, initEmptyTensor,
                indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value index = b.create<linalg::IndexOp>(loc, dim);
                  index = castIndexToInt64(b, loc, index);
                  b.create<linalg::YieldOp>(loc, index);
                })
            .getResult(0);

    // Step 6. Create TMTensor::SortOp.
    SmallVector<Value> operands;
    operands.push_back(inputTensor);
    operands.push_back(indicesTensor);
    SmallVector<Type> elementTypes;
    elementTypes.push_back(inputType.getElementType());
    elementTypes.push_back(indicesType.getElementType());

    // The default value for aten.sort op's `stable` parameter is `false`.
    // Refer: https://pytorch.org/docs/stable/generated/torch.sort.html
    FailureOr<SmallVector<Value>> sortOpValues =
        createTMTensorSortOp(rewriter, loc, operands, elementTypes,
                             /*dimension=*/dim, /*isStable=*/false,
                             /*isDescending=*/descending);
    if (failed(sortOpValues))
      return rewriter.notifyMatchFailure(
          loc, "Only Integer and Floating element type expected.");

    auto sortOpVal = *sortOpValues;
    rewriter.replaceOp(op, sortOpVal);
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

    Location loc = op.getLoc();
    Value input = adaptor.getSelf();
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    Type elementType = resultType.getElementType();
    Type inputElementType =
        input.getType().cast<RankedTensorType>().getElementType();

    // Converting the input element type to the result's element type.
    // The only possible mismatch would be when the input element type is an
    // integer but not `si64`. Therefore, we directly convert the input to
    // `si64`. Rest all cases are handled in the dtype definition for this op.
    if (elementType != inputElementType) {
      Value torchInput = convertTensorToDtype(
          rewriter, loc, op.getSelf(),
          rewriter.getIntegerType(64, IntegerType::Signed));
      input = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(torchInput.getType()),
          torchInput);
    }

    int64_t inputRank = resultType.getRank();
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
          Value sum =
              (input.getType().isa<mlir::FloatType>()
                   ? b.create<arith::AddFOp>(loc, input, acc)->getResult(0)
                   : b.create<arith::AddIOp>(loc, input, acc)->getResult(0));
          b.create<TMTensor::YieldOp>(loc, sum);
        });

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenScaledDotProductAttentionOp
    : public OpConversionPattern<AtenScaledDotProductAttentionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenScaledDotProductAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value mask = op.getAttnMask();
    Value dropoutP = op.getDropoutP();
    Value isCausal = op.getIsCausal();
    Value scale = op.getScale();
    Type elementType =
        adaptor.getQuery().getType().cast<ShapedType>().getElementType();

    // Verify inputs (only support defaults)
    if (!mask.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "attention masking not supported");
    double dropout;
    if (!matchPattern(dropoutP, m_TorchConstantFloat(&dropout)) ||
        dropout > 0.0)
      return rewriter.notifyMatchFailure(op.getLoc(), "dropout not supported");
    bool causal;
    if (!matchPattern(isCausal, m_TorchConstantBool(&causal)) || causal)
      return rewriter.notifyMatchFailure(
          op.getLoc(), "causal attention masking not supported");
    if (!scale.getType().isa<Torch::NoneType>()) {
      double scaleFloat;
      if (!matchPattern(scale, m_TorchConstantFloat(&scaleFloat)) ||
          scaleFloat != 1.0)
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "only default scale supported");
    }

    auto opTy = cast<ValueTensorType>(op.getType()).toBuiltinTensor();
    auto query = adaptor.getQuery();
    auto value = adaptor.getValue();
    auto key = adaptor.getKey();
    auto queryTy = cast<ShapedType>(query.getType());
    auto valueTy = cast<ShapedType>(value.getType());
    auto keyTy = cast<ShapedType>(key.getType());

    if (queryTy.getRank() != valueTy.getRank() ||
        queryTy.getRank() != keyTy.getRank())
      return rewriter.notifyMatchFailure(op, "operand ranks do not match");

    if (queryTy.getRank() < 3)
      return rewriter.notifyMatchFailure(op, "missing batch dimension");

    llvm::SmallVector<ReassociationIndices, 3> reassociation(3);
    for (int i = 0, s = valueTy.getRank() - 2; i < s; ++i)
      reassociation.front().push_back(i);
    reassociation[1].push_back(valueTy.getRank() - 2);
    reassociation[2].push_back(valueTy.getRank() - 1);

    auto loc = op.getLoc();
    auto collapseBatch = [&rewriter, &reassociation,
                          loc](Value value) -> Value {
      auto valueTy = cast<ShapedType>(value.getType());
      if (valueTy.getRank() == 3)
        return value;

      llvm::SmallVector<int64_t, 3> newShape(3, 1);
      newShape[1] = valueTy.getDimSize(valueTy.getRank() - 2);
      newShape[2] = valueTy.getDimSize(valueTy.getRank() - 1);

      for (int i = 0, s = valueTy.getRank() - 2; i < s; ++i) {
        if (valueTy.isDynamicDim(i)) {
          newShape[0] = ShapedType::kDynamic;
          break;
        }
        newShape[0] = newShape[0] * valueTy.getDimSize(i);
      }

      auto collapseTy = valueTy.clone(newShape);
      return rewriter.create<tensor::CollapseShapeOp>(loc, collapseTy, value,
                                                      reassociation);
    };

    query = collapseBatch(query);
    key = collapseBatch(key);
    value = collapseBatch(value);

    SmallVector<int64_t> outSizes(
        query.getType().cast<ShapedType>().getShape());
    SmallVector<int64_t> valueSizes(
        value.getType().cast<ShapedType>().getShape());
    outSizes[outSizes.size() - 1] = valueSizes[valueSizes.size() - 1];
    SmallVector<Value> outSizesDynamic(
        getTensorSizes(rewriter, op.getLoc(), query));
    outSizesDynamic[outSizesDynamic.size() - 1] =
        getTensorSizes(rewriter, op.getLoc(), value)[valueSizes.size() - 1];
    Type outType = RankedTensorType::get(outSizes, elementType);
    Value output = createZeroInitTensor(rewriter, op.getLoc(), outSizesDynamic,
                                        elementType);

    // Overwrite with tm_tensor::attention
    Value attention =
        rewriter
            .create<AttentionOp>(loc, outType,
                                 SmallVector<Value>{query, key, value},
                                 SmallVector<Value>{output})
            .getResult()[0];

    if (opTy != outType) {
      attention = rewriter.create<tensor::ExpandShapeOp>(loc, opTy, attention,
                                                         reassociation);
    }

    rewriter.replaceOp(op, attention);

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
                           math::MathDialect, Torch::TorchDialect,
                           TMTensorDialect>();

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
    target.addIllegalOp<AtenSortOp>();
    patterns.add<ConvertAtenSortOp>(typeConverter, context);
    target.addIllegalOp<AtenCumsumOp>();
    patterns.add<ConvertAtenCumsumOp>(typeConverter, context);
    target.addIllegalOp<AtenScaledDotProductAttentionOp>();
    patterns.add<ConvertAtenScaledDotProductAttentionOp>(typeConverter,
                                                         context);

    target.addIllegalOp<AtenScatterSrcOp>();
    patterns.add<ConvertAtenScatterSrcOp>(typeConverter, context);

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
