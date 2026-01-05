//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir/Conversion/Passes.h"
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
namespace mlir::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOTMTENSOR
#include "torch-mlir/Conversion/Passes.h.inc"

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
  RankedTensorType indexType = cast<RankedTensorType>(indices.getType());
  RankedTensorType srcSelf = cast<RankedTensorType>(src.getType());

  // Store location for insertions
  Location loc = src.getLoc();

  Type indicesElemType = getElementTypeOrSelf(indices);
  Value indexSize = getTensorSize(rewriter, loc, indices);
  indexSize = castIntToIndex(rewriter, loc, indexSize);
  SmallVector<Value> indexShape = getTensorSizes(rewriter, loc, indices);
  Value cstOne = arith::ConstantIndexOp::create(rewriter, loc, 1);

  // We flatten the `src` values from (i, j, k, ...) -> (i * j * k * ...)
  SmallVector<Value> indSliceShape({indexSize, cstOne});
  Value indSlice =
      createZeroInitTensor(rewriter, loc, indSliceShape, indicesElemType);

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
      linalg::GenericOp::create(
          rewriter, loc, outputsType, ValueRange(), outputs, mapping,
          iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            SmallVector<Value> indexValues(indexType.getRank());
            Value ind = linalg::IndexOp::create(b, loc, 0);
            for (int i = indexType.getRank() - 1; i >= 0; i--) {
              indexValues[i] =
                  arith::RemSIOp::create(b, loc, ind, indexShape[i]);
              ind = arith::DivSIOp::create(b, loc, ind, indexShape[i]);
            }
            // Extract the scatter index and update value
            Value extractIndexValue =
                tensor::ExtractOp::create(b, loc, indices, indexValues);
            Value extractSrcValue =
                tensor::ExtractOp::create(b, loc, src, indexValues);
            SmallVector<Value> yieldVals;
            for (Value v : indexValues) {
              Value scalar = castIndexToInt64(b, loc, v);
              yieldVals.push_back(
                  convertScalarToDtype(rewriter, loc, scalar, indicesElemType));
            }
            // Replace the original index with the index specified
            // by the scatter.
            yieldVals[dim] = convertScalarToDtype(
                rewriter, loc, extractIndexValue, indicesElemType);
            yieldVals.push_back(extractSrcValue);
            linalg::YieldOp::create(b, loc, yieldVals);
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
      arith::ConstantIndexOp::create(rewriter, loc, 0),
      arith::ConstantIndexOp::create(rewriter, loc, 0)};
  SmallVector<Value> strides = {
      arith::ConstantIndexOp::create(rewriter, loc, 1),
      arith::ConstantIndexOp::create(rewriter, loc, 1)};
  Value indicesRank =
      arith::ConstantIndexOp::create(rewriter, loc, indexType.getRank());
  Value flattenedIndices = createZeroInitTensor(
      rewriter, loc, SmallVector<Value>({indexSize, indicesRank}),
      indexType.getElementType());
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
  auto originalTensorType = cast<RankedTensorType>(original.getType());
  Type originalElementType = originalTensorType.getElementType();
  auto scatterOp = TMTensor::ScatterOp::create(
      b, loc, originalTensorType, ValueRange{updates, indices},
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
  auto inputType = cast<RankedTensorType>(input.getType());
  Type elementType = inputType.getElementType();
  auto scanOp =
      TMTensor::ScanOp::create(b, loc, ValueRange{input},
                               ValueRange{output, accumulator}, dim, inclusive);

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

static FailureOr<Value> createIntOrFloatCompareOp(PatternRewriter &rewriter,
                                                  Location loc,
                                                  Type elementType, Value lhs,
                                                  Value rhs, bool isDescending,
                                                  bool isEqual) {

  Value compareOp;
  if (auto intType = dyn_cast<mlir::IntegerType>(elementType)) {
    // Case for using arith::CmpIOp.
    arith::CmpIPredicate g =
        isEqual ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sgt;
    arith::CmpIPredicate l =
        isEqual ? arith::CmpIPredicate::sle : arith::CmpIPredicate::slt;
    if (intType.isUnsignedInteger()) {
      g = isEqual ? arith::CmpIPredicate::uge : arith::CmpIPredicate::ugt;
      l = isEqual ? arith::CmpIPredicate::ule : arith::CmpIPredicate::ult;
    }
    arith::CmpIPredicate predicate = isDescending ? g : l;
    compareOp = arith::CmpIOp::create(rewriter, loc, predicate, lhs, rhs);
    return compareOp;
  }

  if (isa<mlir::FloatType>(elementType)) {
    // Case for using arith::CmpFOp.
    arith::CmpFPredicate g =
        isEqual ? arith::CmpFPredicate::OGE : arith::CmpFPredicate::OGT;
    arith::CmpFPredicate l =
        isEqual ? arith::CmpFPredicate::OLE : arith::CmpFPredicate::OLT;

    arith::CmpFPredicate predicate = isDescending ? g : l;
    compareOp = arith::CmpFOp::create(rewriter, loc, predicate, lhs, rhs);
    return compareOp;
  }

  return rewriter.notifyMatchFailure(
      loc, "Only Integer and Floating element type expected.");
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
  auto sortOp =
      TMTensor::SortOp::create(rewriter, sortOpLoc, sortResultTypes, inputs,
                               operands, rewriter.getI64IntegerAttr(dimension));

  // Step 2. Add two arguments for each element type in the SortOp's block.
  Region *body = &sortOp.getRegion();
  Block *block = rewriter.createBlock(body);
  Location loc = body->getLoc();
  for (Type elementType : elementTypes) {
    block->addArguments({elementType, elementType},
                        SmallVector<Location, 2>(2, loc));
  }

  // Step 3. Create comparison op which will be used as the sorting predicate.
  auto compareOpRetVal = createIntOrFloatCompareOp(
      rewriter, loc, elementTypes[0], block->getArgument(0),
      block->getArgument(1), isDescending, true);

  if (failed(compareOpRetVal))
    return rewriter.notifyMatchFailure(
        loc, "Only Integer and Floating element type expected.");

  // Step 4. Create yield op for yielding the sorting predicate.
  TMTensor::YieldOp::create(rewriter, loc, compareOpRetVal.value());
  return SmallVector<Value>(sortOp.getResults());
}

static FailureOr<SmallVector<Value>> createTMTensorTopkOp(
    PatternRewriter &rewriter, Location topkOpLoc, llvm::ArrayRef<Value> inputs,
    llvm::ArrayRef<Value> outputs, llvm::ArrayRef<Type> elementTypes,
    int64_t dimension, bool isMinK) {

  // Generate output types.
  SmallVector<Type> topkResultTypes;
  for (Value val : outputs) {
    topkResultTypes.push_back(val.getType());
  }

  // Create empty TopkOp, add body later.
  auto topkOp =
      TMTensor::TopkOp::create(rewriter, topkOpLoc, topkResultTypes, inputs,
                               outputs, rewriter.getI64IntegerAttr(dimension));

  Region *body = &topkOp.getRegion();
  Block *block = rewriter.createBlock(body);
  Location loc = body->getLoc();
  // Add arguments for each passed body region element type.
  for (Type elementType : elementTypes) {
    block->addArgument({elementType}, {loc});
  }

  // Generate compare operator. If minK is chosen, isDescending should be false.
  // Is equal should be false, because we do not want equality to cause element
  // swap.
  auto compareOpRetVal = createIntOrFloatCompareOp(
      rewriter, loc, elementTypes[0], block->getArgument(0),
      block->getArgument(1), /*isDescending=*/!isMinK, /*isEqual=*/false);

  // Check if correct element types are passed.
  if (failed(compareOpRetVal))
    return rewriter.notifyMatchFailure(
        loc, "Only Integer and Floating element type expected.");

  // Yield the comparison result.
  TMTensor::YieldOp::create(rewriter, loc, compareOpRetVal.value());
  return SmallVector<Value>(topkOp.getResults());
}

static FailureOr<Value>
repeatTensorElementsForDim(Operation *op, ConversionPatternRewriter &rewriter,
                           Type resType, Value self, int64_t repeats,
                           int64_t dim) {
  Location loc = op->getLoc();
  auto context = op->getContext();
  auto selfTy = cast<BaseTensorType>(self.getType());

  int64_t inputRank = selfTy.getSizes().size();
  dim = toPositiveDim(dim, inputRank);
  Value dimValue =
      ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(dim));
  Value dimValuePlusOne =
      ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(dim + 1));

  auto unsqueezedInfo = unsqueezeTensor(rewriter, op, self, dimValuePlusOne);
  if (failed(unsqueezedInfo))
    return rewriter.notifyMatchFailure(op,
                                       "cannot generate unsqueeze tensor op");
  self = *unsqueezedInfo;

  Value constMinusOne =
      ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(-1));
  SmallVector<Value> expandShapeValueList(inputRank + 1, constMinusOne);
  expandShapeValueList[dim + 1] =
      ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(repeats));
  Value expandShapeList = PrimListConstructOp::create(
      rewriter, loc, ListType::get(IntType::get(context)),
      expandShapeValueList);

  SmallVector<int64_t> expandShape(inputRank + 1);
  for (int64_t i = 0; i <= dim; i++) {
    expandShape[i] = selfTy.getSizes()[i];
  }
  expandShape[dim + 1] = repeats;
  for (int64_t i = dim + 1; i < inputRank; i++) {
    expandShape[i + 1] = selfTy.getSizes()[i];
  }

  BaseTensorType expandTy =
      rewriter.getType<ValueTensorType>(expandShape, selfTy.getOptionalDtype());
  Value expandSelf =
      AtenBroadcastToOp::create(rewriter, loc, expandTy, self, expandShapeList);

  Value result = PrimsCollapseOp::create(rewriter, loc, resType, expandSelf,
                                         dimValue, dimValuePlusOne);
  return result;
}

namespace {
template <typename AtenOpT>
class ConvertAtenScatterOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    const TypeConverter *typeConverter =
        OpConversionPattern<AtenOpT>::getTypeConverter();
    Value self = adaptor.getSelf();
    Value index = adaptor.getIndex();
    Value src = adaptor.getSrc();

    RankedTensorType selfType = cast<RankedTensorType>(self.getType());
    RankedTensorType indexType = cast<RankedTensorType>(index.getType());
    RankedTensorType srcType = cast<RankedTensorType>(src.getType());
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
          if (isa<AtenScatterSrcOp>(op)) {
            TMTensor::YieldOp::create(b, loc, updatesElement);
          } else if (isa<AtenScatterAddOp>(op)) {
            if (isa<mlir::IntegerType>(selfType.getElementType())) {
              Value add =
                  arith::AddIOp::create(b, loc, inputElement, updatesElement);
              TMTensor::YieldOp::create(b, loc, add);
            } else if (isa<mlir::FloatType>(selfType.getElementType())) {
              Value add =
                  arith::AddFOp::create(b, loc, inputElement, updatesElement);
              TMTensor::YieldOp::create(b, loc, add);
            }
          }
        });

    auto resultType = cast<RankedTensorType>(
        typeConverter->convertType(op->getResult(0).getType()));
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
    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    if (inputType.getRank() != 1 ||
        !isa<mlir::IntegerType>(inputType.getElementType()))
      return rewriter.notifyMatchFailure(
          op,
          "Input tensor has to be a one-dimensional tensor of integer type.");

    // Check whether the input tensor element type is i64 or not.
    IntegerType inputIntegerType =
        cast<IntegerType>(inputType.getElementType());
    if (inputIntegerType.getWidth() != 64)
      return rewriter.notifyMatchFailure(
          op,
          "Unimplemented: Integer width not equal to 64 are not supported.");

    // TODO: Incorporate the weight argument.
    if (!isa<mlir::torch::Torch::NoneType>(weights.getType()))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: the weights operand is not incorporated.");

    // Finding the maximum value in the input tensor.
    SmallVector<int64_t> maxTensorSizes;
    ValueTensorType maxTensorType = ValueTensorType::get(
        context, llvm::ArrayRef(maxTensorSizes),
        cast<ValueTensorType>(torchTypeInput.getType()).getDtype());
    Value maxTensor =
        AtenMaxOp::create(rewriter, loc, maxTensorType, torchTypeInput);
    maxTensor = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(maxTensor.getType()),
        maxTensor);

    // `maxTensor` is a 0-d tensor, extracting its only element and
    // storing it in `maxInput`.
    Value maxInput = tensor::ExtractOp::create(rewriter, loc, maxTensor);

    // Creating a tm_tensor.scatter op with the following mapping:
    // 1.) `input` tensor maps to the indices in scatter op. `input` is
    // expanded from 1-d to 2-d.
    // 2.) `updates` is a 1-d dummy tensor with the size equivalent to the
    // `input`.
    // 3.) `bincount` a 1-d tensor maps to the original in scatter op
    // with size equal to the max(max(input) + 1, minlength).
    SmallVector<int64_t> expandedInputSizes{
        makeShapeTorchCompatible(inputType.getShape())[0], 1};
    ValueTensorType expandInputType = ValueTensorType::get(
        context, llvm::ArrayRef(expandedInputSizes),
        cast<ValueTensorType>(torchTypeInput.getType()).getDtype());
    Value torchCstOne = Torch::ConstantIntOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(1));
    Value expandedInputTensor = AtenUnsqueezeOp::create(
        rewriter, loc, expandInputType, torchTypeInput, torchCstOne);

    Value indices = typeConverter->materializeTargetConversion(
        rewriter, loc,
        typeConverter->convertType(expandedInputTensor.getType()),
        expandedInputTensor);

    auto resultType = cast<RankedTensorType>(
        typeConverter->convertType(op->getResult(0).getType()));
    Type resultElemType = resultType.getElementType();

    SmallVector<Value, 1> inputSizeDynamic =
        getTensorSizesUntilDim(rewriter, loc, input, 0);
    Value updatesTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(inputSizeDynamic), resultElemType);

    Value constantZero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(resultElemType));
    Value constantOne = arith::ConstantIntOp::create(
        rewriter, loc, 1, resultElemType.getIntOrFloatBitWidth());

    // Bincount size = max(max(input) + 1, minlength)
    Value maxInputPlusOne =
        arith::AddIOp::create(rewriter, loc, maxInput, constantOne);
    Value bincountSize =
        arith::MaxSIOp::create(rewriter, loc, maxInputPlusOne, minlength);
    bincountSize = castIntToIndex(rewriter, loc, bincountSize);
    Value bincountTensor = createInitTensor(rewriter, loc, {bincountSize},
                                            resultElemType, constantZero);

    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, updatesTensor, indices, bincountTensor,
        /*dimensionsMap=*/createDefaultDimMap(indices), /*uniqueIndices=*/false,
        [&](OpBuilder &b, Location loc, Value _, Value bincountElem) {
          Value add = arith::AddIOp::create(b, loc, bincountElem, constantOne);
          TMTensor::YieldOp::create(b, loc, add);
        });
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, scatterOp);
    return success();
  }
};
} // namespace

namespace {

// Determine the common broadcast shape of all the index tensors.
std::pair<llvm::SmallVector<Value>, llvm::SmallVector<int64_t>>
getBroadcastShape(Location loc, llvm::ArrayRef<Value> indices, OpBuilder b) {
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

  Value torchCstOne =
      Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(1));
  llvm::SmallVector<Value> broadcastSizes(indicesRank, torchCstOne);
  llvm::SmallVector<int64_t> broadcastShape(indicesRank, 0);
  for (auto index : indices) {
    auto indexTy = cast<Torch::ValueTensorType>(index.getType());
    auto shape = indexTy.getSizes();
    int32_t rank = shape.size();

    for (int32_t j = 0; j < rank; ++j) {
      Value dim = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(j));
      auto sizeOp = Torch::AtenSizeIntOp::create(b, loc, index, dim);
      auto size = shape[j];

      int32_t idx = broadcastShape.size() - rank + j;
      broadcastSizes[idx] =
          Torch::PrimMaxIntOp::create(b, loc, sizeOp, broadcastSizes[idx]);
      broadcastShape[idx] = maxDim(size, broadcastShape[idx]);
    }
  }
  return std::make_pair(broadcastSizes, broadcastShape);
}

Value combinePutIndices(Location loc, llvm::ArrayRef<Value> indicesRef,
                        OpBuilder b) {
  llvm::SmallVector<Value> indices(indicesRef);
  // Declare commonly used constants up front:
  Value torchCstZero =
      Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(0));
  Value torchCstOne =
      Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(1));
  Value torchCstNegOne =
      Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(-1));

  auto [broadcastSizes, broadcastShape] = getBroadcastShape(loc, indicesRef, b);

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
  Value broadcastSizeList = PrimListConstructOp::create(
      b, loc, Torch::ListType::get(b.getType<Torch::IntType>()),
      broadcastSizes);
  for (Value &index : indices) {
    auto indexTy = cast<Torch::ValueTensorType>(index.getType());
    auto expandTy = b.getType<Torch::ValueTensorType>(
        broadcastShape, indexTy.getOptionalDtype());
    index = Torch::AtenBroadcastToOp::create(b, loc, expandTy, index,
                                             broadcastSizeList);

    auto flattenTy = b.getType<Torch::ValueTensorType>(
        scatterBatchCount, indexTy.getOptionalDtype());
    index = Torch::AtenFlattenUsingIntsOp::create(b, loc, flattenTy, index,
                                                  torchCstZero, torchCstNegOne);
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
        AtenUnsqueezeOp::create(b, loc, unsqueezeTy, tensor, torchCstOne);
    tensor = unsqueezed;
  }

  BaseTensorType unsqueezedTensorType =
      cast<BaseTensorType>(indices[0].getType());
  Value indicesTorchList = PrimListConstructOp::create(
      b, loc, Torch::ListType::get(unsqueezedTensorType), indices);
  llvm::SmallVector<int64_t, 2> concatShape{
      unsqueezedTensorType.getSizes()[0], static_cast<int64_t>(indices.size())};
  ValueTensorType concatIndicesType = b.getType<ValueTensorType>(
      llvm::ArrayRef(concatShape), unsqueezedTensorType.getDtype());
  return AtenCatOp::create(b, loc, concatIndicesType, indicesTorchList,
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
    outDims.push_back(Torch::ConstantIntOp::create(b, loc, 1));
    outShape.push_back(1);
  }

  // Dimensions before the batch stay the same:
  for (int i = 0; i <= batch; i++) {
    auto k = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(i));
    auto dim = Torch::AtenSizeIntOp::create(b, loc, values, k);
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
        Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(batch + i));
    auto dim = Torch::AtenSizeIntOp::create(b, loc, values, k);
    outDims.back() = Torch::AtenMulIntOp::create(b, loc, dim, outDims.back());
  }

  // Add the dimensions after the batch dims:
  for (int i = batch + count, s = inShape.size(); i < s; ++i) {
    auto k = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(i));
    auto dim = Torch::AtenSizeIntOp::create(b, loc, values, k);
    outDims.push_back(dim);
    outShape.push_back(inShape[i]);
  }

  Value outDimsList = PrimListConstructOp::create(
      b, loc, Torch::ListType::get(b.getType<Torch::IntType>()), outDims);

  valuesTy =
      b.getType<Torch::ValueTensorType>(outShape, valuesTy.getOptionalDtype());
  values = AtenViewOp::create(b, loc, valuesTy, values, outDimsList);

  if (batch == 0)
    return values;

  // Batch is already at the front, no need to transpose:
  std::swap(outDims[0], outDims[batch + 1]);
  std::swap(outShape[0], outShape[batch + 1]);

  Value dim0 = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(0));
  Value dimB =
      Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(batch + 1));

  valuesTy =
      b.getType<Torch::ValueTensorType>(outShape, valuesTy.getOptionalDtype());
  values =
      Torch::AtenTransposeIntOp::create(b, loc, valuesTy, values, dim0, dimB);

  outDims.clear();
  outShape.clear();
  auto transposeShape = valuesTy.getSizes();
  int64_t transposeRank = transposeShape.size();
  for (int i = 0; i < transposeRank; ++i) {
    if (i == batch + 1)
      continue;
    Value k = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(i));
    outDims.push_back(AtenSizeIntOp::create(b, loc, values, k));
    outShape.push_back(transposeShape[i]);
  }

  valuesTy =
      b.getType<Torch::ValueTensorType>(outShape, valuesTy.getOptionalDtype());
  outDimsList = PrimListConstructOp::create(
      b, loc, Torch::ListType::get(b.getType<Torch::IntType>()), outDims);
  return AtenViewOp::create(b, loc, valuesTy, values, outDimsList);
}

// Broadcast the `values` tensor to the slice size created by the list of index
// tensors.
static Value broadcastValuesToSliceSize(Location loc, Value input, Value values,
                                        llvm::ArrayRef<Value> indices,
                                        OpBuilder b) {
  auto inputType = cast<ValueTensorType>(input.getType());
  ArrayRef<int64_t> inputStaticShape = inputType.getSizes();
  auto valuesType = cast<ValueTensorType>(values.getType());

  // In the case where the input rank is greater than the number of index
  // tensors, the remaining dimensions of the input are indexed in their
  // entirety. Thus, we need to append the remaining dimensions to get the shape
  // of the indexed slice.
  auto [resultShape, resultStaticShape] = getBroadcastShape(loc, indices, b);
  for (size_t i = indices.size(); i < inputStaticShape.size(); i++) {
    Value dim = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(i));
    resultShape.push_back(AtenSizeIntOp::create(b, loc, input, dim));
    resultStaticShape.push_back(inputStaticShape[i]);
  }

  auto resultType = b.getType<Torch::ValueTensorType>(
      resultStaticShape, valuesType.getOptionalDtype());
  Value broadcastShapeList = PrimListConstructOp::create(
      b, loc, Torch::ListType::get(b.getType<Torch::IntType>()), resultShape);
  return AtenBroadcastToOp::create(b, loc, resultType, values,
                                   broadcastShapeList);
}

class ConvertAtenIndexPutHackedTwinOp
    : public OpConversionPattern<AtenIndexPutHackedTwinOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenIndexPutHackedTwinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value values = op.getValues();
    auto inputType = cast<ValueTensorType>(input.getType());
    auto valuesType = cast<ValueTensorType>(values.getType());
    int64_t inputRank = inputType.getSizes().size();
    auto valuesTensorType = cast<BaseTensorType>(op.getValues().getType());
    auto resultType = cast<RankedTensorType>(
        typeConverter->convertType(op->getResult(0).getType()));

    if (!valuesTensorType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the values tensor type must have sizes.");

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

    values = broadcastValuesToSliceSize(loc, input, values, optionalIndicesList,
                                        rewriter);
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
    Value zero = Torch::ConstantIntOp::create(rewriter, loc,
                                              rewriter.getI64IntegerAttr(0));
    Value one = Torch::ConstantIntOp::create(rewriter, loc,
                                             rewriter.getI64IntegerAttr(1));
    llvm::SmallVector<int64_t> valuesShape;
    llvm::SmallVector<Value> valuesDims;
    int vDim = 0;

    if (optionalIndicesCount + valuesType.getSizes().size() >
        inputType.getSizes().size()) {
      valuesShape.push_back(valuesType.getSizes().front());
      valuesDims.push_back(
          Torch::AtenSizeIntOp::create(rewriter, loc, values, zero));
      vDim++;
    }

    for (int i = 0, s = inputType.getSizes().size(); i < s; ++i) {
      if (i < optionalIndicesCount &&
          !isa<Torch::NoneType>(optionalIndicesList[i].getType())) {
        valuesDims.push_back(one);
        valuesShape.push_back(1);
        continue;
      }

      Value k = Torch::ConstantIntOp::create(rewriter, loc,
                                             rewriter.getI64IntegerAttr(vDim));
      valuesDims.push_back(
          Torch::AtenSizeIntOp::create(rewriter, loc, values, k));
      valuesShape.push_back(inputType.getSizes()[i]);
      vDim++;
    }

    Value valuesDimsList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        valuesDims);

    valuesType = rewriter.getType<Torch::ValueTensorType>(
        valuesShape, valuesType.getOptionalDtype());
    values =
        AtenViewOp::create(rewriter, loc, valuesType, values, valuesDimsList);

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
    // If accumulate == false, the behavior is undefined if the indicies aren't
    // unique.
    bool uniqueIndices = !accumulate;
    Value scatterOp = createTMTensorScatterOp(
        rewriter, loc, values, indices, input, indicesMap,
        /*uniqueIndices=*/uniqueIndices,
        [&](OpBuilder &b, Location loc, Value valuesElement,
            Value inputElement) {
          Value yieldValue = valuesElement;
          if (accumulate) {
            if (isa<mlir::IntegerType>(inputElement.getType())) {
              yieldValue =
                  arith::AddIOp::create(b, loc, inputElement, valuesElement);
            } else if (isa<mlir::FloatType>(inputElement.getType())) {
              yieldValue =
                  arith::AddFOp::create(b, loc, inputElement, valuesElement);
            } else {
              invalidInputTypeFound = true;
              return;
            }
          }
          TMTensor::YieldOp::create(b, loc, yieldValue);
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
    Value gradOutput = adaptor.getGradOutput();
    Value input = adaptor.getSelf();
    RankedTensorType gradOutputType =
        cast<RankedTensorType>(gradOutput.getType());
    Type gradOutputElemType = gradOutputType.getElementType();
    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    Type inputElemType = inputType.getElementType();
    int64_t tensorOperandRank = inputType.getRank();

    Value indices = adaptor.getIndices();
    RankedTensorType indicesType = cast<RankedTensorType>(indices.getType());
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

    Value initTensor = tensor::EmptyOp::create(
        rewriter, loc, updatedIndicesShape, indicesElemType);

    Value wIn = inputShape[tensorOperandRank - 1];
    SmallVector<Value> cstValues;
    for (int64_t i = 0; i < tensorOperandRank; i++)
      cstValues.push_back(arith::ConstantIndexOp::create(rewriter, loc, i));

    Value updatedIndices =
        linalg::GenericOp::create(
            rewriter, loc, initTensor.getType(), indices, initTensor,
            indexingMaps, iteratorTypes,
            [tensorOperandRank, wIn, cstValues,
             indicesElemType](OpBuilder &b, Location loc, ValueRange args) {
              Value index = castIntToIndex(b, loc, args[0]);
              Value updatedIndex = cstValues[0];
              Value lastDim =
                  linalg::IndexOp::create(b, loc, tensorOperandRank);

              for (int64_t i = tensorOperandRank - 1; i >= 0; i--) {
                Value result;
                if (i == tensorOperandRank - 1)
                  result = arith::RemSIOp::create(b, loc, index, wIn);
                if (i == tensorOperandRank - 2)
                  result = arith::FloorDivSIOp::create(b, loc, index, wIn);
                if (i == tensorOperandRank - 3 || i == tensorOperandRank - 4)
                  result = linalg::IndexOp::create(b, loc, i);

                Value pred = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::eq, lastDim, cstValues[i]);
                Value addAmount =
                    arith::SelectOp::create(b, loc, pred, result, cstValues[0]);
                updatedIndex =
                    arith::AddIOp::create(b, loc, updatedIndex, addAmount);
              }

              updatedIndex = arith::IndexCastOp::create(b, loc, indicesElemType,
                                                        updatedIndex);
              linalg::YieldOp::create(b, loc, updatedIndex);
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
    Value gradOutputFlattened =
        tensor::CollapseShapeOp::create(rewriter, loc, gradOutputFlattenedType,
                                        gradOutput, reassociationCollapse);

    // Collapsing updated indices into a 2-d tensor.
    SmallVector<ReassociationIndices> reassociationCollapseIndices(2);
    for (auto i = 0; i < tensorOperandRank; i++)
      reassociationCollapseIndices[0].push_back(i);
    reassociationCollapseIndices[1].push_back(tensorOperandRank);
    int64_t numelIndices = getNumberOfElements(indicesType);
    Value indicesCollapsed = tensor::CollapseShapeOp::create(
        rewriter, loc,
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
          if (isa<mlir::IntegerType>(inputElement.getType())) {
            yieldValue =
                arith::AddIOp::create(b, loc, inputElement, valuesElement);
          } else if (isa<mlir::FloatType>(inputElement.getType())) {
            yieldValue =
                arith::AddFOp::create(b, loc, inputElement, valuesElement);
          } else {
            invalidInputTypeFound = true;
            return;
          }
          TMTensor::YieldOp::create(b, loc, yieldValue);
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
        cast<RankedTensorType>(adaptor.getSelf().getType());
    RankedTensorType indexType =
        cast<RankedTensorType>(adaptor.getIndex().getType());
    RankedTensorType srcType =
        cast<RankedTensorType>(adaptor.getSrc().getType());

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
      Value initElement = arith::ConstantOp::create(rewriter, loc, initAttr);
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
        normalizationValue = arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(srcType.getElementType()));
      } else if (reduceEnum == torch_upstream::ReductionType::PROD) {
        // Set the values in the input tensor to '1' (multiplication identity)
        if (llvm::isa<mlir::FloatType>(srcType.getElementType())) {
          normalizationValue = arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getFloatAttr(srcType.getElementType(), 1.0));
        } else if (llvm::isa<mlir::IntegerType>(srcType.getElementType())) {
          normalizationValue = arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getIntegerAttr(srcType.getElementType(), 1));
        } else {
          llvm_unreachable("Only integer/float types supported!");
        }
      } else if (reduceEnum == torch_upstream::ReductionType::MAX) {
        // Set the values in the input tensor to the smallest element of that
        // type
        TypedAttr minAttr = getNumericLimit(rewriter, srcType.getElementType(),
                                            /*getMin=*/true);
        normalizationValue = arith::ConstantOp::create(rewriter, loc, minAttr);
      } else if (reduceEnum == torch_upstream::ReductionType::MIN) {
        // Set the values in the input tensor to the largest element of that
        // type
        TypedAttr maxAttr = getNumericLimit(rewriter, srcType.getElementType(),
                                            /*getMin=*/false);
        normalizationValue = arith::ConstantOp::create(rewriter, loc, maxAttr);
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
            TMTensor::YieldOp::create(b, loc, update);
          });
      if (reduceEnum == torch_upstream::ReductionType::MEAN) {
        counts = createTMTensorScatterOp(
            rewriter, loc, normalizations, indices, counts,
            /*dimensionsMap=*/createDefaultDimMap(indices),
            /*uniqueIndices=*/false,
            [&](OpBuilder &b, Location loc, Value update, Value current) {
              TMTensor::YieldOp::create(b, loc, update);
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
            if (isa<mlir::IntegerType>(update.getType())) {
              result = arith::AddIOp::create(b, loc, update, current);
            } else if (isa<mlir::FloatType>(update.getType())) {
              result = arith::AddFOp::create(b, loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          } else if (reduceEnum == torch_upstream::ReductionType::PROD) {
            if (isa<mlir::IntegerType>(update.getType())) {
              result = arith::MulIOp::create(b, loc, update, current);
            } else if (isa<mlir::FloatType>(update.getType())) {
              result = arith::MulFOp::create(b, loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          } else if (reduceEnum == torch_upstream::ReductionType::MAX) {
            if (isa<mlir::IntegerType>(update.getType())) {
              result = arith::MaxSIOp::create(b, loc, update, current);
            } else if (isa<mlir::FloatType>(update.getType())) {
              result = arith::MaximumFOp::create(b, loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          } else if (reduceEnum == torch_upstream::ReductionType::MIN) {
            if (isa<mlir::IntegerType>(update.getType())) {
              result = arith::MinSIOp::create(b, loc, update, current);
            } else if (isa<mlir::FloatType>(update.getType())) {
              result = arith::MinimumFOp::create(b, loc, update, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
          }
          TMTensor::YieldOp::create(b, loc, result);
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
              Value constantUpdate = arith::ConstantOp::create(
                  b, loc, b.getIntegerAttr(intType, 1));
              result = arith::AddIOp::create(b, loc, constantUpdate, current);
            } else if (mlir::FloatType floatType =
                           llvm::dyn_cast<mlir::FloatType>(current.getType())) {
              Value constantUpdate = arith::ConstantOp::create(
                  b, loc, b.getFloatAttr(floatType, 1.0));
              result = arith::AddFOp::create(b, loc, constantUpdate, current);
            } else {
              llvm_unreachable("Only integer/float types supported!");
            }
            TMTensor::YieldOp::create(b, loc, result);
          });

      Value output = tensor::EmptyOp::create(
          rewriter, loc, tensor::getMixedSizes(rewriter, loc, self),
          selfType.getElementType());

      // Finally divide the result
      scatterOp =
          linalg::MapOp::create(
              rewriter, loc, ValueRange{scatterOp, counts}, output,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                Value result;
                if (llvm::isa<mlir::IntegerType>(args[0].getType())) {
                  result = arith::DivSIOp::create(b, loc, args[0], args[1]);
                } else if (llvm::isa<mlir::FloatType>(args[0].getType())) {
                  result = arith::DivFOp::create(b, loc, args[0], args[1]);
                } else {
                  llvm_unreachable("Only integer/float types supported!");
                }
                linalg::YieldOp::create(b, loc, result);
              })
              .getResult()[0];
    }
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
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
    auto inputType = cast<RankedTensorType>(inputTensor.getType());
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
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, inputTensor, i));
      }
    }
    Value initEmptyTensor = tensor::EmptyOp::create(
        rewriter, loc, inputType.getShape(), rewriter.getI64Type(), dynDims);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::getMultiDimIdentityMap(inputRank, op.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
    Value indicesTensor =
        linalg::GenericOp::create(
            rewriter, loc, initEmptyTensor.getType(), ValueRange{},
            initEmptyTensor, indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value index = linalg::IndexOp::create(b, loc, dim);
              index = castIndexToInt64(b, loc, index);
              linalg::YieldOp::create(b, loc, index);
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
class ConvertAtenCumprodOp : public OpConversionPattern<AtenCumprodOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenCumprodOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value input = adaptor.getSelf();
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    Type elementType = resultType.getElementType();
    Type inputElementType =
        cast<RankedTensorType>(input.getType()).getElementType();

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
    if (!isa<Torch::NoneType>(dtype.getType()))
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
    Value output = createOneInitTensor(rewriter, loc, sizes, elementType);
    output = tensor::CastOp::create(rewriter, loc, resultType, output);

    SmallVector<Value> accSizes(sizes);
    accSizes.erase(accSizes.begin() + dim);
    SmallVector<int64_t> accStatic(
        makeShapeTorchCompatible(resultType.getShape()));
    accStatic.erase(accStatic.begin() + dim);
    Value acc = createOneInitTensor(rewriter, loc, accSizes, elementType);
    Type accType =
        RankedTensorType::get(makeShapeLLVMCompatible(accStatic), elementType);
    acc = tensor::CastOp::create(rewriter, loc, accType, acc);

    Value result = createTMTensorScanOp(
        rewriter, loc, input, output, acc, dim, /*inclusive=*/true,
        [](OpBuilder &b, Location loc, Value input, Value acc) {
          Value prod =
              (isa<mlir::FloatType>(input.getType())
                   ? arith::MulFOp::create(b, loc, input, acc)->getResult(0)
                   : arith::MulIOp::create(b, loc, input, acc)->getResult(0));
          TMTensor::YieldOp::create(b, loc, prod);
        });

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
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
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    Type elementType = resultType.getElementType();
    auto inputType = cast<RankedTensorType>(input.getType());
    Type inputElementType = inputType.getElementType();

    Value dtype = op.getDtype();
    if (!isa<Torch::NoneType>(dtype.getType())) {
      int64_t dtypeInt;
      if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only constant int dtype value is supported");

      FailureOr<Type> resDtype = getTypeForScalarType(
          op->getContext(), (torch_upstream::ScalarType)dtypeInt);
      if (failed(resDtype))
        return rewriter.notifyMatchFailure(
            op, "unsupported: dtype not defined for the given dtype int value");

      Value torchInput =
          convertTensorToDtype(rewriter, loc, op.getSelf(), resDtype.value());
      input = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(torchInput.getType()),
          torchInput);
    } else if (elementType != inputElementType &&
               isa<mlir::IntegerType>(elementType) &&
               isa<mlir::IntegerType>(inputElementType)) {
      // Converting the input element type to the result's element type.
      // The only possible mismatch would be when the input element type is an
      // integer but not `si64` and the `dtype` is not specified. Therefore, we
      // directly convert the input to `si64`. Rest all cases are handled in the
      // dtype definition for this op.
      Value torchInput = convertTensorToDtype(
          rewriter, loc, op.getSelf(),
          rewriter.getIntegerType(64, IntegerType::Signed));
      input = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(torchInput.getType()),
          torchInput);
    }

    int64_t inputRank = inputType.getRank();
    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant dim value is supported");
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      return rewriter.notifyMatchFailure(op, "invalid dim");

    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, input);
    Value output = createZeroInitTensor(rewriter, loc, sizes, elementType);
    output = tensor::CastOp::create(rewriter, loc, resultType, output);

    SmallVector<Value> accSizes(sizes);
    accSizes.erase(accSizes.begin() + dim);
    SmallVector<int64_t> accStatic(
        makeShapeTorchCompatible(resultType.getShape()));
    accStatic.erase(accStatic.begin() + dim);
    Value acc = createZeroInitTensor(rewriter, loc, accSizes, elementType);
    Type accType =
        RankedTensorType::get(makeShapeLLVMCompatible(accStatic), elementType);
    acc = tensor::CastOp::create(rewriter, loc, accType, acc);

    Value result = createTMTensorScanOp(
        rewriter, loc, input, output, acc, dim, /*inclusive=*/true,
        [](OpBuilder &b, Location loc, Value input, Value acc) {
          Value sum =
              (isa<mlir::FloatType>(input.getType())
                   ? arith::AddFOp::create(b, loc, input, acc)->getResult(0)
                   : arith::AddIOp::create(b, loc, input, acc)->getResult(0));
          TMTensor::YieldOp::create(b, loc, sum);
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

  static LogicalResult
  preProcessGroupQueryAttentionInput(AtenScaledDotProductAttentionOp op,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter *typeConverter,
                                     Value query, Value &key, Value &value) {
    auto queryTy = cast<ShapedType>(query.getType());
    auto valueTy = cast<ShapedType>(value.getType());
    auto keyTy = cast<ShapedType>(key.getType());

    int64_t rank = queryTy.getRank();

    int64_t qNumHeads = queryTy.getDimSize(rank - 3);
    int64_t kNumHeads = valueTy.getDimSize(rank - 3);
    int64_t vNumHeads = keyTy.getDimSize(rank - 3);

    if (llvm::any_of(llvm::ArrayRef<int64_t>{qNumHeads, kNumHeads, vNumHeads},
                     [](int64_t d) { return d == Torch::kUnknownSize; })) {
      return llvm::failure();
    }

    if (llvm::all_equal(
            llvm::ArrayRef<int64_t>{qNumHeads, kNumHeads, vNumHeads}))
      return llvm::success();

    if ((qNumHeads % kNumHeads) && (qNumHeads % vNumHeads))
      return llvm::failure();

    int64_t repeatKeyShape = qNumHeads / kNumHeads;
    int64_t repeatValueShape = qNumHeads / vNumHeads;

    Location loc = op.getLoc();
    FailureOr<Value> keyRepeated = repeatTensorElementsForDim(
        op.getOperation(), rewriter, /*resType=*/op.getQuery().getType(),
        op.getKey(),
        /*repeats=*/repeatKeyShape, /*dim=*/rank - 3);
    if (failed(keyRepeated))
      return rewriter.notifyMatchFailure(
          loc, "Failed to repeat the tensor elements for key.");

    FailureOr<Value> valueRepeated = repeatTensorElementsForDim(
        op.getOperation(), rewriter, /*resType=*/op.getQuery().getType(),
        op.getValue(),
        /*repeats=*/repeatValueShape, /*dim=*/rank - 3);
    if (failed(valueRepeated))
      return rewriter.notifyMatchFailure(
          loc, "Failed to repeat the tensor elements for value.");

    key = typeConverter->materializeTargetConversion(
        rewriter, loc,
        typeConverter->convertType(keyRepeated.value().getType()),
        keyRepeated.value());
    value = typeConverter->materializeTargetConversion(
        rewriter, loc,
        typeConverter->convertType(valueRepeated.value().getType()),
        valueRepeated.value());
    return success();
  }

  LogicalResult
  matchAndRewrite(AtenScaledDotProductAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto opTy = cast<ValueTensorType>(op.getType()).toBuiltinTensor();
    auto query = adaptor.getQuery();
    auto value = adaptor.getValue();
    auto key = adaptor.getKey();
    auto mask = adaptor.getAttnMask();
    auto queryTy = cast<ShapedType>(query.getType());
    auto valueTy = cast<ShapedType>(value.getType());
    auto keyTy = cast<ShapedType>(key.getType());

    auto loc = op.getLoc();
    Value dropoutP = op.getDropoutP();
    Value isCausal = op.getIsCausal();
    Value scale = op.getScale();
    Value enableGQA = op.getEnableGqa();
    Type elementType =
        cast<ShapedType>(adaptor.getQuery().getType()).getElementType();

    double dropout;
    if (!matchPattern(dropoutP, m_TorchConstantFloat(&dropout)) ||
        dropout > 0.0)
      return rewriter.notifyMatchFailure(loc, "dropout not supported");

    bool causal;
    if (!matchPattern(isCausal, m_TorchConstantBool(&causal)) || causal) {
      if (!isa<Torch::NoneType>(mask.getType())) {
        return rewriter.notifyMatchFailure(
            loc, "expected no attention mask when isCausal is true");
      }

      SmallVector<int64_t> maskStatic;
      SmallVector<Value> maskDyn;
      for (int i = 0, s = queryTy.getRank() - 1; i < s; ++i) {
        maskStatic.push_back(queryTy.getDimSize(i));
        if (maskStatic.back() == ShapedType::kDynamic)
          maskDyn.push_back(tensor::DimOp::create(rewriter, loc, query, i));
      }

      maskStatic.push_back(keyTy.getDimSize(keyTy.getRank() - 2));
      if (maskStatic.back() == ShapedType::kDynamic)
        maskDyn.push_back(
            tensor::DimOp::create(rewriter, loc, key, keyTy.getRank() - 2));

      Type maskType = getElementTypeOrSelf(queryTy);
      Value emptyMask =
          tensor::EmptyOp::create(rewriter, loc, maskStatic, maskType, maskDyn);

      Value zero = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(getElementTypeOrSelf(maskType), 0.0));
      Value negInf = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(getElementTypeOrSelf(maskType), -INFINITY));

      mask =
          linalg::FillOp::create(rewriter, loc, zero, emptyMask).getResult(0);

      int64_t rank = cast<ShapedType>(queryTy).getRank();
      AffineMap maskMap = rewriter.getMultiDimIdentityMap(rank);
      SmallVector<utils::IteratorType> iteratorTypes(
          rank, utils::IteratorType::parallel);
      auto genericOp = linalg::GenericOp::create(
          rewriter, loc, mask.getType(), ValueRange{}, mask,
          SmallVector<AffineMap>{maskMap}, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value i = linalg::IndexOp::create(b, loc, queryTy.getRank() - 2);
            Value j = linalg::IndexOp::create(b, loc, queryTy.getRank() - 1);

            Value cond =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, i, j);
            Value select = arith::SelectOp::create(b, loc, cond, zero, negInf);
            linalg::YieldOp::create(b, loc, select);
          });
      mask = genericOp.getResult(0);
    }

    // Broadcast the batch dimensions of the mask:
    if (!isa<Torch::NoneType>(mask.getType())) {
      auto maskTy = cast<RankedTensorType>(mask.getType());
      int64_t rank = maskTy.getRank();
      bool needsBroadcast = false;
      for (int i = 0, s = rank - 2; i < s; ++i) {
        needsBroadcast |= maskTy.getDimSize(i) != queryTy.getDimSize(i);
      }

      if (needsBroadcast) {
        SmallVector<int64_t> maskShape;
        SmallVector<Value> maskDynDims;

        SmallVector<AffineExpr> maskExprs;
        for (int i = 0, s = rank - 2; i < s; ++i) {
          maskShape.push_back(queryTy.getDimSize(i));

          if (maskTy.getDimSize(i) != queryTy.getDimSize(i)) {
            maskExprs.push_back(rewriter.getAffineConstantExpr(0));
          } else {
            maskExprs.push_back(rewriter.getAffineDimExpr(i));
          }

          if (queryTy.isDynamicDim(i)) {
            maskDynDims.push_back(
                tensor::DimOp::create(rewriter, loc, query, i));
          }
        }

        maskExprs.push_back(rewriter.getAffineDimExpr(rank - 2));
        maskExprs.push_back(rewriter.getAffineDimExpr(rank - 1));
        maskShape.push_back(maskTy.getDimSize(rank - 2));
        maskShape.push_back(maskTy.getDimSize(rank - 1));
        if (maskTy.isDynamicDim(rank - 2))
          maskDynDims.push_back(
              tensor::DimOp::create(rewriter, loc, mask, rank - 2));
        if (maskTy.isDynamicDim(rank - 1))
          maskDynDims.push_back(
              tensor::DimOp::create(rewriter, loc, mask, rank - 1));

        SmallVector<AffineMap> affineMaps = {
            AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, maskExprs,
                           op.getContext()),
            rewriter.getMultiDimIdentityMap(rank)};
        SmallVector<utils::IteratorType> findMaxIteratorTypes(
            rank, utils::IteratorType::parallel);

        Value emptyMask = tensor::EmptyOp::create(
            rewriter, loc, maskShape, maskTy.getElementType(), maskDynDims);
        Value newMask =
            linalg::GenericOp::create(
                rewriter, loc, emptyMask.getType(), mask,
                ValueRange({emptyMask}), affineMaps, findMaxIteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  linalg::YieldOp::create(b, loc, args[0]);
                })
                .getResult(0);
        mask = newMask;
      }
    }

    // Verify the scale matches the expected 1/sqrt(headDim).
    // See:
    // https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if (!isa<Torch::NoneType>(scale.getType())) {
      double scaleFloat;
      if (!matchPattern(scale, m_TorchConstantFloat(&scaleFloat)))
        return rewriter.notifyMatchFailure(loc, "scale must be a constant");

      int64_t headDim = queryTy.getDimSize(queryTy.getRank() - 1);
      if (headDim == ShapedType::kDynamic) {
        // With dynamic head dimension, we cannot verify the scale matches
        // 1/sqrt(headDim).
        return rewriter.notifyMatchFailure(
            loc, "cannot verify scale with dynamic head dimension; use "
                 "scale=None or use static head dimension");
      }
      double expectedScale = 1.0 / std::sqrt(static_cast<double>(headDim));
      // Use relative tolerance for floating point comparison to handle
      // varying magnitudes across different head dimensions consistently.
      // 1e-6 relative tolerance is ~10x float32 machine epsilon, which
      // provides a safe margin for:
      // - Different computation orders (a*b vs b*a can differ slightly)
      // - Float64 -> float32 -> float64 round-trips through serialization
      double relativeError =
          std::abs(scaleFloat - expectedScale) / expectedScale;
      if (relativeError > 1e-6) {
        return rewriter.notifyMatchFailure(
            loc, "scale must be None or 1/sqrt(headDim)");
      }
    }

    if (queryTy.getRank() != valueTy.getRank() ||
        queryTy.getRank() != keyTy.getRank())
      return rewriter.notifyMatchFailure(op, "operand ranks do not match");

    if (queryTy.getRank() < 3)
      return rewriter.notifyMatchFailure(op, "missing batch dimension");

    bool isGQAEnabled;
    if (!matchPattern(enableGQA, m_TorchConstantBool(&isGQAEnabled)))
      return rewriter.notifyMatchFailure(
          loc, "Expected enable_gqa flag to be constant bool");

    // For the cases when `enable_gqa` flag is set to true, we have to
    // pre-process the inputs specifically key and value by repeating the
    // elements for the head dim.
    // The reference code is available here:
    // https://github.com/pytorch/pytorch/pull/132689/files#diff-e726853e9795dfb6c74ab1e10945f5d5f24540eb7bc633e5c76f69bc258f24d6R612
    if (enableGQA) {
      if (failed(preProcessGroupQueryAttentionInput(
              op, rewriter, getTypeConverter(), query, key, value)))
        return failure();
    }

    llvm::SmallVector<ReassociationIndices, 3> reassociation(3);
    for (int i = 0, s = valueTy.getRank() - 2; i < s; ++i)
      reassociation.front().push_back(i);
    reassociation[1].push_back(valueTy.getRank() - 2);
    reassociation[2].push_back(valueTy.getRank() - 1);

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
      return tensor::CollapseShapeOp::create(rewriter, loc, collapseTy, value,
                                             reassociation);
    };

    query = collapseBatch(query);
    key = collapseBatch(key);
    value = collapseBatch(value);
    if (!isa<mlir::torch::Torch::NoneType>(mask.getType())) {
      mask = collapseBatch(mask);
    }

    SmallVector<int64_t> outSizes(cast<ShapedType>(query.getType()).getShape());
    SmallVector<int64_t> valueSizes(
        cast<ShapedType>(value.getType()).getShape());
    outSizes[outSizes.size() - 1] = valueSizes[valueSizes.size() - 1];
    SmallVector<Value> outSizesDynamic(getTensorSizes(rewriter, loc, query));
    outSizesDynamic[outSizesDynamic.size() - 1] =
        getTensorSizes(rewriter, loc, value)[valueSizes.size() - 1];
    Type outType = RankedTensorType::get(outSizes, elementType);
    Value output =
        createZeroInitTensor(rewriter, loc, outSizesDynamic, elementType);

    SmallVector<Value> inputs = SmallVector<Value>{query, key, value};

    if (!isa<mlir::torch::Torch::NoneType>(mask.getType())) {
      inputs.push_back(mask);
    }

    // Overwrite with tm_tensor::attention
    Value attention = AttentionOp::create(rewriter, loc, outType, inputs,
                                          SmallVector<Value>{output})
                          .getResult()[0];

    if (opTy != outType) {
      attention = tensor::ExpandShapeOp::create(rewriter, loc, opTy, attention,
                                                reassociation);
    }

    rewriter.replaceOp(op, attention);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenKthvalueOp : public OpConversionPattern<AtenKthvalueOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenKthvalueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const llvm::StringRef opName = op->getName().getStringRef();

    Location loc = op.getLoc();
    auto typec = this->getTypeConverter();

    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    unsigned inputRank = inputType.getRank();
    Type inputElementType = inputType.getElementType();

    auto valResultType =
        cast<RankedTensorType>(typec->convertType(op.getResult(0).getType()));
    auto valResultElementType =
        getElementTypeOrSelf(typec->convertType(valResultType));

    auto idxResultType =
        cast<RankedTensorType>(typec->convertType(op.getResult(1).getType()));
    auto idxResultElementType =
        getElementTypeOrSelf(typec->convertType(idxResultType));

    // get keepdim and check it is bool
    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(
          op, opName + " requires boolean value for keepdim");

    // get dim, check it is constant int
    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant dim value is supported");

    // turn dim into positive if negative, and check it is in the valid range
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank)) {
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");
    }

    // get k, check it is a constant int
    int64_t k;
    if (!matchPattern(op.getK(), m_TorchConstantInt(&k)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only constant k value is supported");

    // check if element type is float, int, or unsigned
    bool isUnsigned = false;
    if (!isa<mlir::FloatType>(inputElementType)) {
      if (!isa<mlir::IntegerType>(inputElementType)) {
        return rewriter.notifyMatchFailure(
            op, opName + " to linalg.* requires Float or Integer "
                         "input element type");
      }

      auto integerTy = dyn_cast<mlir::IntegerType>(
          cast<BaseTensorType>(op.getSelf().getType()).getDtype());
      isUnsigned = integerTy.isUnsigned();
    }

    // Create the values to fill initial output tensors for
    // topk op and linalg generic op for finding max value.
    Value fillValLinalgFindMax;
    Value fillValTopK;
    if (isa<mlir::FloatType>(inputElementType)) {
      // max float for topk tensor
      fillValTopK = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(
              inputElementType,
              APFloat::getInf(
                  cast<mlir::FloatType>(inputElementType).getFloatSemantics(),
                  /*Negative=*/false)));
      // min float for linalg generic op tensor
      fillValLinalgFindMax = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(
              inputElementType,
              APFloat::getInf(
                  cast<mlir::FloatType>(inputElementType).getFloatSemantics(),
                  /*Negative=*/true)));
    } else if (!isUnsigned) {
      auto width = cast<mlir::IntegerType>(inputElementType).getWidth();
      // max signed int for topk op tensor
      auto init = APSInt::getSignedMaxValue(width);
      fillValTopK = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(inputElementType, init));
      // min signed int for linalg generic op tensor
      init = APSInt::getSignedMinValue(width);
      fillValLinalgFindMax = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(inputElementType, init));
    } else if (isUnsigned) {
      auto width = cast<mlir::IntegerType>(inputElementType).getWidth();
      // max unsigned int for topk op tensor
      auto init = APInt::getMaxValue(width);
      fillValTopK = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(inputElementType, init));
      // min unsigned int for linalg generic op tensor
      init = APInt::getMinValue(width);
      fillValLinalgFindMax = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(inputElementType, init));
    }

    auto i32Type = rewriter.getI32Type();

    // ======== BEGIN: Topk op section ========
    // Based on iree docs:
    // https://iree.dev/reference/mlir-dialects/LinalgExt/#iree_linalg_extsort-linalgextsortop

    // Create the output shape of topk op.
    // For every dimension, topkShape[dimension] = inputShape[dimension],
    // except topkShape[dim] = k.
    SmallVector<Value> topkShape;
    for (unsigned i = 0; i < inputRank; i++) {
      auto currentDimSize = tensor::DimOp::create(rewriter, loc, input, i);
      topkShape.push_back(currentDimSize);
    }
    auto dimSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(rewriter.getI64Type(), k));
    topkShape[dim] = dimSize;

    // Fill the initial topk op output tensor.
    Value topkOutputVal = createInitTensor(rewriter, loc, topkShape,
                                           valResultElementType, fillValTopK);

    // Create the initial value to fill the topk output indices tensor.
    // It is equal to the max 32-bit signless integer.
    auto signlessType = mlir::IntegerType::get(op.getContext(), 32,
                                               mlir::IntegerType::Signless);
    auto initIdx = getNumericLimit(rewriter, signlessType, /*getMin=*/false);
    auto fillValTopkIdx = arith::ConstantOp::create(rewriter, loc, initIdx);
    // Fill the initial topk op output indices tensor.
    Value topkOutputIdx =
        createInitTensor(rewriter, loc, topkShape, i32Type, fillValTopkIdx);

    // Input arguments for topk op contain only the input tensor.
    // Input indices will be inferred based on input shape.
    // (See docs link above).
    SmallVector<Value> topkInputs;
    topkInputs.push_back(input);

    // Outputs contain both the values and the indices tensors.
    SmallVector<Value> topkOutputs;
    topkOutputs.push_back(topkOutputVal);
    topkOutputs.push_back(topkOutputIdx);

    // Element types of the arguments passed to the topk op region.
    // The region accepts the next value N, and the current output
    // candidate K (see docs link above).
    // Both N and K are values from the input tensors, thus the
    // element types are the same and are taken from inputType.
    SmallVector<Type> topkElementTypes;
    topkElementTypes.push_back(inputType.getElementType());
    topkElementTypes.push_back(inputType.getElementType());

    // Create the TMTensor TopkOp.
    FailureOr<SmallVector<Value>> topkOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      topkOp = createTMTensorTopkOp(rewriter, loc, topkInputs, topkOutputs,
                                    topkElementTypes, dim, /*isMinK=*/true);
    }
    // Topk op creation fails with invalid element types.
    if (failed(topkOp))
      return rewriter.notifyMatchFailure(
          loc, "Only Integer and Floating element type expected.");

    auto topkOpVal = topkOp.value();
    // ======== END: Topk op section ========

    // ======== BEGIN: Linalg generic to find max in topk result ========

    // Create result shape as both a vector of Value and of int64_t types.
    // We assume that keepdim is false, and fix the result later if true.
    // Result shape is equal to inputShape, with dim dimension removed.
    SmallVector<Value> resultShape;
    SmallVector<int64_t> resultShapeInt;
    for (int64_t i = 0; i < inputType.getRank(); i++) {
      if (dim != i) {
        auto currentDimSize = tensor::DimOp::create(rewriter, loc, input, i);
        resultShape.push_back(currentDimSize);
        resultShapeInt.push_back(inputType.getShape()[i]);
      }
    }

    // Fill the initial output tensor for linalg op for finding max value.
    Value findMaxOutputVal = createInitTensor(
        rewriter, loc, resultShape, inputElementType, fillValLinalgFindMax);

    // Fill the initial output indices tensor for linalg op for finding max
    // value with zeros.
    Value findMaxOutputIdx =
        createZeroInitTensor(rewriter, loc, resultShape, idxResultElementType);

    // Reduce along dim.
    SmallVector<utils::IteratorType> findMaxIteratorTypes(
        inputType.getRank(), utils::IteratorType::parallel);
    findMaxIteratorTypes[dim] = utils::IteratorType::reduction;

    SmallVector<AffineExpr> findMaxMapExprs;
    SmallVector<AffineExpr> findMaxMapResultExprs;
    for (auto size :
         llvm::enumerate(makeShapeTorchCompatible(inputType.getShape()))) {
      findMaxMapExprs.push_back(rewriter.getAffineDimExpr(size.index()));
      if (unsigned(dim) != size.index())
        findMaxMapResultExprs.push_back(
            rewriter.getAffineDimExpr(size.index()));
    }

    auto findMaxMaps = AffineMap::inferFromExprList(
        {findMaxMapExprs, findMaxMapResultExprs, findMaxMapResultExprs},
        rewriter.getContext());

    // Create linalg op for finding the max value in the extracted topk values.
    auto findMaxLinalg = linalg::GenericOp::create(
        rewriter, loc,
        ArrayRef<Type>(
            {findMaxOutputVal.getType(), findMaxOutputIdx.getType()}),
        topkOpVal.front(), ValueRange({findMaxOutputVal, findMaxOutputIdx}),
        findMaxMaps, findMaxIteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          // Linalg generic body is the same as the decomposition for
          // AtenMinDim: lib/Conversion/TorchToLinalg/Reduction.cpp

          Value newValue = blockArgs[0];
          Value oldValue = blockArgs[1];
          Value oldIndex = blockArgs[2];

          Value newIndex = arith::IndexCastOp::create(
              rewriter, nestedLoc, oldIndex.getType(),
              linalg::IndexOp::create(rewriter, nestedLoc, dim));

          Value resultVal, predicate;
          if (isa<mlir::FloatType>(inputElementType)) {
            resultVal = arith::MaximumFOp::create(rewriter, nestedLoc, newValue,
                                                  oldValue);
            predicate = arith::CmpFOp::create(rewriter, nestedLoc,
                                              arith::CmpFPredicate::OGT,
                                              newValue, oldValue);
          } else {
            arith::CmpIPredicate predType;
            predType = isUnsigned ? arith::CmpIPredicate::ugt
                                  : arith::CmpIPredicate::sgt;
            if (isUnsigned) {
              resultVal = arith::MaxUIOp::create(rewriter, nestedLoc, newValue,
                                                 oldValue);
            } else {
              resultVal = arith::MaxSIOp::create(rewriter, nestedLoc, newValue,
                                                 oldValue);
            }
            predicate = arith::CmpIOp::create(rewriter, nestedLoc, predType,
                                              newValue, oldValue);
          }
          auto resultIndex = arith::SelectOp::create(
              rewriter, nestedLoc, predicate, newIndex, oldIndex);
          linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                  ValueRange{resultVal, resultIndex});
        });

    auto findMaxVal = findMaxLinalg.getResult(0);
    auto findMaxIdx = findMaxLinalg.getResult(1);
    auto findMaxIdxType = cast<RankedTensorType>(findMaxIdx.getType());

    // ======== END: Linalg generic to find max in topk result ========

    // ======== BEGIN: Linalg generic for index extraction ========
    // The linalg op for finding max returned idx of max elements in the
    // tensor returned by the topk op. We need the idx of those elements
    // in the original input. The topk op returned the idx of the top k
    // extracted elements in the original input. Using the linalg idx
    // results to index the topk idx results returns the idx of kth
    // max value in the original input. Example:
    // input = [1, 7, 3, 6, 2, 8, 9, 5], k = 4
    // topk_val = [1, 3, 2, 5], topk_idx = [0, 2, 4, 7]
    // linalg_max_val = [5], linalg_max_idx = [3] (5 is at idx 3 in topk_val)
    // index the topk_idx using linalg_max_idx -> topk_idx[3] = 7
    // kth_val = [5], kth_idx = [7]

    // Create a tensor for the resulting idx.
    Value filledTensorExtractedIdx = createZeroInitTensor(
        rewriter, loc, getTensorSizes(rewriter, loc, findMaxIdx), i32Type);

    // We iterate through the idx tensor returned by the linalg generic op for
    // finding max.
    SmallVector<utils::IteratorType> extractedIdxIteratorTypes(
        findMaxIdxType.getRank(), utils::IteratorType::parallel);

    SmallVector<AffineExpr> extractedIdxMapExprs;
    for (auto size :
         llvm::enumerate(makeShapeTorchCompatible(findMaxIdxType.getShape()))) {
      extractedIdxMapExprs.push_back(rewriter.getAffineDimExpr(size.index()));
    }

    auto extractedIdxMaps = AffineMap::inferFromExprList(
        {extractedIdxMapExprs, extractedIdxMapExprs}, rewriter.getContext());

    // Linalg generic op for indexing the topk output idx tensor using
    // the idx tensor returned by the linalg generic op for finding max.
    // Only the idx tensor from the linalg generic op is sent as input.
    auto extractedIdxLinalg = linalg::GenericOp::create(
        rewriter, loc, ArrayRef<Type>({filledTensorExtractedIdx.getType()}),
        findMaxIdx, filledTensorExtractedIdx, extractedIdxMaps,
        extractedIdxIteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          // Get the current input idx.
          Value index = arith::IndexCastOp::create(
              rewriter, loc, rewriter.getIndexType(), blockArgs[0]);

          // Create idx to index the topk idx tensor.
          // Index the dim dimension using the current input idx.
          SmallVector<Value> indexTarget;
          for (unsigned i = 0; i < dim; i++)
            indexTarget.push_back(linalg::IndexOp::create(rewriter, loc, i));
          indexTarget.push_back(index);
          for (unsigned i = dim; i < findMaxIdxType.getRank(); i++)
            indexTarget.push_back(linalg::IndexOp::create(rewriter, loc, i));

          // Extract the element from the topk idx tensor.
          Value extractedElement = tensor::ExtractOp::create(
              rewriter, loc, topkOpVal.back(), indexTarget);
          linalg::YieldOp::create(rewriter, loc, extractedElement);
        });

    auto extractedIdx = extractedIdxLinalg.getResult(0);
    auto extractedIdxType = cast<RankedTensorType>(extractedIdx.getType());

    // ======== END: Linalg generic for index extraction ========

    // ======== BEGIN: Linalg generic for topk idx cast ========
    // Casts from i32 to idx result type of the Kthvalue op.

    // Create the initial tensor for the cast result.
    Value filledTensorCastedIdx = createZeroInitTensor(
        rewriter, loc, getTensorSizes(rewriter, loc, extractedIdx),
        idxResultElementType);

    SmallVector<utils::IteratorType> castedIdxIteratorTypes(
        extractedIdxType.getRank(), utils::IteratorType::parallel);

    SmallVector<AffineExpr> castedIdxMapExprs;
    for (auto size : llvm::enumerate(
             makeShapeTorchCompatible(extractedIdxType.getShape()))) {
      castedIdxMapExprs.push_back(rewriter.getAffineDimExpr(size.index()));
    }

    auto castedIdxMaps = AffineMap::inferFromExprList(
        {castedIdxMapExprs, castedIdxMapExprs}, rewriter.getContext());

    // Linalg generic op for casting topk idx output tensor elements from i32 to
    // result idx tensor element type.
    auto castedIdxLinalg = linalg::GenericOp::create(
        rewriter, loc, ArrayRef<Type>({filledTensorCastedIdx.getType()}),
        extractedIdx, filledTensorCastedIdx, castedIdxMaps,
        castedIdxIteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value oldIdx = blockArgs[0];

          // Cast from i32 to index.
          Value oldIdxToIndexType = arith::IndexCastOp::create(
              rewriter, nestedLoc, rewriter.getIndexType(), oldIdx);
          // Cast from index to result idx element type.
          Value resultIdx = arith::IndexCastOp::create(
              rewriter, nestedLoc, idxResultElementType, oldIdxToIndexType);

          linalg::YieldOp::create(nestedBuilder, nestedLoc, resultIdx);
        });

    auto castedIdx = castedIdxLinalg.getResult(0);

    // ======== END: Linalg generic for topk idx cast ========

    // Create output value type ("squeezed" since we assume keepdim=False).
    auto topkValResultType =
        cast<RankedTensorType>(topkOpVal.front().getType());
    auto squeezedValType = topkValResultType.cloneWith(
        resultShapeInt,
        cast<RankedTensorType>(findMaxVal.getType()).getElementType());

    // Create output idx type ("squeezed" since we assume keepdim=False).
    auto castedIdxType = cast<RankedTensorType>(castedIdx.getType());
    auto squeezedIdxType = castedIdxType.cloneWith(
        resultShapeInt, findMaxIdxType.getElementType());

    if (!keepDim) {
      // If keepdim=false, cast the the outputs to appropriate type and return.
      Value retVal =
          tensor::CastOp::create(rewriter, loc, squeezedValType, findMaxVal);
      Value retIdx =
          tensor::CastOp::create(rewriter, loc, squeezedIdxType, castedIdx);
      llvm::SmallVector<Value> res{retVal, retIdx};
      rewriter.replaceOp(op, res);
      return success();
    }

    // If keepdim is false, unsqueeze.
    // Unsqueezing implementation taken from AteMinMaxDimOp lowering:
    // lib/Conversion/TorchToLinalg/Reduction.cpp
    llvm::SmallVector<int64_t> valShape(valResultType.getShape());
    llvm::SmallVector<int64_t> idxShape(idxResultType.getShape());
    for (int i = dim, s = valShape.size() - 1; i < s; ++i) {
      valShape[i] = valShape[i + 1];
      idxShape[i] = idxShape[i + 1];
    }

    valShape.resize(valShape.size() - 1);
    idxShape.resize(idxShape.size() - 1);

    Value retVal =
        tensor::CastOp::create(rewriter, loc, squeezedValType.clone(valShape),
                               findMaxLinalg.getResult(0));
    Value retIdx = tensor::CastOp::create(
        rewriter, loc, squeezedIdxType.clone(idxShape), castedIdx);

    SmallVector<ReassociationIndices> reassociation(valShape.size());
    if (reassociation.size() > 0) {
      for (int i = 0; i < dim; ++i)
        reassociation[i].push_back(i);
      reassociation[std::max<int64_t>(0, dim - 1)].push_back(dim);
      for (int i = dim, s = reassociation.size(); i < s; ++i)
        reassociation[i].push_back(i + 1);
    }

    valShape.push_back(0);
    idxShape.push_back(0);
    for (int i = dim, s = valShape.size() - 1; i < s; ++i) {
      valShape[i + 1] = valShape[i];
      idxShape[i + 1] = idxShape[i];
    }

    valShape[dim] = 1;
    idxShape[dim] = 1;

    Value unsqueezeVal = tensor::ExpandShapeOp::create(
        rewriter, loc, valResultType, retVal, reassociation);

    Value unsqueezeIdx = tensor::ExpandShapeOp::create(
        rewriter, loc, idxResultType, retIdx, reassociation);

    // Return unsqueezed.
    llvm::SmallVector<Value> unsqueezes = {unsqueezeVal, unsqueezeIdx};
    rewriter.replaceOp(op, unsqueezes);
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTMTensor
    : public impl::ConvertTorchToTMTensorBase<ConvertTorchToTMTensor> {
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
    target.addIllegalOp<AtenIndexPutHackedTwinOp>();
    patterns.add<ConvertAtenIndexPutHackedTwinOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxPool2dWithIndicesBackwardOp>();
    patterns.add<ConvertAtenMaxPool2dWithIndicesBackwardOp>(typeConverter,
                                                            context);
    target.addIllegalOp<AtenScatterReduceTwoOp>();
    patterns.add<ConvertAtenScatterReduceTwoOp>(typeConverter, context);
    target.addIllegalOp<AtenSortOp>();
    patterns.add<ConvertAtenSortOp>(typeConverter, context);
    target.addIllegalOp<AtenCumsumOp>();
    patterns.add<ConvertAtenCumsumOp>(typeConverter, context);
    target.addIllegalOp<AtenCumprodOp>();
    patterns.add<ConvertAtenCumprodOp>(typeConverter, context);
    target.addIllegalOp<AtenScaledDotProductAttentionOp>();
    patterns.add<ConvertAtenScaledDotProductAttentionOp>(typeConverter,
                                                         context);

    target.addIllegalOp<AtenScatterSrcOp>();
    patterns.add<ConvertAtenScatterOp<AtenScatterSrcOp>>(typeConverter,
                                                         context);
    target.addIllegalOp<AtenScatterAddOp>();
    patterns.add<ConvertAtenScatterOp<AtenScatterAddOp>>(typeConverter,
                                                         context);
    target.addIllegalOp<AtenKthvalueOp>();
    patterns.add<ConvertAtenKthvalueOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToTMTensorPass() {
  return std::make_unique<ConvertTorchToTMTensor>();
}

} // namespace mlir::torch
