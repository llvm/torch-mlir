//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

// Simple rewrites for the default domain.
// See: https://onnx.ai/onnx/operators/
// For operators that are effectively version invariant, we register with
// sinceVersion==1. We interpret this to include the following spec
// diffs that are irrelevant to this level of lowering:
//   * Supported element types.
//   * Limited broadcasting to full broadcasting support.
//
// There are a lot of spec revisions that basically generalized elementwise
// to be more normal and a direct translation vs a special case. This
// results in a lot of ONNX test cases that all reduce to the exact same
// thing here, so we simplify.

// utilities
namespace {
// In case the ReduceSum Op was not the first operation performed on the data,
// we provide the original operand through storeResult, which will be modified
// if the result will be passed onto another operation, and will be used for
// noop_with_empty_axes handling before that.
LogicalResult reducedSumImpl(OpBinder binder,
                             ConversionPatternRewriter &rewriter, Value data,
                             Torch::ValueTensorType resultType,
                             Value &storeResult, int64_t keepDims,
                             int64_t noop_with_empty_axes,
                             bool isIntermediateOp) {

  SmallVector<Value> axesList;
  Value axesVal;
  if (!binder.tensorOperandAtIndex(axesVal, 1)) {
    auto inputType = dyn_cast<Torch::ValueTensorType>(data.getType());
    if (!inputType.hasSizes() || !resultType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          binder.op, "unimplemented: expected input and result to have shapes");
    }

    if (inputType.areAllSizesKnown() && resultType.areAllSizesKnown()) {
      SmallVector<int64_t> inputShape{inputType.getSizes()};
      SmallVector<int64_t> resultShape{resultType.getSizes()};
      // if the shapes are equal, none of the dims is reduced
      if (llvm::equal(inputShape, resultShape)) {
        // simply fill in the op and return
        rewriter.replaceOp(binder.op, data);
        return success();
      }
      if (areAllElementsDistinct(inputShape)) {
        // The check for the input shape elements to be distinct is added
        // for the cases like:
        // Input: [3, 2, 2] -> Output: [3, 2]
        // For the above case, from the input and output shape it can't be
        // inferred whether the dim:1 is reduced or dim:2. To avoid these
        // type of cases, the check has been placed.
        SmallVector<int64_t> reduceDims;
        unsigned resultShapeCounter = 0;
        for (unsigned i = 0; i < inputShape.size(); i++) {
          if (resultShapeCounter < resultShape.size() &&
              inputShape[i] == resultShape[resultShapeCounter]) {
            resultShapeCounter++;
          } else {
            reduceDims.push_back(i);
            if (resultShapeCounter < resultShape.size() &&
                resultShape[resultShapeCounter] == 1)
              resultShapeCounter++;
          }
        }
        for (auto i : reduceDims) {
          axesList.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
      }
    }
    if (axesList.empty()) {
      Torch::BaseTensorType axesType =
          cast<Torch::BaseTensorType>(axesVal.getType());
      auto axesTy = dyn_cast<Torch::ValueTensorType>(axesVal.getType());
      auto axesShape = axesTy.getSizes();
      if (axesShape.size() != 1 || axesShape[0] == Torch::kUnknownSize)
        return failure();

      Value zero = rewriter.create<Torch::ConstantIntOp>(
          binder.getLoc(), rewriter.getType<Torch::IntType>(),
          rewriter.getI64IntegerAttr(0));
      SmallVector<int64_t> selectSizes{1};
      auto selType = rewriter.getType<Torch::ValueTensorType>(
          selectSizes, axesType.getOptionalDtype());
      int64_t numAxes = axesShape[0];
      for (int64_t i = 0; i < numAxes; ++i) {
        Value iv = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(i));
        Value extract = rewriter.create<Torch::AtenSelectIntOp>(
            binder.getLoc(), selType, axesVal, zero, iv);
        Value dim = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
        axesList.push_back(dim);
      }
    }
  }

  SmallVector<int64_t> axesInts;
  if (!binder.s64IntegerArrayAttr(axesInts, "axes", {})) {
    for (int64_t i = 0, s = axesInts.size(); i < s; ++i) {
      Value iv = rewriter.create<Torch::ConstantIntOp>(
          binder.getLoc(), rewriter.getType<Torch::IntType>(),
          rewriter.getI64IntegerAttr(axesInts[i]));
      axesList.push_back(iv);
    }
  }

  // Do not include absolute value in the noop
  if (axesList.empty() && noop_with_empty_axes) {
    rewriter.replaceOp(binder.op, storeResult);
    return success();
  }

  Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
      binder.getLoc(),
      Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
      axesList);
  Value keepDimBool =
      rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), keepDims);
  Value dType = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
  // If we are using the ReducedSum as an intermediate op to be passed into
  // another operation, we might not want to replace the Op. So we create a new
  // Op and store the result in a variable.
  if (!isIntermediateOp) {
    rewriter.replaceOpWithNewOp<Torch::AtenSumDimIntListOp>(
        binder.op, resultType, data, dimValueList, keepDimBool,
        /*dtype=*/dType);
  } else {
    storeResult = rewriter.create<Torch::AtenSumDimIntListOp>(
        binder.getLoc(), resultType, data, dimValueList, keepDimBool,
        /*dtype=*/dType);
  }
  return success();
}

Value getValueList(OpBinder binder, ConversionPatternRewriter &rewriter,
                   Value operand) {
  SmallVector<Value> itemList;
  auto sizes = dyn_cast<Torch::ValueTensorType>(operand.getType()).getSizes();
  Torch::BaseTensorType operandType =
      cast<Torch::BaseTensorType>(operand.getType());

  SmallVector<int64_t> selectSizes;
  selectSizes.push_back(1);
  Type selectResultType = operandType.getWithSizesAndDtype(
      llvm::ArrayRef(selectSizes), operandType.getOptionalDtype());

  auto extract = [&rewriter, &binder](Value x, Value v) {
    auto xTy = cast<Torch::ValueTensorType>(x.getType());
    Type extractTy = rewriter.getType<Torch::FloatType>();
    if (isa<IntegerType>(xTy.getDtype()))
      extractTy = rewriter.getType<Torch::IntType>();

    return rewriter.create<Torch::AtenItemOp>(binder.getLoc(), extractTy, v);
  };

  Value zero = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

  MLIRContext *context = binder.op->getContext();
  for (int i = 2; i < sizes[0]; i++) {
    Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
        binder.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
    Value ext = rewriter.create<Torch::AtenSelectIntOp>(
        binder.getLoc(), selectResultType, operand, zero, selectIndex);
    Value item = extract(operand, ext);
    itemList.push_back(item);
  }
  auto xTy = cast<Torch::ValueTensorType>(operand.getType());
  Value ValueList;
  if (isa<IntegerType>(xTy.getDtype())) {
    ValueList = rewriter.create<Torch::PrimListConstructOp>(
        binder.getLoc(), Torch::ListType::get(Torch::IntType::get(context)),
        itemList);
  } else {
    ValueList = rewriter.create<Torch::PrimListConstructOp>(
        binder.getLoc(), Torch::ListType::get(Torch::FloatType::get(context)),
        itemList);
  }
  return ValueList;
}
} // namespace

void mlir::torch::onnx_c::populateDefaultDomainQtoZ(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp(
      "QuantizeLinear", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperands(operands, 3) ||
            binder.tensorResultType(resultType))
          return failure();

        Value operand = operands[0];
        Value scale = operands[1];
        Value zeropoint = operands[2];

        auto scaleTy = dyn_cast<Torch::ValueTensorType>(scale.getType());
        if (!scaleTy || !scaleTy.hasSizes())
          return rewriter.notifyMatchFailure(binder.op, "requires known rank");
        if (!resultType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "requires known result dtype");

        if (scaleTy.getSizes().size() == 0) {
          auto qTensorTy = getQTorchTypeFromTorchIntType(resultType);
          if (!qTensorTy) {
            return rewriter.notifyMatchFailure(binder.op,
                                               "unsupported result dtype");
          }

          auto torchqTy = Torch::getScalarTypeForType(qTensorTy.getDtype());

          Value tyConst = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                      static_cast<int64_t>(torchqTy)));

          scale = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::FloatType>(), scale);
          zeropoint = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), zeropoint);

          auto quantize = rewriter.create<Torch::AtenQuantizePerTensorOp>(
              binder.getLoc(), qTensorTy, operand, scale, zeropoint, tyConst);
          rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(
              binder.op, resultType, quantize);
          return success();
        }

        return failure();
      });
  patterns.onOp(
      "QLinearConv", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if ((binder.tensorOperands(operands, 8) &&
             binder.tensorOperands(operands, 9)) ||
            binder.tensorResultType(resultType))
          return failure();
        Value a = operands[0];
        Value aScale = operands[1];
        Value aZp = operands[2];
        Value b = operands[3];
        Value bScale = operands[4];
        Value bZp = operands[5];
        Value cScale = operands[6];
        Value cZp = operands[7];
        Value c = operands.size() == 9 ? operands[8] : nullptr;

        auto check = [](Value v) {
          auto vTy = cast<Torch::ValueTensorType>(v.getType());
          return llvm::all_of(vTy.getSizes(), [](int64_t d) { return d == 1; });
        };
        if (!check(aScale) || !check(aZp) || !check(bScale) || !check(bZp) ||
            !check(cScale) || !check(cScale))
          return rewriter.notifyMatchFailure(
              binder.op, "not supported for non per-tensor quantization");

        auto extract = [&rewriter, &binder](Value v) {
          auto vTy = cast<Torch::ValueTensorType>(v.getType());
          Type extractTy = rewriter.getType<Torch::FloatType>();
          if (isa<IntegerType>(vTy.getDtype()))
            extractTy = rewriter.getType<Torch::IntType>();

          return rewriter.create<Torch::AtenItemOp>(binder.getLoc(), extractTy,
                                                    v);
        };

        aZp = extract(aZp);
        bZp = extract(bZp);
        cZp = extract(cZp);
        aScale = extract(aScale);
        bScale = extract(bScale);
        cScale = extract(cScale);

        auto make = [&rewriter, &binder](Value v, Value scale,
                                         Value zp) -> Value {
          auto ty = cast<Torch::ValueTensorType>(v.getType());
          auto newTy = getQTorchTypeFromTorchIntType(ty);
          return rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
              binder.getLoc(), newTy, v, scale, zp);
        };

        a = make(a, aScale, aZp);
        b = make(b, bScale, bZp);

        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(),
            rewriter.getIntegerType(32, /*issigned=*/true));

        // TODO(suderman): insert convolution operator.
        llvm::SmallVector<Value> newOperands = {a, b};
        if (c)
          newOperands.push_back(c);

        cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(),
            rewriter.getType<Torch::QInt32Type>());

        llvm::SmallVector<NamedAttribute> newAttributes;
        newAttributes.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr("onnx.Conv")));
        for (auto namedAttr : binder.op->getAttrDictionary()) {
          if (namedAttr.getName().getValue().compare("name") == 0)
            continue;
          llvm::errs() << namedAttr.getName() << "\n";
          newAttributes.push_back(namedAttr);
        }

        c = rewriter
                .create<Torch::OperatorOp>(binder.getLoc(), cTy, newOperands,
                                           newAttributes,
                                           binder.op->getRegions().size())
                .getResult(0);

        Value outScale = rewriter.create<Torch::AtenMulFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(), aScale,
            bScale);
        Value outZp = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        c = rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
            binder.getLoc(), cTy, c, outScale, outZp);
        cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());

        c = rewriter.create<Torch::AtenDequantizeSelfOp>(binder.getLoc(), cTy,
                                                         c);
        cTy = getQTorchTypeFromTorchIntType(resultType);
        Value dtyVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = rewriter.create<Torch::AtenQuantizePerTensorOp>(
            binder.getLoc(), cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp(
      "QLinearMatMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperands(operands, 8) ||
            binder.tensorResultType(resultType))
          return failure();
        Value a = operands[0];
        Value aScale = operands[1];
        Value aZp = operands[2];
        Value b = operands[3];
        Value bScale = operands[4];
        Value bZp = operands[5];
        Value cScale = operands[6];
        Value cZp = operands[7];

        auto check = [](Value v) {
          auto vTy = cast<Torch::ValueTensorType>(v.getType());
          for (auto dim : vTy.getSizes())
            if (dim != 1)
              return false;
          return true;
        };
        if (!check(aScale) || !check(aZp) || !check(bScale) || !check(bZp) ||
            !check(cScale) || !check(cScale))
          return rewriter.notifyMatchFailure(
              binder.op, "not supported for non per-tensor quantization");

        Value emptyList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            ValueRange{});
        auto extract = [&rewriter, &binder, &emptyList](Value v) {
          auto vTy = cast<Torch::ValueTensorType>(v.getType());
          if (!vTy.getSizes().empty()) {
            vTy = rewriter.getType<Torch::ValueTensorType>(
                ArrayRef<int64_t>({}), vTy.getOptionalDtype());
            v = rewriter.create<Torch::AtenReshapeOp>(binder.getLoc(), vTy, v,
                                                      emptyList);
          }

          Type extractTy = rewriter.getType<Torch::FloatType>();
          if (isa<IntegerType>(vTy.getDtype()))
            extractTy = rewriter.getType<Torch::IntType>();

          return rewriter.create<Torch::AtenItemOp>(binder.getLoc(), extractTy,
                                                    v);
        };

        aZp = extract(aZp);
        bZp = extract(bZp);
        cZp = extract(cZp);
        aScale = extract(aScale);
        bScale = extract(bScale);
        cScale = extract(cScale);

        auto make = [&rewriter, &binder](Value v, Value scale,
                                         Value zp) -> Value {
          auto ty = cast<Torch::ValueTensorType>(v.getType());
          auto newTy = getQTorchTypeFromTorchIntType(ty);
          return rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
              binder.getLoc(), newTy, v, scale, zp);
        };

        a = make(a, aScale, aZp);
        b = make(b, bScale, bZp);

        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(),
            rewriter.getIntegerType(32, /*issigned=*/true));

        Value c;
        if (cTy.getSizes().size() == 2) {
          c = rewriter.create<Torch::AtenMmOp>(binder.getLoc(), cTy, a, b);
        } else {
          c = rewriter.create<Torch::AtenBmmOp>(binder.getLoc(), cTy, a, b);
        }

        cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(),
            rewriter.getType<Torch::QInt32Type>());

        Value mmScale = rewriter.create<Torch::AtenMulFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(), aScale,
            bScale);
        Value mmZp = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        c = rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
            binder.getLoc(), cTy, c, mmScale, mmZp);
        cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());

        c = rewriter.create<Torch::AtenDequantizeSelfOp>(binder.getLoc(), cTy,
                                                         c);
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = rewriter.create<Torch::AtenQuantizePerTensorOp>(
            binder.getLoc(), cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp("Reciprocal", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenReciprocalOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Relu", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        if (binder.tensorOperand(x) || binder.tensorResultType(resultType))
          return failure();

        rewriter.replaceOpWithNewOp<Torch::AtenReluOp>(binder.op, resultType,
                                                       x);
        return success();
      });
  patterns.onOp("Round", 11,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenRoundOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("RNN", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  return OnnxRnnExpander(binder, rewriter);
                });
  patterns.onOp(
      "Scatter", 9, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        int64_t axis;
        if (binder.s64IntegerAttr(axis, "axis", {}))
          return rewriter.notifyMatchFailure(binder.op, "axis bind failure");

        Torch::ValueTensorType resultTy;
        Value data, indices, updates;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorOperandAtIndex(updates, 2) ||
            binder.tensorResultType(resultTy))
          return failure();

        auto dataTy = cast<Torch::ValueTensorType>(data.getType()),
             indicesTy = cast<Torch::ValueTensorType>(indices.getType()),
             updatesTy = cast<Torch::ValueTensorType>(updates.getType());

        int64_t dataRank = dataTy.getSizes().size(),
                indicesRank = indicesTy.getSizes().size(),
                updatesRank = updatesTy.getSizes().size();

        if ((dataRank < 1) || (indicesRank < 1) || (updatesRank < 1) ||
            (axis < -dataRank) || (axis >= dataRank))
          return failure();

        Value axisValue = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(axis));

        rewriter.replaceOpWithNewOp<Torch::AtenScatterSrcOp>(
            binder.op, resultTy, data, axisValue, indices, updates);

        return success();
      });
  patterns.onOp(
      "ScatterElements", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        SmallVector<Value> valList;
        int64_t axis;
        std::string reduction;
        int64_t numOperands = binder.op->getNumOperands();
        if (binder.tensorOperands(valList, numOperands) ||
            binder.s64IntegerAttr(axis, "axis", 0) ||
            binder.customOpNameStringAttr(reduction, "reduction", "none") ||
            binder.tensorResultType(resultType))
          return failure();

        auto loc = binder.getLoc();
        Value data = valList[0];
        Value indices = valList[1];
        Value updates = valList[2];

        // ONNX allows negative axis.
        if (axis < 0)
          axis +=
              cast<Torch::ValueTensorType>(data.getType()).getSizes().size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(0));
        Value one = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(1));

        Value axisSize = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), data,
            constAxis);

        auto indicesTy = cast<Torch::ValueTensorType>(indices.getType());
        Value indicesAdd = rewriter.create<Torch::AtenAddScalarOp>(
            loc, indicesTy, indices, axisSize, one);

        Value inputNeg = rewriter.create<Torch::AtenLtScalarOp>(
            loc,
            rewriter.getType<Torch::ValueTensorType>(indicesTy.getSizes(),
                                                     rewriter.getI1Type()),
            indices, zero);

        indices = rewriter.create<Torch::AtenWhereSelfOp>(
            loc, indicesTy, inputNeg, indicesAdd, indices);

        if (reduction == "none") {
          rewriter.replaceOpWithNewOp<Torch::AtenScatterSrcOp>(
              binder.op, resultType, data, constAxis, indices, updates);
          return success();
        }

        // TODO: Implement max and min cases
        if (reduction == "mul") {
          reduction = "multiply";
        } else if (reduction == "max" || reduction == "min") {
          return rewriter.notifyMatchFailure(
              binder.op, "max/min reduction unsupported for scatter elements");
        }

        Value cstStrReduction =
            rewriter.create<Torch::ConstantStrOp>(binder.getLoc(), reduction);

        rewriter.replaceOpWithNewOp<Torch::AtenScatterReduceOp>(
            binder.op, resultType, data, constAxis, indices, updates,
            cstStrReduction);
        return success();
      });
  patterns.onOp(
      "SequenceConstruct", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallVector<Value> operands;
        Torch::ListType resultType;

        if (binder.tensorOperands(operands, binder.getNumOperands()) ||
            binder.tensorListResultType(resultType))
          return failure();

        rewriter.replaceOpWithNewOp<Torch::PrimListConstructOp>(
            binder.op, resultType, operands);
        return success();
      });
  patterns.onOp(
      "SequenceLength", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // onnx.SequenceLength takes a sequence(list) of tensors, and returns
        // a zero rank tensor with the length.
        Torch::ValueTensorType resultType;
        Value x;
        if (binder.tensorListOperand(x) || binder.tensorResultType(resultType))
          return failure();

        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value len = rewriter.create<Torch::AtenLenTOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), x);

        // AtenLenTOp returns a torch.int, so we have to
        // put that in a tensor.
        rewriter.replaceOpWithNewOp<Torch::AtenTensorIntOp>(
            binder.op, resultType, len, none, none, cstFalse);

        return success();
      });
  patterns.onOp(
      "Sigmoid", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        if (binder.tensorOperand(x) || binder.tensorResultType(resultType))
          return failure();

        rewriter.replaceOpWithNewOp<Torch::AtenSigmoidOp>(binder.op, resultType,
                                                          x);
        return success();
      });
  patterns.onOp("Sin", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenSinOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Tanh", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenTanhOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Sqrt", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenSqrtOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Sub", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        Value y;
        if (binder.tensorOperands(x, y) || binder.tensorResultType(resultType))
          return failure();
        Value const1 = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
        rewriter.replaceOpWithNewOp<Torch::AtenSubTensorOp>(
            binder.op, resultType, x, y, const1);
        return success();
      });
  patterns.onOp(
      "Sum", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        if (binder.op->getNumOperands() == 1) {
          Torch::ValueTensorType resultType;
          Value x;
          if (binder.tensorOperand(x) || binder.tensorResultType(resultType))
            return failure();
          rewriter.replaceOp(binder.op, x);
          return success();
        }
        Torch::ValueTensorType resultType;
        SmallVector<Value> valList;
        int64_t numOperands = binder.op->getNumOperands();
        if (binder.tensorOperands(valList, numOperands) ||
            binder.tensorResultType(resultType))
          return failure();
        Value const1 = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
        // Short circuit to binary add
        if (numOperands == 2) {
          rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
              binder.op, resultType, valList[0], valList[1], const1);
          return success();
        }
        // When binder.op->getNumOperands() > 2
        Value curr = rewriter.create<Torch::AtenAddTensorOp>(
            binder.getLoc(), resultType, valList[0], valList[1], const1);
        for (int i = 2; i < numOperands; i++) {
          if (i == numOperands - 1) {
            curr = rewriter.create<Torch::AtenAddTensorOp>(
                binder.getLoc(), resultType, curr, valList[i], const1);
          } else {
            SmallVector<int64_t> resultBroadcastShapeInt;
            SmallVector<Value> resultBroadcastShapeValue;
            Torch::computeBroadcastShape(rewriter, binder.getLoc(), curr,
                                         valList[i], resultBroadcastShapeInt,
                                         resultBroadcastShapeValue);
            auto baseType = Torch::ValueTensorType::get(
                binder.op->getContext(), resultBroadcastShapeInt,
                resultType.getOptionalDtype());
            curr = rewriter.create<Torch::AtenAddTensorOp>(
                binder.getLoc(), baseType, curr, valList[i], const1);
          }
        }
        rewriter.replaceOp(binder.op, curr);
        return success();
      });
  patterns.onOp("Where", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  SmallVector<Value> valList;
                  int64_t numOperands = binder.op->getNumOperands();
                  if (binder.tensorOperands(valList, numOperands) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  Value condition = valList[0];
                  Value x = valList[1];
                  Value y = valList[2];
                  rewriter.replaceOpWithNewOp<Torch::AtenWhereSelfOp>(
                      binder.op, resultType, condition, x, y);
                  return success();
                });
  patterns.onOp(
      "Xor", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        Value y;
        if (binder.tensorOperands(x, y) || binder.tensorResultType(resultType))
          return failure();
        rewriter.replaceOpWithNewOp<Torch::AtenLogicalXorOp>(binder.op,
                                                             resultType, x, y);
        return success();
      });
  patterns.onOp(
      "Squeeze", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        SmallVector<Value> inputOperands;
        if (binder.tensorOperands(inputOperands, binder.op->getNumOperands()) ||
            binder.tensorResultType(resultType))
          return failure();

        Value data = inputOperands[0];
        auto inputType = dyn_cast<Torch::ValueTensorType>(data.getType());
        if (!inputType.hasSizes() || !resultType.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: expected input and result to have shapes");

        int64_t inputRank = inputType.getSizes().size();
        int64_t resultRank = resultType.getSizes().size();
        int64_t rankDiff = inputRank - resultRank;
        if (rankDiff == 0) {
          // In this case, no dimension is squeezed. Hence just replace the op
          // with input.
          rewriter.replaceOp(binder.op, data);
          return success();
        }

        if (inputOperands.size() == 1) {
          // Case: `axes` value is not present which means squeeze all the
          // dimensions with shape value 1.
          rewriter.replaceOpWithNewOp<Torch::AtenSqueezeOp>(binder.op,
                                                            resultType, data);
          return success();
        }

        SmallVector<Value> dimList;
        if (inputType.areAllSizesKnown() && resultType.areAllSizesKnown()) {
          // If the input shape and result shape is statically known then the
          // list of dims to be squeezed can be derived from those shapes. As a
          // result, we don't have to wait for the dim values to be known at
          // runtime which is also expected by the downstream pipeline.
          SmallVector<int64_t> inputShape(inputType.getSizes());
          SmallVector<int64_t> resultShape(resultType.getSizes());
          SmallVector<int64_t> squeezeDims;
          unsigned resultShapeCounter = 0;
          for (unsigned i = 0; i < inputRank; i++) {
            if (resultShapeCounter < resultRank &&
                inputShape[i] == resultShape[resultShapeCounter]) {
              resultShapeCounter++;
            } else {
              squeezeDims.push_back(i);
            }
          }
          for (auto i : squeezeDims) {
            dimList.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i)));
          }
        }

        if (dimList.empty()) {
          Value axes = inputOperands[1];
          Torch::BaseTensorType axesType =
              cast<Torch::BaseTensorType>(axes.getType());
          SmallVector<int64_t> selectSizes{1};
          Type selectResultType = axesType.getWithSizesAndDtype(
              selectSizes, axesType.getOptionalDtype());
          Value zero = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
          for (int i = 0; i < rankDiff; i++) {
            // Go through the axes list and get each dim in the list
            Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
            Value extract = rewriter.create<Torch::AtenSelectIntOp>(
                binder.getLoc(), selectResultType, axes, zero, selectIndex);
            Value dim = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
            dimList.push_back(dim);
          }
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            dimList);
        rewriter.replaceOpWithNewOp<Torch::PrimsSqueezeOp>(
            binder.op, resultType, data, dimValueList);
        return success();
      });
  patterns.onOp(
      "Unsqueeze", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // Unlike squeeze where we are able to lower to Torch::PrimsSqueezeOp,
        // pytorch does not support torch.unsqueeze to insert multiple new dims.
        // discussion can be found here:
        // https://github.com/pytorch/pytorch/issues/9410
        // So, for now, we unroll into multiple unsqueezes.
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value data, axes;
        if (binder.tensorOperands(data, axes) ||
            binder.tensorResultType(resultType))
          return failure();
        auto inputType = dyn_cast<Torch::ValueTensorType>(data.getType());
        if (!inputType.hasSizes() || !resultType.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: expected input and result to have shapes");

        int64_t inputRank = inputType.getSizes().size();
        int64_t resultRank = resultType.getSizes().size();
        int64_t rankDiff = resultRank - inputRank;
        if (rankDiff == 0) {
          // In this case, no dimension is unsqueezed. Hence just replace the op
          // with input.
          rewriter.replaceOp(binder.op, data);
          return success();
        }

        SmallVector<int64_t> unsqueezeDims;
        SmallVector<int64_t> inputShape(inputType.getSizes());
        if (inputType.areAllSizesKnown() && resultType.areAllSizesKnown()) {
          // If the input shape and result shape is statically known then the
          // list of dims to be squeezed can be derived from those shapes. As a
          // result, we don't have to wait for the dim values to be known at
          // runtime which is also expected by the downstream pipeline.
          SmallVector<int64_t> resultShape(resultType.getSizes());
          unsigned inputShapeCounter = 0;
          for (unsigned i = 0; i < resultRank; i++) {
            if (inputShapeCounter < inputRank &&
                inputShape[inputShapeCounter] == resultShape[i]) {
              inputShapeCounter++;
            } else {
              unsqueezeDims.push_back(i);
            }
          }
        } else {
          SmallVector<int64_t> unsqueezeDimsInts;
          if (!matchPattern(axes, m_OnnxListOfConstantInts(unsqueezeDimsInts)))
            return rewriter.notifyMatchFailure(
                binder.op, "only support constant int axes values");

          for (auto dim : unsqueezeDimsInts)
            unsqueezeDims.push_back(dim < 0 ? dim + resultRank : dim);
          // If we don't sort, unsqueezing first on 4 and then on 0 would fail
          // for shape = {x,y,z}, and axes [4,0]
          llvm::sort(unsqueezeDims.begin(), unsqueezeDims.end());
        }
        Value result = data;
        SmallVector<int64_t> unsqueezeShape = inputShape;
        for (auto dim : unsqueezeDims) {
          unsqueezeShape.insert(unsqueezeShape.begin() + dim, 1);
          Type unsqueezeType = resultType.getWithSizesAndDtype(
              unsqueezeShape, resultType.getOptionalDtype());
          Value cstDim = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(dim));
          result = rewriter.create<Torch::AtenUnsqueezeOp>(loc, unsqueezeType,
                                                           result, cstDim);
        }
        rewriter.replaceOp(binder.op, result);
        return success();
      });
  patterns.onOp(
      "Softmax", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        int64_t axis;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(axis, "axis", -1) ||
            binder.tensorResultType(resultType))
          return failure();

        // ONNX allows negative axis.
        if (axis < 0)
          axis +=
              cast<Torch::ValueTensorType>(input.getType()).getSizes().size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));

        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        rewriter.replaceOpWithNewOp<Torch::AtenSoftmaxIntOp>(
            binder.op, resultType, input, constAxis, /*dtype=*/noneVal);
        return success();
      });

  patterns.onOp(
      "Selu", 6, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0
        Torch::ValueTensorType resultType;
        float alpha, gamma;
        Value operand;
        // Refer https://onnx.ai/onnx/operators/onnx__Selu.html for the default
        // alpha and gamma values.
        if (binder.tensorOperand(operand) ||
            binder.f32FloatAttr(alpha, "alpha", 1.67326) ||
            binder.f32FloatAttr(gamma, "gamma", 1.0507) ||
            binder.tensorResultType(resultType))
          return failure();

        Value vAlpha = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), alpha));

        Value vScale = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), gamma));

        Value vInputScale = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), 1.0));

        rewriter.replaceOpWithNewOp<Torch::AtenEluOp>(
            binder.op, resultType, operand, vAlpha, vScale, vInputScale);
        return success();
      });
  patterns.onOp("ReduceL1", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  int64_t keepDims, noop_with_empty_axes;
                  Value operand;
                  if (binder.tensorOperandAtIndex(operand, 0) ||
                      binder.tensorResultType(resultType) ||
                      binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
                      binder.s64IntegerAttr(noop_with_empty_axes,
                                            "noop_with_empty_axes", 0))
                    return failure();

                  Value data = rewriter.create<Torch::AtenAbsOp>(
                      binder.getLoc(), operand.getType(), operand);

                  return reducedSumImpl(binder, rewriter, data, resultType,
                                        /*storeValue=*/operand, keepDims,
                                        noop_with_empty_axes, false);
                });
  patterns.onOp(
      "ReduceL2", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        int64_t keepDims, noop_with_empty_axes;
        if (binder.tensorOperandAtIndex(operand, 0) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();

        // A ReduceL2 op is equivalent to the following sequence of operations:
        // Mul(x, x) -> ReduceSum -> CastF32 -> Sqrt -> CastLike(resultType)
        Value squareOfOperand = rewriter.create<Torch::AtenMulTensorOp>(
            binder.getLoc(), operand.getType(), operand, operand);

        auto reducedSum =
            reducedSumImpl(binder, rewriter, squareOfOperand, resultType,
                           operand, keepDims, noop_with_empty_axes, true);
        if (failed(reducedSum))
          return rewriter.notifyMatchFailure(
              binder.op,
              "Failed to perform sum operation on square of operand");

        Value castDType = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(/*Float32Type*/ 6));

        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value constFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);

        // Perform an AtenToDtype op on the squared sum of the operand, stored
        // now in operand itself.
        auto size = dyn_cast<Torch::ValueTensorType>(operand.getType())
                        .getOptionalSizes();
        auto f32ResultType = rewriter.getType<Torch::ValueTensorType>(
            size, rewriter.getF32Type());
        Value operandCast = rewriter.create<Torch::AtenToDtypeOp>(
            binder.getLoc(), f32ResultType, operand, castDType,
            /*non_blocking=*/constFalse, /*copy=*/constFalse,
            /*memory_format=*/noneVal);

        Value operandSqrt = rewriter.create<Torch::AtenSqrtOp>(
            binder.getLoc(), f32ResultType, operandCast);

        Value resultDtype = Torch::getDtypeIntValueForType(
            rewriter, binder.getLoc(), resultType.getDtype());
        rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
            binder.op, resultType, operandSqrt, resultDtype,
            /*non_blocking=*/constFalse, /*copy=*/constFalse,
            /*memory_format=*/noneVal);
        return success();
      });
  patterns.onOp("ReduceLogSum", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value data;
                  int64_t keepDims, noop_with_empty_axes;
                  if (binder.tensorOperandAtIndex(data, 0) ||
                      binder.tensorResultType(resultType) ||
                      binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
                      binder.s64IntegerAttr(noop_with_empty_axes,
                                            "noop_with_empty_axes", 0))
                    return failure();

                  auto reducedSumBool =
                      reducedSumImpl(binder, rewriter, data, resultType,
                                     /*storeValue=*/data, keepDims,
                                     noop_with_empty_axes, true);

                  if (failed(reducedSumBool))
                    return rewriter.notifyMatchFailure(
                        binder.op,
                        "Failed to perform sum operation on square of operand");

                  rewriter.replaceOpWithNewOp<Torch::AtenLogOp>(
                      binder.op, resultType, data);
                  return success();
                });
  patterns.onOp(
      "ReduceLogSumExp", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        int64_t keepDims, noop_with_empty_axes;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();

        // out = Log(reducesum(exp(data)))
        Value castDType = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(/*Float64Type*/ 7));
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value constFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        auto size =
            dyn_cast<Torch::ValueTensorType>(data.getType()).getOptionalSizes();
        auto f64ResultType = rewriter.getType<Torch::ValueTensorType>(
            size, rewriter.getF64Type());
        Value dataCast = rewriter.create<Torch::AtenToDtypeOp>(
            binder.getLoc(), f64ResultType, data, castDType,
            /*non_blocking=*/constFalse, /*copy=*/constFalse,
            /*memory_format=*/noneVal);
        Value dataExp = rewriter.create<Torch::AtenExpOp>(
            binder.getLoc(), f64ResultType, dataCast);
        auto f64ReduceType = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF64Type());
        auto reducedSumBool = reducedSumImpl(
            binder, rewriter, dataExp, f64ReduceType,
            /*storeValue=*/data, keepDims, noop_with_empty_axes, true);
        if (failed(reducedSumBool))
          return rewriter.notifyMatchFailure(
              binder.op,
              "Failed to perform sum operation on square of operand");
        Value finalResult = rewriter.create<Torch::AtenLogOp>(
            binder.getLoc(), f64ReduceType, data);
        Value resultDtype = Torch::getDtypeIntValueForType(
            rewriter, binder.getLoc(), resultType.getDtype());
        rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
            binder.op, resultType, finalResult, resultDtype,
            /*non_blocking=*/constFalse, /*copy=*/constFalse,
            /*memory_format=*/noneVal);
        return success();
      });
  patterns.onOp("ReduceSum", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value data;
                  int64_t keepDims, noop_with_empty_axes;
                  if (binder.tensorOperandAtIndex(data, 0) ||
                      binder.tensorResultType(resultType) ||
                      binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
                      binder.s64IntegerAttr(noop_with_empty_axes,
                                            "noop_with_empty_axes", 0))
                    return failure();

                  return reducedSumImpl(binder, rewriter, data, resultType,
                                        /*storeValue=*/data, keepDims,
                                        noop_with_empty_axes, false);
                });
  patterns.onOp("ReduceSumSquare", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value data;
                  int64_t keepDims, noop_with_empty_axes;
                  if (binder.tensorOperandAtIndex(data, 0) ||
                      binder.tensorResultType(resultType) ||
                      binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
                      binder.s64IntegerAttr(noop_with_empty_axes,
                                            "noop_with_empty_axes", 0))
                    return failure();

                  Value dataSquare = rewriter.create<Torch::AtenMulTensorOp>(
                      binder.getLoc(), data.getType(), data, data);

                  return reducedSumImpl(binder, rewriter, dataSquare,
                                        resultType,
                                        /*storeValue=*/data, keepDims,
                                        noop_with_empty_axes, false);
                });
  patterns.onOp(
      "ReduceMean", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        int64_t keepDims, noop_with_empty_axes;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();

        SmallVector<Value> axesList;

        Value axesVal;
        if (!binder.tensorOperandAtIndex(axesVal, 1)) {
          auto inputType = dyn_cast<Torch::ValueTensorType>(data.getType());
          if (!inputType.hasSizes() || !resultType.hasSizes()) {
            return rewriter.notifyMatchFailure(
                binder.op,
                "unimplemented: expected input and result to have shapes");
          }

          // If the input shape and result shape is statically known then the
          // list of dims to be squeezed can be derived from those shapes. As a
          // result, we don't have to wait for the dim values to be known at
          // runtime which is also expected by the downstream pipeline.
          if (inputType.areAllSizesKnown() && resultType.areAllSizesKnown()) {
            SmallVector<int64_t> inputShape{inputType.getSizes()};
            SmallVector<int64_t> resultShape{resultType.getSizes()};
            if (llvm::equal(inputShape, resultShape)) {
              // Case: none of the dimension is reduced.
              rewriter.replaceOp(binder.op, data);
              return success();
            }
            if (areAllElementsDistinct(inputShape)) {
              // The check for the input shape elements to be distinct is added
              // for the cases like:
              // Input: [3, 2, 2] -> Output: [3, 2]
              // For the above case, from the input and output shape it can't be
              // inferred whether the dim:1 is reduced or dim:2. To avoid these
              // type of cases, the check has been placed.
              SmallVector<int64_t> reduceDims;
              unsigned resultShapeCounter = 0;
              for (unsigned i = 0; i < inputShape.size(); i++) {
                if (resultShapeCounter < resultShape.size() &&
                    inputShape[i] == resultShape[resultShapeCounter]) {
                  resultShapeCounter++;
                } else {
                  reduceDims.push_back(i);
                  if (resultShapeCounter < resultShape.size() &&
                      resultShape[resultShapeCounter] == 1)
                    resultShapeCounter++;
                }
              }
              for (auto i : reduceDims) {
                axesList.push_back(rewriter.create<Torch::ConstantIntOp>(
                    binder.getLoc(), rewriter.getI64IntegerAttr(i)));
              }
            }
          }

          if (axesList.empty()) {
            Torch::BaseTensorType axesType =
                cast<Torch::BaseTensorType>(axesVal.getType());
            auto axesTy = dyn_cast<Torch::ValueTensorType>(axesVal.getType());
            auto axesShape = axesTy.getSizes();
            if (axesShape.size() != 1 || axesShape[0] == Torch::kUnknownSize)
              return failure();

            Value zero = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getI64IntegerAttr(0));
            SmallVector<int64_t> selectSizes{1};
            auto selType = rewriter.getType<Torch::ValueTensorType>(
                selectSizes, axesType.getOptionalDtype());
            int64_t numAxes = axesShape[0];
            for (int64_t i = 0; i < numAxes; ++i) {
              Value iv = rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getType<Torch::IntType>(),
                  rewriter.getI64IntegerAttr(i));
              Value extract = rewriter.create<Torch::AtenSelectIntOp>(
                  binder.getLoc(), selType, axesVal, zero, iv);
              Value dim = rewriter.create<Torch::AtenItemOp>(
                  binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
              axesList.push_back(dim);
            }
          }
        }

        SmallVector<int64_t> axesInts;
        if (!binder.s64IntegerArrayAttr(axesInts, "axes", {})) {
          for (int64_t i = 0, s = axesInts.size(); i < s; ++i) {
            Value iv = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getI64IntegerAttr(axesInts[i]));
            axesList.push_back(iv);
          }
        }

        // deal with case when axes is empty
        if (axesList.empty() && noop_with_empty_axes) {
          rewriter.replaceOp(binder.op, data);
          return success();
        }

        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            axesList);
        Value keepDimBool =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), keepDims);
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        rewriter.replaceOpWithNewOp<Torch::AtenMeanDimOp>(
            binder.op, resultType, data, dimValueList, keepDimBool,
            /*dtype=*/noneVal);
        return success();
      });
  patterns.onOp(
      "ReduceMax", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // AtenAmaxOp allows us to pass a list of dims
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        int64_t keepDims;
        int64_t noop_with_empty_axes;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();

        auto dataTy = cast<Torch::BaseTensorType>(data.getType());
        Torch::IntType torchIntTy = rewriter.getType<Torch::IntType>();

        // If any of the input dims are 0 we set to the upper limit:
        if (llvm::any_of(dataTy.getSizes(), [](int64_t d) { return d == 0; }) &&
            (llvm::any_of(dataTy.getSizes(),
                          [](int64_t d) { return d == Torch::kUnknownSize; }) ||
             keepDims)) {
          auto dty = dataTy.getDtype();
          Value scalar;
          if (FloatType fpTy = dyn_cast<FloatType>(dty)) {
            auto inf = APFloat::getInf(fpTy.getFloatSemantics());
            scalar = rewriter.create<Torch::ConstantFloatOp>(
                binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                rewriter.getFloatAttr(rewriter.getF64Type(),
                                      inf.convertToDouble()));
          }

          if (IntegerType intTy = dyn_cast<IntegerType>(dty)) {
            auto mx =
                intTy.isSigned()
                    ? APInt::getSignedMaxValue(intTy.getIntOrFloatBitWidth())
                    : APInt::getMaxValue(intTy.getIntOrFloatBitWidth());
            scalar = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), torchIntTy,
                rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                        mx.getSExtValue()));
          }

          llvm::SmallVector<Value> fillDims;
          for (int i = 0, s = resultType.getSizes().size(); i < s; ++i) {
            auto staticDim = resultType.getSizes()[i];
            if (staticDim != Torch::kUnknownSize) {
              fillDims.push_back(rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), torchIntTy,
                  rewriter.getI64IntegerAttr(staticDim)));
              continue;
            }

            Value iv = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), torchIntTy, rewriter.getI64IntegerAttr(i));
            fillDims.push_back(rewriter.create<Torch::AtenSizeIntOp>(
                binder.getLoc(), torchIntTy, data, iv));
          }

          Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
          Value fillDimsList = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(), Torch::ListType::get(torchIntTy), fillDims);
          rewriter.replaceOpWithNewOp<Torch::AtenFullOp>(
              binder.op, resultType, fillDimsList, scalar, none, none, none,
              none);
          return success();
        }

        // Previous version of the operation had the axes as an attribute:
        SmallVector<Value> axesList;
        llvm::SmallVector<int64_t> axesAttr;
        if (!binder.s64IntegerArrayAttr(axesAttr, "axes", {})) {
          for (int i = 0, s = axesAttr.size(); i < s; ++i) {
            axesList.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), torchIntTy,
                rewriter.getI64IntegerAttr(axesAttr[i])));
          }
        }

        // Extract the axes values from the axes operand:
        if (!binder.tensorOperandAtIndex(axes, 1)) {
          Torch::BaseTensorType axesType =
              cast<Torch::BaseTensorType>(axes.getType());
          SmallVector<int64_t> selectSizes{1};
          Type selectResultType = axesType.getWithSizesAndDtype(
              selectSizes, axesType.getOptionalDtype());
          auto sizes = axesType.getSizes();

          Value zero = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

          // Extract the value of each axes:
          for (int i = 0; i < sizes[0]; i++) {
            // Go through the axes list and get each dim in the list
            Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
            Value extract = rewriter.create<Torch::AtenSelectIntOp>(
                binder.getLoc(), selectResultType, axes, zero, selectIndex);
            Value dim = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
            axesList.push_back(dim);
          }
        }

        // Handle the noop case:
        if (axesList.empty() && noop_with_empty_axes) {
          rewriter.replaceOp(binder.op, data);
          return success();
        }

        // Deal with case when no axes arg is passed but not a noop:
        if (axesList.empty()) {
          int64_t numDims = dyn_cast<Torch::ValueTensorType>(data.getType())
                                .getSizes()
                                .size();
          for (int i = 0; i < numDims; i++) {
            Value curr = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
            axesList.push_back(curr);
          }
        }

        // Handle negative axis:
        Value rankVal = rewriter.create<Torch::AtenDimOp>(binder.getLoc(),
                                                          torchIntTy, data);
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(0));
        for (Value &axes : axesList) {
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), axes, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, rankVal);
          axes = rewriter.create<Torch::AtenAddIntOp>(binder.getLoc(), axes,
                                                      finalOffset);
        }

        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(), Torch::ListType::get(torchIntTy), axesList);
        Value keepDimBool =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), keepDims);
        rewriter.replaceOpWithNewOp<Torch::AtenAmaxOp>(
            binder.op, resultType, data, dimValueList, keepDimBool);
        return success();
      });

  patterns.onOp(
      "ReduceMin", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // AtenAminOp allows us to pass a list of dims
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        int64_t keepDims;
        int64_t noop_with_empty_axes;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();

        auto dataTy = cast<Torch::BaseTensorType>(data.getType());
        Torch::IntType torchIntTy = rewriter.getType<Torch::IntType>();

        // If any of the input dims are 0 we set to the upper limit:
        if (llvm::any_of(dataTy.getSizes(), [](int64_t d) { return d == 0; }) &&
            (llvm::any_of(dataTy.getSizes(),
                          [](int64_t d) { return d == Torch::kUnknownSize; }) ||
             keepDims)) {
          auto dty = dataTy.getDtype();
          Value scalar;
          if (FloatType fpTy = dyn_cast<FloatType>(dty)) {
            auto inf = APFloat::getInf(fpTy.getFloatSemantics());
            scalar = rewriter.create<Torch::ConstantFloatOp>(
                binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                rewriter.getFloatAttr(rewriter.getF64Type(),
                                      inf.convertToDouble()));
          }

          if (IntegerType intTy = dyn_cast<IntegerType>(dty)) {
            auto mx =
                intTy.isSigned()
                    ? APInt::getSignedMaxValue(intTy.getIntOrFloatBitWidth())
                    : APInt::getMaxValue(intTy.getIntOrFloatBitWidth());
            scalar = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), torchIntTy,
                rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                        mx.getSExtValue()));
          }

          llvm::SmallVector<Value> fillDims;
          for (int i = 0, s = resultType.getSizes().size(); i < s; ++i) {
            auto staticDim = resultType.getSizes()[i];
            if (staticDim != Torch::kUnknownSize) {
              fillDims.push_back(rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), torchIntTy,
                  rewriter.getI64IntegerAttr(staticDim)));
              continue;
            }

            Value iv = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), torchIntTy, rewriter.getI64IntegerAttr(i));
            fillDims.push_back(rewriter.create<Torch::AtenSizeIntOp>(
                binder.getLoc(), torchIntTy, data, iv));
          }

          Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
          Value fillDimsList = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(), Torch::ListType::get(torchIntTy), fillDims);
          rewriter.replaceOpWithNewOp<Torch::AtenFullOp>(
              binder.op, resultType, fillDimsList, scalar, none, none, none,
              none);
          return success();
        }

        // Previous version of the operation had the axes as an attribute:
        SmallVector<Value> axesList;
        llvm::SmallVector<int64_t> axesAttr;
        if (!binder.s64IntegerArrayAttr(axesAttr, "axes", {})) {
          for (int i = 0, s = axesAttr.size(); i < s; ++i) {
            axesList.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), torchIntTy,
                rewriter.getI64IntegerAttr(axesAttr[i])));
          }
        }

        // Extract the axes values from the axes operand:
        if (!binder.tensorOperandAtIndex(axes, 1)) {
          Torch::BaseTensorType axesType =
              cast<Torch::BaseTensorType>(axes.getType());
          SmallVector<int64_t> selectSizes{1};
          Type selectResultType = axesType.getWithSizesAndDtype(
              selectSizes, axesType.getOptionalDtype());
          auto sizes = axesType.getSizes();

          Value zero = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

          // Extract the value of each axes:
          for (int i = 0; i < sizes[0]; i++) {
            // Go through the axes list and get each dim in the list
            Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
            Value extract = rewriter.create<Torch::AtenSelectIntOp>(
                binder.getLoc(), selectResultType, axes, zero, selectIndex);
            Value dim = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
            axesList.push_back(dim);
          }
        }

        // Handle the noop case:
        if (axesList.empty() && noop_with_empty_axes) {
          rewriter.replaceOp(binder.op, data);
          return success();
        }

        // Deal with case when no axes arg is passed but not a noop:
        if (axesList.empty()) {
          int64_t numDims = dyn_cast<Torch::ValueTensorType>(data.getType())
                                .getSizes()
                                .size();
          for (int i = 0; i < numDims; i++) {
            Value curr = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
            axesList.push_back(curr);
          }
        }

        // Handle negative axis:
        Value rankVal = rewriter.create<Torch::AtenDimOp>(binder.getLoc(),
                                                          torchIntTy, data);
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(0));
        for (Value &axes : axesList) {
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), axes, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, rankVal);
          axes = rewriter.create<Torch::AtenAddIntOp>(binder.getLoc(), axes,
                                                      finalOffset);
        }

        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(), Torch::ListType::get(torchIntTy), axesList);
        Value keepDimBool =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), keepDims);
        rewriter.replaceOpWithNewOp<Torch::AtenAminOp>(
            binder.op, resultType, data, dimValueList, keepDimBool);
        return success();
      });

  patterns.onOp(
      "Shape", 9, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        int64_t start, end;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(start, "start", 0) ||
            binder.s64IntegerAttr(end, "end", -1))
          return failure();

        auto inputType = dyn_cast<Torch::ValueTensorType>(operand.getType());
        int64_t inputRank = inputType.getSizes().size();

        auto shapeType = Torch::ValueTensorType::get(
            binder.op->getContext(), SmallVector<int64_t>{inputRank},
            resultType.getOptionalDtype());

        Value shape = rewriter.create<Torch::Aten_ShapeAsTensorOp>(
            binder.getLoc(), shapeType, operand);

        if (start == 0 && end == -1) {
          rewriter.replaceOp(binder.op, shape);
          return success();
        }

        Value sv = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(start));

        Value ev = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(end));

        Value step = rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), 1);

        Value dim = rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), 0);

        shape = rewriter.create<Torch::AtenSliceTensorOp>(
            binder.getLoc(), resultType, shape, dim, sv, ev, step);

        rewriter.replaceOp(binder.op, shape);
        return success();
      });

  patterns.onOp("Sinh", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();

                  rewriter.replaceOpWithNewOp<Torch::AtenSinhOp>(
                      binder.op, resultType, operand);
                  return success();
                });

  // split with fixed-size parts
  // Arguments:
  // - input: the tensor to split
  // Attributes:
  // - axis: the axis along which to split the input
  // - num_outputs: the number of outputs to produce
  // Outputs:
  // - outputs: the produced outputs. Variadic with num_outputs elements.
  // Note: torch.aten gives a list of tensors, but ONNX gives a variadic list of
  // tensors
  //       so we need to unpack the list
  patterns.onOp(
      "Split", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value self;
        int64_t axis;
        int64_t numOutputs;
        if (binder.tensorOperand(self))
          return rewriter.notifyMatchFailure(
              binder.op, "Not converting to AtenSplitTensorOp due to input "
                         "tensor mismatch");
        if (binder.s64IntegerAttr(axis, "axis", 0))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get axis attribute");
        if (binder.s64IntegerAttr(numOutputs, "num_outputs", 2))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to get num_outputs attribute");

        auto loc = binder.getLoc();
        auto result0Ty =
            cast<Torch::ValueTensorType>(binder.op->getResult(0).getType());
        auto resultNTy = cast<Torch::ValueTensorType>(
            binder.op->getResults().back().getType());
        auto selfTy = cast<Torch::ValueTensorType>(self.getType());

        int64_t dim = axis;
        if (dim < 0)
          dim += selfTy.getSizes().size();

        Value dimValue = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(dim));

        Value vNumOutputs = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(numOutputs));

        Value one = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));

        Value vDimSize = rewriter.create<Torch::AtenSizeIntOp>(
            loc, rewriter.getType<Torch::IntType>(), self, dimValue);

        Value addNumOutputs =
            rewriter.create<Torch::AtenAddIntOp>(loc, vDimSize, vNumOutputs);
        Value subOne =
            rewriter.create<Torch::AtenSubIntOp>(loc, addNumOutputs, one);
        Value splitSize =
            rewriter.create<Torch::AtenFloordivIntOp>(loc, subOne, vNumOutputs);

        llvm::SmallVector<Value> outputs;
        Value step = one;
        Value start = zero;

        for (int i = 0; i < numOutputs - 1; ++i) {
          Value end =
              rewriter.create<Torch::AtenAddIntOp>(loc, start, splitSize);
          Value slice = rewriter.create<Torch::AtenSliceTensorOp>(
              loc, result0Ty, self, dimValue, start, end, step);
          start = end;
          outputs.push_back(slice);
        }

        Value end = vDimSize;
        Value lastSlice = rewriter.create<Torch::AtenSliceTensorOp>(
            loc, resultNTy, self, dimValue, start, end, step);
        outputs.push_back(lastSlice);

        rewriter.replaceOp(binder.op, outputs);

        return success();
      });

  // split with variable parts
  // Arguments:
  // - input: the tensor to split
  // - split: the sizes of the splits to be produced
  // Attributes:
  // - axis: the axis along which to split the input
  // - num_outputs: the number of outputs to produce
  // Outputs:
  // - outputs: the produced outputs. Variadic with num_outputs elements.
  // Note: torch.aten gives a list of tensors, but ONNX gives a variadic list of
  // tensors
  //       so we need to unpack the list
  patterns.onOp(
      "Split", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value self;
        Value split;
        int64_t axis;
        int64_t num_outputs;
        if (binder.tensorOperandAtIndex(self, 0) ||
            binder.tensorOperandAtIndex(split, 1))
          return rewriter.notifyMatchFailure(
              binder.op, "Not converting to AtenSplitWithSizesOp due to input "
                         "tensor mismatch");
        if (binder.s64IntegerAttr(axis, "axis", 0))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get axis attribute");
        if (binder.s64IntegerAttr(num_outputs, "num_outputs", 0))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to get num_outputs attribute");

        auto result0Ty =
            cast<Torch::ValueTensorType>(binder.op->getResult(0).getType());
        auto selfTy =
            cast<Torch::ValueTensorType>(binder.op->getOperand(0).getType());

        int64_t dim = axis;
        if (dim < 0)
          dim += selfTy.getSizes().size();

        llvm::SmallVector<int64_t> intermediateShape(result0Ty.getSizes());
        for (auto result : binder.op->getResultTypes()) {
          int64_t d = cast<Torch::ValueTensorType>(result).getSizes()[dim];
          intermediateShape[dim] = d == intermediateShape[dim] ? d : -1;
        }

        Torch::PrimTolistOp splitToList = rewriter.create<Torch::PrimTolistOp>(
            binder.getLoc(),
            Torch::ListType::get(rewriter.getType<Torch::IntType>()), split);

        Value dimValue = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), dim));

        // TODO: Attempting to use the shape expected by the ONNX mlir as ground
        // truth. For now just use dynamic shapes.
        auto resultOuterType =
            Torch::ListType::get(rewriter.getType<Torch::ValueTensorType>(
                /*std::optional<llvm::ArrayRef<int64_t>>=*/intermediateShape,
                result0Ty.getOptionalDtype()));
        Torch::AtenSplitWithSizesOp new_op =
            rewriter.create<Torch::AtenSplitWithSizesOp>(
                binder.getLoc(), resultOuterType, self,
                splitToList.getResult(0), dimValue);

        // the onnx op is variadic with multiple results, but AtenSplitWithSizes
        // outputs a list so we need to unpack the list
        rewriter.replaceOpWithNewOp<Torch::PrimListUnpackOp>(
            binder.op, binder.op->getResults().getType(), new_op.getResult());

        return success();
      });

  patterns.onOp("Tan", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenTanOp>(
                      binder.op, resultType, operand);
                  return success();
                });

  patterns.onOp(
      "Transpose", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        auto loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType))
          return failure();
        auto operandType = cast<Torch::ValueTensorType>(operand.getType());
        TensorType tensorType = operandType.toBuiltinTensor();
        if (!tensorType || !tensorType.hasRank())
          return failure();

        // Default permutation is to reverse orders:
        int64_t rank = tensorType.getRank();
        llvm::SmallVector<int64_t> reverse(rank);
        for (int64_t i = 0; i < rank; ++i) {
          reverse[i] = rank - i - 1;
        }

        llvm::SmallVector<int64_t> permutations;
        if (failed(binder.s64IntegerArrayAttr(permutations, "perm", reverse)))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to obtain permutations");

        if (static_cast<int64_t>(permutations.size()) != rank)
          return rewriter.notifyMatchFailure(
              binder.op, "Permutation length does not match operand rank");

        llvm::SmallVector<int64_t> shape(tensorType.getShape());
        llvm::SmallVector<int64_t> current(rank);
        for (int64_t i = 0; i < rank; ++i) {
          current[i] = i;
        }

        for (auto &dim : permutations)
          dim = dim < 0 ? dim + rank : dim;

        // We need to override to the destination if known:
        if (resultType.hasSizes()) {
          for (int i = 0; i < rank; ++i) {
            shape[permutations[i]] = resultType.getSizes()[i];
          }
        }

        // Convert dynamic shape dimension:
        for (unsigned i = 0; i < shape.size(); i++) {
          if (shape[i] == ShapedType::kDynamic)
            shape[i] = Torch::kUnknownSize;
        }

        for (int64_t i = 0; i < rank; ++i) {
          if (current[i] == permutations[i])
            continue;

          int64_t target = i + 1;
          for (; target < rank; ++target) {
            if (current[target] == permutations[i])
              break;
          }

          std::swap(shape[i], shape[target]);
          std::swap(current[i], current[target]);

          Value dim0 = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));

          Value dim1 = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), target));

          operand = rewriter.create<Torch::AtenTransposeIntOp>(
              loc,
              Torch::ValueTensorType::get(tensorType.getContext(), shape,
                                          operandType.getOptionalDtype()),
              operand, dim0, dim1);
        }

        rewriter.replaceOp(binder.op, operand);
        return success();
      });
  patterns.onOp(
      "Slice", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultTorchType;
        Value operand, starts, ends;
        // Handle if axes are not provided

        if (binder.tensorOperandAtIndex(operand, 0) ||
            binder.tensorOperandAtIndex(starts, 1) ||
            binder.tensorOperandAtIndex(ends, 2) ||
            binder.tensorResultType(resultTorchType)) {
          return failure();
        }

        auto context = rewriter.getContext();
        auto operandTorchTy = cast<Torch::ValueTensorType>(operand.getType());
        auto operandTy =
            dyn_cast<RankedTensorType>(operandTorchTy.toBuiltinTensor());

        if (!operandTy)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected tensor operator argument to be a ranked tensor type");

        auto startsTorchTy = cast<Torch::ValueTensorType>(starts.getType());
        auto startsTy =
            dyn_cast<RankedTensorType>(startsTorchTy.toBuiltinTensor());
        int startSize = startsTy.getDimSize(0);

        auto endsTorchTy = cast<Torch::ValueTensorType>(ends.getType());
        auto endsTy = dyn_cast<RankedTensorType>(endsTorchTy.toBuiltinTensor());
        int endSize = endsTy.getDimSize(0);
        auto resultTy =
            dyn_cast<RankedTensorType>(resultTorchType.toBuiltinTensor());
        if (!resultTy)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type to be a ranked tensor type");

        Location loc = binder.getLoc();

        // Binding `axes` from its arguments or through a default value
        Value axes;
        if (binder.getNumOperands() >= 4) {
          if (binder.tensorOperandAtIndex(axes, 3)) {
            return failure();
          }
        }

        // Binding `steps` from its arguments or through a default value
        Value steps;
        if (binder.getNumOperands() >= 5) {
          if (binder.tensorOperandAtIndex(steps, 4)) {
            return failure();
          }
        } else {
          // The default `steps` value is a 1d tensor filled with ones with a
          // size equal to the size of `starts` and `ends`.
          Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
          Value sizeStepInput = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), startSize));
          Value sizeStepsInput = rewriter.create<Torch::PrimListConstructOp>(
              loc,
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              sizeStepInput);
          steps = rewriter.create<Torch::AtenOnesOp>(
              loc, startsTorchTy, sizeStepsInput, none, none, none, none);
        }

        if (!(endsTy.getRank() == 1 && startsTy.getRank() == 1 &&
              startSize == endSize))
          return rewriter.notifyMatchFailure(
              binder.op, "Expected the rank of starts and ends tensors to be 1 "
                         "and their dimensions to match");

        if (axes) {
          auto axesTorchTy = cast<Torch::ValueTensorType>(axes.getType());
          auto axesTy =
              dyn_cast<RankedTensorType>(axesTorchTy.toBuiltinTensor());
          int64_t numAxes = axesTy.getDimSize(0);

          if (!(axesTy && numAxes == endSize))
            return rewriter.notifyMatchFailure(
                binder.op, "Axes should be the same size of starts and ends");
        }

        auto stepsTy = dyn_cast<RankedTensorType>(
            cast<Torch::ValueTensorType>(steps.getType()).toBuiltinTensor());

        if (!(stepsTy && stepsTy.getDimSize(0) == endsTy.getDimSize(0)))
          return rewriter.notifyMatchFailure(
              binder.op, "Steps should be the same size of starts and ends");

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

        auto select = [&](Value v, Value k) -> Value {
          auto ty = cast<Torch::ValueTensorType>(v.getType());
          auto sel = rewriter.create<Torch::AtenIndexSelectOp>(
              loc,
              Torch::ValueTensorType::get(ty.getContext(), ArrayRef<int64_t>{1},
                                          ty.getOptionalDtype()),
              v, zero, k);
          Value item = rewriter.create<Torch::AtenItemOp>(
              loc, rewriter.getType<Torch::IntType>(), sel);
          return item;
        };

        llvm::SmallVector<int64_t> intermediateShape(operandTy.getShape());
        for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
          if (operandTy.getDimSize(i) != resultTy.getDimSize(i))
            intermediateShape[i] = -1;
          if (intermediateShape[i] == ShapedType::kDynamic)
            intermediateShape[i] = Torch::kUnknownSize;
        }
        auto intermediateType = Torch::ValueTensorType::get(
            context, intermediateShape, resultTorchType.getOptionalDtype());
        for (int i = 0; i < endSize; ++i) {

          Value k = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value kTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
              loc,
              Torch::ValueTensorType::get(
                  context, ArrayRef<int64_t>{1},
                  rewriter.getIntegerType(64, /*signed*/ 1)),
              k);

          Value start = select(starts, kTensor);
          Value end = select(ends, kTensor);
          Value axis = axes ? select(axes, kTensor) : k;
          Value step = select(steps, kTensor);

          auto sliceType = intermediateType;
          sliceType = i == (endSize - 1) ? resultTorchType : sliceType;
          operand = rewriter.create<Torch::AtenSliceTensorOp>(
              loc, sliceType, operand, axis, start, end, step);
        }

        rewriter.replaceOp(binder.op, operand);
        return success();
      });
  patterns.onOp(
      "Reshape", 5, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        Value shape;
        int64_t allowzero;
        if (binder.tensorOperands(data, shape) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(allowzero, "allowzero", 0))
          return failure();

        // If the result shape is static then we can create a result shape list
        // directly using the result shape values (integers).
        if (resultType.hasSizes()) {
          bool hasStaticShape = resultType.areAllSizesKnown();
          ArrayRef<int64_t> resultShapeInt = resultType.getSizes();
          if (hasStaticShape) {
            SmallVector<Value> resultShape;
            for (int64_t dim : resultShapeInt) {
              resultShape.push_back(rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getI64IntegerAttr(dim)));
            }
            Value resultShapeList = rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                Torch::ListType::get(
                    Torch::IntType::get(binder.op->getContext())),
                resultShape);
            rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(
                binder.op, resultType, data, resultShapeList);
            return success();
          }
        }

        Torch::BaseTensorType shapeType =
            cast<Torch::BaseTensorType>(shape.getType());
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Type selectResultType = shapeType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), shapeType.getOptionalDtype());
        auto shapeSizes =
            dyn_cast<Torch::ValueTensorType>(shape.getType()).getSizes();
        auto dataSizes =
            dyn_cast<Torch::ValueTensorType>(data.getType()).getSizes();
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        if (allowzero == 0) {
          // convert shape (tensor) into torch int list while dealing with zero
          // vals
          for (int i = 0; i < shapeSizes[0]; i++) {
            // Go through the shape list and get each dim in the list
            Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
            Value extract = rewriter.create<Torch::AtenSelectIntOp>(
                binder.getLoc(), selectResultType, shape, zero, selectIndex);
            Value dim = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
            // deal with zero axis values: replace with original dim value in
            // input
            Value isZero =
                rewriter.create<Torch::AtenEqIntOp>(binder.getLoc(), dim, zero);
            isZero =
                rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(), isZero);

            int64_t dataRank = dataSizes.size();
            if (i < dataRank) {
              auto torchIntTy = rewriter.getType<Torch::IntType>();
              auto int64Ty = rewriter.getIntegerType(64, true);
              auto dimTy = rewriter.getType<Torch::ValueTensorType>(
                  ArrayRef<int64_t>(), int64Ty);
              auto boolTy = rewriter.getType<Torch::ValueTensorType>(
                  ArrayRef<int64_t>(), rewriter.getI1Type());
              Value iv = rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getI64IntegerAttr(i));
              Value inDim = rewriter.create<Torch::AtenSizeIntOp>(
                  binder.getLoc(), torchIntTy, data, iv);
              isZero = rewriter.create<Torch::PrimNumToTensorScalarOp>(
                  binder.getLoc(), boolTy, isZero);
              inDim = rewriter.create<Torch::PrimNumToTensorScalarOp>(
                  binder.getLoc(), dimTy, inDim);
              dim = rewriter.create<Torch::PrimNumToTensorScalarOp>(
                  binder.getLoc(), dimTy, dim);
              Value finalDim = rewriter.create<Torch::AtenWhereSelfOp>(
                  binder.getLoc(), dimTy, isZero, inDim, dim);
              dim = rewriter.create<Torch::AtenItemOp>(
                  binder.getLoc(), rewriter.getType<Torch::IntType>(),
                  finalDim);
            }
            dimList.push_back(dim);
          }
          Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(),
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              dimList);
          rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(
              binder.op, resultType, data, dimValueList);
          return success();
        }
        // convert axes (tensor) into torch int list
        for (int i = 0; i < shapeSizes[0]; i++) {
          // Go through the axes list and get each dim in the list
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, shape, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          dimList.push_back(dim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(binder.op, resultType,
                                                          data, dimValueList);
        return success();
      });
  patterns.onOp(
      "ReduceProd", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // ReduceProd allows us to pass a list of dims but AtenProdDimIn only
        // allow one dim as input.
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        int64_t keepDims;
        int64_t noop_with_empty_axes;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();

        auto dataTy = cast<Torch::BaseTensorType>(data.getType());
        Torch::IntType torchIntTy = rewriter.getType<Torch::IntType>();

        if (!resultType.hasSizes() || !resultType.areAllSizesKnown() ||
            !dataTy.areAllSizesKnown())
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected the input and result type to have known sizes");

        int64_t rank = dataTy.getSizes().size();
        SmallVector<Value> axesList;
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));

        // Previous version of the operation had the axes as an attribute:
        llvm::SmallVector<int64_t> axesAttr;
        if (!binder.s64IntegerArrayAttr(axesAttr, "axes", {})) {
          for (int i = 0, s = axesAttr.size(); i < s; ++i) {
            axesList.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), torchIntTy,
                rewriter.getI64IntegerAttr(axesAttr[i])));
          }
        }

        // Handle cases that axes are explicitly specified.
        // Extract the axes values from the axes operand.
        // This really shouldn't happen but it helps pass weird tests.
        // TODO: Derive the chosen axes from the data type and final result type
        // instead of using the dynamic axes at operand[1].
        if (!binder.tensorOperandAtIndex(axes, 1)) {
          Torch::BaseTensorType axesType =
              cast<Torch::BaseTensorType>(axes.getType());
          auto sizes = axesType.getSizes();
          for (int i = 0; i < sizes[0]; i++) {
            Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i));
            Value extract = rewriter.create<Torch::AtenSelectIntOp>(
                binder.getLoc(),
                axesType.getWithSizesAndDtype(llvm::SmallVector<int64_t>{1},
                                              axesType.getOptionalDtype()),
                axes, zero, selectIndex);
            Value dim = rewriter.create<Torch::AtenItemOp>(binder.getLoc(),
                                                           torchIntTy, extract);
            axesList.push_back(dim);
          }
        }

        // Handle the noop case:
        // When axes is empty and noop_with_empty_axes is set to true, input
        // tensor will not be reduced, and the output tensor would be
        // equivalent to input tensor.
        if (axesList.empty() && noop_with_empty_axes) {
          rewriter.replaceOp(binder.op, data);
          return success();
        }

        // Handle case when no axes arg is passed but not a noop:
        // Manually set positive axis to all dims.
        if (axesList.empty()) {
          for (int i = 0; i < rank; i++) {
            Value dimValue = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i));
            axesList.push_back(dimValue);
          }
        }

        // Handle negative axis:
        Value rankVal = rewriter.create<Torch::AtenDimOp>(binder.getLoc(),
                                                          torchIntTy, data);
        for (Value &axes : axesList) {
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), axes, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, rankVal);
          axes = rewriter.create<Torch::AtenAddIntOp>(binder.getLoc(), axes,
                                                      finalOffset);
        }

        // Handle multiple axes case:
        // ReduceProd on each dim, always set keepDimsBool == True to avoid
        // segfault.
        Value trueVal =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        SmallVector<int64_t> intermediateShape(rank, Torch::kUnknownSize);
        Value dataReduceProd = data;
        for (int i = 0, numAxes = axesList.size(); i < numAxes; i++) {
          auto axis = axesList[i];
          if (keepDims && i == numAxes - 1) {
            dataReduceProd = rewriter.create<Torch::AtenProdDimIntOp>(
                binder.getLoc(),
                dataTy.getWithSizesAndDtype(resultType.getSizes(),
                                            dataTy.getOptionalDtype()),
                dataReduceProd, axis, trueVal, noneVal);
            rewriter.replaceOp(binder.op, dataReduceProd);
            return success();
          }
          Type resultTyReduceProd = dataTy.getWithSizesAndDtype(
              ArrayRef(intermediateShape), dataTy.getOptionalDtype());
          dataReduceProd = rewriter.create<Torch::AtenProdDimIntOp>(
              binder.getLoc(), resultTyReduceProd, dataReduceProd, axis,
              trueVal, noneVal);
        }

        // Derived the final shape of the tensor after prod loop of each axis.
        SmallVector<int64_t> dataReduceProdSize;
        auto dataSize = dataTy.getSizes();
        auto resultTypeSizes = resultType.getSizes();
        if (!keepDims) {
          // Handle the keepDimsBool == False case:
          // 2 point algorithm to derive the static shape after prod loop.
          int j = 0;
          for (int i = 0; i < rank; i++) {
            if (resultTypeSizes.size() && dataSize[i] == resultTypeSizes[j]) {
              dataReduceProdSize.push_back(resultTypeSizes[i]);
              j++;
              continue;
            }
            dataReduceProdSize.push_back(1);
          }
        }

        // Handle the keepDimsBool == False case:
        // Reshape the prod loop result to the final result shape.
        SmallVector<Value> dataReduceProdShape;
        for (auto dim : dataReduceProdSize)
          dataReduceProdShape.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(dim)));
        Value dataReduceProdShapeList =
            rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                rewriter.getType<Torch::ListType>(
                    rewriter.getType<Torch::IntType>()),
                dataReduceProdShape);
        rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(
            binder.op, resultType, dataReduceProd, dataReduceProdShapeList);
        return success();
      });
  patterns.onOp(
      "Range", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // ONNX.Range(start, limit, delta) -- limit is exclusive

        Torch::ValueTensorType resultType;
        Value start, limit, delta;
        auto loc = binder.getLoc();
        Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
        if (binder.tensorOperandAtIndex(start, 0) ||
            binder.tensorOperandAtIndex(limit, 1) ||
            binder.tensorOperandAtIndex(delta, 2) ||
            binder.tensorResultType(resultType))
          return failure();

        // Convert a 0-dimensional/Scalar Tensor ([]) to Scalar Torch Numeric
        // Value torch.tensor(1.1) equivalent in ONNX to 1.1 as an example
        // type of start, limit, delta can be one of: double, float, int16,
        // int32, int64 Assuming start, limit and delta to be same type (could
        // they be different?)
        Torch::BaseTensorType startTensorType =
            cast<Torch::BaseTensorType>(start.getType());
        bool isFloatDType = startTensorType.getDtype().isF64() ||
                            startTensorType.getDtype().isF32();
        bool isIntDType = startTensorType.getDtype().isInteger(16) ||
                          startTensorType.getDtype().isInteger(32) ||
                          startTensorType.getDtype().isInteger(64);
        if (!isFloatDType && !isIntDType) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected the start, limit, delta to be one of "
                         "double, float, int16, int32, int64");
        }
        Value scalarStart, scalarLimit, scalarDelta;
        if (isFloatDType) {
          scalarStart = getItemOp<Torch::FloatType>(binder, rewriter, start);
          scalarLimit = getItemOp<Torch::FloatType>(binder, rewriter, limit);
          scalarDelta = getItemOp<Torch::FloatType>(binder, rewriter, delta);
        } else {
          scalarStart = getItemOp<Torch::IntType>(binder, rewriter, start);
          scalarLimit = getItemOp<Torch::IntType>(binder, rewriter, limit);
          scalarDelta = getItemOp<Torch::IntType>(binder, rewriter, delta);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenArangeStartStepOp>(
            binder.op, resultType, scalarStart, scalarLimit, scalarDelta, none,
            none, none, none);
        return success();
      });
  patterns.onOp(
      "Size", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType))
          return failure();

        auto loc = binder.getLoc();
        auto &op = binder.op;
        auto operandTy = cast<Torch::BaseTensorType>(operand.getType());

        if (!operandTy.hasSizes())
          return rewriter.notifyMatchFailure(op, "input rank unknown");

        llvm::SmallVector<Value> dims;
        int64_t rank = operandTy.getSizes().size();
        for (int i = 0; i < rank; ++i) {
          auto iv = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i));
          Value dim = rewriter.create<Torch::AtenSizeIntOp>(
              loc, rewriter.getType<Torch::IntType>(), operand, iv);
          dims.push_back(dim);
        }

        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
        Value none = rewriter.create<Torch::ConstantNoneOp>(loc);

        if (dims.empty()) {
          Value one = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(1));
          rewriter.replaceOpWithNewOp<Torch::AtenTensorIntOp>(
              op, resultType, one, none, none, cstFalse);
          return success();
        }

        Value prod = dims[0];
        for (int i = 1, s = dims.size(); i < s; ++i)
          prod = rewriter.create<Torch::AtenMulIntOp>(loc, prod, dims[i]);

        rewriter.replaceOpWithNewOp<Torch::AtenTensorIntOp>(
            op, resultType, prod, none, none, cstFalse);
        return success();
      });
  patterns.onOp(
      "Tile", 6, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        Value repeatDims;
        if (binder.tensorOperands(operand, repeatDims) ||
            binder.tensorResultType(resultType))
          return failure();

        // convert repeatDims tensor to list of ints
        auto repeatDimsSizes =
            dyn_cast<Torch::ValueTensorType>(repeatDims.getType()).getSizes();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Torch::BaseTensorType shapeType =
            cast<Torch::BaseTensorType>(repeatDims.getType());
        Type selectResultType = shapeType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), shapeType.getOptionalDtype());
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        for (int i = 0; i < repeatDimsSizes[0]; i++) {
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, repeatDims, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          dimList.push_back(dim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);

        rewriter.replaceOpWithNewOp<Torch::AtenTileOp>(binder.op, resultType,
                                                       operand, dimValueList);
        return success();
      });
  patterns.onOp(
      "TopK", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType Values_type, Indices_type;
        Value input, kValue;
        int64_t axis;
        bool largest, sorted;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(kValue, 1) ||
            binder.s64IntegerAttr(axis, "axis", -1) ||
            binder.s64BoolAttr(largest, "largest", true) ||
            binder.s64BoolAttr(sorted, "sorted", true) ||
            binder.tensorResultTypeAtIndex(Values_type, 0) ||
            binder.tensorResultTypeAtIndex(Indices_type, 1))
          return failure();
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        unsigned rank = *maybeRank;
        axis = Torch::toPositiveDim(axis, rank);
        Value cstAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(axis));
        Value cstLargest =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), largest);
        Value cstSorted =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), sorted);
        Value kValueInt = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), kValue);
        rewriter.replaceOpWithNewOp<Torch::AtenTopkOp>(
            binder.op, Values_type, Indices_type, input, kValueInt, cstAxis,
            cstLargest, cstSorted);
        return success();
      });
  patterns.onOp("Sign", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();

                  rewriter.replaceOpWithNewOp<Torch::AtenSignOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Softplus", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        if (binder.tensorOperand(input) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }
        // out = ln(exp(x) + 1)
        Value exp = rewriter.create<Torch::AtenExpOp>(binder.getLoc(),
                                                      resultType, input);
        rewriter.replaceOpWithNewOp<Torch::AtenLog1pOp>(binder.op, resultType,
                                                        exp);
        return success();
      });
  patterns.onOp("Softsign", 22,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value input;
                  if (binder.tensorOperand(input) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }

                  Value absX = rewriter.create<Torch::AtenAbsOp>(
                      binder.getLoc(), resultType, input);

                  Value constOne = rewriter.create<Torch::ConstantIntOp>(
                      binder.getLoc(), rewriter.getI64IntegerAttr(1));

                  Value absXPlusOne = rewriter.create<Torch::AtenAddScalarOp>(
                      binder.getLoc(), resultType, absX, constOne, constOne);

                  rewriter.replaceOpWithNewOp<Torch::AtenDivTensorOp>(
                      binder.op, resultType, input, absXPlusOne);
                  return success();
                });
  patterns.onOp(
      "Trilu", 14, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        int64_t upper;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.s64IntegerAttr(upper, "upper", 1) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        Value diagonal;
        if (binder.tensorOperandAtIndex(diagonal, 1)) {
          diagonal = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(0));
        } else {
          diagonal = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), diagonal);
        }

        if (upper) {
          rewriter.replaceOpWithNewOp<Torch::AtenTriuOp>(binder.op, resultType,
                                                         input, diagonal);
          return success();
        }
        rewriter.replaceOpWithNewOp<Torch::AtenTrilOp>(binder.op, resultType,
                                                       input, diagonal);
        return success();
      });
  patterns.onOp("ThresholdedRelu", 10,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value input;
                  float alpha;
                  if (binder.tensorOperand(input) ||
                      binder.f32FloatAttr(alpha, "alpha", 1.0)) {
                    return failure();
                  }
                  Value cstAlpha = rewriter.create<Torch::ConstantFloatOp>(
                      binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                      rewriter.getFloatAttr(rewriter.getF64Type(), alpha));
                  Value value = rewriter.create<Torch::ConstantFloatOp>(
                      binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                      rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
                  rewriter.replaceOpWithNewOp<Torch::AtenThresholdOp>(
                      binder.op, resultType, input, cstAlpha, value);
                  return success();
                });
  patterns.onOp(
      "RandomNormal", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallString<64> name("torch.onnx.seed");
        auto seedAttr = binder.op->getAttr(name);
        if (seedAttr)
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for seed attribute");

        Torch::ValueTensorType resultType;
        int64_t dtypeIntOnnx;
        float mean, scale;
        SmallVector<int64_t> shape;
        if (binder.s64IntegerAttr(dtypeIntOnnx, "dtype", 1) ||
            binder.f32FloatAttr(mean, "mean", 0.0) ||
            binder.f32FloatAttr(scale, "scale", 1.0) ||
            binder.s64IntegerArrayAttr(shape, "shape", {}) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));

        Value shapeList = createConstantIntList(binder, rewriter, shape);
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value self = rewriter.create<Torch::AtenEmptyMemoryFormatOp>(
            binder.op->getLoc(), resultType, shapeList,
            /*dtype=*/constDtype,
            /*layout=*/cstNone,
            /*device=*/cstNone, /*pinMemory=*/cstNone,
            /*memoryFormat=*/cstNone);

        Value cstMean = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), mean));
        Value cstStd = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), scale));

        rewriter.replaceOpWithNewOp<Torch::AtenNormalFunctionalOp>(
            binder.op, resultType, self, cstMean, cstStd,
            /*generator=*/cstNone);
        return success();
      });
  patterns.onOp(
      "RandomNormalLike", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallString<64> name("torch.onnx.seed");
        auto seedAttr = binder.op->getAttr(name);
        if (seedAttr)
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for seed attribute");

        Torch::ValueTensorType resultType;
        int64_t dtypeIntOnnx;
        float mean, scale;
        SmallVector<int64_t> shape;
        Value input;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(dtypeIntOnnx, "dtype", 1) ||
            binder.f32FloatAttr(mean, "mean", 0.0) ||
            binder.f32FloatAttr(scale, "scale", 1.0) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));

        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        input = rewriter.create<Torch::AtenToDtypeOp>(
            binder.op->getLoc(), resultType, input, constDtype,
            /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
            /*memory_format=*/cstNone);

        Value cstMean = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), mean));
        Value cstStd = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), scale));

        rewriter.replaceOpWithNewOp<Torch::AtenNormalFunctionalOp>(
            binder.op, resultType, input, cstMean, cstStd,
            /*generator=*/cstNone);
        return success();
      });
  patterns.onOp(
      "RandomUniform", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallString<64> name("torch.onnx.seed");
        auto seedAttr = binder.op->getAttr(name);
        if (seedAttr)
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for seed attribute");

        Torch::ValueTensorType resultType;
        int64_t dtypeIntOnnx;
        float high, low;
        SmallVector<int64_t> shape;
        if (binder.s64IntegerAttr(dtypeIntOnnx, "dtype", 1) ||
            binder.f32FloatAttr(high, "high", 1.0) ||
            binder.f32FloatAttr(low, "low", 0.0) ||
            binder.s64IntegerArrayAttr(shape, "shape", {}) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));

        Value shapeList = createConstantIntList(binder, rewriter, shape);
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value self = rewriter.create<Torch::AtenEmptyMemoryFormatOp>(
            binder.op->getLoc(), resultType, shapeList,
            /*dtype=*/constDtype,
            /*layout=*/cstNone,
            /*device=*/cstNone, /*pinMemory=*/cstNone,
            /*memoryFormat=*/cstNone);

        Value cstHigh = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), high));
        Value cstLow = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), low));

        rewriter.replaceOpWithNewOp<Torch::AtenUniformOp>(
            binder.op, resultType, self, cstLow, cstHigh,
            /*generator=*/cstNone);
        return success();
      });
  patterns.onOp(
      "RandomUniformLike", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallString<64> name("torch.onnx.seed");
        auto seedAttr = binder.op->getAttr(name);
        if (seedAttr)
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for seed attribute");

        Torch::ValueTensorType resultType;
        int64_t dtypeIntOnnx;
        float high, low;
        SmallVector<int64_t> shape;
        Value input;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(dtypeIntOnnx, "dtype", 1) ||
            binder.f32FloatAttr(high, "high", 1.0) ||
            binder.f32FloatAttr(low, "low", 0.0) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));

        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        input = rewriter.create<Torch::AtenToDtypeOp>(
            binder.op->getLoc(), resultType, input, constDtype,
            /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
            /*memory_format=*/cstNone);

        Value cstHigh = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), high));
        Value cstLow = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), low));

        rewriter.replaceOpWithNewOp<Torch::AtenUniformOp>(
            binder.op, resultType, input, cstLow, cstHigh,
            /*generator=*/cstNone);
        return success();
      });
  patterns.onOp(
      "SoftmaxCrossEntropyLoss", 12,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        int64_t ignoreIndex;
        std::string reduction;
        SmallVector<int64_t> shape;
        Value scores, labels, weight;
        if (binder.tensorOperandAtIndex(scores, 0) ||
            binder.tensorOperandAtIndex(labels, 1) ||
            binder.s64IntegerAttr(ignoreIndex, "ignore_index", -100) ||
            binder.customOpNameStringAttr(reduction, "reduction", "mean") ||
            binder.tensorResultTypeAtIndex(resultType, 0)) {
          return failure();
        }

        if (binder.tensorOperandAtIndex(weight, 2))
          weight = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value cstIgnoreIndex = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(ignoreIndex));

        int64_t reductionInt = reduction == "none"   ? 0
                               : reduction == "mean" ? 1
                                                     : 2;
        Value cstReductionInt = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(reductionInt));

        // The default PyTorch value for label smoothing is "0.0".
        // Refer:
        // https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        Value cstLabelSmoothing = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));

        Value loss = rewriter.create<Torch::AtenCrossEntropyLossOp>(
            binder.getLoc(), resultType, scores, labels, weight,
            cstReductionInt, cstIgnoreIndex, cstLabelSmoothing);

        if (binder.op->getNumResults() == 1) {
          rewriter.replaceOp(binder.op, loss);
          return success();
        }

        Torch::ValueTensorType resultTypeLogProb;
        if (binder.tensorResultTypeAtIndex(resultTypeLogProb, 1))
          return failure();

        Value dim = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value logProb = rewriter.create<Torch::AtenLogSoftmaxIntOp>(
            binder.getLoc(), resultTypeLogProb, scores, dim, /*dtype=*/cstNone);

        rewriter.replaceOp(binder.op, {loss, logProb});
        return success();
      });
  patterns.onOp(
      "Resize", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        std::string mode, nearest_mode, coordTfMode;
        int64_t antialias, exclude_outside;
        float extrapolation_value;
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        if (auto attr = binder.op->getAttr("torch.onnx.axes")) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for axes attribute");
        }
        if (auto attr =
                binder.op->getAttr("torch.onnx.keep_aspect_ratio_policy")) {
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: support not present for "
                         "keep_aspect_ratio_policy attribute");
        }

        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.customOpNameStringAttr(mode, "mode", "nearest") ||
            binder.customOpNameStringAttr(
                coordTfMode, "coordinate_transformation_mode", "half_pixel") ||
            binder.s64IntegerAttr(antialias, "antialias", 0) ||
            binder.s64IntegerAttr(exclude_outside, "exclude_outside", 0) ||
            binder.f32FloatAttr(extrapolation_value, "extrapolation_value",
                                0.0) ||
            binder.customOpNameStringAttr(nearest_mode, "nearest_mode",
                                          "round_prefer_floor"))
          return failure();
        if (antialias != 0) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for antialias attribute");
        }
        if (exclude_outside != 0) {
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: support not present for "
                         "exclude_outside attribute");
        }
        if (extrapolation_value != 0.0) {
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: support not present for "
                         "extrapolation_value attribute");
        }
        if (coordTfMode == "tf_crop_and_resize")
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: coordinate transformation mode: "
                         "tf_crop_and_resize");

        if (mode == "nearest" && coordTfMode != "asymmetric" &&
            coordTfMode != "half_pixel") {
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: support not present for coord tf mode "
                         "except asymmetric and half_pixel");
        }

        unsigned rank = dyn_cast<Torch::ValueTensorType>(operands[0].getType())
                            .getSizes()
                            .size();

        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value cstTrue =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        Value modeStrValue;

        Value scalesValueList = noneVal;
        Value sizesValueList = noneVal;
        Value alignCorners =
            coordTfMode == "align_corners" ? cstTrue : cstFalse;
        if (mode == "cubic") {
          return rewriter.notifyMatchFailure(binder.op,
                                             "unimplemented: bicubic mode");
        }
        // supported modes:
        // bilinear (half_pixel), bilinear with align_corners,
        // bilinear_pytorch_half_pixel, bilinear_asymmetric nearest
        // (asymmetric), nearest with align_corners, nearest_half_pixel,
        // nearest_pytorch_half_pixel
        if (mode == "linear") {
          std::string modeStr;
          switch (rank) {
          case 3:
            modeStr = "linear";
            break;
          case 4:
            modeStr = "bilinear";
            break;
          case 5:
            modeStr = "trilinear";
            break;
          default:
            return failure();
          }
          // Confusingly enough, the default coordTfMode for pytorch bilinear
          // mode is apparently half_pixel, NOT pytorch_half_pixel
          if (coordTfMode != "half_pixel" && coordTfMode != "align_corners")
            modeStr = (modeStr + "_") + coordTfMode;
          modeStrValue =
              rewriter.create<Torch::ConstantStrOp>(binder.getLoc(), modeStr);
        }
        if (mode == "nearest") {
          std::string modeStr = "nearest";
          // The default coordTfMode for pytorch with mode = nearest is
          // apparently asymmetric
          if (coordTfMode != "asymmetric" && coordTfMode != "align_corners")
            modeStr = (modeStr + "_") + coordTfMode;
          if (nearest_mode != "floor" && nearest_mode != "")
            modeStr = modeStr + "," + nearest_mode;
          modeStrValue =
              rewriter.create<Torch::ConstantStrOp>(binder.getLoc(), modeStr);
        }
        if (operands.size() < 4) {
          Value scaleOperand = operands[2];
          scalesValueList = getValueList(binder, rewriter, scaleOperand);
          sizesValueList = noneVal;
        } else {
          Value sizeOperand = operands[3];
          scalesValueList = noneVal;
          sizesValueList = getValueList(binder, rewriter, sizeOperand);
        }
        if (isa<Torch::NoneType>(scalesValueList.getType()) &&
            isa<Torch::NoneType>(sizesValueList.getType())) {
          return rewriter.notifyMatchFailure(binder.op, "unknown scaling mode");
        }
        rewriter
            .replaceOpWithNewOp<Torch::Aten__InterpolateSizeListScaleListOp>(
                binder.op, resultType, operands[0], sizesValueList,
                scalesValueList, modeStrValue,
                /* AnyTorchOptionalBoolType:$align_corners */ alignCorners,
                /* AnyTorchOptionalBoolType:$recompute_scale_factor */ noneVal,
                /*Torch_BoolType:$antialias*/ cstFalse);
        return success();
      });
  patterns.onOp(
      "RoiAlign", 16, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // operands = input, rois, batch_indices
        SmallVector<Value> operands;
        std::string coordTfMode, mode;
        int64_t outHInt, outWInt, samplingRatioInt;
        float spatialScaleFloat;
        Torch::ValueTensorType resultType;
        if (binder.tensorOperands(operands, 3) ||
            binder.customOpNameStringAttr(
                coordTfMode, "coordinate_transformation_mode", "half_pixel") ||
            binder.customOpNameStringAttr(mode, "mode", "avg") ||
            binder.s64IntegerAttr(outHInt, "output_height", 1) ||
            binder.s64IntegerAttr(outWInt, "output_width", 1) ||
            binder.s64IntegerAttr(samplingRatioInt, "sampling_ratio", 0) ||
            binder.f32FloatAttr(spatialScaleFloat, "spatial_scale", 1.0f) ||
            binder.tensorResultType(resultType))
          return failure();
        Value input = operands[0];
        Value rois = operands[1];
        Value batchIndices = operands[2];

        // the torchvision roi_pool op does not support these features:
        if (mode == "max" &&
            (coordTfMode != "half_pixel" || samplingRatioInt != 0))
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported: roi max pooling without default "
                         "coordTfMode and sampling_ratio");

        Location loc = binder.getLoc();
        // concatenate the batchIndices to the rois to get rois as a num_roisx5
        // tensor. The batchIndices tensor is an int64 tensor, and needs to be
        // converted to float before concatenation.
        auto roisType = dyn_cast<Torch::ValueTensorType>(rois.getType());
        if (!roisType || !roisType.hasSizes())
          return failure();
        Value cstDim = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        FailureOr<Value> unsqueezeIndices =
            Torch::unsqueezeTensor(rewriter, binder.op, batchIndices, cstDim);
        if (failed(unsqueezeIndices))
          return failure();
        batchIndices = unsqueezeIndices.value();
        auto batchIndicesType =
            cast<Torch::ValueTensorType>(batchIndices.getType());
        Value dTypeInt =
            Torch::getDtypeIntValueForType(rewriter, loc, roisType.getDtype());
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value newBatchIndices = rewriter.create<Torch::AtenToDtypeOp>(
            loc,
            batchIndicesType.getWithSizesAndDtype(
                batchIndicesType.getOptionalSizes(),
                roisType.getOptionalDtype()),
            batchIndices, dTypeInt, cstFalse, cstFalse, none);
        SmallVector<int64_t> roiSizes(roisType.getSizes());
        roiSizes.back() = 5;
        auto catType = rewriter.getType<Torch::ValueTensorType>(
            roiSizes, roisType.getDtype());
        Type listElemType =
            roisType.getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                          /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = rewriter.create<Torch::PrimListConstructOp>(
            binder.op->getLoc(), listType, ValueRange{newBatchIndices, rois});
        Value newRois =
            rewriter.create<Torch::AtenCatOp>(loc, catType, tensorList, cstDim);

        // make constants from attributes
        Value cstSpatialScale = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(spatialScaleFloat));
        Value pooledHeight = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(outHInt));
        Value pooledWidth = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(outWInt));
        // this is for consistency with the default pytorch sampling ratio value
        samplingRatioInt = (samplingRatioInt == 0) ? -1 : samplingRatioInt;
        Value samplingRatio = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(samplingRatioInt));
        bool aligned = coordTfMode == "half_pixel";
        Value cstAligned = rewriter.create<Torch::ConstantBoolOp>(loc, aligned);

        if (mode == "avg") {
          rewriter.replaceOpWithNewOp<Torch::TorchvisionRoiAlignOp>(
              binder.op, resultType, input, newRois, cstSpatialScale,
              pooledHeight, pooledWidth, samplingRatio, cstAligned);
          return success();
        }
        // mode == "max"
        auto indicesType = resultType.getWithSizesAndDtype(
            resultType.getOptionalSizes(), batchIndicesType.getDtype());
        auto roiPool = rewriter.create<Torch::TorchvisionRoiPoolOp>(
            loc, TypeRange{resultType, indicesType}, input, newRois,
            cstSpatialScale, pooledHeight, pooledWidth);
        rewriter.replaceOp(binder.op, roiPool.getResult(0));
        return success();
      });
  patterns.onOp(
      "SpaceToDepth", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        int64_t blockSize;
        std::string mode;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(blockSize, "blocksize") ||
            binder.customOpNameStringAttr(mode, "mode", "DCR") ||
            binder.tensorResultType(resultType))
          return failure();
        auto inputTy = dyn_cast<Torch::BaseTensorType>(input.getType());
        if (!inputTy || !inputTy.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input type having sizes");
        }
        SmallVector<int64_t> inputSizes{inputTy.getSizes()};
        if (inputSizes.size() != 4) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected input rank to be 4");
        }

        Value b = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(0)));
        Value c = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(1)));
        Value h = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(2)));
        Value w = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(3)));
        Value cstBlockSize = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(blockSize));
        Value cstBlockSizeSquare = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(blockSize * blockSize));
        Value hDivBlockSize = rewriter.create<Torch::AtenDivIntOp>(
            binder.getLoc(), h, cstBlockSize);
        Value wDivBlockSize = rewriter.create<Torch::AtenDivIntOp>(
            binder.getLoc(), w, cstBlockSize);
        hDivBlockSize = rewriter.create<Torch::AtenIntFloatOp>(binder.getLoc(),
                                                               hDivBlockSize);
        wDivBlockSize = rewriter.create<Torch::AtenIntFloatOp>(binder.getLoc(),
                                                               wDivBlockSize);

        // The implementation is as follows:
        // tmp = np.reshape(
        //     x, [b, c, h // blocksize, blocksize, w // blocksize, blocksize]
        // )
        // tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
        // y = np.reshape(tmp, [b, c * (blocksize**2), h // blocksize, w //
        // blocksize])
        Value reshapeSizesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(input.getContext())),
            llvm::SmallVector<Value>{b, c, hDivBlockSize, cstBlockSize,
                                     wDivBlockSize, cstBlockSize});
        int64_t hDivBlockSizeInt = inputSizes[2] == Torch::kUnknownSize
                                       ? Torch::kUnknownSize
                                       : inputSizes[2] / blockSize;
        int64_t wDivBlockSizeInt = inputSizes[3] == Torch::kUnknownSize
                                       ? Torch::kUnknownSize
                                       : inputSizes[3] / blockSize;
        SmallVector<int64_t, 6> reshapeSizesInt{inputSizes[0],    inputSizes[1],
                                                hDivBlockSizeInt, blockSize,
                                                wDivBlockSizeInt, blockSize};
        Value reshapedInput = rewriter.create<Torch::AtenReshapeOp>(
            binder.getLoc(),
            inputTy.getWithSizesAndDtype(reshapeSizesInt,
                                         inputTy.getOptionalDtype()),
            input, reshapeSizesList);

        SmallVector<int64_t, 6> permuteDimsInt{0, 3, 5, 1, 2, 4};
        Value permutedInput;
        if (failed(createTorchPermuteOp(binder, rewriter, binder.getLoc(),
                                        reshapedInput, permuteDimsInt,
                                        permutedInput)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to create Torch Permute op");

        Value cMulBlockSizeSquare = rewriter.create<Torch::AtenMulIntOp>(
            binder.getLoc(), c, cstBlockSizeSquare);
        reshapeSizesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(input.getContext())),
            llvm::SmallVector<Value>{b, cMulBlockSizeSquare, hDivBlockSize,
                                     wDivBlockSize});
        rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(
            binder.op, resultType, permutedInput, reshapeSizesList);
        return success();
      });
  patterns.onOp(
      "Shrink", 9, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value input;
        float bias, lambd;
        if (binder.tensorOperand(input) ||
            binder.f32FloatAttr(bias, "bias", 0.0) ||
            binder.f32FloatAttr(lambd, "lambd", 0.5) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!isa<mlir::FloatType>(inputType.getDtype()))
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: non-floating point dtype");

        // The formula of this operator is: If x < -lambd, y = x + bias; If x >
        // lambd, y = x - bias; Otherwise, y = 0.
        // The implementation is based on the following algorithm:
        // Shrink <bias,lambd>(input) => (output)
        // {
        //    Lambd = Constant <value_float: float = @lambd> ()
        //    LambdCast = CastLike (Lambd, input)
        //    Bias = Constant <value_float: float = @bias> ()
        //    BiasCast = CastLike (Bias, input)
        //    Zero = Constant <value: tensor = float {0}> ()
        //    ZeroCast = CastLike (Zero, input)
        //    NegLmbda = Neg (LambdCast)
        //    InputLessThanNegLambda = Less (input, NegLmbda)
        //    InputAddBias = Add (input, BiasCast)
        //    InputSubBias = Sub (input, BiasCast)
        //    LambdaLessThanInput = Less (LambdCast, input)
        //    InputSubBiasOrZero = Where (LambdaLessThanInput, InputSubBias,
        //    ZeroCast) output = Where (InputLessThanNegLambda, InputAddBias,
        //    InputSubBiasOrZero)
        // }
        Value constLambd = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getFloatAttr(rewriter.getF64Type(), lambd));
        Value constBias = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getFloatAttr(rewriter.getF64Type(), bias));
        Value constZero = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
        Value constOne = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getFloatAttr(rewriter.getF64Type(), 1.0));
        Value constNegLambd = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getFloatAttr(rewriter.getF64Type(), -lambd));

        Value inputLTNegLambd = rewriter.create<Torch::AtenLtScalarOp>(
            loc, inputType, input, constNegLambd);
        Value inputPlusBias = rewriter.create<Torch::AtenAddScalarOp>(
            loc, inputType, input, constBias, /*alpha=*/constOne);
        Value inputSubBias = rewriter.create<Torch::AtenSubScalarOp>(
            loc, inputType, input, constBias, /*alpha=*/constOne);
        Value inputGTLambd = rewriter.create<Torch::AtenGtScalarOp>(
            loc, inputType, input, constLambd);

        Value inputSubBiasOrZero =
            rewriter.create<Torch::AtenWhereScalarOtherOp>(
                loc, resultType, inputGTLambd, inputSubBias, constZero);
        rewriter.replaceOpWithNewOp<Torch::AtenWhereSelfOp>(
            binder.op, resultType, inputLTNegLambd, inputPlusBias,
            inputSubBiasOrZero);
        return success();
      });
  patterns.onOp("SequenceAt", 11,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value inputSequence, position;
                  if (binder.tensorListOperandAtIndex(inputSequence, 0) ||
                      binder.tensorOperandAtIndex(position, 1) ||
                      binder.tensorResultType(resultType))
                    return failure();

                  Value index = rewriter.create<Torch::AtenItemOp>(
                      binder.getLoc(), rewriter.getType<Torch::IntType>(),
                      position);
                  rewriter.replaceOpWithNewOp<Torch::Aten__Getitem__TOp>(
                      binder.op, resultType, inputSequence, index);
                  return success();
                });
  patterns.onOp(
      "SequenceEmpty", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ListType resultType;
        int64_t dtypeIntOnnx;
        if (binder.s64IntegerAttr(dtypeIntOnnx, "dtype", 1) ||
            binder.tensorListResultType(resultType))
          return failure();

        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));

        Value shapeList = createConstantIntList(binder, rewriter, {});
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value self = rewriter.create<Torch::AtenEmptyMemoryFormatOp>(
            binder.op->getLoc(), resultType.getContainedType(), shapeList,
            /*dtype=*/constDtype,
            /*layout=*/cstNone,
            /*device=*/cstNone, /*pinMemory=*/cstNone,
            /*memoryFormat=*/cstNone);

        rewriter.replaceOpWithNewOp<Torch::PrimListConstructOp>(
            binder.op, resultType, llvm::SmallVector<Value>{self});
        return success();
      });
  patterns.onOp(
      "SequenceErase", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ListType resultType;
        Value inputSequence, position;
        if (binder.tensorListOperandAtIndex(inputSequence, 0) ||
            binder.tensorListResultType(resultType))
          return failure();

        Value length = rewriter.create<Torch::AtenLenTOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), inputSequence);

        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        if (binder.op->getNumOperands() == 1) {
          // If True, it means that the `position` arg is missing and
          // the last tensor from the list has to be erased.
          Value lengthMinusOne = rewriter.create<Torch::AtenSubIntOp>(
              binder.getLoc(), length, cstOne);
          rewriter.replaceOpWithNewOp<Torch::AtenSliceTOp>(
              binder.op, resultType, inputSequence, /*start=*/cstNone,
              /*end=*/lengthMinusOne, /*step=*/cstOne);
          return success();
        }

        if (binder.tensorOperandAtIndex(position, 1))
          return failure();

        Value positionInt = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), position);
        // Handling negative position value.
        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value isPositionNegative = rewriter.create<Torch::AtenLtIntOp>(
            binder.getLoc(), positionInt, cstZero);
        isPositionNegative = rewriter.create<Torch::AtenIntBoolOp>(
            binder.getLoc(), isPositionNegative);
        Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
            binder.getLoc(), isPositionNegative, length);
        positionInt = rewriter.create<Torch::AtenAddIntOp>(
            binder.getLoc(), positionInt, finalOffset);

        Value listBeforePosition = rewriter.create<Torch::AtenSliceTOp>(
            binder.getLoc(), resultType, inputSequence, /*start=*/cstNone,
            /*end=*/positionInt, /*step=*/cstOne);
        Value positionPlusOne = rewriter.create<Torch::AtenAddIntOp>(
            binder.getLoc(), positionInt, cstOne);
        Value listAfterPosition = rewriter.create<Torch::AtenSliceTOp>(
            binder.getLoc(), resultType, inputSequence,
            /*start=*/positionPlusOne,
            /*end=*/length, /*step=*/cstOne);

        rewriter.replaceOpWithNewOp<Torch::AtenAddTOp>(
            binder.op, resultType, listBeforePosition, listAfterPosition);
        return success();
      });
  patterns.onOp(
      "SequenceInsert", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ListType resultType;
        Value inputSequence, position, insertValue;
        if (binder.tensorListOperandAtIndex(inputSequence, 0) ||
            binder.tensorOperandAtIndex(insertValue, 1) ||
            binder.tensorListResultType(resultType))
          return failure();

        if (binder.op->getNumOperands() == 1) {
          // If True, it means that the `position` arg is missing and
          // the tensor has to be inserted at the end of the list.
          Value length = rewriter.create<Torch::AtenLenTOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              inputSequence);
          rewriter.replaceOpWithNewOp<Torch::AtenInsertTOp>(
              binder.op, inputSequence, /*idx=*/length,
              /*el=*/insertValue);
          return success();
        }

        if (binder.tensorOperandAtIndex(position, 2))
          return failure();

        Value positionInt = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), position);
        rewriter.create<Torch::AtenInsertTOp>(binder.getLoc(), inputSequence,
                                              /*idx=*/positionInt,
                                              /*el=*/insertValue);
        rewriter.replaceOp(binder.op, inputSequence);
        return success();
      });
  patterns.onOp(
      "SequenceMap", 17,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        llvm::SmallVector<Value> operands;
        Torch::ListType resultType;
        if (binder.tensorOperandsList(operands) || operands.size() == 0 ||
            binder.tensorListResultType(resultType)) {
          return failure();
        }

        Region *bodyRegion;
        if (binder.getRegionAtIndex(bodyRegion, 0)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed getting Body Region");
        }

        // construct an empty list, append results through the loop
        auto resultTensorType =
            dyn_cast<Torch::ValueTensorType>(resultType.getContainedType());
        Value shapeList = createConstantIntList(binder, rewriter,
                                                resultTensorType.getSizes());
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value self = rewriter.create<Torch::AtenEmptyMemoryFormatOp>(
            binder.op->getLoc(), resultType.getContainedType(), shapeList,
            /*dtype=*/cstNone, /*layout=*/cstNone, /*device=*/cstNone,
            /*pinMemory=*/cstNone, /*memoryFormat=*/cstNone);
        Value result = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(), resultType, llvm::SmallVector<Value>{self});

        // create a for-like primLoopOp
        // with the length of sequence as max iter_num
        Value len = rewriter.create<Torch::AtenLenTOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), operands[0]);
        auto cstTrue = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(true));
        mlir::ImplicitLocOpBuilder b(binder.getLoc(), rewriter);
        auto loop =
            b.create<Torch::PrimLoopOp>(resultType, len, cstTrue, result);
        rewriter.cloneRegionBefore(*bodyRegion, loop.getRegion(),
                                   loop.getRegion().begin());

        // primLoopOp loopBody expects torch.int as first arg
        // remove inputs from the region and use it from outside
        loop.getRegion().front().insertArgument(0U, resultType,
                                                binder.getLoc());
        Value sequenceArg = loop.getRegion().front().getArgument(0);
        loop.getRegion().front().insertArgument(
            0U, rewriter.getType<Torch::IntType>(), binder.getLoc());
        Value indexArg = loop.getRegion().front().getArgument(0);

        // get sequence[i] (and addtionalInput[i]) in each iteration
        rewriter.setInsertionPointToStart(&loop.getRegion().front());
        for (size_t i = 0; i < operands.size(); i++) {
          Value argInput = loop.getRegion().front().getArgument(2);
          if (isa<Torch::ListType>(operands[i].getType())) {
            auto tensorType = dyn_cast<Torch::ValueTensorType>(
                dyn_cast<Torch::ListType>(operands[i].getType())
                    .getContainedType());
            Value item = rewriter.create<Torch::Aten__Getitem__TOp>(
                binder.getLoc(), tensorType, operands[i], indexArg);
            argInput.replaceAllUsesWith(item);
          } else {
            argInput.replaceAllUsesWith(operands[i]);
          }
          loop.getRegion().eraseArgument(2);
        }

        // replace terminator
        PatternRewriter::InsertionGuard guard(rewriter);
        Operation *terminator = loop.getRegion().front().getTerminator();
        rewriter.setInsertionPoint(terminator);
        // update sequence input
        auto terminatorOperands = terminator->getOperands();
        Value append = rewriter.create<Torch::AtenAppendTOp>(
            binder.getLoc(), resultType, sequenceArg, terminatorOperands[0]);
        rewriter.replaceOpWithNewOp<Torch::PrimLoopConditionOp>(
            terminator, cstTrue, append);

        rewriter.replaceOp(binder.op, loop);
        return success();
      });
  patterns.onOp(
      "Upsample", 9, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        std::string mode;
        Value input, scales;
        if (binder.tensorOperands(input, scales) ||
            binder.customOpNameStringAttr(mode, "mode", "nearest") ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        if (mode != "nearest" && mode != "linear")
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported interpolation mode other than nearest, "
                         "linear");

        int64_t resultRank = resultType.getSizes().size();
        if (resultRank > 5)
          return rewriter.notifyMatchFailure(
              binder.op, "supports upto 3d upsampling only");

        Value scalesValueList = getValueList(binder, rewriter, scales);
        if (mode == "linear") {
          if (resultRank == 4)
            mode = "bilinear";
          if (resultRank == 5)
            mode = "trilinear";
        }
        Value modeStrValue =
            rewriter.create<Torch::ConstantStrOp>(binder.getLoc(), mode);
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(false));

        rewriter
            .replaceOpWithNewOp<Torch::Aten__InterpolateSizeListScaleListOp>(
                binder.op, resultType, input, /*size=*/cstNone, scalesValueList,
                modeStrValue,
                /* AnyTorchOptionalBoolType:$align_corners */ cstNone,
                /* AnyTorchOptionalBoolType:$recompute_scale_factor */ cstNone,
                /*Torch_BoolType:$antialias*/ cstFalse);
        return success();
      });
  patterns.onOp(
      "STFT", 17, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // operands in order ->(signal, frameStep, window, frameLength*)
        SmallVector<Value> operands;
        int64_t onesided;
        Torch::ValueTensorType resultType;

        if (binder.tensorOperandsList(operands) ||
            binder.s64IntegerAttr(onesided, "onesided", 1) ||
            binder.tensorResultType(resultType))
          return failure();

        Value signal = operands[0];
        Value frameStep = operands[1];
        auto signalTy = cast<Torch::ValueTensorType>(signal.getType());
        auto signalShape = signalTy.getSizes();
        auto resultShape = resultType.getSizes();

        // There are two possible cases for optional inputs frameLength and
        // window, which are that either 4 operands will be passed with window
        // being !torch.none, or three operands will be passed, with window
        // present and frameLength absent. In the former case, we simply create
        // a rectangular window consisting of ones, and in the latter, we set
        // frameLength equal to the the inputShape[-2] or windowShape[0]
        // depending upon whether window was present or not. Note that it is
        // possible that both window and frameLength can be none, which would
        // mean that either only two operands were passed, or, in case of three
        // operands, window was passed in as none, and frameLength was absent.
        Value window = nullptr, frameLength = nullptr;
        bool windowIsNone = true, frameLengthIsNone = true;
        if (operands.size() == 3) {
          window = operands[2];
          windowIsNone = isa<Torch::NoneType>(window.getType());
        }
        if (operands.size() == 4) {
          window = operands[2];
          frameLength = operands[3];
          windowIsNone = isa<Torch::NoneType>(window.getType());
          frameLengthIsNone = isa<Torch::NoneType>(frameLength.getType());
        }

        ArrayRef<int64_t> windowShape;
        if (frameLengthIsNone) {
          if (windowIsNone) {
            frameLength = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(
                                     signalShape[signalShape.size() - 2]));
          } else {
            windowShape =
                cast<Torch::ValueTensorType>(window.getType()).getSizes();
            frameLength = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(windowShape[0]));
          }
        }

        Value frameLengthItem;
        if (!frameLengthIsNone || windowIsNone) {
          frameLengthItem =
              getItemOp<Torch::IntType>(binder, rewriter, frameLength);
        } else {
          frameLengthItem = frameLength;
        }
        Value frameStepItem =
            getItemOp<Torch::IntType>(binder, rewriter, frameStep);

        if (windowIsNone) {
          auto onesResultTy = rewriter.getType<Torch::ValueTensorType>(
              ArrayRef<int64_t>({-1}), signalTy.getDtype());

          Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
          Value sizes = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(),
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              SmallVector<Value>{frameLengthItem});
          window = rewriter.create<Torch::AtenOnesOp>(
              binder.getLoc(), onesResultTy, sizes, none, none, none, none);
        }

        FailureOr<Type> complexDtype;
        if (signalTy.getDtype().isBF16()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support for bfloat16 type is unimplemented.");
        }
        if (signalTy.getDtype().isF16()) {
          complexDtype = Torch::getTypeForScalarType(
              binder.op->getContext(),
              torch::torch_upstream::ScalarType::ComplexHalf);
        } else if (signalTy.getDtype().isF32()) {
          complexDtype = Torch::getTypeForScalarType(
              binder.op->getContext(),
              torch::torch_upstream::ScalarType::ComplexFloat);
        } else {
          complexDtype = Torch::getTypeForScalarType(
              binder.op->getContext(),
              torch::torch_upstream::ScalarType::ComplexDouble);
        }

        auto complexSignalTy = rewriter.getType<Torch::ValueTensorType>(
            ArrayRef<int64_t>({signalShape[0], signalShape[1]}),
            complexDtype.value());

        // The onnx STFT op always passes in a float input, and if the input
        // is intended to be complex, its shape will be [batch][length][2],
        // where [...][0] is the real component, and [...][1] is the complex
        // component. This complex input has to be made torch compatible before
        // being passed into torch.stft, so it is necessary to call
        // AtenViewAsComplexOp. In case of real input, the shape of the signal
        // will be [batch][length][1], and therefore it will have to be squeezed
        // at dim=2, before being passed into torch.stft.
        if (signalShape[2] == 2) {
          signal = rewriter.create<Torch::AtenViewAsComplexOp>(
              binder.getLoc(), complexSignalTy, signal);
        } else {
          Value two = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(2));
          auto newSignalTy = signalTy.getWithSizesAndDtype(
              ArrayRef<int64_t>({signalShape[0], signalShape[1]}),
              signalTy.getDtype());
          signal = rewriter.create<Torch::AtenSqueezeDimOp>(
              binder.getLoc(), newSignalTy, signal, two);
        }

        // In case the window is not given, we use frameLength
        // as the length of the window.
        Value windowLen;
        if (!windowIsNone) {
          windowLen = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(windowShape[0]));
        } else {
          windowLen = frameLengthItem;
        }

        Value falseVal =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value trueVal =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        auto stftTy = complexSignalTy.getWithSizesAndDtype(
            ArrayRef<int64_t>({resultShape[0], resultShape[2], resultShape[1]}),
            complexSignalTy.getDtype());

        // After torch.stft is called and the result is stored into the value
        // stft, there is one thing to note: The resultType for the onnx op
        // will have shape [batch][num_frames][length][2], while the shape of
        // stft will be [batch][length][num_frames]. Before the value is
        // converted to real through torch.view_as_real, we must permute the
        // shape of stft to match the shape of resultType. Also, it is
        // immaterial whether torch.view_as_real is called after or before the
        // permutation; both outputs will be equivalent.
        Value stft = rewriter.create<Torch::AtenStftOp>(
            binder.getLoc(), stftTy, signal, frameLengthItem, frameStepItem,
            windowLen, window, falseVal, onesided ? trueVal : falseVal,
            trueVal);

        auto permuteStftTy = complexSignalTy.getWithSizesAndDtype(
            ArrayRef<int64_t>({resultShape[0], resultShape[1], resultShape[2]}),
            complexSignalTy.getDtype());
        Value permuteDims = createConstantIntList(binder, rewriter, {0, 2, 1});
        Value permutedStft = rewriter.create<Torch::AtenPermuteOp>(
            binder.getLoc(), permuteStftTy, stft, permuteDims);

        rewriter.replaceOpWithNewOp<Torch::AtenViewAsRealOp>(
            binder.op, resultType, permutedStft);
        return success();
      });
  patterns.onOp(
      "ReverseSequence", 10,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input, sequenceLens;
        int64_t batchAxis, timeAxis;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(sequenceLens, 1) ||
            binder.s64IntegerAttr(batchAxis, "batch_axis", 1) ||
            binder.s64IntegerAttr(timeAxis, "time_axis", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        auto inputTy = cast<Torch::ValueTensorType>(input.getType());
        SmallVector<int64_t> inputShape(inputTy.getSizes());
        auto dtype = resultType.getDtype();

        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value batchAxisVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(batchAxis));
        Value timeAxisVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(timeAxis));

        SmallVector<int64_t> sliceShape(inputShape);
        sliceShape[batchAxis] = 1;
        auto sliceType =
            rewriter.getType<Torch::ValueTensorType>(sliceShape, dtype);
        SmallVector<int64_t> flipShape(sliceShape);
        flipShape[timeAxis] = Torch::kUnknownSize;
        auto flipType =
            rewriter.getType<Torch::ValueTensorType>(flipShape, dtype);
        auto scalarTensorType = rewriter.getType<Torch::ValueTensorType>(
            ArrayRef<int64_t>{1}, rewriter.getIntegerType(64, /*signed*/ 1));

        for (int i = 0; i < inputShape[batchAxis]; i++) {
          // slice i iterating on batch axis
          Value k = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i));
          Value end =
              rewriter.create<Torch::AtenAddIntOp>(binder.getLoc(), k, cstOne);
          Value sliceBatch = rewriter.create<Torch::AtenSliceTensorOp>(
              binder.getLoc(), sliceType, input, batchAxisVal, k, end, cstOne);

          // get sequence length and slice the reversing part
          Value kTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
              binder.getLoc(), scalarTensorType, k);
          Value sel = rewriter.create<Torch::AtenIndexSelectOp>(
              binder.getLoc(), scalarTensorType, sequenceLens, cstZero,
              kTensor);
          Value len = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), sel);
          Value sliceTime = rewriter.create<Torch::AtenSliceTensorOp>(
              binder.getLoc(), flipType, sliceBatch, timeAxisVal, cstZero, len,
              cstOne);
          // flip the sliced reversing tensor
          Value dims = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(),
              rewriter.getType<Torch::ListType>(
                  rewriter.getType<Torch::IntType>()),
              SmallVector<Value>{timeAxisVal});
          Value flip = rewriter.create<Torch::AtenFlipOp>(
              binder.getLoc(), flipType, sliceTime, dims);

          // embeds the reversed tensor to the input
          Value embedTime = rewriter.create<Torch::AtenSliceScatterOp>(
              binder.getLoc(), sliceType, sliceBatch, flip, timeAxisVal,
              /*start=*/cstZero, /*end=*/len, /*step=*/cstOne);
          input = rewriter.create<Torch::AtenSliceScatterOp>(
              binder.getLoc(), resultType, input, embedTime, batchAxisVal,
              /*start=*/k, /*end=*/end, /*step=*/cstOne);
        }

        rewriter.replaceOp(binder.op, input);
        return success();
      });
  patterns.onOp(
      "ScatterND", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, indices, updates;
        std::string reduction;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorOperandAtIndex(updates, 2) ||
            binder.tensorResultType(resultType))
          return failure();

        // Previous to version 16 of ScatterND, reduction attribute was not
        // supported. Setting it as "none" for unsupported versions.
        if (binder.customOpNameStringAttr(reduction, "reduction", "none")) {
          reduction = "none";
        }

        // Map onnx reduction type to torch reduction type.
        if (reduction == "add") {
          reduction = "sum";
        } else if (reduction == "mul") {
          reduction = "prod";
        } else if (reduction == "max") {
          reduction = "amax";
        } else if (reduction == "min") {
          reduction = "amin";
        } else if (reduction != "none") {
          return rewriter.notifyMatchFailure(
              binder.op, "expects reduction to be one of add, mul, max, min, "
                         "none(default)");
        }

        Location loc = binder.getLoc();
        auto dataTy = dyn_cast<Torch::ValueTensorType>(data.getType());
        auto indicesTy = dyn_cast<Torch::ValueTensorType>(indices.getType());
        auto updatesTy = dyn_cast<Torch::ValueTensorType>(updates.getType());
        if (!dataTy || !indicesTy || !updatesTy || !dataTy.hasSizes() ||
            !indicesTy.hasSizes() || !updatesTy.hasSizes())
          return failure();

        // step 1. Get shapes and ranks of data, indices and updates.
        // The last dimension of indices is expected to be static.
        ArrayRef<int64_t> dataShape = dataTy.getSizes();
        int64_t dataRank = dataShape.size();
        ArrayRef<int64_t> updatesShape = updatesTy.getSizes();
        int64_t updatesRank = updatesShape.size();
        ArrayRef<int64_t> indicesShape = indicesTy.getSizes();
        int64_t indicesRank = indicesShape.size();
        int64_t indicesLastDim = indicesShape.back();
        // Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and
        // updates tensor of rank q + r - indices_shape[-1] - 1, the output is
        // produced by creating a copy of the input data, and then updating
        // its value to values specified by updates at specific index positions
        // specified by indices. Its output shape is the same as the shape of
        // data.
        // indices_shape[-1] must be static to have deterministic ranks.
        if (dataRank < 1 || indicesRank < 1 || updatesRank < 1)
          return rewriter.notifyMatchFailure(
              binder.op, "expected data, indices and updates rank to be >= 1");
        if (indicesLastDim == Torch::kUnknownSize || indicesLastDim <= 0)
          return rewriter.notifyMatchFailure(
              binder.op, "expected last dimension of indices to be static and "
                         "greater than zero");

        // step 2. Get dimension list of data.
        SmallVector<Value> dataDims;
        for (int64_t i = 0; i < dataRank; ++i) {
          Value k = rewriter.create<Torch::ConstantIntOp>(loc, i);
          Value dataDim = rewriter.create<Torch::AtenSizeIntOp>(loc, data, k);
          dataDims.push_back(dataDim);
        }

        // step 3. Get dimension list of indices.
        Value constZero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));
        Value constOne = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        SmallVector<Value> indicesDimsMinusOne;
        Value indicesFlattenDim = constOne;
        for (int64_t i = 0; i < indicesRank - 1; ++i) {
          Value k = rewriter.create<Torch::ConstantIntOp>(loc, i);
          Value indicesDim =
              rewriter.create<Torch::AtenSizeIntOp>(loc, indices, k);
          indicesDimsMinusOne.push_back(indicesDim);
          indicesFlattenDim = rewriter.create<Torch::AtenMulIntOp>(
              loc, indicesFlattenDim, indicesDim);
        }
        ArrayRef<int64_t> indicesShapeMinusOne = indicesShape.drop_back();

        // Algorithm: We can not directly perform torch.scatter as it requires
        // the ranks of data(`r`), indices(`q`) and updates to be same.
        // So we will perform collapse and expand operations to match the
        // ranks of data, indices and updates(making sure the semantic of the
        // onnx.scatter_nd is preserved), then perform torch.scatter operation,
        // later unflatten the scatter result to match onnx.scatter_nd output.
        // For example, assuming
        // indices is of shape (4, 5, 3, 2), data is (4, 10, 11, 7, 4) and
        // updates is (4, 5, 3, 11, 7, 4). Firstly, modify indices to 1-D
        // indexing as the torch.scatter op supports only single dimensional
        // indexing(this algorithm would have been simpler if we can get a
        // torch op that supports indexing at multiple dimensions
        // simultaneously). 1-D indexed indices will be of shape (4, 5, 3, 1),
        // now materialize it to `r-indices_shape[-1]` dimension of data i.e.
        // reshaping it to the shape (4, 5, 3, 1, 1, 1). Next step is to
        // flatten+expand the indices and flatten the data to (60, 11, 7, 4) and
        // (40, 11, 7, 4) shapes respectively and then perform the torch.scatter
        // operation. Post the scatter operation, unflatten the first dimension
        // of result to (4, 10, 11, 7, 4) which is our required result.

        // step 4. Convert indices_shape[-1] dimensional indexing to 1D
        // indexing.
        Value sliceDim = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(indicesRank - 1));
        SmallVector<int64_t> indicesSliceShape(indicesShapeMinusOne);
        indicesSliceShape.push_back(1);
        auto indicesSliceTy = rewriter.getType<Torch::ValueTensorType>(
            indicesSliceShape, indicesTy.getOptionalDtype());

        Value start = constZero;
        Value updatedIndices;
        for (int64_t i = 0; i < indicesLastDim; ++i) {
          Value end = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i + 1));
          Value indicesSlice = rewriter.create<Torch::AtenSliceTensorOp>(
              loc, indicesSliceTy, indices, sliceDim, start, end,
              /*step=*/constOne);
          start = end;
          // Apply bounds checking on the indices slice.
          auto boolTy = rewriter.getType<Torch::ValueTensorType>(
              indicesSliceShape, rewriter.getI1Type());
          Value lt = rewriter.create<Torch::AtenLtScalarOp>(
              loc, boolTy, indicesSlice, constZero);
          Value add = rewriter.create<Torch::AtenAddScalarOp>(
              loc, indicesSliceTy, indicesSlice, dataDims[i],
              /*alpha=*/constOne);
          indicesSlice = rewriter.create<Torch::AtenWhereSelfOp>(
              loc, indicesSliceTy, lt, add, indicesSlice);
          if (i == 0) {
            updatedIndices = indicesSlice;
            continue;
          }
          updatedIndices = rewriter.create<Torch::AtenAddTensorOp>(
              loc, indicesSliceTy, indicesSlice, updatedIndices, dataDims[i]);
        }

        // step 5. Compute all the required result types here.
        SmallVector<int64_t> reshapeIndicesShape(indicesShapeMinusOne);
        SmallVector<Value> reshapeIndicesDims(indicesDimsMinusOne);
        // Determine the collapsed dim size of indices(index_shape[-1] is not
        // part of collapsing as we already removed it by 1-D indexing).
        SmallVector<int64_t> flattenIndicesShape;
        auto indicesCt = 1;
        for (int64_t i = 0; i < indicesRank - 1; ++i) {
          if (indicesShape[i] == Torch::kUnknownSize) {
            indicesCt = Torch::kUnknownSize;
            break;
          }
          indicesCt *= indicesShape[i];
        }
        flattenIndicesShape.push_back(indicesCt);
        // Compute the shape of expand op.
        SmallVector<Value> expandIndicesDims;
        expandIndicesDims.push_back(indicesFlattenDim);
        SmallVector<int64_t> expandIndicesShape;
        expandIndicesShape.push_back(indicesCt);
        // Determine the collapsed dim size of data.
        SmallVector<int64_t> flattenDataShape;
        auto dataCt = 1;
        for (int64_t i = 0; i < indicesLastDim; ++i) {
          if (dataShape[i] == Torch::kUnknownSize) {
            dataCt = Torch::kUnknownSize;
            break;
          }
          dataCt *= dataShape[i];
        }
        flattenDataShape.push_back(dataCt);
        // Determine the collapsed dim size of updates.
        SmallVector<int64_t> flattenUpdatesShape;
        auto updatesCt = 1;
        for (int64_t i = 0; i < indicesRank - 1; ++i) {
          if (updatesShape[i] == Torch::kUnknownSize) {
            updatesCt = Torch::kUnknownSize;
            break;
          }
          updatesCt *= updatesShape[i];
        }
        flattenUpdatesShape.push_back(updatesCt);
        flattenUpdatesShape.insert(flattenUpdatesShape.end(),
                                   updatesShape.begin() + indicesRank - 1,
                                   updatesShape.end());
        // Append `r-indices_shape[-1]` unit or data dims appropriately to all
        // result types.
        for (int64_t i = indicesLastDim; i < dataRank; ++i) {
          reshapeIndicesShape.push_back(1);
          flattenIndicesShape.push_back(1);
          flattenDataShape.push_back(dataShape[i]);
          expandIndicesShape.push_back(dataShape[i]);
          reshapeIndicesDims.push_back(constOne);
          expandIndicesDims.push_back(dataDims[i]);
        }

        // step 6. Reshape 1-D indexed indices to match the rank of flattened
        // data by inserting unit dimensions.
        auto intListTy = rewriter.getType<Torch::ListType>(
            rewriter.getType<Torch::IntType>());
        Value reshapeIndicesSizeList =
            rewriter.create<Torch::PrimListConstructOp>(loc, intListTy,
                                                        reshapeIndicesDims);
        auto reshapeIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            reshapeIndicesShape, indicesTy.getOptionalDtype());
        Value reshapedIndices = rewriter.create<Torch::AtenViewOp>(
            loc, reshapeIndicesTy, updatedIndices, reshapeIndicesSizeList);

        // step 7. Flatten `q-1` dimensions of the indices and updates.
        auto flattenIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            flattenIndicesShape, indicesTy.getOptionalDtype());
        auto flattenUpdatesTy = rewriter.getType<Torch::ValueTensorType>(
            flattenUpdatesShape, updatesTy.getOptionalDtype());
        Value flattenedIndices = reshapedIndices;
        Value flattenedUpdates = updates;
        if (indicesRank == 1) {
          flattenedIndices = rewriter.create<Torch::AtenUnsqueezeOp>(
              loc, flattenIndicesTy, reshapedIndices, constZero);
          flattenedUpdates = rewriter.create<Torch::AtenUnsqueezeOp>(
              loc, flattenUpdatesTy, updates, constZero);
        } else if (indicesRank > 1) {
          Value endDim = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(indicesRank - 2));
          flattenedIndices = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
              loc, flattenIndicesTy, reshapedIndices, constZero, endDim);
          flattenedUpdates = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
              loc, flattenUpdatesTy, updates, constZero, endDim);
        }

        // step 8. Expand `r-indices_shape[-1]` dims of flattened indices.
        auto expandIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            expandIndicesShape, indicesTy.getOptionalDtype());
        Value expandIndicesSizeList =
            rewriter.create<Torch::PrimListConstructOp>(loc, intListTy,
                                                        expandIndicesDims);
        Value constFalse = rewriter.create<Torch::ConstantBoolOp>(
            loc, rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));
        Value expandedIndices = rewriter.create<Torch::AtenExpandOp>(
            loc, expandIndicesTy, flattenedIndices, expandIndicesSizeList,
            /*implicit=*/constFalse);

        // step 9. Flatten indices_shape[-1] dimensions of data.
        auto flattenDataTy = rewriter.getType<Torch::ValueTensorType>(
            flattenDataShape, dataTy.getOptionalDtype());
        Value endDim = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(indicesLastDim - 1));
        Value flattenedData = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
            loc, flattenDataTy, data, constZero, endDim);

        // step 10. Now we have flattenedData, expandedIndices and
        // flattenedUpdates of same rank to perform scatter operation.
        auto scatterTy = rewriter.getType<Torch::ValueTensorType>(
            flattenDataShape, dataTy.getOptionalDtype());

        Value scatter;
        if (reduction == "none") {
          scatter = rewriter.create<Torch::AtenScatterSrcOp>(
              loc, scatterTy, flattenedData, /*axis=*/constZero,
              expandedIndices, flattenedUpdates);
        } else {
          Value cstReduction =
              rewriter.create<Torch::ConstantStrOp>(loc, reduction);
          Value constTrue = rewriter.create<Torch::ConstantBoolOp>(
              loc, rewriter.getType<Torch::BoolType>(),
              rewriter.getBoolAttr(true));
          scatter = rewriter.create<Torch::AtenScatterReduceTwoOp>(
              loc, scatterTy, flattenedData, /*axis=*/constZero,
              expandedIndices, flattenedUpdates, cstReduction,
              /*include_self=*/constTrue);
        }

        // step 11. Unflatten the collapsed data dims of scatter result.
        if (indicesLastDim == 1) {
          rewriter.replaceOp(binder.op, scatter);
          return success();
        }
        Value unflattenSizeList = rewriter.create<Torch::PrimListConstructOp>(
            loc, intListTy, dataDims);
        rewriter.replaceOpWithNewOp<Torch::AtenUnflattenIntOp>(
            binder.op, resultType, scatter, constZero, unflattenSizeList);
        return success();
      });
  // split to sequence
  // Arguments:
  // - input: the tensor to split
  // -Split(optional): Length of each output
  // Attributes:
  // - axis: the axis along which to split the input
  // - keepdims: to keep the split dimension or not. Ignored when 'split' is
  // specified Outputs:
  // - outputs: sequence of tensor
  //

  patterns.onOp(
      "SplitToSequence", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value self;
        Value split;
        int64_t axis;
        int64_t keepdims;
        Torch::ListType resultType;

        if (binder.op->getNumOperands() == 1)
          return rewriter.notifyMatchFailure(
              binder.op, "No of operands should be two.Keepdims attribute is "
                         "not yet implemented");

        if (binder.tensorOperandAtIndex(self, 0) ||
            binder.tensorListResultType(resultType) ||
            binder.s64IntegerAttr(keepdims, "keepdims", 1) ||
            binder.tensorOperandAtIndex(split, 1) ||
            binder.s64IntegerAttr(axis, "axis", 0))
          return rewriter.notifyMatchFailure(
              binder.op,
              "Not converting to AtenSplitToSequenceOp due to inputs ");

        Value axisValue = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(axis));
        auto splitTy = cast<Torch::ValueTensorType>(split.getType());

        if (!splitTy || !splitTy.hasSizes())
          return failure();

        auto splitSizes = splitTy.getSizes();
        unsigned splitDim = splitTy.getSizes().size();

        if (splitDim > 1)
          return rewriter.notifyMatchFailure(
              binder.op, "Split should be scalar or 1-D Tensor ");

        if (splitDim == 1) {
          if (splitSizes[0] == Torch::kUnknownSize) {
            return rewriter.notifyMatchFailure(
                binder.op, "Dynamic shapes for Split is not yet supported");
          } else if (splitSizes[0] <=
                     1) { // dealing with 1/0 element in 1-D tensor
            Value splitInt = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), split);
            rewriter.replaceOpWithNewOp<Torch::AtenSplitTensorOp>(
                binder.op, resultType, self, splitInt, axisValue);
            return success();
          } else {
            // Handling multiple elment in split
            Value shapeList =
                createConstantIntList(binder, rewriter, splitSizes);
            rewriter.replaceOpWithNewOp<Torch::AtenSplitSizesOp>(
                binder.op, resultType, self, shapeList, axisValue);
            return success();
          }
        } else if (splitDim == 0) { // Handle 0-D tensor
          Value splitInt = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), split);
          rewriter.replaceOpWithNewOp<Torch::AtenSplitTensorOp>(
              binder.op, resultType, self, splitInt, axisValue);
          return success();
        } else {
          return rewriter.notifyMatchFailure(
              binder.op, "Handling of this kind of inputs is not there");
        }
      });
  patterns.onOp(
      "Unique", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value input;
        int64_t axis, sorted;
        SmallVector<Type> resultTypes;

        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(sorted, "sorted", 1) ||
            binder.tensorResultTypes(resultTypes))
          return failure();

        Value zero = rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), 0);

        auto inputTy = cast<Torch::ValueTensorType>(input.getType());
        if (!inputTy.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input type to have sizes");
        }
        auto inputShape = inputTy.getSizes();
        int64_t inputDim = static_cast<int64_t>(inputShape.size());

        Value axisVal;
        SmallVector<int64_t> outputTensorSizes(inputDim);
        bool axisWasNone;
        if (!binder.optionalS64IntegerAttr(axis, "axis")) {
          if (axis < -1 * inputDim || axis > inputDim - 1)
            return rewriter.notifyMatchFailure(binder.op,
                                               "invalid value for axis");
          axisVal = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(axis));
          axisWasNone = false;
        } else {
          axisVal = zero;
          axisWasNone = true;
        }

        Value sortedVal = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(sorted));
        Value trueVal =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);

        // The shape of inverse_indices is the same as input shape, but
        // resulTypes[2] must be used to avoid live value after conversion.
        Torch::ValueTensorType outputTy;
        outputTy = cast<Torch::ValueTensorType>(resultTypes[0]);
        Torch::ValueTensorType countsTy =
            cast<Torch::ValueTensorType>(resultTypes[3]);
        Torch::ValueTensorType inverseTy =
            cast<Torch::ValueTensorType>(resultTypes[2]);

        if (axisWasNone) {
          int64_t inputNumel = 1;
          for (auto elem : inputShape) {
            if (elem == Torch::kUnknownSize) {
              return rewriter.notifyMatchFailure(
                  binder.op,
                  "Expected all sizes in input shape to be statically known");
            }
            inputNumel *= elem;
          }
          auto flattenResultTy = rewriter.getType<Torch::ValueTensorType>(
              ArrayRef({inputNumel}), inputTy.getDtype());
          Value negativeOne =
              rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), -1);
          input = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
              binder.getLoc(), flattenResultTy, input, zero, negativeOne);
        }

        Torch::AtenUniqueDimOp intermResults =
            rewriter.create<Torch::AtenUniqueDimOp>(
                binder.getLoc(), outputTy, inverseTy, countsTy, input, axisVal,
                sortedVal, trueVal, trueVal);

        SmallVector<Value> uniqueResults = intermResults.getResults();

        // Calculate the indices where each of the unique elements first
        // appeared in the original input tensor. Also, the counts tensor and
        // the indices tensor have the same Dtype, int64, so reuse that here.
        auto arangeResultType = rewriter.getType<Torch::ValueTensorType>(
            ArrayRef<int64_t>({inputShape[0]}), countsTy.getOptionalDtype());

        Value inputDimZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(inputShape[0]));
        Value int64Type = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(4));
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value perm = rewriter.create<Torch::AtenArangeOp>(
            binder.getLoc(), arangeResultType, inputDimZero,
            /*dtype=*/int64Type,
            /*layout=*/noneVal, /*device=*/noneVal, /*pin_memory=*/noneVal);

        // Inverse has the same shape as input, but the dtype is not the same.
        Value flipDims = createConstantIntList(binder, rewriter, {0});
        Value inverse = rewriter.create<Torch::AtenFlipOp>(
            binder.getLoc(),
            inputTy.getWithSizesAndDtype(inputShape, countsTy.getDtype()),
            uniqueResults[1], flipDims);
        perm = rewriter.create<Torch::AtenFlipOp>(
            binder.getLoc(), cast<Torch::ValueTensorType>(perm.getType()), perm,
            flipDims);

        auto newInverseTy = rewriter.getType<Torch::ValueTensorType>(
            ArrayRef<int64_t>({outputTy.getSizes()[0]}), countsTy.getDtype());
        Value newInverseSize =
            createConstantIntList(binder, rewriter, {outputTy.getSizes()[0]});
        Value newInverse = rewriter.create<Torch::AtenNewEmptyOp>(
            binder.getLoc(), newInverseTy, inverse, newInverseSize,
            /*dtype=*/int64Type, /*layout=*/noneVal, /*device=*/noneVal,
            /*pin_memory=*/noneVal);

        Value firstOccurIndices = rewriter.create<Torch::AtenScatterSrcOp>(
            binder.getLoc(), resultTypes[1], newInverse, zero, inverse, perm);

        rewriter.replaceOp(binder.op, {uniqueResults[0], firstOccurIndices,
                                       uniqueResults[1], uniqueResults[2]});
        return success();
      });
  patterns.onOp(
      "Scan", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        SmallVector<Value> operands;
        int64_t numScanInputs;
        if (binder.tensorOperandsList(operands) || operands.size() == 0 ||
            binder.s64IntegerAttr(numScanInputs, "num_scan_inputs")) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required inputs");
        }
        SmallVector<Type> resultTypes;
        if (binder.tensorResultTypes(resultTypes)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");
        }
        Region *loopBodyIn;
        if (binder.getRegionAtIndex(loopBodyIn, 0)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed getting LoopBody Region");
        }

        int64_t numInits = operands.size() - numScanInputs;
        SmallVector<Value> initVals(operands.begin(),
                                    operands.begin() + numInits);
        SmallVector<Value> scanInputs(operands.begin() + numInits,
                                      operands.end());
        if (scanInputs.size() < 1) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expects at least one scan input");
        }

        Value constZero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));
        Value constOne = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        SmallVector<Type> scanOutTypes;
        for (unsigned i = numInits; i < resultTypes.size(); i++) {
          auto scanOutTy = cast<Torch::ValueTensorType>(resultTypes[i]);
          // TODO: Handle dynamic result types.
          if (!scanOutTy.hasSizes() || !scanOutTy.areAllSizesKnown()) {
            return rewriter.notifyMatchFailure(
                binder.op, "Expects result type to be static");
          }
          Value sizeList =
              createConstantIntList(binder, rewriter, scanOutTy.getSizes());
          initVals.push_back(Torch::createInitTensor(rewriter, loc, scanOutTy,
                                                     constZero, sizeList));
          scanOutTypes.push_back(resultTypes[i]);
        }
        // Create torch.prim.Loop op.
        Value maxTripCount = rewriter.create<Torch::AtenSizeIntOp>(
            loc, scanInputs[0], constZero);
        auto constBoolTrue = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(true));
        auto primLoop = rewriter.create<Torch::PrimLoopOp>(
            loc, resultTypes, maxTripCount, constBoolTrue, initVals);
        rewriter.cloneRegionBefore(*loopBodyIn, primLoop.getRegion(),
                                   primLoop.getRegion().begin());

        // Insert index var as torch.int argument in the loop body, as
        // the primLoopOp loopBody expects torch.int as first argument.
        primLoop.getRegion().insertArgument(
            0u, rewriter.getType<Torch::IntType>(), loc);
        auto loopInd = primLoop.getRegion().getArgument(0);

        // The block arguments of onnx.scan needs to be replaced with
        // slice of scan inputs.
        rewriter.setInsertionPointToStart(&primLoop.getRegion().front());
        for (unsigned i = 0; i < numScanInputs; i++) {
          auto loopBlockArg =
              primLoop.getRegion().getArgument(numInits + 1 + i);
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              loc, loopBlockArg.getType(), scanInputs[i], constZero, loopInd);
          loopBlockArg.replaceAllUsesWith(extract);
        }
        primLoop.getRegion().front().eraseArguments(numInits + 1,
                                                    /*count=*/numScanInputs);

        // Collect the output slices to form scan outputs and replace the
        // terminator.
        SmallVector<Location> locs(scanOutTypes.size(), loc);
        primLoop.getRegion().front().addArguments(scanOutTypes, locs);

        PatternRewriter::InsertionGuard guard(rewriter);
        Operation *terminator = primLoop.getRegion().front().getTerminator();
        auto terminatorOperands = terminator->getOperands();
        SmallVector<Value> resTerminatorOperands(
            terminatorOperands.begin(), terminatorOperands.begin() + numInits);
        SmallVector<Value> scanOutSlices(terminatorOperands.begin() + numInits,
                                         terminatorOperands.end());
        rewriter.setInsertionPoint(terminator);
        for (unsigned i = 0; i < scanOutSlices.size(); i++) {
          Value self = BlockArgument::Value(
              primLoop.getRegion().getArgument(numInits + 1 + i));
          FailureOr<Value> src = Torch::unsqueezeTensor(
              rewriter, binder.op, scanOutSlices[i], constZero);
          if (failed(src))
            return failure();
          Value scanOut = rewriter.create<Torch::AtenSliceScatterOp>(
              loc, scanOutTypes[i], self, src.value(), constZero,
              /*start=*/loopInd,
              /*end=*/loopInd, constOne);
          resTerminatorOperands.push_back(scanOut);
        }

        Value terminatorCond = constBoolTrue;
        rewriter.replaceOpWithNewOp<Torch::PrimLoopConditionOp>(
            terminator, terminatorCond, resTerminatorOperands);
        rewriter.replaceOp(binder.op, primLoop);
        return success();
      });
}
