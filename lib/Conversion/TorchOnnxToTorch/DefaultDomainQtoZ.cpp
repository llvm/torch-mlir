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
//  Templatized function to get an item op of a type
namespace {
template <typename T>
Value getItemOp(OpBinder binder, ConversionPatternRewriter &rewriter,
                Value &ofItem) {
  return rewriter.create<Torch::AtenItemOp>(binder.getLoc(),
                                            rewriter.getType<T>(), ofItem);
}

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
    auto inputType = data.getType().dyn_cast<Torch::ValueTensorType>();
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
          axesVal.getType().cast<Torch::BaseTensorType>();
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

        auto scaleTy = scale.getType().dyn_cast<Torch::ValueTensorType>();
        if (!scaleTy || !scaleTy.hasSizes())
          return rewriter.notifyMatchFailure(binder.op, "requires known rank");
        if (!resultType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "requires known result dtype");

        if (scaleTy.getSizes().size() == 0) {
          Type qTy = resultType.getDtype();

          if (qTy.isUnsignedInteger(8)) {
            qTy = rewriter.getType<Torch::QUInt8Type>();
          } else if (qTy.isSignedInteger(8)) {
            qTy = rewriter.getType<Torch::QInt8Type>();
          } else if (qTy.isSignedInteger(32)) {
            qTy = rewriter.getType<Torch::QInt32Type>();
          } else {
            return rewriter.notifyMatchFailure(binder.op,
                                               "unsupported result dtype");
          }

          auto qTensorTy = rewriter.getType<Torch::ValueTensorType>(
              resultType.getOptionalSizes(), qTy);
          auto torchqTy = Torch::getScalarTypeForType(qTy);

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
          auto vTy = v.getType().cast<Torch::ValueTensorType>();
          return llvm::all_of(vTy.getSizes(), [](int64_t d) { return d == 1; });
        };
        if (!check(aScale) || !check(aZp) || !check(bScale) || !check(bZp) ||
            !check(cScale) || !check(cScale))
          return rewriter.notifyMatchFailure(
              binder.op, "not supported for non per-tensor quantization");

        auto extract = [&rewriter, &binder](Value v) {
          auto vTy = v.getType().cast<Torch::ValueTensorType>();
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
          auto ty = v.getType().cast<Torch::ValueTensorType>();
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
          auto vTy = v.getType().cast<Torch::ValueTensorType>();
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
          auto vTy = v.getType().cast<Torch::ValueTensorType>();
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
          auto ty = v.getType().cast<Torch::ValueTensorType>();
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

        Value data = valList[0];
        Value indices = valList[1];
        Value updates = valList[2];

        // ONNX allows negative axis.
        if (axis < 0)
          axis +=
              cast<Torch::ValueTensorType>(data.getType()).getSizes().size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));

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
        auto inputType = data.getType().dyn_cast<Torch::ValueTensorType>();
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
              axes.getType().cast<Torch::BaseTensorType>();
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
        auto inputType = data.getType().dyn_cast<Torch::ValueTensorType>();
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
        Torch::ValueTensorType resultType;
        float alpha, gamma;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.f32FloatAttr(alpha, "alpha") ||
            binder.f32FloatAttr(gamma, "gamma") ||
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
          auto inputType = data.getType().dyn_cast<Torch::ValueTensorType>();
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
                axesVal.getType().cast<Torch::BaseTensorType>();
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
              axes.getType().cast<Torch::BaseTensorType>();
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
              axes.getType().cast<Torch::BaseTensorType>();
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

  patterns.onOp("Shape", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::Aten_ShapeAsTensorOp>(
                      binder.op, resultType, operand);
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
            binder.op->getResult(0).getType().cast<Torch::ValueTensorType>();
        auto resultNTy = binder.op->getResults()
                             .back()
                             .getType()
                             .cast<Torch::ValueTensorType>();
        auto selfTy = self.getType().cast<Torch::ValueTensorType>();

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
            binder.op->getResult(0).getType().cast<Torch::ValueTensorType>();
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
      "Transpose", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        auto loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType))
          return failure();
        auto operandType = operand.getType().cast<Torch::ValueTensorType>();
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
        auto operandTorchTy = operand.getType().cast<Torch::ValueTensorType>();
        auto operandTy =
            operandTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();

        if (!operandTy)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected tensor operator argument to be a ranked tensor type");

        auto startsTorchTy = starts.getType().cast<Torch::ValueTensorType>();
        auto startsTy =
            startsTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();
        int startSize = startsTy.getDimSize(0);

        auto endsTorchTy = ends.getType().cast<Torch::ValueTensorType>();
        auto endsTy =
            endsTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();
        int endSize = endsTy.getDimSize(0);
        auto resultTy =
            resultTorchType.toBuiltinTensor().dyn_cast<RankedTensorType>();
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
          auto axesTorchTy = axes.getType().cast<Torch::ValueTensorType>();
          auto axesTy =
              axesTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();
          int64_t numAxes = axesTy.getDimSize(0);

          if (!(axesTy && numAxes == endSize))
            return rewriter.notifyMatchFailure(
                binder.op, "Axes should be the same size of starts and ends");
        }

        auto stepsTy = steps.getType()
                           .cast<Torch::ValueTensorType>()
                           .toBuiltinTensor()
                           .dyn_cast<RankedTensorType>();

        if (!(stepsTy && stepsTy.getDimSize(0) == endsTy.getDimSize(0)))
          return rewriter.notifyMatchFailure(
              binder.op, "Steps should be the same size of starts and ends");

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

        auto select = [&](Value v, Value k) -> Value {
          auto ty = v.getType().cast<Torch::ValueTensorType>();
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
            shape.getType().cast<Torch::BaseTensorType>();
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
              axes.getType().cast<Torch::BaseTensorType>();
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
            start.getType().cast<Torch::BaseTensorType>();
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
            repeatDims.getType().cast<Torch::BaseTensorType>();
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
}
