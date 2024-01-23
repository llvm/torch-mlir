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
void mlir::torch::onnx_c::populateDefaultDomainGtoP(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp("HardSigmoid", 6,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value tensorOperand;
        float alpha, beta;
        if (binder.tensorOperand(tensorOperand) ||
            binder.f32FloatAttr(alpha, "alpha", 0.2f) ||
            binder.f32FloatAttr(beta, "beta", 0.5f) ||
            binder.tensorResultType(resultType))
          return failure();
        
        // HardSigmoid computes the following expression: max(0, min(1, alpha * x + beta))
        Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(alpha));

        Value constBeta = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(beta));

        // Expression: alpha * x + beta
        Value alpha_x_plus_beta = rewriter.create<Torch::AtenAddScalarOp>(
            binder.getLoc(), resultType, tensorOperand, constBeta, /*alpha=*/constAlpha);

        // Expression: min(1, alpha * x + beta)
        Value constantOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value oneTensor = createRank0Tensor(rewriter, binder.getLoc(),
                                            resultType, constantOne);
        Value minExpression = rewriter.create<Torch::AtenMinimumOp>(
            binder.getLoc(), resultType, oneTensor, alpha_x_plus_beta);

        // Expression: max(0, min(1, alpha * x + beta))
        Value constantZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value zeroTensor = createRank0Tensor(rewriter, binder.getLoc(),
                                             resultType, constantZero);
        rewriter.replaceOpWithNewOp<Torch::AtenMaximumOp>(
            binder.op, resultType, zeroTensor, minExpression);
        return success();
      });
  patterns.onOp(
      "Gelu", 20, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value operand;
        Torch::ValueTensorType resultType;
        std::string approximate;

        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.customOpNameStringAttr(approximate, "approximate", "none"))
          return failure();

        Value vApproximate = rewriter.create<Torch::ConstantStrOp>(
            binder.getLoc(), rewriter.getType<Torch::StringType>(),
            rewriter.getStringAttr(approximate));

        rewriter.replaceOpWithNewOp<Torch::AtenGeluOp>(binder.op, resultType,
                                                       operand, vApproximate);
        return success();
      });
  patterns.onOp("Less", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenLtTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                }); 
  patterns.onOp("LessOrEqual", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
		    rewriter.replaceOpWithNewOp<Torch::AtenLeTensorOp>(
                      binder.op, resultType, lhs, rhs);
                              return success();
                });
  patterns.onOp("Log", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenLogOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("MatMul", 13,
                  [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("Mul", 7,
		[](OpBinder binder, ConversionPatternRewriter &rewriter) {
		  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenMulTensorOp>(
		    binder.op, resultType, lhs, rhs);
		  return success();
		});
  patterns.onOp("NonZero", 13,
    [](OpBinder binder, ConversionPatternRewriter &rewriter) {
      Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenNonzeroOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "MaxPool", 12, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "auto_pad bind failure");
        if (autoPad != "NOTSET")
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");

        Torch::ValueTensorType resultType;
        Value operand;
        bool ceilMode;
        int64_t storageOrder;
        // TODO: Add support for indices output and storage_order
        if (binder.tensorOperand(operand) ||
            binder.s64BoolAttr(ceilMode, "ceil_mode", false) ||
            binder.s64IntegerAttr(storageOrder, "storage_order", 0) ||
            binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(
              binder.op,
              "operand/ceil_mode/storage_order/resultType bind failure");
        if (storageOrder != 0)
          return rewriter.notifyMatchFailure(
              binder.op, "storage_order setting is not supported.");
        // Determine the rank of input tensor.
        std::optional<unsigned> maybeRank = Torch::getTensorRank(operand);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        unsigned rank = *maybeRank;

        SmallVector<int64_t> kernel, padding, strides, dilations;
        if (binder.s64IntegerArrayAttr(kernel, "kernel_shape", {}))
          return rewriter.notifyMatchFailure(binder.op,
                                             "kernel_shape bind failure");
        if (kernel.size() != rank - 2)
          return rewriter.notifyMatchFailure(
              binder.op, "kernel list size does not match the number of axes");
        if (binder.s64IntegerArrayAttr(padding, "pads", {0}))
          return rewriter.notifyMatchFailure(binder.op, "pads bind failure");
        if (padding.size() != 1 && padding.size() != rank - 2)
          return rewriter.notifyMatchFailure(
              binder.op, "padding list size does not match the number of axes");
        if (binder.s64IntegerArrayAttr(strides, "strides", {1}))
          return rewriter.notifyMatchFailure(binder.op, "strides bind failure");
        if (strides.size() != 1 && strides.size() != rank - 2)
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");
        if (binder.s64IntegerArrayAttr(dilations, "dilations", {}))
          return rewriter.notifyMatchFailure(binder.op,
                                             "dilations bind failure");

        Value kernelSizeList = createConstantIntList(binder, rewriter, kernel);
        Value paddingList = createConstantIntList(binder, rewriter, padding);
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        Value dilationsList =
            createConstantIntList(binder, rewriter, dilations);
        Value cstCeilMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), ceilMode);

        if (rank == 3)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: AtenMaxPool1dOp");
        if (rank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxPool2dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, dilationsList, cstCeilMode);
          return success();
        }
        if (rank == 5) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxPool3dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, dilationsList, cstCeilMode);
          return success();
        }
        return rewriter.notifyMatchFailure(binder.op, "No rank is matched.");
      });
  patterns.onOp("Greater", 16,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  std::string direction;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenGtTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("GreaterOrEqual", 16,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  std::string direction;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenGeTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("Max", 1,
		[](OpBinder binder, ConversionPatternRewriter &rewriter) {
		  Torch::ValueTensorType resultType;
		  llvm::SmallVector<Value> operands;
		  if (binder.tensorOperandsList(operands) ||
                      binder.tensorResultType(resultType) ||
		      operands.size() == 0) {
                    return failure();
		  }
		  Value result = operands[0];
		  for (uint64_t i = 1; i < operands.size(); i++) {
		    result = rewriter.create<Torch::AtenMaximumOp>(
		               binder.getLoc(), resultType, result, operands[i]);
                  }
                  rewriter.replaceOp(binder.op, result.getDefiningOp());
                  return success();
                });
  patterns.onOp("Min", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  llvm::SmallVector<Value> operands;
                  if (binder.tensorOperandsList(operands) ||
                      binder.tensorResultType(resultType) ||
                      operands.size() == 0) {
                    return failure();
                  }
                  Value result = operands[0];
                  for (uint64_t i = 1; i < operands.size(); i++) {
                    result = rewriter.create<Torch::AtenMinimumOp>(
                               binder.getLoc(), resultType, result, operands[i]);
                  }
                  rewriter.replaceOp(
                    binder.op, result.getDefiningOp());
                  return success();
		});
  patterns.onOp("Neg", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenNegOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Not", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenBitwiseNotOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Or", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenBitwiseOrTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp(
      "Gather", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, indices;
        int64_t axis;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(axis, "axis", 0))
          return failure();
        Location loc = binder.getLoc();

        // 1. Get data shape and rank.
        auto dataTensorType = data.getType().cast<Torch::ValueTensorType>();
        if (!dataTensorType || !dataTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expect non empty input data");
        }
        ArrayRef<int64_t> dataShape = dataTensorType.getSizes();
        unsigned dataRank = dataShape.size();

        // 2. Get indices shape and rank.
        auto indexType = indices.getType().cast<Torch::ValueTensorType>();
        if (!indexType || !indexType.hasSizes()) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expect non empty index tensor");
        }
        ArrayRef<int64_t> indexShape = indexType.getSizes();
        unsigned indexRank = indexShape.size();

        // 3. Compute total elements in the indices tensor, as we will collapse
        // the indices tensor to a unary tensor. Also compute index shape and
        // data shape tensors as they will be used for creating output types.
        int64_t indexElemCount = 1;
        for (int64_t dim : indexShape) {
          if (dim == Torch::kUnknownSize) {
            indexElemCount = Torch::kUnknownSize;
            break;
          }
          indexElemCount *= dim;
        }

        Value constOne = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        SmallVector<Value> indexShapeTensor;
        Value indexElemCountVal = constOne;
        for (unsigned i = 0; i < indexRank; ++i) {
          Value indexDimVal = rewriter.create<Torch::AtenSizeIntOp>(
              loc, indices,
              rewriter.create<Torch::ConstantIntOp>(
                  loc, rewriter.getI64IntegerAttr(i)));
          indexShapeTensor.emplace_back(indexDimVal);
          indexElemCountVal = rewriter.create<Torch::AtenMulIntOp>(
              loc, indexElemCountVal, indexDimVal);
        }

        SmallVector<Value> dataShapeTensor;
        for (unsigned i = 0; i < dataRank; ++i) {
          dataShapeTensor.emplace_back(rewriter.create<Torch::AtenSizeIntOp>(
              loc, data,
              rewriter.create<Torch::ConstantIntOp>(
                  loc, rewriter.getI64IntegerAttr(i))));
        }

        // 4. We can not directly perform torch.gather as the onnx.gather op
        // collects the input data at different location of output compared to
        // torch.gather op. The output of torch.gather and onnx.gather ops are
        // indexed differently.
        // check https://onnx.ai/onnx/operators/onnx__Gather.html for more
        // details. So we will collapse indices tensor to a unary tensor and
        // materialize to non-axis dimension of data tensor. For example,
        // assuming indices is of shape (4, 5, 6), data is (8, 10, 11, 12) and
        // axis=1. we will collapse indices into a (120,) unary tensor,
        // materialize to non-axis dimension of data i.e. reshaping the unary
        // indices tensor to (1, 120, 1, 1) and then perform the torch.gather
        // operation. Now broadcast the output of gather operation to non-axis
        // dimensions of data tensor. This would make the result of shape (8,
        // 10, 120, 12). Post the broadcasting, expand the indices dimensions by
        // reshaping (8, 10, 120, 12) to (8, 10, 4, 5, 6, 12) tensor, which is
        // our expected final result.
        SmallVector<int64_t> collapsedIndexShape(dataRank, 1);
        collapsedIndexShape[axis] = indexElemCount;
        Type collapsedIndexType = Torch::ValueTensorType::get(
            indexType.getContext(), llvm::ArrayRef(collapsedIndexShape),
            indexType.getOptionalDtype());

        SmallVector<Value> collapsedIndexSize(dataRank, constOne);
        collapsedIndexSize[axis] = indexElemCountVal;
        auto collapsedIndexSizeList =
            rewriter.create<Torch::PrimListConstructOp>(
                loc,
                rewriter.getType<Torch::ListType>(
                    rewriter.getType<Torch::IntType>()),
                collapsedIndexSize);

        auto collapsedIndices = rewriter.create<Torch::AtenViewOp>(
            loc, collapsedIndexType, indices, collapsedIndexSizeList);

        // 5. Compute gather result type and perform gather operation.
        Type gatherResultType = Torch::ValueTensorType::get(
            dataTensorType.getContext(), llvm::ArrayRef(collapsedIndexShape),
            dataTensorType.getOptionalDtype());
        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));
        Value constFalse = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));
        auto gatherOp = rewriter.create<Torch::AtenGatherOp>(
            loc, gatherResultType, data, constAxis, collapsedIndices,
            /*sparseGrad=*/constFalse);

        // 6. Broadcast the gather output to non-axis dimensions of data tensor.
        SmallVector<int64_t> dataShapeVector(dataShape);
        dataShapeVector[axis] = indexElemCount;
        Type expandResultType = Torch::ValueTensorType::get(
            dataTensorType.getContext(), llvm::ArrayRef(dataShapeVector),
            dataTensorType.getOptionalDtype());

        dataShapeTensor[axis] = indexElemCountVal;
        auto expandSizeList = rewriter.create<Torch::PrimListConstructOp>(
            loc, Torch::ListType::get(Torch::IntType::get(data.getContext())),
            dataShapeTensor);
        auto expandedGather = rewriter.create<Torch::AtenExpandOp>(
            loc, expandResultType, gatherOp, expandSizeList,
            /*implicit=*/constFalse);

        // 7. Compute the result type of reshape op which expands the collapsed
        // indices shapes back to the original indices shapes and reshape the
        // output produced at step 6. This will produce our expected result of
        // onnx.gather op.
        SmallVector<Value> resultShapeTensor;
        for (unsigned i = 0; i < dataRank; ++i) {
          if (i == axis) {
            resultShapeTensor.insert(resultShapeTensor.end(),
                                     indexShapeTensor.begin(),
                                     indexShapeTensor.end());
            continue;
          }
          resultShapeTensor.emplace_back(dataShapeTensor[i]);
        }
        auto resultSizeList = rewriter.create<Torch::PrimListConstructOp>(
            loc, Torch::ListType::get(Torch::IntType::get(data.getContext())),
            resultShapeTensor);

        rewriter.replaceOpWithNewOp<Torch::AtenViewOp>(
            binder.op, resultType, expandedGather, resultSizeList);
        return success();
      });
  patterns.onOp(
      "GatherElements", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, indices;
        int64_t axis;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(axis, "axis", 0))
          return failure();
        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));
        Value sparseGrad = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));
        rewriter.replaceOpWithNewOp<Torch::AtenGatherOp>(
            binder.op, resultType, data, constAxis, indices, sparseGrad);
        return success();
      });
  patterns.onOp(
      "Gemm", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value a, b, c;
        float alpha, beta;
        int64_t transA, transB;
        if (binder.tensorOperandAtIndex(a, 0) ||
            binder.tensorOperandAtIndex(b, 1) ||
            binder.tensorOperandAtIndex(c, 2) ||
            binder.s64IntegerAttr(transA, "transA", 0) ||
            binder.s64IntegerAttr(transB, "transB", 0) ||
            binder.f32FloatAttr(alpha, "alpha", 1.0f) ||
            binder.f32FloatAttr(beta, "beta", 1.0f) ||
            binder.tensorResultType(resultType))
          return failure();

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        Value one = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

        auto transpose = [&](Value m) -> Value {
          auto tty = m.getType().cast<Torch::ValueTensorType>();
          auto shape = tty.getOptionalSizes();
          if (shape.has_value()) {
            llvm::SmallVector<int64_t> newShape(shape.value());
            std::reverse(newShape.begin(), newShape.end());
            shape = std::move(newShape);
          }
          auto oty = Torch::ValueTensorType::get(tty.getContext(), shape,
                                                 tty.getOptionalDtype());
          return rewriter.create<Torch::AtenTransposeIntOp>(binder.getLoc(),
                                                            oty, m, zero, one);
        };

        if (transA) {
          a = transpose(a);
        }

        if (transB) {
          b = transpose(b);
        }

        Value mm =
            rewriter.create<Torch::AtenMmOp>(binder.getLoc(), resultType, a, b);
        if (alpha == 1.0 && beta == 1.0) {
          rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
              binder.op, resultType, mm, c, one);
          return success();
        }

        if (alpha != 1.0 && beta != 1.0) {
          Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
              binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(alpha));
          mm = rewriter.create<Torch::AtenMulScalarOp>(
              binder.getLoc(), resultType, mm, constAlpha);
          alpha = 1.0;
        }

        if (alpha != 1.0) {
          std::swap(alpha, beta);
          std::swap(mm, c);
        }

        Value constBeta = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(beta));
        rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
            binder.op, resultType, mm, c, constBeta);
        return success();
      });
  patterns.onOp(
      "GlobalAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType))
          return failure();

        auto inputTensorType = operand.getType().cast<Torch::ValueTensorType>();
        if (!inputTensorType || !inputTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input type having sizes");
        }
        ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
        unsigned inputRank = inputShape.size();
        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        ArrayRef<int64_t> resultShape = resultType.getSizes();

        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        for (unsigned i = 2; i < inputRank; i++) {
          int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
          cstKernel.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(kernelSize)));
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        if (inputRank == 3) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool1dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad);
          return success();
        } else if (inputRank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool2dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
          return success();
        } else if (inputRank == 5) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool3dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
          return success();
        }
        return failure();
      });
  patterns.onOp("LayerNormalization", 17,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType Y_type;
                  Torch::ValueTensorType Mean_type;
                  Torch::ValueTensorType InvStdDev_type;
                  Value X;
                  Value Scale;
                  Value B;
                  int64_t axis;
		  float epsilon;
                  int64_t stash_type;
                  if (binder.tensorOperandAtIndex(X, 0) ||
                      binder.tensorOperandAtIndex(Scale, 1) ||
                      binder.tensorOperandAtIndex(B, 2) ||
                      binder.tensorResultTypeAtIndex(Y_type, 0) || 
		                  binder.tensorResultTypeAtIndex(Mean_type, 1) ||
                      binder.tensorResultTypeAtIndex(InvStdDev_type, 2) || 
		                  binder.s64IntegerAttr(axis, "axis", -1) ||
                      binder.f32FloatAttr(epsilon, "epsilon", 0.00001) ||
                      binder.s64IntegerAttr(stash_type, "stash_type", 1)) 
                    return failure(); 
                  Value constEpsilon = rewriter.create<Torch::ConstantFloatOp>(
                    binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                    rewriter.getF64FloatAttr(epsilon));
                  unsigned rank = 1;
                  if(std::optional<unsigned> maybeRank = Torch::getTensorRank(X))
                    rank = *maybeRank;
                  SmallVector<Value> normalized;
                  axis = Torch::toPositiveDim(axis, rank);
                  auto X_type = X.getType().cast<Torch::ValueTensorType>();
                  ArrayRef<int64_t> X_shape = X_type.getSizes();
                  for (int64_t n = axis; n < rank ; n++) {                      
                    normalized.push_back(rewriter.create<Torch::ConstantIntOp>(
                    binder.getLoc(), rewriter.getI64IntegerAttr(X_shape[n])));
                  }
                  Value normalized_shape = rewriter.create<Torch::PrimListConstructOp>(
                    binder.getLoc(),
                    Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
                    normalized);
                  rewriter.replaceOpWithNewOp<Torch::AtenNativeLayerNormOp>(
                      binder.op, Y_type, Mean_type, InvStdDev_type, X, normalized_shape, Scale, B, constEpsilon);
                  return success();
              });
  patterns.onOp("LeakyRelu", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  float alpha;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType) ||
                      binder.f32FloatAttr(alpha, "alpha", 0.01f))
                    return failure();
                  Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
                      binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                      rewriter.getF64FloatAttr(alpha));
                  rewriter.replaceOpWithNewOp<Torch::AtenLeakyReluOp>(
                      binder.op, resultType, operand, constAlpha);
                  return success();
                });
  patterns.onOp("Pow", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenPowTensorTensorOp>(
                    binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp(
      "Identity", 14, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value tensor;
        if (binder.tensorOperand(tensor) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        rewriter.replaceOpWithNewOp<Torch::AtenCloneOp>(
            binder.op, resultType, tensor, /*memory_format=*/noneVal);
        return success();
      });
}
