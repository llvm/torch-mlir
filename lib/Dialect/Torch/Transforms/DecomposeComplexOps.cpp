//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Helper function to check whether the `dtype` is None or Float type.
static bool isNoneOrFloatDtype(MLIRContext *context, Value dtype) {
  if (dtype.getType().isa<Torch::NoneType>())
    return true;
  int64_t dtypeInt;
  if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    return false;
  FailureOr<Type> resDtype =
      getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
  if (failed(resDtype))
    return false;
  return resDtype->isa<mlir::FloatType>();
}

// Helper function to compute the return type of the reduction function.
// `dim` specifies the dimension to reduce and `keepDim` preserves the rank of
// the input tensor.
static Type computeReductionType(PatternRewriter &rewriter, Operation *op,
                                 BaseTensorType tensorType, Value dim,
                                 bool keepDim) {
  SmallVector<int64_t> sizes;
  int64_t dimInt;
  if (tensorType.hasSizes()) {
    ArrayRef<int64_t> inputShape = tensorType.getSizes();
    int64_t inputRank = inputShape.size();
    if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
      dimInt = toPositiveDim(dimInt, inputRank);
      if (!isValidDim(dimInt, inputRank)) {
        (void)rewriter.notifyMatchFailure(op, "dim is not a valid dim");
        return nullptr;
      }
      sizes.append(inputShape.begin(), inputShape.end());
      // The dimension to be reduced is set to 1 when `keepDim` is true else it
      // is removed.
      if (keepDim)
        sizes[dimInt] = 1;
      else
        sizes.erase(sizes.begin() + dimInt);
    } else {
      unsigned reducedRank = keepDim ? inputRank : inputRank - 1;
      sizes.resize(reducedRank, kUnknownSize);
    }
  }

  Type resultType = tensorType.getWithSizesAndDtype(
      sizes.size() == 0 ? std::optional<ArrayRef<int64_t>>()
                        : llvm::ArrayRef(sizes),
      tensorType.getOptionalDtype());
  return resultType;
}

// Reduction function to calculate sum along given `dim`.
static Value createSumAlongDimension(PatternRewriter &rewriter, Location loc,
                                     Operation *op, Value input, Value dim,
                                     bool keepDim) {
  Value dimList = rewriter.create<PrimListConstructOp>(
      loc, Torch::ListType::get(dim.getType()), dim);
  Value keepDimCst = rewriter.create<ConstantBoolOp>(loc, keepDim);
  Value dtype = rewriter.create<ConstantNoneOp>(loc);
  Type resultType = computeReductionType(
      rewriter, op, input.getType().cast<BaseTensorType>(), dim, keepDim);
  if (!resultType)
    return nullptr;
  return rewriter.create<AtenSumDimIntListOp>(loc, resultType, input, dimList,
                                              keepDimCst, dtype);
}

// Reduction function to calculate max along given `dim`.
static Value createMaxAlongDimension(PatternRewriter &rewriter, Location loc,
                                     Operation *op, Value input, Value dim,
                                     bool keepDim) {
  Value keepDimCst = rewriter.create<ConstantBoolOp>(loc, keepDim);
  BaseTensorType valueType =
      computeReductionType(rewriter, op, input.getType().cast<BaseTensorType>(),
                           dim, keepDim)
          .cast<BaseTensorType>();
  if (!valueType)
    return nullptr;
  BaseTensorType indexType =
      valueType
          .getWithSizesAndDtype(
              !valueType.hasSizes() ? std::optional<ArrayRef<int64_t>>()
                                    : llvm::ArrayRef(valueType.getSizes()),
              IntegerType::get(op->getContext(), 64, IntegerType::Signed))
          .cast<BaseTensorType>();
  return rewriter
      .create<AtenMaxDimOp>(loc, valueType, indexType, input, dim, keepDimCst)
      .getValues();
}

// Helper for creating `aten::sub_tensor_op`.
static Value createTensorSub(PatternRewriter &rewriter, Location loc,
                             Type tensorType, Value lhs, Value rhs) {
  Value alpha =
      rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
  Value sub =
      rewriter.create<AtenSubTensorOp>(loc, tensorType, lhs, rhs, alpha);
  return sub;
}

// Share code between `softmax_backward` and `log_softmax_backward` ops.
// Returns x - y * sum(z, dim).
static Value createSoftmaxBackwardCommonKernel(PatternRewriter &rewriter,
                                               Location loc, Operation *op,
                                               Type tensorType, Value x,
                                               Value y, Value z, Value dim) {
  Value sum =
      createSumAlongDimension(rewriter, loc, op, z, dim, /*keepDim=*/true);
  if (!sum)
    return nullptr;
  auto broadcastSizeType =
      Torch::ListType::get(Torch::IntType::get(op->getContext()));
  Value broadcastSize = rewriter.create<AtenSizeOp>(loc, broadcastSizeType, z);
  Value sumBroadcast =
      rewriter.create<AtenBroadcastToOp>(loc, tensorType, sum, broadcastSize);
  Value temp =
      rewriter.create<AtenMulTensorOp>(loc, tensorType, y, sumBroadcast);

  Value sub = createTensorSub(rewriter, loc, tensorType, x, temp);
  return sub;
}

static SmallVector<int64_t> computeDimsOrderForMoveDim(int64_t srcDimInt,
                                                       int64_t dstDimInt,
                                                       unsigned inputRank) {
  llvm::iota_range<int64_t> dimsOrderIR(0, inputRank, /*inclusive=*/false);
  SmallVector<int64_t> dimsOrder(dimsOrderIR.begin(), dimsOrderIR.end());
  dimsOrder.erase(dimsOrder.begin() + srcDimInt);
  dimsOrder.insert(dimsOrder.begin() + dstDimInt, srcDimInt);
  return dimsOrder;
}

static bool parseEquation(const std::string &equation,
                          SmallVector<SmallVector<char>> &inputTokens,
                          SmallVector<char> &resultTokens) {
  SmallVector<char> inputToken;
  size_t index = 0;
  enum EquationVariable { kIsInput, kIsResult };
  EquationVariable currentVariable = kIsInput;
  while (index < equation.size()) {
    if (std::isalpha(equation[index])) {
      if (currentVariable == kIsInput) {
        inputToken.push_back(equation[index]);
      } else {
        resultTokens.push_back(equation[index]);
      }
    } else if (equation[index] == ',') {
      inputTokens.push_back(inputToken);
      inputToken.clear();
    } else if ((index < (equation.size() - 1)) &&
               (equation.substr(index, 2).find("->") != std::string::npos)) {
      inputTokens.push_back(inputToken);
      inputToken.clear();
      currentVariable = kIsResult;
      index++;
    } else if (equation[index] != ' ') {
      return false;
    }
    index++;
  }
  return true;
}

// [*batchingDims, *lhsOtherDims, *lhsReduceDims, *lhsContractingDims] =>
// [batchingDimsProd, lhsOtherDimsProd, lhsContractingDimsProd]
static Value collapseDimForMatmul(PatternRewriter &rewriter, Location loc,
                                  Value input, int64_t batchDimsLength,
                                  int64_t contractingDimsLength,
                                  int64_t otherDimsLength,
                                  int64_t reduceDimsLength, bool isLhs) {
  auto inputType = input.getType().cast<BaseTensorType>();
  auto inputRank = batchDimsLength + contractingDimsLength + otherDimsLength +
                   reduceDimsLength;
  SmallVector<Value> inputShapeTensor;
  for (auto i = 0; i < inputRank; ++i) {
    inputShapeTensor.emplace_back(rewriter.create<AtenSizeIntOp>(
        loc, input,
        rewriter.create<Torch::ConstantIntOp>(loc,
                                              rewriter.getI64IntegerAttr(i))));
  }

  SmallVector<Value> outShapeTensor;
  Value constOne =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  auto dimOffset = 0;

  auto appendDims = [&](int64_t dimLength) {
    Value prod = constOne;
    for (auto i = 0; i < dimLength; ++i) {
      prod = rewriter.create<AtenMulIntOp>(loc, prod,
                                           inputShapeTensor[i + dimOffset]);
    }
    outShapeTensor.emplace_back(prod);
    dimOffset += dimLength;
  };

  appendDims(batchDimsLength);
  if (!isLhs)
    appendDims(contractingDimsLength);
  appendDims(otherDimsLength + reduceDimsLength);
  if (isLhs)
    appendDims(contractingDimsLength);

  auto outShapeValue = rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(input.getContext())),
      outShapeTensor);

  auto outType = inputType.getWithSizesAndDtype(std::nullopt,
                                                inputType.getOptionalDtype());
  return rewriter.create<Torch::AtenReshapeOp>(loc, outType, input,
                                               outShapeValue);
}

// classify every dim token into different categories. Note that although we
// parse out reduce dims, we delay their execution until
// `performLastPermuteAndReduce`.
static void parseDimTokens(
    SmallVector<char> &lhsTokens, SmallVector<char> &rhsTokens,
    SmallVector<char> &finalResultTokens, SmallVector<char> &contractingDims,
    SmallVector<char> &lhsReduceDims, SmallVector<char> &rhsReduceDims,
    SmallVector<char> &batchingDims, SmallVector<char> &lhsOtherDims,
    SmallVector<char> &rhsOtherDims) {
  llvm::SmallDenseSet<char> lhsTokenSet(lhsTokens.begin(), lhsTokens.end());
  llvm::SmallDenseSet<char> rhsTokenSet(rhsTokens.begin(), rhsTokens.end());
  llvm::SmallDenseSet<char> finalResultTokenSet(finalResultTokens.begin(),
                                                finalResultTokens.end());

  for (size_t i = 0; i < lhsTokens.size(); ++i) {
    bool rhsContains = rhsTokenSet.contains(lhsTokens[i]);
    bool finalResultConatins = finalResultTokenSet.contains(lhsTokens[i]);
    // batching dim
    if (rhsContains && finalResultConatins) {
      batchingDims.push_back(lhsTokens[i]);
      // reduce dim of lhs
    } else if (!rhsContains && !finalResultConatins) {
      lhsReduceDims.push_back(lhsTokens[i]);
      // other dim of lhs
    } else if (finalResultConatins) {
      lhsOtherDims.push_back(lhsTokens[i]);
      // contracting dim of lhs
    } else if (rhsContains) {
      contractingDims.push_back(lhsTokens[i]);
    }
  }

  for (size_t i = 0; i < rhsTokens.size(); ++i) {
    bool lhsContains = lhsTokenSet.contains(rhsTokens[i]);
    bool finalResultConatins = finalResultTokenSet.contains(rhsTokens[i]);
    // batching dim
    if (lhsContains && finalResultConatins) {
      // reduce dim of rhs
    } else if (!lhsContains && !finalResultConatins) {
      rhsReduceDims.push_back(rhsTokens[i]);
      // other dim of rhs
    } else if (finalResultConatins) {
      rhsOtherDims.push_back(rhsTokens[i]);
      // contracting dim of rhs
    } else if (lhsContains) {
    }
  }
}

static void generateIdealReusltDimTokens(SmallVector<char> &batchingDims,
                                         SmallVector<char> &lhsOtherDims,
                                         SmallVector<char> &rhsOtherDims,
                                         SmallVector<char> &lhsReduceDims,
                                         SmallVector<char> &rhsReduceDims,
                                         SmallVector<char> &resultTokens) {
  // generate ideal result dims, i.e.,
  // [*batchingDims, *lhsOtherDims, *lhsReduceDims, *rhsOtherDims,
  // *rhsReduceDims]
  resultTokens.insert(resultTokens.end(), batchingDims.begin(),
                      batchingDims.end());
  resultTokens.insert(resultTokens.end(), lhsOtherDims.begin(),
                      lhsOtherDims.end());
  resultTokens.insert(resultTokens.end(), lhsReduceDims.begin(),
                      lhsReduceDims.end());
  resultTokens.insert(resultTokens.end(), rhsOtherDims.begin(),
                      rhsOtherDims.end());
  resultTokens.insert(resultTokens.end(), rhsReduceDims.begin(),
                      rhsReduceDims.end());
}

static Value permuteTensorForMatmul(PatternRewriter &rewriter, Location loc,
                                    Value input, SmallVector<char> &dimTokens,
                                    SmallVector<char> &batchingDims,
                                    SmallVector<char> &contractingDims,
                                    SmallVector<char> &otherDims,
                                    SmallVector<char> &reduceDims, bool isLhs) {
  auto inputType = input.getType().cast<BaseTensorType>();
  llvm::SmallDenseMap<char, int64_t> dimTokenMap;
  for (size_t idx = 0; idx < dimTokens.size(); ++idx) {
    dimTokenMap[dimTokens[idx]] = idx;
  }

  SmallVector<Value> permuteVec;
  auto appendDims = [&](SmallVector<char> dimTokens) {
    for (auto d : dimTokens) {
      permuteVec.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dimTokenMap[d])));
    }
  };

  appendDims(batchingDims);
  if (!isLhs)
    appendDims(contractingDims);
  appendDims(otherDims);
  appendDims(reduceDims);
  if (isLhs)
    appendDims(contractingDims);

  Value dstDims = rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(rewriter.getContext())),
      permuteVec);
  auto outType = inputType.getWithSizesAndDtype(std::nullopt,
                                                inputType.getOptionalDtype());
  return rewriter.create<Torch::AtenPermuteOp>(loc, outType, input, dstDims);
}

static LogicalResult performMatmul(PatternRewriter &rewriter, Location loc,
                                   Value lhs, SmallVector<char> &lhsTokens,
                                   Value rhs, SmallVector<char> &rhsTokens,
                                   Value &result,
                                   SmallVector<char> &resultTokens,
                                   SmallVector<char> &finalResultTokens) {
  auto lhsType = lhs.getType().cast<BaseTensorType>();
  auto rhsType = rhs.getType().cast<BaseTensorType>();

  Type outputDType = lhsType.hasDtype() ? lhsType.getOptionalDtype()
                                        : rhsType.getOptionalDtype();

  llvm::SmallDenseMap<char, Value> lhsDimShapeMap;
  for (size_t idx = 0; idx < lhsTokens.size(); ++idx) {
    char d = lhsTokens[idx];
    lhsDimShapeMap[d] = rewriter.create<AtenSizeIntOp>(
        loc, lhs,
        rewriter.create<Torch::ConstantIntOp>(loc,
                                              rewriter.getI64IntegerAttr(idx)));
  }
  llvm::SmallDenseMap<char, Value> rhsDimShapeMap;
  for (size_t idx = 0; idx < rhsTokens.size(); ++idx) {
    char d = rhsTokens[idx];
    rhsDimShapeMap[d] = rewriter.create<AtenSizeIntOp>(
        loc, rhs,
        rewriter.create<Torch::ConstantIntOp>(loc,
                                              rewriter.getI64IntegerAttr(idx)));
  }

  // parse batch, contracting, other, reduce dims of lhs and rhs
  SmallVector<char> contractingDims;
  SmallVector<char> lhsReduceDims;
  SmallVector<char> rhsReduceDims;
  SmallVector<char> lhsOtherDims;
  SmallVector<char> rhsOtherDims;
  SmallVector<char> batchingDims;
  parseDimTokens(lhsTokens, rhsTokens, finalResultTokens, contractingDims,
                 lhsReduceDims, rhsReduceDims, batchingDims, lhsOtherDims,
                 rhsOtherDims);

  llvm::SmallDenseMap<char, Value> outDimShapeMap;
  auto generateOutDimShapeMap = [&](SmallVector<char> &dims) {
    for (auto d : dims) {
      bool lhsContains = lhsDimShapeMap.count(d) > 0;
      bool rhsContains = rhsDimShapeMap.count(d) > 0;
      if (lhsContains && rhsContains) {
        outDimShapeMap[d] = rewriter.create<Torch::PrimMaxIntOp>(
            loc, lhsDimShapeMap[d], rhsDimShapeMap[d]);
      } else if (lhsContains) {
        outDimShapeMap[d] = lhsDimShapeMap[d];
      } else if (rhsContains) {
        outDimShapeMap[d] = rhsDimShapeMap[d];
      }
    }
  };

  generateOutDimShapeMap(contractingDims);
  generateOutDimShapeMap(batchingDims);
  generateOutDimShapeMap(lhsReduceDims);
  generateOutDimShapeMap(rhsReduceDims);
  generateOutDimShapeMap(lhsOtherDims);
  generateOutDimShapeMap(rhsOtherDims);

  if (contractingDims.size() == 0 && lhsOtherDims.size() == 0 &&
      rhsOtherDims.size() == 0) {
    return rewriter.notifyMatchFailure(
        loc, "Hadamard product is currently not supported");
  }

  // shape: [*batchingDims, *lhsOtherDims, *lhsReduceDims, *lhsContractingDims]
  lhs = permuteTensorForMatmul(rewriter, loc, lhs, lhsTokens, batchingDims,
                               contractingDims, lhsOtherDims, lhsReduceDims,
                               true);
  // shape: [*batchingDims, *rhsContractingDims, *rhsOtherDims, *rhsReduceDims]
  rhs = permuteTensorForMatmul(rewriter, loc, rhs, rhsTokens, batchingDims,
                               contractingDims, rhsOtherDims, rhsReduceDims,
                               false);
  // shape: [batchingDimsProd, lhsOtherDimsProd, lhsContractingDimsProd]
  lhs = collapseDimForMatmul(rewriter, loc, lhs, batchingDims.size(),
                             contractingDims.size(), lhsOtherDims.size(),
                             lhsReduceDims.size(), true);
  // shape: [batchingDimsProd, rhsContractingDimsProd, rhsOtherDimsProd]
  rhs = collapseDimForMatmul(rewriter, loc, rhs, batchingDims.size(),
                             contractingDims.size(), rhsOtherDims.size(),
                             rhsReduceDims.size(), false);

  // perform matmul
  auto outType = lhsType.getWithSizesAndDtype(std::nullopt, outputDType);
  result = rewriter.create<Torch::AtenMatmulOp>(loc, outType, lhs, rhs);

  // generate ideal result dims.
  generateIdealReusltDimTokens(batchingDims, lhsOtherDims, rhsOtherDims,
                               lhsReduceDims, rhsReduceDims, resultTokens);

  // reshape matmul result to ideal shape:
  // [batchingDimsProd, lhsOtherDimsProd, rhsOtherDimsProd] =>
  // [*batchingDims, *lhsOtherDims, *lhsReduceDims, *rhsOtherDims,
  // *rhsReduceDims]
  SmallVector<Value> outShapeTensors;
  for (char d : resultTokens) {
    outShapeTensors.emplace_back(outDimShapeMap[d]);
  }

  auto outResultShape = rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(lhs.getContext())),
      outShapeTensors);
  result = rewriter.create<Torch::AtenReshapeOp>(
      loc, lhsType.getWithSizesAndDtype(std::nullopt, outputDType), result,
      outResultShape);
  return success();
}

static Value performLastReduceAndPermute(PatternRewriter &rewriter,
                                         Location loc, Type outType,
                                         Value input,
                                         SmallVector<char> &inputTokens,
                                         SmallVector<char> &outTokens) {
  auto inputType = input.getType().cast<BaseTensorType>();

  llvm::SmallDenseSet<char> outTokenSet(outTokens.begin(), outTokens.end());
  SmallVector<int64_t> sumDims;
  llvm::SmallDenseMap<char, int64_t> inputDimToIdx;
  int64_t idx = 0;
  for (size_t i = 0; i < inputTokens.size(); ++i) {
    char d = inputTokens[i];
    if (!outTokenSet.contains(d)) {
      sumDims.emplace_back(i);
    } else {
      inputDimToIdx[d] = idx++;
    }
  }

  if (sumDims.size() > 0) {
    SmallVector<Value> sumDimsTensor;
    for (auto d : sumDims) {
      sumDimsTensor.emplace_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(d)));
    }
    auto sumDimsListValue = rewriter.create<Torch::PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(rewriter.getContext())),
        sumDimsTensor);
    auto falseValue = rewriter.create<Torch::ConstantBoolOp>(
        loc, rewriter.getBoolAttr(false));
    auto noneValue = rewriter.create<Torch::ConstantNoneOp>(loc);
    input = rewriter.create<Torch::AtenSumDimIntListOp>(
        loc,
        inputType.getWithSizesAndDtype(std::nullopt,
                                       inputType.getOptionalDtype()),
        input, sumDimsListValue, falseValue, noneValue);
  }

  SmallVector<Value> permuteDimsTensor;
  for (auto d : outTokens) {
    permuteDimsTensor.emplace_back(rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(inputDimToIdx[d])));
  }
  auto permuteDimsListValue = rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(input.getContext())),
      permuteDimsTensor);
  auto out = rewriter.create<Torch::AtenPermuteOp>(loc, outType, input,
                                                   permuteDimsListValue);
  return out;
}

namespace {
/// We decompose aten.amax into a set of aten.max.dim op(s) depending on the
/// number of dimensions across which the max needs to be computed.
/// Eg:
///   INPUT:
///      final_output = aten.amax(initial_input, dim=(0, 2, 1), keepdim=False)
///
///   OUTPUT:
///      input_1 = aten.max.dim(initial_input, 2, keepdim)  #1
///      input_2 = aten.max.dim(input_1, 1, keepdim)   #2
///      final_output = aten.max.dim(input_2, 0, keepdim) #3
///
/// NOTE: We iterate over, in reverse order, every dimension included in `dim`
///       of the `aten.amax` op and create an `aten.amax.dim` op.
///       Input tensor to the next `aten.amax.dim` op is thus the output of the
///       previous `aten.amax.dim` op.
class DecomposeAtenAmaxOp : public OpRewritePattern<AtenAmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAmaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<int64_t, 4> dims;
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(dims)))

      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");

    bool keepDim;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(
          op, "Expected a constant boolean value for keepDim");

    Value input = op.getSelf();
    auto inputTy = input.getType().dyn_cast<Torch::ValueTensorType>();
    if (!inputTy || !inputTy.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "Expected input type having sizes");
    }
    // For every dimension included in `dim` of the op, iterated over in
    // reverse order, we create a call to aten.max.dim.
    std::sort(dims.rbegin(), dims.rend());
    for (int64_t dimInt : dims) {
      int64_t inputRank = inputTy.getSizes().size();
      dimInt = toPositiveDim(dimInt, inputRank);
      if (!isValidDim(dimInt, inputRank))
        return rewriter.notifyMatchFailure(op, "dim is statically invalid");
      Value dim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dimInt));
      // The input to the next invocation of aten.max.dim is the output of the
      // previous aten.max.dim op.
      input = createMaxAlongDimension(rewriter, loc, op, input, dim, keepDim);
    }
    rewriter.replaceOp(op, input);
    return success();
  }
};
} // end namespace

namespace {
class DecomposeAtenTriuOp : public OpRewritePattern<AtenTriuOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTriuOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    Location loc = op.getLoc();
    Value input = op.getSelf();
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasSizes() || !inputType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "should have shape and dtype");
    }
    if (inputType.getSizes().size() < 2) {
      return rewriter.notifyMatchFailure(op, "the rank of tensor should >= 2");
    }

    auto baseType = ValueTensorType::getWithLeastStaticInformation(context);
    Value cstZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value none = rewriter.create<ConstantNoneOp>(loc);

    Value rowDim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(-2));
    Value colDim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(-1));
    Value rowSize = rewriter.create<AtenSizeIntOp>(loc, input, rowDim);
    Value colSize = rewriter.create<AtenSizeIntOp>(loc, input, colDim);

    Value rowArange = rewriter.create<AtenArangeOp>(
        loc, baseType, rowSize, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none);
    Value colArange = rewriter.create<AtenArangeOp>(
        loc, baseType, colSize, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none);

    Value unsqueezeRowArange =
        rewriter.create<AtenUnsqueezeOp>(loc, baseType, rowArange, cstOne);
    Value unsqueezeColArange =
        rewriter.create<AtenUnsqueezeOp>(loc, baseType, colArange, cstZero);

    Value unsqueezeRowArangePlusDiagonal = rewriter.create<AtenAddScalarOp>(
        loc, baseType, unsqueezeRowArange, op.getDiagonal(), cstOne);

    Value condTensor = rewriter.create<AtenGeTensorOp>(
        loc, baseType, unsqueezeColArange, unsqueezeRowArangePlusDiagonal);

    rewriter.replaceOpWithNewOp<AtenWhereScalarOtherOp>(
        op, op.getResult().getType(), condTensor, input, cstZero);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenSizeOp : public OpRewritePattern<AtenSizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();

    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
    unsigned rank = *maybeRank;
    SmallVector<Value> sizes;
    for (unsigned i = 0; i < rank; i++) {
      Value dim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i));
      sizes.push_back(rewriter.create<AtenSizeIntOp>(loc, self, dim));
    }

    Value sizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), sizes);
    rewriter.replaceOp(op, sizeList);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenSelectIntOp : public OpRewritePattern<AtenSelectIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSelectIntOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value start = op.getIndex();
    Value dim = op.getDim();
    Value self = op.getSelf();

    // convert `start` to non-negative: start += int(start < 0) * dimSize
    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value isNegative = rewriter.create<AtenLtIntOp>(loc, start, zero);
    isNegative = rewriter.create<AtenIntBoolOp>(loc, isNegative);
    Value dimSize = rewriter.create<AtenSizeIntOp>(loc, self, dim);
    Value indexOffset = rewriter.create<AtenMulIntOp>(loc, isNegative, dimSize);
    start = rewriter.create<AtenAddIntOp>(loc, start, indexOffset);

    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value startPlusOne =
        rewriter.create<AtenAddIntOp>(loc, one.getType(), start, one);
    Value slice = rewriter.create<AtenSliceTensorOp>(
        loc,
        computeReductionType(rewriter, op,
                             self.getType().cast<BaseTensorType>(), dim,
                             /*keepDim=*/true),
        op.getSelf(), dim, start, startPlusOne, /*step=*/one);

    // `aten.slice.tensor` doesn't squeeze the dim even when it's size 1 after
    // slicing, while `aten.select.int` does.
    rewriter.replaceOpWithNewOp<AtenSqueezeDimOp>(op, op.getResult().getType(),
                                                  slice, op.getDim());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNarrowOp : public OpRewritePattern<AtenNarrowOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNarrowOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value start = op.getStart();
    Value dim = op.getDim();
    Value length = op.getLength();

    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value startPlusLength =
        rewriter.create<AtenAddIntOp>(loc, one.getType(), start, length);

    rewriter.replaceOpWithNewOp<AtenSliceTensorOp>(
        op, op.getResult().getType(), op.getSelf(), /*dim=*/dim,
        /*start=*/start,
        /*end=*/startPlusLength, /*step=*/one);

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.narrow.Tensor` to `aten.narrow` op
class DecomposeAtenNarrowTensorOp
    : public OpRewritePattern<AtenNarrowTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNarrowTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *context = op.getContext();
    // PyTorch makes sure that `start` param is an 0-dim integral tensor.
    // REF: https://pytorch.org/docs/stable/generated/torch.narrow.html.
    auto start = rewriter.create<Torch::AtenScalarImplicitOp>(
        loc, Torch::IntType::get(context), op.getStart());
    rewriter.replaceOpWithNewOp<Torch::AtenNarrowOp>(
        op, op.getType(), op.getSelf(), op.getDim(), start, op.getLength());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenGluOp : public OpRewritePattern<AtenGluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenGluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value dim = op.getDim();

    auto outputTy = op.getType().dyn_cast<Torch::ValueTensorType>();
    if (!outputTy || !outputTy.hasSizes() || !outputTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "Expected output type having sizes and dtype");
    }

    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value dimSize = rewriter.create<AtenSizeIntOp>(loc, self, dim);
    Value two =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(2));

    Value remainder = rewriter.create<AtenRemainderIntOp>(loc, dimSize, two);
    Value eqOrNot = rewriter.create<AtenEqIntOp>(loc, remainder, zero);

    rewriter.create<RuntimeAssertOp>(
        loc, eqOrNot,
        rewriter.getStringAttr("AtenGluOp's dim size must be multiple of 2"));

    Value splitLength = rewriter.create<AtenFloordivIntOp>(loc, dimSize, two);
    Value a = rewriter.create<AtenNarrowOp>(loc, outputTy, self, dim, zero,
                                            splitLength);
    Value b = rewriter.create<AtenNarrowOp>(loc, outputTy, self, dim,
                                            splitLength, splitLength);
    // a⊗σ(b)
    Value sigmoidB = rewriter.create<AtenSigmoidOp>(loc, outputTy, b);
    Value result = rewriter.create<AtenMulTensorOp>(loc, outputTy, a, sigmoidB);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenZeroOp : public OpRewritePattern<AtenZeroOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenZeroOp op,
                                PatternRewriter &rewriter) const override {
    Value zero = rewriter.create<ConstantIntOp>(op.getLoc(),
                                                rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<AtenFillScalarOp>(op, op.getType(),
                                                  op.getSelf(), zero);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenEyeOp : public OpRewritePattern<AtenEyeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEyeOp op,
                                PatternRewriter &rewriter) const override {
    Value n = op.getN();
    Value m = op.getN();
    rewriter.replaceOpWithNewOp<AtenEyeMOp>(op, op.getType(), n, m,
                                            op.getDtype(), op.getLayout(),
                                            op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenEyeMOp : public OpRewritePattern<AtenEyeMOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEyeMOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t n;

    if (!matchPattern(op.getN(), m_TorchConstantInt(&n)))
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: n must be constant");
    int64_t m;
    if (!matchPattern(op.getM(), m_TorchConstantInt(&m)))
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: m must be constant");
    Value none = rewriter.create<ConstantNoneOp>(loc);
    auto outType = op.getType().dyn_cast<BaseTensorType>();
    if (!outType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types input are currently supported");
    if (!outType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    if (n < 0) {
      return rewriter.notifyMatchFailure(op, "n must be greater or equal to 0");
    }
    if (m < 0) {
      return rewriter.notifyMatchFailure(op, "m must be greater or equal to 0");
    }

    auto context = op.getContext();
    auto int64Dtype = getDtypeIntValueForType(
        rewriter, loc,
        rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true));
    auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);
    auto arangeType = outType.getWithSizesAndDtype(llvm::ArrayRef(n), si64Type);
    Value rangeN = rewriter.create<AtenArangeOp>(
        loc, arangeType, op.getN(), /*dtype=*/int64Dtype, /*layout=*/none,
        /*device=*/op.getDevice(), /*pin_memory=*/none);

    auto arangeType1 =
        outType.getWithSizesAndDtype(llvm::ArrayRef(m), si64Type);
    Value rangeM = rewriter.create<AtenArangeOp>(
        loc, arangeType1, op.getM(), /*dtype=*/int64Dtype, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none);

    Value constMinusOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(-1));
    auto unsqzTensorInfo =
        unsqueezeTensor(rewriter, op, rangeN, /*dim=*/constMinusOne);
    if (failed(unsqzTensorInfo)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor");
    }
    Value unsqzRangeN = *unsqzTensorInfo;

    // compare unsqueezed input with boundaries
    auto eqType = ValueTensorType::get(
        context, op.getType().cast<BaseTensorType>().getSizes(),
        IntegerType::get(context, 1));
    Value eqTensor =
        rewriter.create<AtenEqTensorOp>(loc, eqType, unsqzRangeN, rangeM);

    Value dtype = op.getDtype();
    if (dtype.getType().isa<Torch::BoolType>()) {
      rewriter.replaceOp(op, eqTensor);
      return success();
    } else {
      auto zero =
          rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
      auto one =
          rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
      Value outTensor =
          rewriter.create<AtenWhereScalarOp>(loc, outType, eqTensor, one, zero);
      rewriter.replaceOp(op, outTensor);
      return success();
    }
  }
};
} // namespace

namespace {
class DecomposeAtenIsnanOp : public OpRewritePattern<AtenIsnanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIsnanOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getSelf();
    // Create a new aten.ne operation with the same type and input value.
    rewriter.replaceOpWithNewOp<AtenNeTensorOp>(op, op.getType(), input, input);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenIsinfOp : public OpRewritePattern<AtenIsinfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIsinfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();

    mlir::FloatType f64Type = rewriter.getF64Type();
    Value inf = rewriter.create<ConstantFloatOp>(
        loc, rewriter.getFloatAttr(
                 f64Type, APFloat::getInf(f64Type.getFloatSemantics())));
    Value abs = rewriter.create<AtenAbsOp>(loc, self.getType(), self);
    rewriter.replaceOpWithNewOp<AtenEqScalarOp>(op, op.getType(), abs, inf);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenIsneginfOp : public OpRewritePattern<AtenIsneginfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIsneginfOp op,
                                PatternRewriter &rewriter) const override {
    mlir::FloatType f64Type = rewriter.getF64Type();
    Value inf = rewriter.create<ConstantFloatOp>(
        op.getLoc(),
        rewriter.getFloatAttr(
            f64Type, APFloat::getInf(f64Type.getFloatSemantics(), true)));
    rewriter.replaceOpWithNewOp<AtenEqScalarOp>(op, op.getType(), op.getSelf(),
                                                inf);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenIsposinfOp : public OpRewritePattern<AtenIsposinfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIsposinfOp op,
                                PatternRewriter &rewriter) const override {
    mlir::FloatType f64Type = rewriter.getF64Type();
    Value inf = rewriter.create<ConstantFloatOp>(
        op.getLoc(),
        rewriter.getFloatAttr(f64Type,
                              APFloat::getInf(f64Type.getFloatSemantics())));
    rewriter.replaceOpWithNewOp<AtenEqScalarOp>(op, op.getType(), op.getSelf(),
                                                inf);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenReshapeOp : public OpRewritePattern<AtenReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getSelf();
    // TODO: Handle non value tensor type operands.
    if (!input.getType().isa<ValueTensorType>()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only value tensor type operands are supported");
    }
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), input,
                                            op.getShape());
    return success();
  }
};
} // namespace

namespace {
// Decompose AtenEinsumOp to AtenMatmulOp, and supports possible reduce
// operation and permute operation. Currently, this pass doesn't support
// Hadamard product. The basic idea is that:
//  Step 1: split the string equation to input/result tokens and find
//    batchingDims, contractingDims, otherDims and reduceDims.
//  Step 2: permute and reshape input tensors suitable
//    for matmul operations.
//  Step 3: use AtenMatmulOp to get the result.
//  Step 4: iteratively execute step 2 & 3 until we get the final result.
//  Step 5: perform remaining permute and reduce operations.
// notice: support static shape only

class DecomposeAtenEinsumOp : public OpRewritePattern<AtenEinsumOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEinsumOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    std::string equation;
    if (!matchPattern(op.getEquation(), m_TorchConstantStr(equation))) {
      return rewriter.notifyMatchFailure(op, "Unsupported value of equation");
    }
    SmallVector<char> resultTokens;
    SmallVector<SmallVector<char>> inputTokens;
    if (!parseEquation(equation, inputTokens, resultTokens)) {
      return rewriter.notifyMatchFailure(
          op, "Unexpected character in equations encountered");
    }

    SmallVector<Value> inputTensors;
    if (!getListConstructElements(op.getTensors(), inputTensors)) {
      return rewriter.notifyMatchFailure(
          op, "input should comes from a PrimListConstructOp");
    }

    auto allTensorHasSizes = [](Value tensor) {
      auto type = tensor.getType().dyn_cast<BaseTensorType>();
      if (!type || !type.hasSizes())
        return false;
      return true;
    };

    if (!llvm::all_of(inputTensors, allTensorHasSizes)) {
      return rewriter.notifyMatchFailure(op,
                                         "all input tensors should have sizes");
    }

    SmallVector<char> lhsTokens = inputTokens[0];
    Value lhs = inputTensors[0];
    Value result;

    for (size_t i = 1; i < inputTensors.size(); ++i) {
      auto rhs = inputTensors[i];
      auto rhsTokens = inputTokens[i];
      SmallVector<char> outTokens;
      if (failed(performMatmul(rewriter, loc, lhs, lhsTokens, rhs, rhsTokens,
                               result, outTokens, resultTokens))) {
        return failure();
      }
      lhs = result;
      lhsTokens = outTokens;
    }

    result = performLastReduceAndPermute(rewriter, loc, op.getType(), lhs,
                                         lhsTokens, resultTokens);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Calculate the trace of the input tensor as the sum over its diagonal
// elements. This computation is performed as:
//
// Step1: Obtain the diagonal using AtenDiagonalOp
// Step2: Compute the trace using AtenSumOp.
//
// It is verified that the input tensor has rank two.
class DecomposeAtenTraceOp : public OpRewritePattern<AtenTraceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTraceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    std::optional<unsigned> inRank = getTensorRank(self);
    if (inRank != 2)
      return rewriter.notifyMatchFailure(
          op, "Expected input tensor to have rank 2.");

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    BaseTensorType inputType = self.getType().cast<BaseTensorType>();

    Value output = op.getResult();
    BaseTensorType outputType = output.getType().cast<BaseTensorType>();

    ArrayRef<int64_t> inputShape = inputType.getSizes();
    int64_t diagonalSize = std::min(inputShape[0], inputShape[1]);
    SmallVector<int64_t> diagonalShape{diagonalSize};
    Type elementType = inputType.getOptionalDtype();
    Type diagonalType = inputType.getWithSizesAndDtype(
        llvm::ArrayRef(diagonalShape), elementType);

    Value diagonal = rewriter.create<AtenDiagonalOp>(
        loc, diagonalType, /*input=*/self, /*offset=*/zero, /*dim1=*/zero,
        /*dim2=*/one);
    Value sum = rewriter.create<AtenSumOp>(loc, outputType, /*self=*/diagonal,
                                           /*dtype=*/none);
    rewriter.replaceOp(op, sum);
    return success();
  }
};
} // namespace

// Calculates the softmax function on the given `input` tensor. Softmax(x) =
// exp(x)/sum(exp(x)).
// To avoid overflow we use the following decomposition rule:
//     x_max = max(input, dim, keepdim = True)
//     unnorm = aten.exp(input - x_max)
//     softmax = unnorm / sum(unnorm, dim, keepdim = True)
template <typename OpTy>
static Value getSoftmaxResult(OpTy op, Value self, Type resultType,
                              PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value dim = op.getDim();
  Value xMax =
      createMaxAlongDimension(rewriter, loc, op, self, dim, /*keepDim=*/true);
  if (!xMax)
    return nullptr;
  Value unNormalized = createTensorSub(rewriter, loc, resultType, self, xMax);
  Value unNormalizedExp =
      rewriter.create<AtenExpOp>(loc, resultType, unNormalized);
  Value sum = createSumAlongDimension(rewriter, loc, op, unNormalizedExp, dim,
                                      /*keepDim=*/true);
  if (!sum)
    return nullptr;
  return rewriter.create<AtenDivTensorOp>(loc, resultType, unNormalizedExp,
                                          sum);
}

// Decompose softmax into: exp(x) / sum(exp(x))
namespace {
class DecomposeAtenSoftmaxIntOp : public OpRewritePattern<AtenSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType resultTensorType = op.getType().cast<BaseTensorType>();
    if (!resultTensorType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Type resultTensorDtype = resultTensorType.getDtype();
    if (!resultTensorDtype.isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op,
                                         "Only support floating-point type");

    // If `dtype` arg is non-none then convert the input to `dtype`.
    if (!op.getDtype().getType().isa<Torch::NoneType>()) {
      Location loc = op.getLoc();
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
      self = rewriter.create<AtenToDtypeOp>(
          loc, resultTensorType, self,
          getDtypeIntValueForType(rewriter, loc, resultTensorDtype),
          /*non_blocking=*/cstFalse, /*copy=*/cstFalse, /*memory_format=*/none);
    }

    Value result = getSoftmaxResult(op, self, resultTensorType, rewriter);
    if (!result)
      return failure();
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op.getType(),
                                                        result);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAten_SoftmaxOp : public OpRewritePattern<Aten_SoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");
    bool halfToFloat;
    if (!matchPattern(op.getHalfToFloat(), m_TorchConstantBool(&halfToFloat)))
      return rewriter.notifyMatchFailure(
          op, "Expected a boolean value for half_to_float");

    BaseTensorType resultTensorType = op.getType().cast<BaseTensorType>();
    if (!resultTensorType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Type resultTensorDtype = resultTensorType.getDtype();
    // `torch.ops.aten._softmax`'s softmax with half to float conversion is not
    // supported on CPU, but we go ahead with the decomposing.
    // TODO: Add an e2e test once upstream support is added.
    // If `half_to_float` is set, we convert the input's elemental type to match
    // that of output's.
    if (halfToFloat) {
      Location loc = op.getLoc();
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
      self = rewriter.create<AtenToDtypeOp>(
          loc, resultTensorType, self,
          getDtypeIntValueForType(rewriter, loc, resultTensorDtype),
          /*non_blocking=*/cstFalse, /*copy=*/cstFalse, /*memory_format=*/none);
    }
    Value result = getSoftmaxResult(op, self, resultTensorType, rewriter);
    if (!result)
      return op.emitError("failed to get softmax result");
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTensorType,
                                                        result);
    return success();
  }
};
} // namespace

// Aten_SoftmaxBackwardDataOp(gradOutput, output, dim) =>
//    newGrad = gradOutput * output
//    result = newGrad - output * sum(newGrad, dim))
//
// Refer to
// https://github.com/pytorch/pytorch/blob/15fecc4c830a3907fde4b44c9962dc4144da50a4/torch/csrc/jit/codegen/cuda/ops/normalization.cpp#L31
namespace {
class DecomposeAten_SoftmaxBackwardDataOp
    : public OpRewritePattern<Aten_SoftmaxBackwardDataOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_SoftmaxBackwardDataOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.getGradOutput();
    Value output = op.getOutput();
    Value dim = op.getDim();

    BaseTensorType tensorType = gradOutput.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value newGrad =
        rewriter.create<AtenMulTensorOp>(loc, tensorType, gradOutput, output);
    Value result = createSoftmaxBackwardCommonKernel(
        rewriter, loc, op, tensorType, newGrad, output, newGrad, dim);
    if (!result)
      return rewriter.notifyMatchFailure(
          op,
          "nullptr returned by createSoftmaxBackwardCommonKernel function.");
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// AtenTanhBackwardOp(gradOutput, output) =>
//    result = gradOutput * (1 - output^2)
// To get away from broadcasts the above formula is expanded i.e.,
// result = gradOutput - (gradOutput * output^2)
namespace {
class DecomposeAtenTanhBackwardOp
    : public OpRewritePattern<AtenTanhBackwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTanhBackwardOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.getGradOutput();

    // `output` is the value flowing out from tanh. Hence, tanh(x) = output.
    //  Since, dTanh(x) = (1 - tanh(x)^2) hence, dOutput = (1 - output^2).
    Value output = op.getOutput();

    BaseTensorType tensorType = gradOutput.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value tanhSquare =
        rewriter.create<AtenMulTensorOp>(loc, tensorType, output, output);
    Value gradMulTanhSquare = rewriter.create<AtenMulTensorOp>(
        loc, tensorType, tanhSquare, gradOutput);

    Value newGrad = createTensorSub(rewriter, loc, tensorType, gradOutput,
                                    gradMulTanhSquare);
    rewriter.replaceOp(op, newGrad);
    return success();
  }
};
} // namespace

// Aten_LogSoftmaxBackwardDataOp(gradOutput, output, dim) =>
//    result = gradOutput - (exp(output) * sum(gradOutput, dim))
namespace {
class DecomposeAten_LogSoftmaxBackwardDataOp
    : public OpRewritePattern<Aten_LogSoftmaxBackwardDataOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_LogSoftmaxBackwardDataOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.getGradOutput();
    Value output = op.getOutput();
    Value dim = op.getDim();

    BaseTensorType tensorType = gradOutput.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value expOut = rewriter.create<AtenExpOp>(loc, tensorType, output);
    Value result = createSoftmaxBackwardCommonKernel(
        rewriter, loc, op, tensorType, gradOutput, expOut, gradOutput, dim);
    if (!result)
      return rewriter.notifyMatchFailure(
          op,
          "nullptr returned by createSoftmaxBackwardCommonKernel function.");
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenAMinMaxOp : public OpRewritePattern<Torch::AtenAminOp> {
public:
  using OpRewritePattern<Torch::AtenAminOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Torch::AtenAminOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<int64_t> dimList;
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(dimList))) {
      return rewriter.notifyMatchFailure(op, "dims not foldable constants");
    }

    bool keepdim;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepdim))) {
      return rewriter.notifyMatchFailure(op, "keepdims not foldable constants");
    }

    auto loc = op.getLoc();
    std::sort(dimList.begin(), dimList.end(), std::greater<int64_t>());

    Value reduction = op.getSelf();
    auto resultTy = cast<Torch::ValueTensorType>(op.getType());
    auto reductionTy = cast<Torch::ValueTensorType>(reduction.getType());
    llvm::SmallVector<int64_t> reductionShape(reductionTy.getSizes());

    for (auto dim : dimList) {
      auto dimValue = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dim));
      reductionShape[dim] = 1;
      if (!keepdim) {
        for (int i = dim, s = reductionShape.size() - 1; i < s; ++i)
          reductionShape[i] = reductionShape[i + 1];
        reductionShape.resize(reductionShape.size() - 1);
      }

      reductionTy = rewriter.getType<Torch::ValueTensorType>(
          reductionShape, resultTy.getOptionalDtype());
      auto idxTy = rewriter.getType<Torch::ValueTensorType>(
          reductionShape, rewriter.getIntegerType(32, /*is_signed*/ true));
      llvm::SmallVector<Type, 2> types{reductionTy, idxTy};
      reduction = rewriter
                      .create<Torch::AtenMinDimOp>(loc, types, reduction,
                                                   dimValue, op.getKeepdim())
                      .getResult(0);
    }

    rewriter.replaceOp(op, reduction);
    return success();
  }
};
} // namespace

// Decompose `AtenArgMaxOp` into `AtenMaxDimOp` as well as `AtenArgMinOp` into
// `AtenMinDimOp`
namespace {
template <typename OpTy, typename DecompOpTy>
class DecomposeAtenArgMinMaxOp : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value dim = op.getDim();
    Value keepDim = op.getKeepdim();
    Value result = op.getResult();

    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    BaseTensorType indicesTensorType = result.getType().cast<BaseTensorType>();
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(
          op, "expected input tensor to have a rank");
    }
    unsigned inputRank = *maybeInputRank;
    if (!indicesTensorType.hasSizes())
      return failure();
    BaseTensorType valueTensorType =
        inputType
            .getWithSizesAndDtype(indicesTensorType.getOptionalSizes(),
                                  inputType.getOptionalDtype())
            .cast<BaseTensorType>();

    // If the dim type is `NoneType` i.e. reduce along all the dimensions.
    // `AtenMaxDimOp` and `AtenMinDimOp` do not support dim as `NoneType` so
    // first the input tensor is flattened to 1d tensor and then the reduction
    // happens on the 0th dimension.
    if (dim.getType().isa<Torch::NoneType>()) {
      BaseTensorType flattenType =
          inputType
              .getWithSizesAndDtype({kUnknownSize},
                                    inputType.getOptionalDtype())
              .cast<BaseTensorType>();
      dim = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
      Value end = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(inputRank - 1));
      input = rewriter.create<AtenFlattenUsingIntsOp>(loc, flattenType, input,
                                                      dim, end);
    }

    Value resultArg =
        rewriter
            .create<DecompOpTy>(loc, valueTensorType, indicesTensorType, input,
                                dim, keepDim)
            .getIndices();

    rewriter.replaceOp(op, resultArg);
    return success();
  }
};
} // namespace

// Decompose `aten.bucketize` into the following op sequence:
//
// def aten_bucketize(input, boundaries, out_int32, right):
//     unsqz_input = input.unsqueeze(-1)
//     if not right:
//         comparison = unsqz_input <= boundaries
//     else:
//         comparison = unsqz_input < boundaries
//     indices = torch.argmax(comparison.float(), dim=-1)
//     within_bound = comparison[..., -1]
//     result = torch.where(within_bound, indices, boundaries.shape[0])
//     if out_int32:
//         result = result.int()
//     return result
//
namespace {
class DecomposeAtenBucketizeTensorOp
    : public OpRewritePattern<AtenBucketizeTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBucketizeTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getSelf();
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: input must have known sizes");
    }
    ArrayRef<int64_t> inputShape = inputType.getSizes();

    Value boundaries = op.getBoundaries();
    auto boundariesType = boundaries.getType().cast<BaseTensorType>();
    if (!boundariesType.hasSizes() || boundariesType.getSizes().size() != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: boundaries must have "
                                         "known sizes and must be a 1D array");
    }
    int64_t boundariesSize = boundariesType.getSizes()[0];

    bool outInt32;
    if (!matchPattern(op.getOutInt32(), m_TorchConstantBool(&outInt32))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: out_int32 must be a constant bool");
    }

    bool right;
    if (!matchPattern(op.getRight(), m_TorchConstantBool(&right))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: right must be a constant bool");
    }

    // unsqueeze input at the last dim to make it broadcastable with boundaries
    Value constMinusOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(-1));
    auto unsqzTensorInfo =
        unsqueezeTensor(rewriter, op, input, /*dim=*/constMinusOne);
    if (failed(unsqzTensorInfo)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor");
    }
    Value unsqzInput = *unsqzTensorInfo;

    // compare unsqueezed input with boundaries
    SmallVector<int64_t> compareShape(inputShape);
    compareShape.push_back(boundariesSize);
    Type compareType =
        inputType.getWithSizesAndDtype(compareShape, rewriter.getI1Type());
    Value compare;
    if (!right) {
      compare = rewriter.create<AtenLeTensorOp>(loc, compareType, unsqzInput,
                                                boundaries);
    } else {
      compare = rewriter.create<AtenLtTensorOp>(loc, compareType, unsqzInput,
                                                boundaries);
    }

    // convert the comparison results to float32 as the argmax op input,
    // which does not support integer dtype in LINALG backend
    Value compareF32 =
        convertTensorToDtype(rewriter, loc, compare, rewriter.getF32Type());

    // get the first boundary index where the input element is less than (or
    // equal to) the boundary value
    Type indicesType = inputType.getWithSizesAndDtype(
        inputShape, rewriter.getIntegerType(64, IntegerType::Signed));
    Value constFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value indices = rewriter.create<AtenArgmaxOp>(loc, indicesType, compareF32,
                                                  /*dim=*/constMinusOne,
                                                  /*keepdim=*/constFalse);

    // get the comparison results between each input element and the rightmost
    // boundary value
    Type withinUpperBoundType =
        inputType.getWithSizesAndDtype(inputShape, rewriter.getI1Type());
    Value withinUpperBound = rewriter.create<AtenSelectIntOp>(
        loc, withinUpperBoundType, compare, /*dim=*/constMinusOne,
        /*index=*/constMinusOne);

    // If the input element is less than (or equal to) the rightmost boundary,
    // take the max index as result. Otherwise, the element is beyond the
    // rightmost boundary, so take the boundary size.
    Value constZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value upperBound =
        rewriter.create<AtenSizeIntOp>(loc, boundaries, /*dim=*/constZero);
    Value result = rewriter.create<AtenWhereScalarOtherOp>(
        loc, indicesType, withinUpperBound, indices, upperBound);

    if (outInt32) {
      result = convertTensorToDtype(
          rewriter, loc, result,
          rewriter.getIntegerType(32, IntegerType::Signed));
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// To avoid overflow we use the following decomposition rule:
//  x_max = aten.max(x, dim, keepdim=True)[0]
//  shifted = x - x_max
//  shifted_logsumexp = aten.log(aten.sum(aten.exp(shifted), dim, keepdim=True))
//  log_softmax = shifted - shifted_logsumexp
template <typename OpTy>
static Value getLogSoftmaxResult(OpTy op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value dim = op.getDim();
  Value self = op.getSelf();
  BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
  Value xMax =
      createMaxAlongDimension(rewriter, loc, op, self, dim, /*keepDim=*/true);
  if (!xMax)
    return nullptr;

  Value shifted = createTensorSub(rewriter, loc, tensorType, self, xMax);
  Value shiftedExp = rewriter.create<AtenExpOp>(loc, tensorType, shifted);
  Value shiftedSumExp =
      createSumAlongDimension(rewriter, loc, op, shiftedExp, dim,
                              /*keepDim=*/true);
  if (!shiftedSumExp)
    return nullptr;

  Value shiftedLogSumExp =
      rewriter.create<AtenLogOp>(loc, shiftedSumExp.getType(), shiftedSumExp);
  Value result =
      createTensorSub(rewriter, loc, op.getType(), shifted, shiftedLogSumExp);
  return result;
}

namespace {
class DecomposeAtenLogSoftmaxIntOp
    : public OpRewritePattern<AtenLogSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLogSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    if (!op.getDtype().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for log_softmax");

    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value logSoftmax = getLogSoftmaxResult(op, rewriter);
    if (!logSoftmax)
      return rewriter.notifyMatchFailure(
          op, "getLogSoftmaxResult function returned nullptr");
    rewriter.replaceOp(op, logSoftmax);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAten_LogSoftmaxOp : public OpRewritePattern<Aten_LogSoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_LogSoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    bool halfToFloat;
    if (!matchPattern(op.getHalfToFloat(), m_TorchConstantBool(&halfToFloat)))
      return rewriter.notifyMatchFailure(
          op, "Expected a boolean value for half_to_float");

    // Currently, setting `halfToFloat` is not supported as the E2E testing for
    // the same is not present on CPU.
    if (halfToFloat)
      return rewriter.notifyMatchFailure(
          op, "halfToFloat is currently not supported.");
    Value _logSoftmax = getLogSoftmaxResult(op, rewriter);
    if (!_logSoftmax)
      return rewriter.notifyMatchFailure(
          op, "getLogSoftmaxResult function returned nullptr");
    rewriter.replaceOp(op, _logSoftmax);
    return success();
  }
};
} // namespace

// Decompose aten.matmul into: aten.mm and aten.bmm according to ranks.
namespace {
class DecomposeAtenMatmulOp : public OpRewritePattern<AtenMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getSelf();
    Value rhs = op.getOther();

    std::optional<unsigned> maybeLhsRank = getTensorRank(lhs);
    std::optional<unsigned> maybeRhsRank = getTensorRank(rhs);
    if (!maybeLhsRank || !maybeRhsRank) {
      return rewriter.notifyMatchFailure(
          op, "expected input tensors to have a rank");
    }
    unsigned lhsRank = *maybeLhsRank;
    unsigned rhsRank = *maybeRhsRank;

    if (lhsRank == 2 && rhsRank == 2) {
      // If both lhs and rhs ranks are 2 then map it to `aten.mm` op.
      rewriter.replaceOpWithNewOp<AtenMmOp>(op, op.getType(), lhs, rhs);
    } else if (lhsRank == 3 && rhsRank == 3) {
      // If both lhs and rhs ranks are 3 then map it to `aten.bmm` op.
      rewriter.replaceOpWithNewOp<AtenBmmOp>(op, op.getType(), lhs, rhs);
    } else {
      return failure();
    }

    return success();
  }
};
} // namespace

// Decompose aten.mv into: aten.matmul.
namespace {
class DecomposeAtenMvOp : public OpRewritePattern<AtenMvOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMvOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getSelf();
    Value rhs = op.getVec();
    rewriter.replaceOpWithNewOp<AtenMatmulOp>(op, op.getType(), lhs, rhs);
    return success();
  }
};
} // namespace

// Decompose aten.pixel_shuffle into: prims.split_dim, aten.permute, and
// prims.collapse operations.
//
// If input is a tensor of shape
//     (*leading_dims, C*r*r, H, W),
//
// where leading_dims is of size N, then
//    X = pixel_shuffle(input, upscale_factor)
//
// gets replaced with
//    X = input.split_dim(...)  # shape (*leading_dims, C, r*r, H, W)
//    X = X.split_dim(...)      # shape (*leading_dims, C, r, r, H, W)
//    X = X.permute(0, ..., N, N+3, N+1, N+4, N+2)
//                              # shape (*leading_dims, C, H, r, W, r)
//    X = X.collapse(...)       # shape (*leading_dims, C, r, H, r*W)
//    X = X.collapse(...)       # shape (*leading_dims, C, r*H, r*W)
//
// 'r' above is referred to as the 'upscale factor' or just 'factor' below.
namespace {
class DecomposeAtenPixelShuffleOp
    : public OpRewritePattern<AtenPixelShuffleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenPixelShuffleOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value inValue = op.getSelf();
    auto inType = inValue.getType().cast<BaseTensorType>();
    auto maybeSizes = inType.getOptionalSizes();
    if (!maybeSizes) {
      return rewriter.notifyMatchFailure(
          op, "Expected input tensor to have known rank.");
    }
    auto inShape = maybeSizes.value();
    auto inRank = inShape.size();

    // The input tensor must have at least 3 dimensions: (1) the channel
    // dimension which gets smaller by 'factor*factor', (2) the H channel which
    // gets larger by 'factor' and (3) the W channel which get larger by
    // 'factor'. The total number of dimensions is 3 + N, where N is the number
    // of leading dimensions, and N >= 0 so the input must have rank at least 3.
    if (inRank < 3)
      return rewriter.notifyMatchFailure(
          op, "Expected input tensor to have rank greater than 2.");

    const auto inOptionalDType = inType.getOptionalDtype();

    auto getTypeFromShape = [inOptionalDType](auto &&vals) {
      // Get a vector of integers from a vector of Values.
      auto getIntShape = [](auto &&vals) {
        SmallVector<int64_t> shape;
        shape.reserve(vals.size());
        for (auto v : vals) {
          int64_t cst_val;
          if (matchPattern(v, m_TorchConstantInt(&cst_val))) {
            shape.push_back(cst_val);
          } else {
            shape.push_back(kUnknownSize);
          }
        }
        return shape;
      };

      const auto intShape = getIntShape(vals);
      return ValueTensorType::get(vals[0].getContext(),
                                  llvm::ArrayRef(intShape), inOptionalDType);
    };

    auto nLeadingDims = inRank - 3;

    // Get the size of the dimension 'i'. Note the use of 'createOrFold' instead
    // of 'create': if the dimension size is known, then the AtenSizeIntOp is
    // folded to a ConstantOp.
    auto getDimSize = [&](uint64_t i) -> Value {
      Value dim =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i));
      return rewriter.createOrFold<AtenSizeIntOp>(loc, inValue, dim);
    };

    auto inC = getDimSize(inRank - 3);
    auto inH = getDimSize(inRank - 2);
    auto inW = getDimSize(inRank - 1);

    auto factor = op.getUpscaleFactor();

    Value factorSquared =
        rewriter.createOrFold<AtenMulIntOp>(loc, factor, factor);

    Value outC =
        rewriter.createOrFold<AtenFloordivIntOp>(loc, inC, factorSquared);

    Value outH = rewriter.createOrFold<AtenMulIntOp>(loc, inH, factor);
    Value outW = rewriter.createOrFold<AtenMulIntOp>(loc, inW, factor);

    SmallVector<Value> dimensionConstants;
    dimensionConstants.reserve(inRank + 2);
    for (unsigned i = 0; i < inRank + 2; ++i) {
      dimensionConstants.push_back(
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i)));
    }

    SmallVector<Value> leadingDims;
    leadingDims.reserve(nLeadingDims);
    for (unsigned i = 0; i < nLeadingDims; ++i) {
      Value leadingDimSize = rewriter.createOrFold<AtenSizeIntOp>(
          loc, inValue, dimensionConstants[i]);
      leadingDims.push_back(leadingDimSize);
    }

    SmallVector<Value> partiallyExpandedShape = leadingDims;
    partiallyExpandedShape.append({outC, factorSquared, inH, inW});

    SmallVector<Value> prePermuteShape = leadingDims;
    prePermuteShape.append({outC, factor, factor, inH, inW});

    SmallVector<Value> postPermuteShape = leadingDims;
    postPermuteShape.append({outC, inH, factor, inW, factor});

    SmallVector<Value> partiallyCollapsedShape = leadingDims;
    partiallyCollapsedShape.append({outC, inH, factor, outW});

    SmallVector<Value> outShape = leadingDims;
    outShape.append({outC, outH, outW});

    SmallVector<Value> permutation{dimensionConstants.begin(),
                                   dimensionConstants.begin() + nLeadingDims};
    SmallVector<uint64_t> permutationTail{0, 3, 1, 4, 2};
    for (uint64_t d : permutationTail) {
      permutation.push_back(dimensionConstants[nLeadingDims + d]);
    }

    Value permuteDimsOrder = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op->getContext())),
        permutation);

    // Split input channel inC -> (inC, factorSquared)
    auto partiallyExpanded =
        rewriter
            .create<PrimsSplitDimOp>(
                loc, getTypeFromShape(partiallyExpandedShape), inValue,
                dimensionConstants[nLeadingDims], outC)
            .getResult();

    // Split new dimension factorSquared -> (factor, factor)
    auto fullyExpanded = rewriter.create<PrimsSplitDimOp>(
        loc, getTypeFromShape(prePermuteShape), partiallyExpanded,
        dimensionConstants[nLeadingDims + 1], factor);

    // Perform the permutation
    auto permuted =
        rewriter.create<AtenPermuteOp>(loc, getTypeFromShape(postPermuteShape),
                                       fullyExpanded, permuteDimsOrder);

    // Collapse final 2 dimension
    auto partiallyCollapsed = rewriter.create<PrimsCollapseOp>(
        loc, getTypeFromShape(partiallyCollapsedShape), permuted,
        dimensionConstants[nLeadingDims + 3],
        dimensionConstants[nLeadingDims + 4]);

    // Collapse back to original rank
    rewriter.replaceOpWithNewOp<PrimsCollapseOp>(
        op, op.getType(), partiallyCollapsed,
        dimensionConstants[nLeadingDims + 1],
        dimensionConstants[nLeadingDims + 2]);

    return success();
  }
};
} // namespace

// ReLU6(x) = min(max(0, x), 6) = min(Relu(x), 6)
static Value getRelu6Results(PatternRewriter &rewriter, Location loc,
                             Value input) {
  BaseTensorType inputType = input.getType().cast<BaseTensorType>();

  Value relu = rewriter.create<AtenReluOp>(loc, inputType, input);
  Value cst6 =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(6));
  Value sixTensor = createRank0Tensor(rewriter, loc, inputType, cst6);
  Value relu6Out =
      rewriter.create<AtenMinimumOp>(loc, inputType, relu, sixTensor);
  return relu6Out;
}

namespace {
class DecomposeAtenRelu6Op : public OpRewritePattern<AtenRelu6Op> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRelu6Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value relu6 = getRelu6Results(rewriter, loc, op.getSelf());
    rewriter.replaceOp(op, relu6);
    return success();
  }
};
} // namespace

// Hardswish(x) = x * Relu6(x+3)/6
namespace {
class DecomposeAtenHardswishOp : public OpRewritePattern<AtenHardswishOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardswishOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Type inputType = input.getType();

    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantThree = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(3));
    Value constantSix = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(6));
    Value inputPlusThree = rewriter.create<AtenAddScalarOp>(
        loc, inputType, input, constantThree, /*alpha=*/constantOne);
    Value relu6 = getRelu6Results(rewriter, loc, inputPlusThree);
    Value divTensor =
        rewriter.create<AtenDivScalarOp>(loc, inputType, relu6, constantSix);
    Value mulTensor =
        rewriter.create<AtenMulTensorOp>(loc, inputType, divTensor, input);

    rewriter.replaceOp(op, mulTensor);
    return success();
  }
};
} // namespace

// LeakyRelu = max(0,x) + negative_slope * min(0,x)
namespace {
class DecomposeAtenLeakyReluOp : public OpRewritePattern<AtenLeakyReluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLeakyReluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value negativeSlope = op.getNegativeSlope();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    Value constantZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value constantOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value zeroTensor = createRank0Tensor(rewriter, loc, resType, constantZero);
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, input);
    Value negativeOutput =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, input);
    Value scaledNegativeOutput = rewriter.create<AtenMulScalarOp>(
        loc, resType, negativeOutput, negativeSlope);
    Value leakyReluOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, scaledNegativeOutput, constantOne);

    rewriter.replaceOp(op, leakyReluOutput);
    return success();
  }
};
} // namespace

// LeakyReluBackward = max(0,grad) + negative_slope * min(0,x)
namespace {
class DecomposeAtenLeakyReluBackwardOp
    : public OpRewritePattern<AtenLeakyReluBackwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLeakyReluBackwardOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.getGradOutput();
    Value input = op.getSelf();
    Value negativeSlope = op.getNegativeSlope();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    bool selfIsResult = false;
    if (!matchPattern(op.getSelfIsResult(),
                      m_TorchConstantBool(&selfIsResult)) ||
        selfIsResult)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: self_is_result should be false");

    Value constantZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value constantOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value zeroTensor = createRank0Tensor(rewriter, loc, resType, constantZero);
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, gradOutput);
    Value negativeOutput =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, input);
    Value scaledNegativeOutput = rewriter.create<AtenMulScalarOp>(
        loc, resType, negativeOutput, negativeSlope);
    Value leakyReluBackwardOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, scaledNegativeOutput, constantOne);

    rewriter.replaceOp(op, leakyReluBackwardOutput);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenLerpScalarOp : public OpRewritePattern<AtenLerpScalarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLerpScalarOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    auto start = op.getSelf();
    auto inputType = start.getType().cast<BaseTensorType>();

    auto delta = rewriter.create<AtenSubTensorOp>(loc, inputType, op.getEnd(),
                                                  start, cstOne);

    auto weightedDelta =
        rewriter.create<AtenMulScalarOp>(loc, inputType, delta, op.getWeight());
    auto lerp = rewriter.create<AtenAddTensorOp>(loc, inputType, start,
                                                 weightedDelta, cstOne);
    rewriter.replaceOp(op, lerp);
    return success();
  }
};
} // namespace

// Elu = scale * max(0,x) + alpha * scale * (exp(min(0,x) * input_scale) - 1)
namespace {
class DecomposeAtenEluOp : public OpRewritePattern<AtenEluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value alpha = op.getAlpha();
    Value scale = op.getScale();
    Value inputScale = op.getInputScale();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    Value constantZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value constantOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value zeroTensor = createRank0Tensor(rewriter, loc, resType, constantZero);
    Value maxZeroX =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, input);
    Value positiveOutput =
        rewriter.create<AtenMulScalarOp>(loc, resType, maxZeroX, scale);
    Value minZeroX =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, input);
    Value scaledMinZeroX =
        rewriter.create<AtenMulScalarOp>(loc, resType, minZeroX, inputScale);
    Value expX = rewriter.create<AtenExpOp>(loc, resType, scaledMinZeroX);
    Value expXM1 = rewriter.create<AtenSubScalarOp>(loc, resType, expX,
                                                    constantOne, constantOne);
    Value scaledExpXM1 =
        rewriter.create<AtenMulScalarOp>(loc, resType, expXM1, scale);
    Value negativeOutput =
        rewriter.create<AtenMulScalarOp>(loc, resType, scaledExpXM1, alpha);

    Value eluOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, negativeOutput, constantOne);

    rewriter.replaceOp(op, eluOutput);
    return success();
  }
};
} // namespace

// Selu = scale * (max(0,x) + min(0,alpha * (exp(x) − 1)))
namespace {
class DecomposeAtenSeluOp : public OpRewritePattern<AtenSeluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSeluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    // Define λ and α
    double scale = 1.0507009873554804934193349852946;
    double alpha = 1.6732632423543772848170429916717;

    // Create constants for λ and α
    Value scaleVal = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(scale));
    Value alphaVal = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(alpha));

    // Create zero tensor for comparison
    Value constantZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value zeroTensor = createRank0Tensor(rewriter, loc, resType, constantZero);

    // Calculate positive and negative parts
    Value constantOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, input);
    Value minZeroX =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, input);
    Value expInput = rewriter.create<AtenExpOp>(loc, resType, minZeroX);
    Value expInputMinusOne = rewriter.create<AtenSubScalarOp>(
        loc, resType, expInput, constantOne, constantOne);
    Value negativeOutput = rewriter.create<AtenMulScalarOp>(
        loc, resType, expInputMinusOne, alphaVal);

    // Multiply the result by λ
    Value seluOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, negativeOutput, constantOne);
    seluOutput =
        rewriter.create<AtenMulScalarOp>(loc, resType, seluOutput, scaleVal);

    // Replace the original operation
    rewriter.replaceOp(op, seluOutput);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenTOp : public OpRewritePattern<AtenTOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getSelf();
    std::optional<unsigned> lhsRank = getTensorRank(lhs);
    auto loc = op.getLoc();

    if (!lhsRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    } else if (*lhsRank > 2) {
      std::string errorMessage =
          "t() expects a tensor with <=2 dimensions, but self is " +
          std::to_string(*lhsRank) + "D";
      return rewriter.notifyMatchFailure(op, errorMessage.c_str());
    } else if (*lhsRank < 2)
      rewriter.replaceOp(op, lhs);
    else {
      Value zero =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
      Value one =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
      rewriter.replaceOpWithNewOp<AtenTransposeIntOp>(op, op.getType(), lhs,
                                                      zero, one);
    }
    return success();
  }
};
} // namespace

// Decompose `aten.stack` into `aten.unsqueeze` and `aten.cat`.
namespace {
class DecomposeAtenStackOp : public OpRewritePattern<AtenStackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenStackOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> tensors;
    if (!getListConstructElements(op.getTensors(), tensors)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the tensor list is not from list construct");
    }
    // Ensure all tensors have known sizes
    for (Value tensor : tensors) {
      BaseTensorType tensorType = tensor.getType().cast<BaseTensorType>();
      if (!tensorType.hasSizes()) {
        return rewriter.notifyMatchFailure(
            op, "unimplemented: one tensor does not have known sizes");
      }
    }

    SmallVector<Value> unsqueezedTensors;
    for (Value tensor : tensors) {
      auto unsqueezedInfo = unsqueezeTensor(rewriter, op, tensor, op.getDim());
      if (failed(unsqueezedInfo)) {
        return rewriter.notifyMatchFailure(
            op, "cannot generate unsqueeze tensor op");
      }
      unsqueezedTensors.push_back(*unsqueezedInfo);
    }

    Type listElemType =
        op.getType().cast<BaseTensorType>().getWithSizesAndDtype(
            /*optionalSizes=*/std::nullopt, /*optionalDtype=*/nullptr);
    Type listType = Torch::ListType::get(listElemType);
    Value unsqueezedTensorList = rewriter.create<PrimListConstructOp>(
        op.getLoc(), listType, unsqueezedTensors);
    rewriter.replaceOpWithNewOp<AtenCatOp>(op, op.getType(),
                                           unsqueezedTensorList, op.getDim());
    return success();
  }
};
} // namespace

// Decompose aten.roll into aten.slice and aten.cat ops.
// https://pytorch.org/docs/stable/generated/torch.roll.html
namespace {
class DecomposeAtenRollOp : public OpRewritePattern<AtenRollOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRollOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> shifts;
    if (!getListConstructElements(op.getShifts(), shifts))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: shifts not list of Scalar");
    SmallVector<Value> dims;
    if (!getListConstructElements(op.getDims(), dims))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dims not list of Scalar");

    if (shifts.size() != dims.size())
      return op.emitError("list sizes of shifts and dims are not the same");

    auto loc = op.getLoc();
    Value constNone = rewriter.create<ConstantNoneOp>(loc);
    Value constZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value constOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    auto self = op.getSelf();
    auto selfTy = self.getType().cast<BaseTensorType>();
    // roll(input, shift, dim) = cat({
    //   slice(input, dim, -shift, none),
    //   slice(input, dim, 0, -shift)}, dim)
    auto imitateRoll = [&](Value input, Value shift, Value dim,
                           int64_t cstDim) {
      Value negShift = rewriter.create<AtenNegIntOp>(loc, shift);
      ArrayRef<int64_t> inputShape = selfTy.getSizes();
      SmallVector<int64_t> sizes;
      sizes.append(inputShape.begin(), inputShape.end());
      sizes[cstDim] = kUnknownSize;
      Type sliceTy = selfTy.getWithSizesAndDtype(llvm::ArrayRef(sizes),
                                                 selfTy.getOptionalDtype());
      Value slice0 = rewriter.create<AtenSliceTensorOp>(
          loc, sliceTy, input, dim, negShift, constNone, constOne);
      Value slice1 = rewriter.create<AtenSliceTensorOp>(
          loc, sliceTy, input, dim, constZero, negShift, constOne);

      Type listType = Torch::ListType::get(sliceTy);
      Value slices = rewriter.create<PrimListConstructOp>(
          loc, listType, llvm::ArrayRef<Value>{slice0, slice1});
      return rewriter.create<AtenCatOp>(loc, self.getType(), slices, dim);
    };
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
    unsigned rank = *maybeRank;
    Value output = self;
    auto nShifts = shifts.size();
    for (size_t k = 0; k < nShifts; ++k) {
      auto dim = dims[k];
      int64_t cstDim = -1;
      if (!matchPattern(dim, m_TorchConstantInt(&cstDim)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: dim must be constant");

      cstDim = toPositiveDim(cstDim, rank);
      output = imitateRoll(output, shifts[k], dim, cstDim);
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

// Decompose aten.repeat into aten.expand and aten.view ops.
//
// Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
//
// For shape [S1, S2, S3] and repeats [M0, M1, M2, M3]
//     MS0 = M0; MS1 = M1 * S1; MS2 = M2 * S2; MS3 = M3 * S3
//
// def aten_repeat(self, repeats):
//     sizes = self.size()
//     unsqueezed_sizes = []
//     expanded_sizes = []
//     reshape_sizes = []
//     leading_rank = repeats.size() - sizes.size()
//     for r in range(leading_rank):
//         unsqueezed_sizes.append(1)
//         expanded_sizes.append(repeats[r])
//         reshaped_sizes.append(repeats[r])
//
//     for s, m in zip(sizes, repeats[leading_rank:]):
//         unsqueezed_sizes += [1, s]
//         expanded_sizes += [m, s]
//         reshaped_sizes += [m * s]
//     return
//     self.view(unsqueezed_sizes).expand(expanded_sizes).view(reshaped_sizes)
//
namespace {
class DecomposeAtenRepeatOp : public OpRewritePattern<AtenRepeatOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRepeatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
    unsigned rank = *maybeRank;

    SmallVector<Value> repeats;
    if (!getListConstructElements(op.getRepeats(), repeats))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: repeats not list of Scalar");

    if (rank > repeats.size()) {
      return rewriter.notifyMatchFailure(
          op, "repeats are not matched with self's rank");
    }

    auto insertDimSizes = [](SmallVector<Value> &dimSizes,
                             SmallVector<int64_t> &shape,
                             const ArrayRef<Value> &vals) {
      dimSizes.insert(dimSizes.end(), vals.begin(), vals.end());
      std::transform(vals.begin(), vals.end(), std::back_inserter(shape),
                     [&](Value val) -> int64_t {
                       int64_t cst_val;
                       if (matchPattern(val, m_TorchConstantInt(&cst_val))) {
                         return cst_val;
                       } else {
                         return kUnknownSize;
                       }
                     });
    };

    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));

    SmallVector<Value> unsqueezedSizes, expandedSizes, reshapedSizes;
    SmallVector<int64_t> unsqueezedIntSizes, expandedIntSizes;
    assert(repeats.size() >= rank && "leadingRank should greater than 0");
    auto leadingRank = repeats.size() - rank;
    for (size_t i = 0; i < leadingRank; ++i) {
      insertDimSizes(unsqueezedSizes, unsqueezedIntSizes, ArrayRef<Value>{one});
      insertDimSizes(expandedSizes, expandedIntSizes,
                     ArrayRef<Value>{repeats[i]});
      reshapedSizes.push_back(repeats[i]);
    }

    auto selfType = self.getType().dyn_cast<BaseTensorType>();
    auto selfShape = selfType.getSizes();
    for (unsigned i = 0; i < rank; i++) {
      auto scale = repeats[i + leadingRank];
      Value dimSize;
      if (selfShape[i] == kUnknownSize) {
        Value dim = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(i));
        dimSize = rewriter.create<AtenSizeIntOp>(loc, self, dim);
      } else {
        dimSize = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(selfShape[i]));
      }

      insertDimSizes(unsqueezedSizes, unsqueezedIntSizes,
                     ArrayRef<Value>{one, dimSize});
      insertDimSizes(expandedSizes, expandedIntSizes,
                     ArrayRef<Value>{scale, dimSize});

      Value scaledSize = rewriter.create<AtenMulIntOp>(loc, dimSize, scale);
      reshapedSizes.push_back(scaledSize);
    }

    Type dtype = self.getType().cast<ValueTensorType>().getOptionalDtype();
    Type unsqueezedType = ValueTensorType::get(
        context, llvm::ArrayRef(unsqueezedIntSizes), dtype);
    Type expandedType =
        ValueTensorType::get(context, llvm::ArrayRef(expandedIntSizes), dtype);

    auto listType = Torch::ListType::get(Torch::IntType::get(op.getContext()));
    Value unsqueezedDims =
        rewriter.create<PrimListConstructOp>(loc, listType, unsqueezedSizes);
    Value expandedDims =
        rewriter.create<PrimListConstructOp>(loc, listType, expandedSizes);
    Value reshapedDims =
        rewriter.create<PrimListConstructOp>(loc, listType, reshapedSizes);
    auto reshaped = rewriter.create<AtenViewOp>(loc, unsqueezedType,
                                                op.getSelf(), unsqueezedDims);
    auto expanded = rewriter.create<AtenBroadcastToOp>(loc, expandedType,
                                                       reshaped, expandedDims);

    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), expanded,
                                            reshapedDims);
    return success();
  }
};
} // namespace

// Decompose aten.flatten.using_ints into aten.view op.
namespace {
class DecomposeAtenFlattenUsingIntsOp
    : public OpRewritePattern<AtenFlattenUsingIntsOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFlattenUsingIntsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "unimplemented: unranked tensor");
    unsigned rank = *maybeRank;

    int64_t start, end;
    if (!matchPattern(op.getStartDim(), m_TorchConstantInt(&start)) ||
        !matchPattern(op.getEndDim(), m_TorchConstantInt(&end))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires start and end dims to be constants");
    }

    SmallVector<Value, 4> newSizes;
    if (rank == 0) {
      Value one =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
      newSizes.push_back(one);
    } else {
      start = toPositiveDim(start, rank);
      end = toPositiveDim(end, rank);

      if (start > end) {
        return rewriter.notifyMatchFailure(
            op, "expected end dim larger than start dim");
      }

      newSizes.reserve(rank - end + start);
      for (int64_t k = 0; k < start; ++k) {
        Value dim =
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(k));
        newSizes.push_back(
            rewriter.create<AtenSizeIntOp>(loc, self, /*dim=*/dim));
      }
      Value flattenDimSize =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(-1));
      newSizes.push_back(flattenDimSize);
      for (int64_t k = end + 1; k < rank; ++k) {
        Value dim =
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(k));
        newSizes.push_back(
            rewriter.create<AtenSizeIntOp>(loc, self, /*dim=*/dim));
      }
    }
    Value newSizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), newSizes);
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), op.getSelf(),
                                            newSizeList);
    return success();
  }
};
} // namespace

// Decompose aten.unflatten.int into aten.view op.
namespace {
class DecomposeAtenUnflattenIntOp
    : public OpRewritePattern<AtenUnflattenIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenUnflattenIntOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    BaseTensorType outputTensorType = op.getType().cast<BaseTensorType>();
    if (!outputTensorType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: output must have known sizes");

    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "unimplemented: unranked tensor");
    unsigned inputRank = *maybeRank;
    auto inputTensorType = self.getType().cast<Torch::ValueTensorType>();
    if (!inputTensorType || !inputTensorType.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "Expected input type having sizes");
    }
    ArrayRef<int64_t> inputShape = inputTensorType.getSizes();

    SmallVector<int64_t> sizesInts;
    if (!matchPattern(op.getSizes(), m_TorchListOfConstantInts(sizesInts)))
      return rewriter.notifyMatchFailure(
          op, "sizes must be a list of constant ints");

    bool inferred = false;
    if (llvm::count(sizesInts, -1) > 1)
      return rewriter.notifyMatchFailure(
          op, "only one of sizes' elements can be -1");

    int64_t dimInt;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires dim to be constants");

    dimInt = toPositiveDim(dimInt, inputRank);
    if (!isValidDim(dimInt, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is not a valid dim");

    SmallVector<Value> sizesTorchInt;
    if (!getListConstructElements(op.getSizes(), sizesTorchInt))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: sizes not list of Scalar");

    // Create new sizes based on the unflattened dimension.
    SmallVector<Value> newSizes;
    for (int64_t i = 0; i < inputRank; ++i) {
      Value dimValue =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i));
      Value dimSize =
          rewriter.create<AtenSizeIntOp>(loc, self, /*dim=*/dimValue);
      if (i == dimInt) {
        int64_t inferredSizeInt = inputShape[i];
        int64_t inferredDim;
        for (unsigned j = 0; j < sizesInts.size(); ++j) {
          if (sizesInts[j] == -1) {
            inferred = true;
            inferredDim = j;
          } else {
            Value sizeValue = rewriter.create<ConstantIntOp>(
                loc, rewriter.getI64IntegerAttr(sizesInts[j]));
            newSizes.push_back(sizeValue);
            inferredSizeInt = inferredSizeInt / sizesInts[j];
          }
        }
        if (inferred) {
          Value inferredSize = rewriter.create<ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(inferredSizeInt));
          newSizes.insert(newSizes.begin() + inferredDim + i, inferredSize);
        }
      } else {
        newSizes.push_back(dimSize);
      }
    }

    // Create the AtenViewOp to replace the original op.
    Value newSizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), newSizes);
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), op.getSelf(),
                                            newSizeList);
    return success();
  }
};
} // namespace

// Decompose aten.expand into aten.broadcast_to op.
namespace {
class DecomposeAtenExpandOp : public OpRewritePattern<AtenExpandOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenExpandOp op,
                                PatternRewriter &rewriter) const override {
    bool implicit = false;
    if (!matchPattern(op.getImplicit(), m_TorchConstantBool(&implicit)) ||
        implicit) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires implicit to be false");
    }
    rewriter.replaceOpWithNewOp<AtenBroadcastToOp>(op, op.getType(),
                                                   op.getSelf(), op.getSize());
    return success();
  }
};
} // namespace

// Decompose aten.where.Scalar into aten.where.self op.
namespace {
class DecomposeAtenWhereScalarOp : public OpRewritePattern<AtenWhereScalarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereScalarOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value selfTensor = createRank0Tensor(rewriter, loc, resType, op.getSelf());
    Value otherTensor =
        createRank0Tensor(rewriter, loc, resType, op.getOther());
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, op.getCondition(),
                                                 selfTensor, otherTensor);
    return success();
  }
};
} // namespace

// Decompose aten.where.ScalarOther into aten.where.self op.
namespace {
class DecomposeAtenWhereScalarOtherOp
    : public OpRewritePattern<AtenWhereScalarOtherOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereScalarOtherOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value otherTensor =
        createRank0Tensor(rewriter, loc, resType, op.getOther());
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, op.getCondition(),
                                                 op.getSelf(), otherTensor);
    return success();
  }
};
} // namespace

// Decompose aten.where.ScalarSelf into aten.where.self op.
namespace {
class DecomposeAtenWhereScalarSelfOp
    : public OpRewritePattern<AtenWhereScalarSelfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereScalarSelfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value selfTensor = createRank0Tensor(rewriter, loc, resType, op.getSelf());
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, op.getCondition(),
                                                 selfTensor, op.getOther());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNanToNumOp : public OpRewritePattern<AtenNanToNumOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNanToNumOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    mlir::FloatType f64Type = rewriter.getF64Type();
    Value nan = op.getNan();
    Value posinf = op.getPosinf();
    Value neginf = op.getNeginf();
    auto baseType =
        ValueTensorType::getWithLeastStaticInformation(op.getContext());
    if (dyn_cast_or_null<ConstantNoneOp>(nan.getDefiningOp()))
      nan = rewriter.create<ConstantFloatOp>(
          loc, rewriter.getFloatAttr(
                   f64Type, APFloat::getZero(f64Type.getFloatSemantics())));
    if (dyn_cast_or_null<ConstantNoneOp>(posinf.getDefiningOp()))
      posinf = rewriter.create<ConstantFloatOp>(
          loc, rewriter.getFloatAttr(
                   f64Type, APFloat::getInf(f64Type.getFloatSemantics())));
    if (dyn_cast_or_null<ConstantNoneOp>(neginf.getDefiningOp()))
      neginf = rewriter.create<ConstantFloatOp>(
          loc,
          rewriter.getFloatAttr(
              f64Type, APFloat::getInf(f64Type.getFloatSemantics(), true)));
    Value isNan =
        rewriter.create<Torch::AtenIsnanOp>(loc, baseType, op.getSelf());
    Value where = rewriter.create<Torch::AtenWhereScalarSelfOp>(
        loc, baseType, isNan, nan, op.getSelf());
    Value isposinf =
        rewriter.create<Torch::AtenIsposinfOp>(loc, baseType, where);
    where = rewriter.create<Torch::AtenWhereScalarSelfOp>(
        loc, baseType, isposinf, posinf, where);
    Value isneginf =
        rewriter.create<Torch::AtenIsneginfOp>(loc, baseType, where);
    rewriter.replaceOpWithNewOp<Torch::AtenWhereScalarSelfOp>(
        op, op.getType(), isneginf, neginf, where);
    return success();
  }
};
} // namespace

// Decompose aten.masked_fill.Scalar into aten.where.self op.
namespace {
class DecomposeAtenMaskedFillScalarOp
    : public OpRewritePattern<AtenMaskedFillScalarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMaskedFillScalarOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value mask = op.getMask();
    Value value = createRank0Tensor(rewriter, loc, resType, op.getValue());
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, mask, value,
                                                 op.getSelf());
    return success();
  }
};
} // namespace

// Decompose aten._convolution-like to aten.convolution
namespace {
template <typename ConvolutionLikeOp>
class DecomposeAten_ConvolutionLikeOp
    : public OpRewritePattern<ConvolutionLikeOp> {
public:
  using OpRewritePattern<ConvolutionLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvolutionLikeOp op,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), op.getTransposed(),
        op.getOutputPadding(), op.getGroups());

    return success();
  }
};
} // namespace

namespace {

static LogicalResult createTorchTransposeOpForConvTbc(PatternRewriter &rewriter,
                                                      Location loc, Value input,
                                                      int64_t dimA,
                                                      int64_t dimB,
                                                      Value &transposed) {
  Type transposedType;
  if (failed(getTransposedType(input.getType().cast<Torch::BaseTensorType>(),
                               dimA, dimB, transposedType)))
    return failure();
  Value cstDimA = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(dimA));
  Value cstDimB = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(dimB));
  transposed = rewriter.create<Torch::AtenTransposeIntOp>(
      loc, transposedType, input, cstDimA, cstDimB);
  return success();
}

class DecomposeAtenConvTbcOp : public OpRewritePattern<AtenConvTbcOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvTbcOp op,
                                PatternRewriter &rewriter) const override {
    Value emptyList = rewriter.create<PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(Torch::IntType::get(op.getContext())),
        SmallVector<Value>());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    Value oneList = rewriter.create<PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(Torch::IntType::get(op.getContext())),
        SmallVector<Value>{rewriter.create<Torch::ConstantIntOp>(
            op.getLoc(), rewriter.getI64IntegerAttr(1))});
    Value padding = rewriter.create<PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(Torch::IntType::get(op.getContext())),
        SmallVector<Value>{op.getPad()});
    Value groups = rewriter.create<Torch::ConstantIntOp>(
        op.getLoc(), rewriter.getI64IntegerAttr(1));

    // convtbc has WNC layout for input and output
    // and WCF layout for weight
    // whereas Convolution is going to use Conv1DNcwFcwOp for 1d
    // which means we need the inputs in NCW and the weight in FCW
    Value selfWnc = op.getSelf();
    Value selfNwc;
    Value selfNcw;
    if (failed(createTorchTransposeOpForConvTbc(rewriter, op.getLoc(), selfWnc,
                                                0, 1, selfNwc)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to transpose input to Nwc");
    if (failed(createTorchTransposeOpForConvTbc(rewriter, op.getLoc(), selfNwc,
                                                1, 2, selfNcw)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to transpose input to Ncw");

    Value weightWcf = op.getWeight();
    Value weightFcw;
    if (failed(createTorchTransposeOpForConvTbc(rewriter, op.getLoc(),
                                                weightWcf, 0, 2, weightFcw)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to transpose weight to Fcw");

    Value outputNcw = rewriter.create<AtenConvolutionOp>(
        op.getLoc(), op->getResultTypes(), selfNcw, weightFcw, op.getBias(),
        /*stride*/ oneList,
        /*padding*/ padding, /*dilation*/ oneList,
        /*transpose*/ cstFalse, /*output_padding*/ emptyList, groups);

    // convert output from Ncw to Wnc
    Value outputNwc;
    Value outputWnc;
    if (failed(createTorchTransposeOpForConvTbc(rewriter, op.getLoc(),
                                                outputNcw, 1, 2, outputNwc)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to transpose output to Nwc");
    if (failed(createTorchTransposeOpForConvTbc(rewriter, op.getLoc(),
                                                outputNwc, 0, 1, outputWnc)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to transpose output to Wnc");
    rewriter.replaceOp(op, outputWnc);

    return success();
  }
};
} // namespace

// Decompose aten.conv1d to aten.convolution
namespace {
class DecomposeAtenConv1dOp : public OpRewritePattern<AtenConv1dOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConv1dOp op,
                                PatternRewriter &rewriter) const override {

    Value emptyList = rewriter.create<PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(Torch::IntType::get(op.getContext())),
        SmallVector<Value>());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), cstFalse, emptyList,
        op.getGroups());

    return success();
  }
};
} // namespace

// Decompose aten.conv2d to aten.convolution
namespace {
class DecomposeAtenConv2dOp : public OpRewritePattern<AtenConv2dOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConv2dOp op,
                                PatternRewriter &rewriter) const override {

    Value emptyList = rewriter.create<PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(Torch::IntType::get(op.getContext())),
        SmallVector<Value>());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), cstFalse, emptyList,
        op.getGroups());

    return success();
  }
};
} // namespace

// Decompose aten.conv3d to aten.convolution
namespace {
class DecomposeAtenConv3dOp : public OpRewritePattern<AtenConv3dOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConv3dOp op,
                                PatternRewriter &rewriter) const override {

    Value emptyList = rewriter.create<PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(Torch::IntType::get(op.getContext())),
        SmallVector<Value>());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), cstFalse, emptyList,
        op.getGroups());

    return success();
  }
};
} // namespace

// Decompose aten.conv_transpose2d to aten.convolution
namespace {
class DecomposeAtenConvTranspose2dOp
    : public OpRewritePattern<AtenConvTranspose2dInputOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvTranspose2dInputOp op,
                                PatternRewriter &rewriter) const override {

    Value cstTrue = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), true);
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(),
        /*transposed=*/cstTrue, op.getOutputPadding(), op.getGroups());
    return success();
  }
};
} // namespace

// The convolution backward op is decomposed as follows:
// inputH, inputW = input.shape[2:]
// output_padding_ = [
//     inputH
//     - 1
//     + 2 * padding_[0]
//     - dilation_[0] * (weight.shape[2] - 1)
//     - (grad_output.shape[2] - 1) * stride_[0],
//     inputW
//     - 1
//     + 2 * padding_[1]
//     - dilation_[1] * (weight.shape[3] - 1)
//     - (grad_output.shape[3] - 1) * stride_[1],
// ]
//
// decomp_grad_input = torch.nn.functional.conv_transpose2d(
//     grad_output,
//     weight,
//     None,
//     stride_,
//     padding_,
//     output_padding_,
//     groups_,
//     dilation_,
// )
//
// input_transposed = torch.ops.aten.transpose(input, 0, 1)
// grad_output_transposed = grad_output.view(
//     grad_output.shape[0] * grad_output.shape[1], 1, *grad_output.shape[2:]
// )
// decomp_grad_weight = torch.ops.aten.convolution(
//     input_transposed,
//     grad_output_transposed,
//     bias=None,
//     stride=dilation_,
//     padding=padding_,
//     dilation=stride_,
//     transposed=False,
//     output_padding=[0, 0],
//     groups=input.shape[0],
// )
// decomp_grad_weight = torch.narrow(decomp_grad_weight, 2, 0, weight.shape[2])
// decomp_grad_weight = torch.narrow(decomp_grad_weight, 3, 0, weight.shape[3])
// decomp_grad_weight = decomp_grad_weight.view(
//     input_transposed.shape[0],
//     input_transposed.shape[1],
//     grad_output.shape[1],
//     *decomp_grad_weight.shape[2:]
// )
// decomp_grad_weight = decomp_grad_weight.movedim(0, 2)
// decomp_grad_weight = decomp_grad_weight.sum(dim=0)
//
// decomp_grad_bias = torch.sum(grad_output, dim=[0, 2, 3])
namespace {
class DecomposeAtenConvolutionBackwardOp
    : public OpRewritePattern<AtenConvolutionBackwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvolutionBackwardOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value gradOutput = op.getGradOutput();
    std::optional<unsigned> maybeGradRank = getTensorRank(gradOutput);
    if (!maybeGradRank) {
      return rewriter.notifyMatchFailure(op,
                                         "expected grad output to have a rank");
    }
    unsigned gradRank = *maybeGradRank;
    if (gradRank != 4)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only 2D convolutions supported.");

    Value cstZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value cstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value cstTwo = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(2));
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
        loc, rewriter.getBoolAttr(false));

    SmallVector<Value> padding, dilation, stride;
    SmallVector<int64_t, 2> paddingInt, dilationInt, strideInt,
        outputPaddingInt;

    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingInt)))
      return rewriter.notifyMatchFailure(
          op, "padding must be a list of constant ints");

    if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(strideInt)))
      return rewriter.notifyMatchFailure(
          op, "stride must be a list of constant ints");

    if (!matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilationInt)))
      return rewriter.notifyMatchFailure(
          op, "dilation must be a list of constant ints");
    if (!llvm::all_of(dilationInt,
                      [](int64_t dilationVal) { return dilationVal == 1; }))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only dilations of 1 supported.");

    if (!matchPattern(op.getOutputPadding(),
                      m_TorchListOfConstantInts(outputPaddingInt)))
      return rewriter.notifyMatchFailure(
          op, "output padding must be a list of constant ints");
    if (!llvm::all_of(outputPaddingInt,
                      [](int64_t outPad) { return outPad == 0; }))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only output padding of 0 supported.");

    SmallVector<bool> outMask;
    if (!matchPattern(op.getOutputMask(), m_TorchListOfConstantBools(outMask)))
      return rewriter.notifyMatchFailure(
          op, "only constant bool output_mask is supported.");
    for (unsigned i = 0; i < outMask.size(); i++) {
      if (outMask[i] == false) {
        Value result = op->getResults()[i];
        if (!result.getUsers().empty())
          return rewriter.notifyMatchFailure(
              op, "unimplemented: false value supported for output_mask only "
                  "when the result tensor corresponding to that has no users.");
      }
    }

    bool transposed;
    if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
      return rewriter.notifyMatchFailure(
          op, "transposed arg should be a constant bool.");
    if (transposed)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: transposed convolutions are not supported.");

    getListConstructElements(op.getPadding(), padding);
    getListConstructElements(op.getStride(), stride);
    getListConstructElements(op.getDilation(), dilation);

    // Computing Grad Input.
    // Calculate output padding for first convolution.
    // output_padding_ = [
    //     inputH - 1 + (2 * padding_[0]) - (dilation_[0] * (weight.size()[2]
    //     - 1)) - ((grad_out.size()[2] - 1) * stride_[0]), inputW - 1 + (2 *
    //     padding_[1]) - (dilation_[1] * (weight.size()[3] - 1)) -
    //     ((grad_out.size()[3] - 1) * stride_[1]),
    // ]
    SmallVector<Value> outputPaddingValues;
    for (unsigned i = 2; i < gradRank; i++) {
      Value dim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i));
      Value inputVecDim =
          rewriter.create<Torch::AtenSizeIntOp>(loc, input, dim);
      Value gradOutDim =
          rewriter.create<Torch::AtenSizeIntOp>(loc, gradOutput, dim);
      Value weightDim = rewriter.create<Torch::AtenSizeIntOp>(loc, weight, dim);
      Value inputVecDimMinusOne =
          rewriter.create<Torch::AtenSubIntOp>(loc, inputVecDim, cstOne);
      Value gradOutDimMinusOne =
          rewriter.create<Torch::AtenSubIntOp>(loc, gradOutDim, cstOne);
      Value weightDimMinusOne =
          rewriter.create<Torch::AtenSubIntOp>(loc, weightDim, cstOne);
      Value twoTimesPadding =
          rewriter.create<Torch::AtenMulIntOp>(loc, padding[i - 2], cstTwo);
      Value tmpA = rewriter.create<Torch::AtenMulIntOp>(loc, weightDimMinusOne,
                                                        dilation[i - 2]);
      Value tmpB = rewriter.create<Torch::AtenMulIntOp>(loc, gradOutDimMinusOne,
                                                        stride[i - 2]);
      Value outputPaddingVal = rewriter.create<AtenAddIntOp>(
          loc, inputVecDimMinusOne, twoTimesPadding);
      outputPaddingVal =
          rewriter.create<AtenSubIntOp>(loc, outputPaddingVal, tmpA);
      outputPaddingVal =
          rewriter.create<AtenSubIntOp>(loc, outputPaddingVal, tmpB);
      outputPaddingValues.push_back(outputPaddingVal);
    }
    Value outputPaddingForGradInput =
        rewriter.create<Torch::PrimListConstructOp>(
            loc, ListType::get(IntType::get(context)), outputPaddingValues);

    Value gradInput = rewriter.create<Torch::AtenConvTranspose2dInputOp>(
        loc, op.getResultTypes()[0], gradOutput, weight, cstNone,
        op.getStride(), op.getPadding(), outputPaddingForGradInput,
        op.getGroups(), op.getDilation());

    Type transposedType;
    if (failed(getTransposedType(input.getType().cast<BaseTensorType>(), 0, 1,
                                 transposedType)))
      return failure();
    Value inputTransposed = rewriter.create<Torch::AtenTransposeIntOp>(
        loc, transposedType, input, cstZero, cstOne);

    // For the cases where the stride is non-unit, we compute the `GradWeight`
    // through this implementation.
    Value gradWeight;
    if (!llvm::all_of(strideInt, [](int64_t stride) { return stride == 1; })) {
      // Computing Grad Weight.
      SmallVector<Value, 4> gradOutputSize;
      for (unsigned i = 0; i < gradRank; i++) {
        gradOutputSize.push_back(rewriter.create<Torch::AtenSizeIntOp>(
            loc, gradOutput,
            rewriter.create<Torch::ConstantIntOp>(
                loc, rewriter.getI64IntegerAttr(i))));
      }

      Value gradOutputViewDimZero = rewriter.create<Torch::AtenMulIntOp>(
          loc, gradOutputSize[0], gradOutputSize[1]);
      Value gradOutputViewShapeList =
          rewriter.create<Torch::PrimListConstructOp>(
              loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
              ValueRange{gradOutputViewDimZero, cstOne, gradOutputSize[2],
                         gradOutputSize[3]});

      BaseTensorType gradOutputTy = gradOutput.getType().cast<BaseTensorType>();
      if (!gradOutputTy.hasSizes())
        return failure();
      SmallVector<int64_t> gradOutputSizesInt(gradOutputTy.getSizes());
      SmallVector<int64_t> gradOutputViewSizesInt(gradOutputSizesInt);
      if (gradOutputViewSizesInt[0] != kUnknownSize &&
          gradOutputViewSizesInt[1] != kUnknownSize)
        gradOutputViewSizesInt[0] *= gradOutputViewSizesInt[1];
      else
        gradOutputViewSizesInt[0] = kUnknownSize;
      gradOutputViewSizesInt[1] = 1;
      BaseTensorType gradOutputTypeForView =
          gradOutputTy
              .getWithSizesAndDtype(llvm::ArrayRef(gradOutputViewSizesInt),
                                    gradOutputTy.getOptionalDtype())
              .cast<BaseTensorType>();
      Value gradOutputView = rewriter.create<Torch::AtenViewOp>(
          loc, gradOutputTypeForView, gradOutput, gradOutputViewShapeList);

      BaseTensorType inputTransposedTy =
          inputTransposed.getType().cast<BaseTensorType>();
      if (!inputTransposedTy.hasSizes())
        return failure();
      SmallVector<int64_t> inputTransposedSizesInt(
          inputTransposedTy.getSizes());
      SmallVector<int64_t> gradWeightSizesInt{inputTransposedSizesInt[0],
                                              gradOutputViewSizesInt[0]};
      for (unsigned i = 2; i < gradRank; i++) {
        if (inputTransposedSizesInt[i] != kUnknownSize &&
            gradOutputViewSizesInt[i] != kUnknownSize) {
          int64_t kernelSizeInt =
              strideInt[i - 2] * (gradOutputViewSizesInt[i] - 1) + 1;
          gradWeightSizesInt.push_back(
              ((inputTransposedSizesInt[i] + (paddingInt[i - 2] * 2) -
                kernelSizeInt) /
               dilationInt[i - 2]) +
              1);
        } else {
          gradWeightSizesInt.push_back(kUnknownSize);
        }
      }

      BaseTensorType gradWeightTy =
          inputTransposedTy
              .getWithSizesAndDtype(llvm::ArrayRef(gradWeightSizesInt),
                                    inputTransposedTy.getOptionalDtype())
              .cast<BaseTensorType>();

      Value numGroup = rewriter.create<AtenSizeIntOp>(loc, input, cstZero);
      gradWeight = rewriter.create<Torch::AtenConvolutionOp>(
          loc, gradWeightTy, inputTransposed, gradOutputView, cstNone,
          /*stride=*/op.getDilation(), op.getPadding(),
          /*dilation=*/op.getStride(), op.getTransposed(),
          op.getOutputPadding(), numGroup);

      BaseTensorType weightTy = weight.getType().cast<BaseTensorType>();
      if (!weightTy.hasSizes())
        return failure();
      SmallVector<int64_t> weightSizes(weightTy.getSizes());
      for (unsigned i = 0; i < gradWeightTy.getSizes().size() - 2; i++) {
        gradWeightSizesInt[i + 2] = weightSizes[i + 2];
        BaseTensorType gradWeightNarrowTy =
            gradWeightTy
                .getWithSizesAndDtype(llvm::ArrayRef(gradWeightSizesInt),
                                      gradWeightTy.getOptionalDtype())
                .cast<BaseTensorType>();

        Value dim = rewriter.create<ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(i + 2));
        Value length = rewriter.create<Torch::AtenSizeIntOp>(loc, weight, dim);
        gradWeight = rewriter.create<Torch::AtenNarrowOp>(
            loc, gradWeightNarrowTy, gradWeight, dim, /*start=*/cstZero,
            length);
      }

      SmallVector<int64_t, 5> gradWeightViewShapeInt{
          inputTransposedSizesInt[0], inputTransposedSizesInt[1]};
      gradWeightViewShapeInt.push_back(gradOutputSizesInt[1]);
      gradWeightViewShapeInt.insert(
          gradWeightViewShapeInt.end(),
          {gradWeightSizesInt[2], gradWeightSizesInt[3]});

      SmallVector<Value> gradWeightViewShapeValue;
      for (unsigned i = 0; i < gradWeightViewShapeInt.size(); i++) {
        gradWeightViewShapeValue.push_back(
            rewriter.create<Torch::ConstantIntOp>(
                loc, rewriter.getI64IntegerAttr(gradWeightViewShapeInt[i])));
      }

      Value gradWeightViewShapeList =
          rewriter.create<Torch::PrimListConstructOp>(
              loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
              gradWeightViewShapeValue);

      BaseTensorType gradWeightTypeForView =
          gradWeightTy
              .getWithSizesAndDtype(llvm::ArrayRef(gradWeightViewShapeInt),
                                    gradWeightTy.getOptionalDtype())
              .cast<BaseTensorType>();
      gradWeight = rewriter.create<Torch::AtenViewOp>(
          loc, gradWeightTypeForView, gradWeight, gradWeightViewShapeList);

      gradWeightTy = gradWeight.getType().cast<BaseTensorType>();
      SmallVector<int64_t, 5> gradWeightDimsOrder =
          computeDimsOrderForMoveDim(0, 2, gradWeightViewShapeInt.size());
      SmallVector<int64_t, 5> gradWeightMoveDimShape;
      for (unsigned i = 0; i < gradWeightDimsOrder.size(); i++) {
        gradWeightMoveDimShape.push_back(
            gradWeightViewShapeInt[gradWeightDimsOrder[i]]);
      }
      BaseTensorType gradWeightTypeForMoveDim =
          gradWeightTy
              .getWithSizesAndDtype(llvm::ArrayRef(gradWeightMoveDimShape),
                                    gradWeightTy.getOptionalDtype())
              .cast<BaseTensorType>();

      gradWeight = rewriter.create<AtenMovedimIntOp>(
          loc, gradWeightTypeForMoveDim, gradWeight, /*source=*/cstZero,
          /*destination=*/cstTwo);

      Value gradIntList = rewriter.create<Torch::PrimListConstructOp>(
          loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
          llvm::ArrayRef{cstZero});
      gradWeight = rewriter.create<Torch::AtenSumDimIntListOp>(
          loc, op.getResultTypes()[1], /*self=*/gradWeight, /*dim=*/gradIntList,
          /*keepdim=*/cstFalse,
          /*dtype=*/cstNone);
    } else {
      if (failed(getTransposedType(gradOutput.getType().cast<BaseTensorType>(),
                                   0, 1, transposedType)))
        return failure();
      Value gradOutputTransposed = rewriter.create<Torch::AtenTransposeIntOp>(
          loc, transposedType, gradOutput, cstZero, cstOne);
      // Convolve input with grad_output.
      if (failed(
              getTransposedType(op.getResultTypes()[1].cast<BaseTensorType>(),
                                0, 1, transposedType)))
        return failure();
      gradWeight = rewriter.create<Torch::AtenConvolutionOp>(
          loc, transposedType, inputTransposed, gradOutputTransposed, cstNone,
          op.getStride(), op.getPadding(), op.getDilation(), op.getTransposed(),
          op.getOutputPadding(), op.getGroups());
      gradWeight = rewriter.create<Torch::AtenTransposeIntOp>(
          loc, op.getResultTypes()[1], gradWeight, cstZero, cstOne);
    }

    // Computing Grad Bias.
    SmallVector<Value> dimIntList{cstZero};
    for (unsigned i = 2; i < gradRank; i++)
      dimIntList.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    Value gradIntList = rewriter.create<Torch::PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        dimIntList);

    // Sum grad_output along dim 1.
    Value gradBias = rewriter.create<Torch::AtenSumDimIntListOp>(
        loc, op.getResultTypes()[2], gradOutput, gradIntList, cstFalse,
        cstNone);

    rewriter.replaceOp(op, {gradInput, gradWeight, gradBias});
    return success();
  }
};
} // namespace

// Decompose aten.addmm into aten.mm and aten.add.Tensor op.
namespace {
class DecomposeAtenAddmmOp : public OpRewritePattern<AtenAddmmOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAddmmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value mat1 = op.getMat1();
    Value mat2 = op.getMat2();
    std::optional<unsigned> mat1Rank = getTensorRank(mat1);
    std::optional<unsigned> mat2Rank = getTensorRank(mat2);

    // The operands `mat1`, `mat2` to aten.addmm must be of rank 2.
    if (!mat1Rank || !mat2Rank || *mat1Rank != 2 || *mat2Rank != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected mat1, mat2 operands to aten.addmm to be rank 2");
    }

    // TODO: Handle integer type operands.
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: non-floating point dtype");
    }

    // matrix multiplication: matmul = mat1 @ mat2
    Value matmul = rewriter.create<AtenMmOp>(loc, op.getType(), mat1, mat2);
    // scaledInput = self * beta
    Value scaledInput = rewriter.create<AtenMulScalarOp>(loc, input.getType(),
                                                         input, op.getBeta());
    // result = scaledInput + alpha * matmul
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), scaledInput,
                                                 matmul, op.getAlpha());
    return success();
  }
};
} // namespace

// Decompose aten.mean into: sum(x)/div(numTensorElements).
namespace {
class DecomposeAtenMeanOp : public OpRewritePattern<AtenMeanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMeanOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value output = op.getResult();
    BaseTensorType outputTensorType = output.getType().cast<BaseTensorType>();
    Value sum =
        rewriter.create<AtenSumOp>(loc, outputTensorType, input, op.getDtype());
    Value numTensorElements = rewriter.create<AtenNumelOp>(loc, input);
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, outputTensorType, sum,
                                                 numTensorElements);
    return success();
  }
};
} // namespace

// productDimSize = product(size(dim) for dim in dims)
// aten.mean(x, dims) = aten.sum(x, dims) / productDimSize.
namespace {
class DecomposeAtenMeanDimOp : public OpRewritePattern<AtenMeanDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMeanDimOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned inputRank = *maybeInputRank;

    Value dimList = op.getDim();
    Value keepDim = op.getKeepdim();
    Value dtype = op.getDtype();
    Type outputType = op.getType();
    MLIRContext *context = op.getContext();

    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>() ||
        !isNoneOrFloatDtype(context, dtype)) {
      return rewriter.notifyMatchFailure(
          op, "only floating-point type is supported");
    }

    SmallVector<Value> dimListElements;
    if (!getListConstructElements(dimList, dimListElements) &&
        !dimList.getType().isa<Torch::NoneType>()) {
      return rewriter.notifyMatchFailure(
          op, "expected `dim` to be `None` or constructed from list construct");
    }

    // Compute sum along dimensions specified in `dimList`.
    Value sumAlongDims = rewriter.create<AtenSumDimIntListOp>(
        loc, outputType, input, dimList, keepDim, dtype);

    // `productDimSize` is product of sizes of dimensions to be reduced.
    Value productDimSize;
    // Case: Reduce along all dims.
    if (dimListElements.empty() && inputRank != 0) {
      productDimSize = rewriter.create<AtenNumelOp>(loc, input);
    } else {
      productDimSize = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(1));
      for (Value dim : dimListElements) {
        Value dimSize = rewriter.create<AtenSizeIntOp>(loc, input, dim);
        productDimSize =
            rewriter.create<AtenMulIntOp>(loc, productDimSize, dimSize);
      }
    }
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, outputType, sumAlongDims,
                                                 productDimSize);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenSquareOp : public OpRewritePattern<AtenSquareOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSquareOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, op.getType(), self, self);
    return success();
  }
};
} // namespace

// Silu(x) = sigmoid(x) * x
namespace {
class DecomposeAtenSiluOp : public OpRewritePattern<AtenSiluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSiluOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    Value sigmoid =
        rewriter.create<AtenSigmoidOp>(op.getLoc(), op.getType(), self);
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, op.getType(), sigmoid,
                                                 self);
    return success();
  }
};
} // namespace

// pDash = 1.0 - p
// boolMask = aten.rand_like(input) < pDash
// dropout(input, p, train=True) = (boolMask * input) / pDash
// dropout(input, p, train=False) = input
namespace {
class DecomposeAtenDropoutOp : public OpRewritePattern<AtenDropoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenDropoutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value prob = op.getP();
    bool train = false;
    if (!matchPattern(op.getTrain(), m_TorchConstantBool(&train)))
      return rewriter.notifyMatchFailure(op,
                                         "train must be a boolean constant");
    if (!train) {
      rewriter.replaceOp(op, input);
      return success();
    }
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "only support floating type input for training mode");
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value floatOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value oneMinusP = rewriter.create<AtenSubFloatOp>(loc, floatOne, prob);
    Value boolMask = rewriter.create<ValsemVariantAtenBernoulliFloatOp>(
        loc, inputType, input, oneMinusP, /*generator=*/noneVal);
    Value maskedInput =
        rewriter.create<AtenMulTensorOp>(loc, inputType, boolMask, input);
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, op.getType(), maskedInput,
                                                 oneMinusP);
    return success();
  }
};

class DeomposeAtenNativeDropoutOp
    : public OpRewritePattern<AtenNativeDropoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNativeDropoutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();
    Value input = op.getInput();
    Value prob = op.getP();
    bool train = false;
    if (!op.getTrain().getType().isa<Torch::NoneType>()) {
      if (!matchPattern(op.getTrain(), m_TorchConstantBool(&train))) {
        return rewriter.notifyMatchFailure(
            op, "train must be a boolean constant or none");
      }
    }
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    if (!train) {
      Value i1Type =
          getDtypeIntValueForType(rewriter, loc, IntegerType::get(context, 1));
      Value inputSize = rewriter.create<AtenSizeOp>(
          loc, Torch::ListType::get(Torch::IntType::get(context)), input);
      Value trueValue = rewriter.create<ConstantIntOp>(loc, 1);
      Value trueMask = rewriter.create<AtenFullOp>(
          loc, op->getResultTypes()[1], inputSize, trueValue, i1Type,
          /*layout=*/noneVal, /*device=*/noneVal, /*pin_memory=*/noneVal);
      rewriter.replaceOp(op, ArrayRef<Value>{input, trueMask});
      return success();
    }
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "only support floating type input for training mode");
    }
    Value floatOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value oneMinusP = rewriter.create<AtenSubFloatOp>(loc, floatOne, prob);
    Value boolMask = rewriter.create<ValsemVariantAtenBernoulliFloatOp>(
        loc, inputType, input, oneMinusP, /*generator=*/noneVal);
    Value maskedInput =
        rewriter.create<AtenMulTensorOp>(loc, inputType, boolMask, input);
    Value output = rewriter.create<AtenDivScalarOp>(
        loc, op->getResultTypes()[0], maskedInput, oneMinusP);
    rewriter.replaceOp(
        op, ArrayRef<Value>{
                output, convertTensorToDtype(rewriter, loc, boolMask,
                                             IntegerType::get(context, 1))});
    return success();
  }
};
} // namespace

// Decompose aten.var into: aten.var.dim op.
namespace {
class DecomposeAtenVarOp : public OpRewritePattern<AtenVarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    std::optional<unsigned> maybeInputRank = getTensorRank(self);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned inputRank = *maybeInputRank;
    BaseTensorType rank0FloatTensorTy = op.getType().cast<BaseTensorType>();
    if (!rank0FloatTensorTy.hasSizes() ||
        rank0FloatTensorTy.getSizes().size() != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected aten.var to have a rank 0 tensor type");
    }

    SmallVector<Value> dims;
    for (unsigned i = 0; i < inputRank; i++)
      dims.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())), dims);

    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<AtenVarDimOp>(op, rank0FloatTensorTy, self,
                                              dimList, op.getUnbiased(),
                                              /*keepdim=*/cstFalse);
    return success();
  }
};
} // namespace

// Decompose aten.std to sqrt(var(x))
namespace {
class DecomposeAtenStdOp : public OpRewritePattern<AtenStdOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenStdOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType inputTensorTy = self.getType().cast<BaseTensorType>();
    if (!inputTensorTy.hasDtype() ||
        !inputTensorTy.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "Only aten.std support floating type");
    }
    Value var = rewriter.create<AtenVarOp>(op->getLoc(), op.getType(),
                                           op.getSelf(), op.getUnbiased());
    rewriter.replaceOpWithNewOp<AtenSqrtOp>(op, op.getType(), var);
    return success();
  }
};
} // namespace

// Softplus(x, beta, threshold) =
//   x * beta > threshold ? x : log(1 + exp(x * beta)) / beta
namespace {
class DecomposeAtenSoftplusOp : public OpRewritePattern<AtenSoftplusOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftplusOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();

    Value inputTimesBeta =
        rewriter.create<AtenMulScalarOp>(loc, inputType, input, op.getBeta());

    // out = log1p(exp(input * beta)) / beta
    Value exp = rewriter.create<AtenExpOp>(loc, inputType, inputTimesBeta);
    Value log1p = rewriter.create<AtenLog1pOp>(loc, inputType, exp);
    Value out =
        rewriter.create<AtenDivScalarOp>(loc, inputType, log1p, op.getBeta());

    // Select where x * beta > threshold
    auto boolResType = inputType.getWithSizesAndDtype(inputType.getSizes(),
                                                      rewriter.getI1Type());
    Value condition = rewriter.create<AtenGtScalarOp>(
        loc, boolResType, inputTimesBeta, op.getThreshold());

    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, op.getType(), condition,
                                                 input, out);
    return success();
  }
};
} // namespace

// Decompose aten.std.dim to sqrt(var.dim(x))
namespace {
class DecomposeAtenStdDimOp : public OpRewritePattern<AtenStdDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenStdDimOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType inputTensorType = self.getType().cast<BaseTensorType>();
    if (!inputTensorType.hasDtype() ||
        !inputTensorType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "aten.std.dim expects input tensor of floating-point type");
    }

    Value varDim = rewriter.create<AtenVarDimOp>(
        op->getLoc(), op.getType(), self, op.getDim(), op.getUnbiased(),
        op.getKeepdim());
    rewriter.replaceOpWithNewOp<AtenSqrtOp>(op, op.getType(), varDim);
    return success();
  }
};
} // namespace

// Decompose aten.std.correction to sqrt(var.correction(x))
namespace {
class DecomposeAtenStdCorrectionOp
    : public OpRewritePattern<AtenStdCorrectionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenStdCorrectionOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType inputTensorType = self.getType().cast<BaseTensorType>();
    if (!inputTensorType.hasDtype() ||
        !inputTensorType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op,
          "aten.std.correction expects input tensor of floating-point type");
    }

    Value varCorrection = rewriter.create<AtenVarCorrectionOp>(
        op->getLoc(), op.getType(), self, op.getDim(), op.getCorrection(),
        op.getKeepdim());
    rewriter.replaceOpWithNewOp<AtenSqrtOp>(op, op.getType(), varCorrection);
    return success();
  }
};
} // namespace

// Hardsigmoid(x) = max(0, min(1, (x+3)/6))
namespace {
class DecomposeAtenHardsigmoidOp : public OpRewritePattern<AtenHardsigmoidOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardsigmoidOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    // outputTensor = (input + 3) / 6.
    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantThree = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(3));
    Value constantSix = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(6));
    Value inputPlusThree = rewriter.create<AtenAddScalarOp>(
        loc, inputType, input, constantThree, /*alpha=*/constantOne);
    Value outputTensor = rewriter.create<AtenDivScalarOp>(
        loc, inputType, inputPlusThree, constantSix);

    // result = max(0, min(1, (input+3)/6))
    Value constantZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value oneTensor = createRank0Tensor(rewriter, loc, inputType, constantOne);
    Value minResult =
        rewriter.create<AtenMinimumOp>(loc, inputType, oneTensor, outputTensor);
    Value zeroTensor =
        createRank0Tensor(rewriter, loc, inputType, constantZero);
    rewriter.replaceOpWithNewOp<AtenMaximumOp>(op, op.getType(), zeroTensor,
                                               minResult);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenHardtanhOp : public OpRewritePattern<AtenHardtanhOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardtanhOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    auto resType = op.getType().cast<BaseTensorType>();
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    // result = min(maxVal, max(minVal, x))
    Value minVal = createRank0Tensor(rewriter, loc, inputType, op.getMinVal());
    Value maxResult =
        rewriter.create<AtenMaximumOp>(loc, inputType, input, minVal);
    Value maxVal = createRank0Tensor(rewriter, loc, inputType, op.getMaxVal());
    rewriter.replaceOpWithNewOp<AtenMinimumOp>(op, op.getType(), maxVal,
                                               maxResult);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenRandLikeOp : public OpRewritePattern<AtenRandLikeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandLikeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Type resultType = op.getType();
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "only support floating-point type");
    }

    // Create a uniform random op with low and high set to 0.0 and 1.0,
    // respectively.
    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value one =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value emptyTensor = rewriter.create<AtenFullLikeOp>(
        loc, resultType, input, zero, op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory(), op.getMemoryFormat());
    rewriter.replaceOpWithNewOp<AtenUniformOp>(op, resultType, emptyTensor,
                                               /*from=*/zero, /*to=*/one,
                                               /*generator=*/none);
    return success();
  }
};
} // namespace

namespace {
// Bernoulli(x, p) = (randLike(float(x)) < p).cast(type(x)). Here,
// 1. p must be a float tensor.
// 2. The shape of p should be broadcastable to the shape of x.
// 3. Bernoulli(x, p) returns a tensor of the same type as that of x.
static LogicalResult decomposeBernoulliLikeOp(PatternRewriter &rewriter,
                                              Operation *op, Location loc,
                                              Value input, Value prob,
                                              Value &output) {
  auto inputType = input.getType().cast<BaseTensorType>();
  auto probType = prob.getType().cast<BaseTensorType>();
  // Both the `input` and `prob` must be ranked tensors.
  if (!inputType.hasSizes() || !inputType.hasDtype() || !probType.hasSizes() ||
      !probType.hasDtype()) {
    return rewriter.notifyMatchFailure(
        op, "can't decompose bernoulli like ops without sizes or dtype");
  }
  // The `prob` is expected to be a float type tensor.
  if (!probType.getDtype().isa<mlir::FloatType>()) {
    return rewriter.notifyMatchFailure(
        op, "probabilities must be a float type tensor");
  }

  // Since the `aten.randLike` op expects float-type operand, create a
  // float-type tensor with the same shape as that of the `input`.
  Value floatTensor =
      convertTensorToDtype(rewriter, loc, input, rewriter.getF64Type());
  Value none = rewriter.create<ConstantNoneOp>(loc);
  Value randomVal = rewriter.create<AtenRandLikeOp>(
      loc, floatTensor.getType(), floatTensor, /*dtype=*/none, /*layout=*/none,
      /*device=*/none, /*pinMemory=*/none, /*memoryFormat=*/none);

  // Bernoulli(x, p) = randLike(float(x)) < p.
  auto boolResType = inputType.getWithSizesAndDtype(inputType.getSizes(),
                                                    rewriter.getI1Type());
  Value lessThanP =
      rewriter.create<AtenLtTensorOp>(loc, boolResType, randomVal, prob);

  // As the `output` is expected to be of the `input` type, convert the boolean
  // tensor `lessThanP` to a `input` type tensor.
  output = convertTensorToDtype(rewriter, loc, lessThanP, inputType.getDtype());
  return success();
}

// aten.bernoulli(x) = randLike(x) < x. Here, the input x is a tensor
// containing probabilities to be used for drawing the binary random number.
class DecomposeAtenBernoulliOp : public OpRewritePattern<AtenBernoulliOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBernoulliOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    if (!op.getGenerator().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");
    Value output;
    if (failed(
            decomposeBernoulliLikeOp(rewriter, op, loc, input, input, output)))
      return rewriter.notifyMatchFailure(
          op, "decomposeBernoulliLikeOp failed to decompose the op");
    rewriter.replaceOp(op, output);
    return success();
  }
};

// aten.bernoulli.float(x, p) = (randLike(float(x)) < tensor(p)).cast(type(x)).
// Since the input x can be an integer tensor, it's important to cast it to
// float type before passing it to the `aten.randLike` op.
template <typename BernoulliLikeOp>
class DecomposeAtenBernoulliLikeOp : public OpRewritePattern<BernoulliLikeOp> {
public:
  using OpRewritePattern<BernoulliLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BernoulliLikeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value p = op.getP();
    if (!op.getGenerator().getType().template isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");

    auto inputType = input.getType().cast<BaseTensorType>();
    SmallVector<int64_t> empty;
    Type tensorType = inputType.getWithSizesAndDtype(llvm::ArrayRef(empty),
                                                     rewriter.getF64Type());
    Value prob = rewriter.create<PrimNumToTensorScalarOp>(loc, tensorType, p);
    Value output;
    if (failed(
            decomposeBernoulliLikeOp(rewriter, op, loc, input, prob, output)))
      return rewriter.notifyMatchFailure(
          op, "decomposeBernoulliLikeOp failed to decompose the op");
    rewriter.replaceOp(op, output);
    return success();
  }
};

// aten.bernoulli.Tensor(x, p) = (randLike(float(x)) < p).cast(type(x)).
// Since the input x can be an integer tensor, it's important to cast it to
// float type before passing it to the `aten.randLike` op.
class DecomposeAtenBernoulliTensorOp
    : public OpRewritePattern<AtenBernoulliTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBernoulliTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value prob = op.getP();
    if (!op.getGenerator().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");
    Value output;
    if (failed(
            decomposeBernoulliLikeOp(rewriter, op, loc, input, prob, output)))
      return rewriter.notifyMatchFailure(
          op, "decomposeBernoulliLikeOp failed to decompose the op");
    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

namespace {
// Decompose exponential() to do inverse transform sampling.
// - https://en.wikipedia.org/wiki/Inverse_transform_sampling
// With the exponential distribution, F(x) = 1 - exp(-lambda * x). Thus,
// exponential() = - ln(1 - uniform(0, 1)) / lambda.
class DecomposeAtenExponentialOp : public OpRewritePattern<AtenExponentialOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenExponentialOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getGenerator().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");

    Location loc = op.getLoc();
    Type resultType = op.getType();

    // Create a uniform random op with low and high set to 0.0 and 1.0,
    // respectively.
    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value one =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value emptyTensor = rewriter.create<AtenFullLikeOp>(
        loc, resultType, op.getSelf(), zero, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pin_memoty=*/none, /*memory_format=*/none);
    Value x = rewriter.create<AtenUniformOp>(loc, resultType, emptyTensor,
                                             /*from=*/zero, /*to=*/one,
                                             /*generator=*/none);

    Value negX = rewriter.create<AtenNegOp>(loc, resultType, x);
    Value oneMinusX =
        rewriter.create<AtenAddScalarOp>(loc, resultType, negX, one,
                                         /*alpha=*/one);
    Value lnOneMinusX = rewriter.create<AtenLogOp>(loc, resultType, oneMinusX);
    Value negLambda = rewriter.create<AtenNegFloatOp>(loc, op.getLambd());
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, resultType, lnOneMinusX,
                                                 negLambda);
    return success();
  }
};

// aten.normal_functional(mean, sigma) = randn() * sigma + mean.
class DecomposeAtenNormalFunctionalOp
    : public OpRewritePattern<AtenNormalFunctionalOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNormalFunctionalOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getGenerator().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");

    Location loc = op.getLoc();
    Type resultType = op.getType();
    Value std = op.getStd();
    Value mean = op.getMean();

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value one =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value randN = rewriter.create<AtenRandnLikeOp>(
        loc, resultType, op.getSelf(), /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none, /*memory_format=*/none);
    Value stdRandN =
        rewriter.create<AtenMulScalarOp>(loc, resultType, randN, std);
    rewriter.replaceOpWithNewOp<AtenAddScalarOp>(op, resultType, stdRandN, mean,
                                                 /*alpha=*/one);
    return success();
  }
};

template <typename OpTy, typename T1T2Op>
class DecomposeAtenAddCLikeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value tensor1 = op.getTensor1();
    Value tensor2 = op.getTensor2();
    Value value = op.getValue();

    Value product =
        rewriter.create<T1T2Op>(loc, op.getType(), tensor1, tensor2);
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), input,
                                                 product, value);
    return success();
  }
};

class DecomposeAtenLayerNormOp : public OpRewritePattern<AtenLayerNormOp> {
  using OpRewritePattern<AtenLayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLayerNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto input = op.getInput().getType().cast<BaseTensorType>();
    if (!input.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "input tensor should have known sizes.");
    int64_t inputRank = input.getSizes().size();
    Value normalizedShape = op.getNormalizedShape();
    SmallVector<Value> normalizedShapeSizesTorchInt;
    getListConstructElements(normalizedShape, normalizedShapeSizesTorchInt);
    int64_t axis = inputRank - normalizedShapeSizesTorchInt.size();
    std::vector<int64_t> meanVarSizes(inputRank, 1);
    for (int i = 0; i < axis; i++)
      meanVarSizes[i] = input.getSizes()[i];
    auto meanVarType = input.getWithSizesAndDtype(llvm::ArrayRef(meanVarSizes),
                                                  input.getOptionalDtype());
    auto nativeLayerNorm = rewriter.create<AtenNativeLayerNormOp>(
        loc, op.getType(), meanVarType, meanVarType, op.getInput(),
        op.getNormalizedShape(), op.getWeight(), op.getBias(), op.getEps());
    rewriter.replaceOp(op, nativeLayerNorm.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenInstanceNormOp
    : public OpRewritePattern<AtenInstanceNormOp> {
  using OpRewritePattern<AtenInstanceNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenInstanceNormOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto context = op.getContext();

    auto inputTy = op.getInput().getType().cast<BaseTensorType>();
    int64_t inputRank = inputTy.getSizes().size();
    SmallVector<int64_t> reducedShape(inputTy.getSizes());
    SmallVector<int64_t> reduceDimInts;
    SmallVector<Value> reduceDimVals;
    for (int i = 2; i < inputRank; ++i) {
      reducedShape[i] = 1;
      reduceDimVals.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    }

    Type dtype = inputTy.getOptionalDtype();
    Type reducedTy = ValueTensorType::get(op.getContext(),
                                          llvm::ArrayRef(reducedShape), dtype);

    auto sizeListType = ListType::get(IntType::get(context));
    Value reduceDimList =
        rewriter.create<PrimListConstructOp>(loc, sizeListType, reduceDimVals);
    Value cstTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);

    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));

    // mean(x)
    Value inputMean = rewriter.create<AtenMeanDimOp>(
        loc, reducedTy, op.getInput(), reduceDimList, cstTrue, none);

    // x - mean(x)
    Value inputMeanExpanded =
        rewriter.create<AtenExpandAsOp>(loc, inputTy, inputMean, op.getInput());
    Value inputSubMean = rewriter.create<AtenSubTensorOp>(
        loc, inputTy, op.getInput(), inputMeanExpanded, one);
    // (x - mean(x))^2
    Value inputSubMeanSquare = rewriter.create<AtenMulTensorOp>(
        loc, inputTy, inputSubMean, inputSubMean);

    Value variancesum = rewriter.create<AtenSumDimIntListOp>(
        loc, reducedTy, inputSubMeanSquare, reduceDimList, cstTrue,
        /*dtype=*/none);

    int64_t elemCount = 1;
    for (int i = 2; i < inputRank; ++i)
      elemCount *= inputTy.getSizes()[i];

    Value hw = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(elemCount));
    Value inputVar =
        rewriter.create<AtenDivScalarOp>(loc, reducedTy, variancesum, hw);

    // rsqrt(var(x) + eps)
    Value inputVarPlusEps = rewriter.create<AtenAddScalarOp>(
        loc, reducedTy, inputVar, op.getEps(), one);
    Value inputRsqrtVar =
        rewriter.create<AtenRsqrtOp>(loc, reducedTy, inputVarPlusEps);

    // (x - mean(x)) * rsqrt(var(x) + eps)
    Value inputRsqrtVarExpanded = rewriter.create<AtenExpandAsOp>(
        loc, inputTy, inputRsqrtVar, op.getInput());
    Value inputNormalized = rewriter.create<AtenMulTensorOp>(
        loc, inputTy, inputSubMean, inputRsqrtVarExpanded);
    Value out = rewriter.create<TensorStaticInfoCastOp>(
        loc, op.getResult().getType(), inputNormalized);

    Value weight = op.getWeight();
    auto weightTy = weight.getType().cast<BaseTensorType>();
    dtype = weightTy.getOptionalDtype();

    SmallVector<int64_t> weightShape(weightTy.getSizes());
    SmallVector<int64_t> newWeightShape;
    newWeightShape.push_back(1);
    newWeightShape.append(weightShape);

    Value zero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Type newWeightTy = ValueTensorType::get(
        op.getContext(), llvm::ArrayRef(newWeightShape), dtype);
    weight = rewriter.create<AtenUnsqueezeOp>(loc, newWeightTy, weight, zero);

    while (static_cast<int64_t>(newWeightShape.size()) < inputRank) {
      Value i = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(newWeightShape.size()));
      newWeightShape.push_back(1);
      newWeightTy = ValueTensorType::get(op.getContext(),
                                         llvm::ArrayRef(newWeightShape), dtype);
      weight = rewriter.create<AtenUnsqueezeOp>(loc, newWeightTy, weight, i);
    }

    Value weightExpanded =
        rewriter.create<AtenExpandAsOp>(loc, inputTy, weight, op.getInput());

    Value bias = op.getBias();
    auto biasTy = bias.getType().cast<BaseTensorType>();
    dtype = biasTy.getOptionalDtype();

    SmallVector<int64_t> biasShape(biasTy.getSizes());
    SmallVector<int64_t> newBiasShape;
    newBiasShape.push_back(1);
    newBiasShape.append(biasShape);

    Type newBiasTy = ValueTensorType::get(op.getContext(),
                                          llvm::ArrayRef(newBiasShape), dtype);
    bias = rewriter.create<AtenUnsqueezeOp>(loc, newBiasTy, bias, zero);

    while (static_cast<int64_t>(newBiasShape.size()) < inputRank) {
      Value i = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(newBiasShape.size()));
      newBiasShape.push_back(1);
      newBiasTy = ValueTensorType::get(op.getContext(),
                                       llvm::ArrayRef(newBiasShape), dtype);
      bias = rewriter.create<AtenUnsqueezeOp>(loc, newBiasTy, bias, i);
    }

    Value biasExpanded =
        rewriter.create<AtenExpandAsOp>(loc, inputTy, bias, op.getInput());

    out = rewriter.create<AtenMulTensorOp>(loc, out.getType(), out,
                                           weightExpanded);
    out = rewriter.create<AtenAddTensorOp>(loc, out.getType(), out,
                                           biasExpanded, one);

    rewriter.replaceOp(op, out);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNativeLayerNormOp
    : public OpRewritePattern<AtenNativeLayerNormOp> {
  using OpRewritePattern<AtenNativeLayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNativeLayerNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();

    auto inputTy = op.getInput().getType().cast<BaseTensorType>();
    if (!inputTy.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "input tensor should have known sizes.");
    int64_t inputRank = inputTy.getSizes().size();
    Value normalizedShape = op.getNormalizedShape();
    SmallVector<Value> normalizedShapeSizesTorchInt;
    getListConstructElements(normalizedShape, normalizedShapeSizesTorchInt);
    int64_t axis = inputRank - normalizedShapeSizesTorchInt.size();
    auto reduceDimInts =
        llvm::to_vector<4>(llvm::seq<int64_t>(axis, inputRank));
    auto reducedTy = op.getResult(1).getType();
    auto sizeListType = ListType::get(IntType::get(context));

    // build reduce dims
    SmallVector<Value> reduceDimVals;
    reduceDimVals.reserve(reduceDimInts.size());
    std::transform(reduceDimInts.begin(), reduceDimInts.end(),
                   std::back_inserter(reduceDimVals), [&](int64_t d) {
                     return rewriter.create<Torch::ConstantIntOp>(
                         loc, rewriter.getI64IntegerAttr(d));
                   });
    Value reduceDimList =
        rewriter.create<PrimListConstructOp>(loc, sizeListType, reduceDimVals);
    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));

    Value cstTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
    // mean(x)
    Value inputMean = rewriter.create<AtenMeanDimOp>(
        loc, reducedTy, op.getInput(), reduceDimList, cstTrue, none);

    // x - mean(x)
    Value inputMeanExpanded =
        rewriter.create<AtenExpandAsOp>(loc, inputTy, inputMean, op.getInput());
    Value inputZeroMean = rewriter.create<AtenSubTensorOp>(
        loc, inputTy, op.getInput(), inputMeanExpanded, one);
    // var(x) = mean((x - mean(x))^2)
    Value inputZeroMeanSquare = rewriter.create<AtenMulTensorOp>(
        loc, inputTy, inputZeroMean, inputZeroMean);
    Value inputVar = rewriter.create<AtenMeanDimOp>(
        loc, reducedTy, inputZeroMeanSquare, reduceDimList, cstTrue, none);

    // rsqrt(var(x) + eps)
    Value inputVarPlusEps = rewriter.create<AtenAddScalarOp>(
        loc, reducedTy, inputVar, op.getEps(), one);
    Value inputRsqrtVar =
        rewriter.create<AtenRsqrtOp>(loc, reducedTy, inputVarPlusEps);

    // (x - mean(x)) * rsqrt(var(x) + eps)
    Value inputRsqrtVarExpanded = rewriter.create<AtenExpandAsOp>(
        loc, inputTy, inputRsqrtVar, op.getInput());
    Value inputNormalized = rewriter.create<AtenMulTensorOp>(
        loc, inputTy, inputZeroMean, inputRsqrtVarExpanded);
    Value out = rewriter.create<TensorStaticInfoCastOp>(
        loc, op.getResult(0).getType(), inputNormalized);

    Value weight = op.getWeight();
    Value bias = op.getBias();
    if (!weight.getType().isa<Torch::NoneType>()) {
      out = rewriter.create<AtenMulTensorOp>(loc, out.getType(), out, weight);
    }
    if (!bias.getType().isa<Torch::NoneType>()) {
      out =
          rewriter.create<AtenAddTensorOp>(loc, out.getType(), out, bias, one);
    }
    rewriter.replaceOp(op, {out, inputMean, inputRsqrtVar});

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.emptyLike` op into `aten.size` and `aten.empty` ops.
class DecomposeAtenEmptyLikeOp : public OpRewritePattern<AtenEmptyLikeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEmptyLikeOp op,
                                PatternRewriter &rewriter) const override {
    auto sizeListType =
        Torch::ListType::get(Torch::IntType::get(op.getContext()));
    Value sizeList =
        rewriter.create<AtenSizeOp>(op.getLoc(), sizeListType, op.getSelf());
    rewriter.replaceOpWithNewOp<AtenEmptyMemoryFormatOp>(
        op, op.getType(), sizeList, op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory(), op.getMemoryFormat());
    return success();
  }
};
} // namespace

namespace {
// The `aten.arange` op is converted to `aten.arange.startStep` op.
class DecomposeAtenArangeOp : public OpRewritePattern<AtenArangeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenArangeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The AtenArangeOp doesn't have a start and step value. Therefore we set
    // them as default values 0 and 1, respectively.
    Value start, step;
    start = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    step = rewriter.create<Torch::ConstantIntOp>(loc,
                                                 rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<AtenArangeStartStepOp>(
        op, op.getType(), start, op.getEnd(), step, op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
// The `aten.arange.start` op is converted to `aten.arange.startStep` op.
class DecomposeAtenArangeStartOp : public OpRewritePattern<AtenArangeStartOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenArangeStartOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The AtenArangeStartOp doesn't have a step value. Therefore we set it as
    // default value 1.
    Value step;
    step = rewriter.create<Torch::ConstantIntOp>(loc,
                                                 rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<AtenArangeStartStepOp>(
        op, op.getType(), op.getStart(), op.getEnd(), step, op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
// Decompose constant tensor full like ops.
template <typename OpTy, int fillVal>
class DecomposeConstantTensorAllocLikeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value constVal = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(fillVal));
    rewriter.replaceOpWithNewOp<AtenFullLikeOp>(
        op, op.getType(), op.getSelf(), constVal, op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory(), op.getMemoryFormat());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenGroupNormOp : public OpRewritePattern<AtenGroupNormOp> {
  using OpRewritePattern<AtenGroupNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenGroupNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    Value numGroups = op.getNumGroups();
    Value eps = op.getEps();

    Value cstZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    auto baseType = ValueTensorType::getWithLeastStaticInformation(context);

    Value N = rewriter.create<AtenSizeIntOp>(loc, input, cstZero);
    Value C = rewriter.create<AtenSizeIntOp>(loc, input, cstOne);
    Value numElements = rewriter.create<AtenNumelOp>(loc, input);
    Value numElementsDivN =
        rewriter.create<AtenFloordivIntOp>(loc, numElements, N);
    Value HxW = rewriter.create<AtenFloordivIntOp>(loc, numElementsDivN, C);

    AtenNativeGroupNormOp newOp = rewriter.create<AtenNativeGroupNormOp>(
        loc, ArrayRef<Type>{op.getResult().getType(), baseType, baseType},
        input, weight, bias, N, C, HxW, numGroups, eps);

    rewriter.replaceOp(op, newOp.getResult0());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNativeGroupNormOp
    : public OpRewritePattern<AtenNativeGroupNormOp> {
  using OpRewritePattern<AtenNativeGroupNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNativeGroupNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    Value numGroups = op.getGroup();
    Value eps = op.getEps();

    // Check the rank of the input/outputs tensor.
    auto inputType = input.getType().cast<BaseTensorType>();
    auto outputType = op.getResult0().getType().cast<BaseTensorType>();
    auto meanType = op.getResult1().getType().cast<BaseTensorType>();
    auto rsqrtVarType = op.getResult2().getType().cast<BaseTensorType>();
    if (!inputType.hasSizes() || !outputType.hasSizes() ||
        !meanType.hasSizes() || !rsqrtVarType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "input/outputs tensor should have known sizes.");
    }

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value cstZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value cstNegtiveOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(-1));
    Value cstTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    auto baseType = ValueTensorType::getWithLeastStaticInformation(context);

    // GroupNorm requires the channel dimension (C) to be exactly divisible by
    // the number of groups.
    Value channel = rewriter.create<AtenSizeIntOp>(loc, input, cstOne);
    Value remainder =
        rewriter.create<AtenRemainderIntOp>(loc, channel, numGroups);
    Value eqOrNot = rewriter.create<AtenEqIntOp>(loc, remainder, cstZero);
    rewriter.create<RuntimeAssertOp>(
        loc, eqOrNot,
        rewriter.getStringAttr("the number of channels must be divisible by "
                               "the number of groups"));

    // Reshape the input tensor to (N, numGroups, -1) to apply normalization.
    SmallVector<Value> newShape;
    newShape.push_back(rewriter.create<AtenSizeIntOp>(loc, input, cstZero));
    newShape.push_back(numGroups);
    newShape.push_back(cstNegtiveOne);
    Value reshapedInput = rewriter.create<AtenViewOp>(
        loc, baseType, input,
        rewriter.create<PrimListConstructOp>(
            loc, Torch::ListType::get(IntType::get(context)), newShape));

    // Now we proceed with the normalization steps across the 'groupSize'
    // Compute the mean and variance for each group
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        ArrayRef<Value>{cstNegtiveOne});
    auto mean = rewriter.create<AtenMeanDimOp>(
        loc, baseType, reshapedInput, /*dims=*/dimList, /*keepdim=*/cstTrue,
        /*dtype=*/none);
    auto var = rewriter.create<AtenVarDimOp>(
        loc, baseType, reshapedInput, /*dims=*/dimList, /*unbiased=*/cstFalse,
        /*keepdim=*/cstTrue);

    // Compute the normalized output: (input - mean) * rsqrt(var + eps)
    auto varPlusEps = rewriter.create<AtenAddScalarOp>(loc, baseType, var, eps,
                                                       /*alpha=*/cstOne);
    auto invStd = rewriter.create<AtenRsqrtOp>(loc, baseType, varPlusEps);
    auto inputSubMean = rewriter.create<AtenSubTensorOp>(
        loc, baseType, reshapedInput, mean, /*alpha=*/cstOne);
    auto normalizedOutput =
        rewriter.create<AtenMulTensorOp>(loc, baseType, inputSubMean, invStd);

    // Reshape normalized output back to the original input shape
    auto inputShape = rewriter.create<AtenSizeOp>(
        loc, Torch::ListType::get(IntType::get(context)), input);
    auto reshapedOutput = rewriter.create<AtenViewOp>(
        loc, inputType, normalizedOutput, /*shape=*/inputShape);

    // Apply weight and bias if they are not None
    // Reshape weight and bias to C,1,1,...
    SmallVector<Value> viewShape = {channel};
    for (unsigned i = 2; i < inputType.getSizes().size(); i++) {
      viewShape.push_back(cstOne);
    }
    Value viewShapeSizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), viewShape);

    Value groupNormOutput = reshapedOutput;
    if (!weight.getType().isa<Torch::NoneType>()) {
      auto weightReshaped = rewriter.create<AtenViewOp>(
          loc, baseType, weight, /*shape=*/viewShapeSizeList);
      groupNormOutput = rewriter.create<AtenMulTensorOp>(
          loc, inputType, groupNormOutput, weightReshaped);
    }
    if (!bias.getType().isa<Torch::NoneType>()) {
      auto biasReshaped = rewriter.create<AtenViewOp>(
          loc, baseType, bias, /*shape=*/viewShapeSizeList);
      groupNormOutput = rewriter.create<AtenAddTensorOp>(
          loc, inputType, groupNormOutput, biasReshaped,
          /*alpha=*/cstOne);
    }

    Value squeezedMean =
        rewriter.create<AtenSqueezeDimOp>(loc, meanType, mean, cstNegtiveOne);
    Value squeezedRsqrtVar = rewriter.create<AtenSqueezeDimOp>(
        loc, rsqrtVarType, invStd, cstNegtiveOne);

    rewriter.replaceOp(
        op, ArrayRef<Value>{groupNormOutput, squeezedMean, squeezedRsqrtVar});

    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNativeBatchNormOp
    : public OpRewritePattern<AtenNativeBatchNormOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNativeBatchNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    Value runningMean = op.getRunningMean();
    Value runningVar = op.getRunningVar();
    Value eps = op.getEps();

    // TODO: Add support for `training` mode.
    bool training = false;
    if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training)) ||
        training)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: training mode is not supported");

    // Rank of the input tensor must be greater than or equal to 2. The shape of
    // the `input` is supposed to be (N, C, D?, H?, W?).
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank || *maybeInputRank < 2)
      return rewriter.notifyMatchFailure(
          op, "input must have rank greater than or equal to 2");
    unsigned inputRank = *maybeInputRank;

    // In the inference mode, the `runningMean` and `runningVar` must not be
    // None.
    if (runningMean.getType().isa<Torch::NoneType>() ||
        runningVar.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "running stats must not be None in inference mode");

    // Rank of `runningMean` and `runningVar` must be exactly 1.
    std::optional<unsigned> runningMeanRank = getTensorRank(runningMean);
    std::optional<unsigned> runningVarRank = getTensorRank(runningVar);
    if (!runningMeanRank || !runningVarRank || *runningMeanRank != 1 ||
        *runningVarRank != 1)
      return rewriter.notifyMatchFailure(
          op, "expected runningMean and runningVar to be rank 1");

    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value numFeatures = rewriter.create<AtenSizeIntOp>(loc, input, /*dim=*/one);
    // TODO: Add Runtime Asserts to check the shape of weight, bias,
    // runningMean and runningVar to be (numFeatures).

    // The `runningMean` and `runningVar` must be reshaped to (1, C, 1?, 1?, 1?)
    // to make it broadcast-compatible with (N, C, D?, H?, W?).
    // 1. runningMean = runningMean.view(1, C, 1?, 1?, 1?)
    // 2. runningVar = runningVar.view(1, C, 1?, 1?, 1?)
    SmallVector<Value> runningStatsShape(inputRank, one);
    runningStatsShape[1] = numFeatures;
    Value runningStatsSizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), runningStatsShape);

    SmallVector<int64_t> runningStatsShapeInt(inputRank, 1);
    runningStatsShapeInt[1] =
        runningMean.getType().cast<BaseTensorType>().getSizes()[0];
    Type dtype = input.getType().cast<ValueTensorType>().getOptionalDtype();
    Type reshapeType = ValueTensorType::get(
        context, llvm::ArrayRef(runningStatsShapeInt), dtype);

    runningMean = rewriter.create<AtenViewOp>(loc, reshapeType, runningMean,
                                              runningStatsSizeList);
    runningVar = rewriter.create<AtenViewOp>(loc, reshapeType, runningVar,
                                             runningStatsSizeList);

    // normalizedInput = (input - runningMean) / (sqrt(runningVar + eps)).
    Value inputSubMean = rewriter.create<AtenSubTensorOp>(
        loc, input.getType(), input, runningMean, /*alpha=*/one);
    Value varEps = rewriter.create<AtenAddScalarOp>(
        loc, runningVar.getType(), runningVar, eps, /*alpha=*/one);
    Value invStd = rewriter.create<AtenRsqrtOp>(loc, varEps.getType(), varEps);
    Value normalizedInput = rewriter.create<AtenMulTensorOp>(
        loc, inputSubMean.getType(), inputSubMean, invStd);

    // The `weight` and `bias` must be reshaped to (1, C, 1?, 1?, 1?) to make it
    // broadcast-compatible with (N, C, D?, H?, W?).
    // 1. weight = weight.view(1, C, 1?, 1?, 1?)
    // 2. bias = bias.view(1, C, 1?, 1?, 1?)
    // 3. output = normalizedInput * weight + bias
    Value batchNormOutput = normalizedInput;
    if (!weight.getType().isa<Torch::NoneType>()) {
      // Rank of `weight` must be exactly 1.
      std::optional<unsigned> weightRank = getTensorRank(weight);
      if (!weightRank || *weightRank != 1)
        return rewriter.notifyMatchFailure(op, "expected weight to be rank 1");
      weight = rewriter.create<AtenViewOp>(loc, reshapeType, weight,
                                           runningStatsSizeList);
      batchNormOutput = rewriter.create<AtenMulTensorOp>(
          loc, batchNormOutput.getType(), batchNormOutput, weight);
    }
    if (!bias.getType().isa<Torch::NoneType>()) {
      // Rank of `bias` must be exactly 1.
      std::optional<unsigned> biasRank = getTensorRank(bias);
      if (!biasRank || *biasRank != 1)
        return rewriter.notifyMatchFailure(op, "expected bias to be rank 1");
      bias = rewriter.create<AtenViewOp>(loc, reshapeType, bias,
                                         runningStatsSizeList);
      batchNormOutput = rewriter.create<AtenAddTensorOp>(
          loc, batchNormOutput.getType(), batchNormOutput, bias, /*alpha=*/one);
    }

    // The `mean` and `invstd` outputs are empty tensors in inference mode.
    Value zeroList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(zero.getType()), zero);
    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value emptyMeanTensor = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, op.getType(1), zeroList, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pinMemory=*/none, /*memoryFormat=*/none);
    Value emptyInvStdTensor = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, op.getType(2), zeroList, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pinMemory=*/none, /*memoryFormat=*/none);

    rewriter.replaceOp(op,
                       {batchNormOutput, emptyMeanTensor, emptyInvStdTensor});
    return success();
  }
};
} // namespace

// Decompse `Aten_UnsafeViewOp` into `AtenViewOp`. UnsafeView() differs from
// view() in that the returned tensor isn't treated as a view for the purposes
// of automatic differentiation.  It's only safe to use if the `self` tensor is
// temporary. For example, the viewed tensor here (a + b) is discarded
// immediately after viewing:
//
//  res = UnsafeView(a + b, size);
//
// This is a hack because in-place operations on tensors treated like views
// can be much more expensive than the same operations on non-view tensors.

// Refer to
// https://github.com/pytorch/pytorch/blob/364055b2771ecf9b54f1d67a8bf44bb5496476d4/aten/src/ATen/native/TensorShape.cpp#L2072
namespace {
class DecomposeAten_UnsafeViewOp : public OpRewritePattern<Aten_UnsafeViewOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_UnsafeViewOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), op.getSelf(),
                                            op.getSize());
    return success();
  }
};
} // namespace

// In PyTorch, ReshapeAlias just uses an already computed stride.
// See
// https://github.com/pytorch/pytorch/blob/d8c31a819d4a65e732b5901e3b994e1869851f1a/aten/src/ATen/native/TensorShape.cpp#L1153
// Note that this is the same decomposition as in AOTAutograd
// https://github.com/pytorch/functorch/blob/a3042d94e616d4143813668b1372d9d4545be14e/functorch/Src/aotAutograd.py#L104
namespace {
class DecomposeAten_ReshapeAliasOp
    : public OpRewritePattern<Aten_ReshapeAliasOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ReshapeAliasOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), op.getSelf(),
                                            op.getSize());
    return success();
  }
};
} // namespace

namespace {
// Decompose constant tensor like ops.
template <typename OpTy, typename NewOpTy>
class DecomposeConstantTensorNewLikeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value dtype = op.getDtype();
    if (dtype.getType().isa<Torch::NoneType>()) {
      BaseTensorType tensorType =
          op.getSelf().getType().template cast<BaseTensorType>();
      if (!tensorType.hasDtype()) {
        return rewriter.notifyMatchFailure(
            op, "expected input tensor to have a dtype");
      }
      dtype =
          getDtypeIntValueForType(rewriter, op.getLoc(), tensorType.getDtype());
    }
    rewriter.replaceOpWithNewOp<NewOpTy>(op, op.getType(), op.getSize(), dtype,
                                         op.getLayout(), op.getDevice(),
                                         op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.full` op into `aten.broadcastTo`
class DecomposeAtenFullOp : public OpRewritePattern<AtenFullOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFullOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    BaseTensorType outTy = op.getType().template cast<BaseTensorType>();
    if (!outTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    SmallVector<int64_t> empty;
    auto dtype =
        getTypeForTorchType(op.getContext(), op.getFillValue().getType());
    Type tensorType = outTy.getWithSizesAndDtype(llvm::ArrayRef(empty), dtype);
    Value fillVal = rewriter.create<PrimNumToTensorScalarOp>(loc, tensorType,
                                                             op.getFillValue());
    fillVal = convertTensorToDtype(rewriter, loc, fillVal, outTy.getDtype());
    rewriter.replaceOpWithNewOp<AtenBroadcastToOp>(op, op.getType(), fillVal,
                                                   op.getSize());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.linear` op into `aten.matmul` and `aten.add` ops.
class DecomposeAtenLinearOp : public OpRewritePattern<AtenLinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLinearOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();

    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasSizes() || inputType.getSizes().size() < 2)
      return rewriter.notifyMatchFailure(
          op, "expected input to be rank 2 or greater");

    BaseTensorType weightType = weight.getType().cast<BaseTensorType>();
    // `weight` must be a rank 2 matrix.
    if (!weightType.hasSizes() || weightType.getSizes().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected weight to be a rank 2");

    SmallVector<int64_t> transposeShape =
        llvm::to_vector(llvm::reverse(weightType.getSizes()));
    Type transposeType = weightType.getWithSizesAndDtype(
        llvm::ArrayRef(transposeShape), weightType.getOptionalDtype());
    Value transposeWeight =
        rewriter.create<AtenTOp>(loc, transposeType, weight);

    Value matmul = rewriter.create<AtenMatmulOp>(loc, op.getType(), input,
                                                 transposeWeight);
    if (bias.getType().isa<Torch::NoneType>()) {
      rewriter.replaceOp(op, matmul);
      return success();
    }

    BaseTensorType biasType = bias.getType().cast<BaseTensorType>();
    if (!biasType.hasSizes() || biasType.getSizes().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected bias to be rank 1");

    Value alpha =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), matmul,
                                                 op.getBias(), alpha);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.mish` op into `aten.tanh` and `aten.softplus` ops.
// Mish(x) = x * Tanh(Softplus(x))
class DecomposeAtenMishOp : public OpRewritePattern<AtenMishOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMishOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Type type = op.getType();

    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype())
      return rewriter.notifyMatchFailure(op, "Dtype not present");

    Type dType = inputType.getDtype();
    // Form default Value tensors for `beta` and `threshold` operands
    // of `aten.softplus` op.
    Value beta = getConstantWithGivenDtypeAndValue(rewriter, loc, 1.0, dType);
    Value threshold =
        getConstantWithGivenDtypeAndValue(rewriter, loc, 20.0, dType);
    Value softplusOp =
        rewriter.create<AtenSoftplusOp>(loc, type, input, beta, threshold);
    Value tanhOp = rewriter.create<AtenTanhOp>(loc, type, softplusOp);
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, type, input, tanhOp);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.fullLike` op into `aten.emptyLike` and `aten.fill` ops.
class DecomposeAtenFullLikeOp : public OpRewritePattern<AtenFullLikeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFullLikeOp op,
                                PatternRewriter &rewriter) const override {
    BaseTensorType outTy = op.getType().template cast<BaseTensorType>();
    if (!outTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    SmallVector<int64_t> empty;
    auto dtype =
        getTypeForTorchType(op.getContext(), op.getFillValue().getType());
    Type tensorType = outTy.getWithSizesAndDtype(llvm::ArrayRef(empty), dtype);
    Value fillVal = rewriter.create<PrimNumToTensorScalarOp>(
        op.getLoc(), tensorType, op.getFillValue());
    fillVal =
        convertTensorToDtype(rewriter, op.getLoc(), fillVal, outTy.getDtype());
    rewriter.replaceOpWithNewOp<AtenExpandAsOp>(op, op.getType(), fillVal,
                                                op.getSelf());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.new_full` op into `aten.full` op.
class DecomposeAtenNewFullOp : public OpRewritePattern<AtenNewFullOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNewFullOp op,
                                PatternRewriter &rewriter) const override {
    Value dtype = op.getDtype();
    if (dtype.getType().isa<Torch::NoneType>()) {
      BaseTensorType tensorType = op.getSelf().getType().cast<BaseTensorType>();
      if (!tensorType.hasDtype()) {
        return rewriter.notifyMatchFailure(
            op, "expected input tensor to have a dtype");
      }
      dtype =
          getDtypeIntValueForType(rewriter, op.getLoc(), tensorType.getDtype());
    }
    rewriter.replaceOpWithNewOp<AtenFullOp>(
        op, op.getType(), op.getSize(), op.getFillValue(), dtype,
        op.getLayout(), op.getDevice(), op.getPinMemory());

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.indexPut` op into `valsem.aten.indexPutImpl` op.
class DecomposeAtenIndexPutOp : public OpRewritePattern<AtenIndexPutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIndexPutOp op,
                                PatternRewriter &rewriter) const override {
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<Aten_IndexPutImplOp>(
        op, op.getType(), op.getSelf(), op.getIndices(), op.getValues(),
        op.getAccumulate(),
        /*unsafe=*/cstFalse);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenExpandAsOp : public OpRewritePattern<AtenExpandAsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenExpandAsOp op,
                                PatternRewriter &rewriter) const override {

    auto sizeListType =
        Torch::ListType::get(Torch::IntType::get(op.getContext()));
    Value sizeList =
        rewriter.create<AtenSizeOp>(op.getLoc(), sizeListType, op.getOther());
    rewriter.replaceOpWithNewOp<AtenBroadcastToOp>(op, op.getType(),
                                                   op.getSelf(), sizeList);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.ToCopy` op into `valsem.aten.copy` op.
class DecomposeAten_ToCopyOp : public OpRewritePattern<Aten_ToCopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ToCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<BaseTensorType>();
    if (!resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Type resultDtype = resultType.getDtype();
    Value zero = getConstantWithGivenDtypeAndValue(rewriter, op.getLoc(), 0.0,
                                                   resultDtype);
    Value emptyTensor = rewriter.create<AtenFullLikeOp>(
        op.getLoc(), op.getType(), op.getSelf(), zero, op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory(),
        op.getMemoryFormat());
    rewriter.replaceOpWithNewOp<AtenCopyOp>(op, op.getType(), emptyTensor,
                                            op.getSelf(), op.getNonBlocking());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.copy` op into `aten.to.dtype` and `aten.expand_as`.
class DecomposeAtenCopyOp : public OpRewritePattern<AtenCopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<BaseTensorType>();
    if (!resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    auto srcTy = op.getSrc().getType().cast<BaseTensorType>();
    if (!srcTy.hasSizes() || !srcTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected src type to have a known rank and dtype");
    }
    Type resultDtype = resultType.getDtype();
    Value srcToDtype =
        convertTensorToDtype(rewriter, op.getLoc(), op.getSrc(), resultDtype);
    rewriter.replaceOpWithNewOp<AtenExpandAsOp>(op, op.getType(), srcToDtype,
                                                op.getSelf());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.newEmpty` op into `aten.empty.memoryFormat` op.
class DecomposeAtenNewEmptyOp : public OpRewritePattern<AtenNewEmptyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNewEmptyOp op,
                                PatternRewriter &rewriter) const override {
    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value dtype = op.getDtype();
    if (dtype.getType().isa<Torch::NoneType>()) {
      BaseTensorType tensorType = op.getSelf().getType().cast<BaseTensorType>();
      if (!tensorType.hasDtype()) {
        return rewriter.notifyMatchFailure(
            op, "expected input tensor to have a dtype");
      }
      dtype =
          getDtypeIntValueForType(rewriter, op.getLoc(), tensorType.getDtype());
    }
    rewriter.replaceOpWithNewOp<AtenEmptyMemoryFormatOp>(
        op, op.getType(), op.getSize(), dtype, op.getLayout(), op.getDevice(),
        op.getPinMemory(), /*memoryFormat=*/noneVal);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.indexPut.hackedTwin` op into `valsem.aten.indexPutImpl`
// op.
class DecomposeAtenIndexPutHackedTwinOp
    : public OpRewritePattern<AtenIndexPutHackedTwinOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIndexPutHackedTwinOp op,
                                PatternRewriter &rewriter) const override {
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<Aten_IndexPutImplOp>(
        op, op.getType(), op.getSelf(), op.getIndices(), op.getValues(),
        op.getAccumulate(),
        /*unsafe=*/cstFalse);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten._unsafe_indexPut.hackedTwin` op into `aten._index_put_impl`
// op.
class DecomposeAten_UnsafeIndexPutHackedTwinOp
    : public OpRewritePattern<Aten_UnsafeIndexPutHackedTwinOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_UnsafeIndexPutHackedTwinOp op,
                                PatternRewriter &rewriter) const override {
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<Aten_IndexPutImplOp>(
        op, op.getType(), op.getSelf(), op.getIndices(), op.getValues(),
        op.getAccumulate(),
        /*unsafe=*/cstFalse);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.pad` op into `aten.constantPadNd` op.
class DecomposeAtenPadOp : public OpRewritePattern<AtenPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenPadOp op,
                                PatternRewriter &rewriter) const override {

    Value value = op.getValue();
    if (value.getType().isa<Torch::OptionalType>())
      return rewriter.notifyMatchFailure(op, "optional type not supported");
    if (value.getType().isa<Torch::NoneType>())
      value = rewriter.create<Torch::ConstantFloatOp>(
          op.getLoc(), rewriter.getF64FloatAttr(0));

    rewriter.replaceOpWithNewOp<AtenConstantPadNdOp>(
        op, op.getType(), op.getSelf(), op.getPad(), value);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.to.dtypeLayout` op into `aten.to.dtype` op.
class DecomposeAtenToDtypeLayoutOp
    : public OpRewritePattern<AtenToDtypeLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenToDtypeLayoutOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Add support for pinMemory arg equal to `True`.
    if (!op.getPinMemory().getType().isa<Torch::NoneType>()) {
      bool pinMemory;
      if (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: pinMemory must be a constant");
      else if (pinMemory)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: pinMemory is expected to be false");
    }

    // TODO: Add support for device arg other than cpu.
    if (!op.getDevice().getType().isa<Torch::NoneType>()) {
      std::string device;
      if (!matchPattern(op.getDevice(), m_TorchConstantDevice(device)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: device must be a constant str");
      else if (device != "cpu")
        return rewriter.notifyMatchFailure(
            op, "unimplemented: device is expected to be cpu");
    }

    // TODO: Add support for non-strided layout.
    // torch.layout is by default strided i.e. 0.
    if (!op.getLayout().getType().isa<Torch::NoneType>()) {
      int64_t tensorLayout;
      if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: layout must be a constant");
      else if (tensorLayout != torch_upstream::Layout::Strided)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: layout is expected to be strided");
    }

    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(
        op, op.getType(), op.getSelf(), op.getDtype(), op.getNonBlocking(),
        op.getCopy(), op.getMemoryFormat());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.to.device` op into `aten.to.dtype` op.
class DecomposeAtenToDeviceOp : public OpRewritePattern<AtenToDeviceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenToDeviceOp op,
                                PatternRewriter &rewriter) const override {

    // Device information isn't relevant to torch-mlir, so we can drop that info
    // here.
    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(
        op, op.getType(), op.getSelf(), op.getDtype(), op.getNonBlocking(),
        op.getCopy(), op.getMemoryFormat());

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.adaptive_avg_pool1d` op into `aten.avg_pool1d` op.

// The logic of this decomposition is totally same with
// the DecomposeAtenAdaptiveAvgPool2dOp, that means currently only following two
// cases are supported:
//  1. inputSize = outputSize
//  2. outputSize = 1
class DecomposeAtenAdaptiveAvgPool1dOp
    : public OpRewritePattern<AtenAdaptiveAvgPool1dOp> {
  using OpRewritePattern<AtenAdaptiveAvgPool1dOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAdaptiveAvgPool1dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op.getContext();

    Value input = op.getSelf();
    std::optional<unsigned> maybeRank = getTensorRank(input);
    if (!maybeRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned rank = *maybeRank;
    Value sizeDim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rank - 1));
    Value inputSize = rewriter.create<AtenSizeIntOp>(loc, input, sizeDim);

    Value outputShape = op.getOutputSize();
    SmallVector<Value> outputShapeSizesTorchInt;
    getListConstructElements(outputShape, outputShapeSizesTorchInt);
    Value outputSize = outputShapeSizesTorchInt[0];

    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value constantFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value constantTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);

    int64_t outputSizeInt;
    if (!matchPattern(outputSize, m_TorchConstantInt(&outputSizeInt))) {
      return rewriter.notifyMatchFailure(
          op, "the output size of adaptive_pool_1d must be a constant int");
    }

    SmallVector<Value, 1> kernelSize;
    if (outputSizeInt == 1) {
      BaseTensorType inputTensorType = input.getType().cast<BaseTensorType>();
      ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
      kernelSize.push_back(
          inputShape[rank - 1] == kUnknownSize
              ? inputSize
              : rewriter.create<Torch::ConstantIntOp>(
                    loc, rewriter.getI64IntegerAttr(inputShape[rank - 1])));
    } else {
      if (!isAssumingStrictSymbolicShapes(rewriter)) {
        Value cond = rewriter.create<AtenEqIntOp>(loc, inputSize, outputSize);
        rewriter.create<RuntimeAssertOp>(
            loc, cond,
            "unimplemented: only support cases where input and output size are "
            "equal for non-unit output size");
      }
      kernelSize.push_back(constantOne);
    }

    Value kernelSizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), kernelSize);
    Value strideList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        ValueRange{constantOne});
    Value paddingSizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        ValueRange{constantZero});

    rewriter.replaceOpWithNewOp<AtenAvgPool1dOp>(
        op, op.getType(), input, kernelSizeList, strideList, paddingSizeList,
        /*ceil_mode=*/constantFalse, /*count_include_pad=*/constantTrue);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.adaptiveAvgPool2d` op into `aten.avgPool2d` op.
//
// For AdaptiveAvgPool2d op, when the input size is an integer multiple of
// output size the kernelSize, stride and padding is calculated as follows:
// strideH = inH // outH
// strideW = inH // outH
// kernelH = inH - [(outH - 1) * strideH]
// kernelW = inW - [(outW - 1) * strideW]
// paddingH = 0, paddingW = 0
//
// For the special case, when the output size is one for all dimensions,
// the kernel size is same as the input size.
class DecomposeAtenAdaptiveAvgPool2dOp
    : public OpRewritePattern<AtenAdaptiveAvgPool2dOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAdaptiveAvgPool2dOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value input = op.getSelf();
    std::optional<unsigned> maybeRank = getTensorRank(input);
    if (!maybeRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned rank = *maybeRank;
    SmallVector<Value, 2> inputHW;
    Value dimH = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rank - 2));
    inputHW.push_back(
        /*inH=*/rewriter.create<AtenSizeIntOp>(loc, input, dimH));
    Value dimW = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rank - 1));
    inputHW.push_back(
        /*inW=*/rewriter.create<AtenSizeIntOp>(loc, input, dimW));

    Value outputShape = op.getOutputSize();
    SmallVector<Value> outputShapeSizesTorchInt;
    getListConstructElements(outputShape, outputShapeSizesTorchInt);

    // TODO: Add support for cases other than:
    // 1.) inH == outH and inW == outW.
    // 2.) outH == outW == 1
    bool unitOutputSize = true;
    for (Value outShape : outputShapeSizesTorchInt) {
      int64_t outShapeInt;
      if (!matchPattern(outShape, m_TorchConstantInt(&outShapeInt))) {
        return rewriter.notifyMatchFailure(
            op, "output size is expected to be a constant");
      }
      if (outShapeInt != 1) {
        unitOutputSize = false;
        break;
      }
    }

    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value constantFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value constantTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    SmallVector<Value, 2> kernelSize;

    for (unsigned i = 0; i < inputHW.size(); i++) {
      if (unitOutputSize) {
        BaseTensorType inputTensorType = input.getType().cast<BaseTensorType>();
        ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
        kernelSize.push_back(inputShape[rank - 2 + i] == kUnknownSize
                                 ? inputHW[i]
                                 : rewriter.create<Torch::ConstantIntOp>(
                                       loc, rewriter.getI64IntegerAttr(
                                                inputShape[rank - 2 + i])));
      } else {
        if (!isAssumingStrictSymbolicShapes(rewriter)) {
          Value cond = rewriter.create<AtenEqIntOp>(
              loc, inputHW[i], outputShapeSizesTorchInt[i]);
          rewriter.create<RuntimeAssertOp>(loc, cond,
                                           "unimplemented: only support cases "
                                           "where input and output size are "
                                           "equal for non-unit output size");
        }
        Value outMinusOne = rewriter.create<AtenSubIntOp>(
            loc, outputShapeSizesTorchInt[i], constantOne);
        kernelSize.push_back(
            rewriter.create<AtenSubIntOp>(loc, inputHW[i], outMinusOne));
      }
    }

    Value kernelSizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), kernelSize);
    // Currently we only support cases where input size is equal to the output
    // size or unit output size. For the former case, stride is always equal to
    // one and for the latter the stride value doesn't matter, since the kernel
    // size is same as the input size. Therfore, keeping the stride as one for
    // the latter case as well for the ease of implementation.
    Value strideList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        ValueRange{constantOne, constantOne});
    Value paddingSizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        ValueRange{constantZero, constantZero});

    rewriter.replaceOpWithNewOp<AtenAvgPool2dOp>(
        op, op.getType(), input, kernelSizeList, strideList, paddingSizeList,
        /*ceilMode=*/constantFalse, /*countIncludePad=*/constantTrue,
        /*divisorOverride=*/constantNone);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.clampMin` op into `aten.clamp` op.
class DecomposeAtenClampMinOp : public OpRewritePattern<AtenClampMinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenClampMinOp op,
                                PatternRewriter &rewriter) const override {
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenClampOp>(op, op.getType(), op.getSelf(),
                                             op.getMin(), /*max=*/constantNone);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.clamp_min.Tensor` op into `aten.clamp.Tensor` op.
class DecomposeAtenClampMinTensorOp
    : public OpRewritePattern<AtenClampMinTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenClampMinTensorOp op,
                                PatternRewriter &rewriter) const override {
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenClampTensorOp>(
        op, op.getType(), op.getSelf(), op.getMin(), /*max=*/constantNone);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.clampMax` op into `aten.clamp` op.
class DecomposeAtenClampMaxOp : public OpRewritePattern<AtenClampMaxOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenClampMaxOp op,
                                PatternRewriter &rewriter) const override {
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenClampOp>(op, op.getType(), op.getSelf(),
                                             /*min=*/constantNone, op.getMax());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenCosineSimilarityOp
    : public OpRewritePattern<AtenCosineSimilarityOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCosineSimilarityOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x1 = op.getX1();
    Value x2 = op.getX2();
    Value dim = op.getDim();

    // Broadcast x1 and x2 to the same shape
    SmallVector<int64_t> indexBroadcastShapeInt;
    SmallVector<Value> indexBroadcastShapeValue;
    computeBroadcastShape(rewriter, loc, x1, x2, indexBroadcastShapeInt,
                          indexBroadcastShapeValue);
    Type dtype = x1.getType().cast<BaseTensorType>().getOptionalDtype();
    Type broadcastType = ValueTensorType::get(
        op.getContext(), llvm::ArrayRef(indexBroadcastShapeInt), dtype);
    Value indexBroadcastShapeTorchList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        indexBroadcastShapeValue);
    x1 = rewriter.create<AtenBroadcastToOp>(loc, broadcastType, x1,
                                            indexBroadcastShapeTorchList);
    x2 = rewriter.create<AtenBroadcastToOp>(loc, broadcastType, x2,
                                            indexBroadcastShapeTorchList);

    // Compute the mul of A and B
    Value dotProduct =
        rewriter.create<AtenMulTensorOp>(loc, broadcastType, x1, x2);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op->getContext())),
        ValueRange{dim});
    Value sumDotProduct = rewriter.create<Torch::AtenSumDimIntListOp>(
        loc, op.getType(), /*self=*/dotProduct, /*dim=*/dimList,
        /*keepdim=*/cstFalse,
        /*dtype=*/cstNone);

    // Compute the norm of A and B
    Value ord = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(2.0));
    Value normA = rewriter.create<AtenLinalgVectorNormOp>(
        loc, op.getType(), x1, ord, dimList, /*keepdim=*/cstFalse,
        /*dtype=*/cstNone);
    Value normB = rewriter.create<AtenLinalgVectorNormOp>(
        loc, op.getType(), x2, ord, dimList, /*keepdim=*/cstFalse,
        /*dtype=*/cstNone);

    // Compute the product of the norms
    Value normProduct =
        rewriter.create<AtenMulTensorOp>(loc, op.getType(), normA, normB);
    Value normProductClamp = rewriter.create<AtenClampOp>(
        loc, op.getType(), normProduct, op.getEps(), /*max=*/cstNone);
    // Compute the final cosine similarity by division
    rewriter.replaceOpWithNewOp<AtenDivTensorOp>(
        op, op.getType(), sumDotProduct, normProductClamp);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.baddbmm` op into `aten.bmm`, `aten.mul.Scalar`, and
// `aten.add.Tensor` op.
class DecomposeAtenBaddbmmOp : public OpRewritePattern<AtenBaddbmmOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBaddbmmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bmm = rewriter.create<AtenBmmOp>(loc, op.getType(), op.getBatch1(),
                                           op.getBatch2());
    Value alphaTimesBmm =
        rewriter.create<AtenMulScalarOp>(loc, op.getType(), bmm, op.getAlpha());
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    BaseTensorType resultType =
        op->getResult(0).getType().cast<BaseTensorType>();
    if (inputType.hasDtype() && resultType.hasDtype() &&
        inputType.getDtype() != resultType.getDtype()) {
      input = convertTensorToDtype(rewriter, loc, input, resultType.getDtype());
    }
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(
        op, op.getType(), alphaTimesBmm, op.getSelf(), op.getBeta());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.floorDivide` op into `aten.div.TensorMode` op.
class DecomposeAtenFloorDivideOp : public OpRewritePattern<AtenFloorDivideOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFloorDivideOp op,
                                PatternRewriter &rewriter) const override {
    // https://pytorch.org/docs/stable/generated/torch.floorDivide.html
    // PyTorch aten.floorDivide is a misnomer because it actually rounds
    // the quotient towards zero instead of taking its floor.
    Value cstStrFloor =
        rewriter.create<Torch::ConstantStrOp>(op.getLoc(), "trunc");
    rewriter.replaceOpWithNewOp<AtenDivTensorModeOp>(
        op, op.getType(), op.getSelf(), op.getOther(),
        /*roundingMode=*/cstStrFloor);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.numpyT` op into `aten.permute` op.
class DecomposeAtenNumpyTOp : public OpRewritePattern<AtenNumpyTOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNumpyTOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    std::optional<unsigned> maybeInputRank = getTensorRank(self);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned inputRank = *maybeInputRank;

    SmallVector<Value> dimListElements;
    SmallVector<int> dimListInts(llvm::reverse(
        llvm::iota_range<int>(0, inputRank, /*inclusive=*/false)));
    for (int dimListInt : dimListInts) {
      dimListElements.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dimListInt)));
    }
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op->getContext())),
        dimListElements);
    rewriter.replaceOpWithNewOp<AtenPermuteOp>(op, op.getType(), self, dimList);
    return success();
  }
};
} // namespace

template <typename OpTy>
static LogicalResult calculateVariance(OpTy op, PatternRewriter &rewriter,
                                       bool unbiased, double correction) {
  Location loc = op.getLoc();
  Value self = op.getSelf();
  Value dimList = op.getDim();
  Value keepDim = op.getKeepdim();
  BaseTensorType inputTensorTy = self.getType().cast<BaseTensorType>();
  Type outputType = op.getType();
  BaseTensorType outputTensorType = outputType.cast<BaseTensorType>();
  if (!outputTensorType.hasDtype()) {
    return rewriter.notifyMatchFailure(op,
                                       "expected result type to have a dtype");
  }
  Type newOutputType = outputTensorType.getWithSizesAndDtype(
      outputTensorType.getSizes(), rewriter.getF64Type());
  if (!inputTensorTy.hasDtype() ||
      !inputTensorTy.getDtype().isa<mlir::FloatType>()) {
    return rewriter.notifyMatchFailure(
        op, "support floating-point type input only");
  }

  // Upcasting the input tensor to `F64` dtype for higher precision during the
  // computation of the result.
  if (inputTensorTy.getDtype().getIntOrFloatBitWidth() != 64) {
    self = convertTensorToDtype(rewriter, loc, self, rewriter.getF64Type());
    inputTensorTy = self.getType().cast<BaseTensorType>();
  }

  std::optional<unsigned> maybeInputRank = getTensorRank(self);
  if (!maybeInputRank) {
    return rewriter.notifyMatchFailure(op, "expected input to have a rank");
  }
  unsigned inputRank = *maybeInputRank;
  SmallVector<Value> dimListElements;
  bool isNoneOrEmpty = true;
  if (!dimList.getType().template isa<Torch::NoneType>()) {
    if (!getListConstructElements(dimList, dimListElements))
      return rewriter.notifyMatchFailure(
          op, "expect dimList to be constructed from list construct");
    if (!dimListElements.empty() || inputRank == 0)
      isNoneOrEmpty = false;
  }
  if (isNoneOrEmpty) {
    for (unsigned i = 0; i < inputRank; i++)
      dimListElements.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        dimListElements);
  }
  Type meanDimResultType = inputTensorTy;
  for (unsigned i = 0; i < dimListElements.size(); i++)
    meanDimResultType = computeReductionType(
        rewriter, op, meanDimResultType.cast<BaseTensorType>(),
        dimListElements[i],
        /*keepDim=*/true);

  Value constantNone = rewriter.create<ConstantNoneOp>(loc);
  Value constantTrue = rewriter.create<ConstantBoolOp>(loc, true);
  Value meanAlongDims = rewriter.create<AtenMeanDimOp>(
      loc, meanDimResultType, self, dimList, /*keepDim=*/constantTrue,
      /*dtype=*/constantNone);
  Value subMean =
      createTensorSub(rewriter, loc, inputTensorTy, self, meanAlongDims);
  Value square = rewriter.create<AtenSquareOp>(loc, inputTensorTy, subMean);

  if (!unbiased) {
    Value result = rewriter.create<AtenMeanDimOp>(
        loc, newOutputType, square, dimList, keepDim, /*dtype=*/constantNone);
    result = convertTensorToDtype(rewriter, loc, result,
                                  outputTensorType.getDtype());
    rewriter.replaceOp(op, result);
    return success();
  }
  // Divide the square sum by productDimSize - correction.
  Value squareSum = rewriter.create<AtenSumDimIntListOp>(
      loc, newOutputType, square, dimList, keepDim, /*dtype=*/constantNone);

  // `productDimSize` is product of sizes of dimensions to be reduced.
  Value constantOne =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value productDimSize = constantOne;
  for (Value dim : dimListElements) {
    Value dimSize = rewriter.create<AtenSizeIntOp>(loc, self, dim);
    productDimSize =
        rewriter.create<AtenMulIntOp>(loc, productDimSize, dimSize);
  }
  productDimSize = rewriter.create<AtenFloatScalarOp>(loc, productDimSize);
  constantOne = rewriter.create<Torch::ConstantFloatOp>(
      loc, rewriter.getF64FloatAttr(1.0));
  Value cstCorrection = rewriter.create<Torch::ConstantFloatOp>(
      loc, rewriter.getF64FloatAttr(correction));
  // The `correction` value should be less than or equal to `productDimSize +
  // 1`.
  if (!isAssumingStrictSymbolicShapes(rewriter)) {
    Value productDimSizePlusOne = rewriter.create<AtenAddOp>(
        loc, productDimSize.getType(), productDimSize, constantOne);
    Value cond = rewriter.create<AtenGeFloatOp>(loc, productDimSizePlusOne,
                                                cstCorrection);
    rewriter.create<RuntimeAssertOp>(
        loc, cond,
        "correction value should be less than or equal to productDimSize + 1");
  }
  Value productDimSizeSubCorrection =
      rewriter.create<AtenSubFloatOp>(loc, productDimSize, cstCorrection);
  Value result = rewriter.create<AtenDivScalarOp>(loc, newOutputType, squareSum,
                                                  productDimSizeSubCorrection);
  result =
      convertTensorToDtype(rewriter, loc, result, outputTensorType.getDtype());
  rewriter.replaceOp(op, result);
  return success();
}

// Decompose aten.var(x, dims) into:
// sub = aten.sub(x, aten.mean(x, dims))
// square = aten.square(sub)
// For Unbiased case:
// out = aten.sum(square, dims) / (productDimSize-1)
// For Biased case:
// out = aten.mean(square, dims)
namespace {
class DecomposeAtenVarDimOp : public OpRewritePattern<AtenVarDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarDimOp op,
                                PatternRewriter &rewriter) const override {
    bool unbiased;
    if (!matchPattern(op.getUnbiased(), m_TorchConstantBool(&unbiased))) {
      return rewriter.notifyMatchFailure(
          op, "Only support constant unbiased for aten.var");
    }
    double correction = unbiased ? 1.0 : 0.0;
    if (failed(calculateVariance<AtenVarDimOp>(op, rewriter, unbiased,
                                               correction)))
      return rewriter.notifyMatchFailure(op, "invalid variance parameters");
    return success();
  }
};
} // namespace

// Decompose aten.var(x, dims) into:
// sub = aten.sub(x, aten.mean(x, dims))
// square = aten.square(sub)
// out = aten.sum(square, dims) / (productDimSize - correction)
namespace {
class DecomposeAtenVarCorrectionOp
    : public OpRewritePattern<AtenVarCorrectionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarCorrectionOp op,
                                PatternRewriter &rewriter) const override {
    int64_t correctionValInt;
    double correctionValFloat = 1.0;
    if (!op.getCorrection().getType().isa<Torch::NoneType>()) {
      if (op.getCorrection().getType().isa<Torch::FloatType>()) {
        if (!matchPattern(op.getCorrection(),
                          m_TorchConstantFloat(&correctionValFloat)))
          return rewriter.notifyMatchFailure(
              op, "Only support constant int or float correction value for "
                  "aten.var");
      } else if (op.getCorrection().getType().isa<Torch::IntType>()) {
        if (!matchPattern(op.getCorrection(),
                          m_TorchConstantInt(&correctionValInt)))
          return rewriter.notifyMatchFailure(
              op, "Only support constant int or float correction value for "
                  "aten.var");
        correctionValFloat = (double)correctionValInt;
      } else {
        return rewriter.notifyMatchFailure(
            op, "unimplemented: correction value should be only constant int "
                "or float for aten.var");
      }
    }

    bool unbiased = correctionValFloat == 0.0 ? false : true;
    if (failed(calculateVariance<AtenVarCorrectionOp>(op, rewriter, unbiased,
                                                      correctionValFloat)))
      return rewriter.notifyMatchFailure(op, "invalid variance parameters");
    return success();
  }
};
} // namespace

namespace {
// Decompose the `aten.selectScatter` operation into `aten.sliceScatter` op.
class DecomposeAtenSelectScatterOp
    : public OpRewritePattern<AtenSelectScatterOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSelectScatterOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value start = op.getIndex();
    Value dim = op.getDim();
    Value self = op.getSelf();
    Value src = op.getSrc();

    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value startPlusOne =
        rewriter.create<AtenAddIntOp>(loc, one.getType(), start, one);

    auto unsqueezedInfo = unsqueezeTensor(rewriter, op, src, dim);
    if (failed(unsqueezedInfo)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor op");
    }
    src = *unsqueezedInfo;
    rewriter.replaceOpWithNewOp<AtenSliceScatterOp>(
        op, op.getSelf().getType(), self, src, dim, start, startPlusOne,
        /*step=*/one);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAten_EmbeddingBagOp
    : public OpRewritePattern<Aten_EmbeddingBagOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_EmbeddingBagOp op,
                                PatternRewriter &rewriter) const override {
    Value weight = op.getWeight();
    Value indices = op.getIndices();
    Value offsets = op.getOffsets();
    Value scaleGradByFreq = op.getScaleGradByFreq();
    Value mode = op.getMode();
    Value sparse = op.getSparse();
    Value perSampleWeights = op.getPerSampleWeights();
    Value includeLastOffset = op.getIncludeLastOffset();
    Value paddingIdx = op.getPaddingIdx();

    auto resultType0 = op->getResult(0).getType();
    auto resultType1 = op->getResult(1).getType();
    auto resultType2 = op->getResult(2).getType();
    auto resultType3 = op->getResult(3).getType();

    llvm::SmallVector<Type> returnTypes{resultType0, resultType1, resultType2,
                                        resultType3};

    rewriter.replaceOpWithNewOp<AtenEmbeddingBagPaddingIdxOp>(
        op, returnTypes, weight, indices, offsets, scaleGradByFreq, mode,
        sparse, perSampleWeights, includeLastOffset, paddingIdx);

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.liftFreshCopy` op into `aten.clone` op.
class DecomposeAtenLiftFreshCopyOp
    : public OpRewritePattern<AtenLiftFreshCopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLiftFreshCopyOp op,
                                PatternRewriter &rewriter) const override {
    Value constantNone = rewriter.create<ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenCloneOp>(op, op.getType(), op.getSelf(),
                                             /*memoryFormat=*/constantNone);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenMseLossOp : public OpRewritePattern<AtenMseLossOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMseLossOp op,
                                PatternRewriter &rewriter) const override {

    // The `reduction` arg would have only three valid values.
    // 0 means no reduction.
    // 1 means mean reduction.
    // 2 means sum reduction.
    int64_t reductionType;
    if (!matchPattern(op.getReduction(), m_TorchConstantInt(&reductionType)))
      return rewriter.notifyMatchFailure(
          op, "Expected a constant integer value for reduction");

    Location loc = op.getLoc();
    BaseTensorType resultType = op.getType().cast<BaseTensorType>();
    BaseTensorType inputType = op.getSelf().getType().cast<BaseTensorType>();
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "Expected the input tensor to have sizes");
    BaseTensorType subType =
        inputType
            .getWithSizesAndDtype(llvm::ArrayRef(inputType.getSizes()),
                                  resultType.getOptionalDtype())
            .cast<BaseTensorType>();

    Value sub =
        createTensorSub(rewriter, loc, subType, op.getSelf(), op.getTarget());
    Value result = rewriter.create<AtenSquareOp>(loc, subType, sub);
    if (reductionType == torch_upstream::Reduction::None) {
      rewriter.replaceOp(op, result);
      return success();
    }
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    if (reductionType == torch_upstream::Reduction::Mean)
      result = rewriter.create<AtenMeanDimOp>(loc, resultType, result,
                                              /*dim=*/cstNone,
                                              /*keepdim=*/cstFalse,
                                              /*dtype=*/cstNone);
    else
      result = rewriter.create<AtenSumDimIntListOp>(
          loc, resultType, result, /*dim=*/cstNone, /*keepdim=*/cstFalse,
          /*dtype=*/cstNone);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.norm.ScalarOpt_dim` op to `aten.linalg_vector_norm` op
class DecomposeAtenNormScalarOptDimOp
    : public OpRewritePattern<AtenNormScalarOptDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNormScalarOptDimOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value ord = op.getP();
    if (ord.getType().isa<Torch::NoneType>()) {
      ord = rewriter.create<Torch::ConstantFloatOp>(
          loc, rewriter.getF64FloatAttr(2.0));
    }
    rewriter.replaceOpWithNewOp<AtenLinalgVectorNormOp>(
        op, op.getType(), op.getSelf(), ord, op.getDim(), op.getKeepdim(),
        /*dtype=*/none);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenRandintLowOp : public OpRewritePattern<AtenRandintLowOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandintLowOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Type resultType = op.getType();
    BaseTensorType resultTensorType = resultType.cast<BaseTensorType>();
    if (!resultTensorType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }

    int64_t cstLow, cstHigh;
    if (!matchPattern(op.getLow(), m_TorchConstantInt(&cstLow)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: low must be a constant integer");
    if (!matchPattern(op.getHigh(), m_TorchConstantInt(&cstHigh)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: high must be a constant integer");

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
    Value low = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)cstLow));
    Value high = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)cstHigh));

    BaseTensorType floatResultType =
        resultTensorType
            .getWithSizesAndDtype(resultTensorType.getSizes(),
                                  rewriter.getF32Type())
            .cast<BaseTensorType>();
    Value emptyTensor = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, floatResultType, op.getSize(), /*dtype=*/none,
        /*layout=*/op.getLayout(),
        /*device=*/op.getDevice(), /*pinMemory=*/op.getPinMemory(),
        /*memoryFormat=*/none);

    Value result =
        rewriter.create<AtenUniformOp>(loc, floatResultType, emptyTensor,
                                       /*from=*/low,
                                       /*to=*/high,
                                       /*generator=*/none);
    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(
        op, resultType, result,
        getDtypeIntValueForType(rewriter, loc, resultTensorType.getDtype()),
        /*nonBlocking=*/cstFalse, /*copy=*/cstFalse, /*memoryFormat=*/none);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenRandintOp : public OpRewritePattern<AtenRandintOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandintOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Type resultType = op.getType();

    Value low = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));

    rewriter.replaceOpWithNewOp<AtenRandintLowOp>(
        op, resultType, low, op.getHigh(), op.getSize(), op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory());

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.varMean.correction` op into `aten.var.correction` and
// `aten.mean.dim` op.
class DecomposeAtenVarMeanCorrectionOp
    : public OpRewritePattern<AtenVarMeanCorrectionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarMeanCorrectionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value var = rewriter.create<AtenVarCorrectionOp>(
        loc, op.getType(0), op.getSelf(), op.getDim(), op.getCorrection(),
        op.getKeepdim());
    Value mean = rewriter.create<AtenMeanDimOp>(
        loc, op.getType(0), op.getSelf(), op.getDim(), op.getKeepdim(),
        /*dtype=*/noneVal);
    rewriter.replaceOp(op, {var, mean});
    return success();
  }
};
} // namespace

namespace {
// Decompose `prims.convertElementType` op into `aten.to.dtype` op.
class DecomposePrimsConvertElementTypeOp
    : public OpRewritePattern<PrimsConvertElementTypeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimsConvertElementTypeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(
        op, op.getType(), op.getA(), op.getDtype(), /*nonBlocking=*/cstFalse,
        /*copy=*/cstFalse, /*memoryFormat=*/cstNone);
    return success();
  }
};
} // namespace

namespace {
// Decompose `prims.var` op into `aten.var.correction` op.
class DecomposePrimsVarOp : public OpRewritePattern<PrimsVarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimsVarOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getOutputDtype().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for prims::var op");
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<AtenVarCorrectionOp>(
        op, op.getType(), op.getInp(), op.getDims(), op.getCorrection(),
        /*keepdim=*/cstFalse);
    return success();
  }
};
} // namespace

namespace {
// Decompose `prims.sqrt` op into `aten.sqrt` op.
class DecomposePrimsSqrtOp : public OpRewritePattern<PrimsSqrtOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimsSqrtOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenSqrtOp>(op, op.getType(), op.getSelf());
    return success();
  }
};
} // namespace

namespace {
// The op is decomposed using the Box-Muller transform.
// Refer: https://en.wikipedia.org/wiki/Box-Muller_transform
class DecomposeAtenRandnGeneratorOp
    : public OpRewritePattern<AtenRandnGeneratorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandnGeneratorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = op.getType().cast<BaseTensorType>();

    if (!resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }

    Value dtype = getDtypeIntValueForType(rewriter, loc, resultType.getDtype());
    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value low = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)0.0));
    Value high = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)1.0));
    Value cstMinusTwo = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)-2.0));
    Value cstTwoPie = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)(2.0 * 3.14159)));

    Value emptyTensorA = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, resultType, op.getSize(), /*dtype=*/dtype,
        /*layout=*/op.getLayout(),
        /*device=*/op.getDevice(), /*pin_memory=*/op.getPinMemory(),
        /*memory_format=*/none);
    Value emptyTensorB = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, resultType, op.getSize(), /*dtype=*/dtype,
        /*layout=*/op.getLayout(),
        /*device=*/op.getDevice(), /*pin_memory=*/op.getPinMemory(),
        /*memory_format=*/none);

    Value uOne =
        rewriter.create<AtenUniformOp>(loc, resultType, emptyTensorA,
                                       /*from=*/low,
                                       /*to=*/high,
                                       /*generator=*/op.getGenerator());
    Value uTwo =
        rewriter.create<AtenUniformOp>(loc, resultType, emptyTensorB,
                                       /*from=*/low,
                                       /*to=*/high,
                                       /*generator=*/op.getGenerator());

    Value logUOne = rewriter.create<AtenLogOp>(loc, resultType, uOne);
    Value minusTwoLogUOne =
        rewriter.create<AtenMulScalarOp>(loc, resultType, logUOne, cstMinusTwo);
    Value r = rewriter.create<AtenSqrtOp>(loc, resultType, minusTwoLogUOne);
    Value theta =
        rewriter.create<AtenMulScalarOp>(loc, resultType, uTwo, cstTwoPie);
    Value cosTheta = rewriter.create<AtenCosOp>(loc, resultType, theta);
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, op.getType(), r, cosTheta);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.randn` op into `aten.randn.generator` op.
class DecomposeAtenRandnOp : public OpRewritePattern<AtenRandnOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandnOp op,
                                PatternRewriter &rewriter) const override {
    Value none = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenRandnGeneratorOp>(
        op, op.getType(), op.getSize(), /*generator=*/none, op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.randn_like` op into `aten.randn.generator` op.
class DecomposeAtenRandnLikeOp : public OpRewritePattern<AtenRandnLikeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandnLikeOp op,
                                PatternRewriter &rewriter) const override {
    // Only `none`, `contiguous` and `preserve` memory_format is supported.
    if (!op.getMemoryFormat().getType().isa<Torch::NoneType>()) {
      int64_t memoryFormat;
      if (!matchPattern(op.getMemoryFormat(),
                        m_TorchConstantInt(&memoryFormat)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the memory format should be specified in "
                "an integer constant");
      if (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
          memoryFormat != torch_upstream::MemoryFormat::Preserve)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only none, contiguous and preserve "
                "memory_format is supported");
    }
    Value none = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    auto sizeListType =
        Torch::ListType::get(Torch::IntType::get(op.getContext()));
    Value sizeList =
        rewriter.create<AtenSizeOp>(op.getLoc(), sizeListType, op.getSelf());
    rewriter.replaceOpWithNewOp<AtenRandnGeneratorOp>(
        op, op.getType(), sizeList, /*generator=*/none, op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenRandOp : public OpRewritePattern<AtenRandOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto resultType = op.getType().cast<BaseTensorType>();

    if (!resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Value noneVal = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value low = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)0.0));
    Value high = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)1.0));
    Value emptyTensor = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, resultType, op.getSize(), /*dtype=*/op.getDtype(),
        /*layout=*/op.getLayout(),
        /*device=*/op.getDevice(), /*pin_memory=*/op.getPinMemory(),
        /*memory_format=*/noneVal);
    rewriter.replaceOpWithNewOp<AtenUniformOp>(op, resultType, emptyTensor,
                                               /*from=*/low,
                                               /*to=*/high,
                                               /*generator=*/noneVal);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenVarMeanOp : public OpRewritePattern<AtenVarMeanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarMeanOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value var = rewriter.create<AtenVarDimOp>(loc, op.getType(0), op.getSelf(),
                                              /*dim=*/noneVal, op.getUnbiased(),
                                              /*keepdim=*/falseVal);
    Value mean = rewriter.create<AtenMeanOp>(loc, op.getType(0), op.getSelf(),
                                             /*dtype=*/noneVal);
    rewriter.replaceOp(op, {var, mean});
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNewEmptyStridedOp
    : public OpRewritePattern<AtenNewEmptyStridedOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNewEmptyStridedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value opSize = op.getSize();
    Value opStride = op.getStride();

    if (failed(checkDefaultStrideHelper(op, rewriter, opSize, opStride, loc)))
      return rewriter.notifyMatchFailure(
          op, "Unable to determine if stride is default");

    rewriter.replaceOpWithNewOp<AtenNewEmptyOp>(
        op, op.getType(), op.getSelf(), op.getSize(), op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory());

    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenEmptyStridedOp
    : public OpRewritePattern<AtenEmptyStridedOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEmptyStridedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value opSize = op.getSize();
    Value opStride = op.getStride();

    if (failed(checkDefaultStrideHelper(op, rewriter, opSize, opStride, loc)))
      return rewriter.notifyMatchFailure(
          op, "Unable to determine if stride is default");

    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());

    rewriter.replaceOpWithNewOp<AtenEmptyMemoryFormatOp>(
        op, op.getType(), op.getSize(), op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory(), /*memoryFormat=*/noneVal);
    return success();
  }
};
} // namespace

namespace {
class DecomposePrimsSqueezeOp : public OpRewritePattern<PrimsSqueezeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimsSqueezeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getA();
    SmallVector<int64_t> dimensions;
    if (!matchPattern(op.getDimensions(),
                      m_TorchListOfConstantInts(dimensions)))
      return rewriter.notifyMatchFailure(
          op, "all dimensions must be constant ints");

    std::sort(dimensions.rbegin(), dimensions.rend());

    if (dimensions.size() == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }
    Value result = input;
    for (unsigned i = 0; i < dimensions.size(); i++) {
      auto squeezeTensorInfo =
          squeezeTensor(rewriter, op, loc, dimensions[i], result);
      if (failed(squeezeTensorInfo)) {
        return rewriter.notifyMatchFailure(op,
                                           "cannot generate unsqueeze tensor");
      }
      result = *squeezeTensorInfo;
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenMovedimIntOp : public OpRewritePattern<AtenMovedimIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMovedimIntOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(
          op, "expected input tensor to have a rank");
    }
    unsigned inputRank = *maybeInputRank;
    if (inputRank <= 1) {
      rewriter.replaceOp(op, input);
      return success();
    }

    int64_t srcDimInt, dstDimInt;
    if (matchPattern(op.getSource(), m_TorchConstantInt(&srcDimInt))) {
      srcDimInt = toPositiveDim(srcDimInt, inputRank);
      if (!isValidDim(srcDimInt, inputRank))
        return rewriter.notifyMatchFailure(op, "source is not a valid dim");
    } else {
      return rewriter.notifyMatchFailure(op, "source is not a constant int");
    }
    if (matchPattern(op.getDestination(), m_TorchConstantInt(&dstDimInt))) {
      dstDimInt = toPositiveDim(dstDimInt, inputRank);
      if (!isValidDim(dstDimInt, inputRank))
        return rewriter.notifyMatchFailure(op,
                                           "destination is not a valid dim");
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "destination is not a constant int");
    }

    SmallVector<int64_t> dimsOrder =
        computeDimsOrderForMoveDim(srcDimInt, dstDimInt, inputRank);
    SmallVector<Value> cstDimsOrder;
    for (int64_t dim : dimsOrder)
      cstDimsOrder.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dim)));
    Value permuteDimsOrder = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op->getContext())),
        cstDimsOrder);
    rewriter.replaceOpWithNewOp<AtenPermuteOp>(op, op.getType(), input,
                                               permuteDimsOrder);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenCrossEntropyLossOp
    : public OpRewritePattern<AtenCrossEntropyLossOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCrossEntropyLossOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value target = op.getTarget();
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: unranked input tensor");
    unsigned selfRank = maybeRank.value();
    maybeRank = getTensorRank(target);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: unranked target tensor");
    unsigned targetRank = maybeRank.value();

    // When the input is 2-d i.e. of the form [minibatch, C] and target is 1-d
    // of the form [minibatch] the cross entropy loss decomposes to the
    // combination of softmax and nll loss as follows:
    // cross_entropy_loss = NLLLoss(LogSoftmax(input, dim=1), target)
    // Currently, we only support the above-mentioned case.
    if (selfRank != 2 || targetRank != 1) {
      return rewriter.notifyMatchFailure(
          op,
          "unimplemented: only support cases with 2-d input and 1-d target");
    }

    // TODO: Add support for label_smoothing value other than 0.0 (default
    // value).
    double labelSmoothing;
    if (!matchPattern(op.getLabelSmoothing(),
                      m_TorchConstantFloat(&labelSmoothing))) {
      return rewriter.notifyMatchFailure(
          op, "Only support constant float label_smoothing value");
    } else if (labelSmoothing != 0.0) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: only support default "
                                         "value of 0.0 for label_smoothing");
    }

    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value dim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value logSoftmax = rewriter.create<AtenLogSoftmaxIntOp>(
        loc, self.getType(), self, dim, /*dtype=*/noneVal);
    Value nllLoss =
        rewriter
            .create<AtenNllLossForwardOp>(
                loc, op.getType(), target.getType(), logSoftmax, target,
                op.getWeight(), op.getReduction(), op.getIgnoreIndex())
            ->getResult(0);
    rewriter.replaceOp(op, nllLoss);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenOneHotOp : public OpRewritePattern<AtenOneHotOp> {
  using OpRewritePattern<AtenOneHotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenOneHotOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();

    Value input = op.getSelf();
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "input tensor should have known sizes.");
    int64_t inputRank = inputType.getSizes().size();
    int64_t numClasses;
    if (!matchPattern(op.getNumClasses(), m_TorchConstantInt(&numClasses)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: num_classes must be constant");
    Value none = rewriter.create<ConstantNoneOp>(loc);

    // arange tensor
    auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);
    auto arangeType =
        ValueTensorType::get(context, llvm::ArrayRef(numClasses), si64Type);
    Value arangeTensor = rewriter.create<AtenArangeOp>(
        loc, arangeType, op.getNumClasses(), /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none);

    // unsqueeze input
    llvm::SmallVector<int64_t> unsqueezeShape(inputType.getSizes());
    unsqueezeShape.push_back(1);
    auto unsqueezeType =
        ValueTensorType::get(context, unsqueezeShape, si64Type);
    Value unsqueezeTensor = rewriter.create<AtenUnsqueezeOp>(
        loc, unsqueezeType, input,
        rewriter.create<ConstantIntOp>(loc,
                                       rewriter.getI64IntegerAttr(inputRank)));

    // compare
    auto eqType = ValueTensorType::get(
        context, op.getType().cast<BaseTensorType>().getSizes(),
        IntegerType::get(context, 1));
    Value eqTensor = rewriter.create<AtenEqTensorOp>(
        loc, eqType, unsqueezeTensor, arangeTensor);

    // convert to si64
    Value result = convertTensorToDtype(rewriter, loc, eqTensor, si64Type);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

namespace {
// Decompose `aten.var_mean.dim` op into `aten.var.dim` and
// `aten.mean.dim` op.
class DecomposeAtenVarMeanDimOp : public OpRewritePattern<AtenVarMeanDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarMeanDimOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value var = rewriter.create<AtenVarDimOp>(loc, op.getType(0), op.getSelf(),
                                              op.getDim(), op.getUnbiased(),
                                              op.getKeepdim());
    Value mean = rewriter.create<AtenMeanDimOp>(
        loc, op.getType(0), op.getSelf(), op.getDim(), op.getKeepdim(),
        /*dtype=*/noneVal);
    rewriter.replaceOp(op, {var, mean});
    return success();
  }
};
} // namespace

namespace {
// decompose aten.scalar_tensor to prim.NumToTensor.Scalar and
// aten.to.dtype_layout
class DecomposeAtenScalarTensor : public OpRewritePattern<AtenScalarTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenScalarTensorOp op,
                                PatternRewriter &rewriter) const override {

    auto resultTy = op.getResult().getType().cast<BaseTensorType>();
    auto scalarTy = getBuiltInTypeForTorchScalar(op.getS().getType());
    Value numToTensor = rewriter.create<PrimNumToTensorScalarOp>(
        op.getLoc(),
        resultTy.getWithSizesAndDtype(resultTy.getOptionalSizes(), scalarTy),
        op.getS());

    Value cstNone = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    Value dtype =
        getDtypeIntValueForType(rewriter, op.getLoc(), resultTy.getDtype());
    Value toDTypeLayout = rewriter.create<AtenToDtypeLayoutOp>(
        op.getLoc(), op.getType(), numToTensor, dtype, op.getLayout(),
        op.getDevice(), op.getPinMemory(), /*non_blocking=*/cstFalse,
        /*copy=*/cstFalse, /*memory_format=*/cstNone);
    rewriter.replaceOp(op, toDTypeLayout);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.topk` op into `aten.sort` and `aten.slice.Tensor` op.
class DecomposeAtenTopkOp : public OpRewritePattern<AtenTopkOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTopkOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();

    bool sorted;
    if (!matchPattern(op.getSorted(), m_TorchConstantBool(&sorted)))
      return rewriter.notifyMatchFailure(
          op, "Expected a constant boolean value for sorted");
    if (!sorted)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: sorted value arg must be set to True");

    Value self = op.getSelf();
    Value dim = op.getDim();
    auto selfType = self.getType().cast<BaseTensorType>();
    auto sortIndicesType = selfType.getWithSizesAndDtype(
        selfType.getOptionalSizes(),
        IntegerType::get(context, 64, IntegerType::Signed));
    auto sortOpResult = rewriter.create<AtenSortOp>(
        loc, self.getType(), sortIndicesType, self, dim,
        /*descending=*/op.getLargest());
    Value start = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value step = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value resultValue = rewriter.create<AtenSliceTensorOp>(
        loc, op->getResultTypes()[0], sortOpResult->getResult(0), dim, start,
        /*end=*/op.getK(), step);
    Value resultIndices = rewriter.create<AtenSliceTensorOp>(
        loc, op->getResultTypes()[1], sortOpResult->getResult(1), dim, start,
        /*end=*/op.getK(), step);
    rewriter.replaceOp(op, {resultValue, resultIndices});
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.scatter.value` op into `aten.scatter.src` op.
class DecomposeAtenScatterValueOp
    : public OpRewritePattern<AtenScatterValueOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenScatterValueOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    Value self = op.getSelf();
    Value index = op.getIndex();
    std::optional<unsigned> maybeIndexRank = getTensorRank(index);
    if (!maybeIndexRank) {
      return rewriter.notifyMatchFailure(
          op, "expected index tensor to have a rank");
    }
    unsigned indexRank = *maybeIndexRank;
    SmallVector<Value> sizes;
    for (int64_t i = 0; i < indexRank; ++i) {
      Value dim =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i));
      sizes.push_back(rewriter.create<AtenSizeIntOp>(loc, index, /*dim=*/dim));
    }
    Value sizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), sizes);

    auto selfType = self.getType().cast<BaseTensorType>();
    auto indexType = index.getType().cast<BaseTensorType>();
    BaseTensorType srcType =
        selfType
            .getWithSizesAndDtype(indexType.getOptionalSizes(),
                                  selfType.getOptionalDtype())
            .cast<BaseTensorType>();
    Value src =
        createInitTensor(rewriter, loc, srcType, op.getValue(), sizeList);
    rewriter.replaceOpWithNewOp<AtenScatterSrcOp>(op, op.getType(), self,
                                                  op.getDim(), index, src);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.sign` op into comparisons and aten.where.
class DecomposeAtenSignOp : public OpRewritePattern<AtenSignOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSignOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outType = op.getType().dyn_cast<BaseTensorType>();
    if (!outType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types input are currently supported");

    auto zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    auto one =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    auto minusOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(-1.0));

    auto compTy = outType.getWithSizesAndDtype(outType.getOptionalSizes(),
                                               rewriter.getI1Type());

    auto greater =
        rewriter.create<AtenGtScalarOp>(loc, compTy, op.getSelf(), zero);
    auto greaterEqual =
        rewriter.create<AtenGeScalarOp>(loc, compTy, op.getSelf(), zero);

    // Pseudo code:
    // if (in >= 0)
    //   if (in > 0)
    //     return 1
    //   else
    //     return 0
    // else
    //   return -1
    auto selectGreater =
        rewriter.create<AtenWhereScalarOp>(loc, outType, greater, one, zero);

    rewriter.replaceOpWithNewOp<AtenWhereScalarOtherOp>(
        op, outType, greaterEqual, selectGreater, minusOne);
    return success();
  }
};
} // namespace

namespace {
// Unconditionally decompose `torch.type_as` into `prim.dtype` +
// `torch.to.dtype`.
class DecomposeAtenTypeAsOp : public OpRewritePattern<AtenTypeAsOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTypeAsOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getSelf();
    auto other = op.getOther();
    Location loc = op.getLoc();

    Value targetDtype = rewriter.create<Torch::PrimDtypeOp>(loc, other);
    Value nonBlocking = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value copy = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value memoryFormat = rewriter.create<Torch::ConstantNoneOp>(loc);
    rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
        op, op.getType(), input, targetDtype, nonBlocking, copy, memoryFormat);
    return success();
  }
};
} // namespace

// AtenIndexTensorOp
namespace {
// The goal of this pattern is to eliminate none index in aten.Index.Tensor's
// `indices` param for the ease of various backend. The detailed steps are:
//    1. reorder input tensor so that the non-none index appears at adjacent
//    positions.
//    2. manually generate index tensor with some ops like iota, to replace the
//    none index in `indices`
//    3. replace the old aten.Index.Tensor with a new
//    aten.Index.Tensor_hacked_twin.
class DecomposeAtenIndexTensorOp : public OpRewritePattern<AtenIndexTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // TODO: It might be better to use aten.view op instead of mulitple
  // aten.unsqueeze. But currently, torch-to-linalg pass has limited support for
  // view on dynamic shapes, such as [?] -> [?,1,1,1]. Using aten.view op will
  // cause relevant e2e tests fail.
  static FailureOr<Value>
  unsqueezeTensorAtTrailingDim(Operation *op, PatternRewriter &rewriter,
                               Value input, int count) {
    Location loc = op->getLoc();
    Value constMinusOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(-1));
    Value result = input;
    while (count--) {
      auto unsqzTensorInfo =
          unsqueezeTensor(rewriter, op, result, /*dim=*/constMinusOne);
      if (failed(unsqzTensorInfo)) {
        return failure();
      }

      result = *unsqzTensorInfo;
    }
    return result;
  }

  static Value createIndexToReplaceNone(Operation *op,
                                        PatternRewriter &rewriter, Value input,
                                        int dimInt, int64_t dimSize) {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
    auto int64Dtype = getDtypeIntValueForType(
        rewriter, loc,
        rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true));

    auto resultType = ValueTensorType::get(
        context, {dimSize},
        rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true));
    auto dim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(dimInt));
    auto end = rewriter.create<Torch::AtenSizeIntOp>(loc, input, dim);
    auto v = rewriter.create<Torch::AtenArangeOp>(
        loc, resultType, /*end=*/end, /*dtype=*/int64Dtype, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none);
    return v;
  }

  LogicalResult matchAndRewrite(AtenIndexTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get elements of `indices`");

    auto input = op.getSelf();
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "only input with shape information is supported");
    }
    auto inputSizes = inputType.getSizes();
    int64_t inputRank = inputSizes.size();
    auto outputType = op.getType().cast<BaseTensorType>();
    if (!outputType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "only output with shape information is supported");
    }
    auto outputRank = outputType.getSizes().size();

    auto isTensor = [](Value v) {
      return v.getType().isa<Torch::BaseTensorType>();
    };

    // directly replace aten.Index.Tensor with aten.index.Tensor_hacked_twin
    if (llvm::all_of(indices, isTensor)) {
      if (indices.size() == 0) {
        return rewriter.notifyMatchFailure(
            op, "the indices is empty, it should be folded as a nop");
      }
      // By default, we regard the first index type as the list element type.
      auto indexElemType = indices[0]
                               .getType()
                               .template cast<BaseTensorType>()
                               .getWithSizesAndDtype(std::nullopt, nullptr);
      auto newIndex = rewriter.create<PrimListConstructOp>(
          loc, Torch::ListType::get(indexElemType), indices);
      rewriter.replaceOpWithNewOp<AtenIndexTensorHackedTwinOp>(op, op.getType(),
                                                               input, newIndex);
      return success();
    }

    SmallVector<bool> indexUsed =
        llvm::to_vector(llvm::map_range(indices, isTensor));
    for (int64_t i = indices.size(); i < inputRank; ++i)
      indexUsed.emplace_back(false);
    bool indexIsConsecutive = true;
    int64_t firstUsedIndex = -1;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indexUsed[i] && firstUsedIndex == -1) {
        firstUsedIndex = i;
      } else if (indexUsed[i] && !indexUsed[i - 1]) {
        indexIsConsecutive = false;
        break;
      }
    }

    // use aten.permute to reorder the input
    Value newInput;
    // `dims` stores the mapping from new index to the old index of input
    // tensor.
    SmallVector<int64_t> dims;
    if (!indexIsConsecutive) {
      SmallVector<Value> dimValues;
      SmallVector<int64_t> permutedSizes;
      for (int i = 0; i < inputRank; i++) {
        if (indexUsed[i]) {
          dims.emplace_back(i);
          dimValues.emplace_back(rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i)));
          permutedSizes.emplace_back(inputSizes[i]);
        }
      }
      for (int i = 0; i < inputRank; i++) {
        if (!indexUsed[i]) {
          dims.emplace_back(i);
          dimValues.emplace_back(rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i)));
          permutedSizes.emplace_back(inputSizes[i]);
        }
      }
      auto dimValueList = rewriter.create<Torch::PrimListConstructOp>(
          loc, Torch::ListType::get(Torch::IntType::get(context)), dimValues);
      newInput = rewriter.create<Torch::AtenPermuteOp>(
          loc,
          inputType.getWithSizesAndDtype(permutedSizes,
                                         inputType.getOptionalDtype()),
          input, dimValueList);
    } else {
      newInput = input;
      for (int i = 0; i < inputRank; i++) {
        dims.emplace_back(i);
      }
    }

    // manually generate new indices.
    SmallVector<Value> listElements(inputRank);

    int64_t trailingDimCnt = 0;
    int64_t i;
    // handle trailing none index.
    for (i = inputRank - 1; i >= 0; --i) {
      int64_t oldI = dims[i];
      if (indexUsed[oldI])
        break;
      Value v =
          createIndexToReplaceNone(op, rewriter, newInput, i, inputSizes[oldI]);
      auto vInfo =
          unsqueezeTensorAtTrailingDim(op, rewriter, v, trailingDimCnt);
      if (failed(vInfo)) {
        return rewriter.notifyMatchFailure(op, "failed to unsqueeze tensor");
      }
      listElements[i] = *vInfo;
      trailingDimCnt++;
    }
    // handle non-none index in between.
    for (; i >= 0; --i) {
      int64_t oldI = dims[i];
      if (!indexUsed[oldI])
        break;
      auto vInfo = unsqueezeTensorAtTrailingDim(op, rewriter, indices[oldI],
                                                trailingDimCnt);
      if (failed(vInfo)) {
        return rewriter.notifyMatchFailure(op, "failed to unsqueeze tensor");
      }
      listElements[i] = *vInfo;
    }

    // handle possible leading none dimensions.
    for (; i >= 0; --i) {
      int64_t oldI = dims[i];
      if (indexUsed[oldI]) {
        return rewriter.notifyMatchFailure(
            op, "the indices are still unconsecutive after reordering input "
                "tensor");
      }
      Value v =
          createIndexToReplaceNone(op, rewriter, newInput, i, inputSizes[oldI]);
      auto vInfo =
          unsqueezeTensorAtTrailingDim(op, rewriter, v, outputRank - 1 - i);
      if (failed(vInfo)) {
        return rewriter.notifyMatchFailure(op, "failed to unsqueeze tensor");
      }
      listElements[i] = *vInfo;
    }

    auto listElemType = ValueTensorType::get(context, std::nullopt, nullptr);
    auto newIndexList = rewriter.create<Torch::PrimListConstructOp>(
        loc, Torch::ListType::get(listElemType), listElements);
    rewriter.replaceOpWithNewOp<Torch::AtenIndexTensorHackedTwinOp>(
        op, op.getType(), newInput, newIndexList);
    return success();
  }
};
} // namespace

namespace {
// Unconditionally decompose `aten.tile` into `aten.repeat`.
class DecomposeAtenTileOp : public OpRewritePattern<AtenTileOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTileOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getSelf();
    auto repeats = op.getDims();
    SmallVector<Value> dimsElements;
    if (!getListConstructElements(repeats, dimsElements)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get elements of `dims` param");
    }
    auto dimsSize = dimsElements.size();
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "only support input tensor with shape information");
    }
    auto inputRank = inputType.getSizes().size();
    if (dimsSize < inputRank) {
      auto constantOne = rewriter.create<Torch::ConstantIntOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(1));
      for (auto i = dimsSize; i < inputRank; ++i) {
        dimsElements.insert(dimsElements.begin(), constantOne);
      }
      repeats = rewriter.create<Torch::PrimListConstructOp>(
          op.getLoc(),
          Torch::ListType::get(Torch::IntType::get(op.getContext())),
          dimsElements);
    }
    rewriter.replaceOpWithNewOp<Torch::AtenRepeatOp>(op, op.getType(), input,
                                                     repeats);
    return success();
  }
};
} // namespace

namespace {
// Unconditionally decompose `aten.reshape_as` into `aten.size` +
// `aten.reshape`.
class DecomposeAtenReshapeAsOp : public OpRewritePattern<AtenReshapeAsOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenReshapeAsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = op.getSelf();
    Value other = op.getOther();

    auto otherShape = rewriter.create<Torch::AtenSizeOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), other);
    rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(op, op.getType(), input,
                                                      otherShape);
    return success();
  }
};
} // namespace

namespace {
class DecomposeComplexOpsPass
    : public DecomposeComplexOpsBase<DecomposeComplexOpsPass> {
private:
  llvm::StringSet<> legalOpsSet;

  template <typename DecomposePattern>
  void addPatternIfTargetOpIsIllegal(RewritePatternSet &patterns) {
    MLIRContext *context = &getContext();
    std::optional<OperationName> opName =
        DecomposePattern(context).getRootKind();
    // Because the `DecomposeComplexOpsPass` uses a greedy algorithm
    // to apply patterns, only patterns that we for sure know we want to run
    // must be added. This restricts the set of patterns allowed in this file to
    // patterns that apply to a single op. In other words, patterns that match
    // on `Operation *` are not allowed, since there is no way of telling if
    // that pattern will match on an op in the `legalOpsSet` or not.
    assert(opName && "All decomposition patterns must target a single op");
    if (!legalOpsSet.contains(opName->getStringRef().ltrim(kTorchOpPrefix)))
      patterns.add<DecomposePattern>(context);
  }

public:
  DecomposeComplexOpsPass() = default;
  DecomposeComplexOpsPass(ArrayRef<std::string> legalOps) {
    this->legalOps = legalOps;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // The strings in the `legalOps` ArrayRef don't exist during the call to the
    // constructor `DecomposeComplexOpsPass`, so the creation of the
    // `legalOpsSet` must be delayed to when `runOnOperation` gets called.
    legalOpsSet.clear();
    legalOpsSet.insert(legalOps.begin(), legalOps.end());

    addPatternIfTargetOpIsIllegal<DecomposeAtenSoftmaxIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_SoftmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_LogSoftmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLogSoftmaxIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEmptyLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorAllocLikeOp<AtenOnesLikeOp, 1>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorAllocLikeOp<AtenZerosLikeOp, 0>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStackOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRollOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRepeatOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenExpandOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFlattenUsingIntsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenUnflattenIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarOtherOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarSelfOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNanToNumOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMaskedFillScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSizeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenReshapeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_SoftmaxBackwardDataOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTanhBackwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAddmmOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMeanOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMeanDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAMinMaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSelectIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMatmulOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMvOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenPixelShuffleOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_LogSoftmaxBackwardDataOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenAddCLikeOp<AtenAddcmulOp, AtenMulTensorOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenAddCLikeOp<AtenAddcdivOp, AtenDivTensorOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenInstanceNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLayerNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNativeLayerNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenGroupNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNativeGroupNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNativeBatchNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAten_ConvolutionLikeOp<Aten_ConvolutionOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAten_ConvolutionLikeOp<Aten_ConvolutionDeprecatedOp>>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvolutionBackwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvTranspose2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenArangeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenArangeStartOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenArgMinMaxOp<AtenArgmaxOp, AtenMaxDimOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenArgMinMaxOp<AtenArgminOp, AtenMinDimOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSquareOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_UnsafeViewOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_ReshapeAliasOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBernoulliOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenBernoulliLikeOp<ValsemVariantAtenBernoulliFloatOp>>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenBernoulliLikeOp<AtenBernoulliPOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBernoulliTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenExponentialOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenZeroOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEyeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEyeMOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIsnanOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIsinfOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIsneginfOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIsposinfOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHardsigmoidOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRelu6Op>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEinsumOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTraceOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHardswishOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSoftplusOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSiluOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorNewLikeOp<AtenNewZerosOp, AtenZerosOp>>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorNewLikeOp<AtenNewOnesOp, AtenOnesOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHardtanhOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFullOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLinearOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMishOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFullLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNewFullOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexPutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenExpandAsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_ToCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenDropoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DeomposeAtenNativeDropoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNewEmptyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexPutHackedTwinOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_UnsafeIndexPutHackedTwinOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenPadOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenToDtypeLayoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenToDeviceOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAdaptiveAvgPool1dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAdaptiveAvgPool2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMinOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMinTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCosineSimilarityOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBaddbmmOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFloorDivideOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNumpyTOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSelectScatterOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNarrowOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNarrowTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenGluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_EmbeddingBagOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLiftFreshCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMseLossOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNormScalarOptDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandintOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandintLowOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarMeanCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimsConvertElementTypeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimsVarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimsSqrtOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandnOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandnGeneratorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandnLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNormalFunctionalOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarMeanOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSeluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLeakyReluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLeakyReluBackwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLerpScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNewEmptyStridedOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEmptyStridedOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBucketizeTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimsSqueezeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMovedimIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenOneHotOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCrossEntropyLossOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarMeanDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTopkOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenScalarTensor>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenScatterValueOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSignOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTypeAsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTileOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenReshapeAsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTriuOp>(patterns);
    // More specific conv ops
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvTbcOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConv1dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConv2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConv3dOp>(patterns);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createDecomposeComplexOpsPass(
    ArrayRef<std::string> legalOps) {
  return std::make_unique<DecomposeComplexOpsPass>(legalOps);
}
