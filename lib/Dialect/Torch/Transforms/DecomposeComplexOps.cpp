//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
#include <set>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Helper function to check whether the `dtype` is None or Float type.
static bool isNoneOrFloatDtype(MLIRContext *context, Value dtype) {
  if (isa<Torch::NoneType>(dtype.getType()))
    return true;
  int64_t dtypeInt;
  if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    return false;
  FailureOr<Type> resDtype =
      getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
  if (failed(resDtype))
    return false;
  return isa<mlir::FloatType>(*resDtype);
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

  Type resultType = tensorType.getWithSizesAndDtypeAndSparsity(
      !tensorType.hasSizes() ? std::optional<ArrayRef<int64_t>>()
                             : llvm::ArrayRef(sizes),
      tensorType.getOptionalDtype(), tensorType.getOptionalSparsity());
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
      rewriter, op, cast<BaseTensorType>(input.getType()), dim, keepDim);
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
  BaseTensorType valueType = cast<BaseTensorType>(computeReductionType(
      rewriter, op, cast<BaseTensorType>(input.getType()), dim, keepDim));
  if (!valueType)
    return nullptr;
  BaseTensorType indexType =
      cast<BaseTensorType>(valueType.getWithSizesAndDtype(
          !valueType.hasSizes() ? std::optional<ArrayRef<int64_t>>()
                                : llvm::ArrayRef(valueType.getSizes()),
          IntegerType::get(op->getContext(), 64, IntegerType::Signed)));
  return rewriter
      .create<AtenMaxDimOp>(loc, valueType, indexType, input, dim, keepDimCst)
      .getValues();
}

// Reduction function to calculate min along given `dim`.
static Value createMinAlongDimension(PatternRewriter &rewriter, Location loc,
                                     Operation *op, Value input, Value dim,
                                     bool keepDim) {
  Value keepDimCst = rewriter.create<ConstantBoolOp>(loc, keepDim);
  BaseTensorType valueType = cast<BaseTensorType>(computeReductionType(
      rewriter, op, cast<BaseTensorType>(input.getType()), dim, keepDim));
  if (!valueType)
    return nullptr;
  BaseTensorType indexType =
      cast<BaseTensorType>(valueType.getWithSizesAndDtype(
          !valueType.hasSizes() ? std::optional<ArrayRef<int64_t>>()
                                : llvm::ArrayRef(valueType.getSizes()),
          IntegerType::get(op->getContext(), 64, IntegerType::Signed)));
  return rewriter
      .create<AtenMinDimOp>(loc, valueType, indexType, input, dim, keepDimCst)
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

static bool
rewriteEquationWithEllipsisSlicing(std::string &equation,
                                   SmallVector<int64_t> &inputRanks) {
  // split equation into input and result
  size_t arrowPos = equation.find("->");
  if (arrowPos == std::string::npos) {
    return false;
  }
  std::string inputStr = equation.substr(0, arrowPos);
  std::string resultStr = equation.substr(arrowPos + 2);

  // split input into tokens
  SmallVector<std::string> inputTokens;
  size_t start = 0;
  size_t end = 0;
  std::set<char> usedTokens;
  while (end < inputStr.size()) {
    end = inputStr.find(",", start);
    if (end == std::string::npos) {
      end = inputStr.size();
    }
    std::string token = inputStr.substr(start, end - start);
    inputTokens.push_back(token);
    start = end + 1;
  }
  if (inputTokens.size() != inputRanks.size()) {
    return false;
  }

  // find the rank which ellipsis represents, and max ellipsis rank because a
  // tensor can be broadcasted
  SmallVector<int64_t> ellipsisRanks;
  int maxEllipsisRank = 0;
  for (const auto &[token, inputRank] : llvm::zip(inputTokens, inputRanks)) {
    int explictRank = 0;
    for (auto c : token) {
      if (std::isalpha(c)) {
        usedTokens.insert(c);
        explictRank++;
      } else if (c == '.' || c == ' ') {
        continue;
      } else {
        return false;
      }
    }
    int ellipsisRank = inputRank - explictRank;
    if (ellipsisRank > maxEllipsisRank) {
      maxEllipsisRank = ellipsisRank;
    }
    if (ellipsisRank < 0) {
      return false;
    }
    ellipsisRanks.push_back(inputRank - explictRank);
  }

  auto isTokenUsed = [&usedTokens](char c) {
    return usedTokens.find(c) != usedTokens.end();
  };
  std::string ellipsisToken;
  int usedCount = 0;
  // Iterate over the alphabet to create a new token for ellipsis
  for (char c = 'a'; c <= 'z'; ++c) {
    if (!isTokenUsed(c)) {
      ellipsisToken.push_back(c);
      usedCount++;
      if (usedCount == maxEllipsisRank) {
        break;
      }
    }
  }

  // replace ellipsis with ellipsisToken
  for (size_t i = 0; i < inputTokens.size(); i++) {
    size_t ellipsisPos = inputTokens[i].find("...");
    if (ellipsisPos == std::string::npos) {
      continue;
    }
    if (ellipsisRanks[i] == maxEllipsisRank) {
      inputTokens[i].replace(ellipsisPos, 3, ellipsisToken);
    } else if (ellipsisRanks[i] == 0) {
      inputTokens[i].replace(ellipsisPos, 3, "");
    } else {
      inputTokens[i].replace(
          ellipsisPos, 3,
          ellipsisToken.substr(ellipsisToken.size() - ellipsisRanks[i]));
    }
  }

  // replace ellipsis in result
  size_t ellipsisPos = resultStr.find("...");
  if (ellipsisPos != std::string::npos) {
    resultStr.replace(ellipsisPos, 3, ellipsisToken);
  }

  // join input and result
  equation = llvm::join(inputTokens, ",") + " -> " + resultStr;
  return true;
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
      inputTokens.push_back(std::move(inputToken));
      inputToken.clear();
      currentVariable = kIsResult;
      index++;
    } else if (equation[index] != ' ') {
      return false;
    }
    index++;
  }

  if (!inputToken.empty() && currentVariable == kIsInput) {
    inputTokens.push_back(std::move(inputToken));
  }

  return true;
}

static bool
diagonalizeInputAndRewriteEquation(Location loc, PatternRewriter &rewriter,
                                   std::string &equation,
                                   SmallVector<Value> &inputTensors) {
  SmallVector<char> resultTokens;
  SmallVector<SmallVector<char>> inputTokens;

  if (!parseEquation(equation, inputTokens, resultTokens)) {
    return false;
  }

  for (size_t i = 0, d = inputTokens.size(); i < d; ++i) {
    SmallVector<char> inputStr = inputTokens[i];
    Value input = inputTensors[i];

    for (size_t d0 = 0; d0 < inputStr.size(); ++d0) {
      char id = inputStr[d0];

      size_t d1;
      for (d1 = d0 + 1; d1 < inputStr.size(); d1++) {
        if (id == inputStr[d1])
          break;
      }

      // No duplicate found so we can continue.
      if (d1 == inputStr.size())
        continue;

      // Remove the ID and move to the end:
      for (size_t i = d0 + 1; i < d1; ++i)
        inputStr[i - 1] = inputStr[i];
      for (size_t i = d1 + 1, s = inputStr.size(); i < s; ++i)
        inputStr[i - 2] = inputStr[i];

      inputStr[inputStr.size() - 2] = id;
      inputStr.resize(inputStr.size() - 1);

      auto inputTy = cast<ValueTensorType>(input.getType());
      llvm::SmallVector<int64_t> newShape;
      for (size_t i = 0, s = inputTy.getSizes().size(); i < s; ++i) {
        if (i == d0 || i == d1)
          continue;
        newShape.push_back(inputTy.getSizes()[i]);
      }
      newShape.push_back(inputTy.getSizes()[d0]);

      inputTy = rewriter.getType<ValueTensorType>(newShape, inputTy.getDtype());

      Value zero = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(0));

      Value d0Val = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(d0));
      Value d1Val = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(d1));

      input = rewriter.create<AtenDiagonalOp>(loc, inputTy, /*input=*/input,
                                              /*offset=*/zero, /*dim1=*/d0Val,
                                              /*dim2=*/d1Val);

      // Frontmost token will have changed:
      d0--;
    }

    inputTokens[i] = inputStr;
    inputTensors[i] = input;
  }

  llvm::SmallVector<std::string> inputStrings;
  for (auto inputStr : inputTokens)
    inputStrings.emplace_back(inputStr.begin(), inputStr.end());

  std::string resultString(resultTokens.begin(), resultTokens.end());

  equation = llvm::join(inputStrings, ",");
  if (!resultString.empty())
    equation = equation + "->" + resultString;
  return true;
}

// [*batchingDims, *lhsOtherDims, *lhsReduceDims, *lhsContractingDims] =>
// [batchingDimsProd, lhsOtherDimsProd, lhsContractingDimsProd]
static Value collapseDimForMatmul(PatternRewriter &rewriter, Location loc,
                                  Value input, int64_t batchDimsLength,
                                  int64_t contractingDimsLength,
                                  int64_t otherDimsLength,
                                  int64_t reduceDimsLength, bool isLhs) {
  auto inputType = cast<ValueTensorType>(input.getType());
  auto inputRank = batchDimsLength + contractingDimsLength + otherDimsLength +
                   reduceDimsLength;
  SmallVector<OpFoldResult> inputShapeTensor;
  for (auto i = 0; i < inputRank; ++i) {
    inputShapeTensor.emplace_back(rewriter.createOrFold<AtenSizeIntOp>(
        loc, input,
        rewriter.create<Torch::ConstantIntOp>(loc,
                                              rewriter.getI64IntegerAttr(i))));
  }

  SmallVector<Value> outShapeTensor;
  Value constOne =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  auto dimOffset = 0;

  auto materializeIntFold = [&](OpFoldResult thing) {
    if (auto attr = dyn_cast<mlir::Attribute>(thing)) {
      Value result = rewriter.create<Torch::ConstantIntOp>(
          loc, cast<mlir::IntegerAttr>(attr));
      return result;
    }
    return cast<mlir::Value>(thing);
  };

  auto appendDims = [&](int64_t dimLength) {
    OpFoldResult prod = getAsOpFoldResult(constOne);
    for (auto i = 0; i < dimLength; ++i) {
      prod = rewriter.createOrFold<AtenMulIntOp>(
          loc, materializeIntFold(prod),
          materializeIntFold(inputShapeTensor[i + dimOffset]));
    }
    outShapeTensor.emplace_back(materializeIntFold(prod));
    dimOffset += dimLength;
  };

  appendDims(batchDimsLength);
  if (!isLhs)
    appendDims(contractingDimsLength);
  appendDims(otherDimsLength + reduceDimsLength);
  if (isLhs)
    appendDims(contractingDimsLength);

  SmallVector<int64_t> resultShape;
  for (auto value : outShapeTensor) {
    int64_t v;
    if (matchPattern(value, m_TorchConstantInt(&v))) {
      resultShape.push_back(v);
      continue;
    }
    resultShape.push_back(Torch::kUnknownSize);
  }

  auto outShapeValue = rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(input.getContext())),
      outShapeTensor);

  auto outType =
      inputType.getWithSizesAndDtype(resultShape, inputType.getOptionalDtype());
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
  auto inputType = cast<ValueTensorType>(input.getType());
  llvm::SmallDenseMap<char, int64_t> dimTokenMap;
  for (size_t idx = 0; idx < dimTokens.size(); ++idx) {
    dimTokenMap[dimTokens[idx]] = idx;
  }

  SmallVector<int64_t> permuteShape;
  SmallVector<Value> permuteVec;
  auto appendDims = [&](SmallVector<char> dimTokens) {
    for (auto d : dimTokens) {
      permuteVec.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dimTokenMap[d])));
      permuteShape.push_back(inputType.getSizes()[dimTokenMap[d]]);
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

  auto outType = inputType.getWithSizesAndDtype(permuteShape,
                                                inputType.getOptionalDtype());
  return rewriter.create<Torch::AtenPermuteOp>(loc, outType, input, dstDims);
}

static LogicalResult performMatmul(PatternRewriter &rewriter, Location loc,
                                   Value lhs, SmallVector<char> &lhsTokens,
                                   Value rhs, SmallVector<char> &rhsTokens,
                                   Value &result,
                                   SmallVector<char> &resultTokens,
                                   SmallVector<char> &finalResultTokens) {
  auto lhsType = cast<ValueTensorType>(lhs.getType());
  auto rhsType = cast<ValueTensorType>(rhs.getType());

  Type outputDType = lhsType.hasDtype() ? lhsType.getOptionalDtype()
                                        : rhsType.getOptionalDtype();

  auto materializeIntFold = [&](OpFoldResult thing) {
    if (auto attr = dyn_cast<mlir::Attribute>(thing)) {
      Value result = rewriter.create<Torch::ConstantIntOp>(
          loc, cast<mlir::IntegerAttr>(attr));
      return result;
    }
    return cast<mlir::Value>(thing);
  };

  llvm::SmallDenseMap<char, Value> lhsDimShapeMap;
  for (size_t idx = 0; idx < lhsTokens.size(); ++idx) {
    char d = lhsTokens[idx];
    OpFoldResult lhsFold = rewriter.createOrFold<AtenSizeIntOp>(
        loc, lhs,
        rewriter.create<Torch::ConstantIntOp>(loc,
                                              rewriter.getI64IntegerAttr(idx)));
    lhsDimShapeMap[d] = materializeIntFold(lhsFold);
  }
  llvm::SmallDenseMap<char, Value> rhsDimShapeMap;
  for (size_t idx = 0; idx < rhsTokens.size(); ++idx) {
    char d = rhsTokens[idx];
    OpFoldResult rhsFold = rewriter.createOrFold<AtenSizeIntOp>(
        loc, rhs,
        rewriter.create<Torch::ConstantIntOp>(loc,
                                              rewriter.getI64IntegerAttr(idx)));
    rhsDimShapeMap[d] = materializeIntFold(rhsFold);
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
        OpFoldResult out = rewriter.createOrFold<Torch::PrimMaxIntOp>(
            loc, lhsDimShapeMap[d], rhsDimShapeMap[d]);
        outDimShapeMap[d] = materializeIntFold(out);
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

  lhsType = cast<ValueTensorType>(lhs.getType());
  rhsType = cast<ValueTensorType>(rhs.getType());

  SmallVector<int64_t> outShape;
  outShape.push_back(lhsType.getSizes()[0]);
  outShape.push_back(lhsType.getSizes()[1]);
  outShape.push_back(rhsType.getSizes()[2]);

  // perform matmul
  auto outType = lhsType.getWithSizesAndDtype(outShape, outputDType);

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

  SmallVector<int64_t> resultShape;
  for (auto value : outShapeTensors) {
    int64_t v;
    if (matchPattern(value, m_TorchConstantInt(&v))) {
      resultShape.push_back(v);
      continue;
    }
    resultShape.push_back(Torch::kUnknownSize);
  }

  auto outResultShape = rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(lhs.getContext())),
      outShapeTensors);
  result = rewriter.create<Torch::AtenReshapeOp>(
      loc, lhsType.getWithSizesAndDtype(resultShape, outputDType), result,
      outResultShape);
  return success();
}

static Value performLastReduceAndPermute(PatternRewriter &rewriter,
                                         Location loc, Type outType,
                                         Value input,
                                         SmallVector<char> &inputTokens,
                                         SmallVector<char> &outTokens) {
  auto inputType = cast<BaseTensorType>(input.getType());

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
class DecomposeAtenTriuOp : public OpRewritePattern<AtenTriuOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTriuOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasSizes() || !inputType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "should have shape and dtype");
    }
    if (inputType.getSizes().size() < 2) {
      return rewriter.notifyMatchFailure(op, "the rank of tensor should >= 2");
    }

    Value cstZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value none = rewriter.create<ConstantNoneOp>(loc);

    Value rowSize = getTensorDimSize(rewriter, input, -2);
    Value colSize = getTensorDimSize(rewriter, input, -1);

    auto si64Type = rewriter.getIntegerType(/*width=*/64, /*isSigned*/ true);
    auto int64DtypeInt = getDtypeIntValueForType(rewriter, loc, si64Type);
    auto rowArrangeType = getTensorTypeFromShapeValues({rowSize}, si64Type);
    auto colArrangeType = getTensorTypeFromShapeValues({colSize}, si64Type);

    Value rowArange =
        rewriter.create<AtenArangeOp>(loc, rowArrangeType, rowSize,
                                      /*dtype=*/int64DtypeInt, /*layout=*/none,
                                      /*device=*/none, /*pin_memory=*/none);
    Value colArange =
        rewriter.create<AtenArangeOp>(loc, colArrangeType, colSize,
                                      /*dtype=*/int64DtypeInt, /*layout=*/none,
                                      /*device=*/none, /*pin_memory=*/none);

    auto unsqueezeRowArangeInfo =
        unsqueezeTensor(rewriter, op, rowArange, cstOne);
    auto unsqueezeColArangeInfo =
        unsqueezeTensor(rewriter, op, colArange, cstZero);

    if (failed(unsqueezeRowArangeInfo) || failed(unsqueezeColArangeInfo)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor");
    }

    Value unsqueezeRowArange = unsqueezeRowArangeInfo.value();
    Value unsqueezeColArange = unsqueezeColArangeInfo.value();

    Value unsqueezeRowArangePlusDiagonal = rewriter.create<AtenAddScalarOp>(
        loc, unsqueezeRowArange.getType(), unsqueezeRowArange, op.getDiagonal(),
        cstOne);

    auto boolType = rewriter.getI1Type();
    auto condType = getTensorTypeFromShapeValues({rowSize, colSize}, boolType);
    Value condTensor = rewriter.create<AtenGeTensorOp>(
        loc, condType, unsqueezeColArange, unsqueezeRowArangePlusDiagonal);

    rewriter.replaceOpWithNewOp<AtenWhereScalarOtherOp>(
        op, op.getResult().getType(), condTensor, input, cstZero);
    return success();
  }
};
} // namespace

/*
 This function calculates the number of elements in the lower triangle (below
 the main diagonal) of a tensor with dimensions [row, col]. The main diagonal
 can be shifted using the 'offset' parameter. The lower triangle is divided into
 two parts: a trapezoid and a rectangle. The return tuple includes the number of
 elements in the trapezoid, the number of elements in the rectangle, and the
 index of the first row such that the element [mFirstRow, 0] is below the main
 diagonal.
 */
static std::tuple<int64_t, int64_t, int64_t>
getTrilSizes(int64_t row, int64_t col, int64_t offset) {

  // Base case
  if (row == 0 || col == 0) {
    return std::make_tuple(0, 0, 0);
  }

  // Calculate mFirstRow size
  int64_t mFirstRow;
  if (offset > 0)
    mFirstRow = (col < offset + 1) ? col : offset + 1;
  else
    mFirstRow = (row + offset > 0) ? 1 : 0;

  // Calculate mLastRow size
  int64_t minimum = (col < row + offset) ? col : row + offset;
  int64_t mLastRow = (minimum > 0) ? minimum : 0;

  // Calculate nRowAll
  minimum = (row < row + offset) ? row : row + offset;
  int64_t nRowAll = (minimum > 0) ? minimum : 0;

  // Calucltae nRowTrapezoid
  int64_t nRowTrapezoid = mLastRow - mFirstRow + 1;

  // Number of elements in top trapezoid - trapezoidSize
  int64_t trapezoidSize = (mFirstRow + mLastRow) * nRowTrapezoid / 2;

  // Number of elements in bottom rectangle - rectangleSize
  int64_t diffRow = nRowAll - nRowTrapezoid;
  int64_t rectangleSize = (diffRow * col > 0) ? diffRow * col : 0;

  // Create return value
  return std::make_tuple(trapezoidSize, rectangleSize, mFirstRow);
}

/*
 This function calculates the number of elements in the upper triangle (above
 the main diagonal) of a tensor with dimensions [row, col]. The main diagonal
 can be shifted using the 'offset' parameter. The upper triangle is divided into
 two parts: a trapezoid and a rectangle. The return tuple includes the number of
 elements in the trapezoid, the number of elements in the rectangle, and the
 index of the first row such that the element [mFirstRow, 0] is above the main
 diagonal.
 */
static std::tuple<int64_t, int64_t, int64_t>
getTriuSizes(int64_t row, int64_t col, int64_t offset) {

  // Base case
  if (row == 0 || col == 0)
    return std::make_tuple(0, 0, 0);

  // Calculate mFirstRow size
  int64_t maximum = (col - offset > 0) ? col - offset : 0;
  int64_t mFirstRow = (offset > 0) ? maximum : col;

  // Number of elements in top rectangle - calculate rectangle size
  int64_t minimum = (row < -offset) ? row : -offset;
  int64_t rectangleSize = (minimum * col > 0) ? minimum * col : 0;

  // Number of elements in bottom trapezoid - calculte trapezoid size
  std::tuple<int64_t, int64_t, int64_t> trilSizes =
      getTrilSizes(row, col, offset - 1);
  int64_t trapezoidSizeTril = std::get<0>(trilSizes);
  int64_t rectangleSizeTril = std::get<1>(trilSizes);

  int64_t triuSize = row * col - (trapezoidSizeTril + rectangleSizeTril);
  int64_t trapezoidSize = triuSize - rectangleSize;

  // Create return value
  return std::make_tuple(trapezoidSize, rectangleSize, mFirstRow);
}

// decomposition of torch.triu_indices
// https://github.com/pytorch/pytorch/blob/67ef2683d970fc541b6d266d4b3f8ba9d13844ca/torch/_refs/__init__.py#L5829
namespace {
class DecomposeAtenTriuIndicesOp : public OpRewritePattern<AtenTriuIndicesOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTriuIndicesOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    // Required parameters
    Value row = op.getRow();
    Value col = op.getCol();
    Value offset = op.getOffset();

    // Check if row, col and offset are constant ints
    int64_t rowInt;
    if (!matchPattern(row, m_TorchConstantInt(&rowInt)))
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented: row not constant int");

    int64_t colInt;
    if (!matchPattern(col, m_TorchConstantInt(&colInt)))
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented: col not constant int");

    int64_t offsetInt;
    if (!matchPattern(offset, m_TorchConstantInt(&offsetInt)))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: offset not constant int");

    // Optional parameters
    Value dtype = op.getDtype();
    Value layout = op.getLayout();
    Value device = op.getDevice();
    Value pinMemory = op.getPinMemory();

    // Get int value for dtype
    int64_t dtypeInt;
    if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: dtype not constant int");

    FailureOr<Type> dtypeType =
        getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
    if (failed(dtypeType))
      return rewriter.notifyMatchFailure(op, "dtype is undefined");

    // Constants
    Value cstZero = rewriter.create<Torch::ConstantIntOp>(loc, 0);
    Value cstOne = rewriter.create<Torch::ConstantIntOp>(loc, 1);
    Value cstTwo = rewriter.create<Torch::ConstantIntOp>(loc, 2);
    Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
    Value cstMinusZeroPointFive = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(-0.5));
    Value cstMinusTwoFloat = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(-2.0));

    // Calculte trapezoidSize, rectangleSize and mFirstRow
    std::tuple<int64_t, int64_t, int64_t> triuSizes =
        getTriuSizes(rowInt, colInt, offsetInt);

    int64_t trapezoidSizeInt = std::get<0>(triuSizes);
    int64_t rectangleSizeInt = std::get<1>(triuSizes);
    int64_t mFirstRowInt = std::get<2>(triuSizes);

    // Create const int Values from ints
    Value trapezoidSize = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(trapezoidSizeInt));
    Value rectangleSize = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rectangleSizeInt));
    Value mFirstRow = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(mFirstRowInt));

    // Calculte column offset
    Value colOffset = (offsetInt > 0) ? offset : cstZero;

    // Calculate indices for top rectangle
    auto arrangeType =
        getTensorTypeFromShapeValues({rectangleSize}, *dtypeType);
    Value xs2 =
        rewriter.create<AtenArangeOp>(loc, arrangeType, rectangleSize,
                                      /*dtype=*/dtype, /*layout=*/layout,
                                      /*device=*/device,
                                      /*pin_memory=*/pinMemory);

    // Calculate row_indices2 and column_idices 2
    Value rowInds2 =
        rewriter.create<AtenFloorDivideScalarOp>(loc, xs2.getType(), xs2, col);
    Value colInds2 =
        rewriter.create<AtenRemainderScalarOp>(loc, xs2.getType(), xs2, col);

    // Bottom trapezoid
    auto f64DtypeInt =
        getDtypeIntValueForType(rewriter, loc, rewriter.getF64Type());
    arrangeType =
        getTensorTypeFromShapeValues({trapezoidSize}, rewriter.getF64Type());
    Value xs1 =
        rewriter.create<AtenArangeOp>(loc, arrangeType, trapezoidSize,
                                      /*dtype=*/f64DtypeInt, /*layout=*/layout,
                                      /*device=*/device,
                                      /*pin_memory=*/pinMemory);

    // b = -0.5 - m_first_row
    Value mFirstRowFloat = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(mFirstRowInt));
    Value b = rewriter.create<AtenSubFloatOp>(loc, cstMinusZeroPointFive,
                                              mFirstRowFloat);

    // Implements this piece of code: row_inds1 = torch.floor(-b - torch.sqrt(b
    // * b - 2 * xs1))
    Value bSquare = rewriter.create<AtenMulFloatOp>(loc, b, b);

    Value twoTimesXs1 = rewriter.create<AtenMulScalarOp>(loc, xs1.getType(),
                                                         xs1, cstMinusTwoFloat);
    Value sqrtInput = rewriter.create<AtenAddScalarOp>(
        loc, twoTimesXs1.getType(), twoTimesXs1, bSquare, cstOne);

    Value sqrt =
        rewriter.create<AtenSqrtOp>(loc, sqrtInput.getType(), sqrtInput);
    Value negativeSqrt = rewriter.create<AtenNegOp>(loc, sqrt.getType(), sqrt);

    Value rowInds1 = rewriter.create<AtenSubScalarOp>(
        loc, negativeSqrt.getType(), negativeSqrt, b, cstOne);
    rowInds1 = rewriter.create<AtenFloorOp>(loc, rowInds1.getType(), rowInds1);

    // Implements this piece of code: col_inds1 = torch.floor(xs1 - ((2 *
    // m_first_row - 1 - row_inds1) * row_inds1) * 0.5)
    Value twoTimesMFirstRow =
        rewriter.create<AtenMulIntOp>(loc, cstTwo, mFirstRow);
    twoTimesMFirstRow =
        rewriter.create<AtenSubIntOp>(loc, twoTimesMFirstRow, cstOne);
    Value negativeRowInds1 =
        rewriter.create<AtenNegOp>(loc, rowInds1.getType(), rowInds1);

    negativeRowInds1 = rewriter.create<AtenAddScalarOp>(
        loc, negativeRowInds1.getType(), negativeRowInds1, twoTimesMFirstRow,
        cstOne);
    negativeRowInds1 = rewriter.create<AtenMulTensorOp>(
        loc, negativeRowInds1.getType(), negativeRowInds1, rowInds1);
    negativeRowInds1 = rewriter.create<AtenMulScalarOp>(
        loc, negativeRowInds1.getType(), negativeRowInds1,
        cstMinusZeroPointFive);

    Value colInds1 = rewriter.create<AtenAddTensorOp>(loc, xs1.getType(), xs1,
                                                      negativeRowInds1, cstOne);
    colInds1 = rewriter.create<AtenFloorOp>(loc, colInds1.getType(), colInds1);

    // Convert to dtype
    Type int64Type = rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true);

    auto rowInds1Type = cast<BaseTensorType>(rowInds1.getType());
    ArrayRef<int64_t> sizes = rowInds1Type.getSizes();
    Type finalRowType = rowInds1Type.getWithSizesAndDtype(sizes, int64Type);
    rowInds1 = rewriter.create<AtenToDtypeOp>(
        loc, finalRowType, rowInds1, dtype,
        /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
        /*memory_format=*/cstOne);

    auto colInds1Type = cast<BaseTensorType>(colInds1.getType());
    sizes = colInds1Type.getSizes();
    Type finalColType = colInds1Type.getWithSizesAndDtype(sizes, int64Type);
    colInds1 = rewriter.create<AtenToDtypeOp>(
        loc, finalColType, colInds1, dtype,
        /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
        /*memory_format=*/cstOne);

    // Final calculation for row and col indices
    if (colInt) {

      Value rectangleSizeDivCol =
          rewriter.create<Torch::ConstantIntOp>(loc, rectangleSizeInt / colInt);

      rowInds1 = rewriter.create<AtenAddScalarOp>(
          loc, rowInds1.getType(), rowInds1, rectangleSizeDivCol, cstOne);
    }

    colInds1 = rewriter.create<AtenAddScalarOp>(loc, colInds1.getType(),
                                                colInds1, colOffset, cstOne);

    Type listElemType =
        cast<Torch::BaseTensorType>(rowInds1.getType())
            .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                  /*optionalDtype=*/nullptr);
    Type listType = Torch::ListType::get(listElemType);

    Value sequenceRow = rewriter.create<Torch::PrimListConstructOp>(
        loc, listType, SmallVector<Value>{rowInds2, rowInds1});
    Value sequenceCol = rewriter.create<Torch::PrimListConstructOp>(
        loc, listType, SmallVector<Value>{colInds2, colInds1});

    // Concatenate row and col indices
    Type finalCatType = colInds1Type.getWithSizesAndDtype(
        {rectangleSizeInt + trapezoidSizeInt}, int64Type);

    Value catRow = rewriter.create<AtenCatOp>(loc, finalCatType, sequenceRow,
                                              /*dim=*/cstZero);
    Value catCol = rewriter.create<AtenCatOp>(loc, finalCatType, sequenceCol,
                                              /*dim=*/cstZero);

    // Make return value
    Value sequence = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(context, rowInds1.getType()),
        ValueRange{catRow, catCol});
    Type finalStackType = colInds1Type.getWithSizesAndDtype(
        ArrayRef<int64_t>{2, rectangleSizeInt + trapezoidSizeInt}, int64Type);

    rewriter.replaceOpWithNewOp<AtenStackOp>(op, finalStackType, sequence,
                                             cstZero);

    return success();
  }
};
} // namespace

// decomposition of torch.tril_indices
// https://github.com/pytorch/pytorch/blob/67ef2683d970fc541b6d266d4b3f8ba9d13844ca/torch/_refs/__init__.py#L5797
namespace {
class DecomposeAtenTrilIndicesOp : public OpRewritePattern<AtenTrilIndicesOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTrilIndicesOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    // Required parameters
    Value row = op.getRow();
    Value col = op.getCol();
    Value offset = op.getOffset();

    // Check if row, col and offset are constant ints
    int64_t rowInt;
    if (!matchPattern(row, m_TorchConstantInt(&rowInt)))
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented: row not constant int");

    int64_t colInt;
    if (!matchPattern(col, m_TorchConstantInt(&colInt)))
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented: col not constant int");

    int64_t offsetInt;
    if (!matchPattern(offset, m_TorchConstantInt(&offsetInt)))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: offset not constant int");

    // Optional parameters
    Value dtype = op.getDtype();
    Value layout = op.getLayout();
    Value device = op.getDevice();
    Value pinMemory = op.getPinMemory();

    // Constants
    Value cstZero = rewriter.create<Torch::ConstantIntOp>(loc, 0);
    Value cstOne = rewriter.create<Torch::ConstantIntOp>(loc, 1);
    Value cstTwo = rewriter.create<Torch::ConstantIntOp>(loc, 2);
    Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
    Value cstZeroPointFive = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(0.5));
    Value cstTwoFloat = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(2.0));

    // Get int value for dtype
    int64_t dtypeInt;
    if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: dtype not constant int");

    FailureOr<Type> dtypeType =
        getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
    if (failed(dtypeType))
      return rewriter.notifyMatchFailure(op, "dtype is undefined");

    // Calculte trapezoidSize, rectangleSize and mFirstRow
    std::tuple<int64_t, int64_t, int64_t> triuSizes =
        getTrilSizes(rowInt, colInt, offsetInt);

    int64_t trapezoidSizeInt = std::get<0>(triuSizes);
    int64_t rectangleSizeInt = std::get<1>(triuSizes);
    int64_t mFirstRowInt = std::get<2>(triuSizes);

    // Create const int Values from ints
    Value trapezoidSize = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(trapezoidSizeInt));
    Value rectangleSize = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rectangleSizeInt));
    Value mFirstRow = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(mFirstRowInt));

    // Calculte column offset
    int64_t rowOffsetInt = (-offsetInt > 0) ? (-offsetInt) : 0;
    Value rowOffset = rewriter.create<Torch::ConstantIntOp>(loc, rowOffsetInt);

    // First we do the indices for TOP trapezoid
    auto f64DtypeInt =
        getDtypeIntValueForType(rewriter, loc, rewriter.getF64Type());
    auto arrangeType =
        getTensorTypeFromShapeValues({trapezoidSize}, rewriter.getF64Type());
    Value xs1 =
        rewriter.create<AtenArangeOp>(loc, arrangeType, trapezoidSize,
                                      /*dtype=*/f64DtypeInt, /*layout=*/layout,
                                      /*device=*/device,
                                      /*pin_memory=*/pinMemory);

    // b = m_first_row - 0.5
    Value mFirstRowFloat = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(mFirstRowInt));
    Value b =
        rewriter.create<AtenSubFloatOp>(loc, mFirstRowFloat, cstZeroPointFive);

    // Implements this piece of code: row_inds1 = torch.floor(-b + torch.sqrt(b
    // * b + 2 * xs1))
    Value bSquare = rewriter.create<AtenMulFloatOp>(loc, b, b);

    Value twoTimesXs1 =
        rewriter.create<AtenMulScalarOp>(loc, xs1.getType(), xs1, cstTwoFloat);
    Value sqrtInput = rewriter.create<AtenAddScalarOp>(
        loc, twoTimesXs1.getType(), twoTimesXs1, bSquare, cstOne);

    Value sqrt =
        rewriter.create<AtenSqrtOp>(loc, sqrtInput.getType(), sqrtInput);

    Value rowInds1 =
        rewriter.create<AtenSubScalarOp>(loc, sqrt.getType(), sqrt, b, cstOne);
    rowInds1 = rewriter.create<AtenFloorOp>(loc, rowInds1.getType(), rowInds1);

    // Implements this piece of code: col_inds1 = torch.floor(xs1 - (2 *
    // m_first_row - 1 + row_inds1) * row_inds1 * 0.5)
    Value twoTimesMFirstRow =
        rewriter.create<AtenMulIntOp>(loc, cstTwo, mFirstRow);
    twoTimesMFirstRow =
        rewriter.create<AtenSubIntOp>(loc, twoTimesMFirstRow, cstOne);
    twoTimesMFirstRow = rewriter.create<AtenAddScalarOp>(
        loc, rowInds1.getType(), rowInds1, twoTimesMFirstRow, cstOne);
    twoTimesMFirstRow = rewriter.create<AtenMulTensorOp>(
        loc, twoTimesMFirstRow.getType(), twoTimesMFirstRow, rowInds1);
    twoTimesMFirstRow = rewriter.create<AtenMulScalarOp>(
        loc, twoTimesMFirstRow.getType(), twoTimesMFirstRow, cstZeroPointFive);

    Value colInds1 = rewriter.create<AtenSubTensorOp>(
        loc, xs1.getType(), xs1, twoTimesMFirstRow, cstOne);
    colInds1 = rewriter.create<AtenFloorOp>(loc, colInds1.getType(), colInds1);

    // Convert top trapezoid indices to dtype
    Type int64Type = rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true);

    auto rowInds1Type = cast<BaseTensorType>(rowInds1.getType());
    ArrayRef<int64_t> sizes = rowInds1Type.getSizes();
    Type finalRowType = rowInds1Type.getWithSizesAndDtype(sizes, int64Type);
    rowInds1 = rewriter.create<AtenAddScalarOp>(loc, rowInds1.getType(),
                                                rowInds1, rowOffset, cstOne);
    rowInds1 = rewriter.create<AtenToDtypeOp>(
        loc, finalRowType, rowInds1, dtype,
        /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
        /*memory_format=*/cstOne);

    auto colInds1Type = cast<BaseTensorType>(colInds1.getType());
    sizes = colInds1Type.getSizes();
    Type finalColType = colInds1Type.getWithSizesAndDtype(sizes, int64Type);
    colInds1 = rewriter.create<AtenToDtypeOp>(
        loc, finalColType, colInds1, dtype,
        /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
        /*memory_format=*/cstOne);

    // Calculate indices for BOTTOM rectangle
    arrangeType = getTensorTypeFromShapeValues({rectangleSize}, *dtypeType);
    Value xs2 =
        rewriter.create<AtenArangeOp>(loc, arrangeType, rectangleSize,
                                      /*dtype=*/dtype, /*layout=*/layout,
                                      /*device=*/device,
                                      /*pin_memory=*/pinMemory);

    // Implements this line of code: row_inds2 = xs2 // col + (col - m_first_row
    // + 1 + row_offset)
    Value rowInds2 =
        rewriter.create<AtenFloorDivideScalarOp>(loc, xs2.getType(), xs2, col);
    int64_t addInt = colInt - mFirstRowInt + 1 + rowOffsetInt;
    Value cstAdd = rewriter.create<Torch::ConstantIntOp>(loc, addInt);
    rowInds2 = rewriter.create<AtenAddScalarOp>(loc, rowInds2.getType(),
                                                rowInds2, cstAdd, cstOne);

    // Implements this line of code: col_inds2 = xs2 % col
    Value colInds2 =
        rewriter.create<AtenRemainderScalarOp>(loc, xs2.getType(), xs2, col);

    // Prepare tensors for concatenation
    Type listElemType =
        cast<Torch::BaseTensorType>(rowInds1.getType())
            .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                  /*optionalDtype=*/nullptr);
    Type listType = Torch::ListType::get(listElemType);

    Value sequenceRow = rewriter.create<Torch::PrimListConstructOp>(
        loc, listType, SmallVector<Value>{rowInds1, rowInds2});
    Value sequenceCol = rewriter.create<Torch::PrimListConstructOp>(
        loc, listType, SmallVector<Value>{colInds1, colInds2});

    // Concatenate row and col indices
    Type finalCatType = colInds1Type.getWithSizesAndDtype(
        {rectangleSizeInt + trapezoidSizeInt}, int64Type);

    Value catRow = rewriter.create<AtenCatOp>(loc, finalCatType, sequenceRow,
                                              /*dim=*/cstZero);
    Value catCol = rewriter.create<AtenCatOp>(loc, finalCatType, sequenceCol,
                                              /*dim=*/cstZero);

    // Make return value - stack row and col indices
    Value sequence = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(context, rowInds1.getType()),
        ValueRange{catRow, catCol});
    Type finalStackType = colInds1Type.getWithSizesAndDtype(
        ArrayRef<int64_t>{2, rectangleSizeInt + trapezoidSizeInt}, int64Type);

    rewriter.replaceOpWithNewOp<AtenStackOp>(op, finalStackType, sequence,
                                             cstZero);

    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenDeg2radOp : public OpRewritePattern<AtenDeg2radOp> {
public:
  using OpRewritePattern<AtenDeg2radOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenDeg2radOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    auto selfTy = dyn_cast<BaseTensorType>(self.getType());
    if (!selfTy || !selfTy.getDtype()) {
      return rewriter.notifyMatchFailure(op, "requires tensor types input.");
    }

    auto outTy = dyn_cast<BaseTensorType>(op.getType());
    if (!outTy || !outTy.getDtype()) {
      return rewriter.notifyMatchFailure(
          op, "requires output is a tensor with dtype.");
    }

    if (selfTy.getDtype() != outTy.getDtype()) {
      self = convertTensorToDtype(rewriter, loc, self, outTy.getDtype());
    }

    Value pi =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(M_PI));
    Value basic =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(180.0));
    Value rad =
        rewriter.create<AtenDivScalarOp>(loc, op.getType(), self, basic);
    Value result = rewriter.create<AtenMulScalarOp>(loc, op.getType(), rad, pi);

    rewriter.replaceOp(op, result);

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

    auto resultTy = cast<BaseTensorType>(op.getType());
    if (!resultTy.hasSizes() || !resultTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have sizes and dtype");
    }

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
        computeReductionType(rewriter, op, cast<BaseTensorType>(self.getType()),
                             dim,
                             /*keepDim=*/true),
        op.getSelf(), dim, start, startPlusOne, /*step=*/one);

    auto sliceTy = cast<BaseTensorType>(slice.getType());
    if (sliceTy.getSizes().size() == resultTy.getSizes().size()) {
      rewriter.replaceOp(op, slice);
      return success();
    }

    // `aten.slice.tensor` doesn't squeeze the dim even when it's size 1 after
    // slicing, while `aten.select.int` does.
    rewriter.replaceOpWithNewOp<AtenSqueezeDimOp>(op, op.getResult().getType(),
                                                  slice, op.getDim());
    return success();
  }
};
} // namespace

namespace {
class DecomposePrimTolistOp : public OpRewritePattern<PrimTolistOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimTolistOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto self = op.getOperands()[0];
    auto selfTy = dyn_cast<Torch::BaseTensorType>(self.getType());
    if (!selfTy || !selfTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "Unknown self shape");

    int64_t rank = selfTy.getSizes().size();
    if (rank != 1)
      return rewriter.notifyMatchFailure(op, "Expected rank-1");

    int64_t length = selfTy.getSizes().back();
    if (length == Torch::kUnknownSize)
      return rewriter.notifyMatchFailure(op, "Tolist length is unknown");

    auto resultTy = dyn_cast<Torch::ListType>(op.getType(0));
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "Result type is not list");

    auto scalarTy = resultTy.getContainedType();
    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    auto extractTy = rewriter.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{1}, selfTy.getOptionalDtype());
    llvm::SmallVector<Value> results;
    llvm::SmallVector<int64_t> sizes(selfTy.getSizes());
    for (int64_t i = 0; i < length; ++i) {
      Value iv =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i));
      Value extract = rewriter.create<AtenSelectIntOp>(
          loc, extractTy, self, /*dim=*/zero, /*index=*/iv);
      Value scalar = rewriter.create<AtenItemOp>(loc, scalarTy, extract);
      results.push_back(scalar);
    }

    rewriter.replaceOpWithNewOp<PrimListConstructOp>(op, resultTy, results);
    return failure();
  }
};
} // namespace

namespace {
class DecomposeAtenSplitWithSizesOp
    : public OpRewritePattern<AtenSplitWithSizesOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSplitWithSizesOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value self = op.getSelf();
    SmallVector<Value> splitSizes;
    if (!getListConstructElements(op.getSplitSizes(), splitSizes))
      return rewriter.notifyMatchFailure(op, "Unable to get sizes");

    if (splitSizes.empty())
      return rewriter.notifyMatchFailure(op, "No split sizes");

    auto selfTy = dyn_cast<BaseTensorType>(self.getType());
    if (!selfTy || !selfTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "Self shape unknown");

    int64_t rank = selfTy.getSizes().size();
    auto resultTy = dyn_cast<Torch::ListType>(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "Result type not a list");

    auto sliceTy =
        dyn_cast_or_null<Torch::BaseTensorType>(resultTy.getContainedType());
    if (!sliceTy || !sliceTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "Slice type is unknown");

    int64_t dimInt = 0;
    bool hasDim = matchPattern(op.getDim(), m_TorchConstantInt(&dimInt));
    if (dimInt < 0)
      dimInt += rank;

    auto intTy = rewriter.getType<Torch::IntType>();
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value begin =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));

    llvm::SmallVector<Value> slices;
    llvm::SmallVector<int64_t> sliceSizes(sliceTy.getSizes());
    int64_t defaultLength = !hasDim ? Torch::kUnknownSize : sliceSizes[dimInt];
    for (auto size : splitSizes) {
      Value end = rewriter.create<AtenAddIntOp>(loc, intTy, begin, size);

      int64_t sizeInt;
      if (hasDim && matchPattern(size, m_TorchConstantInt(&sizeInt))) {
        sliceSizes[dimInt] = sizeInt;
      } else if (hasDim) {
        sliceSizes[dimInt] = defaultLength;
      }

      sliceTy = rewriter.getType<ValueTensorType>(sliceSizes,
                                                  sliceTy.getOptionalDtype());
      Value slice = rewriter.create<AtenSliceTensorOp>(
          loc, sliceTy, op.getSelf(),
          /*dim=*/op.getDim(), /*start=*/begin, /*end=*/end, /*step=*/one);
      slices.push_back(slice);
      begin = end;
    }

    rewriter.replaceOpWithNewOp<PrimListConstructOp>(op, resultTy, slices);
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

    auto outputTy = dyn_cast<Torch::ValueTensorType>(op.getType());
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
    auto outType = dyn_cast<BaseTensorType>(op.getType());
    if (!outType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types input are currently supported");
    if (!outType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value none = rewriter.create<ConstantNoneOp>(loc);
    auto context = op.getContext();
    auto int64Dtype = getDtypeIntValueForType(
        rewriter, loc,
        rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true));
    auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);

    int64_t n = kUnknownSize;
    int64_t m = kUnknownSize;
    // prioritize getting shape from output shape
    if (outType.hasSizes() && outType.getSizes().size() == 2) {
      n = outType.getSizes().front();
      m = outType.getSizes().back();
    }
    // if output shape is not available, try to get shape from input
    if (n == kUnknownSize)
      matchPattern(op.getN(), m_TorchConstantInt(&n));
    if (m == kUnknownSize)
      matchPattern(op.getM(), m_TorchConstantInt(&m));

    // prepare two unsqueezed ranges that are equal on and only on the diagonal
    auto rangeNSize = llvm::SmallVector<int64_t, 1>({n});
    Type rangeNType = outType.getWithSizesAndDtype(rangeNSize, si64Type);
    Value rangeN = rewriter.create<AtenArangeOp>(
        loc, rangeNType, op.getN(), /*dtype=*/int64Dtype, /*layout=*/none,
        /*device=*/op.getDevice(), /*pin_memory=*/none);

    auto rangeMSize = llvm::SmallVector<int64_t, 1>({m});
    Type rangeMType = outType.getWithSizesAndDtype(rangeMSize, si64Type);
    Value rangeM = rewriter.create<AtenArangeOp>(
        loc, rangeMType, op.getM(), /*dtype=*/int64Dtype, /*layout=*/none,
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

    auto eqType = ValueTensorType::get(
        context, cast<BaseTensorType>(op.getType()).getSizes(),
        IntegerType::get(context, 1));
    Value eqTensor =
        rewriter.create<AtenEqTensorOp>(loc, eqType, unsqzRangeN, rangeM);

    Value dtype = op.getDtype();
    if (isa<Torch::BoolType>(dtype.getType())) {
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
    if (!isa<ValueTensorType>(input.getType())) {
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
// Decompose aten.atleast_1d into: aten.reshape. See
// https://github.com/pytorch/pytorch/blob/9a8ab778d34bd24c5caceb340837483decc4c311/torch/_refs/__init__.py#L2591
// def atleast_1d(
//     arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args:
//     TensorLikeType
// ) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
//     """Refrence implementation of :func:`torch.atleast_1d`."""
//     if not args and isinstance(arg, collections.abc.Sequence):
//         args_ = arg
//     else:
//         assert not isinstance(arg, collections.abc.Sequence)
//         args_ = (arg,) + args
//     res = tuple(a if a.ndim >= 1 else unsqueeze(a, 0) for a in args_)
//     return res if len(res) > 1 else res[0]
class DecomposeAtenAtleast1dOp : public OpRewritePattern<AtenAtleast1dOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAtleast1dOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getSelf();
    Location loc = op.getLoc();
    Type opType = op.getType();
    auto inpType = cast<BaseTensorType>(input.getType());
    SmallVector<int64_t> inputShape(inpType.getSizes());
    if (inputShape.empty()) {
      Value zero = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(0));
      rewriter.replaceOpWithNewOp<AtenUnsqueezeOp>(op, opType, input, zero);
      return success();
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};
} // namespace

namespace {
// Decompose aten.atleast_2d into: aten.reshape. See
// https://github.com/pytorch/pytorch/blob/9a8ab778d34bd24c5caceb340837483decc4c311/torch/_refs/__init__.py#L2604
// def atleast_2d(
//     arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args:
//     TensorLikeType
// ) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
//     """Reference implementation of :func:`torch.atleast_2d`."""
//     if not args and isinstance(arg, collections.abc.Sequence):
//         args_ = arg
//     else:
//         assert not isinstance(arg, collections.abc.Sequence)
//         args_ = (arg,) + args
//     unsqueeze_atleast_1d = partial(_unsqueeze_atleast, atleast_1d, 0)
//     res = tuple(a if a.ndim >= 2 else unsqueeze_atleast_1d(a) for a in args_)
//     return res if len(res) > 1 else res[0]
class DecomposeAtenAtleast2dOp : public OpRewritePattern<AtenAtleast2dOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAtleast2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Type opType = op.getType();

    auto inputType = cast<BaseTensorType>(input.getType());
    SmallVector<int64_t> inputShape(inputType.getSizes());

    if (inputShape.size() >= 2) {
      rewriter.replaceOp(op, input);
      return success();
    }
    auto atleast1dResShape =
        inputShape.empty() ? SmallVector<int64_t, 1>{1} : inputShape;
    auto atleast1dResType = rewriter.getType<ValueTensorType>(
        atleast1dResShape, inputType.getOptionalDtype());
    auto atleast1dRes =
        rewriter.create<AtenAtleast1dOp>(loc, atleast1dResType, input);
    Value zero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<AtenUnsqueezeOp>(op, opType, atleast1dRes,
                                                 zero);
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

    SmallVector<Value> inputTensors;
    if (!getListConstructElements(op.getTensors(), inputTensors)) {
      return rewriter.notifyMatchFailure(
          op, "input should comes from a PrimListConstructOp");
    }

    auto allTensorHasSizes = [](Value tensor) {
      auto type = dyn_cast<BaseTensorType>(tensor.getType());
      if (!type || !type.hasSizes())
        return false;
      return true;
    };

    if (!llvm::all_of(inputTensors, allTensorHasSizes)) {
      return rewriter.notifyMatchFailure(op,
                                         "all input tensors should have sizes");
    }

    std::string equation;
    if (!matchPattern(op.getEquation(), m_TorchConstantStr(equation))) {
      return rewriter.notifyMatchFailure(op, "Unsupported value of equation");
    }
    // if "..." in equation, modify it
    if (equation.find("...") != std::string::npos) {
      SmallVector<int64_t> inputRanks;
      for (Value tensor : inputTensors) {
        auto type = cast<BaseTensorType>(tensor.getType());
        inputRanks.push_back(type.getSizes().size());
      }

      if (!rewriteEquationWithEllipsisSlicing(equation, inputRanks)) {
        return rewriter.notifyMatchFailure(
            op, "Unexpected character in equations encountered");
      }
    }

    if (!diagonalizeInputAndRewriteEquation(op.getLoc(), rewriter, equation,
                                            inputTensors)) {
      return rewriter.notifyMatchFailure(op,
                                         "Failed to handle diagonalization");
    }

    SmallVector<char> resultTokens;
    SmallVector<SmallVector<char>> inputTokens;
    if (!parseEquation(equation, inputTokens, resultTokens)) {
      return rewriter.notifyMatchFailure(
          op, "Unexpected character in equations encountered");
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
// Trilinear einstein sum, decomposed to:
// (i1.unsqueeze(expand1) * i2.unsqueeze(expand2) * i3.unsqueeze(expand3))
//    .sum(sumdim)
// The unrollDim operand does not impact the output of the operation, so
// it is ignored.

class DecomposeAten_TrilinearOp : public OpRewritePattern<Aten_TrilinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_TrilinearOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    Value input1 = op.getI1();
    Value input2 = op.getI2();
    Value input3 = op.getI3();

    // Expansions
    SmallVector<int64_t> expand1;
    SmallVector<int64_t> expand2;
    SmallVector<int64_t> expand3;
    if (!matchPattern(op.getExpand1(), m_TorchListOfConstantInts(expand1))) {
      return rewriter.notifyMatchFailure(op, "expand1 should be constant");
    }
    if (!matchPattern(op.getExpand2(), m_TorchListOfConstantInts(expand2))) {
      return rewriter.notifyMatchFailure(op, "expand2 should be constant");
    }
    if (!matchPattern(op.getExpand3(), m_TorchListOfConstantInts(expand3))) {
      return rewriter.notifyMatchFailure(op, "expand3 should be constant");
    }

    SmallVector<int64_t> sumDim;
    if (!matchPattern(op.getSumdim(), m_TorchListOfConstantInts(sumDim))) {
      return rewriter.notifyMatchFailure(op, "sumDim should be constant");
    }

    // Check if there are any dimensions that intersect between expand1,
    // expand2, and expand3.
    int64_t totalDims =
        cast<BaseTensorType>(input1.getType()).getSizes().size() +
        expand1.size();
    if (sharedExpandDims(totalDims, expand1, expand2, expand3, sumDim)) {
      // pytorch issue filed: https://github.com/pytorch/pytorch/issues/138353
      // TODO: Remove warning when issue gets resolved.
      op->emitWarning("aten::_trilinear implementation in this case is "
                      "non-functional (returns an empty dimension). We will "
                      "intentionally deviate from this behavior.");
    }

    // Apply unsqueeze to respective input tensors at the specified dimensions
    SmallVector<int64_t> sortedExpand1 = expand1;
    std::sort(sortedExpand1.begin(), sortedExpand1.end());
    for (auto expand : sortedExpand1) {
      Value expandDim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(expand));
      input1 = *unsqueezeTensor(rewriter, op, input1, expandDim);
    }
    SmallVector<int64_t> sortedExpand2 = expand2;
    std::sort(sortedExpand2.begin(), sortedExpand2.end());
    for (auto expand : sortedExpand2) {
      Value expandDim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(expand));
      input2 = *unsqueezeTensor(rewriter, op, input2, expandDim);
    }
    SmallVector<int64_t> sortedExpand3 = expand3;
    std::sort(sortedExpand3.begin(), sortedExpand3.end());
    for (auto expand : sortedExpand3) {
      Value expandDim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(expand));
      input3 = *unsqueezeTensor(rewriter, op, input3, expandDim);
    }

    // Apply multiplication operation.
    auto mul1 =
        rewriter.create<AtenMulTensorOp>(loc, op.getType(), input1, input2);
    auto mul2 =
        rewriter.create<AtenMulTensorOp>(loc, op.getType(), mul1, input3);

    // Apply sum operation.
    // Parse sumDim in descending order to avoid any issues with the
    // dimensions being removed.
    Value result = mul2;
    SmallVector<int64_t> sortedSumDims = sumDim;
    std::sort(sortedSumDims.rbegin(), sortedSumDims.rend());
    for (int64_t dim : sortedSumDims) {
      Value dimValue = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dim));
      result =
          createSumAlongDimension(rewriter, loc, op, result, dimValue, false);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  // Determine if there are any dimensions that intersect between expand1,
  // expand2, and expand3.
  bool sharedExpandDims(const int64_t &totalDims,
                        const SmallVector<int64_t> &expand1,
                        const SmallVector<int64_t> &expand2,
                        const SmallVector<int64_t> &expand3,
                        const SmallVector<int64_t> &sumDim) const {
    for (int64_t i = 0; i < totalDims; ++i) {
      if (!contains(sumDim, i) && contains(expand1, i) &&
          contains(expand2, i) && contains(expand3, i)) {
        return true;
      }
    }
    return false;
  }
  bool contains(const SmallVector<int64_t> &vec, int64_t value) const {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
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
    BaseTensorType inputType = cast<BaseTensorType>(self.getType());

    Value output = op.getResult();
    BaseTensorType outputType = cast<BaseTensorType>(output.getType());

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
                              Type accumulatorType, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value dim = op.getDim();
  if (resultType != accumulatorType)
    self = convertTensorToDtype(rewriter, loc, self, accumulatorType);
  Value xMax =
      createMaxAlongDimension(rewriter, loc, op, self, dim, /*keepDim=*/true);

  if (!xMax)
    return nullptr;
  Value unNormalized =
      createTensorSub(rewriter, loc, self.getType(), self, xMax);
  Value unNormalizedExp =
      rewriter.create<AtenExpOp>(loc, self.getType(), unNormalized);
  Value sum = createSumAlongDimension(rewriter, loc, op, unNormalizedExp, dim,
                                      /*keepDim=*/true);
  if (!sum)
    return nullptr;

  Value result = rewriter.create<AtenDivTensorOp>(loc, self.getType(),
                                                  unNormalizedExp, sum);
  if (resultType != accumulatorType)
    result = convertTensorToDtype(rewriter, loc, result,
                                  cast<BaseTensorType>(resultType).getDtype());

  return result;
}

// Decompose softmax into: exp(x) / sum(exp(x))
namespace {
class DecomposeAtenSoftmaxIntOp : public OpRewritePattern<AtenSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType resultTensorType = cast<BaseTensorType>(op.getType());
    if (!resultTensorType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Type resultTensorDtype = resultTensorType.getDtype();
    if (!isa<mlir::FloatType>(resultTensorDtype))
      return rewriter.notifyMatchFailure(op,
                                         "Only support floating-point type");

    // If `dtype` arg is non-none then convert the input to `dtype`.
    if (!isa<Torch::NoneType>(op.getDtype().getType())) {
      Location loc = op.getLoc();
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
      self = rewriter.create<AtenToDtypeOp>(
          loc, resultTensorType, self,
          getDtypeIntValueForType(rewriter, loc, resultTensorDtype),
          /*non_blocking=*/cstFalse, /*copy=*/cstFalse, /*memory_format=*/none);
    }

    Type accumulatorTensorType = getDefaultAccType(rewriter, resultTensorDtype);

    Value result = getSoftmaxResult(op, self, resultTensorType,
                                    accumulatorTensorType, rewriter);
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
    BaseTensorType tensorType = cast<BaseTensorType>(self.getType());
    if (!tensorType.hasDtype() || !isa<mlir::FloatType>(tensorType.getDtype()))
      return rewriter.notifyMatchFailure(op, "Only support floating type");
    bool halfToFloat;
    if (!matchPattern(op.getHalfToFloat(), m_TorchConstantBool(&halfToFloat)))
      return rewriter.notifyMatchFailure(
          op, "Expected a boolean value for half_to_float");

    BaseTensorType resultTensorType = cast<BaseTensorType>(op.getType());
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

    Type accumulatorTensorType = getDefaultAccType(rewriter, resultTensorDtype);

    Value result = getSoftmaxResult(op, self, resultTensorType,
                                    accumulatorTensorType, rewriter);
    if (!result)
      return op.emitError("failed to get softmax result");
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTensorType,
                                                        result);
    return success();
  }
};
} // namespace

// Ref:
// https://github.com/pytorch/pytorch/blob/5314ae2660a778b87987030182f787bb6cb092c0/aten/src/ATen/native/transformers/attention.cpp#L663-L673
namespace {
class DecomposeAten_SafeSoftmaxOp
    : public OpRewritePattern<Aten_SafeSoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_SafeSoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    BaseTensorType resultTensorType = cast<BaseTensorType>(op.getType());
    if (!resultTensorType.hasDtype() || !resultTensorType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have sizes and dtype");
    }
    SmallVector<int64_t> sizes(resultTensorType.getSizes());

    int64_t dimInt;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
      return rewriter.notifyMatchFailure(op, "Unsupported: non-constant dim");

    dimInt = toPositiveDim(dimInt, sizes.size());
    if (!isValidDim(dimInt, sizes.size()))
      return rewriter.notifyMatchFailure(op, "dim int is not valid");

    Location loc = op.getLoc();
    Value softmax = rewriter.create<AtenSoftmaxIntOp>(
        loc, op.getType(), op.getSelf(), op.getDim(), op.getDtype());

    Type resultTensorDtype = resultTensorType.getDtype();

    Value negInfinity = getConstantWithGivenDtypeAndValue(
        rewriter, loc, -std::numeric_limits<double>::infinity(),
        resultTensorDtype);

    auto boolDtype = rewriter.getI1Type();
    auto boolTensorType =
        resultTensorType.getWithSizesAndDtype(sizes, boolDtype);
    Value masked = rewriter.create<AtenEqScalarOp>(loc, boolTensorType,
                                                   op.getSelf(), negInfinity);

    sizes[dimInt] = 1;
    auto maskedRowsType =
        resultTensorType.getWithSizesAndDtype(sizes, boolDtype);
    Value cstTrue =
        rewriter.create<Torch::ConstantBoolOp>(loc, rewriter.getBoolAttr(true));
    Value maskedRows = rewriter.create<AtenAllDimOp>(
        loc, maskedRowsType, masked, op.getDim(), cstTrue);
    Value cstZero = getConstantWithGivenDtypeAndValue(rewriter, loc, 0.0,
                                                      resultTensorDtype);
    rewriter.replaceOpWithNewOp<AtenWhereScalarSelfOp>(
        op, resultTensorType, maskedRows, cstZero, softmax);
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

    BaseTensorType tensorType = cast<BaseTensorType>(gradOutput.getType());
    if (!tensorType.hasDtype() || !isa<mlir::FloatType>(tensorType.getDtype()))
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

    BaseTensorType tensorType = cast<BaseTensorType>(gradOutput.getType());
    if (!tensorType.hasDtype() || !isa<mlir::FloatType>(tensorType.getDtype()))
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

    BaseTensorType tensorType = cast<BaseTensorType>(gradOutput.getType());
    if (!tensorType.hasDtype() || !isa<mlir::FloatType>(tensorType.getDtype()))
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
template <typename OpTy, typename DecompOpTy>
class DecomposeAtenAminAmaxOp : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool keepDim;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(
          op, "Expected a constant boolean value for keepDim");

    Value input = op.getSelf();
    auto inputTy = dyn_cast<Torch::ValueTensorType>(input.getType());
    if (!inputTy || !inputTy.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "Expected input type having sizes");
    }

    SmallVector<int64_t, 4> dims;
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(dims)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    if (dims.size() == 0) {
      dims = llvm::to_vector(llvm::seq<int64_t>(0, inputTy.getSizes().size()));
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
      static_assert(std::is_same_v<OpTy, AtenAmaxOp> ||
                    std::is_same_v<OpTy, AtenAminOp>);
      if (std::is_same_v<OpTy, AtenAmaxOp>) {
        input = createMaxAlongDimension(rewriter, loc, op, input, dim, keepDim);
      } else if (std::is_same_v<OpTy, AtenAminOp>) {
        input = createMinAlongDimension(rewriter, loc, op, input, dim, keepDim);
      }
    }
    rewriter.replaceOp(op, input);
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
    Value result = op.getResult();

    bool keepDim;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
      return rewriter.notifyMatchFailure(
          op, "expected keepdim to be a constant bool");
    }
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    BaseTensorType indicesTensorType = cast<BaseTensorType>(result.getType());
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank || *maybeInputRank == 0) {
      return rewriter.notifyMatchFailure(
          op, "expected input tensor to have a rank > 0");
    }
    unsigned inputRank = *maybeInputRank;
    if (!indicesTensorType.hasSizes())
      return failure();
    BaseTensorType valueTensorType = cast<BaseTensorType>(
        inputType.getWithSizesAndDtype(indicesTensorType.getOptionalSizes(),
                                       inputType.getOptionalDtype()));

    // If the dim type is `NoneType` i.e. reduce along all the dimensions.
    // `AtenMaxDimOp` and `AtenMinDimOp` do not support dim as `NoneType` so
    // first the input tensor is flattened to 1d tensor and then the reduction
    // happens on the 0th dimension.
    if (isa<Torch::NoneType>(dim.getType())) {
      Value zero = rewriter.create<ConstantIntOp>(loc, 0);
      Value falseValue = rewriter.create<ConstantBoolOp>(loc, false);
      if (inputType.getSizes().size() > 1) {
        int64_t flattenSize = Torch::kUnknownSize;
        if (inputType.areAllSizesKnown()) {
          flattenSize = 1;
          for (int64_t sze : inputType.getSizes())
            flattenSize *= sze;
        }
        auto flattenType = cast<BaseTensorType>(inputType.getWithSizesAndDtype(
            {flattenSize}, inputType.getOptionalDtype()));
        Value end = rewriter.create<ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(inputRank - 1));
        input = rewriter.create<AtenFlattenUsingIntsOp>(loc, flattenType, input,
                                                        zero, end);
      }
      Value resultIndices =
          rewriter
              .create<DecompOpTy>(
                  loc,
                  valueTensorType.getWithSizesAndDtype(
                      ArrayRef<int64_t>{}, valueTensorType.getOptionalDtype()),
                  indicesTensorType.getWithSizesAndDtype(
                      ArrayRef<int64_t>{},
                      indicesTensorType.getOptionalDtype()),
                  input, /*dim=*/zero, /*keepdim=*/falseValue)
              .getIndices();
      if (keepDim) {
        Value one =
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
        Value dimList = rewriter.create<PrimListConstructOp>(
            loc,
            Torch::ListType::get(Torch::IntType::get(rewriter.getContext())),
            SmallVector<Value>(inputRank, one));
        resultIndices = rewriter.create<AtenReshapeOp>(
            loc,
            indicesTensorType.getWithSizesAndDtype(
                SmallVector<int64_t>(inputRank, 1),
                indicesTensorType.getOptionalDtype()),
            resultIndices, dimList);
      }
      rewriter.replaceOp(op, resultIndices);
      return success();
    } else {
      Value resultIndices =
          rewriter
              .create<DecompOpTy>(loc, valueTensorType, indicesTensorType,
                                  input, dim, op.getKeepdim())
              .getIndices();
      rewriter.replaceOp(op, resultIndices);
      return success();
    }
  }
};
} // namespace

// Decompose `AtenAminmaxOp` to `AtenAminOp` + `AtenAmaxOp`
namespace {
class DecomposeAtenAminmaxOp : public OpRewritePattern<AtenAminmaxOp> {
public:
  using OpRewritePattern<AtenAminmaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAminmaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Torch::ListType listType =
        rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>());
    Value dimList;
    if (isa<Torch::NoneType>(op.getDim().getType())) {
      dimList = rewriter.create<Torch::PrimListConstructOp>(loc, listType,
                                                            ArrayRef<Value>{});
    } else {
      dimList = rewriter.create<Torch::PrimListConstructOp>(
          loc, listType, ArrayRef<Value>{op.getDim()});
    }

    auto amin = rewriter.create<AtenAminOp>(
        loc, op.getMin().getType(), op.getSelf(), dimList, op.getKeepdim());
    auto amax = rewriter.create<AtenAmaxOp>(
        loc, op.getMax().getType(), op.getSelf(), dimList, op.getKeepdim());
    rewriter.replaceOp(op, {amin, amax});
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
    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: input must have known sizes");
    }
    ArrayRef<int64_t> inputShape = inputType.getSizes();

    Value boundaries = op.getBoundaries();
    auto boundariesType = cast<BaseTensorType>(boundaries.getType());
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
  BaseTensorType tensorType = cast<BaseTensorType>(self.getType());
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
    if (!isa<Torch::NoneType>(op.getDtype().getType()))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for log_softmax");

    BaseTensorType tensorType = cast<BaseTensorType>(self.getType());
    if (!tensorType.hasDtype() || !isa<mlir::FloatType>(tensorType.getDtype()))
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

namespace {
class DecomposeAtenLogSigmoidOp : public OpRewritePattern<AtenLogSigmoidOp> {
public:
  using OpRewritePattern<AtenLogSigmoidOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLogSigmoidOp op,
                                PatternRewriter &rewriter) const override {
    Value sigmoid =
        rewriter.create<AtenSigmoidOp>(op.getLoc(), op.getType(), op.getSelf());
    rewriter.replaceOpWithNewOp<AtenLogOp>(op, op.getType(), sigmoid);
    return success();
  }
};
} // namespace

// SoftShrink(x, lambda) function:
// Applies a shrinkage function where:
// - If x > lambda, returns x - lambda
// - If x < -lambda, returns x + lambda
// - Otherwise, returns 0
namespace {
class DecomposeAtenSoftshrinkOp : public OpRewritePattern<AtenSoftshrinkOp> {
public:
  using OpRewritePattern<AtenSoftshrinkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftshrinkOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value lambdValue = op.getLambd();

    auto resTy = dyn_cast<ValueTensorType>(op.getType());
    if (!resTy || !resTy.hasDtype() || !resTy.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "result should have dtype and size");
    }

    double lambd;
    if (!matchPattern(lambdValue, m_TorchConstantFloat(&lambd))) {
      return rewriter.notifyMatchFailure(
          op, "expected lambd to be a constant float");
    }

    Value zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value neglambd = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(-lambd));
    Value poslambd = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(lambd));

    Value constOneFloat =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));

    auto boolResType =
        resTy.getWithSizesAndDtype(resTy.getSizes(), rewriter.getI1Type());

    Value posMask =
        rewriter.create<AtenGtScalarOp>(loc, boolResType, self, poslambd);
    Value negMask =
        rewriter.create<AtenLtScalarOp>(loc, boolResType, self, neglambd);

    Value posValue = rewriter.create<AtenSubScalarOp>(loc, resTy, self,
                                                      poslambd, constOneFloat);
    Value negValue = rewriter.create<AtenAddScalarOp>(loc, resTy, self,
                                                      neglambd, constOneFloat);

    Value result = rewriter.create<AtenWhereScalarOtherOp>(loc, resTy, posMask,
                                                           posValue, zero);
    result =
        rewriter.create<AtenWhereSelfOp>(loc, resTy, negMask, negValue, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// HardShrink(x, lambda) function:
// Applies a shrinkage function where:
// - If x > lambda, returns x
// - If x < -lambda, returns x
// - Otherwise, returns 0
namespace {
class DecomposeAtenHardshrinkOp : public OpRewritePattern<AtenHardshrinkOp> {
public:
  using OpRewritePattern<AtenHardshrinkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardshrinkOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value lambdValue = op.getLambd();

    auto resTy = dyn_cast<ValueTensorType>(op.getType());
    if (!resTy || !resTy.hasDtype() || !resTy.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "result should have dtype and size");
    }

    double lambd;
    if (!matchPattern(lambdValue, m_TorchConstantFloat(&lambd))) {
      return rewriter.notifyMatchFailure(
          op, "expected lambd to be a constant float");
    }

    Value zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value neglambd = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(-lambd));
    Value poslambd = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(lambd));

    auto boolResType =
        resTy.getWithSizesAndDtype(resTy.getSizes(), rewriter.getI1Type());

    Value posMask =
        rewriter.create<AtenGtScalarOp>(loc, boolResType, self, poslambd);
    Value negMask =
        rewriter.create<AtenLtScalarOp>(loc, boolResType, self, neglambd);

    Value result = rewriter.create<AtenWhereScalarOtherOp>(loc, resTy, posMask,
                                                           self, zero);
    result =
        rewriter.create<AtenWhereSelfOp>(loc, resTy, negMask, self, result);

    rewriter.replaceOp(op, result);
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

// https://github.com/pytorch/pytorch/blob/9dec41b684a4284c4e052e295314c23f0f942fec/torch/_refs/__init__.py#L3229
// Decompose aten.renorm into: linalg_vector_norm
namespace {
class DecomposeAtenRenormOp : public OpRewritePattern<AtenRenormOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRenormOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value dim = op.getDim();
    Value p = op.getP();
    Value maxnorm = op.getMaxnorm();

    // Prepare all necessary variables
    auto ndim = getTensorRank(self);
    auto resType = cast<BaseTensorType>(self.getType());

    if (!resType.hasDtype() || !resType.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "result should have dtype and sizes");
    }

    Type dtype = resType.getDtype();
    if (isa<mlir::ComplexType>(dtype)) {
      return rewriter.notifyMatchFailure(
          op, "lowering of aten.renorm for complex inputs dtype is "
              "currently unimplemented");
    }

    SmallVector<int64_t> inputSize(resType.getSizes());

    // Convert dim from Value to int
    int64_t dimInt;
    if (!matchPattern(dim, m_TorchConstantInt(&dimInt)))
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented: dim not constant int");

    // Define all constants
    Value cstTrue = rewriter.create<ConstantBoolOp>(loc, true);
    Value cstZero = rewriter.create<Torch::ConstantIntOp>(loc, 0);
    Value cstOne = rewriter.create<Torch::ConstantIntOp>(loc, 1);
    Value cstNone = rewriter.create<ConstantNoneOp>(loc);

    // Arragne reduce_dims tensor (vector), [0, 1, ... , dim-1, dim+1, ... ,
    // ndim-1]
    llvm::SmallVector<Value> reduceDimsVector;
    for (uint64_t i = 0; i < ndim; i++) {
      if (i == (uint64_t)dimInt)
        continue;

      Value constI = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i));

      reduceDimsVector.push_back(constI);
    }

    Value reduceDimsList = rewriter.create<Torch::PrimListConstructOp>(
        loc,
        rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
        reduceDimsVector);

    // Make output shape for linalg.vector_norm operation
    SmallVector<Value> inputSizeValue;
    for (uint64_t i = 0; i < inputSize.size(); i++) {
      if (i != (uint64_t)dimInt)
        inputSize[i] = 1;

      inputSizeValue.push_back(
          rewriter.create<Torch::ConstantIntOp>(loc, inputSize[i]));
    }

    // Prepare arguments for linalg.vector_norm
    Value dtypeValue;
    Type vectorNormOutType;

    if (isa<mlir::Float16Type, mlir::BFloat16Type>(dtype)) {
      dtype = cast<Type>(rewriter.getF32Type());
      dtypeValue = getDtypeIntValueForType(rewriter, loc, dtype);
      vectorNormOutType = resType.getWithSizesAndDtype(inputSize, dtype);
    } else {
      dtypeValue = cstNone;
      vectorNormOutType = resType.getWithSizesAndDtype(inputSize, dtype);
    }

    auto norm = rewriter.create<AtenLinalgVectorNormOp>(
        loc, vectorNormOutType, self, p, reduceDimsList, cstTrue, dtypeValue);

    // Define epsiolon constant 10^-7
    mlir::FloatType f64Type = rewriter.getF64Type();
    Value epsValue = rewriter.create<ConstantFloatOp>(
        loc, rewriter.getFloatAttr(f64Type, 1e-7));

    Value normPlusEps = rewriter.create<AtenAddScalarOp>(
        loc, vectorNormOutType, norm, epsValue, cstOne);

    Value maxnormTensorValue = rewriter.create<AtenFullLikeOp>(
        loc, normPlusEps.getType(), normPlusEps, maxnorm, cstNone, cstNone,
        cstNone, cstNone, cstNone);

    // Divide maxnorm and normPlusEps
    auto divideMaxnormAndNorm = rewriter.create<AtenDivTensorOp>(
        loc, vectorNormOutType, maxnormTensorValue, normPlusEps);

    // Next few lines corespond to this pythorch code: norm_factor =
    // torch.where(norm > maxnorm, maxnorm / (norm + eps), 1.0)
    auto boolTensorType = rewriter.getType<ValueTensorType>(
        cast<BaseTensorType>(vectorNormOutType).getOptionalSizes(),
        rewriter.getI1Type());

    Value greaterThanMaxnorm =
        rewriter.create<AtenGtScalarOp>(loc, boolTensorType, norm, maxnorm);

    Value cstOnetensor = rewriter.create<AtenFullLikeOp>(
        loc, normPlusEps.getType(), normPlusEps, cstOne, cstNone, cstNone,
        cstNone, cstNone, cstNone);

    auto normFactor = rewriter.create<AtenWhereSelfOp>(
        loc, vectorNormOutType, greaterThanMaxnorm, divideMaxnormAndNorm,
        cstOnetensor);

    // Converte norm_factor to input dtype
    Value normFactorFinal = rewriter.create<PrimsConvertElementTypeOp>(
        loc, resType.getWithSizesAndDtype(inputSize, resType.getDtype()),
        normFactor, getDtypeIntValueForType(rewriter, loc, resType.getDtype()));

    // Multiply input tensor with norm factor
    auto output = rewriter.create<AtenMulTensorOp>(loc, self.getType(), self,
                                                   normFactorFinal);

    rewriter.replaceOpWithNewOp<AtenContiguousOp>(op, self.getType(), output,
                                                  /*memory_format*/ cstZero);

    return success();
  }
};
} // namespace

// Decompose aten.linalg_cross into: aten.broadcast_to, aten.index_select,
// aten.add.Tensor and aten.mull.Tensor. See
// https://github.com/pytorch/pytorch/blob/ed3c256b61f05720843454a9282aa7c903da2c81/torch/_refs/linalg/__init__.py#L70.
// def linalg_cross(self: Tensor, other: Tensor, dim: int = -1):
//     broadcast_shape = compute_broadcast_shape(self, other)
//     a = torch.broadcast_to(self, broadcast_shape)
//     b = torch.broadcast_to(other, broadcast_shape)
//     idx = torch.arange(3)
//     return a.index_select(dim, (idx + 1) % 3) *
//            b.index_select(dim, (idx + 2) % 3) -
//            a.index_select(dim, (idx + 2) % 3) *
//            b.index_select(dim, (idx + 1) % 3)
namespace {
class DecomposeAtenLinalgCrossOp : public OpRewritePattern<AtenLinalgCrossOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLinalgCrossOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value other = op.getOther();
    Type opType = op.getType();
    Value dim = op.getDim();

    auto resType = cast<BaseTensorType>(self.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    Type dtype = resType.getDtype();
    if (isa<mlir::ComplexType>(dtype)) {
      return rewriter.notifyMatchFailure(
          op, "lowering of aten.linalg_cross for complex inputs dtype is "
              "currently unimplemented");
    }

    // calculate common shape for broadcast
    SmallVector<int64_t> broadcastShape;
    SmallVector<Value> broadcastShapeValue;
    computeBroadcastShape(rewriter, loc, self, other, broadcastShape,
                          broadcastShapeValue);

    Type broadcastType = ValueTensorType::get(
        op.getContext(), llvm::ArrayRef(broadcastShape), dtype);

    Value indexBroadcastShapeTorchList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        broadcastShapeValue);

    // broadcast tensors to common shape
    auto a = rewriter.create<AtenBroadcastToOp>(loc, broadcastType, self,
                                                indexBroadcastShapeTorchList);
    auto b = rewriter.create<AtenBroadcastToOp>(loc, broadcastType, other,
                                                indexBroadcastShapeTorchList);

    // create constants
    Value constOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constTwo = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(2));
    Value constThree = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(3));
    Value none = rewriter.create<ConstantNoneOp>(loc);

    // idx = torch.arange(3)
    auto outType = dyn_cast<BaseTensorType>(opType);
    auto arangeType = outType.getWithSizesAndDtype(
        llvm::ArrayRef<int64_t>(3),
        IntegerType::get(op.getContext(), 64, IntegerType::Signed));
    auto idx = rewriter.create<AtenArangeOp>(
        loc, arangeType, constThree, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none);

    // (idx + 1) and (idx + 2)
    auto idxPlusOne = rewriter.create<AtenAddScalarOp>(loc, arangeType, idx,
                                                       constOne, constOne);
    auto idxPlusTwo = rewriter.create<AtenAddScalarOp>(loc, arangeType, idx,
                                                       constTwo, constOne);

    // (idx + 1) % 3 and (idx + 2) % 3
    auto idxPlusOneRemainderThree = rewriter.create<AtenRemainderScalarOp>(
        loc, arangeType, idxPlusOne, constThree);
    auto idxPlusTwoRemainderThree = rewriter.create<AtenRemainderScalarOp>(
        loc, arangeType, idxPlusTwo, constThree);

    // a.index_select(dim, (idx + 1) % 3) * b.index_select(dim, (idx + 2) % 3)
    auto idxSelectAPlusOne = rewriter.create<AtenIndexSelectOp>(
        loc, opType, a, dim, idxPlusOneRemainderThree);
    auto idxSelectBPlusTwo = rewriter.create<AtenIndexSelectOp>(
        loc, opType, b, dim, idxPlusTwoRemainderThree);
    auto firstMul = rewriter.create<AtenMulTensorOp>(
        loc, opType, idxSelectAPlusOne, idxSelectBPlusTwo);

    // a.index_select(dim, (idx + 2) % 3) * b.index_select(dim, (idx + 1) % 3)
    auto idxSelectAPlusTwo = rewriter.create<AtenIndexSelectOp>(
        loc, opType, a, dim, idxPlusTwoRemainderThree);
    auto idxSelectBPlusOne = rewriter.create<AtenIndexSelectOp>(
        loc, opType, b, dim, idxPlusOneRemainderThree);
    auto secondMul = rewriter.create<AtenMulTensorOp>(
        loc, opType, idxSelectAPlusTwo, idxSelectBPlusOne);

    // subtract the results of the two multiplications from above
    rewriter.replaceOpWithNewOp<AtenSubTensorOp>(op, opType, firstMul,
                                                 secondMul, constOne);

    return success();
  }
};
} // namespace

// decompose aten.linalg_slogdet into: aten.sgn, aten.log, aten.abs
// aten.linalg_det
namespace {

class DecomposeAtenLinalgSlogdetOp
    : public OpRewritePattern<AtenLinalgSlogdetOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLinalgSlogdetOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 2> results = op.getResults();
    Location loc = op.getLoc();
    Value input = op.getA();
    Value determinant = rewriter.create<Torch::AtenLinalgDetOp>(
        loc, results[0].getType(), input);
    Value sign =
        rewriter.create<AtenSgnOp>(loc, determinant.getType(), determinant);
    Value abs_det =
        rewriter.create<AtenAbsOp>(loc, determinant.getType(), determinant);
    Value ln_abs_det =
        rewriter.create<AtenLogOp>(loc, abs_det.getType(), abs_det);
    rewriter.replaceAllUsesWith(results[0], sign);
    rewriter.replaceAllUsesWith(results[1], ln_abs_det);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {

class DecomposeAten_LinalgDetOp : public OpRewritePattern<Aten_LinalgDetOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_LinalgDetOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 3> results = op.getResults();
    if (!results[1].use_empty() || !results[2].use_empty())
      return rewriter.notifyMatchFailure(
          op, "unsupported: _linalg_det results: LU and pivot");
    Location loc = op.getLoc();
    Value input = op.getA();
    Value determinant = rewriter.create<Torch::AtenLinalgDetOp>(
        loc, results[0].getType(), input);
    rewriter.replaceAllUsesWith(results[0], determinant);
    rewriter.eraseOp(op);
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
    auto inType = cast<BaseTensorType>(inValue.getType());
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
  BaseTensorType inputType = cast<BaseTensorType>(input.getType());

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
    auto resType = cast<BaseTensorType>(op.getType());
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
    auto resType = cast<BaseTensorType>(op.getType());
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
    auto resType = cast<BaseTensorType>(op.getType());
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
class DecomposeAtenRreluWithNoiseBackwardOp
    : public OpRewritePattern<AtenRreluWithNoiseBackwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRreluWithNoiseBackwardOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.getGradOutput();
    Value self = op.getSelf();
    Value noise = op.getNoise();
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    bool training;
    if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training))) {
      return rewriter.notifyMatchFailure(op,
                                         "training should be a bool constant");
    }

    bool selfIsResult = false;
    if (!matchPattern(op.getSelfIsResult(),
                      m_TorchConstantBool(&selfIsResult)) ||
        selfIsResult)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: self_is_result should be false");

    double lower, upper;
    if (!matchPattern(op.getLower(), m_TorchConstantFloat(&lower)) ||
        !matchPattern(op.getUpper(), m_TorchConstantFloat(&upper))) {
      return rewriter.notifyMatchFailure(
          op, "lower and upper should be float constants");
    }

    if (training && (upper - lower > 0.000001)) {
      Value rreluWithNoiseBackwardOutput =
          rewriter.create<AtenMulTensorOp>(loc, resType, gradOutput, noise);
      rewriter.replaceOp(op, rreluWithNoiseBackwardOutput);
    } else {
      double negative_slope = (upper + lower) / 2;
      Value cstNegativeSlope = rewriter.create<ConstantFloatOp>(
          loc, rewriter.getF64FloatAttr(negative_slope));
      rewriter.replaceOpWithNewOp<AtenLeakyReluBackwardOp>(
          op, resType, gradOutput, self, cstNegativeSlope,
          op.getSelfIsResult());
    }
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenPreluOp : public OpRewritePattern<AtenPreluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenPreluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value weight = op.getWeight();
    auto resType = cast<ValueTensorType>(op.getType());
    auto boolTensorType = rewriter.getType<ValueTensorType>(
        resType.getOptionalSizes(), rewriter.getI1Type());
    Value zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value inputMulWeight =
        rewriter.create<AtenMulTensorOp>(loc, resType, input, weight);
    Value lessThanZero =
        rewriter.create<AtenLtScalarOp>(loc, boolTensorType, input, zero);
    Value preluOutput = rewriter.create<AtenWhereSelfOp>(
        loc, resType, lessThanZero, inputMulWeight, input);

    rewriter.replaceOp(op, preluOutput);
    return success();
  }
};

} // namespace

// rrelu = max(0, x) + min(0, alpha * x)
// if in training mode, the alpha is sampled from uniform distribution (lower,
// upper) if in testing mode, the alpha is (lower + upper) / 2
namespace {
class DecomposeAtenRreluOp : public OpRewritePattern<AtenRreluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRreluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value lower = op.getLower();
    Value upper = op.getUpper();
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    bool training;
    if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training))) {
      return rewriter.notifyMatchFailure(op, "training should be a constant");
    }

    Value constantZeroFloat =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value constantOneFloat =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value constantTwoFloat =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(2.0));

    Value alpha;
    if (training) {
      // Create a uniform random op with low and high set to `lower` and
      // `upper`, respectively.
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value emptyTensor = rewriter.create<AtenFullLikeOp>(
          loc, resType, self, constantZeroFloat, /*dtype=*/none,
          /*layout=*/none,
          /*device=*/none, /*pin_memoty=*/none, /*memory_format=*/none);
      alpha = rewriter.create<AtenUniformOp>(loc, resType, emptyTensor,
                                             /*from=*/lower, /*to=*/upper,
                                             /*generator=*/none);
    } else {
      Value half = rewriter.create<AtenAddOp>(loc, constantTwoFloat.getType(),
                                              lower, upper);
      alpha = rewriter.create<AtenDivOp>(loc, constantTwoFloat.getType(), half,
                                         constantTwoFloat);
    }

    Value zeroTensor =
        createRank0Tensor(rewriter, loc, resType, constantZeroFloat);
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, self);

    Value scaledSelf;
    if (training) {
      scaledSelf = rewriter.create<AtenMulTensorOp>(loc, resType, self, alpha);
    } else {
      scaledSelf = rewriter.create<AtenMulScalarOp>(loc, resType, self, alpha);
    }

    Value negativeOutput =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, scaledSelf);
    Value rreluOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, negativeOutput, constantOneFloat);
    rewriter.replaceOp(op, rreluOutput);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenRreluWithNoiseOp
    : public OpRewritePattern<AtenRreluWithNoiseOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRreluWithNoiseOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value noise = op.getNoise();
    Value lower = op.getLower();
    Value upper = op.getUpper();
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    bool training;
    if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training))) {
      return rewriter.notifyMatchFailure(op, "training should be a constant");
    }

    Value constantZeroFloat =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value constantOneFloat =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value constantTwoFloat =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(2.0));

    Value alpha;
    if (training) {
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value emptyTensor = rewriter.create<AtenFullLikeOp>(
          loc, resType, self, constantZeroFloat, /*dtype=*/none,
          /*layout=*/none,
          /*device=*/none, /*pin_memoty=*/none, /*memory_format=*/none);
      alpha = rewriter.create<AtenUniformOp>(loc, resType, emptyTensor,
                                             /*from=*/lower, /*to=*/upper,
                                             /*generator=*/none);
    } else {
      Value half = rewriter.create<AtenAddOp>(loc, constantTwoFloat.getType(),
                                              lower, upper);
      alpha = rewriter.create<AtenDivOp>(loc, constantTwoFloat.getType(), half,
                                         constantTwoFloat);
    }

    Value zeroTensor =
        createRank0Tensor(rewriter, loc, resType, constantZeroFloat);
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, self);

    Value scaledSelf;
    if (training) {
      scaledSelf = rewriter.create<AtenMulTensorOp>(loc, resType, self, alpha);
      auto boolResType = resType.getWithSizesAndDtype(resType.getSizes(),
                                                      rewriter.getI1Type());
      Value oneTensor =
          createRank0Tensor(rewriter, loc, resType, constantOneFloat);
      Value not_positive = rewriter.create<AtenLtScalarOp>(
          loc, boolResType, self, constantZeroFloat);
      noise = rewriter.create<AtenWhereSelfOp>(loc, resType, not_positive,
                                               alpha, oneTensor);
    } else {
      scaledSelf = rewriter.create<AtenMulScalarOp>(loc, resType, self, alpha);
    }

    Value negativeOutput =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, scaledSelf);
    Value rreluOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, negativeOutput, constantOneFloat);
    rewriter.replaceOp(op, rreluOutput);
    return success();
  }
};
} // namespace

// CELU(x)=max(0,x)+min(0,alpha∗(exp(x/alpha)−1))
namespace {
class DecomposeAtenCeluOp : public OpRewritePattern<AtenCeluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCeluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value alpha = op.getAlpha();
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }

    Value constantZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value constantOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));

    // positiveOutput = max(0,x)
    Value zeroTensor = createRank0Tensor(rewriter, loc, resType, constantZero);
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, input);

    // negativeOutput = min(0,alpha∗(exp(x/alpha)−1))
    Value scaledInput =
        rewriter.create<AtenDivScalarOp>(loc, resType, input, alpha);
    Value expX = rewriter.create<AtenExpOp>(loc, resType, scaledInput);
    Value expXM1 = rewriter.create<AtenSubScalarOp>(loc, resType, expX,
                                                    constantOne, constantOne);
    Value scaledExpXM1 =
        rewriter.create<AtenMulScalarOp>(loc, resType, expXM1, alpha);
    Value negativeOutput =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, scaledExpXM1);
    Value celuOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, negativeOutput, constantOne);

    rewriter.replaceOp(op, celuOutput);
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
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    auto start = op.getSelf();
    auto inputType = cast<BaseTensorType>(start.getType());

    auto delta = rewriter.create<AtenSubTensorOp>(loc, inputType, op.getEnd(),
                                                  start, cstOne);

    auto weightedDelta =
        rewriter.create<AtenMulScalarOp>(loc, inputType, delta, op.getWeight());
    auto lerp = rewriter.create<AtenAddTensorOp>(loc, resType, start,
                                                 weightedDelta, cstOne);
    rewriter.replaceOp(op, lerp);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenLerpTensorOp : public OpRewritePattern<AtenLerpTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLerpTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    auto start = op.getSelf();
    auto inputType = cast<BaseTensorType>(start.getType());

    auto delta = rewriter.create<AtenSubTensorOp>(loc, inputType, op.getEnd(),
                                                  start, cstOne);

    auto weightedDelta =
        rewriter.create<AtenMulTensorOp>(loc, inputType, delta, op.getWeight());
    auto lerp = rewriter.create<AtenAddTensorOp>(loc, resType, start,
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
    auto resType = cast<BaseTensorType>(op.getType());
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
    auto resType = cast<BaseTensorType>(op.getType());
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
      BaseTensorType tensorType = cast<BaseTensorType>(tensor.getType());
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
        cast<BaseTensorType>(op.getType())
            .getWithSizesAndDtype(
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

// Decompose `aten.hstack` into `aten.at_least1d` and `aten.cat`.
// https://github.com/pytorch/pytorch/blob/207564bab1c4fe42750931765734ee604032fb69/torch/_refs/__init__.py#L3908
namespace {
class DecomposeAtenHstackOp : public OpRewritePattern<AtenHstackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHstackOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get SmallVector<Value> from Value.
    SmallVector<Value> tensors;
    if (!getListConstructElements(op.getTensors(), tensors))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the tensor list is not from list construct");

    // Execute AtenAtleast1dOp on every tensor inside tensors.
    SmallVector<Value> atleast1dTensors;
    for (auto tensor : tensors) {
      std::optional<unsigned> tensorRank = getTensorRank(tensor);

      // Check if the tensor is already of rank >= 1.
      if (*tensorRank < 1) {
        auto atleast1dTensor =
            rewriter.create<AtenAtleast1dOp>(loc, tensor.getType(), tensor);
        atleast1dTensors.push_back(atleast1dTensor);
      } else {
        atleast1dTensors.push_back(tensor);
      }
    }

    // Make Value list from atleast1dTensors variable.
    auto elemType = cast<BaseTensorType>(atleast1dTensors[0].getType())
                        .getWithSizesAndDtype(std::nullopt, nullptr);
    Value atleast1dTensorList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(elemType), atleast1dTensors);

    // Replace hstack with cat operator.
    if (getTensorRank(atleast1dTensors[0]) == 1)
      rewriter.replaceOpWithNewOp<AtenCatOp>(
          op, op.getType(), atleast1dTensorList,
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0)));
    else
      rewriter.replaceOpWithNewOp<AtenCatOp>(
          op, op.getType(), atleast1dTensorList,
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1)));

    return success();
  }
};
} // namespace

// Decompose `aten.column_stack` into `aten.reshape` and `aten.cat`.
// https://github.com/pytorch/pytorch/blob/207564bab1c4fe42750931765734ee604032fb69/torch/_refs/__init__.py#L2822
namespace {
class DecomposeAtenColumnStackOp : public OpRewritePattern<AtenColumnStackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenColumnStackOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> tensors;
    if (!getListConstructElements(op.getTensors(), tensors))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the tensor list is not from list construct");

    for (auto tensor : tensors) {
      auto tTy = dyn_cast<BaseTensorType>(tensor.getType());
      if (!tTy || !tTy.hasSizes())
        return rewriter.notifyMatchFailure(
            op, "unimplemented: one tensor does not have known sizes");
    }

    SmallVector<Value> tensors2d;
    for (auto tensor : tensors) {
      auto tTy = dyn_cast<BaseTensorType>(tensor.getType());
      SmallVector<int64_t> tSizes(tTy.getSizes());
      if (tSizes.size() <= 1) {
        if (tSizes.size() == 0) {
          tSizes.push_back(1);
        }
        tSizes.push_back(1);
        auto newTy = tTy.getWithSizesAndDtype(tSizes, tTy.getDtype());
        SmallVector<Value> newShapeList;
        for (auto tSize : tSizes) {
          newShapeList.push_back(rewriter.create<ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(tSize)));
        }
        auto newShape = rewriter.create<PrimListConstructOp>(
            loc, Torch::ListType::get(rewriter.getType<IntType>()),
            newShapeList);
        Value tensor2d =
            rewriter.create<AtenReshapeOp>(loc, newTy, tensor, newShape);
        tensors2d.push_back(tensor2d);
      } else {
        tensors2d.push_back(tensor);
      }
    }

    auto elemType = cast<BaseTensorType>(tensors2d[0].getType())
                        .getWithSizesAndDtype(std::nullopt, nullptr);
    Value newTensors = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(elemType), tensors2d);

    rewriter.replaceOpWithNewOp<AtenCatOp>(
        op, op.getType(), newTensors,
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1)));

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
    auto selfTy = cast<BaseTensorType>(self.getType());
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

// Decompose aten.repeat into aten.squeeze, aten.unsqueeze, and aten.broadcast.
//
// Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
namespace {
class DecomposeAtenRepeatOp : public OpRewritePattern<AtenRepeatOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRepeatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    auto selfTy = cast<BaseTensorType>(self.getType());
    if (!selfTy.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: no implementation for rankless tensor");

    SmallVector<Value> repeats;
    if (!getListConstructElements(op.getRepeats(), repeats))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: repeats not list of Scalar");

    int64_t rank = selfTy.getSizes().size();
    if (rank > static_cast<int64_t>(repeats.size())) {
      return rewriter.notifyMatchFailure(
          op, "repeats are not matched with self's rank");
    }

    int64_t repeatSz = repeats.size();
    int64_t batch = repeatSz - rank;

    if (!selfTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "input sizes unknown");

    // Materialize out 1 dimensions to broadcast along. This includes
    // materializing out preceding batch dimensions:
    for (int i = 0; i < repeatSz; ++i) {
      auto oldSizes = selfTy.getSizes();
      llvm::SmallVector<int64_t> sizes;
      int64_t squeezeDim = i < batch ? i : i * 2 - batch;

      for (int j = 0; j < squeezeDim; ++j)
        sizes.push_back(oldSizes[j]);
      sizes.push_back(1);
      for (int j = squeezeDim, s = oldSizes.size(); j < s; j++)
        sizes.push_back(oldSizes[j]);

      Value dim = rewriter.create<Torch::ConstantIntOp>(loc, squeezeDim);
      selfTy =
          rewriter.getType<ValueTensorType>(sizes, selfTy.getOptionalDtype());
      self = rewriter.create<AtenUnsqueezeOp>(loc, selfTy, self, dim);
    }

    llvm::SmallVector<Value> lengths;
    for (int i = 0; i < repeatSz; ++i) {
      if (i < batch) {
        lengths.push_back(repeats[i]);
        continue;
      }

      Value iv = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i * 2 + 1 - batch));
      Value dim = rewriter.create<AtenSizeIntOp>(loc, self, /*dim=*/iv);
      lengths.push_back(repeats[i]);
      lengths.push_back(dim);
    }

    Value lengthv = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(rewriter.getType<IntType>()), lengths);

    llvm::SmallVector<int64_t> expandShape(selfTy.getSizes());
    for (int i = 0; i < repeatSz; ++i) {
      int64_t repeatDim = i < batch ? i : i * 2 - batch;
      int64_t repeat;
      if (!matchPattern(repeats[i], m_TorchConstantInt(&repeat)))
        repeat = Torch::kUnknownSize;
      expandShape[repeatDim] = repeat;
    }

    auto mulDim = [](int64_t lhs, int64_t rhs) {
      if (lhs == Torch::kUnknownSize || rhs == Torch::kUnknownSize)
        return Torch::kUnknownSize;
      return lhs * rhs;
    };

    BaseTensorType expandTy = rewriter.getType<ValueTensorType>(
        expandShape, selfTy.getOptionalDtype());
    Value expand =
        rewriter.create<AtenBroadcastToOp>(loc, expandTy, self, lengthv);

    for (int i = 0; i < rank; ++i) {
      auto oldShape = expandTy.getSizes();
      llvm::SmallVector<int64_t> newShape;
      int64_t flattenDim = i + batch;
      for (int j = 0; j < flattenDim; ++j)
        newShape.push_back(oldShape[j]);
      newShape.push_back(
          mulDim(oldShape[flattenDim], oldShape[flattenDim + 1]));
      for (int j = flattenDim + 2, s = oldShape.size(); j < s; ++j)
        newShape.push_back(oldShape[j]);

      expandTy = rewriter.getType<ValueTensorType>(newShape,
                                                   expandTy.getOptionalDtype());

      // Used to keep the return type the same on the last flatten:
      expandTy = i < rank - 1 ? expandTy : cast<BaseTensorType>(op.getType());

      Value start = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(flattenDim));
      Value end = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(flattenDim + 1));
      expand = rewriter.create<AtenFlattenUsingIntsOp>(loc, expandTy, expand,
                                                       start, end);
    }

    rewriter.replaceOp(op, expand);
    return success();
  }
};
} // namespace

// decompose aten.repeat_interleave.self_int into following ops:
// aten.flatten.using_ints, aten.unsqueeze, aten.tile, aten.reshape
namespace {

class DecomposeAtenRepeatInterleaveSelfIntOp
    : public OpRewritePattern<AtenRepeatInterleaveSelfIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRepeatInterleaveSelfIntOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();
    Value self = op.getSelf();
    auto selfTy = cast<BaseTensorType>(self.getType());
    if (!selfTy.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: no implementation for rankless tensor");
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: no implementation for rankless tensor");

    int64_t inputRank = selfTy.getSizes().size();
    int64_t repeats;
    if (!matchPattern(op.getRepeats(), m_TorchConstantInt(&repeats)))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: repeats not constant int");

    bool dimIsNone = false;
    int64_t dim;
    Value dimValue = op.getDim();
    if (isa<Torch::NoneType>(dimValue.getType())) {
      dimIsNone = true;
      dim = inputRank - 1;
    } else {
      if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
        return rewriter.notifyMatchFailure(
            op, "Unimplemented: dim not constant int");
      dim = toPositiveDim(dim, inputRank);
    }

    dimValue =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(dim));
    Value dimValuePlusOne = rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(dim + 1));

    auto unsqueezedInfo = unsqueezeTensor(rewriter, op, self, dimValuePlusOne);
    if (failed(unsqueezedInfo))
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor op");
    self = *unsqueezedInfo;

    Value constMinusOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(-1));
    SmallVector<Value> expandShapeValueList(inputRank + 1, constMinusOne);
    expandShapeValueList[dim + 1] = rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(repeats));
    Value expandShapeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), expandShapeValueList);
    Value constFalse =
        rewriter.create<ConstantBoolOp>(loc, rewriter.getBoolAttr(false));

    SmallVector<int64_t> expandShape(inputRank + 1);
    for (int64_t i = 0; i <= dim; i++) {
      expandShape[i] = selfTy.getSizes()[i];
    }
    expandShape[dim + 1] = repeats;
    for (int64_t i = dim + 1; i < inputRank; i++) {
      expandShape[i + 1] = selfTy.getSizes()[i];
    }

    BaseTensorType expandTy = rewriter.getType<ValueTensorType>(
        expandShape, selfTy.getOptionalDtype());

    Value expandSelf = rewriter.create<AtenExpandOp>(
        loc, expandTy, self, expandShapeList, constFalse);

    Value result;
    if (dimIsNone) {
      Value constZero =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
      result = rewriter.create<AtenFlattenUsingIntsOp>(
          loc, resType, expandSelf, constZero, constMinusOne);
    } else {
      result = rewriter.create<PrimsCollapseOp>(loc, resType, expandSelf,
                                                dimValue, dimValuePlusOne);
    }

    rewriter.replaceOp(op, result);
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
    BaseTensorType outputTensorType = cast<BaseTensorType>(op.getType());
    if (!outputTensorType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: output must have known sizes");

    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "unimplemented: unranked tensor");
    unsigned inputRank = *maybeRank;
    auto inputTensorType = cast<Torch::ValueTensorType>(self.getType());
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

    if (inputShape[dimInt] == Torch::kUnknownSize &&
        llvm::count(sizesInts, -1) > 0)
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: dynamic unflatten dim with an inferred size.");

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
    auto resType = cast<BaseTensorType>(op.getType());
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
    auto resType = cast<BaseTensorType>(op.getType());
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
    auto resType = cast<BaseTensorType>(op.getType());
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
    Value nan = op.getNan();
    Value posinf = op.getPosinf();
    Value neginf = op.getNeginf();
    auto outputType = cast<BaseTensorType>(op.getResult().getType());
    if (!outputType.hasDtype() ||
        !isa<mlir::FloatType>(outputType.getDtype())) {
      return rewriter.notifyMatchFailure(
          op, "expect output type to have float dtype");
    }
    mlir::FloatType outputElementType =
        cast<mlir::FloatType>(outputType.getDtype());

    if (isa<Torch::NoneType>(nan.getType())) {
      nan =
          rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    }
    if (isa<Torch::NoneType>(posinf.getType())) {
      posinf = rewriter.create<ConstantFloatOp>(
          loc, rewriter.getF64FloatAttr(
                   APFloat::getLargest(outputElementType.getFloatSemantics())
                       .convertToDouble()));
    }
    if (isa<Torch::NoneType>(neginf.getType())) {
      neginf = rewriter.create<ConstantFloatOp>(
          loc, rewriter.getF64FloatAttr(
                   APFloat::getLargest(outputElementType.getFloatSemantics(),
                                       /*Negative=*/true)
                       .convertToDouble()));
    }

    auto compareType = outputType.getWithSizesAndDtype(
        outputType.getOptionalSizes(), rewriter.getI1Type());
    Value isNan =
        rewriter.create<Torch::AtenIsnanOp>(loc, compareType, op.getSelf());
    Value where = rewriter.create<Torch::AtenWhereScalarSelfOp>(
        loc, outputType, isNan, nan, op.getSelf());
    Value isposinf =
        rewriter.create<Torch::AtenIsposinfOp>(loc, compareType, where);
    where = rewriter.create<Torch::AtenWhereScalarSelfOp>(
        loc, outputType, isposinf, posinf, where);
    Value isneginf =
        rewriter.create<Torch::AtenIsneginfOp>(loc, compareType, where);
    rewriter.replaceOpWithNewOp<Torch::AtenWhereScalarSelfOp>(
        op, outputType, isneginf, neginf, where);
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
    auto resType = cast<BaseTensorType>(op.getType());
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

// Decompose aten.masked_fill.Tensor into aten.where.self op.
namespace {
class DecomposeAtenMaskedFillTensorOp
    : public OpRewritePattern<AtenMaskedFillTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMaskedFillTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto resType = cast<BaseTensorType>(op.getType());
    if (!resType.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, op.getMask(),
                                                 op.getValue(), op.getSelf());

    return success();
  }
};
} // namespace

// Decompose aten.masked_scatter:
// def masked_scatter(self: Tensor, mask: Tensor, source: Tensor) -> Tensor:
//     mask_int = mask + torch.zeros_like(self)
//     prefix_sum = torch.cumsum(mask_int.flatten(), dim=0)
//     mask_prefix = torch.clamp(prefix_sum - 1, min=0)
//     mask = mask.to(torch.bool)
//     source = source.flatten()[mask_prefix].reshape(mask.shape)
//     return torch.where(mask, source, self)
namespace {
class DecomposeAtenMaskedScatterOp
    : public OpRewritePattern<AtenMaskedScatterOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMaskedScatterOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();
    Value mask = op.getMask();
    Value source = op.getSource();
    Value self = op.getSelf();

    auto selfTy = cast<BaseTensorType>(self.getType());
    auto resTy = cast<BaseTensorType>(op.getType());
    auto sourceTy = cast<BaseTensorType>(source.getType());

    if (!resTy || !resTy.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result should have dtype");
    }
    if (!selfTy || !selfTy.areAllSizesKnown())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: no implementation for rankless tensor");
    if (!sourceTy || !sourceTy.areAllSizesKnown() || !sourceTy.hasDtype())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: no implementation for rankless tensor");

    int64_t selfNumel = getTensorNumel(self).value(); // as selfTy has sizes
    int64_t sourceNumel =
        getTensorNumel(source).value(); // as sourceTy has sizes
    int64_t selfRank = selfTy.getSizes().size();
    int64_t sourceRank = sourceTy.getSizes().size();

    Value constZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value constOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constNone = rewriter.create<ConstantNoneOp>(loc);
    Value selfLastDim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(selfRank - 1));
    Value sourceLastDim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(sourceRank - 1));

    auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);
    auto int64Dtype = getDtypeIntValueForType(
        rewriter, loc,
        rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true));
    auto selfIntType = selfTy.getWithSizesAndDtype(selfTy.getSizes(), si64Type);

    Value zerosLike = rewriter.create<Torch::AtenZerosLikeOp>(
        loc, selfIntType, self, int64Dtype, constNone, constNone, constNone,
        constNone);
    Value maskInt = rewriter.create<Torch::AtenAddTensorOp>(
        loc, selfIntType, mask, zerosLike, constOne);

    auto flattenMaskedType = selfTy.getWithSizesAndDtype(
        /*optionalSizes=*/{selfNumel}, si64Type);
    Value maskIntFlatten = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
        loc, flattenMaskedType, maskInt, constZero, selfLastDim);
    Value prefixSum = rewriter.create<Torch::AtenCumsumOp>(
        loc, flattenMaskedType, maskIntFlatten,
        /*dim=*/constZero, constNone);
    Value prefixSumMinusOne = rewriter.create<Torch::AtenSubScalarOp>(
        loc, flattenMaskedType, prefixSum, constOne, constOne);
    Value maskPrefix = rewriter.create<Torch::AtenClampOp>(
        loc, flattenMaskedType, prefixSumMinusOne, /*min=*/constZero,
        /*max=*/constNone);

    auto sourceFlattenType = sourceTy.getWithSizesAndDtype(
        /*optionalSizes=*/{sourceNumel}, sourceTy.getDtype());
    Value sourceFlatten = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
        loc, sourceFlattenType, source, constZero, sourceLastDim);

    auto selectSourceType = sourceTy.getWithSizesAndDtype(
        /*optionalSizes=*/{selfNumel}, sourceTy.getDtype());
    Value selectSource = rewriter.create<Torch::AtenIndexSelectOp>(
        loc, selectSourceType, sourceFlatten, constZero, maskPrefix);

    // Reshape normalized output back to the original input shape
    auto selfShape = rewriter.create<AtenSizeOp>(
        loc, Torch::ListType::get(IntType::get(context)), self);
    Value sourceReshape = rewriter.create<Torch::AtenViewOp>(
        loc, selfTy, selectSource, selfShape);
    rewriter.replaceOpWithNewOp<Torch::AtenWhereSelfOp>(op, resTy, mask,
                                                        sourceReshape, self);
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
  if (failed(getTransposedType(cast<Torch::BaseTensorType>(input.getType()),
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

// Decompose aten.conv_transpose1d to aten.convolution
namespace {
class DecomposeAtenConvTranspose1dOp
    : public OpRewritePattern<AtenConvTranspose1dOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvTranspose1dOp op,
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

// Decompose aten.conv_transpose3d to aten.convolution
namespace {
class DecomposeAtenConvTranspose3dOp
    : public OpRewritePattern<AtenConvTranspose3dInputOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvTranspose3dInputOp op,
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
    if (failed(getTransposedType(cast<BaseTensorType>(input.getType()), 0, 1,
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

      BaseTensorType gradOutputTy = cast<BaseTensorType>(gradOutput.getType());
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
          cast<BaseTensorType>(gradOutputTy.getWithSizesAndDtype(
              llvm::ArrayRef(gradOutputViewSizesInt),
              gradOutputTy.getOptionalDtype()));
      Value gradOutputView = rewriter.create<Torch::AtenViewOp>(
          loc, gradOutputTypeForView, gradOutput, gradOutputViewShapeList);

      BaseTensorType inputTransposedTy =
          cast<BaseTensorType>(inputTransposed.getType());
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
          cast<BaseTensorType>(inputTransposedTy.getWithSizesAndDtype(
              llvm::ArrayRef(gradWeightSizesInt),
              inputTransposedTy.getOptionalDtype()));

      Value numGroup = rewriter.create<AtenSizeIntOp>(loc, input, cstZero);
      gradWeight = rewriter.create<Torch::AtenConvolutionOp>(
          loc, gradWeightTy, inputTransposed, gradOutputView, cstNone,
          /*stride=*/op.getDilation(), op.getPadding(),
          /*dilation=*/op.getStride(), op.getTransposed(),
          op.getOutputPadding(), numGroup);

      BaseTensorType weightTy = cast<BaseTensorType>(weight.getType());
      if (!weightTy.hasSizes())
        return failure();
      SmallVector<int64_t> weightSizes(weightTy.getSizes());
      for (unsigned i = 0; i < gradWeightTy.getSizes().size() - 2; i++) {
        gradWeightSizesInt[i + 2] = weightSizes[i + 2];
        BaseTensorType gradWeightNarrowTy =
            cast<BaseTensorType>(gradWeightTy.getWithSizesAndDtype(
                llvm::ArrayRef(gradWeightSizesInt),
                gradWeightTy.getOptionalDtype()));

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
          cast<BaseTensorType>(gradWeightTy.getWithSizesAndDtype(
              llvm::ArrayRef(gradWeightViewShapeInt),
              gradWeightTy.getOptionalDtype()));
      gradWeight = rewriter.create<Torch::AtenViewOp>(
          loc, gradWeightTypeForView, gradWeight, gradWeightViewShapeList);

      gradWeightTy = cast<BaseTensorType>(gradWeight.getType());
      SmallVector<int64_t, 5> gradWeightDimsOrder =
          computeDimsOrderForMoveDim(0, 2, gradWeightViewShapeInt.size());
      SmallVector<int64_t, 5> gradWeightMoveDimShape;
      for (unsigned i = 0; i < gradWeightDimsOrder.size(); i++) {
        gradWeightMoveDimShape.push_back(
            gradWeightViewShapeInt[gradWeightDimsOrder[i]]);
      }
      BaseTensorType gradWeightTypeForMoveDim =
          cast<BaseTensorType>(gradWeightTy.getWithSizesAndDtype(
              llvm::ArrayRef(gradWeightMoveDimShape),
              gradWeightTy.getOptionalDtype()));

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
      if (failed(getTransposedType(cast<BaseTensorType>(gradOutput.getType()),
                                   0, 1, transposedType)))
        return failure();
      Value gradOutputTransposed = rewriter.create<Torch::AtenTransposeIntOp>(
          loc, transposedType, gradOutput, cstZero, cstOne);
      // Convolve input with grad_output.
      if (failed(getTransposedType(cast<BaseTensorType>(op.getResultTypes()[1]),
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
    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasDtype() || !isa<mlir::FloatType>(inputType.getDtype())) {
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
    BaseTensorType outputTensorType = cast<BaseTensorType>(output.getType());
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

    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasDtype() || !isa<mlir::FloatType>(inputType.getDtype()) ||
        !isNoneOrFloatDtype(context, dtype)) {
      return rewriter.notifyMatchFailure(
          op, "only floating-point type is supported");
    }

    SmallVector<Value> dimListElements;
    if (!getListConstructElements(dimList, dimListElements) &&
        !isa<Torch::NoneType>(dimList.getType())) {
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
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasDtype() || !isa<mlir::FloatType>(inputType.getDtype()))
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
    if (!isa<Torch::NoneType>(op.getTrain().getType())) {
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
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasDtype() || !isa<mlir::FloatType>(inputType.getDtype())) {
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
    BaseTensorType rank0FloatTensorTy = cast<BaseTensorType>(op.getType());
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
    BaseTensorType inputTensorTy = cast<BaseTensorType>(self.getType());
    if (!inputTensorTy.hasDtype() ||
        !isa<mlir::FloatType>(inputTensorTy.getDtype())) {
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
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());

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
    BaseTensorType inputTensorType = cast<BaseTensorType>(self.getType());
    if (!inputTensorType.hasDtype() ||
        !isa<mlir::FloatType>(inputTensorType.getDtype())) {
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

// Decompose aten.rot90
// github:
// https://github.com/pytorch/pytorch/blob/207564bab1c4fe42750931765734ee604032fb69/torch/_refs/__init__.py#L3830
namespace {
class DecomposeAtenRot90Op : public OpRewritePattern<AtenRot90Op> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRot90Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();

    // Convert dims from Value to SmallVector.
    SmallVector<Value> dims;
    if (!getListConstructElements(op.getDims(), dims))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dims not list of Scalar");

    // Convert k from Value to int
    int64_t k;
    if (!matchPattern(op.getK(), m_TorchConstantInt(&k)))
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented: k not constant int");

    k = (k % 4 + 4) %
        4; // This is equal to python code k = k % 4, because python and c++
           // have different implementation for operand %.

    if (k == 1) {
      Value flipDimList = rewriter.create<Torch::PrimListConstructOp>(
          loc,
          rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
          ArrayRef{dims[1]});

      Value flip =
          rewriter.create<AtenFlipOp>(loc, self.getType(), self, flipDimList);

      rewriter.replaceOpWithNewOp<Torch::AtenTransposeIntOp>(
          op, op.getType(), flip, dims[0], dims[1]);
    } else if (k == 2) {
      rewriter.replaceOpWithNewOp<AtenFlipOp>(op, op.getType(), self,
                                              op.getDims());
    } else if (k == 3) {
      Value flipDimList = rewriter.create<Torch::PrimListConstructOp>(
          loc,
          rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
          ArrayRef{dims[0]});

      Value flip =
          rewriter.create<AtenFlipOp>(loc, self.getType(), self, flipDimList);

      rewriter.replaceOpWithNewOp<Torch::AtenTransposeIntOp>(
          op, op.getType(), flip, dims[0], dims[1]);
    } else {
      rewriter.replaceOpWithNewOp<AtenCloneOp>(
          op, op.getType(), self,
          /*memory_format=*/
          rewriter.create<Torch::ConstantIntOp>(loc,
                                                rewriter.getI64IntegerAttr(0)));
    }

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
    BaseTensorType inputTensorType = cast<BaseTensorType>(self.getType());
    if (!inputTensorType.hasDtype() ||
        !isa<mlir::FloatType>(inputTensorType.getDtype())) {
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
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    auto resType = cast<BaseTensorType>(op.getType());
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
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    auto resType = cast<BaseTensorType>(op.getType());
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
    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasDtype() || !isa<mlir::FloatType>(inputType.getDtype())) {
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
  auto inputType = cast<BaseTensorType>(input.getType());
  auto probType = cast<BaseTensorType>(prob.getType());
  // Both the `input` and `prob` must be ranked tensors.
  if (!inputType.hasSizes() || !inputType.hasDtype() || !probType.hasSizes() ||
      !probType.hasDtype()) {
    return rewriter.notifyMatchFailure(
        op, "can't decompose bernoulli like ops without sizes or dtype");
  }
  // The `prob` is expected to be a float type tensor.
  if (!isa<mlir::FloatType>(probType.getDtype())) {
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
    if (!isa<Torch::NoneType>(op.getGenerator().getType()))
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
    if (!isa<Torch::NoneType>(op.getGenerator().getType()))
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");

    auto inputType = cast<BaseTensorType>(input.getType());
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
    if (!isa<Torch::NoneType>(op.getGenerator().getType()))
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
    if (!isa<Torch::NoneType>(op.getGenerator().getType()))
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
    if (!isa<Torch::NoneType>(op.getGenerator().getType()))
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

    auto input = cast<BaseTensorType>(op.getInput().getType());
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

    auto inputTy = cast<BaseTensorType>(op.getInput().getType());
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
    auto weightTy = cast<BaseTensorType>(weight.getType());
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
    auto biasTy = cast<BaseTensorType>(bias.getType());
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
class DecomposeAten_WeightNormInterfaceOp
    : public OpRewritePattern<Aten_WeightNormInterfaceOp> {
  using OpRewritePattern<Aten_WeightNormInterfaceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_WeightNormInterfaceOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value v = op.getV();
    Value g = op.getG();
    Value dim = op.getDim();

    auto inputType = cast<BaseTensorType>(v.getType());
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(op, "expected input to have sizes");

    if (!cast<ConstantIntOp>(dim.getDefiningOp()))
      return rewriter.notifyMatchFailure(op, "dim is not a ConstantIntOp");

    auto sizes = inputType.getSizes();
    SmallVector<Value> keepDims;
    for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()); ++i) {
      if (i !=
          static_cast<int64_t>(dim.getDefiningOp<ConstantIntOp>().getValue()))
        keepDims.push_back(
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i)));
    }

    Value ord =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(2));
    Value keepdim =
        rewriter.create<ConstantBoolOp>(loc, rewriter.getBoolAttr(true));
    Value dtypeNone = rewriter.create<ConstantNoneOp>(loc);

    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op->getContext())),
        keepDims);

    Value norm = rewriter.create<AtenLinalgVectorNormOp>(
        loc, v.getType(), v, ord, dimList, keepdim, dtypeNone);

    auto vShape = rewriter.create<AtenSizeOp>(
        loc, Torch::ListType::get(rewriter.getI64Type()), v);

    Value gDivNorm =
        rewriter.create<AtenDivTensorOp>(loc, g.getType(), g, norm);
    Value broadcastedGDivNorm =
        rewriter.create<AtenBroadcastToOp>(loc, v.getType(), gDivNorm, vShape);
    Value vMulBroadcastedGDivNorm = rewriter.create<AtenMulTensorOp>(
        loc, v.getType(), v, broadcastedGDivNorm);

    rewriter.replaceOp(op, ArrayRef<Value>{vMulBroadcastedGDivNorm, norm});
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

    auto inputTy = cast<BaseTensorType>(op.getInput().getType());
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
    if (!isa<Torch::NoneType>(weight.getType())) {
      out = rewriter.create<AtenMulTensorOp>(loc, out.getType(), out, weight);
    }
    if (!isa<Torch::NoneType>(bias.getType())) {
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
// The `prims.iota` op is converted to `aten.arange.startStep` op.
class DecomposePrimsIotaOp : public OpRewritePattern<PrimsIotaOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimsIotaOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int64_t length, start, step;
    if (!matchPattern(op.getLength(), m_TorchConstantInt(&length)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: low must be a constant integer");
    if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: low must be a constant integer");
    if (!matchPattern(op.getStep(), m_TorchConstantInt(&step)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: low must be a constant integer");
    auto endVal = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(start + length * step));
    auto none = rewriter.create<ConstantNoneOp>(loc);
    rewriter.replaceOpWithNewOp<AtenArangeStartStepOp>(
        op, op.getType(), op.getStart(), endVal, op.getStep(), op.getDtype(),
        none, op.getDevice(), none);
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

    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    Value numGroups = op.getNumGroups();
    Value eps = op.getEps();

    int64_t numGroupsInt;
    if (!matchPattern(numGroups, m_TorchConstantInt(&numGroupsInt)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: num_groups must be a constant int");

    Value cstZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));

    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(op, "input should have sizes.");

    SmallVector<int64_t> baseTypeSizes{inputType.getSizes()[0], numGroupsInt};
    auto baseType = inputType.getWithSizesAndDtype(
        baseTypeSizes, inputType.getOptionalDtype());

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
    auto inputType = cast<BaseTensorType>(input.getType());
    auto outputType = cast<BaseTensorType>(op.getResult0().getType());
    auto meanType = cast<BaseTensorType>(op.getResult1().getType());
    auto rsqrtVarType = cast<BaseTensorType>(op.getResult2().getType());
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
    int64_t numGroupsInt;
    if (!matchPattern(numGroups, m_TorchConstantInt(&numGroupsInt)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: num_groups must be a constant int");

    SmallVector<Value> newShape;
    SmallVector<int64_t> inputShapeInt{inputType.getSizes()};
    SmallVector<int64_t> reshapeInputShape{inputShapeInt[0], numGroupsInt};
    int64_t reshapeInputLastDim = 1;
    for (size_t i = 1; i < inputShapeInt.size(); i++) {
      if (inputShapeInt[i] == Torch::kUnknownSize) {
        reshapeInputLastDim = Torch::kUnknownSize;
        break;
      }
      reshapeInputLastDim *= inputShapeInt[i];
    }
    reshapeInputLastDim = reshapeInputLastDim == Torch::kUnknownSize
                              ? reshapeInputLastDim
                              : reshapeInputLastDim / numGroupsInt;
    reshapeInputShape.push_back(reshapeInputLastDim);

    newShape.push_back(rewriter.create<AtenSizeIntOp>(loc, input, cstZero));
    newShape.push_back(numGroups);
    newShape.push_back(cstNegtiveOne);
    Type reshapeInputType = inputType.getWithSizesAndDtype(
        reshapeInputShape, inputType.getOptionalDtype());
    Value reshapedInput = rewriter.create<AtenViewOp>(
        loc, reshapeInputType, input,
        rewriter.create<PrimListConstructOp>(
            loc, Torch::ListType::get(IntType::get(context)), newShape));

    // Now we proceed with the normalization steps across the 'groupSize'
    // Compute the mean and variance for each group
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        ArrayRef<Value>{cstNegtiveOne});

    reshapeInputShape[2] = 1;
    Type reductionType = inputType.getWithSizesAndDtype(
        reshapeInputShape, inputType.getOptionalDtype());
    auto mean =
        rewriter.create<AtenMeanDimOp>(loc, reductionType, reshapedInput,
                                       /*dims=*/dimList, /*keepdim=*/cstTrue,
                                       /*dtype=*/none);
    auto var =
        rewriter.create<AtenVarDimOp>(loc, reductionType, reshapedInput,
                                      /*dims=*/dimList, /*unbiased=*/cstFalse,
                                      /*keepdim=*/cstTrue);

    // Compute the normalized output: (input - mean) * rsqrt(var + eps)
    auto varPlusEps =
        rewriter.create<AtenAddScalarOp>(loc, reductionType, var, eps,
                                         /*alpha=*/cstOne);
    auto invStd = rewriter.create<AtenRsqrtOp>(loc, reductionType, varPlusEps);
    auto inputSubMean = rewriter.create<AtenSubTensorOp>(
        loc, reshapeInputType, reshapedInput, mean, /*alpha=*/cstOne);
    auto normalizedOutput = rewriter.create<AtenMulTensorOp>(
        loc, reshapeInputType, inputSubMean, invStd);

    // Reshape normalized output back to the original input shape
    auto inputShape = rewriter.create<AtenSizeOp>(
        loc, Torch::ListType::get(IntType::get(context)), input);
    auto reshapedOutput = rewriter.create<AtenViewOp>(
        loc, inputType, normalizedOutput, /*shape=*/inputShape);

    // Apply weight and bias if they are not None
    // Reshape weight and bias to C,1,1,...
    SmallVector<Value> viewShape = {channel};
    SmallVector<int64_t> viewShapeInt{inputShapeInt[1]};
    for (unsigned i = 2; i < inputType.getSizes().size(); i++) {
      viewShape.push_back(cstOne);
      viewShapeInt.push_back(1);
    }
    Value viewShapeSizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), viewShape);

    Type viewType = inputType.getWithSizesAndDtype(
        viewShapeInt, inputType.getOptionalDtype());
    Value groupNormOutput = reshapedOutput;
    if (!isa<Torch::NoneType>(weight.getType())) {
      auto weightReshaped = rewriter.create<AtenViewOp>(
          loc, viewType, weight, /*shape=*/viewShapeSizeList);
      groupNormOutput = rewriter.create<AtenMulTensorOp>(
          loc, inputType, groupNormOutput, weightReshaped);
    }
    if (!isa<Torch::NoneType>(bias.getType())) {
      auto biasReshaped = rewriter.create<AtenViewOp>(
          loc, viewType, bias, /*shape=*/viewShapeSizeList);
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
    if (isa<Torch::NoneType>(runningMean.getType()) ||
        isa<Torch::NoneType>(runningVar.getType()))
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
        cast<BaseTensorType>(runningMean.getType()).getSizes()[0];
    Type dtype = cast<ValueTensorType>(input.getType()).getOptionalDtype();
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
    if (!isa<Torch::NoneType>(weight.getType())) {
      // Rank of `weight` must be exactly 1.
      std::optional<unsigned> weightRank = getTensorRank(weight);
      if (!weightRank || *weightRank != 1)
        return rewriter.notifyMatchFailure(op, "expected weight to be rank 1");
      weight = rewriter.create<AtenViewOp>(loc, reshapeType, weight,
                                           runningStatsSizeList);
      batchNormOutput = rewriter.create<AtenMulTensorOp>(
          loc, batchNormOutput.getType(), batchNormOutput, weight);
    }
    if (!isa<Torch::NoneType>(bias.getType())) {
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
    if (isa<Torch::NoneType>(dtype.getType())) {
      BaseTensorType tensorType = cast<BaseTensorType>(op.getSelf().getType());
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
    BaseTensorType outTy = cast<BaseTensorType>(op.getType());
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

    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(op, "expected input to have sizes");

    BaseTensorType weightType = cast<BaseTensorType>(weight.getType());
    if (!weightType.hasSizes())
      return rewriter.notifyMatchFailure(op, "expected weight to have sizes");

    auto transposeWeight = [&]() -> Value {
      SmallVector<int64_t> transposeShape =
          llvm::to_vector(llvm::reverse(weightType.getSizes()));
      Type transposeType = weightType.getWithSizesAndDtype(
          llvm::ArrayRef(transposeShape), weightType.getOptionalDtype());
      Value transposeWeight =
          rewriter.create<AtenTOp>(loc, transposeType, weight);
      return transposeWeight;
    };

    if (isa<Torch::NoneType>(bias.getType())) {
      auto weightRank = weightType.getSizes().size();
      if (weightRank > 2 || weightRank <= 0)
        return rewriter.notifyMatchFailure(
            op, "expected weight's rank <= 2 && >= 1");
      if (weightRank == 1) {
        rewriter.replaceOpWithNewOp<AtenMatmulOp>(op, op.getType(), input,
                                                  weight);
        return success();
      } else if (weightRank == 2) {
        rewriter.replaceOpWithNewOp<AtenMatmulOp>(op, op.getType(), input,
                                                  transposeWeight());
        return success();
      }
      llvm_unreachable("unsupported weightRank");
    } else {
      BaseTensorType biasType = cast<BaseTensorType>(bias.getType());
      if (!biasType.hasSizes() || biasType.getSizes().size() != 1)
        return rewriter.notifyMatchFailure(op, "expected bias to be rank 1");

      // `weight` must be a rank 2 matrix.
      auto weightRank = weightType.getSizes().size();
      if (weightRank != 2)
        return rewriter.notifyMatchFailure(op,
                                           "expected weight to be a rank 2");

      Value matmul = rewriter.create<AtenMatmulOp>(loc, op.getType(), input,
                                                   transposeWeight());
      Value alpha =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
      rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), matmul,
                                                   op.getBias(), alpha);
      return success();
    }
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

    auto inputType = cast<BaseTensorType>(input.getType());
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
    BaseTensorType outTy = cast<BaseTensorType>(op.getType());
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
    if (isa<Torch::NoneType>(dtype.getType())) {
      BaseTensorType tensorType = cast<BaseTensorType>(op.getSelf().getType());
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
    auto resultType = cast<BaseTensorType>(op.getType());
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
    auto resultType = cast<BaseTensorType>(op.getType());
    if (!resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    auto srcTy = cast<BaseTensorType>(op.getSrc().getType());
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
    if (isa<Torch::NoneType>(dtype.getType())) {
      BaseTensorType tensorType = cast<BaseTensorType>(op.getSelf().getType());
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
// Decompose `aten.pad` op into `aten.constantPadNd` op.
class DecomposeAtenPadOp : public OpRewritePattern<AtenPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenPadOp op,
                                PatternRewriter &rewriter) const override {
    std::string mode;
    if (!matchPattern(op.getMode(), m_TorchConstantStr(mode)))
      return rewriter.notifyMatchFailure(op, "mode must be a constant string");

    if (mode == "constant") {
      Value value = op.getValue();
      if (isa<Torch::OptionalType>(value.getType()))
        return rewriter.notifyMatchFailure(op, "optional type not supported");
      if (isa<Torch::NoneType>(value.getType()))
        value = rewriter.create<Torch::ConstantFloatOp>(
            op.getLoc(), rewriter.getF64FloatAttr(0));

      rewriter.replaceOpWithNewOp<AtenConstantPadNdOp>(
          op, op.getType(), op.getSelf(), op.getPad(), value);
      return success();
    }

    SmallVector<Value> padValues;
    if (!getListConstructElements(op.getPad(), padValues))
      return failure();
    SmallVector<int64_t> padInts;
    Value usefulPads = op.getPad();
    uint64_t usefulPadIndexEnd = padValues.size();

    // try to reduce the number of padding dims if possible
    if (matchPattern(op.getPad(), m_TorchListOfConstantInts(padInts))) {
      if ((padInts.size() % 2) == 1)
        return rewriter.notifyMatchFailure(op,
                                           "expected an even number of pads");

      for (uint64_t i = padInts.size() - 1; i > 0; i -= 2) {
        if (padInts[i] != 0 || padInts[i - 1] != 0)
          break;
        usefulPadIndexEnd = i - 1;
      }
      if (usefulPadIndexEnd == 0) {
        rewriter.replaceOp(op, op.getSelf());
        return success();
      }
    }

    // we don't have support for 1-D replicate pad, so pass it as 2d if
    // possible.
    // TODO: add support for AtenReplicatePad1dOp and remove this.
    if (mode == "replicate" && usefulPadIndexEnd == 2 && padValues.size() >= 4)
      usefulPadIndexEnd = 4;

    // make a new list of padding ints if dimensionality reduction can be
    // performed
    if (usefulPadIndexEnd < padValues.size()) {
      ArrayRef<Value> usefulPadValues(padValues.begin(),
                                      padValues.begin() + usefulPadIndexEnd);
      usefulPads = rewriter.create<PrimListConstructOp>(
          op.getLoc(),
          rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
          usefulPadValues);
    }

    uint64_t numPadDims = usefulPadIndexEnd / 2;

    if (mode == "reflect") {
      // only support for relectionpad 1d and 2d
      if (numPadDims == 2) {
        rewriter.replaceOpWithNewOp<AtenReflectionPad2dOp>(
            op, op.getType(), op.getSelf(), usefulPads);
        return success();
      }
      if (numPadDims == 1) {
        rewriter.replaceOpWithNewOp<AtenReflectionPad1dOp>(
            op, op.getType(), op.getSelf(), usefulPads);
        return success();
      }
      return failure();
    }

    if (mode == "replicate") {
      // only support for replication pad 2d
      if (numPadDims != 2)
        return failure();
      rewriter.replaceOpWithNewOp<AtenReplicationPad2dOp>(
          op, op.getType(), op.getSelf(), usefulPads);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported mode: " + mode);
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
    if (!isa<Torch::NoneType>(op.getPinMemory().getType())) {
      bool pinMemory;
      if (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: pinMemory must be a constant");
      else if (pinMemory)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: pinMemory is expected to be false");
    }

    // TODO: Add support for device arg other than cpu.
    if (!isa<Torch::NoneType>(op.getDevice().getType())) {
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
    if (!isa<Torch::NoneType>(op.getLayout().getType())) {
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
// Decompose `aten.to.prim_Device` op into `aten.to.dtype` op.
class DecomposeAtenToPrimDeviceOp
    : public OpRewritePattern<AtenToPrimDeviceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenToPrimDeviceOp op,
                                PatternRewriter &rewriter) const override {

    // Device information isn't relevant to torch-mlir, so we can drop that info
    // here.
    auto loc = op.getLoc();
    Value constNone = rewriter.create<ConstantNoneOp>(loc);

    Value dtype = op.getDtype();
    if (isa<Torch::NoneType>(dtype.getType())) {
      dtype = rewriter.create<Torch::PrimDtypeOp>(loc, op.getSelf());
    }
    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(op, op.getType(), op.getSelf(),
                                               dtype, op.getNonBlocking(),
                                               op.getCopy(), constNone);

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
// Decompose `aten.adaptive_max_pool1d` op into `aten.max_pool1d_with_indices`
// op.
class DecomposeAtenAdaptiveMaxPool1dOp
    : public OpRewritePattern<AtenAdaptiveMaxPool1dOp> {
  using OpRewritePattern<AtenAdaptiveMaxPool1dOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAdaptiveMaxPool1dOp op,
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

    int64_t outputSizeInt;
    if (!matchPattern(outputSize, m_TorchConstantInt(&outputSizeInt))) {
      return rewriter.notifyMatchFailure(
          op, "the output size of adaptive_max_pool1d must be a constant int");
    }

    SmallVector<Value, 1> kernelSize;
    if (outputSizeInt == 1) {
      BaseTensorType inputTensorType = cast<BaseTensorType>(input.getType());
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
    Value dialationList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        ValueRange{constantOne});

    if (op.getResult(1).use_empty()) {
      auto maxPool = rewriter.create<AtenMaxPool1dOp>(
          loc, op.getType(0), input, kernelSizeList, strideList,
          paddingSizeList, dialationList,
          /*ceil_mode=*/constantFalse);
      rewriter.replaceOp(op, {maxPool.getResult(), Value()});
    } else {
      auto maxPool = rewriter.create<AtenMaxPool1dWithIndicesOp>(
          loc, op.getType(0), op.getType(1), input, kernelSizeList, strideList,
          paddingSizeList, dialationList,
          /*ceil_mode=*/constantFalse);
      rewriter.replaceOp(op, maxPool.getResults());
    }
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
      BaseTensorType inputTensorType = cast<BaseTensorType>(input.getType());
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
// kernelH = inH - [(outH - 1) * strideH] = strideH
// kernelW = inW - [(outW - 1) * strideW] = strideW
// paddingH = 0, paddingW = 0
//
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
    // inH % outH != 0 or inW % outW != 0 where
    // the stride/kernel size is not fixed.
    // The following logic of stride/kernel size derivation is consistent
    // with torch/_decomp/decomposations.py:adaptive_avg_pool2d.
    Value constantZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value constantTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(loc);

    SmallVector<Value, 2> strideSize;
    SmallVector<Value, 2> kernelSize;
    for (unsigned i = 0; i < inputHW.size(); i++) {
      Value remainder = rewriter.create<AtenRemainderIntOp>(
          loc, inputHW[i], outputShapeSizesTorchInt[i]);

      // Filter cases with fixed stride size.
      Value cond1 = rewriter.create<Torch::AtenGtIntOp>(
          loc, outputShapeSizesTorchInt[i],
          rewriter.create<Torch::AtenMulIntOp>(
              loc, remainder,
              rewriter.create<Torch::AtenSubIntOp>(
                  loc, outputShapeSizesTorchInt[i], constantOne)));
      rewriter.create<RuntimeAssertOp>(
          loc, cond1,
          "unimplemented: only support cases with fixed stride size.");

      // Filter cases with fixed kernel size.
      // cond2: whether input_size % output_size == 0.
      Value cond2 =
          rewriter.create<Torch::AtenEqIntOp>(loc, remainder, constantZero);
      // cond3: whether output_size % (input_size % output_size) == 0.
      // To avoid potential crash (eg. tosa) happens,choose to mod 1 (add
      // offset) when remainder equals 0, which has no side effect on
      // effectiveness.
      Value offset = rewriter.create<Torch::AtenIntBoolOp>(
          loc, rewriter.create<Torch::Aten__Not__Op>(
                   loc, rewriter.create<Torch::AtenBoolIntOp>(loc, remainder)));
      Value remainder_not_zero =
          rewriter.create<Torch::AtenAddIntOp>(loc, remainder, offset);
      Value cond3 = rewriter.create<Torch::AtenEqIntOp>(
          loc,
          rewriter.create<Torch::AtenRemainderIntOp>(
              loc, outputShapeSizesTorchInt[i], remainder_not_zero),
          constantZero);
      Value cond = rewriter.create<Torch::Aten__Or__BoolOp>(loc, cond2, cond3);

      rewriter.create<RuntimeAssertOp>(
          loc, cond,
          "unimplemented: only support cases with fixed kernel size.");

      Value stride = rewriter.create<Torch::AtenFloordivIntOp>(
          loc, inputHW[i], outputShapeSizesTorchInt[i]);
      strideSize.emplace_back(stride);

      Value kernel = rewriter.create<Torch::AtenFloordivIntOp>(
          loc, inputHW[i], outputShapeSizesTorchInt[i]);

      // When remainder equals 0, it is no need for kernel to add 1
      // and just keep the same as stride, otherwise it is necessary
      // to add 1 (torch/_decomp/decomposations.py:adaptive_avg_pool2d).
      Value boolMod = rewriter.create<Torch::AtenBoolIntOp>(loc, remainder);
      Value intMod = rewriter.create<Torch::AtenIntBoolOp>(loc, boolMod);

      kernel = rewriter.create<Torch::AtenAddIntOp>(loc, kernel, intMod);
      kernelSize.emplace_back(kernel);
    }

    Value kernelSizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), kernelSize);
    Value strideList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), strideSize);
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
class DecomposeAtenRad2degOp : public OpRewritePattern<AtenRad2degOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRad2degOp op,
                                PatternRewriter &rewriter) const override {
    Value constant180OverPi = rewriter.create<Torch::ConstantFloatOp>(
        op.getLoc(), rewriter.getF64FloatAttr(180 / 3.14159));
    rewriter.replaceOpWithNewOp<AtenMulScalarOp>(op, op.getType(), op.getSelf(),
                                                 constant180OverPi);
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
    Type dtype = cast<BaseTensorType>(x1.getType()).getOptionalDtype();
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
// decompose `trunc(x)` to `sign(x) * floor(abs(x))`
class DecomposeAtenTruncOp : public OpRewritePattern<AtenTruncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTruncOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();

    auto resultTy = dyn_cast<ValueTensorType>(op.getType());
    if (!resultTy || !resultTy.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result must have dtype");
    }

    if (isa<mlir::FloatType>(resultTy.getDtype())) {
      Value sign = rewriter.create<AtenSgnOp>(loc, resultTy, self);
      Value abs = rewriter.create<AtenAbsOp>(loc, resultTy, self);
      Value floor = rewriter.create<AtenFloorOp>(loc, resultTy, abs);
      rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, resultTy, sign, floor);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
// decompose `signbit(x)` to `view.dtype(x, si32/si64) < 0 `
class DecomposeAtenSignbitOp : public OpRewritePattern<AtenSignbitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSignbitOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();

    auto operandTy = dyn_cast<ValueTensorType>(self.getType());
    auto resultTy = dyn_cast<ValueTensorType>(op.getType());
    if (!operandTy || !operandTy.hasDtype() || !resultTy ||
        !resultTy.hasDtype()) {
      return rewriter.notifyMatchFailure(op,
                                         "operand and result must have dtype");
    }

    if (isa<mlir::FloatType>(operandTy.getDtype())) {
      mlir::IntegerType intType = rewriter.getIntegerType(
          operandTy.getDtype().getIntOrFloatBitWidth(), /*isSigned*/ true);
      Value dtype = getDtypeIntValueForType(rewriter, loc, intType);
      Value view = rewriter.create<AtenViewDtypeOp>(
          loc,
          operandTy.getWithSizesAndDtype(operandTy.getOptionalSizes(), intType),
          self, dtype);
      Value zero =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
      Value shift = rewriter.create<AtenLtScalarOp>(loc, resultTy, view, zero);
      rewriter.replaceOp(op, shift);
      return success();
    } else if (isa<mlir::IntegerType>(operandTy.getDtype())) {
      Value zero =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
      Value shift = rewriter.create<AtenLtScalarOp>(loc, resultTy, self, zero);
      rewriter.replaceOp(op, shift);
    }
    return failure();
  }
};
} // namespace

namespace {
// decompose `frac(x)` to `x - trunc(x)`
class DecomposeAtenFracOp : public OpRewritePattern<AtenFracOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFracOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    auto resultTy = op.getType();

    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value trunc = rewriter.create<AtenTruncOp>(loc, resultTy, self);
    rewriter.replaceOpWithNewOp<AtenSubTensorOp>(op, resultTy, self, trunc,
                                                 /*alpha=*/one);
    return success();
  }
};
} // namespace

namespace {
// decompose `copysign(x, y)` to `signbit(y) ? -abs(x) : abs(x)`
class DecomposeAtenCopysignTensorOp
    : public OpRewritePattern<AtenCopysignTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCopysignTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value other = op.getOther();
    auto selfTy = self.getType();
    auto otherTy = cast<BaseTensorType>(other.getType());
    auto resultTy = op.getType();

    Value signbit = rewriter.create<AtenSignbitOp>(
        loc,
        otherTy.getWithSizesAndDtype(otherTy.getOptionalSizes(),
                                     rewriter.getI1Type()),
        other);
    Value abs = rewriter.create<AtenAbsOp>(loc, selfTy, self);
    Value neg = rewriter.create<AtenNegOp>(loc, selfTy, abs);
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resultTy, signbit, neg,
                                                 abs);
    return success();
  }
};
} // namespace

namespace {
// decompose `ldexp(x, y)` to `x * 2^y`
class DecomposeAtenLdexpTensorOp : public OpRewritePattern<AtenLdexpTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLdexpTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value other = op.getOther();

    auto otherTy = dyn_cast<BaseTensorType>(other.getType());
    auto resultTy = dyn_cast<ValueTensorType>(op.getType());
    if (!resultTy || !resultTy.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result must have dtype");
    }

    Value exp2 = rewriter.create<AtenExp2Op>(
        loc,
        resultTy.getWithSizesAndDtype(otherTy.getOptionalSizes(),
                                      resultTy.getDtype()),
        other);
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, resultTy, self, exp2);
    return success();
  }
};
} // namespace

namespace {
// decompose `fmod(x, y)` to `x - trunc(x/y) * y`
class DecomposeAtenFmodTensorOp : public OpRewritePattern<AtenFmodTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFmodTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value other = op.getOther();

    auto resultTy = dyn_cast<ValueTensorType>(op.getType());
    if (!resultTy || !resultTy.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result must have dtype");
    }

    if (isa<mlir::IntegerType>(resultTy.getDtype())) {
      Value div = rewriter.create<AtenDivTensorOp>(loc, resultTy, self, other);
      Value mul = rewriter.create<AtenMulTensorOp>(loc, resultTy, div, other);
      Value alpha =
          rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
      rewriter.replaceOpWithNewOp<AtenSubTensorOp>(op, resultTy, self, mul,
                                                   alpha);
      return success();
    } else if (isa<mlir::FloatType>(resultTy.getDtype())) {
      Value div = rewriter.create<AtenDivTensorOp>(loc, resultTy, self, other);
      Value trunc = rewriter.create<AtenTruncOp>(loc, resultTy, div);
      Value mul = rewriter.create<AtenMulTensorOp>(loc, resultTy, trunc, other);
      Value alpha =
          rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
      rewriter.replaceOpWithNewOp<AtenSubTensorOp>(op, resultTy, self, mul,
                                                   alpha);
      return success();
    }
    return failure();
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
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    BaseTensorType resultType =
        cast<BaseTensorType>(op->getResult(0).getType());
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
        rewriter.create<Torch::ConstantStrOp>(op.getLoc(), "floor");
    rewriter.replaceOpWithNewOp<AtenDivTensorModeOp>(
        op, op.getType(), op.getSelf(), op.getOther(),
        /*roundingMode=*/cstStrFloor);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenFloorDivideScalarOp
    : public OpRewritePattern<AtenFloorDivideScalarOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFloorDivideScalarOp op,
                                PatternRewriter &rewriter) const override {
    Value cstStrFloor =
        rewriter.create<Torch::ConstantStrOp>(op.getLoc(), "floor");
    rewriter.replaceOpWithNewOp<AtenDivScalarModeOp>(
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
  BaseTensorType inputTensorTy = cast<BaseTensorType>(self.getType());
  Type outputType = op.getType();
  BaseTensorType outputTensorType = cast<BaseTensorType>(outputType);
  if (!outputTensorType.hasDtype()) {
    return rewriter.notifyMatchFailure(op,
                                       "expected result type to have a dtype");
  }
  Type newOutputType = outputTensorType.getWithSizesAndDtype(
      outputTensorType.getSizes(), rewriter.getF64Type());
  if (!inputTensorTy.hasDtype() ||
      !isa<mlir::FloatType>(inputTensorTy.getDtype())) {
    return rewriter.notifyMatchFailure(
        op, "support floating-point type input only");
  }

  // Upcasting the input tensor to `F64` dtype for higher precision during the
  // computation of the result.
  if (inputTensorTy.getDtype().getIntOrFloatBitWidth() != 64) {
    self = convertTensorToDtype(rewriter, loc, self, rewriter.getF64Type());
    inputTensorTy = cast<BaseTensorType>(self.getType());
  }

  std::optional<unsigned> maybeInputRank = getTensorRank(self);
  if (!maybeInputRank) {
    return rewriter.notifyMatchFailure(op, "expected input to have a rank");
  }
  unsigned inputRank = *maybeInputRank;
  SmallVector<Value> dimListElements;
  bool isNoneOrEmpty = true;
  if (!isa<Torch::NoneType>(dimList.getType())) {
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
        rewriter, op, cast<BaseTensorType>(meanDimResultType),
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
    if (!isa<Torch::NoneType>(op.getCorrection().getType())) {
      if (isa<Torch::FloatType>(op.getCorrection().getType())) {
        if (!matchPattern(op.getCorrection(),
                          m_TorchConstantFloat(&correctionValFloat)))
          return rewriter.notifyMatchFailure(
              op, "Only support constant int or float correction value for "
                  "aten.var");
      } else if (isa<Torch::IntType>(op.getCorrection().getType())) {
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
    BaseTensorType resultType = cast<BaseTensorType>(op.getType());
    BaseTensorType inputType = cast<BaseTensorType>(op.getSelf().getType());
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "Expected the input tensor to have sizes");
    BaseTensorType subType = cast<BaseTensorType>(
        inputType.getWithSizesAndDtype(llvm::ArrayRef(inputType.getSizes()),
                                       resultType.getOptionalDtype()));

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
class DecomposeAtenL1LossOp : public OpRewritePattern<AtenL1LossOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenL1LossOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    auto selfTy = dyn_cast<BaseTensorType>(self.getType());
    if (!selfTy || !selfTy.hasSizes() || !selfTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "Expected self to be a tensor with sizes and a dtype");
    }

    Value target = op.getTarget();
    auto targetTy = dyn_cast<BaseTensorType>(target.getType());
    if (!targetTy || !targetTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "Expected target to be a tensor with sizes and a dtype");
    }

    auto outTy = dyn_cast<BaseTensorType>(op.getType());
    if (!outTy || !outTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "Expected output type to be a tensor with a dtype");
    }

    auto outDtype = outTy.getDtype();
    if (selfTy.getDtype() != outDtype) {
      self = convertTensorToDtype(rewriter, loc, self, outDtype);
    }
    if (targetTy.getDtype() != outDtype) {
      target = convertTensorToDtype(rewriter, loc, target, outDtype);
    }

    Value reduction = op.getReduction();
    int64_t reductionInt;
    if (!matchPattern(reduction, m_TorchConstantInt(&reductionInt))) {
      return rewriter.notifyMatchFailure(
          op, "Expected reduction to be a constant int");
    }

    auto subTy = outTy.getWithSizesAndDtype(selfTy.getSizes(), outDtype);
    Value sub = createTensorSub(rewriter, loc, subTy, self, target);
    Value abs = rewriter.create<AtenAbsOp>(loc, subTy, sub);

    if (reductionInt == 0) {
      rewriter.replaceOp(op, abs);
    } else if (reductionInt == 1) {
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value sum = rewriter.create<AtenSumOp>(loc, outTy, abs, none);
      Value numel = rewriter.create<AtenNumelOp>(loc, abs);
      Value mean = rewriter.create<AtenDivScalarOp>(loc, outTy, sum, numel);
      rewriter.replaceOp(op, mean);
    } else {
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value sum = rewriter.create<AtenSumOp>(loc, outTy, abs, none);
      rewriter.replaceOp(op, sum);
    }

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
    if (isa<Torch::NoneType>(ord.getType())) {
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
    BaseTensorType resultTensorType = cast<BaseTensorType>(resultType);
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
        cast<BaseTensorType>(resultTensorType.getWithSizesAndDtype(
            resultTensorType.getSizes(), rewriter.getF32Type()));
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
    if (!isa<Torch::NoneType>(op.getOutputDtype().getType()))
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
    auto resultType = cast<BaseTensorType>(op.getType());

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
    if (!isa<Torch::NoneType>(op.getMemoryFormat().getType())) {
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
    auto resultType = cast<BaseTensorType>(op.getType());

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
class DecomposeAtenLinspaceOp : public OpRewritePattern<AtenLinspaceOp> {
public:
  using OpRewritePattern<AtenLinspaceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLinspaceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = getContext();

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));

    Value addStart;
    int64_t steps;
    auto si64Type = rewriter.getIntegerType(/*width=*/64, /*isSigned*/ true);
    auto fp32Type = rewriter.getF32Type();
    auto arangeIntType =
        getTensorTypeFromShapeValues({op.getSteps()}, si64Type);
    auto arangeFp32Type =
        getTensorTypeFromShapeValues({op.getSteps()}, fp32Type);
    if (matchPattern(op.getSteps(), m_TorchConstantInt(&steps)) && steps == 1) {
      // specically handle steps == 1
      Value arange = rewriter.create<AtenArangeStartOp>(
          loc, arangeIntType, zero, op.getSteps(), /*dtype=*/none,
          op.getLayout(), op.getDevice(), op.getPinMemory());
      if (isa<Torch::FloatType>(op.getEnd().getType()) ||
          isa<Torch::FloatType>(op.getStart().getType())) {
        addStart = rewriter.create<AtenAddScalarOp>(loc, arangeFp32Type, arange,
                                                    op.getStart(), one);
      } else {
        addStart = rewriter.create<AtenAddScalarOp>(loc, arangeIntType, arange,
                                                    op.getStart(), one);
      }
    } else {
      // handle steps != 1 or dynamic steps
      Value neOrNot = rewriter.create<AtenNeIntOp>(loc, op.getSteps(), one);
      rewriter.create<RuntimeAssertOp>(
          loc, neOrNot,
          rewriter.getStringAttr("linspace's dynamic steps must not be 1"));
      // create arange: [0, ..., steps - 1]
      Value arange = rewriter.create<AtenArangeStartOp>(
          loc, arangeIntType, zero, op.getSteps(), /*dtype=*/none,
          op.getLayout(), op.getDevice(), op.getPinMemory());
      // calculate (end - start) / (steps - 1)
      Value sub;
      if (isa<Torch::FloatType>(op.getEnd().getType()) ||
          isa<Torch::FloatType>(op.getStart().getType())) {
        sub = rewriter.create<AtenSubOp>(loc, Torch::FloatType::get(context),
                                         op.getEnd(), op.getStart());
      } else {
        sub = rewriter.create<AtenSubIntOp>(loc, op.getEnd(), op.getStart());
      }
      Value div = rewriter.create<AtenDivOp>(
          loc, sub, rewriter.create<AtenSubIntOp>(loc, op.getSteps(), one));
      // calculate [0, ..., steps - 1] * ((end - start) / (steps - 1)) + start
      Value mulScalar =
          rewriter.create<AtenMulScalarOp>(loc, arangeFp32Type, arange, div);
      addStart = rewriter.create<AtenAddScalarOp>(
          loc, arangeFp32Type, mulScalar, op.getStart(), one);
    }
    // to dtype
    Value result;
    if (!isa<Torch::NoneType>(op.getDtype().getType())) {
      result = rewriter.create<AtenToDtypeOp>(
          loc, op.getType(), addStart, op.getDtype(),
          /*non_blocking=*/falseVal,
          /*copy=*/falseVal, /*memory_format=*/none);
    } else {
      Value f32Type = rewriter.create<ConstantIntOp>(
          loc, (int)torch_upstream::ScalarType::Float);
      result = rewriter.create<AtenToDtypeOp>(
          loc, op.getType(), addStart, f32Type, /*non_blocking=*/falseVal,
          /*copy=*/falseVal, /*memory_format=*/none);
    }
    rewriter.replaceOp(op, result);
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
    rewriter.replaceOpWithNewOp<Torch::TensorStaticInfoCastOp>(op, op.getType(),
                                                               result);
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
    Value reduction = op.getReduction();
    int64_t reductionInt;
    if (!matchPattern(reduction, m_TorchConstantInt(&reductionInt))) {
      return rewriter.notifyMatchFailure(op,
                                         "reduction should be a constant int!");
    }

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

    Type secondType;
    if (reductionInt == 0) {
      secondType = target.getType();
    } else {
      auto targetType = dyn_cast<BaseTensorType>(target.getType());
      secondType = targetType.getWithSizesAndDtype({}, targetType.getDtype());
    }

    Value nllLoss =
        rewriter
            .create<AtenNllLossForwardOp>(
                loc, op.getType(), secondType, logSoftmax, target,
                op.getWeight(), op.getReduction(), op.getIgnoreIndex())
            ->getResult(0);
    rewriter.replaceOp(op, nllLoss);
    return success();
  }
};
} // namespace

namespace {
// Decompose aten::nll_loss_forward according to :
//  torch/_decomp/decompositions.py and
//  https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html.
// The (self, target) can be:
//  1. [N, C] and [C],
//   or
//  2. [N] or [].
// The weight must be None or 1d where the numel must keep consistent with the
// number of classes.
class DecomposeAtenNllLossForwardOp
    : public OpRewritePattern<AtenNllLossForwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNllLossForwardOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = op.getContext();

    auto self = op.getSelf();
    auto target = op.getTarget();

    auto selfType = dyn_cast<BaseTensorType>(self.getType());
    auto targetType = dyn_cast<BaseTensorType>(target.getType());

    // constraints.
    if (!selfType.hasSizes() || !targetType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "require self and target having sizes!");
    }

    if (!selfType.hasDtype() || !targetType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "require self and target having dtype!");
    }

    auto selfSizes = selfType.getSizes();
    auto targetSizes = targetType.getSizes();
    int64_t selfRank = selfSizes.size();
    int64_t targetRank = targetSizes.size();
    if (selfRank <= 0 or selfRank > 2) {
      return rewriter.notifyMatchFailure(op, "input tensor should be 1D or 2D");
    }
    if (targetRank > 1) {
      return rewriter.notifyMatchFailure(op,
                                         "target tensor shoule be 0D or 1D!");
    }

    if (selfRank != 1 or targetRank != 0) {
      if (!(selfSizes[0] == kUnknownSize and targetSizes[0] == kUnknownSize) and
          selfSizes[0] != targetSizes[0]) {
        return rewriter.notifyMatchFailure(
            op,
            "input tensor and target tensor should have the same batch size!");
      }
    }

    int64_t numClasses = selfSizes.back();
    auto weight = op.getWeight();
    auto weightT = weight.getType();
    if (!isa<Torch::NoneType>(weightT) && numClasses != kUnknownSize) {
      auto weightType = dyn_cast<BaseTensorType>(weightT);
      if (weightType.areAllSizesKnown()) {
        auto weightSizes = weightType.getSizes();
        int64_t weightNumel = 1;
        for (size_t i = 0; i < weightSizes.size(); i++) {
          weightNumel *= weightSizes[i];
        }
        if (weightNumel != numClasses) {
          return rewriter.notifyMatchFailure(
              op, "weight tensor should be defined either for all classes or "
                  "no classes!");
        }
      }
    }

    Value reductionValue = op.getReduction();
    int64_t reduction;
    if (!matchPattern(reductionValue, m_TorchConstantInt(&reduction))) {
      return rewriter.notifyMatchFailure(op,
                                         "reduction should be a constant int!");
    }

    // decomposation.
    uint64_t channelDim = 1;
    if (selfRank < 2) {
      channelDim = 0;
    }
    Value channelDimValue = rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(channelDim));

    auto ignoreIndex = op.getIgnoreIndex();
    Value w;
    if (!isa<Torch::NoneType>(weightT)) {
      if (selfRank > 1) {
        auto weightType = dyn_cast<BaseTensorType>(weightT);
        auto weightSizes = weightType.getSizes();
        SmallVector<int64_t> newShapeList(selfRank, 1);
        newShapeList[channelDim] = weightSizes[0];
        SmallVector<Value> newShapeListValue;
        for (size_t i = 0; i < newShapeList.size(); ++i) {
          newShapeListValue.push_back(rewriter.create<ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(newShapeList[i])));
        }
        Value newShape = rewriter.create<PrimListConstructOp>(
            loc,
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            newShapeListValue);
        auto newType = weightType.getWithSizesAndDtype(newShapeList,
                                                       weightType.getDtype());
        w = rewriter.create<AtenViewOp>(loc, newType, weight, newShape);
      } else {
        w = weight;
      }

      self = rewriter.create<AtenMulTensorOp>(loc, self.getType(), self, w);
    }

    SmallVector<int64_t> targetDimSizes(targetSizes);
    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    auto condType =
        ValueTensorType::get(ctx, targetDimSizes, rewriter.getI1Type());
    auto unequalCond =
        rewriter.create<AtenNeScalarOp>(loc, condType, target, ignoreIndex);
    auto zeroTensorType =
        ValueTensorType::get(ctx, {}, rewriter.getIntegerType(64, true));
    Value zeroTensor =
        rewriter.create<PrimNumToTensorScalarOp>(loc, zeroTensorType, zero);
    auto safeTarget = rewriter.create<AtenWhereSelfOp>(
        loc, target.getType(), unequalCond, target, zeroTensor);

    SmallVector<int64_t> safeTargetShape;
    for (size_t i = 0; i < targetSizes.size(); ++i) {
      if (channelDim == i) {
        safeTargetShape.push_back(1);
      }
      safeTargetShape.push_back(targetSizes[i]);
    }
    if (channelDim == safeTargetShape.size()) {
      safeTargetShape.push_back(1);
    }

    auto gatherType =
        ValueTensorType::get(ctx, safeTargetShape, targetType.getDtype());
    auto safeTarget_ = rewriter.create<AtenUnsqueezeOp>(
        loc, gatherType, safeTarget, channelDimValue);
    auto falseValue =
        rewriter.create<ConstantBoolOp>(loc, rewriter.getBoolAttr(false));
    auto none = rewriter.create<ConstantNoneOp>(loc);
    auto _gather = rewriter.create<AtenGatherOp>(
        loc, ValueTensorType::get(ctx, safeTargetShape, selfType.getDtype()),
        self, channelDimValue, safeTarget_, falseValue);
    Value gather = rewriter.create<AtenNegOp>(loc, _gather.getType(), _gather);
    auto unequalCondType = cast<BaseTensorType>(unequalCond.getType());
    auto result = rewriter.create<AtenWhereSelfOp>(
        loc,
        unequalCondType.getWithSizesAndDtype(unequalCondType.getSizes(),
                                             selfType.getDtype()),
        unequalCond,
        rewriter.create<AtenSqueezeDimOp>(
            loc, ValueTensorType::get(ctx, targetSizes, selfType.getDtype()),
            gather, channelDimValue),
        zeroTensor);

    Value totalWeight;
    if (reduction == 0 and selfRank > 1) {
      auto zeroFloat =
          rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
      Value twSize = rewriter.create<PrimListConstructOp>(
          loc,
          rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
          ValueRange({}));

      totalWeight = rewriter.create<AtenNewFullOp>(
          loc, op.getType(1), self, twSize, zeroFloat, none, none, none, none);
      rewriter.replaceOp(op, {result, totalWeight});

      return success();
    }

    if (!isa<Torch::NoneType>(weightT)) {
      auto wType = cast<BaseTensorType>(w.getType());
      auto newWType = wType.getWithSizesAndDtype(selfSizes, wType.getDtype());
      SmallVector<Value> selfSizesValue;
      for (size_t i = 0; i < selfSizes.size(); ++i) {
        selfSizesValue.push_back(rewriter.create<ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(selfSizes[i])));
      }
      auto wSize = rewriter.create<PrimListConstructOp>(
          loc,
          rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
          selfSizesValue);
      w = rewriter.create<AtenExpandOp>(loc, newWType, w, wSize, falseValue);
      auto wSumGather = rewriter.create<AtenGatherOp>(
          loc, ValueTensorType::get(ctx, safeTargetShape, wType.getDtype()), w,
          channelDimValue, safeTarget_, falseValue);
      auto wSumSq = rewriter.create<AtenSqueezeDimOp>(
          loc, ValueTensorType::get(ctx, targetSizes, wType.getDtype()),
          wSumGather, channelDimValue);
      auto wSum = rewriter.create<AtenWhereSelfOp>(
          loc,
          ValueTensorType::get(ctx, unequalCondType.getSizes(),
                               wType.getDtype()),
          unequalCond, wSumSq, zeroTensor);

      totalWeight = rewriter.create<AtenSumOp>(loc, op.getType(1), wSum, none);
    } else {
      totalWeight =
          rewriter.create<AtenSumOp>(loc, op.getType(1), unequalCond, none);
    }

    auto resultSum =
        rewriter.create<AtenSumOp>(loc, op.getType(0), result, none);
    if (reduction == 1) {
      auto resultMean = rewriter.create<AtenDivTensorOp>(
          loc, op.getType(0), resultSum, totalWeight);
      rewriter.replaceOp(op, {resultMean, totalWeight});

      return success();
    }

    rewriter.replaceOp(op, {resultSum, totalWeight});
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenBinaryCrossEntropyWithLogitsOp
    : public OpRewritePattern<AtenBinaryCrossEntropyWithLogitsOp> {
  using OpRewritePattern<AtenBinaryCrossEntropyWithLogitsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBinaryCrossEntropyWithLogitsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto self = op.getSelf();
    auto target = op.getTarget();
    auto posWeight = op.getPosWeight();
    auto weight = op.getWeight();
    auto reduction = op.getReduction();

    Value loss;
    auto one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    auto _one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(-1));

    auto _target =
        rewriter.create<AtenMulScalarOp>(loc, target.getType(), target, _one);
    auto _target_1 = rewriter.create<AtenAddScalarOp>(loc, _target.getType(),
                                                      _target, one, one);
    Value mm =
        rewriter.create<AtenMulTensorOp>(loc, self.getType(), _target_1, self);
    Value logSigm =
        rewriter.create<AtenLogSigmoidOp>(loc, self.getType(), self);

    if (!isa<Torch::NoneType>(posWeight.getType())) {
      auto logWeight = rewriter.create<AtenAddScalarOp>(
          loc, posWeight.getType(),
          rewriter.create<AtenSubScalarOp>(loc, posWeight.getType(), posWeight,
                                           one, one),
          one, one);
      loss = rewriter.create<AtenSubTensorOp>(
          loc, mm.getType(), mm,
          rewriter.create<AtenMulTensorOp>(loc, logWeight.getType(), logWeight,
                                           logSigm),
          one);
    } else {
      loss =
          rewriter.create<AtenSubTensorOp>(loc, mm.getType(), mm, logSigm, one);
    }

    if (!isa<Torch::NoneType>(weight.getType())) {
      loss =
          rewriter.create<AtenMulTensorOp>(loc, loss.getType(), loss, weight);
    }

    // apply loss reduction.
    int64_t reductionInt;
    if (!matchPattern(reduction, m_TorchConstantInt(&reductionInt))) {
      return rewriter.notifyMatchFailure(op, "no reduction type is appointed!");
    }

    auto none = rewriter.create<ConstantNoneOp>(loc);
    Value res;
    if (reductionInt == 1) {
      res = rewriter.create<AtenMeanOp>(loc, op.getType(), loss, none);
    } else if (reductionInt == 2) {
      res = rewriter.create<AtenSumOp>(loc, op.getType(), loss, none);
    } else {
      res = loss;
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenExp2Op : public OpRewritePattern<AtenExp2Op> {
  using OpRewritePattern<AtenExp2Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenExp2Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();

    auto resultTy = dyn_cast<ValueTensorType>(op.getType());
    if (!resultTy || !resultTy.hasDtype()) {
      return rewriter.notifyMatchFailure(op, "result must have dtype");
    }

    auto two =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(2));
    Value to = convertTensorToDtype(rewriter, loc, self, resultTy.getDtype());
    Value pow = rewriter.create<AtenPowScalarOp>(loc, resultTy, two, to);
    rewriter.replaceOp(op, pow);
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
    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "input tensor should have known sizes.");
    int64_t inputRank = inputType.getSizes().size();
    int64_t numClasses = Torch::kUnknownSize;
    auto resultType = cast<ValueTensorType>(op.getType());
    matchPattern(op.getNumClasses(), m_TorchConstantInt(&numClasses));
    Value none = rewriter.create<ConstantNoneOp>(loc);

    // arange tensor
    auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);
    auto arangeType =
        ValueTensorType::get(context, llvm::ArrayRef(numClasses), si64Type);
    Value arangeTensor = rewriter.create<AtenArangeOp>(
        loc, arangeType, op.getNumClasses(), /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pin_memory=*/none);

    // unsqueeze input
    Value rankV = rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(inputRank));
    auto unsqueeze = Torch::unsqueezeTensor(rewriter, op, input, rankV);
    if (failed(unsqueeze))
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor");

    Value unsqueezeTensor =
        convertTensorToDtype(rewriter, loc, *unsqueeze, si64Type);

    // compare
    auto eqType = ValueTensorType::get(
        context, cast<BaseTensorType>(op.getType()).getSizes(),
        IntegerType::get(context, 1));
    Value eqTensor = rewriter.create<AtenEqTensorOp>(
        loc, eqType, unsqueezeTensor, arangeTensor);

    // convert to si64
    Value result =
        convertTensorToDtype(rewriter, loc, eqTensor, resultType.getDtype());
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

    auto resultTy = cast<BaseTensorType>(op.getResult().getType());
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
    auto selfType = cast<BaseTensorType>(self.getType());
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
// Decompose `aten.hann_window` into `aten.arange.start`, `aten.mul.Scalar`,
// `aten.sin` and `aten.square` or into `aten.ones` in the trivial case
class DecomposeAtenHannWindowPeriodicOp
    : public OpRewritePattern<AtenHannWindowPeriodicOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHannWindowPeriodicOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    Type opType = op.getType();

    Value opWindowLength = op.getWindowLength();
    Value opDtype = op.getDtype();
    Value opLayout = op.getLayout();
    Value opDevice = op.getDevice();
    Value opPinMemory = op.getPinMemory();

    int64_t window_length;
    if (!matchPattern(opWindowLength, m_TorchConstantInt(&window_length)) ||
        window_length <= 0)
      return rewriter.notifyMatchFailure(
          op, "Expected a constant integer greater than zero");
    bool periodic;
    if (!matchPattern(op.getPeriodic(), m_TorchConstantBool(&periodic)))
      return rewriter.notifyMatchFailure(
          op, "Expected a constant boolean value for periodic");

    if (window_length == 1) {
      Value one =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
      SmallVector<Value> sizes({one});
      Value sizeList = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)), sizes);
      rewriter.replaceOpWithNewOp<AtenOnesOp>(op, opType, sizeList, opDtype,
                                              opLayout, opDevice, opPinMemory);
      return success();
    }

    Value zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));

    Value arange = rewriter.create<AtenArangeStartOp>(
        loc, opType, zero, op.getWindowLength(), opDtype, opLayout, opDevice,
        opPinMemory);

    double denominator = !periodic ? window_length - 1 : window_length;

    double piOverDenominator = 3.14159 / denominator;

    Value cstFactor = rewriter.create<ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr(piOverDenominator));

    Value fraction =
        rewriter.create<AtenMulScalarOp>(loc, opType, arange, cstFactor);
    Value sine = rewriter.create<AtenSinOp>(loc, opType, fraction);

    rewriter.replaceOpWithNewOp<AtenSquareOp>(op, opType, sine);

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

    auto selfType = cast<BaseTensorType>(self.getType());
    auto indexType = cast<BaseTensorType>(index.getType());
    BaseTensorType srcType = cast<BaseTensorType>(selfType.getWithSizesAndDtype(
        indexType.getOptionalSizes(), selfType.getOptionalDtype()));
    Value src =
        createInitTensor(rewriter, loc, srcType, op.getValue(), sizeList);
    rewriter.replaceOpWithNewOp<AtenScatterSrcOp>(op, op.getType(), self,
                                                  op.getDim(), index, src);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.sgn` op into comparisons and aten.where.
class DecomposeAtenSgnOp : public OpRewritePattern<AtenSgnOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSgnOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outType = cast<BaseTensorType>(op.getType());
    if (!outType.hasDtype()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected result type to have dtype");
    }
    // TODO: support complex type in future.
    if (isa<mlir::ComplexType>(outType.getDtype())) {
      return rewriter.notifyMatchFailure(op,
                                         "doesn't support complex type now");
    }

    auto zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    auto one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    auto minusOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(-1));

    auto compTy = outType.getWithSizesAndDtype(outType.getOptionalSizes(),
                                               rewriter.getI1Type());

    auto greater =
        rewriter.create<AtenGtScalarOp>(loc, compTy, op.getSelf(), zero);
    auto less =
        rewriter.create<AtenLtScalarOp>(loc, compTy, op.getSelf(), zero);

    // Pseudo code:
    // if (in > 0)
    //     return 1
    // else if (in < 0)
    //   return -1
    // else
    //   return 0
    // note: return 0 if nan/0.0/-0.0
    //       return 1 if inf
    //       return -1 if -inf
    auto selectGreater =
        rewriter.create<AtenWhereScalarOp>(loc, outType, greater, one, zero);

    rewriter.replaceOpWithNewOp<AtenWhereScalarSelfOp>(op, outType, less,
                                                       minusOne, selectGreater);
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

// Torch ops related to indexing tensors, e.g., AtenIndexTensor, AtenIndexPut.
namespace {

// unsqueeze is more easily optimized than a generic view, and we prefer to
// enjoy ops with more structure than less in compositions.
static FailureOr<Value> unsqueezeTensorAtTrailingDim(Operation *op,
                                                     PatternRewriter &rewriter,
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

static Value createIndexToReplaceNone(Operation *op, PatternRewriter &rewriter,
                                      Value input, int dimInt,
                                      int64_t dimSize) {
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
  auto int64Dtype = getDtypeIntValueForType(
      rewriter, loc, rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true));

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

static FailureOr<Value> createNewIndices(Operation *op,
                                         PatternRewriter &rewriter, Value input,
                                         llvm::ArrayRef<Value> oldIndices,
                                         llvm::ArrayRef<int64_t> newToOldDimMap,
                                         llvm::ArrayRef<bool> oldIndexUsed) {
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();

  auto inputType = cast<BaseTensorType>(input.getType());
  if (!inputType.hasSizes()) {
    return failure();
  }
  auto inputSizes = inputType.getSizes();
  int64_t inputRank = inputSizes.size();

  int64_t maxIndexRank = 0;
  for (auto index : oldIndices) {
    auto indexType = dyn_cast<BaseTensorType>(index.getType());
    if (!indexType) // None index
      continue;
    if (!indexType.hasSizes())
      return failure();
    int64_t indexRank = indexType.getSizes().size();
    maxIndexRank = maxIndexRank > indexRank ? maxIndexRank : indexRank;
  }

  // manually generate new indices.
  SmallVector<Value> listElements(inputRank);

  int64_t noneIndexCnt = 0;
  int64_t i;
  // handle trailing none indices.
  for (i = inputRank - 1; i >= 0; --i) {
    int64_t oldI = newToOldDimMap[i];
    if (oldIndexUsed[oldI])
      break;
    Value v = createIndexToReplaceNone(op, rewriter, input, i, inputSizes[i]);
    auto vInfo = unsqueezeTensorAtTrailingDim(op, rewriter, v, noneIndexCnt);
    if (failed(vInfo)) {
      return failure();
    }
    listElements[i] = *vInfo;
    noneIndexCnt++;
  }
  // handle non-none index in between.
  for (; i >= 0; --i) {
    int64_t oldI = newToOldDimMap[i];
    if (!oldIndexUsed[oldI])
      break;
    auto vInfo = unsqueezeTensorAtTrailingDim(op, rewriter, oldIndices[oldI],
                                              noneIndexCnt);
    if (failed(vInfo)) {
      return failure();
    }
    listElements[i] = *vInfo;
  }

  // handle possible leading none indices.
  for (; i >= 0; --i) {
    int64_t oldI = newToOldDimMap[i];
    if (oldIndexUsed[oldI]) {
      return failure();
    }
    Value v = createIndexToReplaceNone(op, rewriter, input, i, inputSizes[i]);
    auto vInfo = unsqueezeTensorAtTrailingDim(op, rewriter, v,
                                              noneIndexCnt + maxIndexRank);
    if (failed(vInfo)) {
      return failure();
    }
    listElements[i] = *vInfo;
    noneIndexCnt++;
  }

  auto listElemType = ValueTensorType::get(context, std::nullopt, nullptr);
  Value newIndexList = rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(listElemType), listElements);

  return newIndexList;
}

// The goal of this pattern is to eliminate `None` index in aten.Index.Tensor's
// `indices` param and transform it to aten.index.Tensor_hacked_twin, for the
// ease of various backend.
class DecomposeAtenIndexTensorOp : public OpRewritePattern<AtenIndexTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtenIndexTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get elements of `indices`");

    auto input = op.getSelf();
    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "only input with shape information is supported");
    }
    auto inputSizes = inputType.getSizes();
    int64_t inputRank = inputSizes.size();

    auto isTensor = [](Value v) {
      return isa<Torch::BaseTensorType>(v.getType());
    };

    // directly replace aten.Index.Tensor with aten.index.Tensor_hacked_twin
    if (llvm::all_of(indices, isTensor)) {
      // By default, we regard the first index type as the list element type.
      auto indexElemType = cast<BaseTensorType>(indices[0].getType())
                               .getWithSizesAndDtype(std::nullopt, nullptr);
      auto newIndices = rewriter.create<PrimListConstructOp>(
          loc, Torch::ListType::get(indexElemType), indices);
      rewriter.replaceOpWithNewOp<AtenIndexTensorHackedTwinOp>(
          op, op.getType(), input, newIndices);
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

    Value newInput;
    SmallVector<int64_t> newToOldDimMap;
    // permute input to make the non-none indices consecutive.
    if (!indexIsConsecutive) {
      SmallVector<Value> dimValues;
      SmallVector<int64_t> permutedSizes;
      for (int i = 0; i < inputRank; i++) {
        if (indexUsed[i]) {
          newToOldDimMap.emplace_back(i);
          dimValues.emplace_back(rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i)));
          permutedSizes.emplace_back(inputSizes[i]);
        }
      }
      for (int i = 0; i < inputRank; i++) {
        if (!indexUsed[i]) {
          newToOldDimMap.emplace_back(i);
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
        newToOldDimMap.emplace_back(i);
      }
    }

    auto newIndeicesInfo = createNewIndices(op, rewriter, newInput, indices,
                                            newToOldDimMap, indexUsed);
    if (failed(newIndeicesInfo)) {
      return rewriter.notifyMatchFailure(op, "failed to replcae `None` index");
    }
    rewriter.replaceOpWithNewOp<Torch::AtenIndexTensorHackedTwinOp>(
        op, op.getType(), newInput, *newIndeicesInfo);
    return success();
  }
};

// The goal of this pattern is to eliminate `None` index in aten.inde_put-like
// ops' `indices` param and transform it to aten.index_put.hacked_twin, for the
// ease of various backend.
template <typename AtenIndexPutLikeOpT>
class DecomposeAtenIndexPutLikeOp
    : public OpRewritePattern<AtenIndexPutLikeOpT> {
public:
  using OpRewritePattern<AtenIndexPutLikeOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(AtenIndexPutLikeOpT op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get elements of `indices`");

    auto input = op.getSelf();
    auto inputType = cast<BaseTensorType>(input.getType());
    if (!inputType.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "only input with shape information is supported");
    }
    auto inputSizes = inputType.getSizes();
    int64_t inputRank = inputSizes.size();

    auto isTensor = [](Value v) {
      return isa<Torch::BaseTensorType>(v.getType());
    };

    // directly replace current op with aten.index_put.hacked_twin
    if (llvm::all_of(indices, isTensor)) {
      // By default, we regard the first index type as the list element type.
      auto indexElemType = cast<BaseTensorType>(indices[0].getType())
                               .getWithSizesAndDtype(std::nullopt, nullptr);
      auto newIndex = rewriter.create<PrimListConstructOp>(
          loc, Torch::ListType::get(indexElemType), indices);
      rewriter.replaceOpWithNewOp<AtenIndexPutHackedTwinOp>(
          op, op.getType(), input, newIndex, op.getValues(),
          op.getAccumulate());
      return success();
    }

    SmallVector<bool> indexUsed =
        llvm::to_vector(llvm::map_range(indices, isTensor));
    for (int64_t i = indices.size(); i < inputRank; ++i)
      indexUsed.emplace_back(false);

    // check if non-None index is consecutive
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
    if (!indexIsConsecutive) {
      return rewriter.notifyMatchFailure(
          op, "non consecutive indices is not supported");
    }

    SmallVector<int64_t> newToOldDimMap;
    for (int i = 0; i < inputRank; i++) {
      newToOldDimMap.emplace_back(i);
    }

    auto newIndicesInfo = createNewIndices(op, rewriter, input, indices,
                                           newToOldDimMap, indexUsed);
    if (failed(newIndicesInfo)) {
      return rewriter.notifyMatchFailure(op, "failed to replace `None` index");
    }
    rewriter.replaceOpWithNewOp<AtenIndexPutHackedTwinOp>(
        op, op.getType(), input, *newIndicesInfo, op.getValues(),
        op.getAccumulate());
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
    auto inputType = cast<BaseTensorType>(input.getType());
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
// Decompose AtenLinalgNormOp to AtenLinalgVectorNormOp only
class DecomposeAtenLinalgNormOp : public OpRewritePattern<AtenLinalgNormOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLinalgNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> dimList;
    if (!getListConstructElements(op.getDim(), dimList)) {
      return rewriter.notifyMatchFailure(
          op, "dim should comes from a PrimListConstructOp");
    }
    if (dimList.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: only dim size of 1 is supported");
    }

    // default ord value is 2 for vector_norm
    auto ord = op.getOrd();
    if (isa<Torch::NoneType>(ord.getType())) {
      ord = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(2));
    }
    rewriter.replaceOpWithNewOp<Torch::AtenLinalgVectorNormOp>(
        op, op.getType(), op.getSelf(), ord, op.getDim(), op.getKeepdim(),
        op.getDtype());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenFakeQuantizePerTensorAffineOp
    : public OpRewritePattern<AtenFakeQuantizePerTensorAffineOp> {
public:
  using OpRewritePattern<AtenFakeQuantizePerTensorAffineOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFakeQuantizePerTensorAffineOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = getContext();

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));

    // input/scale
    Value divScale = rewriter.create<AtenDivScalarOp>(
        loc, op.getType(), op.getSelf(), op.getScale());
    // std::nearby_int(input/scale)
    Value round = rewriter.create<AtenRoundOp>(loc, op.getType(), divScale);
    // std::nearby_int(input/scale) + zero_point
    Value addZeroPoint = rewriter.create<AtenAddScalarOp>(
        loc, op.getType(), round, op.getZeroPoint(), one);
    // max(quant_min, std::nearby_int(input/scale) + zero_point)
    auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);
    auto tensorIntType =
        ValueTensorType::get(context, ArrayRef<int64_t>{1}, si64Type);
    Value max = rewriter.create<AtenMaximumOp>(
        loc, op.getType(), addZeroPoint,
        rewriter.create<AtenTensorIntOp>(loc, tensorIntType, op.getQuantMin(),
                                         /*dtype=*/none,
                                         /*device=*/none,
                                         /*requires_grad=*/falseVal));
    // min(quant_max, max(quant_min, std::nearby_int(input/scale) + zero_point))
    Value min = rewriter.create<AtenMinimumOp>(
        loc, op.getType(), max,
        rewriter.create<AtenTensorIntOp>(loc, tensorIntType, op.getQuantMax(),
                                         /*dtype=*/none, /*device=*/none,
                                         /*requires_grad=*/falseVal));
    // min(quant_max, max(quant_min, std::nearby_int(input/scale) + zero_point))
    // - zero_point
    Value subZeroPoint = rewriter.create<AtenSubScalarOp>(
        loc, op.getType(), min, op.getZeroPoint(), one);
    // (min(quant_max, max(quant_min, std::nearby_int(input/scale) +
    // zero_point)) - zero_point) * scale
    Value result = rewriter.create<AtenMulScalarOp>(
        loc, op.getType(), subZeroPoint, op.getScale());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Decompose aten.fake_quantize_per_tensor_affine_cachemask
// into aten.fake_quantize_per_tensor_affine
// when the second result is unused.
class DecomposeAtenFakeQuantizePerTensorAffineCachemaskOp
    : public OpRewritePattern<AtenFakeQuantizePerTensorAffineCachemaskOp> {
public:
  using OpRewritePattern<
      AtenFakeQuantizePerTensorAffineCachemaskOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFakeQuantizePerTensorAffineCachemaskOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getResult(1).use_empty())
      return failure();

    auto newOp = rewriter.create<AtenFakeQuantizePerTensorAffineOp>(
        op.getLoc(), op->getResult(0).getType(), op.getSelf(), op.getScale(),
        op.getZeroPoint(), op.getQuantMin(), op.getQuantMax());

    rewriter.replaceAllUsesWith(op->getResult(0), newOp);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
// Decompose aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams
// into aten.fake_quantize_per_tensor_affine.tensor_qparams
// when the second result is unused.
class DecomposeAten_FakeQuantizePerTensorAffineCachemaskTensorQparamsOp
    : public OpRewritePattern<
          Aten_FakeQuantizePerTensorAffineCachemaskTensorQparamsOp> {
public:
  using OpRewritePattern<
      Aten_FakeQuantizePerTensorAffineCachemaskTensorQparamsOp>::
      OpRewritePattern;
  LogicalResult
  matchAndRewrite(Aten_FakeQuantizePerTensorAffineCachemaskTensorQparamsOp op,
                  PatternRewriter &rewriter) const override {
    if (!op->getResult(1).use_empty())
      return failure();

    auto newOp =
        rewriter.create<AtenFakeQuantizePerTensorAffineTensorQparamsOp>(
            op.getLoc(), op->getResult(0).getType(), op.getSelf(),
            op.getScale(), op.getZeroPoint(), op.getQuantMin(),
            op.getQuantMax());

    rewriter.replaceAllUsesWith(op->getResult(0), newOp);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
// Decompose aten.fake_quantize_per_channel_affine_cachemask
// into aten.fake_quantize_per_channel_affine
// when the second result is unused.
class DecomposeAtenFakeQuantizePerChannelAffineCachemaskOp
    : public OpRewritePattern<AtenFakeQuantizePerChannelAffineCachemaskOp> {
public:
  using OpRewritePattern<
      AtenFakeQuantizePerChannelAffineCachemaskOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFakeQuantizePerChannelAffineCachemaskOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getResult(1).use_empty())
      return failure();

    auto newOp = rewriter.create<AtenFakeQuantizePerChannelAffineOp>(
        op.getLoc(), op->getResult(0).getType(), op.getSelf(), op.getScale(),
        op.getZeroPoint(), op.getAxis(), op.getQuantMin(), op.getQuantMax());

    rewriter.replaceAllUsesWith(op->getResult(0), newOp);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
// Decompose aten.fmax/fmin to aten.maximum/minimum + aten.where(nanMask)
template <typename AtenFOpT, typename AtenOpT>
class DecomposeAtenFMaxMinOp : public OpRewritePattern<AtenFOpT> {
public:
  using OpRewritePattern<AtenFOpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFOpT op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    BaseTensorType outType = cast<BaseTensorType>(op.getType());
    Type nanMaskType = outType.getWithSizesAndDtype(
        !outType.hasSizes() ? std::optional<ArrayRef<int64_t>>()
                            : llvm::ArrayRef(outType.getSizes()),
        rewriter.getI1Type());

    Value self = op.getSelf();
    Value other = op.getOther();

    Value normalResult =
        rewriter.create<AtenOpT>(loc, outType, self, other).getResult();
    Value selfIsNan =
        rewriter.create<Torch::AtenIsnanOp>(loc, nanMaskType, self).getResult();
    Value otherIsNan =
        rewriter.create<Torch::AtenIsnanOp>(loc, nanMaskType, other)
            .getResult();
    normalResult = rewriter.create<Torch::AtenWhereSelfOp>(
        loc, outType, otherIsNan, self, normalResult);
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, outType, selfIsNan, other,
                                                 normalResult);

    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenThresholdOp : public OpRewritePattern<AtenThresholdOp> {
public:
  using OpRewritePattern<AtenThresholdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenThresholdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    auto selfType = dyn_cast<BaseTensorType>(self.getType());
    if (!selfType || !selfType.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "requires input is tensor with sizes");
    }

    Value threshold = op.getThreshold();
    Value value = op.getValue();

    auto comOp = rewriter.create<AtenGtScalarOp>(
        loc,
        selfType.getWithSizesAndDtype(selfType.getSizes(),
                                      rewriter.getI1Type()),
        self, threshold);

    rewriter.replaceOpWithNewOp<AtenWhereScalarOtherOp>(op, op.getType(), comOp,
                                                        self, value);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenFloatPowerTensorTensorOp
    : public OpRewritePattern<AtenFloatPowerTensorTensorOp> {
public:
  using OpRewritePattern<AtenFloatPowerTensorTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFloatPowerTensorTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value exp = op.getExponent();

    auto selfTy = dyn_cast<BaseTensorType>(self.getType());
    if (!selfTy || !selfTy.hasDtype() || !selfTy.hasSizes()) {
      return rewriter.notifyMatchFailure(
          op, "requires input is tensor with dtype and sizes");
    }

    Value selfF64 =
        convertTensorToDtype(rewriter, loc, self, rewriter.getF64Type());
    rewriter.replaceOpWithNewOp<AtenPowTensorTensorOp>(op, op.getType(),
                                                       selfF64, exp);

    return success();
  }
};
} // namespace

static Value calculateIoU(PatternRewriter &rewriter, Location loc, Value box1,
                          Value box2) {
  // box format: 1x4xf32 with [x1, y1, x2, y2], 0 <= x1 < x2 and 0 <= y1 < y2
  Value cst0 =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value cst1 =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value cst2 =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(2));
  Value cst3 =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(3));
  auto scalarTensorType = rewriter.getType<Torch::ValueTensorType>(
      ArrayRef<int64_t>{1}, rewriter.getIntegerType(64, /*signed*/ true));
  Value cst0Tensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
      loc, scalarTensorType, cst0);
  Value cst1Tensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
      loc, scalarTensorType, cst1);
  Value cst2Tensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
      loc, scalarTensorType, cst2);
  Value cst3Tensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
      loc, scalarTensorType, cst3);
  auto boxesTensorType = dyn_cast<Torch::ValueTensorType>(box1.getType());
  auto extractTy = rewriter.getType<Torch::ValueTensorType>(
      ArrayRef<int64_t>{1}, boxesTensorType.getDtype());
  Value b1x1 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box1, /*dim=*/cst1, /*index=*/cst0Tensor);
  Value b1y1 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box1, /*dim=*/cst1, /*index=*/cst1Tensor);
  Value b1x2 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box1, /*dim=*/cst1, /*index=*/cst2Tensor);
  Value b1y2 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box1, /*dim=*/cst1, /*index=*/cst3Tensor);
  Value b2x1 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box2, /*dim=*/cst1, /*index=*/cst0Tensor);
  Value b2y1 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box2, /*dim=*/cst1, /*index=*/cst1Tensor);
  Value b2x2 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box2, /*dim=*/cst1, /*index=*/cst2Tensor);
  Value b2y2 = rewriter.create<AtenIndexSelectOp>(
      loc, extractTy, box2, /*dim=*/cst1, /*index=*/cst3Tensor);

  // Calculate intersection width and height
  Value intersectX1 =
      rewriter.create<Torch::AtenMaximumOp>(loc, extractTy, b1x1, b2x1);
  Value intersectY1 =
      rewriter.create<Torch::AtenMaximumOp>(loc, extractTy, b1y1, b2y1);
  Value intersectX2 =
      rewriter.create<Torch::AtenMinimumOp>(loc, extractTy, b1x2, b2x2);
  Value intersectY2 =
      rewriter.create<Torch::AtenMinimumOp>(loc, extractTy, b1y2, b2y2);
  // Width = max(0, intersectX2 - intersectX1)
  Value float0 = rewriter.create<Torch::ConstantFloatOp>(
      loc, rewriter.getF32FloatAttr(0.0));
  auto scalarFloatType = rewriter.getType<Torch::ValueTensorType>(
      ArrayRef<int64_t>{1}, rewriter.getF32Type());
  Value float0Tensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
      loc, scalarFloatType, float0);
  Value width = rewriter.create<Torch::AtenSubTensorOp>(
      loc, extractTy, intersectX2, intersectX1, cst1);
  width = rewriter.create<Torch::AtenMaximumOp>(loc, extractTy, width,
                                                float0Tensor);
  // Height = max(0, intersectY2 - intersectY1)
  Value height = rewriter.create<Torch::AtenSubTensorOp>(
      loc, extractTy, intersectY2, intersectY1, cst1);
  height = rewriter.create<Torch::AtenMaximumOp>(loc, extractTy, height,
                                                 float0Tensor);
  // Intersection area = width * height
  Value intersectionArea =
      rewriter.create<Torch::AtenMulTensorOp>(loc, extractTy, width, height);

  // Calculate area of box1: (b1x2 - b1x1) * (b1y2 - b1y1)
  Value width1 =
      rewriter.create<Torch::AtenSubTensorOp>(loc, extractTy, b1x2, b1x1, cst1);
  Value height1 =
      rewriter.create<Torch::AtenSubTensorOp>(loc, extractTy, b1y2, b1y1, cst1);
  Value area1 =
      rewriter.create<Torch::AtenMulTensorOp>(loc, extractTy, width1, height1);
  // Calculate area of box2: (b2x2 - b2x1) * (b2y2 - b2y1)
  Value width2 =
      rewriter.create<Torch::AtenSubTensorOp>(loc, extractTy, b2x2, b2x1, cst1);
  Value height2 =
      rewriter.create<Torch::AtenSubTensorOp>(loc, extractTy, b2y2, b2y1, cst1);
  Value area2 =
      rewriter.create<Torch::AtenMulTensorOp>(loc, extractTy, width2, height2);
  // Union area = area1 + area2 - intersectionArea
  Value unionArea = rewriter.create<Torch::AtenAddTensorOp>(loc, extractTy,
                                                            area1, area2, cst1);
  unionArea = rewriter.create<Torch::AtenSubTensorOp>(loc, extractTy, unionArea,
                                                      intersectionArea, cst1);

  Value iouTensor = rewriter.create<Torch::AtenDivTensorOp>(
      loc, extractTy, intersectionArea, unionArea);
  return rewriter.create<Torch::AtenItemOp>(
      loc, rewriter.getType<Torch::FloatType>(), iouTensor);
}

namespace {
class DecomposeTorchvisionNmsOp : public OpRewritePattern<TorchvisionNmsOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TorchvisionNmsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value boxes = op.getDets();
    Value scores = op.getScores();
    Value iouThreshold = op.getIouThreshold();

    Value cst0 = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value cst1 = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstTrue =
        rewriter.create<Torch::ConstantBoolOp>(loc, rewriter.getBoolAttr(true));

    // Sort scores in descending order
    // Use the sorted indices to iterate boxes
    auto scoresType = dyn_cast<BaseTensorType>(scores.getType());
    auto sortIndicesType = scoresType.getWithSizesAndDtype(
        scoresType.getOptionalSizes(),
        IntegerType::get(context, 64, IntegerType::Signed));
    auto sortResult = rewriter.create<Torch::AtenSortOp>(
        loc, TypeRange({scores.getType(), sortIndicesType}), scores,
        /*dim=*/cst0, /*descending=*/cstTrue);

    // Get number of boxes for the loop count
    auto boxesTensorType = dyn_cast<Torch::ValueTensorType>(boxes.getType());
    int64_t boxesSize = boxesTensorType.getSizes()[0];
    Value len = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(boxesSize));

    // Create a mask to mark if we keep the boxes
    Value maskShapeList = rewriter.create<Torch::PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        SmallVector<Value>{len});
    auto maskTy =
        ValueTensorType::get(context, ArrayRef<int64_t>{boxesSize},
                             rewriter.getIntegerType(64, /*signed*/ true));
    Value mask = rewriter.create<Torch::AtenOnesOp>(
        loc, maskTy, maskShapeList, cstNone, cstNone, cstNone, cstNone);

    llvm::SmallVector<int64_t> sliceSizes = {1, 4};
    auto sliceTy = rewriter.getType<ValueTensorType>(
        sliceSizes, boxesTensorType.getDtype());

    // 1. Loop through the boxes based on sorted indices
    // 2. Check the mask if it's marked as suppressed
    // 3. Loop through the rest boxes in sorted indices
    // 4. Extract the coordinates of two boxes and calculate IoU
    // 5. Mark the second box as suppressed if IOU is larger than threshold
    auto loop1 =
        rewriter.create<Torch::PrimLoopOp>(loc, maskTy, len, cstTrue, mask);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      Block *rowLoopBody = rewriter.createBlock(
          &loop1.getRegion(), loop1.getRegion().begin(),
          TypeRange({rewriter.getType<Torch::IntType>(), mask.getType()}),
          {loc, loc});
      Value i = rowLoopBody->getArgument(0);

      // Extract the mask to check if the base box is suppressed
      auto extractTy = rewriter.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{1}, rewriter.getIntegerType(64, true));
      Value extract = rewriter.create<AtenSelectIntOp>(
          loc, extractTy, mask, /*dim=*/cst0, /*index=*/i);
      Value scalar = rewriter.create<Torch::AtenItemOp>(
          loc, rewriter.getType<Torch::IntType>(), extract);
      Value iskept = rewriter.create<Torch::AtenBoolIntOp>(
          loc, rewriter.getType<Torch::BoolType>(), scalar);
      auto ifFilterOthers =
          rewriter.create<Torch::PrimIfOp>(loc, TypeRange({maskTy}), iskept);
      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.createBlock(&ifFilterOthers.getThenRegion(),
                             ifFilterOthers.getThenRegion().begin());

        // Extract the index from sorted indices to get the coordinates
        Value extractIdx1 = rewriter.create<AtenSelectIntOp>(
            loc, extractTy, sortResult.getResults()[1], /*dim=*/cst0,
            /*index=*/i);
        Value idx1 = rewriter.create<Torch::AtenItemOp>(
            loc, rewriter.getType<Torch::IntType>(), extractIdx1);
        Value end1 = rewriter.create<Torch::AtenAddIntOp>(loc, idx1, cst1);
        Value slice1 = rewriter.create<AtenSliceTensorOp>(
            loc, sliceTy, boxes,
            /*dim=*/cst0, /*start=*/idx1, /*end=*/end1, /*step=*/cst1);

        // Loop through the rest of boxes
        auto loop2 =
            rewriter.create<Torch::PrimLoopOp>(loc, maskTy, len, cstTrue, mask);
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          Block *colLoopBody = rewriter.createBlock(
              &loop2.getRegion(), loop2.getRegion().begin(),
              TypeRange({rewriter.getType<Torch::IntType>(), mask.getType()}),
              {loc, loc});

          // Check if current index is out of range
          Value j = colLoopBody->getArgument(0);
          j = rewriter.create<Torch::AtenAddIntOp>(loc, j, i);
          j = rewriter.create<Torch::AtenAddIntOp>(loc, j, cst1);
          Value isInRange = rewriter.create<Torch::AtenLtIntOp>(loc, j, len);
          auto ifCalculateIou = rewriter.create<Torch::PrimIfOp>(
              loc, TypeRange({maskTy}), isInRange);
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.createBlock(&ifCalculateIou.getThenRegion(),
                                 ifCalculateIou.getThenRegion().begin());

            // Extract the coordinates for the second box
            Value extractIdx2 = rewriter.create<AtenSelectIntOp>(
                loc, extractTy, sortResult.getResults()[1], /*dim=*/cst0,
                /*index=*/j);
            Value idx2 = rewriter.create<Torch::AtenItemOp>(
                loc, rewriter.getType<Torch::IntType>(), extractIdx2);
            Value end2 = rewriter.create<Torch::AtenAddIntOp>(loc, idx2, cst1);
            Value slice2 = rewriter.create<AtenSliceTensorOp>(
                loc, sliceTy, boxes,
                /*dim=*/cst0, /*start=*/idx2, /*end=*/end2, /*step=*/cst1);

            // Calculate IoU and decide if suppress it
            Value iou = calculateIoU(rewriter, loc, slice1, slice2);
            Value isSuppressed =
                rewriter.create<Torch::AtenGtFloatOp>(loc, iou, iouThreshold);
            auto ifUnmask = rewriter.create<Torch::PrimIfOp>(
                loc, TypeRange({maskTy}), isSuppressed);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.createBlock(&ifUnmask.getThenRegion(),
                                   ifUnmask.getThenRegion().begin());

              Value zerosShapeList =
                  rewriter.create<Torch::PrimListConstructOp>(
                      loc, Torch::ListType::get(Torch::IntType::get(context)),
                      SmallVector<Value>{cst1});
              auto zeroTy =
                  ValueTensorType::get(context, ArrayRef<int64_t>{1},
                                       rewriter.getIntegerType(64, true));
              Value falseMask = rewriter.create<Torch::AtenZerosOp>(
                  loc, zeroTy, zerosShapeList, cstNone, cstNone, cstNone,
                  cstNone);
              Value end3 = rewriter.create<Torch::AtenAddIntOp>(loc, j, cst1);
              Value thenMask = rewriter.create<Torch::AtenSliceScatterOp>(
                  loc, maskTy, mask, falseMask, cst0,
                  /*start=*/j, /*end=*/end3, /*step=*/cst1);
              rewriter.create<Torch::PrimIfYieldOp>(loc, thenMask);
            }
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.createBlock(&ifUnmask.getElseRegion(),
                                   ifUnmask.getElseRegion().begin());
              Value elseMask = mask;
              rewriter.create<Torch::PrimIfYieldOp>(loc, elseMask);
            }

            rewriter.create<Torch::PrimIfYieldOp>(loc, ifUnmask.getResult(0));
          }
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.createBlock(&ifCalculateIou.getElseRegion(),
                                 ifCalculateIou.getElseRegion().begin());
            Value elseMask = mask;
            rewriter.create<Torch::PrimIfYieldOp>(loc, elseMask);
          }

          rewriter.create<Torch::PrimLoopConditionOp>(
              loc, cstTrue, ifCalculateIou.getResult(0));
        }

        rewriter.create<Torch::PrimIfYieldOp>(loc, loop2.getResult(0));
      }
      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.createBlock(&ifFilterOthers.getElseRegion(),
                             ifFilterOthers.getElseRegion().begin());
        Value elseMask = mask;
        rewriter.create<Torch::PrimIfYieldOp>(loc, elseMask);
      }

      rewriter.create<Torch::PrimLoopConditionOp>(loc, cstTrue,
                                                  ifFilterOthers.getResult(0));
    }

    rewriter.replaceOpWithNewOp<Torch::AtenMaskedSelectOp>(
        op, op.getType(), sortResult.getResults()[1], loop1.getResult(0));
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

    addPatternIfTargetOpIsIllegal<DecomposeAten_WeightNormInterfaceOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSoftmaxIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_SoftmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_SafeSoftmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_LogSoftmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLogSoftmaxIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLogSigmoidOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHardshrinkOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSoftshrinkOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEmptyLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorAllocLikeOp<AtenOnesLikeOp, 1>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorAllocLikeOp<AtenZerosLikeOp, 0>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStackOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHstackOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenColumnStackOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRollOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRepeatOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRepeatInterleaveSelfIntOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenExpandOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFlattenUsingIntsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenUnflattenIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarOtherOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarSelfOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNanToNumOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMaskedFillScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMaskedFillTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMaskedScatterOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSizeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenReshapeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_SoftmaxBackwardDataOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTanhBackwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAddmmOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMeanOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMeanDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSelectIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMatmulOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMvOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRenormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLinalgCrossOp>(patterns);
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
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvTranspose1dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvTranspose2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvTranspose3dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenArangeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenArangeStartOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimsIotaOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLinspaceOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenAminAmaxOp<AtenAmaxOp, AtenMaxDimOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenAminAmaxOp<AtenAminOp, AtenMinDimOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenArgMinMaxOp<AtenArgmaxOp, AtenMaxDimOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenArgMinMaxOp<AtenArgminOp, AtenMinDimOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAminmaxOp>(patterns);
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
    addPatternIfTargetOpIsIllegal<DecomposeAtenPreluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRreluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRreluWithNoiseOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRreluWithNoiseBackwardOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCeluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAtleast1dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAtleast2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEinsumOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_TrilinearOp>(patterns);
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
    addPatternIfTargetOpIsIllegal<DecomposeAtenExpandAsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_ToCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenDropoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DeomposeAtenNativeDropoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNewEmptyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexPutLikeOp<AtenIndexPutOp>>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenIndexPutLikeOp<Aten_UnsafeIndexPutHackedTwinOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenIndexPutLikeOp<Aten_IndexPutImplOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenPadOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenToDtypeLayoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenToDeviceOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenToPrimDeviceOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAdaptiveMaxPool1dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAdaptiveAvgPool1dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAdaptiveAvgPool2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMinOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMinTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRad2degOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCosineSimilarityOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTruncOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSignbitOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFracOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCopysignTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLdexpTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFmodTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBaddbmmOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFloorDivideOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFloorDivideScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNumpyTOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSelectScatterOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRot90Op>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSplitWithSizesOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNarrowOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNarrowTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenGluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_EmbeddingBagOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLiftFreshCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMseLossOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenL1LossOp>(patterns);
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
    addPatternIfTargetOpIsIllegal<DecomposeAtenFakeQuantizePerTensorAffineOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSeluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLeakyReluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLeakyReluBackwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLerpScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLerpTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNewEmptyStridedOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEmptyStridedOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBucketizeTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimTolistOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimsSqueezeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMovedimIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenExp2Op>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenOneHotOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCrossEntropyLossOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNllLossForwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBinaryCrossEntropyWithLogitsOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarMeanDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTopkOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHannWindowPeriodicOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenScalarTensor>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenScatterValueOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSgnOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTypeAsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTileOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenReshapeAsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTriuOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTriuIndicesOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTrilIndicesOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenDeg2radOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLinalgNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_LinalgDetOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLinalgSlogdetOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenFakeQuantizePerTensorAffineCachemaskOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAten_FakeQuantizePerTensorAffineCachemaskTensorQparamsOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenFakeQuantizePerChannelAffineCachemaskOp>(patterns);
    // More specific conv ops
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvTbcOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConv1dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConv2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConv3dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenThresholdOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFloatPowerTensorTensorOp>(
        patterns);

    addPatternIfTargetOpIsIllegal<
        DecomposeAtenFMaxMinOp<AtenFmaxOp, AtenMaximumOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenFMaxMinOp<AtenFminOp, AtenMinimumOp>>(patterns);

    // Torchvision ops
    addPatternIfTargetOpIsIllegal<DecomposeTorchvisionNmsOp>(patterns);

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
