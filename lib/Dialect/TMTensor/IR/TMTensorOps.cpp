//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TMTensor;

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

Value TMTensor::getDimValue(OpBuilder &builder, Location loc, Value v,
                            int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      })
      .Default([&](Type t) { return Value(); });
}

OpFoldResult TMTensor::getDim(OpBuilder &builder, Location loc, Value v,
                              int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

LogicalResult AttentionOp::verify() {
  Operation *op = getOperation();
  ShapedType queryType = getQueryType();
  ShapedType keyType = getKeyType();
  ArrayRef<int64_t> queryShape = queryType.getShape();
  ArrayRef<int64_t> keyShape = keyType.getShape();
  for (int i = 0, s = queryShape.size() - 2; i < s; ++i) {
    if (keyShape[i] != queryShape[i])
      return op->emitOpError("query and key batch mismatch");
  }
  if (keyShape.back() != queryShape.back())
    return op->emitOpError("query and key head dimension mismatch");
  return success();
}

SmallVector<Range> AttentionOp::getIterationDomain(OpBuilder &builder) {
  SmallVector<Range> loopBounds;
  return loopBounds;
}

SmallVector<utils::IteratorType> AttentionOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes;
  return iteratorTypes;
}

bool AttentionOp::payloadUsesValueFromOperand(OpOperand *opOperand) {
  Value operand = opOperand->get();
  return operand == getQuery() || operand == getKey() || operand == getValue();
}

// Performs a matmul between lhs and rhs
// Note that "transposed" means the last two dims of rhs are swapped
static void matmul(OpBuilder &b, Location loc, Value lhs, ValueRange lhsSizes,
                   Value rhs, ValueRange rhsSizes, Value output,
                   ValueRange outputSizes, bool transposed = false) {
  auto elementType = lhs.getType().cast<MemRefType>().getElementType();
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  auto rank = outputSizes.size();
  Value reductionDimSize = lhsSizes[lhsSizes.size() - 1];

  // Loop over output
  b.create<scf::ParallelOp>(
      loc, SmallVector<Value>(rank, zero), outputSizes,
      SmallVector<Value>(rank, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        Value acc = b.create<arith::ConstantOp>(
            loc, elementType, b.getFloatAttr(elementType, 0.0));
        Value sum =
            b.create<scf::ForOp>(
                 loc, zero, reductionDimSize, one, SmallVector<Value>{acc},
                 [&](OpBuilder &b, Location loc, Value i, ValueRange accs) {
                   SmallVector<Value> lhsIVs(localIVs), rhsIVs(localIVs);
                   lhsIVs[lhsIVs.size() - 1] = i;
                   rhsIVs[rhsIVs.size() - 2] = i;
                   if (transposed)
                     std::swap(rhsIVs[rhsIVs.size() - 1],
                               rhsIVs[rhsIVs.size() - 2]);

                   Value acc = accs[0];
                   Value rElem = b.create<memref::LoadOp>(loc, lhs, lhsIVs);
                   Value cElem = b.create<memref::LoadOp>(loc, rhs, rhsIVs);
                   Value x = b.create<arith::MulFOp>(loc, rElem, cElem);
                   x = b.create<arith::AddFOp>(loc, x, acc);

                   b.create<scf::YieldOp>(loc, x);
                 })
                ->getResult(0);
        b.create<memref::StoreOp>(loc, sum, output, localIVs);
      });
}

LogicalResult AttentionOp::generateScalarImplementation(OpBuilder &b,
                                                        Location loc,
                                                        ValueRange ivs) {

  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  Value output = getOutput();
  auto queryType = query.getType().cast<MemRefType>();
  auto keyType = key.getType().cast<MemRefType>();
  auto valueType = value.getType().cast<MemRefType>();
  auto queryRank = queryType.getRank();
  auto keyRank = keyType.getRank();
  auto valueRank = valueType.getRank();
  auto keySizes = keyType.getShape();
  Type elementType = queryType.getElementType();

  Value zeroF = b.create<arith::ConstantOp>(loc, elementType,
                                            b.getFloatAttr(elementType, 0.0));

  // TODO: This needs to be fixed, it assumes everything is dynamic however if
  // any shapes are static the `memref.alloc` generated is illegal.
  SmallVector<Value> queryDynSizes, keyDynSizes, valueDynSizes, outputDynSizes;
  for (auto i = 0; i < queryRank; i++)
    queryDynSizes.push_back(b.create<memref::DimOp>(loc, query, i));
  for (auto i = 0; i < keyRank; i++)
    keyDynSizes.push_back(b.create<memref::DimOp>(loc, key, i));
  for (auto i = 0; i < valueRank; i++)
    valueDynSizes.push_back(b.create<memref::DimOp>(loc, value, i));
  for (auto i = 0; i < queryRank; i++)
    outputDynSizes.push_back(b.create<memref::DimOp>(loc, output, i));

  // weight = query @ key
  auto weightRank = queryRank;
  auto weightSizes = SmallVector<int64_t>(queryType.getShape());
  weightSizes[weightRank - 1] = keySizes[keyRank - 2];
  auto weightType = MemRefType::get(weightSizes, queryType.getElementType());

  // Setup the weight dynamic sizes:
  SmallVector<Value> weightDynSizes(queryDynSizes);
  weightDynSizes[weightRank - 1] = keyDynSizes[keyRank - 2];

  SmallVector<Value> weightFilteredDynSizes;
  for (int i = 0; i < weightRank; ++i)
    if (weightSizes[i] == ShapedType::kDynamic)
      weightFilteredDynSizes.push_back(weightDynSizes[i]);

  Value weight =
      b.create<memref::AllocOp>(loc, weightType, weightFilteredDynSizes);
  matmul(b, loc, query, queryDynSizes, key, keyDynSizes, weight, weightDynSizes,
         /*transposed=*/true);

  // weight = softmax(weight)
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value dim = weightDynSizes[weightRank - 1];
  Value scaleFactor = b.create<math::SqrtOp>(
      loc, b.create<arith::UIToFPOp>(
               loc, elementType,
               b.create<arith::IndexCastUIOp>(loc, b.getI32Type(),
                                              queryDynSizes[queryRank - 1])));
  // calculate max(weight)
  Value init = b.create<memref::LoadOp>(loc, weight,
                                        SmallVector<Value>(weightRank, zero));
  Value globalMax =
      b.create<scf::ParallelOp>(
           loc, SmallVector<Value>(weightRank, zero), weightDynSizes,
           SmallVector<Value>(weightRank, one), init,
           [&](OpBuilder &b, Location loc, ValueRange localIVs,
               ValueRange accs) {
             auto reduceOp = b.create<scf::ReduceOp>(loc, init);
             // Build reduce body.
             Block &reductionBody = reduceOp.getReductions()[0].front();
             auto bodyBuilder = OpBuilder::atBlockEnd(&reductionBody);
             Value acc = reductionBody.getArgument(0);
             Value x =
                 bodyBuilder.create<memref::LoadOp>(loc, weight, localIVs);
             Value max = bodyBuilder.create<arith::MaximumFOp>(loc, x, acc);
             bodyBuilder.create<scf::ReduceReturnOp>(loc, max);
           })
          .getResult(0);
  // weight = (weight - max(weight)) / math.sqrt(querySizes[-1])
  b.create<scf::ParallelOp>(
      loc, SmallVector<Value>(weightRank, zero), weightDynSizes,
      SmallVector<Value>(weightRank, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        Value x = b.create<memref::LoadOp>(loc, weight, localIVs);
        x = b.create<arith::SubFOp>(loc, x, globalMax);
        x = b.create<arith::DivFOp>(loc, x, scaleFactor);
        b.create<memref::StoreOp>(loc, x, weight, localIVs);
      });
  // calculate exp(weight)
  SmallVector<Value> min(weightRank, zero),
      max(weightDynSizes.begin(), weightDynSizes.end()), steps(weightRank, one);
  b.create<scf::ParallelOp>(
      loc, min, max, steps,
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        Value x = b.create<memref::LoadOp>(loc, weight, localIVs);
        x = b.create<math::ExpOp>(loc, x);
        b.create<memref::StoreOp>(loc, x, weight, localIVs);
      });

  llvm::SmallVector<Value> expWeightDynDims(weightFilteredDynSizes);
  if (weightSizes.back() == ShapedType::kDynamic)
    expWeightDynDims.resize(expWeightDynDims.size() - 1);

  Value expWeightSum = b.create<memref::AllocOp>(
      loc,
      MemRefType::get(
          SmallVector<int64_t>(weightSizes.begin(), weightSizes.end() - 1),
          elementType),
      expWeightDynDims);
  b.create<scf::ParallelOp>(
      loc, SmallVector<Value>(weightRank - 1, zero),
      SmallVector<Value>{weightDynSizes.begin(), weightDynSizes.end() - 1},
      SmallVector<Value>(weightRank - 1, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        b.create<memref::StoreOp>(loc, zeroF, expWeightSum, localIVs);
      });
  // Loop over all dims but -1
  b.create<scf::ParallelOp>(
      loc, SmallVector<Value>(weightRank - 1, zero),
      SmallVector<Value>(weightDynSizes.begin(), weightDynSizes.end() - 1),
      SmallVector<Value>(weightRank - 1, one),
      [&](OpBuilder &b, Location loc, ValueRange outsideDims) {
        // Sum over last dim
        b.create<scf::ParallelOp>(
            loc, zero, dim, one,
            [&](OpBuilder &b, Location loc, ValueRange localIVs) {
              SmallVector<Value> coords(outsideDims);
              coords.push_back(localIVs[0]);
              Value x =
                  b.create<memref::LoadOp>(loc, expWeightSum, outsideDims);
              Value y = b.create<memref::LoadOp>(loc, weight, coords);
              Value sum = b.create<arith::AddFOp>(loc, x, y);
              b.create<memref::StoreOp>(loc, sum, expWeightSum, outsideDims);
            });
      });
  // calculate exp(weight) / sum(exp(weight))
  b.create<scf::ParallelOp>(
      loc, SmallVector<Value>(weightRank, zero),
      SmallVector<Value>(weightDynSizes.begin(), weightDynSizes.end()),
      SmallVector<Value>(weightRank, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        SmallVector<Value> sumIVs(localIVs);
        sumIVs.pop_back();
        Value x = b.create<memref::LoadOp>(loc, weight, localIVs);
        Value sum = b.create<memref::LoadOp>(loc, expWeightSum, sumIVs);
        x = b.create<arith::DivFOp>(loc, x, sum);
        b.create<memref::StoreOp>(loc, x, weight, localIVs);
      });

  // output = weight @ value
  matmul(b, loc, weight, weightDynSizes, value, valueDynSizes, output,
         outputDynSizes, /*transposed=*/false);

  return success();
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

LogicalResult ScanOp::verify() {
  if (getNumInputs() != 1) {
    return emitOpError("expected one input operands");
  }
  if (getNumOutputs() != 2) {
    return emitOpError("expected two output operands");
  }
  if (!input().getType().isa<ShapedType>()) {
    return emitOpError("expected first input element type to be shaped");
  }
  auto accumulatorType = accumulator().getType().cast<ShapedType>();
  auto inputType = input().getType().cast<ShapedType>();
  auto outputType = output().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (accumulatorType.getElementType() != inputType.getElementType()) {
    return emitOpError(
        "expected input/accumulator element types to be identical");
  }
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  if (accumulatorRank != inputType.getRank() - 1) {
    return emitOpError(
        "expected accumulator rank to be equal to input rank - 1");
  }
  SmallVector<int64_t> expectedAccumulatorShape;
  for (size_t i = 0; i < (size_t)inputType.getRank(); i++) {
    if (i != getDimension())
      expectedAccumulatorShape.push_back(inputShapes[i]);
  }
  if (llvm::any_of(llvm::zip(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamic &&
                            std::get<1>(s) != ShapedType::kDynamic &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return emitOpError("incompatible input/accumulator shapes");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("expected input/output element types to be identical");
  }
  if (inputShapes.size() != outputShapes.size()) {
    return emitOpError("expected input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamic &&
                            std::get<1>(s) != ShapedType::kDynamic &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return emitOpError("incompatible input/output shapes");
  }
  return success();
}

SmallVector<Range> ScanOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = input();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> ScanOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

bool ScanOp::payloadUsesValueFromOperand(OpOperand *opOperand) {
  Value operand = opOperand->get();
  if (operand == accumulator())
    return !getInclusive();
  else if (operand == output())
    return false;
  else {
    assert(operand == input() &&
           "operand must belong to the current tm_tensor.scan op");
    return true;
  }
}

// Generates naive scalar implementation of scan for a given operator f.
// For inclusive,
//     output[0] = input[0]
//     output[i] = f(output[i-1], input[i])
//
// For exclusive,
//     output[0] = 0
//     output[i] = f(output[i-1], input[i-1])

LogicalResult ScanOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  SmallVector<Value> indices, scanBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  uint64_t scanDim = getDimension();
  Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                       indices[scanDim], zero);
  bool isInclusive = getInclusive();
  SmallVector<Value> accIndices;
  for (size_t i = 0; i < indices.size(); i++) {
    if (i != scanDim)
      accIndices.push_back(indices[i]);
  }

  auto scfIf = b.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        if (isInclusive) {
          auto value = b.create<memref::LoadOp>(loc, input(), indices);
          b.create<memref::StoreOp>(loc, value, output(), indices);
        } else {
          auto value = b.create<memref::LoadOp>(loc, accumulator(), accIndices);
          b.create<memref::StoreOp>(loc, value, output(), indices);
        }
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value iv = indices[scanDim];
        Value ivMinusOne = b.create<arith::SubIOp>(loc, iv, one);
        indices[scanDim] = ivMinusOne;
        scanBlkArgs.push_back(b.create<memref::LoadOp>(loc, output(), indices));
        Value i0;
        if (!isInclusive)
          i0 = b.create<memref::LoadOp>(loc, input(), indices);
        indices[scanDim] = iv;
        if (isInclusive)
          i0 = b.create<memref::LoadOp>(loc, input(), indices);
        scanBlkArgs.push_back(i0);
      });

  auto &srcBlock = getRegion().front();
  Region &thisRegion = scfIf.getElseRegion();
  IRMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = thisRegion.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip(srcBlock.getArguments(), scanBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        output(), indices);
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        accumulator(), accIndices);
    b.create<scf::YieldOp>(loc);
  }
  return success();
}

static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

LogicalResult ScanOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//
static Type getComplexElementTypeOrSelf(Type ty) {
  if (auto complex = dyn_cast_or_null<ComplexType>(ty))
    return complex.getElementType();
  return ty;
}

static bool isInvalid(ArrayRef<int64_t> dimsPos, int64_t rank) {
  // early exit.
  if (static_cast<int64_t>(dimsPos.size()) > rank)
    return true;
  DenseSet<int64_t> uniqued;
  for (int64_t dim : dimsPos)
    uniqued.insert(dim);
  if (static_cast<int64_t>(dimsPos.size()) != uniqued.size())
    return true;
  return llvm::any_of(
      dimsPos, [rank](int64_t dimPos) { return dimPos < 0 || dimPos >= rank; });
}

LogicalResult ScatterOp::verify() {
  Operation *op = getOperation();
  if (getInputs().size() != 2) {
    return op->emitOpError("expected two input operands");
  }
  if (getOutputs().size() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto checkDimensionsMatch = [&](ShapedType t1, ShapedType t2, unsigned dim) {
    return t1.getShape()[dim] == t2.getShape()[dim];
  };

  auto indicesType = getIndicesType();
  if (indicesType.getRank() != 2 ||
      !indicesType.getElementType().isInteger(32)) {
    return emitOpError("expected indices to be of rank 2 of i32 element type");
  }
  auto indexDepth = getIndexDepth();
  if (ShapedType::isDynamic(indexDepth)) {
    return emitOpError("expected index depth is static");
  }

  ArrayRef<int64_t> dimMap = getDimensionMap();
  if (static_cast<int64_t>(dimMap.size()) != indexDepth) {
    return op->emitOpError("invalid number of dimension map entries ");
  }

  auto originalType = getOriginalType();
  if (isInvalid(dimMap, originalType.getRank()))
    return op->emitOpError("dimension map is invalid");

  // The first dimension of the indices should match the first dimension of the
  // output. They indicate to the number of updates.
  auto updateType = getUpdateType();
  if (updateType.getRank() < 1) {
    return emitOpError("expected update value to be at least rank 1");
  }
  if (!checkDimensionsMatch(indicesType, updateType, 0)) {
    return emitOpError(
        "mismatch in shape of indices and update value at dim#0");
  }
  if (updateType.getRank() - 1 > originalType.getRank()) {
    return emitOpError(
        "update value rank exceeds the rank of the original value");
  }

  // indexDepth + update dims should cover the original dims. The first dim of
  // update is the number of updates.
  if (originalType.getRank() > indexDepth + updateType.getRank() - 1) {
    return emitOpError(
        "index depth and update value does not cover rank of original value");
  }

  // Validate the non-indexed update dims cover the full slice size of the
  // original tensor.
  int64_t fullSliceDims = originalType.getRank() - indexDepth;
  for (auto it :
       llvm::zip(llvm::seq<unsigned>(indexDepth, originalType.getRank()),
                 llvm::seq<unsigned>(updateType.getRank() - fullSliceDims,
                                     updateType.getRank()))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (!originalType.isDynamicDim(originalDim) &&
        updateType.getDimSize(updateDim) >
            originalType.getDimSize(originalDim)) {
      return op->emitOpError("shape of update value dim#")
             << updateDim << " exceeds original value at dim#" << originalDim;
    }
  }

  // Check that the remaining update indices do not exceed the update length.
  int64_t insertDims = originalType.getRank() - updateType.getRank() + 1;
  for (auto it : llvm::zip(
           llvm::seq<unsigned>(insertDims, indexDepth),
           llvm::seq<unsigned>(1, updateType.getRank() - fullSliceDims))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (!originalType.isDynamicDim(originalDim) &&
        updateType.getDimSize(updateDim) >
            originalType.getDimSize(originalDim)) {
      return op->emitOpError("indexed shape of update value dim#")
             << updateDim << " exceeds original value at dim#" << originalDim
             << " " << updateType.getDimSize(updateDim) << " "
             << originalType.getDimSize(originalDim);
    }
  }

  Region &region = this->getRegion();
  Block *body = &region.front();
  if (body->getNumArguments() != 2) {
    return op->emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!getComplexElementTypeOrSelf(arg0Type).isIntOrFloat() ||
      !getComplexElementTypeOrSelf(arg1Type).isIntOrFloat()) {
    return emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != updateType.getElementType()) {
    return emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of update value "
           << updateType.getElementType();
  }
  if (arg1Type != originalType.getElementType()) {
    return emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of original value "
           << originalType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<TMTensor::YieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }
  return success();
}

SmallVector<utils::IteratorType> ScatterOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getUpdateType().getRank(),
                                                 utils::IteratorType::parallel);
  if (!getUniqueIndices()) {
    iteratorTypes[0] = utils::IteratorType::reduction;
  }
  return iteratorTypes;
}

bool ScatterOp::payloadUsesValueFromOperand(OpOperand *opOperand) {
  unsigned bbArgNumber;
  Value operand = opOperand->get();
  if (operand == updates())
    bbArgNumber = 0; // block arg 0 is `update`.
  else {
    bool isValidOperand = operand == indices() || operand == original();
    (void)isValidOperand;
    assert(isValidOperand &&
           "operand must belong to the current tm_tensor.scatter op");
    return true;
  }

  assert(this->getOperation()->getNumRegions() == 1 &&
         "unexpected "
         "missing region (calling `payloadUsesValueFromOperand` on "
         "manually defined named TMTensor op?)");
  Block &block = this->getOperation()->getRegion(0).front();
  return !block.getArgument(bbArgNumber).use_empty();
}

SmallVector<Range> ScatterOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getUpdateType().getRank())) {
    Value ub = getDimValue(builder, loc, updates(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

LogicalResult ScatterOp::generateScalarImplementation(OpBuilder &b,
                                                      Location loc,
                                                      ValueRange ivs) {
  auto indexDepth = getIndexDepth();
  Value update = b.create<memref::LoadOp>(loc, updates(), ivs);
  SmallVector<Value> starts;
  SmallVector<Value> loadIndices;
  loadIndices.push_back(ivs.front());
  loadIndices.push_back(Value());

  // Populate with empty values.
  auto originalTy = original().getType().cast<ShapedType>();
  starts.resize(originalTy.getRank(), Value());
  auto updateIvs = ivs.drop_front(1);

  int64_t offset = starts.size() - updateIvs.size();
  for (auto it : llvm::enumerate(updateIvs)) {
    starts[it.index() + offset] = it.value();
  }

  ArrayRef<int64_t> dimMap = getDimensionMap();
  for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
    loadIndices.back() = b.create<arith::ConstantIndexOp>(loc, i);
    Value idx = b.create<memref::LoadOp>(loc, indices(), loadIndices);
    Value ret = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);

    auto dim = dimMap[i];
    if (starts[dim])
      ret = b.create<arith::AddIOp>(loc, ret, starts[dim]);
    starts[dim] = ret;
  }

  Value init = b.create<memref::LoadOp>(loc, original(), starts);

  IRMapping bvm;
  Block &block = getRegion().front();
  bvm.map(block.getArgument(0), update);
  bvm.map(block.getArgument(1), init);
  for (auto &blockOp : block.without_terminator()) {
    b.clone(blockOp, bvm);
  }
  // The last op is linalg_ext.yield op. Store the operand to
  // destination.
  b.create<memref::StoreOp>(
      loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)),
      original(), starts);
  return success();
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

LogicalResult SortOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs()) {
    return op->emitOpError("does not expect to take any inputs");
  }
  if (getNumOutputs() == 0) {
    return op->emitOpError("expected at least one `outs` operand");
  }

  Block &block = getRegion().front();
  size_t numOutputs = getNumOutputs();
  if (block.getNumArguments() != 2 * numOutputs) {
    return op->emitOpError("region block should have ")
           << 2 * numOutputs << " arguments";
  }

  int64_t rank = getOperandRank();
  int sortDim = getDimension();
  if (sortDim < 0 || sortDim >= rank) {
    return op->emitOpError("dimension must be within (0, ") << rank << "]";
  }

  ArrayRef<int64_t> shape = getOperandShape();
  for (auto indexedOperand : llvm::enumerate(getOutputs())) {
    int index = indexedOperand.index();
    auto operandType = getOperandType(index);
    if (operandType.getRank() != rank) {
      return op->emitOpError("expected operand ")
             << index << " to be rank " << rank << ", same as other operands";
    }
    if (operandType.getShape() != shape) {
      return op->emitOpError("expected operand ")
             << index << " to have same shape as other operands";
    }
    Type elemType = operandType.getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType) {
        return op->emitOpError("region block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
      }
    }
  }

  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return op->emitOpError("should yield exactly one operand");
  }
  auto ty = yieldOp.getOperand(0).getType().dyn_cast<IntegerType>();
  if (!ty || ty.getWidth() != 1) {
    return op->emitOpError("should yield i1 type");
  }

  return success();
}

SmallVector<utils::IteratorType> SortOp::getLoopIteratorTypes() {
  // All loops except the dimension to sort along are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

SmallVector<Range> SortOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = operand(0);
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

LogicalResult SortOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  auto sortDim = getDimension();
  SmallVector<Value> indices, sortBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  // Bubble sort innermost loop.
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value ub;
  if (getOperandType(0).isDynamicDim(sortDim)) {
    ub = b.create<memref::DimOp>(loc, operand(0), sortDim);
  } else {
    ub = b.create<arith::ConstantIndexOp>(
        loc, getOperandType(0).getDimSize(sortDim));
  }
  ub = b.create<arith::SubIOp>(loc, ub, one);
  auto scfFor = b.create<scf::ForOp>(
      loc, zero, ub, one, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iters) {
        SmallVector<Value> indices(ivs);
        Value ivPlusOne = b.create<arith::AddIOp>(loc, iv, one);
        for (auto output : getOutputOperands()) {
          indices[sortDim] = iv;
          sortBlkArgs.push_back(
              b.create<memref::LoadOp>(loc, output->get(), indices));
          indices[sortDim] = ivPlusOne;
          sortBlkArgs.push_back(
              b.create<memref::LoadOp>(loc, output->get(), indices));
        }
      });

  auto &srcBlock = getRegion().front();
  Region &region = scfFor.getRegion();
  IRMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = region.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip(srcBlock.getArguments(), sortBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
  }
  Value cond = bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0));

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(&region.front());
  b.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        // Do not swap the pairs if true.
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        // Swap the pairs if false.
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value ivPlusOne =
            b.create<arith::AddIOp>(loc, scfFor.getInductionVar(), one);
        for (int i = 0, e = getNumOutputs(); i < e; ++i) {
          Value v1 = sortBlkArgs[i * 2];
          Value v2 = sortBlkArgs[i * 2 + 1];
          indices[sortDim] = scfFor.getInductionVar();
          b.create<memref::StoreOp>(loc, v2, getOutputOperand(i)->get(),
                                    indices);
          indices[sortDim] = ivPlusOne;
          b.create<memref::StoreOp>(loc, v1, getOutputOperand(i)->get(),
                                    indices);
        }
        b.create<scf::YieldOp>(loc);
      });
  b.create<scf::YieldOp>(loc);
  return success();
}

bool SortOp::payloadUsesValueFromOperand(OpOperand *opOperand) {
  // All operands of SortOp will be sorted. So, we'll end up loading/storing
  // from them - hence setting this utility to always return `true`.
  return true;
}

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    SmallVector<Value> inputBuffers = getInputBufferOperands();                \
    SmallVector<Value> outputBuffers = getOutputBufferOperands();              \
    getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,        \
                   outputBuffers);                                             \
  }

DEFINE_OP_GET_EFFECTS(AttentionOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(ScatterOp)
DEFINE_OP_GET_EFFECTS(SortOp)

namespace {
/// This is derived from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp without any
/// changes.
struct FoldTensorCastOp : public OpInterfaceRewritePattern<TMTensorOp> {
  using OpInterfaceRewritePattern<TMTensorOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(TMTensorOp op,
                                PatternRewriter &rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getInputAndOutputOperands(), [&](OpOperand *opOperand) {
          if (opOperand->get().isa<BlockArgument>())
            return false;
          auto castOp = opOperand->get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (OpOperand *opOperand : op.getInputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (OpOperand *opOperand : op.getOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand()
                                 : opOperand->get());
      newResultTypes.push_back(newOperands.back().getType());
    }
    // Clone op.
    Operation *newOp =
        op.clone(rewriter, op->getLoc(), newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto result : llvm::zip(op->getResults(), newOp->getResults())) {
      Value oldResult = std::get<0>(result);
      Value newResult = std::get<1>(result);
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TMTensorDialect
//===----------------------------------------------------------------------===//

void TMTensorDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<FoldTensorCastOp>(getContext());
}

#define GET_OP_CLASSES
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.cpp.inc"
