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
    ResultRange results, ArrayRef<OpOperand *> inputBuffers,
    ArrayRef<OpOperand *> outputBuffers) {
  for (OpResult value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand *value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand *value : outputBuffers) {
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
        return tensor::DimOp::create(builder, loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return memref::DimOp::create(builder, loc, v, dim);
      })
      .Default([&](Type t) { return Value(); });
}

OpFoldResult TMTensor::getDim(OpBuilder &builder, Location loc, Value v,
                              int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
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
  ShapedType valueType = getValueType();

  auto optionalMaskType = getAttnMaskType();
  ShapedType maskType = optionalMaskType ? *optionalMaskType : ShapedType();

  ArrayRef<int64_t> queryShape = queryType.getShape();
  ArrayRef<int64_t> keyShape = keyType.getShape();
  ArrayRef<int64_t> valueShape = valueType.getShape();
  ArrayRef<int64_t> maskShape =
      optionalMaskType ? maskType.getShape() : ArrayRef<int64_t>();

  for (int i = 0, s = queryShape.size() - 2; i < s; ++i) {
    if (keyShape[i] != queryShape[i]) {
      return op->emitOpError("query and key batch mismatch");
    }
  }
  if (keyShape.back() != queryShape.back()) {
    return op->emitOpError("query and key head dimension mismatch");
  }

  for (int i = 0, s = queryShape.size() - 2; i < s; ++i) {
    if (valueShape[i] != queryShape[i]) {
      return op->emitOpError("query and value batch dimension mismatch");
    }
  }
  if (keyShape[keyShape.size() - 2] != valueShape[valueShape.size() - 2]) {
    return op->emitOpError("key and value sequence length dimension mismatch");
  }
  if (optionalMaskType) {
    for (int i = 0, s = maskShape.size() - 2; i < s; ++i) {
      if (maskShape[i] != queryShape[i]) {
        return op->emitOpError("query and mask batch dimension mismatch");
      }
    }
    if (maskShape[maskShape.size() - 2] != queryShape[queryShape.size() - 2]) {
      return op->emitOpError(
          "mask sequence length and query sequence length mismatch");
    }
    if (maskShape[maskShape.size() - 1] != keyShape[keyShape.size() - 2]) {
      return op->emitOpError(
          "mask sequence lengt and key sequence length mismatch");
    }
  }
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
  auto elementType = cast<MemRefType>(lhs.getType()).getElementType();
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  auto rank = outputSizes.size();
  Value reductionDimSize = lhsSizes[lhsSizes.size() - 1];

  // Loop over output
  scf::ParallelOp::create(
      b, loc, SmallVector<Value>(rank, zero), outputSizes,
      SmallVector<Value>(rank, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        Value acc = arith::ConstantOp::create(b, loc, elementType,
                                              b.getFloatAttr(elementType, 0.0));
        Value sum =
            scf::ForOp::create(
                b, loc, zero, reductionDimSize, one, SmallVector<Value>{acc},
                [&](OpBuilder &b, Location loc, Value i, ValueRange accs) {
                  SmallVector<Value> lhsIVs(localIVs), rhsIVs(localIVs);
                  lhsIVs[lhsIVs.size() - 1] = i;
                  rhsIVs[rhsIVs.size() - 2] = i;
                  if (transposed)
                    std::swap(rhsIVs[rhsIVs.size() - 1],
                              rhsIVs[rhsIVs.size() - 2]);

                  Value acc = accs[0];
                  Value rElem = memref::LoadOp::create(b, loc, lhs, lhsIVs);
                  Value cElem = memref::LoadOp::create(b, loc, rhs, rhsIVs);
                  Value x = arith::MulFOp::create(b, loc, rElem, cElem);
                  x = arith::AddFOp::create(b, loc, x, acc);

                  scf::YieldOp::create(b, loc, x);
                })
                ->getResult(0);
        memref::StoreOp::create(b, loc, sum, output, localIVs);
      });
}

LogicalResult AttentionOp::generateScalarImplementation(OpBuilder &b,
                                                        Location loc,
                                                        ValueRange ivs) {

  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();

  auto optionalMask = getAttnMask();
  Value mask = optionalMask ? *optionalMask : Value();

  Value output = getOutput();
  auto queryType = cast<MemRefType>(query.getType());
  auto keyType = cast<MemRefType>(key.getType());
  auto valueType = cast<MemRefType>(value.getType());
  auto maskType = mask ? cast<MemRefType>(mask.getType()) : MemRefType();
  auto queryRank = queryType.getRank();
  auto keyRank = keyType.getRank();
  auto valueRank = valueType.getRank();
  auto keySizes = keyType.getShape();
  Type elementType = queryType.getElementType();

  Value zeroF = arith::ConstantOp::create(b, loc, elementType,
                                          b.getFloatAttr(elementType, 0.0));
  Value negInfF = arith::ConstantOp::create(
      b, loc, elementType,
      b.getFloatAttr(elementType, -std::numeric_limits<double>::infinity()));

  // TODO: This needs to be fixed, it assumes everything is dynamic however if
  // any shapes are static the `memref.alloc` generated is illegal.
  SmallVector<Value> queryDynSizes, keyDynSizes, valueDynSizes, outputDynSizes;
  for (auto i = 0; i < queryRank; i++)
    queryDynSizes.push_back(memref::DimOp::create(b, loc, query, i));
  for (auto i = 0; i < keyRank; i++)
    keyDynSizes.push_back(memref::DimOp::create(b, loc, key, i));
  for (auto i = 0; i < valueRank; i++)
    valueDynSizes.push_back(memref::DimOp::create(b, loc, value, i));
  for (auto i = 0; i < queryRank; i++)
    outputDynSizes.push_back(memref::DimOp::create(b, loc, output, i));

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
      memref::AllocOp::create(b, loc, weightType, weightFilteredDynSizes);
  matmul(b, loc, query, queryDynSizes, key, keyDynSizes, weight, weightDynSizes,
         /*transposed=*/true);

  // weight = softmax(weight)
  Value dim = weightDynSizes[weightRank - 1];
  Value scaleFactor = math::SqrtOp::create(
      b, loc,
      arith::UIToFPOp::create(
          b, loc, elementType,
          arith::IndexCastUIOp::create(b, loc, b.getI32Type(),
                                       queryDynSizes[queryRank - 1])));

  // weight = (weight - max(weight)) / math.sqrt(querySizes[-1])
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  scf::ParallelOp::create(
      b, loc, SmallVector<Value>(weightRank, zero), weightDynSizes,
      SmallVector<Value>(weightRank, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        Value x = memref::LoadOp::create(b, loc, weight, localIVs);
        x = arith::DivFOp::create(b, loc, x, scaleFactor);
        memref::StoreOp::create(b, loc, x, weight, localIVs);
      });

  // Apply mask to weights if mask is given
  if (mask) {
    scf::ParallelOp::create(
        b, loc, SmallVector<Value>(weightRank, zero), weightDynSizes,
        SmallVector<Value>(weightRank, one),
        [&](OpBuilder &b, Location loc, ValueRange localIVs) {
          Value weightValue = memref::LoadOp::create(b, loc, weight, localIVs);
          Value maskValue = memref::LoadOp::create(b, loc, mask, localIVs);
          if (maskType.getElementType().isInteger(1)) {
            maskValue =
                arith::SelectOp::create(b, loc, maskValue, zeroF, negInfF);
          }
          Value maskedWeight =
              arith::AddFOp::create(b, loc, weightValue, maskValue);
          memref::StoreOp::create(b, loc, maskedWeight, weight, localIVs);
        });
  }

  // calculate max(weight)
  Value init = memref::LoadOp::create(b, loc, weight,
                                      SmallVector<Value>(weightRank, zero));
  Value globalMax =
      scf::ParallelOp::create(
          b, loc, SmallVector<Value>(weightRank, zero), weightDynSizes,
          SmallVector<Value>(weightRank, one), init,
          [&](OpBuilder &b, Location loc, ValueRange localIVs,
              ValueRange accs) {
            auto reduceOp = scf::ReduceOp::create(b, loc, init);
            // Build reduce body.
            Block &reductionBody = reduceOp.getReductions()[0].front();
            auto bodyBuilder = OpBuilder::atBlockEnd(&reductionBody);
            Value acc = reductionBody.getArgument(0);
            Value x =
                memref::LoadOp::create(bodyBuilder, loc, weight, localIVs);
            Value max = arith::MaximumFOp::create(bodyBuilder, loc, x, acc);
            scf::ReduceReturnOp::create(bodyBuilder, loc, max);
          })
          .getResult(0);
  // weight = (weight - max(weight)) / math.sqrt(querySizes[-1])
  scf::ParallelOp::create(
      b, loc, SmallVector<Value>(weightRank, zero), weightDynSizes,
      SmallVector<Value>(weightRank, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        Value x = memref::LoadOp::create(b, loc, weight, localIVs);
        x = arith::SubFOp::create(b, loc, x, globalMax);
        memref::StoreOp::create(b, loc, x, weight, localIVs);
      });
  // calculate exp(weight)
  SmallVector<Value> min(weightRank, zero),
      max(weightDynSizes.begin(), weightDynSizes.end()), steps(weightRank, one);
  scf::ParallelOp::create(
      b, loc, min, max, steps,
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        Value x = memref::LoadOp::create(b, loc, weight, localIVs);
        x = math::ExpOp::create(b, loc, x);
        memref::StoreOp::create(b, loc, x, weight, localIVs);
      });

  llvm::SmallVector<Value> expWeightDynDims(weightFilteredDynSizes);
  if (weightSizes.back() == ShapedType::kDynamic)
    expWeightDynDims.resize(expWeightDynDims.size() - 1);

  Value expWeightSum = memref::AllocOp::create(
      b, loc,
      MemRefType::get(
          SmallVector<int64_t>(weightSizes.begin(), weightSizes.end() - 1),
          elementType),
      expWeightDynDims);
  scf::ParallelOp::create(
      b, loc, SmallVector<Value>(weightRank - 1, zero),
      SmallVector<Value>{weightDynSizes.begin(), weightDynSizes.end() - 1},
      SmallVector<Value>(weightRank - 1, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        memref::StoreOp::create(b, loc, zeroF, expWeightSum, localIVs);
      });
  // Loop over all dims but -1
  scf::ParallelOp::create(
      b, loc, SmallVector<Value>(weightRank - 1, zero),
      SmallVector<Value>(weightDynSizes.begin(), weightDynSizes.end() - 1),
      SmallVector<Value>(weightRank - 1, one),
      [&](OpBuilder &b, Location loc, ValueRange outsideDims) {
        // Sum over last dim
        scf::ParallelOp::create(
            b, loc, zero, dim, one,
            [&](OpBuilder &b, Location loc, ValueRange localIVs) {
              SmallVector<Value> coords(outsideDims);
              coords.push_back(localIVs[0]);
              Value x =
                  memref::LoadOp::create(b, loc, expWeightSum, outsideDims);
              Value y = memref::LoadOp::create(b, loc, weight, coords);
              Value sum = arith::AddFOp::create(b, loc, x, y);
              memref::StoreOp::create(b, loc, sum, expWeightSum, outsideDims);
            });
      });
  // calculate exp(weight) / sum(exp(weight))
  scf::ParallelOp::create(
      b, loc, SmallVector<Value>(weightRank, zero),
      SmallVector<Value>(weightDynSizes.begin(), weightDynSizes.end()),
      SmallVector<Value>(weightRank, one),
      [&](OpBuilder &b, Location loc, ValueRange localIVs) {
        SmallVector<Value> sumIVs(localIVs);
        sumIVs.pop_back();

        Value x = memref::LoadOp::create(b, loc, weight, localIVs);
        Value sum = memref::LoadOp::create(b, loc, expWeightSum, sumIVs);
        Value divResult = arith::DivFOp::create(b, loc, x, sum);

        // Set to 0 if sum is 0 (can occur during boolean mask / large negative
        // QK)
        Value isSumZero = arith::CmpFOp::create(
            b, loc, arith::CmpFPredicate::OEQ, sum, zeroF);
        Value result =
            arith::SelectOp::create(b, loc, isSumZero, zeroF, divResult);

        memref::StoreOp::create(b, loc, result, weight, localIVs);
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
  if (!isa<ShapedType>(input().getType())) {
    return emitOpError("expected first input element type to be shaped");
  }
  auto accumulatorType = cast<ShapedType>(accumulator().getType());
  auto inputType = cast<ShapedType>(input().getType());
  auto outputType = cast<ShapedType>(output().getType());
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
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
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
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  uint64_t scanDim = getDimension();
  Value cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                     indices[scanDim], zero);
  bool isInclusive = getInclusive();
  SmallVector<Value> accIndices;
  for (size_t i = 0; i < indices.size(); i++) {
    if (i != scanDim)
      accIndices.push_back(indices[i]);
  }

  auto scfIf = scf::IfOp::create(
      b, loc, cond,
      [&](OpBuilder &b, Location loc) {
        if (isInclusive) {
          auto value = memref::LoadOp::create(b, loc, input(), indices);
          memref::StoreOp::create(b, loc, value, output(), indices);
        } else {
          auto value =
              memref::LoadOp::create(b, loc, accumulator(), accIndices);
          memref::StoreOp::create(b, loc, value, output(), indices);
        }
        scf::YieldOp::create(b, loc);
      },
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value iv = indices[scanDim];
        Value ivMinusOne = arith::SubIOp::create(b, loc, iv, one);
        indices[scanDim] = ivMinusOne;
        scanBlkArgs.push_back(
            memref::LoadOp::create(b, loc, output(), indices));
        Value i0;
        if (!isInclusive)
          i0 = memref::LoadOp::create(b, loc, input(), indices);
        indices[scanDim] = iv;
        if (isInclusive)
          i0 = memref::LoadOp::create(b, loc, input(), indices);
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
    memref::StoreOp::create(
        b, loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        output(), indices);
    memref::StoreOp::create(
        b, loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        accumulator(), accIndices);
    scf::YieldOp::create(b, loc);
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
      !isa<IntegerType>(indicesType.getElementType())) {
    return emitOpError(
        "expected indices to be of rank 2 of integer element type");
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
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
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
  Value update = memref::LoadOp::create(b, loc, updates(), ivs);
  SmallVector<Value> starts;
  SmallVector<Value> loadIndices;
  loadIndices.push_back(ivs.front());
  loadIndices.push_back(Value());

  // Populate with empty values.
  auto originalTy = cast<ShapedType>(original().getType());
  starts.resize(originalTy.getRank(), Value());
  auto updateIvs = ivs.drop_front(1);

  int64_t offset = starts.size() - updateIvs.size();
  for (auto it : llvm::enumerate(updateIvs)) {
    starts[it.index() + offset] = it.value();
  }

  ArrayRef<int64_t> dimMap = getDimensionMap();
  for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
    loadIndices.back() = arith::ConstantIndexOp::create(b, loc, i);
    Value idx = memref::LoadOp::create(b, loc, indices(), loadIndices);
    Value ret = arith::IndexCastOp::create(b, loc, b.getIndexType(), idx);

    auto dim = dimMap[i];
    if (starts[dim])
      ret = arith::AddIOp::create(b, loc, ret, starts[dim]);
    starts[dim] = ret;
  }

  Value init = memref::LoadOp::create(b, loc, original(), starts);

  IRMapping bvm;
  Block &block = getRegion().front();
  bvm.map(block.getArgument(0), update);
  bvm.map(block.getArgument(1), init);
  for (auto &blockOp : block.without_terminator()) {
    b.clone(blockOp, bvm);
  }
  // The last op is linalg_ext.yield op. Store the operand to
  // destination.
  memref::StoreOp::create(
      b, loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)),
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
  auto ty = dyn_cast<IntegerType>(yieldOp.getOperand(0).getType());
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
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
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
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value ub;
  if (getOperandType(0).isDynamicDim(sortDim)) {
    ub = memref::DimOp::create(b, loc, operand(0), sortDim);
  } else {
    ub = arith::ConstantIndexOp::create(b, loc,
                                        getOperandType(0).getDimSize(sortDim));
  }
  ub = arith::SubIOp::create(b, loc, ub, one);
  auto scfFor = scf::ForOp::create(
      b, loc, zero, ub, one, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iters) {
        SmallVector<Value> indices(ivs);
        Value ivPlusOne = arith::AddIOp::create(b, loc, iv, one);
        for (auto output : getOutputOperands()) {
          indices[sortDim] = iv;
          sortBlkArgs.push_back(
              memref::LoadOp::create(b, loc, output->get(), indices));
          indices[sortDim] = ivPlusOne;
          sortBlkArgs.push_back(
              memref::LoadOp::create(b, loc, output->get(), indices));
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
  scf::IfOp::create(
      b, loc, cond,
      [&](OpBuilder &b, Location loc) {
        // Do not swap the pairs if true.
        scf::YieldOp::create(b, loc);
      },
      [&](OpBuilder &b, Location loc) {
        // Swap the pairs if false.
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value ivPlusOne =
            arith::AddIOp::create(b, loc, scfFor.getInductionVar(), one);
        for (int i = 0, e = getNumOutputs(); i < e; ++i) {
          Value v1 = sortBlkArgs[i * 2];
          Value v2 = sortBlkArgs[i * 2 + 1];
          indices[sortDim] = scfFor.getInductionVar();
          memref::StoreOp::create(b, loc, v2, getOutputOperand(i)->get(),
                                  indices);
          indices[sortDim] = ivPlusOne;
          memref::StoreOp::create(b, loc, v1, getOutputOperand(i)->get(),
                                  indices);
        }
        scf::YieldOp::create(b, loc);
      });
  scf::YieldOp::create(b, loc);
  return success();
}

bool SortOp::payloadUsesValueFromOperand(OpOperand *opOperand) {
  // All operands of SortOp will be sorted. So, we'll end up loading/storing
  // from them - hence setting this utility to always return `true`.
  return true;
}

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

LogicalResult TopkOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs() != 1 && getNumInputs() != 2) {
    return op->emitOpError("expected one or two input operands");
  }
  if (getNumOutputs() != 2) {
    return op->emitOpError("expected two output operands");
  }
  // First check added to eliminate comparison of different int types
  if (getInputRank() < 0 ||
      (getDimension() >= static_cast<uint64_t>(getInputRank()))) {
    return op->emitOpError("dimension exceeds rank");
  }
  // Ensure input/output element types match
  auto inputValuesType = cast<ShapedType>(values().getType());
  auto outputValuesType = cast<ShapedType>(outputValues().getType());
  if (inputValuesType.getElementType() != outputValuesType.getElementType()) {
    return op->emitOpError("expected input/output value types to be identical");
  }
  // Indices must be int if provided
  auto outputIndicesType = cast<ShapedType>(outputIndices().getType());
  if (auto inputIndices = indices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (!inputIndicesType.getElementType().isInteger(32) ||
        !outputIndicesType.getElementType().isInteger(32)) {
      return op->emitOpError("expected input/output indices types to be int32");
    }
  }

  // Ranks must match
  if (inputValuesType.getRank() != outputValuesType.getRank()) {
    return op->emitOpError("expected input/output to have the same rank");
  }
  if (auto inputIndices = indices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (inputIndicesType.getRank() != outputIndicesType.getRank()) {
      return op->emitOpError("expected input/output to have the same rank");
    }
  }
  // Input indicies and values must have the same shape.
  if (auto inputIndices = indices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (failed(verifyCompatibleShape(inputValuesType, inputIndicesType)))
      return op->emitOpError("input indices/values shape must match");
  }
  // Output indicies and values must have the same shape.
  if (failed(verifyCompatibleShape(outputValuesType, outputIndicesType)))
    return op->emitOpError("output indices/values shape must match");
  // Input shape must match the output shape except for the dimension()
  uint64_t dim = getDimension();
  if (!llvm::all_of(llvm::enumerate(llvm::zip(inputValuesType.getShape(),
                                              outputValuesType.getShape())),
                    [dim](auto e) {
                      if (e.index() == dim) {
                        return true;
                      }
                      std::tuple<int64_t, int64_t> s = e.value();
                      return succeeded(verifyCompatibleShape(std::get<0>(s),

                                                             std::get<1>(s)));
                    })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  // Check region compatibility
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return op->emitOpError("region block should have 2 arguments");
  }
  if (block.getArgument(0).getType() != inputValuesType.getElementType() ||
      block.getArgument(1).getType() != inputValuesType.getElementType()) {
    return op->emitOpError("region block types must match input");
  }
  auto terminatorOp = llvm::dyn_cast<YieldOp>(block.getTerminator());
  if (!terminatorOp || !terminatorOp.getOperand(0).getType().isInteger(1)) {
    return op->emitOpError("region block must end with a linalg_ext.yield i1!");
  }
  return success();
}

SmallVector<utils::IteratorType> TopkOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

SmallVector<Range> TopkOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getInputRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value source = values();
  for (auto dim : llvm::enumerate(getInputType().getShape())) {
    loopBounds[dim.index()].offset = zero;
    loopBounds[dim.index()].size =
        getDimValue(builder, loc, source, dim.index());
    loopBounds[dim.index()].stride = one;
  }
  return loopBounds;
}

LogicalResult TopkOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  uint64_t kDim = getDimension();
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value initialValue = memref::LoadOp::create(b, loc, values(), ivs);

  // If the indices tensor is not provided, the value index is derived from the
  // loop induction variables.
  Value initialIndex;
  if (indices()) {
    initialIndex = memref::LoadOp::create(b, loc, *indices(), ivs);
  } else {
    Value rawInitialIndex = ivs[kDim];
    initialIndex =
        arith::IndexCastOp::create(b, loc, b.getI32Type(), rawInitialIndex);
  }

  // Compute K (ub) from the selected dim of the output
  Value ub = memref::DimOp::create(b, loc, outputValues(), getDimension());

  // Inner K loop functions:
  //   Load current K value and index
  //   Compare N/K using inserted block compare
  //   Check if N == K using strict weak ordering, select which index came first
  //   Select new K value from N/K comparison
  //   Select new K index from N/K comparison or which index came first
  //   Store new k value and index
  //   Yield loop carry values after K selection
  Value kValue, kIndex;
  auto scfFor = scf::ForOp::create(
      b, loc, zero, ub, one, ValueRange{initialValue, initialIndex},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopCarryValues) {
        SmallVector<Value> indices(ivs);
        indices[kDim] = iv;
        kValue = memref::LoadOp::create(b, loc, outputValues(), indices);
        kIndex = memref::LoadOp::create(b, loc, outputIndices(), indices);
      });

  SmallVector<Value> indices(ivs);
  indices[kDim] = scfFor.getInductionVar();
  auto loopCarryValues = scfFor.getRegionIterArgs();

  // Retrieve region as black box comparision function f(x,y). Plug into op.
  auto &srcBlock = getRegion().front();
  IRMapping bvmF; // f(x,y)
  IRMapping bvmR; // f(y,x)
  {
    // Save previous insertion point. Continue within loop body.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(&scfFor.getRegion().front());
    SmallVector<Value> forwardValues{loopCarryValues[0], kValue};
    SmallVector<Value> reverseValues{kValue, loopCarryValues[0]};
    for (auto it : llvm::zip(srcBlock.getArguments(), forwardValues)) {
      bvmF.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto it : llvm::zip(srcBlock.getArguments(), reverseValues)) {
      bvmR.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvmF);
      b.clone(blockOp, bvmR);
    }
    Value forwardCmpRes = bvmF.lookup(srcBlock.getTerminator()->getOperand(0));
    Value reverseCmpRes = bvmR.lookup(srcBlock.getTerminator()->getOperand(0));

    // Check value equality using strictly weak ordering from the region:
    //   f(x,y) --> forwardCmpRes
    //   f(y,x) --> reverseCmpRes
    //   if forwardCmpRes == reverseCmpRes then select which came first
    Value cmpValuesEqual = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, forwardCmpRes, reverseCmpRes);
    Value cmpFirstIndex = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::slt, loopCarryValues[1], kIndex);
    Value combinedCmpEqRes =
        arith::AndIOp::create(b, loc, cmpValuesEqual, cmpFirstIndex);
    // True if N > K or N came before K
    Value indexCmpRes =
        arith::OrIOp::create(b, loc, forwardCmpRes, combinedCmpEqRes);
    // Select results for K based on comparisons
    Value resultKValue = arith::SelectOp::create(b, loc, forwardCmpRes,
                                                 loopCarryValues[0], kValue);
    Value resultKIndex = arith::SelectOp::create(b, loc, indexCmpRes,
                                                 loopCarryValues[1], kIndex);
    memref::StoreOp::create(b, loc, resultKValue, outputValues(), indices);
    memref::StoreOp::create(b, loc, resultKIndex, outputIndices(), indices);
    // Select loop carry, opposite of K results
    Value resultCarryValue = arith::SelectOp::create(
        b, loc, forwardCmpRes, kValue, loopCarryValues[0]);
    Value resultCarryIndex = arith::SelectOp::create(
        b, loc, indexCmpRes, kIndex, loopCarryValues[1]);
    scf::YieldOp::create(b, loc,
                         ValueRange{resultCarryValue, resultCarryIndex});
  }
  return success();
}

bool TopkOp::payloadUsesValueFromOperand(OpOperand *opOperand) {
  // Set to true so that output operands are always initialized.
  return true;
}

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    OpOperandVector inputBuffers = getInputBufferOperands();                   \
    OpOperandVector outputBuffers = getOutputBufferOperands();                 \
    getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,        \
                   outputBuffers);                                             \
  }

DEFINE_OP_GET_EFFECTS(AttentionOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(ScatterOp)
DEFINE_OP_GET_EFFECTS(SortOp)
DEFINE_OP_GET_EFFECTS(TopkOp)

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
          if (isa<BlockArgument>(opOperand->get()))
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
        replacements.push_back(tensor::CastOp::create(
            rewriter, op->getLoc(), oldResult.getType(), newResult));
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
