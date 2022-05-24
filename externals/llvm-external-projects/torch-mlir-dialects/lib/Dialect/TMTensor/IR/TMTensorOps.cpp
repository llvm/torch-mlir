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
#include "mlir/Dialect/SCF/SCF.h"
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
    if (i != dimension())
      expectedAccumulatorShape.push_back(inputShapes[i]);
  }
  if (llvm::any_of(llvm::zip(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamicSize &&
                            std::get<1>(s) != ShapedType::kDynamicSize &&
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
                     return std::get<0>(s) != ShapedType::kDynamicSize &&
                            std::get<1>(s) != ShapedType::kDynamicSize &&
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

SmallVector<StringRef> ScanOp::getLoopIteratorTypes() {
  SmallVector<StringRef> iteratorTypes(getOperandRank(),
                                       getParallelIteratorTypeName());
  iteratorTypes[dimension()] = getReductionIteratorTypeName();
  return iteratorTypes;
}

bool ScanOp::payloadUsesValueFromOperand(OpOperand *opOperand) {
  Value operand = opOperand->get();
  if (operand == accumulator())
    return !inclusive();
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
  uint64_t scanDim = dimension();
  Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                       indices[scanDim], zero);
  bool isInclusive = inclusive();
  SmallVector<Value> accIndices;
  for (size_t i = 0; i < indices.size(); i++) {
    if (i != scanDim)
      accIndices.push_back(indices[i]);
  }

  auto scfIf = b.create<scf::IfOp>(
      loc, TypeRange{}, cond,
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

  auto &srcBlock = region().front();
  Region &thisRegion = scfIf.getElseRegion();
  BlockAndValueMapping bvm;
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

LogicalResult ScanOp::fold(ArrayRef<Attribute>,
                           SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//
LogicalResult ScatterOp::verify() {
  if (inputs().size() != 2) {
    return emitOpError("expected two input operands");
  }
  if (outputs().size() != 1) {
    return emitOpError("expected one output operand");
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
  if (indexDepth == ShapedType::kDynamicSize) {
    return emitOpError("expected index depth is static");
  }

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
  auto originalType = getOriginalType();
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

  // Validate the non-indexed update dims covier the full slice size of the
  // original tensor.
  int64_t fullSliceDims = originalType.getRank() - indexDepth;
  for (auto it :
       llvm::zip(llvm::seq<unsigned>(indexDepth, originalType.getRank()),
                 llvm::seq<unsigned>(updateType.getRank() - fullSliceDims,
                                     updateType.getRank()))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (updateType.getDimSize(updateDim) !=
        originalType.getDimSize(originalDim)) {
      return emitOpError("mismatch in shape of update value dim#")
             << updateDim << " and original value at dim#" << originalDim;
    }
  }

  // Check that the remaining update indices do not exceed the update length.
  int64_t insertDims = originalType.getRank() - updateType.getRank() + 1;
  for (auto it : llvm::zip(
           llvm::seq<unsigned>(insertDims, indexDepth),
           llvm::seq<unsigned>(1, updateType.getRank() - fullSliceDims))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (updateType.getDimSize(updateDim) >
        originalType.getDimSize(originalDim)) {
      return emitOpError("indexed shape of update value dim#")
             << updateDim << " exceeds original value at dim#" << originalDim
             << " " << updateType.getDimSize(updateDim) << " "
             << originalType.getDimSize(originalDim);
    }
  }

  Region &thisRegion = region();
  Block *body = &thisRegion.front();
  if (body->getNumArguments() != 2) {
    return emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!arg0Type.isIntOrFloat() || !arg1Type.isIntOrFloat()) {
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

SmallVector<StringRef> ScatterOp::getLoopIteratorTypes() {
  SmallVector<StringRef> iteratorTypes(getUpdateType().getRank(),
                                       getParallelIteratorTypeName());
  if (!unique_indices()) {
    iteratorTypes[0] = getReductionIteratorTypeName();
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

  for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
    loadIndices.back() = b.create<arith::ConstantIndexOp>(loc, i);
    Value idx = b.create<memref::LoadOp>(loc, indices(), loadIndices);
    Value cast = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);

    if (starts[i])
      cast = b.create<arith::AddIOp>(loc, cast, starts[i]);
    starts[i] = cast;
  }

  Value init = b.create<memref::LoadOp>(loc, original(), starts);

  BlockAndValueMapping bvm;
  Block &block = region().front();
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

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    SmallVector<Value> inputBuffers = getInputBufferOperands();                \
    SmallVector<Value> outputBuffers = getOutputBufferOperands();              \
    getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,        \
                   outputBuffers);                                             \
  }

DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(ScatterOp)

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
                                ? tensorCastOp.source()
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
