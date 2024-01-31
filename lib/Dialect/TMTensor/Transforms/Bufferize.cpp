//===- Bufferize.cpp - Bufferization of tmtensor ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/PassDetail.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"

using namespace ::mlir;
using namespace ::mlir::torch::TMTensor;

static Value cloneMemref(Location loc, Value memref, OpBuilder &b) {
  auto memrefType = memref.getType().cast<MemRefType>();
  auto alloc = b.create<memref::AllocOp>(
      loc, memref::getMixedSizes(b, loc, memref), memrefType.getElementType());
  b.create<memref::CopyOp>(loc, memref, alloc);
  return alloc;
}

static LogicalResult
allocateBuffersForResults(Location loc, TMTensorOp tmtensorOp,
                          ValueRange outputs,
                          SmallVectorImpl<Value> &resultBuffers, OpBuilder &b) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Allocate a buffer for every tensor result.
  assert(tmtensorOp.getNumOutputs() == tmtensorOp->getNumResults());
  for (const auto &en : llvm::enumerate(tmtensorOp->getResultTypes())) {
    size_t resultIndex = en.index();
    Type resultType = en.value();

    auto tensorType = resultType.dyn_cast<RankedTensorType>();
    if (tensorType == nullptr) {
      tmtensorOp.emitOpError()
          << "tensor to buffer conversion expects ranked tensor results";
      return failure();
    }
    auto tensorShape = tensorType.getShape();
    auto memrefType = MemRefType::get(tensorShape, tensorType.getElementType());
    Value resultTensor = outputs[resultIndex];

    // Clone output buffers whose value is actually used.
    OpOperand *tiedOpOperand = tmtensorOp.getOutputOperand(resultIndex);
    if (tmtensorOp.payloadUsesValueFromOperand(tiedOpOperand)) {
      resultBuffers.push_back(cloneMemref(loc, resultTensor, b));
      continue;
    }

    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers.push_back(b.create<memref::AllocOp>(loc, memrefType));
      continue;
    }

    resultBuffers.push_back(b.create<memref::AllocOp>(
        loc, memref::getMixedSizes(b, loc, resultTensor),
        memrefType.getElementType()));
  }
  return success();
}

/// Create TMTensor op on buffers given the original tensor-based operation and
/// the buffers for the outputs.
static TMTensorOp createTMTensorOpOnBuffers(ConversionPatternRewriter &rewriter,
                                            TMTensorOp tmtensorOp,
                                            ValueRange inputs,
                                            ValueRange outputs) {
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  return cast<TMTensorOp>(
      tmtensorOp.clone(rewriter, tmtensorOp->getLoc(), {}, newOperands));
}

/// Generic conversion pattern that matches any TMTensorOp. This avoids template
/// instantiating one pattern for each TMTensorOp.
class BufferizeAnyTMTensorOp : public OpInterfaceConversionPattern<TMTensorOp> {
public:
  using OpInterfaceConversionPattern<TMTensorOp>::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(TMTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    SmallVector<Value, 2> newOutputBuffers;

    SmallVector<Value> outputs(operands.begin() + op.getNumInputs(),
                               operands.end());
    if (failed(allocateBuffersForResults(loc, op, outputs, newOutputBuffers,
                                         rewriter))) {
      return op.emitOpError()
             << "Failed to allocate buffers for tensor results.";
    }

    SmallVector<Value> inputs(operands.begin(),
                              operands.begin() + op.getNumInputs());
    createTMTensorOpOnBuffers(rewriter, op, inputs, newOutputBuffers);
    // Replace the results of the old op with the new output buffers.
    rewriter.replaceOp(op, newOutputBuffers);
    return success();
  }
};

namespace {
/// Converts TMTensor operations that work on tensor-type operands or results to
/// work on buffers.
struct TMTensorBufferizePass
    : public TMTensorBufferizeBase<TMTensorBufferizePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    torch::TMTensor::TMTensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    bufferization::BufferizeTypeConverter typeConverter;

    // Mark all Standard operations legal.
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect, tensor::TensorDialect>();

    // Mark all TMTensor operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<TMTensorDialect>(isLegalOperation);
    RewritePatternSet patterns(&context);
    patterns.add<BufferizeAnyTMTensorOp>(typeConverter, patterns.getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
torch::TMTensor::createTMTensorBufferizePass() {
  return std::make_unique<TMTensorBufferizePass>();
}
