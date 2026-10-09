//===- Bufferize.cpp - Bufferization of tmtensor ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"

using namespace ::mlir;
using namespace ::mlir::torch::TMTensor;
namespace mlir::torch::TMTensor {

#define GEN_PASS_DEF_TMTENSORBUFFERIZE
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h.inc"

static LogicalResult
allocateBuffersForResults(Location loc, TMTensorOp tmtensorOp,
                          SmallVectorImpl<Value> &resultBuffers, OpBuilder &b) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Allocate a buffer for every tensor result.
  assert(tmtensorOp.getNumOutputs() == tmtensorOp->getNumResults());
  for (const auto &en : llvm::enumerate(tmtensorOp->getResultTypes())) {
    size_t resultIndex = en.index();
    Type resultType = en.value();

    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (tensorType == nullptr) {
      tmtensorOp.emitOpError()
          << "tensor to buffer conversion expects ranked tensor results";
      return failure();
    }
    auto tensorShape = tensorType.getShape();
    auto memrefType = MemRefType::get(tensorShape, tensorType.getElementType());

    // Clone output buffers whose value is actually used.
    OpOperand *tiedOpOperand = tmtensorOp.getOutputOperand(resultIndex);
    if (tmtensorOp.payloadUsesValueFromOperand(tiedOpOperand)) {
      // The op updates this operand in place, so the buffer it writes has to
      // start out holding the operand's incoming value. Copy from the *tensor*
      // operand, not from the buffer the conversion driver handed us: a copy
      // with a tensor source is itself the read of that tensor, so the read
      // lands here, at the op, instead of at the operand's definition -- the
      // same reason the `ins` crossings below are built here.
      Value alloc = memref::AllocOp::create(
          b, loc, tensor::getMixedSizes(b, loc, tiedOpOperand->get()),
          memrefType.getElementType());
      bufferization::MaterializeInDestinationOp::create(
          b, loc, /*result=*/TypeRange{}, tiedOpOperand->get(), alloc,
          /*restrict=*/true, /*writable=*/true);
      resultBuffers.push_back(alloc);
      continue;
    }

    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers.push_back(memref::AllocOp::create(b, loc, memrefType));
      continue;
    }

    // The payload does not read this output, so the buffer needs the operand's
    // shape but not its contents. Take the dynamic sizes from the *tensor*
    // operand; the destination-passing-style contract guarantees they match the
    // result's -- "Init operands and their tied OpResults have the same type.
    // Dynamic dimension sizes also match at runtime."
    resultBuffers.push_back(memref::AllocOp::create(
        b, loc, tensor::getMixedSizes(b, loc, tiedOpOperand->get()),
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

    if (failed(
            allocateBuffersForResults(loc, op, newOutputBuffers, rewriter))) {
      return op.emitOpError()
             << "Failed to allocate buffers for tensor results.";
    }

    // Take each tensor `ins` operand's buffer here, at the op that reads it,
    // instead of using the operand the conversion driver converted for us. The
    // driver pins a target materialization to the operand's *definition*, which
    // leaves the read invisible to a later full bufferization: this pass is
    // partial, so the TMTensor op it feeds reads raw memory, and One-Shot
    // Bufferize analyses tensor operands only. Any write between the definition
    // and this op may then be bufferized in place over the very buffer we were
    // handed, silently changing the value the op observes. Stating the crossing
    // here instead puts the read at the program point where it happens, so
    // One-Shot sees the read-after-write and preserves the value. `read_only`
    // is truthful -- a TMTensor op never writes an `ins` operand -- and keeps
    // the crossing from counting as a write as well.
    SmallVector<Value> inputs;
    for (OpOperand &opOperand :
         op->getOpOperands().take_front(op.getNumInputs())) {
      auto tensorType = dyn_cast<RankedTensorType>(opOperand.get().getType());
      if (!tensorType) {
        inputs.push_back(operands[opOperand.getOperandNumber()]);
        continue;
      }
      inputs.push_back(bufferization::ToBufferOp::create(
          rewriter, loc,
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()),
          opOperand.get(), /*read_only=*/true));
    }
    createTMTensorOpOnBuffers(rewriter, op, inputs, newOutputBuffers);
    // Replace the results of the old op with the new output buffers.
    rewriter.replaceOp(op, newOutputBuffers);
    return success();
  }
};

namespace {

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<BaseMemRefType>(inputs[0].getType()));
  return bufferization::ToTensorOp::create(builder, loc, type, inputs[0],
                                           /*restrict=*/true);
}

/// Converts TMTensor operations that work on tensor-type operands or results to
/// work on buffers.
struct TMTensorBufferizePass
    : public impl::TMTensorBufferizeBase<TMTensorBufferizePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    torch::TMTensor::TMTensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    // Since the `BufferizeTypeConverter` has been removed here
    // https://github.com/llvm/llvm-project/commit/2ff2e871f5e632ea493efaf4f2192f8b18a54ab1,
    // hence we have inlined the converter here.
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    // Convert RankedTensorType to MemRefType.
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    // Convert UnrankedTensorType to UnrankedMemRefType.
    typeConverter.addConversion([](UnrankedTensorType type) -> Type {
      return UnrankedMemRefType::get(type.getElementType(), 0);
    });
    typeConverter.addSourceMaterialization(materializeToTensor);
    typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                              BaseMemRefType type,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      assert(inputs.size() == 1 && "expected exactly one input");
      if (auto inputType = dyn_cast<MemRefType>(inputs[0].getType())) {
        // MemRef to MemRef cast.
        assert(inputType != type && "expected different types");
        // Ranked to unranked casts must be explicit.
        auto rankedDestType = dyn_cast<MemRefType>(type);
        if (!rankedDestType)
          return nullptr;
        bufferization::BufferizationOptions options;
        options.bufferAlignment = 0;
        FailureOr<Value> replacement = castOrReallocMemRefValue(
            builder, inputs[0], rankedDestType, options);
        if (failed(replacement))
          return nullptr;
        return *replacement;
      }
      if (isa<TensorType>(inputs[0].getType())) {
        // Tensor to MemRef cast.
        return bufferization::ToBufferOp::create(builder, loc, type, inputs[0]);
      }
      llvm_unreachable("only tensor/memref input types supported");
    });

    // Mark all Standard operations legal.
    target.addLegalDialect<
        arith::ArithDialect, bufferization::BufferizationDialect,
        func::FuncDialect, memref::MemRefDialect, tensor::TensorDialect>();

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

std::unique_ptr<OperationPass<func::FuncOp>> createTMTensorBufferizePass() {
  return std::make_unique<TMTensorBufferizePass>();
}

} // namespace mlir::torch::TMTensor
