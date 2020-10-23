//===- RecognizeKernels.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Dialect/Torch/IR/OpInterfaces.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aten-recognize-kernels"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::aten;
using namespace mlir::NPCOMP::Torch;

namespace {

struct TypeConversion {
  Type targetType;
  std::function<Value(Location loc, Value originalValue,
                      PatternRewriter &rewriter)>
      callback;
};

/// Converts a Torch argument type so it is compatible with a target type
/// and flags. "Source" refers to the original operand value/type. "Target"
/// refers to the new op's operand value/type.
Optional<TypeConversion>
convertTorchArgType(StringRef sourceTorchType, StringRef targetTorchType,
                    KernelValueConversion::BitMask flag, Type sourceMlirType) {
  using KVC = KernelValueConversion::BitMask;
  // Default trivial case.
  if (sourceTorchType == targetTorchType && flag == 0)
    return TypeConversion{sourceMlirType, nullptr};

  // Immutable tensor conversion.
  if (flag & KVC::kImmutableTensor) {
    if (sourceTorchType != "Tensor" || targetTorchType != "Tensor")
      return None;

    // Already immutable.
    if (sourceMlirType.isa<TensorType>())
      return TypeConversion{sourceMlirType, nullptr};

    // Convert NdArray type.
    if (auto ndArrayType = sourceMlirType.dyn_cast<Numpy::NdArrayType>()) {
      auto tensorType = ndArrayType.toTensorType();
      auto callback = [=](Location loc, Value originalValue,
                          PatternRewriter &rewriter) -> Value {
        return rewriter.create<Numpy::CopyToTensorOp>(loc, tensorType,
                                                      originalValue);
      };
      return TypeConversion{tensorType, callback};
    }

    return None;
  }

  // TODO: Special case promotions and conversions.
  return None;
}

/// Converts a Torch result type so it is compatible with the target type and
/// flags of a new op. "Source" refers to the original result value/type.
/// "Target" refers to the new ops's result value/type. The conversions
/// supported for results are, in general, much more constrained than those
/// supported for operands since these vary far less. The returned conversion
/// callback will convert from the target's type to the source's type.
Optional<TypeConversion>
convertTorchReturnType(StringRef sourceTorchType, StringRef targetTorchType,
                       KernelValueConversion::BitMask flag,
                       Type sourceMlirType) {
  using KVC = KernelValueConversion::BitMask;
  // Default trivial case.
  if (sourceTorchType == targetTorchType && flag == 0)
    return TypeConversion{sourceMlirType, nullptr};

  // Immutable tensor conversion.
  if (flag & KVC::kImmutableTensor) {
    if (sourceTorchType != "Tensor" || targetTorchType != "Tensor")
      return None;

    // Already immutable.
    if (sourceMlirType.isa<TensorType>())
      return TypeConversion{sourceMlirType, nullptr};

    // Convert NdArray type.
    if (auto ndArrayType = sourceMlirType.dyn_cast<Numpy::NdArrayType>()) {
      auto tensorType = ndArrayType.toTensorType();
      auto callback = [=](Location loc, Value newOpResultValue,
                          PatternRewriter &rewriter) -> Value {
        return rewriter.create<Numpy::CreateArrayFromTensorOp>(
            loc, ndArrayType, newOpResultValue);
      };
      return TypeConversion{tensorType, callback};
    }
  }

  return None;
}

/// Transforms from torch.kernel_call to recognized ops that implement the
/// TorchBuildableKernelOpInterface.
class KernelCallTransformer {
  struct CandidateTransform;

public:
  KernelCallTransformer(MLIRContext &context) : context(context) {}

  template <typename DialectTy> void addDialectOps() {
    Dialect *dialect = context.getOrLoadDialect<DialectTy>();
    // TODO: This is not efficient. We should have a mechanism for dialects to
    // track their own ops and allow a more fine grained mechanism.
    auto allOps = context.getRegisteredOperations();
    for (AbstractOperation *absOp : allOps) {
      if (&absOp->dialect != dialect)
        continue;
      auto *concept = absOp->getInterface<TorchBuildableKernelOpInterface>();
      if (!concept)
        continue;
      const BuildKernelMetadata &buildMetadata =
          concept->getTorchBuildKernelMetadata();
      addBuildableOperation(absOp->name, buildMetadata);
    }
  }

  void addBuildableOperation(Identifier opName,
                             const BuildKernelMetadata &buildMetadata) {
    LLVM_DEBUG(llvm::dbgs()
               << "Register kernel call translation for: " << opName << "\n");
    CandidateTransformList &candidates =
        kernelTransforms[buildMetadata.kernelName];
    candidates.emplace_back(opName, buildMetadata);
  }

  LogicalResult transformKernelCall(KernelCallOp kernelCall,
                                    PatternRewriter &rewriter) const {
    StringRef kernelName = kernelCall.kernelName();
    LLVM_DEBUG(llvm::dbgs()
               << "Evaluate kernel transform '" << kernelName << "':\n");
    auto it = kernelTransforms.find(kernelName);
    if (it == kernelTransforms.end()) {
      LLVM_DEBUG(llvm::dbgs() << "  - No candidate ops for kernel name\n");
      return failure();
    }

    const CandidateTransformList &candidates = it->second;
    for (const CandidateTransform &candidate : candidates) {
      if (succeeded(rewriteForCandidateOp(kernelCall, candidate, rewriter)))
        return success();
    }
    return failure();
  }

  LogicalResult rewriteForCandidateOp(KernelCallOp kernelCall,
                                      const CandidateTransform &candidate,
                                      PatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "  * Evaluate op " << candidate.targetOpName << "\n");
    Torch::KernelMetadata sourceMetadata = kernelCall.getTorchKernelMetadata();
    // Fail on presently unsupported cases.
    if (sourceMetadata.isVararg || candidate.buildMetadata.isVararg) {
      LLVM_DEBUG(llvm::dbgs() << "    - Skip candidate op: vararg kernels "
                                 "presently not supported\n");
      return failure();
    }
    if (sourceMetadata.isVarret || candidate.buildMetadata.isVarret) {
      LLVM_DEBUG(llvm::dbgs() << "    - Skip candidate op: varret kernels "
                                 "presently not supported\n");
      return failure();
    }

    // In none of the special forms do return arity mismatch.
    if (sourceMetadata.returnTypes.size() !=
        candidate.buildMetadata.returnTypes.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    - Skip candidate op: return arity mismatch\n");
      return failure();
    }

    // TODO: Detect trailing outref.
    bool sourceHasTrailingOutRef = false;
    if (sourceHasTrailingOutRef ||
        sourceMetadata.argTypes.size() ==
            candidate.buildMetadata.argTypes.size()) {
      // Arg arity match.
      LLVM_DEBUG(llvm::dbgs() << "    + Candidate matches based on arity\n");

      return rewriteMatchingArity(
          kernelCall, sourceMetadata, candidate,
          /*fixedArgArity=*/candidate.buildMetadata.argTypes.size(),
          /*fixedRetArity=*/candidate.buildMetadata.returnTypes.size(),
          /*sourceHasTrailingOutRef=*/sourceHasTrailingOutRef, rewriter);
    }

    return failure();
  }

  LogicalResult rewriteMatchingArity(KernelCallOp kernelCall,
                                     const Torch::KernelMetadata sourceMetadata,
                                     const CandidateTransform &candidate,
                                     size_t fixedArgArity, size_t fixedRetArity,
                                     bool sourceHasTrailingOutRef,
                                     PatternRewriter &rewriter) const {
    using KVC = KernelValueConversion::BitMask;
    // Pre-conditions.
    assert(!sourceHasTrailingOutRef && "trailing outref not yet implemented");
    if (sourceHasTrailingOutRef)
      assert((sourceMetadata.argTypes.size() ==
              candidate.buildMetadata.argTypes.size() + 1) &&
             "arg arity mismatch for trailing outref conversion");
    else
      assert(sourceMetadata.argTypes.size() ==
                 candidate.buildMetadata.argTypes.size() &&
             "arg arity mismatch");

    // Convert fixed return types.
    struct ConversionInfo {
      Value originalValue;
      TypeConversion conversion;
    };
    SmallVector<Type, 4> resultTypes;
    SmallVector<ConversionInfo, 4> resultConversions;
    for (size_t i = 0; i < fixedRetArity; ++i) {
      StringRef sourceTorchType = sourceMetadata.returnTypes[i];
      StringRef targetTorchType = candidate.buildMetadata.returnTypes[i];
      KVC flag = candidate.buildMetadata.getReturnConversion(i);
      Value sourceValue = kernelCall.getResult(i);
      Type sourceMlirType = kernelCall.getResultTypes()[i];
      auto conversion = convertTorchReturnType(sourceTorchType, targetTorchType,
                                               flag, sourceMlirType);
      if (!conversion) {
        LLVM_DEBUG(llvm::dbgs() << "    - Return type[" << i
                                << "] incompatible: source=" << sourceTorchType
                                << ", target=" << targetTorchType
                                << ", flag=" << flag << "\n");
        return failure();
      }
      resultTypes.push_back(conversion->targetType);
      resultConversions.push_back({sourceValue, std::move(*conversion)});
    }

    // Convert fixed arg types.
    SmallVector<ConversionInfo, 4> operandInfos;
    for (size_t i = 0; i < fixedArgArity; ++i) {
      operandInfos.emplace_back();
      ConversionInfo &info = operandInfos.back();
      info.originalValue = kernelCall.getOperand(i);
      Type sourceMlirType = info.originalValue.getType();
      auto conversion = convertTorchArgType(
          /*sourceTorchType=*/sourceMetadata.argTypes[i],
          /*targetTorchType=*/candidate.buildMetadata.argTypes[i],
          /*flag=*/candidate.buildMetadata.getArgConversion(i),
          /*sourceMlirType=*/sourceMlirType);
      if (!conversion) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    - Arg type[" << i
                   << "] incompatible: source=" << sourceMetadata.argTypes[i]
                   << ", target=" << candidate.buildMetadata.argTypes[i]
                   << ", flag=" << candidate.buildMetadata.getArgConversion(i)
                   << "\n");
        return failure();
      }
      info.conversion = std::move(*conversion);
    }

    // Match criteria are satisfied. The IR can now be rewritten.
    // Materialize conversions for operands.
    SmallVector<Value, 4> operands;
    for (ConversionInfo &info : operandInfos) {
      if (!info.conversion.callback) {
        // Identity conversion.
        operands.push_back(info.originalValue);
      } else {
        // Computed conversion.
        Value newValue = info.conversion.callback(kernelCall.getLoc(),
                                                  info.originalValue, rewriter);
        operands.push_back(newValue);
      }
    }

    // Create the op.
    OperationState state(kernelCall.getLoc(), candidate.targetOpName, operands,
                         resultTypes, {});
    Operation *newOp = rewriter.createOperation(state);

    // Materialize conversions for results.
    for (auto it : llvm::enumerate(resultConversions)) {
      ConversionInfo &info = it.value();
      Value origOpResultValue = info.originalValue;
      Value newOpResultValue = newOp->getOpResult(it.index());
      Value convertedValue = newOpResultValue;
      if (info.conversion.callback) {
        // Conversion required.
        convertedValue = info.conversion.callback(kernelCall.getLoc(),
                                                  newOpResultValue, rewriter);
      }
      origOpResultValue.replaceAllUsesWith(convertedValue);
    }

    // Done.
    rewriter.eraseOp(kernelCall);
    return success();
  }

private:
  struct CandidateTransform {
    CandidateTransform(Identifier targetOpName,
                       const BuildKernelMetadata &buildMetadata)
        : targetOpName(targetOpName), buildMetadata(buildMetadata) {}
    Identifier targetOpName;
    const BuildKernelMetadata &buildMetadata;
  };
  using CandidateTransformList = SmallVector<CandidateTransform, 1>;

  MLIRContext &context;
  // Map of the torch.kernel_call op's kernelName() attribute to a list of
  // candidate transforms describing how to map to a specific, recognized op.
  // Note that a single generic kernel name can map to more than one candidate
  // kernels based on signature. PyTorch has many such overloads that vary
  // by signature. Some are handled transparently (such as special handling
  // for variants with an out= parameter), while others may map differently
  // (i.e. variants that operate on scalars vs tensors, have different arities,
  // etc).
  llvm::StringMap<CandidateTransformList> kernelTransforms;
};

class RecognizeOpPattern : public OpRewritePattern<Torch::KernelCallOp> {
public:
  RecognizeOpPattern(MLIRContext *context,
                     const KernelCallTransformer &callTransformer)
      : OpRewritePattern(context), callTransformer(callTransformer) {}

  LogicalResult matchAndRewrite(Torch::KernelCallOp kernelCall,
                                PatternRewriter &rewriter) const override {
    return callTransformer.transformKernelCall(kernelCall, rewriter);
  }

private:
  const KernelCallTransformer &callTransformer;
};

class ATenRecognizeKernelsPass
    : public ATenRecognizeKernelsBase<ATenRecognizeKernelsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ATenDialect>();
  }

  void runOnOperation() override {
    auto &context = getContext();
    KernelCallTransformer transformer(context);
    transformer.addDialectOps<ATenDialect>();

    OwningRewritePatternList patterns;
    patterns.insert<RecognizeOpPattern>(&context, transformer);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::aten::createRecognizeKernelsPass() {
  return std::make_unique<ATenRecognizeKernelsPass>();
}
