//===- RecognizeKernels.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
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

bool isTorchTensorType(StringRef torchType) {
  return torchType == "Tensor" || torchType == "Tensor?";
}

bool isTorchOptionalType(StringRef torchType) {
  return torchType.endswith("?");
}

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
    // TODO: Support the kPromoteScalar flag.
    if (!isTorchTensorType(sourceTorchType) ||
        !isTorchTensorType(targetTorchType))
      return None;

    // If the target is optional and the type is NoneType, passthrough.
    if (isTorchOptionalType(targetTorchType) &&
        sourceMlirType.isa<Basicpy::NoneType>())
      return TypeConversion{sourceMlirType, nullptr};

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

  if (flag & KVC::kMutableTensor) {
    if (!isTorchTensorType(sourceTorchType) ||
        !isTorchTensorType(targetTorchType))
      return None;
    // If the type is already mutable, passthrough.
    if (sourceMlirType.isa<Numpy::NdArrayType>())
      return TypeConversion{sourceMlirType, nullptr};
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
  if (sourceTorchType == targetTorchType && flag == 0) {
    LLVM_DEBUG(llvm::dbgs() << "      * Return types already match\n");
    return TypeConversion{sourceMlirType, nullptr};
  }

  // Immutable tensor conversion.
  if (flag & KVC::kImmutableTensor) {
    LLVM_DEBUG(llvm::dbgs()
               << "      * Return conversion flag kImmutableTensor\n");
    if (!isTorchTensorType(sourceTorchType) ||
        !isTorchTensorType(targetTorchType)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "      * Source or target not a Tensor type\n");
      return None;
    }

    // Already immutable.
    if (sourceMlirType.isa<TensorType>()) {
      LLVM_DEBUG(llvm::dbgs() << "      * Source is already immutable\n");
      return TypeConversion{sourceMlirType, nullptr};
    }

    // Convert NdArray type.
    if (sourceMlirType.isa<Basicpy::NoneType>() &&
        isTorchOptionalType(targetTorchType)) {
      LLVM_DEBUG(llvm::dbgs() << "      * None Tensor type passthrough\n");
      return TypeConversion{sourceMlirType, nullptr};
    } else if (auto ndArrayType =
                   sourceMlirType.dyn_cast<Numpy::NdArrayType>()) {
      auto tensorType = ndArrayType.toTensorType();
      auto callback = [=](Location loc, Value newOpResultValue,
                          PatternRewriter &rewriter) -> Value {
        return rewriter.create<Numpy::CreateArrayFromTensorOp>(
            loc, ndArrayType, newOpResultValue);
      };
      LLVM_DEBUG(llvm::dbgs() << "      * Convert return type\n");
      return TypeConversion{tensorType, callback};
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "      * Return type is not a supported tensor type\n");
      return None;
    }
  }

  if (flag & KVC::kMutableTensor) {
    if (!isTorchTensorType(sourceTorchType) ||
        !isTorchTensorType(targetTorchType)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "      * Source or target not a Tensor type\n");
      return None;
    }
    // If the type is already mutable, passthrough.
    if (sourceMlirType.isa<Numpy::NdArrayType>()) {
      LLVM_DEBUG(llvm::dbgs() << "      * Source is already mutable\n");
      return TypeConversion{sourceMlirType, nullptr};
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "      * Return type conversion fallthrough\n");
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
    {
      CandidateTransformList &candidates =
          kernelTransforms[buildMetadata.kernelName];
      candidates.emplace_back(opName, buildMetadata);
    }

    for (StringRef aliasKernelName : buildMetadata.aliasKernelNames) {
      CandidateTransformList &candidates = kernelTransforms[aliasKernelName];
      candidates.emplace_back(opName, buildMetadata);
    }

    if (buildMetadata.inplaceVariantKernelName) {
      CandidateTransformList &candidates =
          kernelTransforms[*buildMetadata.inplaceVariantKernelName];
      candidates.emplace_back(opName, buildMetadata);
    }
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

    bool sourceHasTrailingOutRef =
        candidate.buildMetadata.promoteTrailingOutTensor &&
        sourceMetadata.argTypes.size() ==
            candidate.buildMetadata.argTypes.size() + 1;
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
    if (sourceHasTrailingOutRef)
      assert((sourceMetadata.argTypes.size() ==
              candidate.buildMetadata.argTypes.size() + 1) &&
             "arg arity mismatch for trailing outref conversion");
    else
      assert(sourceMetadata.argTypes.size() ==
                 candidate.buildMetadata.argTypes.size() &&
             "arg arity mismatch");
    bool isInplaceVariant =
        candidate.buildMetadata.inplaceVariantKernelName &&
        kernelCall.kernelName() ==
            *candidate.buildMetadata.inplaceVariantKernelName;

    // Convert fixed return types.
    using PostConversionCallback = std::function<void()>;
    SmallVector<PostConversionCallback, 4> postConversionCallbacks;
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
      if (flag & KVC::kDropReturnAndAliasArg0) {
        // Reduce result arity and alias any uses to arg0.
        if (kernelCall.args().empty()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    - Cannot alias arg0 (no arguments)\n");
          return failure();
        }
        Value arg0 = kernelCall.args()[0];
        postConversionCallbacks.push_back(
            [sourceValue, arg0]() { sourceValue.replaceAllUsesWith(arg0); });
      } else {
        // General, arity-preserving type conversion.
        auto conversion = convertTorchReturnType(
            sourceTorchType, targetTorchType, flag, sourceMlirType);
        if (!conversion) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    - Return type[" << i << "] incompatible: source="
                     << sourceTorchType << ", target=" << targetTorchType
                     << ", flag=" << flag << "\n");
          return failure();
        }
        resultTypes.push_back(conversion->targetType);
        resultConversions.push_back({sourceValue, std::move(*conversion)});
      }
    }

    // Convert fixed arg types.
    SmallVector<ConversionInfo, 4> operandInfos;
    for (size_t i = 0, operandIndex = 0; i < fixedArgArity; ++i) {
      // Drop this arg?
      if (candidate.buildMetadata.argConversions[i] & KVC::kDrop)
        continue;
      if (kernelCall.getNumOperands() <= operandIndex) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    - Arg operand " << i
                   << " does not exist in kernel call (missing default?)\n");
        return failure();
      }

      // Normal type conversion of the operand.
      operandInfos.emplace_back();
      ConversionInfo &info = operandInfos.back();
      info.originalValue = kernelCall.getOperand(operandIndex++);
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
    // For out params, we need to save off the converted first result -- we will
    // just RAUW it with the out param later.
    Value firstResultConverted;
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
      if (it.index() == 0)
        firstResultConverted = convertedValue;
    }

    // Post conversion callbacks.
    for (auto &callback : postConversionCallbacks)
      callback();

    if (sourceHasTrailingOutRef || isInplaceVariant) {
      assert(newOp->getNumResults() > 0 &&
             newOp->getResultTypes()[0].isa<TensorType>() &&
             "must have tensor first result");
      LLVM_DEBUG(llvm::dbgs()
                 << "    - Ovewriting out param with result tensor.\n");
      Value out;
      if (sourceHasTrailingOutRef)
        out = kernelCall.getOperand(fixedArgArity);
      else // isInplaceVariant
        out = kernelCall.getOperand(0);
      rewriter.create<Numpy::OverwriteArrayOp>(kernelCall.getLoc(),
                                               newOp->getResult(0), out);
      assert(firstResultConverted && "must have a first result");
      firstResultConverted.replaceAllUsesWith(out);
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
    MLIRContext *context = &getContext();
    KernelCallTransformer transformer(*context);
    transformer.addDialectOps<ATenDialect>();

    RewritePatternSet patterns(context);
    patterns.add<RecognizeOpPattern>(context, transformer);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::aten::createRecognizeKernelsPass() {
  return std::make_unique<ATenRecognizeKernelsPass>();
}
