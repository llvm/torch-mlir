//===- LowerToBackendContract.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torch-lower-to-backend-contract"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// Checking the backend contract.
//===----------------------------------------------------------------------===//

static void markDecomposedOpsAsIllegal(MLIRContext *context,
                                       ConversionTarget &target,
                                       ArrayRef<std::string> backendLegalOps);

static LogicalResult checkType(Operation *op, Type type,
                               bool actuallyEmitDiagnostics) {
  // Allow various scalar types that backends are expected to be able to handle.
  if (type.isa<Torch::IntType, Torch::FloatType, Torch::BoolType,
               Torch::DeviceType>())
    return success();

  // Backends are not expected to support dynamic computations on these types,
  // but they frequently appear as parameters to ops which backends
  // can statically pattern match and eliminate from the program.
  // For example, a tensor operand might be optional, and the backend
  // will pattern-match statically whether it is passed as a tensor or None.
  if (type.isa<Torch::NoneType, Torch::StringType>())
    return success();

  // We blanket prohibit non-value-semantic tensors.
  // All of our backends are currently based on value-semantic tensors, so
  // we consider it our responsibility to lower all non-value-semantic tensors
  // to value-semantic tensors.
  if (type.isa<NonValueTensorType>()) {
    if (actuallyEmitDiagnostics) {
      return op
          ->emitError("unsupported by backend contract: non-value tensor type")
          .attachNote()
          .append("this is likely due to a missing case in the "
                  "MaximizeValueSemantics pass");
    } else {
      return failure();
    }
  }

  // For value-semantic tensors, we require at least a known rank and dtype.
  // We are not aware of a situation where our backends can handle an unranked
  // tensor type or a tensor with a dynamic dtype.
  //
  // There are somewhat fundamental reasons for this. In particular, the problem
  // of unranked codegen is completely different from the problem of ranked
  // codegen (since ranked corresponds to a fixed loop nest structure). For all
  // codegen systems we are aware of, the program must be reduced to operate
  // on ranked tensors at some point in compilation, and we are not aware of
  // any backend with a general solution to this problem before it reaches
  // codegen. So we consider it our responsibility to eliminate unranked tensor
  // from the program.
  //
  // We aren't aware of any backend with any infrastructure to represent dynamic
  // dtypes, let alone transform and optimize them. Additionally, it is unlikely
  // that any backend, even if it supports dynamic dtypes in some form, will
  // have an sufficiently rich system for representing PyTorch type promotion
  // rules. So we consider it our responsibility to ensure that all dtypes are
  // statically known.
  if (auto tensorType = type.dyn_cast<ValueTensorType>()) {
    if (!tensorType.hasSizes()) {
      if (actuallyEmitDiagnostics) {
        return op
            ->emitError(
                "unsupported by backend contract: tensor with unknown rank")
            .attachNote()
            .append("this is likely due to a missing transfer function "
                    "in abstract_interp_lib_gen.py");
      } else {
        return failure();
      }
    }
    if (!tensorType.hasDtype()) {
      if (actuallyEmitDiagnostics) {
        return op
            ->emitError(
                "unsupported by backend contract: tensor with unknown dtype")
            .attachNote()
            .append("this is likely due to a missing case in RefineTypes");
      } else {
        return failure();
      }
    }
    return success();
  }

  // Optional types are also in the category of types which we don't expect
  // backends to dynamically compute with, but they can be pattern matched
  // in many cases that are practically necessary.
  if (auto optionalType = type.dyn_cast<OptionalType>()) {
    // TODO: Be stricter about tensor types.
    // See comment below for ListType.
    if (optionalType.getContainedType().isa<ValueTensorType>())
      return success();
    return checkType(op, optionalType.getContainedType(),
                     actuallyEmitDiagnostics);
  }
  // List types are also in the category of types which we don't expect
  // backends to dynamically compute with, but they can be pattern matched
  // in many cases that are practically necessary. For example, the
  // strides of a convolution op are represented as a list.
  if (auto listType = type.dyn_cast<ListType>()) {
    // TODO: Be stricter about tensor types.
    // For the moment, there are cases (such as for torch.cat) where we end
    // up with `!torch.list<vtensor>` which doesn't have shape or dtype in
    // the contained type information. Somehow this slips through and works.
    // We should be stricter about this and properly infer the contained type
    // and shape.
    if (listType.getContainedType().isa<ValueTensorType>())
      return success();
    return checkType(op, listType.getContainedType(), actuallyEmitDiagnostics);
  }
  // Tuple types are also in the category of types which we don't expect
  // backends to dynamically compute with, but they can be pattern matched
  // in many cases that are practically necessary.
  if (auto tupleType = type.dyn_cast<Torch::TupleType>()) {
    for (auto containedType : tupleType.getContainedTypes()) {
      if (failed(checkType(op, containedType, actuallyEmitDiagnostics)))
        return failure();
    }
    return success();
  }

  // Unsupported type.
  if (actuallyEmitDiagnostics) {
    return op->emitError("unsupported by backend contract: type ") << type;
  } else {
    return failure();
  }
}

static LogicalResult checkOpIsBackendLegal(Operation *op,
                                           const ConversionTarget &target,
                                           bool actuallyEmitDiagnostics) {
  if (target.isLegal(op))
    return success();

  if (actuallyEmitDiagnostics) {
    return op->emitError("found an op that was marked as backend illegal")
        .attachNote()
        .append("this is likely due to DecomposeComplexOps being unable to "
                "decompose this op");
  } else {
    return failure();
  }
}

static bool satisfiesBackendContract(ModuleOp module,
                                     const ConversionTarget &target,
                                     bool actuallyEmitDiagnostics = false) {
  // We do not permit `torch.global_slot`'s in the backend contract, since
  // support for them is not widespread, and this does not align with PyTorch's
  // more tracing-based direction.
  //
  // We just check for the GlobalSlotModuleInitializerOp since its verifier
  // ensures that the set of global slots matches those initialized by the
  // module initializer.
  auto walkResult0 = module.walk([&](Torch::GlobalSlotModuleInitializerOp op) {
    if (actuallyEmitDiagnostics) {
      // Report the error on the terminator to avoid dumping the whole
      // initializer itself, which can have pages of ops in it.
      op.getBody()
          ->getTerminator()
          ->emitError("unsupported by backend contract: module initializers")
          .attachNote()
          .append("this is likely due to InlineGlobalSlots being unable to "
                  "inline a global slot");
    }
    return WalkResult::interrupt();
  });
  if (walkResult0.wasInterrupted())
    return false;

  // Check all the types of all Value's in the program and the legality of all
  // the ops.
  //
  // A pre-order walk gives a more intuitive "first error".
  // TODO: Should we report more than the first error?
  // How do we avoid making it too spammy?
  auto walkResult1 = module.walk<WalkOrder::PreOrder>([&](Block *block) {
    for (BlockArgument arg : block->getArguments())
      if (failed(checkType(block->getParentOp(), arg.getType(),
                           actuallyEmitDiagnostics))) {
        return WalkResult::interrupt();
      }
    for (Operation &op : *block) {
      if (failed(checkOpIsBackendLegal(&op, target, actuallyEmitDiagnostics)))
        return WalkResult::interrupt();

      for (OpResult result : op.getResults())
        if (failed(checkType(&op, result.getType(), actuallyEmitDiagnostics)))
          return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if (walkResult1.wasInterrupted())
    return false;
  return true;
}

// Explicitly set ops and dialects allowed and not allowed in backend contract.
static ConversionTarget
getBackendContractTarget(MLIRContext *context, bool decompose,
                         ArrayRef<std::string> backendLegalOps) {
  ConversionTarget target(*context);
  target.addLegalDialect<func::FuncDialect, Torch::TorchDialect>();
  if (decompose)
    markDecomposedOpsAsIllegal(context, target, backendLegalOps);
  return target;
}

namespace {
class LowerToBackendContractPass
    : public LowerToBackendContractBase<LowerToBackendContractPass> {
public:
  LowerToBackendContractPass() = default;
  LowerToBackendContractPass(int maxIterations, bool decompose,
                             ArrayRef<std::string> backendLegalOps) {
    this->maxIterations = maxIterations;
    this->decompose = decompose;
    this->backendLegalOps = backendLegalOps;
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    ConversionTarget target =
        getBackendContractTarget(context, decompose, backendLegalOps);

    OpPassManager pm(module.getOperationName());
    TorchLoweringPipelineOptions options;
    options.decompose = decompose;
    options.backendLegalOps = backendLegalOps;
    createTorchSimplificationPipeline(pm, options);

    int i = 0;
    do {
      if (i++ == maxIterations) {
        LLVM_DEBUG({
          llvm::dbgs() << "LowerToBackendContractPass: "
                       << "failed to satisfy backend contract after "
                       << maxIterations
                       << " iterations of the simplification pipeline\n";
        });
        // Show the diagnostics.
        (void)satisfiesBackendContract(module, target,
                                       /*actuallyEmitDiagnostics=*/true);
        return signalPassFailure();
      }

      if (failed(runPipeline(pm, module)))
        return signalPassFailure();
    } while (!satisfiesBackendContract(module, target));
    LLVM_DEBUG({
      llvm::dbgs() << "LowerToBackendContractPass: "
                   << "succeeded after " << i
                   << " iterations of the simplification pipeline\n";
    });
  }
};

class VerifyBackendContractPass
    : public VerifyBackendContractBase<VerifyBackendContractPass> {
public:
  VerifyBackendContractPass() = default;
  VerifyBackendContractPass(bool decompose,
                            ArrayRef<std::string> backendLegalOps) {
    this->decompose = decompose;
    this->backendLegalOps = backendLegalOps;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target =
        getBackendContractTarget(context, decompose, backendLegalOps);

    if (!satisfiesBackendContract(getOperation(), target,
                                  /*actuallyEmitDiagnostics=*/true)) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createLowerToBackendContractPass(
    int maxIterations, bool decompose, ArrayRef<std::string> backendLegalOps) {
  return std::make_unique<LowerToBackendContractPass>(maxIterations, decompose,
                                                      backendLegalOps);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createVerifyBackendContractPass(
    bool decompose, ArrayRef<std::string> backendLegalOps) {
  return std::make_unique<VerifyBackendContractPass>(decompose,
                                                     backendLegalOps);
}

// The backend contract guarantees that ops with decompositions available will
// be decomposed. The only way to have an op reach the backend contract without
// getting decomposed is by having the user explicitly specify that op in the
// `backendLegalOps` argument to the `LowerToBackendContractPass`. Therefore,
// here we mark as illegal all ops with decompositions except for those in
// `backendLegalOps`.
//
// The legality check takes place here instead of in the `DecomposeComplexOps`
// pass for two reasons:
// 1. Makes sure the `DecomposeComplexOps` pass always succeeds, allowing it to
//   run multiple times. This is needed for graphs where static information such
//   as dtypes and shapes takes multiple iterations to propagate through the
//   entire graph. `DecomposeComplexOps` pass failing would cause the entire
//   `LowerToBackendContractPass` to fail
// 2. Makes the legality requirements in the backend contract for ops with
//   decompositions explicit in this file
static void markDecomposedOpsAsIllegal(MLIRContext *context,
                                       ConversionTarget &target,
                                       ArrayRef<std::string> backendLegalOps) {
  target.addIllegalOp<AtenSoftmaxIntOp>();
  target.addIllegalOp<Aten_SoftmaxOp>();
  target.addIllegalOp<Aten_LogSoftmaxOp>();
  target.addIllegalOp<AtenLogSoftmaxIntOp>();
  target.addIllegalOp<AtenEmptyLikeOp>();
  target.addIllegalOp<AtenOnesLikeOp>();
  target.addIllegalOp<AtenZerosLikeOp>();
  target.addIllegalOp<AtenRollOp>();
  target.addIllegalOp<AtenRepeatOp>();
  target.addIllegalOp<AtenExpandOp>();
  target.addIllegalOp<AtenFlattenUsingIntsOp>();
  target.addIllegalOp<AtenWhereScalarOp>();
  target.addIllegalOp<AtenWhereScalarOtherOp>();
  target.addIllegalOp<AtenWhereScalarSelfOp>();
  target.addIllegalOp<AtenConvolutionBackwardOverrideableOp>();
  target.addIllegalOp<AtenSizeOp>();
  target.addIllegalOp<AtenReshapeOp>();
  target.addIllegalOp<Aten_SoftmaxBackwardDataOp>();
  target.addIllegalOp<AtenTanhBackwardOp>();
  target.addIllegalOp<AtenAddmmOp>();
  target.addIllegalOp<AtenMeanOp>();
  target.addIllegalOp<AtenMeanDimOp>();
  target.addIllegalOp<AtenSelectIntOp>();
  target.addIllegalOp<AtenMvOp>();
  target.addIllegalOp<AtenTOp>();
  target.addIllegalOp<Aten_LogSoftmaxBackwardDataOp>();
  target.addDynamicallyLegalOp<AtenMatmulOp>([](AtenMatmulOp op) {
    std::optional<unsigned> lhsRank = getTensorRank(op.getSelf());
    std::optional<unsigned> rhsRank = getTensorRank(op.getOther());
    if (!lhsRank || !rhsRank)
      return false;
    // Make aten.matmul legal if the following condition is satisfied.
    return (*lhsRank != 2 || *rhsRank != 2) && (*lhsRank != 3 || *rhsRank != 3);
  });
  target.addIllegalOp<AtenAddcmulOp>();
  target.addIllegalOp<AtenAddcdivOp>();
  target.addIllegalOp<AtenLayerNormOp>();
  target.addIllegalOp<AtenNativeLayerNormOp>();
  target.addIllegalOp<AtenNativeBatchNormOp>();
  target.addIllegalOp<AtenConvolutionOverrideableOp>();
  target.addIllegalOp<Aten_ConvolutionOp, Aten_ConvolutionDeprecatedOp>();
  target.addIllegalOp<AtenConvolutionBackwardOp>();
  target.addIllegalOp<AtenConv2dOp>();
  target.addIllegalOp<AtenConvTranspose2dInputOp>();
  target.addIllegalOp<AtenArangeOp>();
  target.addIllegalOp<AtenArangeStartOp>();
  target.addIllegalOp<AtenArgmaxOp>();
  target.addIllegalOp<AtenSquareOp>();
  target.addIllegalOp<AtenVarOp>();
  target.addIllegalOp<AtenStdOp>();
  target.addIllegalOp<Aten_UnsafeViewOp>();
  target.addIllegalOp<Aten_ReshapeAliasOp>();
  target.addIllegalOp<AtenBernoulliOp>();
  target.addIllegalOp<ValsemVariantAtenBernoulliFloatOp>();
  target.addIllegalOp<AtenBernoulliTensorOp>();
  target.addIllegalOp<AtenZeroOp>();
  target.addIllegalOp<AtenRandLikeOp>();
  target.addIllegalOp<AtenHardsigmoidOp>();
  target.addIllegalOp<AtenRelu6Op>();
  target.addIllegalOp<AtenHardswishOp>();
  target.addIllegalOp<AtenSoftplusOp>();
  target.addIllegalOp<AtenSiluOp>();
  target.addIllegalOp<AtenNewZerosOp>();
  target.addIllegalOp<AtenNewOnesOp>();
  target.addIllegalOp<AtenHardtanhOp>();
  target.addIllegalOp<AtenFullOp>();
  target.addIllegalOp<AtenLinearOp>();
  target.addIllegalOp<AtenMishOp>();
  target.addIllegalOp<AtenFullLikeOp>();
  target.addIllegalOp<AtenIndexPutOp>();
  target.addIllegalOp<AtenExpandAsOp>();
  target.addIllegalOp<Aten_ToCopyOp>();
  target.addIllegalOp<AtenDropoutOp>();
  target.addIllegalOp<AtenNewEmptyOp>();
  target.addIllegalOp<AtenIndexPutHackedTwinOp>();
  target.addIllegalOp<AtenPadOp>();
  target.addIllegalOp<AtenToDtypeLayoutOp>();
  target.addIllegalOp<AtenToDeviceOp>();
  target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
  target.addIllegalOp<AtenClampMinOp>();
  target.addIllegalOp<AtenClampMaxOp>();
  target.addIllegalOp<AtenBaddbmmOp>();
  target.addIllegalOp<AtenFloorDivideOp>();
  target.addIllegalOp<AtenNumpyTOp>();
  target.addIllegalOp<AtenSelectScatterOp>();
  target.addIllegalOp<AtenVarDimOp>();
  target.addIllegalOp<AtenAmaxOp>();
  target.addIllegalOp<AtenVarCorrectionOp>();
  target.addIllegalOp<AtenStdDimOp>();
  target.addIllegalOp<AtenStdCorrectionOp>();
  target.addIllegalOp<AtenNarrowOp>();
  target.addIllegalOp<Aten_EmbeddingBagOp>();
  target.addIllegalOp<AtenLiftFreshCopyOp>();
  target.addIllegalOp<AtenIndexTensorHackedTwinOp>();
  target.addIllegalOp<AtenMseLossOp>();
  target.addIllegalOp<AtenRandintLowOp>();
  target.addIllegalOp<AtenVarMeanCorrectionOp>();
  target.addIllegalOp<PrimsConvertElementTypeOp>();
  target.addIllegalOp<PrimsVarOp>();
  target.addIllegalOp<PrimsSqrtOp>();
  target.addIllegalOp<AtenRandnOp>();
  target.addIllegalOp<AtenRandnGeneratorOp>();
  target.addIllegalOp<AtenVarMeanOp>();
  for (std::string opName : backendLegalOps) {
    target.addLegalOp(OperationName(opName, context));
  }
}
