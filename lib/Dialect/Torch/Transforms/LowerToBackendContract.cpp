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
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torch-lower-to-backend-contract"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// Checking the backend contract.
//===----------------------------------------------------------------------===//

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
            .append("this is likely due to a missing shape transfer function "
                    "in shape_lib_gen.py");
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

static bool satisfiesBackendContract(ModuleOp module,
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

  // Check all the type of all Value's in the program.
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
    for (Operation &op : *block)
      for (OpResult result : op.getResults())
        if (failed(checkType(&op, result.getType(), actuallyEmitDiagnostics)))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (walkResult1.wasInterrupted())
    return false;
  return true;
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
        (void)satisfiesBackendContract(module,
                                       /*actuallyEmitDiagnostics=*/true);
        return signalPassFailure();
      }

      if (failed(runPipeline(pm, module)))
        return signalPassFailure();
    } while (!satisfiesBackendContract(module));
    LLVM_DEBUG({
      llvm::dbgs() << "LowerToBackendContractPass: "
                   << "succeeded after " << i
                   << " iterations of the simplification pipeline\n";
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createLowerToBackendContractPass(
    int maxIterations, bool decompose, ArrayRef<std::string> backendLegalOps) {
  return std::make_unique<LowerToBackendContractPass>(maxIterations, decompose,
                                                      backendLegalOps);
}
