//===- ReduceOpVariants.cpp --------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "ReifyAbstractInterpCalculationsUtils.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Create an overwrite in a manner that preserves the
// `OverwriteTensorContentsOp` invariant that both arguments
// must have the same shape and dtype.
static void createOverwriteTensorContents(PatternRewriter &rewriter,
                                          Location loc, Value overwriterTensor,
                                          Value overwrittenTensor) {
  Type overwriterTensorType = overwriterTensor.getType();
  Type overwrittenTensorType = overwrittenTensor.getType()
                                   .dyn_cast<NonValueTensorType>()
                                   .getWithValueSemantics();
  if (overwriterTensorType != overwrittenTensorType) {
    overwriterTensor = rewriter.create<TensorStaticInfoCastOp>(
        loc, overwrittenTensorType, overwriterTensor);
  }
  rewriter.create<OverwriteTensorContentsOp>(loc, overwriterTensor,
                                             overwrittenTensor);
}

static Type getContainerOrTensorTypeWithValueSemantics(Type type) {
  if (auto optionalType = type.dyn_cast<OptionalType>()) {
    Type newContainedType = getContainerOrTensorTypeWithValueSemantics(
        optionalType.getContainedType());
    return OptionalType::get(newContainedType);
  } else if (auto listType = type.dyn_cast<ListType>()) {
    Type newContainedType =
        getContainerOrTensorTypeWithValueSemantics(listType.getContainedType());
    return ListType::get(newContainedType);
  } else if (auto tensorType = type.dyn_cast<NonValueTensorType>()) {
    return tensorType.getWithValueSemantics();
  } else {
    return nullptr;
  }
}

static bool
operatorOpHasValueSemantics(OperatorOp opOp,
                            std::optional<SymbolTable> extraLibrary) {
  if (!extraLibrary.has_value())
    return false;
  auto opName = opOp->getAttr("name").cast<StringAttr>().getValue();
  std::string libFuncName = (mlir::torch::Torch::getLibraryFunctionPrefix(
                                 LibraryFunctionKind::HasValueSemantics) +
                             Twine(opName))
                                .str();
  auto libFunc = extraLibrary->lookup<func::FuncOp>(libFuncName);
  return bool(libFunc);
}

namespace {
// Convert value semantic ops operating on mutable arrays to instead operate on
// immutable tensors.
class ConvertHasValueSemanticsOpsToValueTensors : public RewritePattern {
public:
  ConvertHasValueSemanticsOpsToValueTensors(MLIRContext *context,
                                            const std::optional<SymbolTable>& extraLibrary)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {
    this->extraLibrary = extraLibrary;
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<OperatorOp>(op)) {
      if (!operatorOpHasValueSemantics(cast<OperatorOp>(op), extraLibrary)) {
        return rewriter.notifyMatchFailure(op, "does not have value semantics");
      }
    } else if (!op->hasTrait<Torch::OpTrait::HasValueSemantics>()) {
      return rewriter.notifyMatchFailure(op, "does not have value semantics");
    }

    rewriter.startRootUpdate(op);
    // Convert all operands.
    SmallVector<Value> newOperands;
    for (OpOperand &opOperand : op->getOpOperands()) {
      Type operandType = opOperand.get().getType();
      if (operandType.isa<NonValueTensorType>()) {
        opOperand.set(rewriter.create<CopyToValueTensorOp>(op->getLoc(),
                                                           opOperand.get()));
      } else if (auto listType = operandType.dyn_cast<ListType>()) {
        if (!(listType.getContainedType().isa<NonValueTensorType>() ||
              listType.getContainedType().isa<OptionalType>()))
          continue;

        // Construct a new list whose elements are value tensors copied from
        // the non-value tensors of the original list.
        auto listConstruct =
            opOperand.get().getDefiningOp<PrimListConstructOp>();
        if (!listConstruct) {
          rewriter.cancelRootUpdate(op);
          return rewriter.notifyMatchFailure(
              op, "unimplemented: list of non vtensor type not constructed "
                  "from list construct");
        }

        if (listConstruct.getElements().empty())
          continue;

        // TODO: Handle optional type in list type.
        if (auto optionalType =
                listType.getContainedType().dyn_cast<OptionalType>()) {
          if (!llvm::all_of(listConstruct.getElements(), [](Value val) {
                return val.getType().isa<NonValueTensorType, Torch::NoneType>();
              })) {
            rewriter.cancelRootUpdate(op);
            return rewriter.notifyMatchFailure(
                op, "unimplemented: list containing optional type is not "
                    "handled.");
          }
        }

        auto newListElements = llvm::to_vector(llvm::map_range(
            listConstruct.getElements(), [&](Value tensor) -> Value {
              if (tensor.getType().isa<NonValueTensorType>()) {
                return rewriter.create<CopyToValueTensorOp>(op->getLoc(),
                                                            tensor);
              }
              return tensor;
            }));

        Type newListType = getContainerOrTensorTypeWithValueSemantics(listType);
        if (!newListType) {
          rewriter.cancelRootUpdate(op);
          return rewriter.notifyMatchFailure(
              op, "Unable to convert list type to value semantics.");
        }
        opOperand.set(rewriter.create<PrimListConstructOp>(
            op->getLoc(), newListType, newListElements));
      } else if (auto optionalType = operandType.dyn_cast<OptionalType>()) {
        // TODO: A more general way to handle the optional type is to
        // introduce a `copy.to_optional_vtensor` op.
        if (!optionalType.getContainedType().isa<NonValueTensorType>())
          continue;

        // Create a new optional value whose input is a value tensor copied
        // from the non value tensor of the original optional value.
        auto derefine = opOperand.get().getDefiningOp<DerefineOp>();
        if (!derefine) {
          rewriter.cancelRootUpdate(op);
          return rewriter.notifyMatchFailure(
              op, "unimplemented: optional of non vtensor type not from "
                  "derefine");
        }

        if (!derefine.getOperand().getType().isa<NonValueTensorType>())
          continue;
        auto newOperand = rewriter.create<CopyToValueTensorOp>(
            op->getLoc(), derefine.getOperand());
        opOperand.set(rewriter.create<DerefineOp>(
            op->getLoc(), Torch::OptionalType::get(newOperand.getType()),
            newOperand));
      }
    }
    // Convert all results.
    rewriter.setInsertionPointAfter(op);
    for (Value result : op->getResults()) {
      auto tensorType = result.getType().dyn_cast<NonValueTensorType>();
      if (!tensorType)
        continue;
      result.setType(tensorType.getWithValueSemantics());
      auto nonValueTensor =
          rewriter.create<CopyToNonValueTensorOp>(op->getLoc(), result);
      result.replaceAllUsesExcept(nonValueTensor, nonValueTensor);
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }
private:
  std::optional<SymbolTable> extraLibrary;
};
} // namespace

// Reduce Ops without value semantics but the corresponding without trailing
// underscore variant doesn't exist.
namespace {
class ReduceNonValueSemanticOps : public RewritePattern {
public:
  ReduceNonValueSemanticOps(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Operation *newOp;
    if (isa<AtenBernoulli_FloatOp>(op)) {
      newOp = rewriter.create<ValsemVariantAtenBernoulliFloatOp>(
          loc, op->getResultTypes(), op->getOperands());
    } else {
      return failure();
    }

    auto tensor =
        rewriter.create<CopyToValueTensorOp>(loc, newOp->getResult(0));
    createOverwriteTensorContents(rewriter, loc, tensor, op->getOperand(0));
    rewriter.replaceOp(op, op->getOperand(0));
    return success();
  }
};
} // namespace

namespace {
// Reduce the "trailing underscore inplace variant" to the value semantic
// variant + an overwrite of the original "self" argument.
class ReduceTrailingUnderscoreInplaceVariant : public RewritePattern {
public:
  ReduceTrailingUnderscoreInplaceVariant(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<Torch::OpTrait::IsTrailingUnderscoreInplaceVariant>())
      return rewriter.notifyMatchFailure(op, "is not trailing_ variant");

    SmallVector<StringRef> fragments;
    llvm::SplitString(op->getName().getStringRef(), fragments, ".");
    assert(fragments.size() >= 3 && fragments[2].endswith("_") &&
           "IsTrailingUnderscoreInplaceVariant incorrectly applied");
    fragments[2] = fragments[2].drop_back();
    std::string noUnderscoreName = llvm::join(fragments, ".");

    OperationState state(op->getLoc(), noUnderscoreName);
    state.addTypes(op->getResultTypes());
    state.addOperands(op->getOperands());
    state.addAttributes(op->getAttrDictionary().getValue());
    // Note: No successors or regions. Torch JIT operators don't have any.
    assert(op->getNumRegions() == 0 && op->getNumSuccessors() == 0 &&
           "Torch JIT operators shouldn't have regions or successors");

    Operation *newOp = rewriter.create(state);
    auto tensor =
        rewriter.create<CopyToValueTensorOp>(op->getLoc(), newOp->getResult(0));
    createOverwriteTensorContents(rewriter, op->getLoc(), tensor,
                                  op->getOperand(0));
    rewriter.replaceOp(op, op->getOperand(0));

    return success();
  }
};
} // namespace

static LogicalResult
reduceNonValueTensorLiteralOpToValueTensorLiteralOp(NonValueTensorLiteralOp op,
                                                    PatternRewriter &rewriter) {
  Value valueTensor =
      rewriter.create<ValueTensorLiteralOp>(op->getLoc(), op.getValue());
  Value tensor =
      copyTensorToType(rewriter, op->getLoc(), op.getType(), valueTensor);
  rewriter.replaceOp(op, {tensor});
  return success();
}

namespace {
struct ReduceOpVariantsPass
    : public ReduceOpVariantsBase<ReduceOpVariantsPass> {
  ReduceOpVariantsPass() = default;
  ReduceOpVariantsPass(StringRef extraLibrary) {
    this->extraLibrary = extraLibrary.str();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    OwningOpRef<ModuleOp> extraLibraryModule =
        ModuleOp::create(UnknownLoc::get(context));
    std::optional<SymbolTable> extraLibraryModuleSymTable = std::nullopt;
    if (!extraLibrary.empty()) {
      if (failed(loadExtraLibrary(extraLibrary, extraLibraryModule))) {
        emitError(getOperation()->getLoc(),
                  "Failed to load extra-library file at " + extraLibrary);
        return signalPassFailure();
      }

      extraLibraryModuleSymTable =
          SymbolTable(extraLibraryModule->getOperation());
    }
    patterns.add<ConvertHasValueSemanticsOpsToValueTensors>(
        context, extraLibraryModuleSymTable);
    patterns.add<ReduceTrailingUnderscoreInplaceVariant>(context);
    patterns.add(reduceNonValueTensorLiteralOpToValueTensorLiteralOp);
    patterns.add<ReduceNonValueSemanticOps>(context);

    ConversionTarget target(*context);
    target.addIllegalOp<NonValueTensorLiteralOp>();
    target.addIllegalOp<AtenBernoulli_FloatOp>();
    target.markUnknownOpDynamicallyLegal([&extraLibraryModuleSymTable](
                                             Operation *op) {
      if (op->hasTrait<Torch::OpTrait::HasValueSemantics>() ||
          (isa<OperatorOp>(op) &&
           operatorOpHasValueSemantics(cast<OperatorOp>(op),
                                       extraLibraryModuleSymTable))) {
        auto hasValueSemantics = [](Type t) {
          // TODO: Make this an allowlist based on a closed torch dialect
          // type system.
          if (auto tensorType = t.dyn_cast<NonValueTensorType>()) {
            return false;
          }
          return true;
        };
        return llvm::all_of(op->getOperandTypes(), hasValueSemantics) &&
               llvm::all_of(op->getResultTypes(), hasValueSemantics);
      }
      if (op->hasTrait<Torch::OpTrait::IsTrailingUnderscoreInplaceVariant>()) {
        return false;
      }
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createReduceOpVariantsPass(StringRef extraLibrary) {
  return std::make_unique<ReduceOpVariantsPass>(extraLibrary);
}
