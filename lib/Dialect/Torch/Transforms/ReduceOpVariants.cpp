//===- ReduceOpVariants.cpp --------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "ReifyAbstractInterpCalculationsUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
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
  ConvertHasValueSemanticsOpsToValueTensors(
      MLIRContext *context, const std::optional<SymbolTable> &extraLibrary)
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

    rewriter.startOpModification(op);
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
          rewriter.cancelOpModification(op);
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
            rewriter.cancelOpModification(op);
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
          rewriter.cancelOpModification(op);
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
          rewriter.cancelOpModification(op);
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
    rewriter.finalizeOpModification(op);
    return success();
  }

private:
  std::optional<SymbolTable> extraLibrary;
};
} // namespace

namespace {

class TorchMatchSpecializedBackendOp
    : public OpConversionPattern<Torch::OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  using HandlerFn = LogicalResult (*)(OperatorOp op,
                                      ConversionPatternRewriter &rewriter);

  LogicalResult
  matchAndRewrite(Torch::OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (namedHandlers.contains(op.getNameAttr())) {
      return namedHandlers.lookup(op.getNameAttr()).front()(op, rewriter);
    }

    return failure();
  }

  static void
  populateSpecializedConversions(TorchMatchSpecializedBackendOp &matcher);

  static std::unique_ptr<TorchMatchSpecializedBackendOp>
  getPopulatedMatcher(MLIRContext *context) {
    auto matcher = std::make_unique<TorchMatchSpecializedBackendOp>(context);
    populateSpecializedConversions(*matcher);
    return matcher;
  };

  void populate(StringRef name, HandlerFn fn) {
    namedHandlers[StringAttr::get(getContext(), name)].push_back(fn);
  }

  void populateLegalizedNames(llvm::DenseSet<StringAttr> &set) {
    for (auto handle : namedHandlers) {
      set.insert(handle.first);
    }
  }

private:
  DenseMap<StringAttr, SmallVector<HandlerFn, 1>> namedHandlers;
};

void TorchMatchSpecializedBackendOp::populateSpecializedConversions(
    TorchMatchSpecializedBackendOp &matcher) {
  matcher.populate(
      "torch.aten._scaled_dot_product_flash_attention_for_cpu",
      [](Torch::OperatorOp op,
         ConversionPatternRewriter &rewriter) -> LogicalResult {
        auto uses = op.getResult(1).getUses();
        if (uses.end() == uses.begin()) {
          auto oldOperands = op->getOperands();
          llvm::SmallVector<Value> newOperands{
              oldOperands[0], oldOperands[1], oldOperands[2], oldOperands[5],
              oldOperands[3], oldOperands[4], oldOperands[6]};

          auto newOp = rewriter.create<Torch::AtenScaledDotProductAttentionOp>(
              op.getLoc(), op->getResultTypes()[0], newOperands,
              op->getAttrs());
          rewriter.replaceAllUsesWith(op.getResult(0), newOp.getResult());
          rewriter.eraseOp(op);
          return success();
        }
        return failure();
      });
}

bool isSpecializedOperation(Torch::OperatorOp op) { return true; }
} // namespace

// Reduce Ops without value semantics but the corresponding without trailing
// underscore variant doesn't exist.
namespace {

// int(ceil((end - start) / step))
Value calculateArangeResultNumElements(PatternRewriter &rewriter, Location loc,
                                       Value start, Value end, Value step) {
  Value sub = rewriter.create<AtenSubOp>(
      loc, Torch::NumberType::get(rewriter.getContext()), end, start);
  Value div = rewriter.create<AtenDivOp>(loc, sub, step);
  return rewriter.create<AtenCeilFloatOp>(loc, div);
}

class ReduceNonValueSemanticOps : public RewritePattern {
public:
  ReduceNonValueSemanticOps(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();
    if (isa<AtenBernoulli_FloatOp>(op)) {
      Operation *newOp = rewriter.create<ValsemVariantAtenBernoulliFloatOp>(
          loc, op->getResultTypes(), op->getOperands());
      auto tensor =
          rewriter.create<CopyToValueTensorOp>(loc, newOp->getResult(0));
      createOverwriteTensorContents(rewriter, loc, tensor, op->getOperand(0));
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    } else if (auto arangeOutOp = dyn_cast<AtenArangeStartOutOp>(op)) {
      Value start = arangeOutOp.getStart();
      Value end = arangeOutOp.getEnd();
      Value step = arangeOutOp.getStep();
      Value out = arangeOutOp.getOut();

      // `overwrite.tensor.contents` cannot change the tensor shape,
      // so `out` tensor should have same num_elements with result tensor.
      // It means that we don't support code like:
      //   `x = torch.randn(12)`
      //   `y = torch.arange(13, out=x)`
      Value resultNumElements =
          calculateArangeResultNumElements(rewriter, loc, start, end, step);
      Value outNumElements = rewriter.create<AtenNumelOp>(loc, out);
      Value eqOrNot =
          rewriter.create<AtenEqIntOp>(loc, resultNumElements, outNumElements);
      rewriter.create<RuntimeAssertOp>(
          loc, eqOrNot,
          rewriter.getStringAttr("`out` tensor should have the same "
                                 "num_elements with result tenosr"));

      auto dtype = rewriter.create<PrimDtypeOp>(loc, out);
      auto device = rewriter.create<PrimDeviceOp>(loc, out);
      auto shape = rewriter.create<AtenSizeOp>(
          loc, Torch::ListType::get(Torch::IntType::get(ctx)), out);
      auto none = rewriter.create<ConstantNoneOp>(loc);
      Value newArange = rewriter.create<AtenArangeStartStepOp>(
          loc, arangeOutOp.getResult().getType(), start, end, step, dtype,
          /*layout=*/none, device, /*pin_memory=*/none);
      Value reshape = rewriter.create<AtenReshapeOp>(
          loc, arangeOutOp.getResult().getType(), newArange, shape);

      auto vtensor = rewriter.create<CopyToValueTensorOp>(loc, reshape);
      createOverwriteTensorContents(rewriter, loc, vtensor, out);
      rewriter.replaceOp(arangeOutOp, out);
      return success();
    } else {
      return failure();
    }
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
    assert(fragments.size() >= 3 && fragments[2].ends_with("_") &&
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
    // Note: need to convert result to first input's dtype because mix precision
    // compute would result in different behaviors.
    // For example:
    // a = torch.randn(3, 3).half() # float16
    // b = torch.randn(3, 3) # float32
    // a += b # i.e. torch.ops.aten.add_(a, b), result is float16
    // c = a + b # i.e. torch.ops.aten.add(a, b), result is float32
    Value none = rewriter.create<ConstantNoneOp>(op->getLoc());
    Value cstFalse = rewriter.create<ConstantBoolOp>(op->getLoc(), false);
    auto aDtype = rewriter.create<PrimDtypeOp>(op->getLoc(), op->getOperand(0));
    auto toDtype = rewriter.create<AtenToDtypeOp>(
        op->getLoc(), newOp->getResult(0).getType(), newOp->getResult(0),
        aDtype, /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
        /*memory_format=*/none);
    auto tensor = rewriter.create<CopyToValueTensorOp>(op->getLoc(), toDtype);
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

    // Create specialized matcher:
    auto specialized =
        TorchMatchSpecializedBackendOp::getPopulatedMatcher(context);
    DenseSet<StringAttr> specializedNames;
    specialized->populateLegalizedNames(specializedNames);
    patterns.insert(std::move(specialized));

    ConversionTarget target(*context);
    target.addIllegalOp<NonValueTensorLiteralOp>();
    target.addIllegalOp<AtenBernoulli_FloatOp>();
    target.addIllegalOp<AtenArangeStartOutOp>();
    target.markUnknownOpDynamicallyLegal([&extraLibraryModuleSymTable,
                                          &specializedNames](Operation *op) {
      if (isa<OperatorOp>(op)) {
        if (specializedNames.contains(cast<OperatorOp>(op).getNameAttr())) {
          return false;
        }
      }
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

      if (isa<OperatorOp>(op) && isSpecializedOperation(cast<OperatorOp>(op)))
        return false;
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
