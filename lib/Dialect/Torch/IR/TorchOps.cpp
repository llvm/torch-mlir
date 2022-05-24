//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// see https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h#L28
static int64_t getDtypeIntegerFromMlirType(Type dtype) {
  if (dtype.isa<Float32Type>())
    return 6;

  if (auto integerType = dtype.dyn_cast<IntegerType>()) {
    if (integerType.isSignedInteger(64))
      return 4;
    if (integerType.isSignlessInteger(1))
      return 11;
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

Value mlir::torch::Torch::copyTensorToType(OpBuilder &builder, Location loc,
                                           BaseTensorType newType,
                                           Value tensor) {
  auto originalType = tensor.getType().cast<BaseTensorType>();
  // Adjust the static information in the type to match between the original and
  // new types.
  if (!originalType.hasSameSizesAndDtype(newType)) {
    tensor = builder.create<TensorStaticInfoCastOp>(
        loc, originalType.getWithSizesAndDtypeFrom(newType), tensor);
  }

  // Unless both the original and new types are both value tensors, we end
  // up creating one op that converts between the value and non-value tensor
  // domains. If both the original and new types are both non-value tensors,
  // then we do the copy by going to a value tensor and back.
  if (tensor.getType().isa<NonValueTensorType>())
    tensor = builder.create<CopyToValueTensorOp>(loc, tensor);
  if (newType.isa<NonValueTensorType>())
    tensor = builder.create<CopyToNonValueTensorOp>(loc, tensor);

  return tensor;
}

bool mlir::torch::Torch::isListPotentiallyMutated(Value list) {
  assert(list.getType().isa<Torch::ListType>());
  return llvm::any_of(list.getUsers(), potentiallyMutatesListOperands);
}

bool mlir::torch::Torch::potentiallyMutatesListOperands(Operation *op) {
  // TODO: Find a better place to put this assertion.
  assert((!op->hasTrait<Torch::OpTrait::HasValueSemantics>() ||
          op->hasTrait<OpTrait::ReadOnly>()) &&
         "HasValueSemantics should imply ReadOnly!");
  // ReadOnly ops trivially do not mutate any list operands.
  if (op->hasTrait<Torch::OpTrait::ReadOnly>())
    return false;

  // Ops with no MemoryEffectOpInterface effects also do not mutate any list
  // operands.
  if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (effects.hasNoEffect())
      return false;
  }

  // Conservatively assume that an op might mutate any list operands.
  return true;
}

static IntegerAttr getI64IntegerAttr(MLIRContext *context, int64_t value) {
  return IntegerAttr::get(IntegerType::get(context, 64), value);
}

static FloatAttr getF64FloatAttr(MLIRContext *context, double value) {
  return FloatAttr::get(Float64Type::get(context), value);
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

LogicalResult MethodOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto func =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, functionAttr());
  if (!func)
    return emitError() << "'@" << function()
                       << "' does not reference a valid function";
  if (func.getVisibility() != SymbolTable::Visibility::Private)
    return emitError() << "'@" << function()
                       << "' must reference a private function";
  if (func.isDeclaration())
    return emitError() << "'@" << function()
                       << "' must reference a function that is defined (not "
                          "merely declared)";
  auto expectedReceiverArgType = NnModuleType::get(
      getContext(), getOperation()->getParentOfType<ClassTypeOp>().getName());
  if (func.getFunctionType().getNumInputs() == 0 ||
      func.getFunctionType().getInput(0) != expectedReceiverArgType) {
    return emitError() << "the referenced function '" << function()
                       << "' must have a first argument of type "
                       << expectedReceiverArgType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NnModuleOp
//===----------------------------------------------------------------------===//

LogicalResult NnModuleOp::verify() {
  for (Operation &child : *getBody())
    if (!isa<SlotOp, NnModuleTerminatorOp>(&child))
      return child.emitOpError() << "is not allowed inside 'torch.nn_module'";
  return success();
}

LogicalResult NnModuleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto classType = symbolTable.lookupNearestSymbolFrom<ClassTypeOp>(
      *this, SymbolRefAttr::get(getContext(), getClassName()));
  if (!classType)
    return emitError() << "'" << getClassName()
                       << "' does not reference a valid class type";

  auto attrs = llvm::to_vector<6>(getBody()->getOps<SlotOp>());
  auto attrDefs = llvm::to_vector<6>(classType.getBody()->getOps<AttrOp>());
  if (attrs.size() != attrDefs.size())
    return emitError() << "number of 'torch.slot's in a 'torch.nn_module' must "
                          "match number of 'torch.attr's in "
                          "the corresponding 'torch.class_type'";
  for (int i = 0, e = attrs.size(); i != e; i++) {
    SlotOp attr = attrs[i];
    AttrOp attrDef = attrDefs[i];
    if (!isValidSubtype(attr.value().getType(), attrDef.type()) ||
        attr.name() != attrDef.name()) {
      return attr.emitOpError()
          .append("is expected to match type and name of '",
                  attrDef.getOperation(), "'")
          .attachNote(attrDef.getLoc())
          .append("see torch.attr at corresponding index ", i, " here");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrimListConstructOp
//===----------------------------------------------------------------------===//

LogicalResult PrimListConstructOp::verify() {
  auto resultType = getResult().getType();
  auto resultElementType = resultType.dyn_cast<ListType>().getContainedType();
  auto matchResultElementType = [&](Type type) {
    return isValidSubtype(type, resultElementType);
  };
  if (!llvm::all_of(getOperandTypes(), matchResultElementType)) {
    return emitError() << "operand types should have the same type as the "
                          "list contained type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrimDictConstructOp
//===----------------------------------------------------------------------===//

LogicalResult PrimDictConstructOp::verify() {
  auto isValidSubTypeOf = [](Type expectedType) {
    return [=](Type type) { return isValidSubtype(type, expectedType); };
  };

  if (!llvm::all_of(keys().getTypes(), isValidSubTypeOf(getKeyType())))
    return emitError() << "keys should be of Dict key type";

  if (!llvm::all_of(values().getTypes(), isValidSubTypeOf(getValueType())))
    return emitError() << "values  should be of Dict value type";

  return success();
}

//===----------------------------------------------------------------------===//
// ClassTypeOp
//===----------------------------------------------------------------------===//

LogicalResult ClassTypeOp::verify() {
  llvm::StringMap<Operation *> namesToOps;
  for (Operation &child : getBody()->without_terminator()) {
    if (!isa<AttrOp, MethodOp>(&child))
      return child.emitOpError() << "is not allowed inside `torch.class_type`";
    StringRef name;
    if (auto attr = dyn_cast<AttrOp>(child))
      name = attr.name();
    else
      name = cast<MethodOp>(child).name();
    auto itAndWasInserted = namesToOps.insert({name, &child});
    auto it = itAndWasInserted.first;
    bool wasInserted = itAndWasInserted.second;
    if (!wasInserted) {
      auto diag = emitOpError().append("has duplicate attr/method with name '",
                                       name, "'");
      diag.attachNote(it->second->getLoc())
          .append("see first conflicting attr/method here");
      diag.attachNote(child.getLoc())
          .append("see second conflicting attr/method here");
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrimLoopOp
//===----------------------------------------------------------------------===//

OperandRange PrimLoopOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0);
  return iterArgsInit();
}

void PrimLoopOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  (void)operands;

  if (!index.hasValue()) {
    regions.emplace_back(&region(), region().getArguments().slice(1));
    return;
  }
  assert(*index == 0);
  regions.emplace_back(&region(), region().getArguments().slice(1));
  regions.emplace_back(getResults());
}

bool PrimLoopOp::isForLike() {
  bool b;
  return matchPattern(initialCondition(), m_TorchConstantBool(&b)) && b;
}

//===----------------------------------------------------------------------===//
// PrimLoopConditionOp
//===----------------------------------------------------------------------===//

MutableOperandRange
PrimLoopConditionOp::getMutableSuccessorOperands(Optional<unsigned> index) {
  // Pass all operands except the condition to the successor which is the
  // parent loop op.
  return iterArgsMutable();
}

//===----------------------------------------------------------------------===//
// PrimIfOp
//===----------------------------------------------------------------------===//

ParseResult PrimIfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type boolType = builder.getType<Torch::BoolType>();
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, boolType, result.operands))
    return failure();
  // Parse results type list.
  if (parser.parseArrowTypeList(result.types))
    return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  // Parse the 'else' region.
  if (parser.parseKeyword("else"))
    return failure();
  if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void PrimIfOp::print(OpAsmPrinter &p) {
  p << " " << condition();
  p << " -> (" << getResultTypes() << ") ";
  p.printRegion(thenRegion(), /*printEntryBlockArgs=*/false);
  p << " else ";
  p.printRegion(elseRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict((*this)->getAttrs());
}

void PrimIfOp::getSuccessorRegions(Optional<unsigned> index,
                                   ArrayRef<Attribute> operands,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // If the condition is constant, we can give a more precise answer.
  if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    Region *executedRegion =
        condAttr.getValue().isOneValue() ? &thenRegion() : &elseRegion();
    regions.push_back(RegionSuccessor(executedRegion));
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&thenRegion()));
  regions.push_back(RegionSuccessor(&elseRegion()));
  return;
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

void PrimIfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  // If the condition is constant, delete the dead branch and inline the live
  // branch.
  patterns.add(+[](PrimIfOp op, PatternRewriter &rewriter) {
    auto constantBool = op.condition().getDefiningOp<Torch::ConstantBoolOp>();
    if (!constantBool)
      return rewriter.notifyMatchFailure(op, "non-constant condition");
    replaceOpWithRegion(
        rewriter, op, constantBool.value() ? op.thenRegion() : op.elseRegion());
    return success();
  });
  // If the thenRegion and elseRegion yield the same Value's, then use those
  // directly.
  patterns.add(+[](PrimIfOp op, PatternRewriter &rewriter) {
    auto trueTerminator = op.thenRegion().front().getTerminator();
    auto falseTerminator = op.elseRegion().front().getTerminator();
    bool madeChange = false;
    SmallVector<int> resultsToErase;
    for (auto t : llvm::zip(trueTerminator->getOperands(),
                            falseTerminator->getOperands(), op->getResults())) {
      auto trueVal = std::get<0>(t);
      auto falseVal = std::get<1>(t);
      auto resultToBeReplaced = std::get<2>(t);
      if (trueVal == falseVal) {
        madeChange |= !resultToBeReplaced.use_empty();
        resultToBeReplaced.replaceAllUsesWith(trueVal);
      }
    }
    // We leave it up to a separate pattern (not yet implemented) to erase the
    // results that are now dead. That transformation is independently useful,
    // and also pretty tricky to implement because it changes the number of
    // results.
    return success(madeChange);
  });
  // Erase any dead results.
  patterns.add(+[](PrimIfOp op, PatternRewriter &rewriter) {
    llvm::BitVector resultsToErase(op.getNumResults());
    for (auto result : llvm::enumerate(op->getResults())) {
      if (result.value().use_empty())
        resultsToErase.set(result.index());
    }

    // If no results have uses and there are no side effects, just erase the op.
    // Approximate the body having no side effects by checking if it is just a
    // terminator.
    // Note: We don't want to make this logic too fancy, because in general,
    // checking for recursive side effects can result in a quadratic amount of
    // work (N nested If's each resulting in O(N) work). It should probably be
    // split into its own pattern if we want to make it fancier.
    if (resultsToErase.all() &&
        llvm::hasSingleElement(op.thenRegion().front()) &&
        llvm::hasSingleElement(op.elseRegion().front())) {
      rewriter.eraseOp(op);
      return success();
    }

    // If there are no results to erase, we're done.
    if (!resultsToErase.any())
      return failure();

    SmallVector<Type> newResultTypes;
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      if (resultsToErase[i])
        continue;
      newResultTypes.push_back(op->getResult(i).getType());
    }
    auto newIf =
        rewriter.create<PrimIfOp>(op->getLoc(), newResultTypes, op.condition());
    rewriter.inlineRegionBefore(op.thenRegion(), newIf.thenRegion(),
                                newIf.thenRegion().end());
    rewriter.inlineRegionBefore(op.elseRegion(), newIf.elseRegion(),
                                newIf.elseRegion().end());
    newIf.thenRegion().front().getTerminator()->eraseOperands(resultsToErase);
    newIf.elseRegion().front().getTerminator()->eraseOperands(resultsToErase);
    SmallVector<Value> replacementValues;
    for (int i = 0, e = op->getNumResults(), nextNewValue = 0; i < e; ++i) {
      if (resultsToErase[i])
        replacementValues.push_back(nullptr);
      else
        replacementValues.push_back(newIf->getResult(nextNewValue++));
    }
    rewriter.replaceOp(op, replacementValues);

    return success();
  });
}

//===----------------------------------------------------------------------===//
// DerefineOp
//===----------------------------------------------------------------------===//

bool DerefineOp::areCastCompatible(mlir::TypeRange inputs,
                                   mlir::TypeRange outputs) {
  return isValidSubtype(inputs[0], outputs[0]);
}

OpFoldResult DerefineOp::fold(ArrayRef<Attribute> operands) {
  auto uncheckedCast = getOperand().getDefiningOp<PrimUncheckedCastOp>();
  if (!uncheckedCast)
    return nullptr;
  if (uncheckedCast.getOperand().getType() == getType())
    return uncheckedCast.getOperand();
  return nullptr;
}

void DerefineOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add(+[](DerefineOp op, PatternRewriter &rewriter) {
    bool madeChange = false;
    for (OpOperand &use : llvm::make_early_inc_range(op->getUses())) {
      if (use.getOwner()->hasTrait<OpTrait::AllowsTypeRefinement>()) {
        use.set(op.getOperand());
        madeChange = true;
      }
    }
    return success(madeChange);
  });
}

static OpFoldResult atenIsOrIsNotFoldHelper(Operation *op, bool equalIsTrue) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  // Look through DerefineOp's to get more refined static information.
  if (auto derefine = lhs.getDefiningOp<DerefineOp>())
    lhs = derefine.getOperand();
  if (auto derefine = rhs.getDefiningOp<DerefineOp>())
    rhs = derefine.getOperand();
  Type lhsType = lhs.getType();
  Type rhsType = rhs.getType();

  // If either type is a NoneType, make it be the lhsType.
  if (rhsType.isa<Torch::NoneType>()) {
    std::swap(lhsType, rhsType);
    std::swap(lhs, rhs);
  }

  // For now, check a few specific cases.

  // If both types are the singleton `!torch.none` type, then we don't even need
  // to look at the values.
  if (lhsType.isa<Torch::NoneType>() && rhsType.isa<Torch::NoneType>())
    return IntegerAttr::get(IntegerType::get(op->getContext(), 1), equalIsTrue);

  // If neither type is a subtype of the other, then the result is false.
  // TODO: Implement and use subtype infra for this.
  // For now, check a specific case.
  // If the rhs is not OptionalType, then we know it cannot be None.
  if (lhsType.isa<Torch::NoneType>() && !rhsType.isa<Torch::OptionalType>()) {
    return IntegerAttr::get(IntegerType::get(op->getContext(), 1),
                            !equalIsTrue);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Aten__RangeLengthOp
//===----------------------------------------------------------------------===//

OpFoldResult Aten__RangeLengthOp::fold(ArrayRef<Attribute> operands) {
  auto lo = operands[0];
  auto hi = operands[1];
  auto step = operands[2];
  if (!lo || !hi || !step)
    return nullptr;
  auto loInt = lo.dyn_cast_or_null<IntegerAttr>().getValue();
  auto hiInt = hi.dyn_cast_or_null<IntegerAttr>().getValue();
  auto stepInt = step.dyn_cast_or_null<IntegerAttr>().getValue();
  // TODO: Implement folding for negative steps.
  if (stepInt.isNegative())
    return nullptr;
  // From Python language spec:
  // r[i] = lo + step*i such that i >= 0 and r[i] < hi
  // So maximize `i` such that lo + step * i < hi
  // ==> i == ceildiv(hi - lo, step)
  return IntegerAttr::get(lo.getType(),
                          llvm::APIntOps::RoundingSDiv(hiInt - loInt, stepInt,
                                                       APInt::Rounding::UP));
}

//===----------------------------------------------------------------------===//
// Aten__DeriveIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult Aten__DeriveIndexOp::fold(ArrayRef<Attribute> operands) {
  auto index = operands[0];
  auto start = operands[1];
  auto step = operands[2];
  if (!index || !start || !step)
    return nullptr;
  auto indexInt = index.dyn_cast_or_null<IntegerAttr>().getValue();
  auto startInt = start.dyn_cast_or_null<IntegerAttr>().getValue();
  auto stepInt = step.dyn_cast_or_null<IntegerAttr>().getValue();
  return IntegerAttr::get(index.getType(), startInt + stepInt * indexInt);
}

//===----------------------------------------------------------------------===//
// Aten__Is__Op
//===----------------------------------------------------------------------===//

OpFoldResult Aten__Is__Op::fold(ArrayRef<Attribute> operands) {
  return atenIsOrIsNotFoldHelper(*this, /*equalIsTrue=*/true);
}

//===----------------------------------------------------------------------===//
// Aten__Isnot__Op
//===----------------------------------------------------------------------===//

OpFoldResult Aten__Isnot__Op::fold(ArrayRef<Attribute> operands) {
  return atenIsOrIsNotFoldHelper(*this, /*equalIsTrue=*/false);
}

//===----------------------------------------------------------------------===//
// Aten__Not__Op
//===----------------------------------------------------------------------===//

OpFoldResult Aten__Not__Op::fold(ArrayRef<Attribute> operands) {
  bool value;
  if (!matchPattern(getOperand(), m_TorchConstantBool(&value)))
    return nullptr;
  return IntegerAttr::get(IntegerType::get(getContext(), 1), !value);
}

//===----------------------------------------------------------------------===//
// AtenNeBoolOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenNeBoolOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand(0) == getOperand(1))
    return IntegerAttr::get(IntegerType::get(getContext(), 1), false);

  bool a, b;
  if (!matchPattern(getOperand(0), m_TorchConstantBool(&a)))
    return nullptr;
  if (!matchPattern(getOperand(1), m_TorchConstantBool(&b)))
    return nullptr;
  return IntegerAttr::get(IntegerType::get(getContext(), 1), a != b);
}

//===----------------------------------------------------------------------===//
// AtenSqueezeOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenSqueezeOp::fold(ArrayRef<Attribute> operands) {
  if (auto tensorType = getOperand().getType().dyn_cast<BaseTensorType>()) {
    if (tensorType.hasSizes() && tensorType.getSizes().size() == 0)
      return getOperand();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenSqueezeDimOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenSqueezeDimOp::fold(ArrayRef<Attribute> operands) {
  if (auto tensorType = getOperand(0).getType().dyn_cast<BaseTensorType>()) {
    if (tensorType.hasSizes() && tensorType.getSizes().size() == 0)
      return getOperand(0);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenToDtypeOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenToDtypeOp::fold(ArrayRef<Attribute> operands) {
  bool nonBlocking, copyArg;
  // The non_blocking arg must be `False`.
  if (!matchPattern(non_blocking(), m_TorchConstantBool(&nonBlocking)) ||
      nonBlocking)
    return nullptr;
  // The copy arg must be `False`.
  if (!matchPattern(copy(), m_TorchConstantBool(&copyArg)) || copyArg)
    return nullptr;
  // The memory_format arg must be `none`.
  if (!memory_format().getType().isa<Torch::NoneType>())
    return nullptr;

  auto inputType = self().getType().cast<BaseTensorType>();
  auto resType = getType().cast<BaseTensorType>();
  // If the types aren't equal, then we can't fold.
  if (inputType != resType)
    return nullptr;
  // If the type does not have a statically known dtype, then we cannot fold.
  // For example, folding `tensor<*,unk>` to `tensor<*,unk>` would be wrong,
  // since the `unk` could be dynamically different for the operand and result.
  if (!inputType.hasDtype())
    return nullptr;
  // Fold when both the input tensor and result are of the same type.
  return getOperand(0);
}

//===----------------------------------------------------------------------===//
// AtenToDtypeLayoutOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenToDtypeLayoutOp::fold(ArrayRef<Attribute> operands) {
  // The pin_memory arg should be either constant `False` or `none`.
  if (!pin_memory().getType().isa<Torch::NoneType>()) {
    bool pinMemory;
    if (!matchPattern(pin_memory(), m_TorchConstantBool(&pinMemory)))
      return nullptr;
    else if (pinMemory)
      return nullptr;
  }

  // The non_blocking arg should be constant `False`.
  bool nonBlocking;
  if (!matchPattern(non_blocking(), m_TorchConstantBool(&nonBlocking)))
    return nullptr;
  else if (nonBlocking)
    return nullptr;

  // The copy arg should be constant `False`.
  bool copyArg;
  if (!matchPattern(copy(), m_TorchConstantBool(&copyArg)))
    return nullptr;
  else if (copyArg)
    return nullptr;

  // The device arg must be `none`.
  if (!device().getType().isa<Torch::NoneType>())
    return nullptr;

  // The memory_format arg must be `none`.
  if (!memory_format().getType().isa<Torch::NoneType>())
    return nullptr;

  auto inputType = self().getType().cast<BaseTensorType>();
  auto resType = getType().cast<BaseTensorType>();
  // If the types aren't equal, then we can't fold.
  if (inputType != resType)
    return nullptr;
  // If the type does not have a statically known dtype, then we cannot fold.
  // For example, folding `tensor<*,unk>` to `tensor<*,unk>` would be wrong,
  // since the `unk` could be dynamically different for the operand and result.
  if (!inputType.hasDtype())
    return nullptr;

  // The layout arg should be either `none` or `0` i.e. strided.
  if (!layout().getType().isa<Torch::NoneType>()) {
    int64_t tensorLayout;
    if (!matchPattern(layout(), m_TorchConstantInt(&tensorLayout)))
      return nullptr;
    else if (tensorLayout != torch_upstream::Layout::Strided)
      return nullptr;
  }

  // Fold when both the input tensor and result are of the same type and the
  // layout arg is strided.
  return getOperand(0);
}

//===----------------------------------------------------------------------===//
// AtenViewOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenViewOp::fold(ArrayRef<Attribute> operands) {
  auto inputType = getOperand(0).getType().dyn_cast<BaseTensorType>();
  if (!inputType || !inputType.hasSizes() || inputType.getSizes().size() != 1)
    return nullptr;
  auto resType = getType().dyn_cast<BaseTensorType>();
  if (!resType || !resType.hasSizes() || resType.getSizes().size() != 1)
    return nullptr;
  // Fold when both the input tensor and result are unity rank tensors.
  return getOperand(0);
}

//===----------------------------------------------------------------------===//
// AtenDimOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenDimOp::fold(ArrayRef<Attribute> operands) {
  if (auto tensorType = getOperand().getType().dyn_cast<BaseTensorType>()) {
    if (tensorType.hasSizes())
      return IntegerAttr::get(IntegerType::get(getContext(), 64),
                              tensorType.getSizes().size());
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenLenTOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenLenTOp::fold(ArrayRef<Attribute> operands) {
  // `len([1,1,1])` -> `3`, if it is not mutated.
  if (auto listConstruct =
          getOperand().getDefiningOp<Torch::PrimListConstructOp>()) {
    if (!isListPotentiallyMutated(listConstruct)) {
      return IntegerAttr::get(IntegerType::get(getContext(), 64),
                              listConstruct.getNumOperands());
    }
  }
  return nullptr;
}

void AtenLenTOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  // `len(t.size())` -> `t.ndim`
  patterns.add(+[](AtenLenTOp op, PatternRewriter &rewriter) {
    auto size = op.getOperand().getDefiningOp<AtenSizeOp>();
    if (!size)
      return rewriter.notifyMatchFailure(op, "operand not AtenSizeOp");
    rewriter.replaceOpWithNewOp<AtenDimOp>(op, size.getOperand());
    return success();
  });
}

//===----------------------------------------------------------------------===//
// AtenSizeOp
//===----------------------------------------------------------------------===//

void AtenSizeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add(+[](AtenSizeOp op, PatternRewriter &rewriter) {
    auto type = op.getOperand().getType().dyn_cast<BaseTensorType>();
    if (!type || !type.areAllSizesKnown())
      return rewriter.notifyMatchFailure(op, "all sizes not known");
    SmallVector<Value> listElements;
    for (int64_t size : type.getSizes()) {
      listElements.push_back(rewriter.create<Torch::ConstantIntOp>(
          op->getLoc(), rewriter.getI64IntegerAttr(size)));
    }
    rewriter.replaceOpWithNewOp<Torch::PrimListConstructOp>(
        op, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        listElements);
    return success();
  });
  // One-off pattern to erase if dead.
  // TODO: Use the effects infra to express the semantics of this op and enable
  // a centralized "erase if dead" canonicalization.
  // Specifically, we need to mark the op as only MemoryEffects::Allocate
  // so that `mlir::wouldOpBeTriviallyDead` does the right thing.
  patterns.add(+[](AtenSizeOp op, PatternRewriter &rewriter) {
    if (!op.use_empty())
      return failure();
    rewriter.eraseOp(op);
    return failure();
  });
}

//===----------------------------------------------------------------------===//
// AtenSizeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenSizeIntOp::fold(ArrayRef<Attribute> operands) {
  auto type = getOperand(0).getType().dyn_cast<BaseTensorType>();
  if (!type || !type.hasSizes())
    return nullptr;

  llvm::Optional<int64_t> dimOpt = matchLegalConstantIndexIntoListOfSize(
      this->dim(), type.getSizes().size());
  if (!dimOpt)
    return nullptr;
  if (type.getSizes()[*dimOpt] == kUnknownSize)
    return nullptr;
  return IntegerAttr::get(IntegerType::get(getContext(), 64),
                          type.getSizes()[*dimOpt]);
}

//===----------------------------------------------------------------------===//
// AtenGtIntOp
//===----------------------------------------------------------------------===//

static IntegerAttr getI1IntegerAttr(MLIRContext *context, bool value) {
  return IntegerAttr::get(IntegerType::get(context, 1),
                          static_cast<int64_t>(value));
}

using ConstantFloatComparator = std::function<bool(double, double)>;
template <typename OpTy>
static OpFoldResult
floatComparatorFoldHelper(OpTy op, ConstantFloatComparator comparator) {
  if (op.getOperand(0) == op.getOperand(1))
    return getI1IntegerAttr(op.getContext(), comparator(0, 0));

  double lhs, rhs;
  if (!matchPattern(op.getOperand(0), m_TorchConstantFloat(&lhs)) ||
      !matchPattern(op.getOperand(1), m_TorchConstantFloat(&rhs)))
    return nullptr;

  return getI1IntegerAttr(op.getContext(), comparator(lhs, rhs));
}

//===----------------------------------------------------------------------===//
// AtenLtFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenLtFloatOp::fold(ArrayRef<Attribute> operands) {
  return floatComparatorFoldHelper(*this,
                                   [](double a, double b) { return a < b; });
}

//===----------------------------------------------------------------------===//
// AtenGtFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenGtFloatOp::fold(ArrayRef<Attribute> operands) {
  return floatComparatorFoldHelper(*this,
                                   [](double a, double b) { return a > b; });
}

//===----------------------------------------------------------------------===//
// AtenGeFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenGeFloatOp::fold(ArrayRef<Attribute> operands) {
  return floatComparatorFoldHelper(*this,
                                   [](double a, double b) { return a >= b; });
}

//===----------------------------------------------------------------------===//
// AtenEqFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenEqFloatOp::fold(ArrayRef<Attribute> operands) {
  return floatComparatorFoldHelper(*this,
                                   [](double a, double b) { return a == b; });
}

using ConstantIntComparator = std::function<bool(int64_t, int64_t)>;
template <typename OpTy>
static OpFoldResult intComparatorFoldHelper(OpTy op,
                                            ConstantIntComparator comparator) {

  Value lhsValue = op->getOperand(0);
  Value rhsValue = op->getOperand(1);
  if (lhsValue == rhsValue)
    return getI1IntegerAttr(op.getContext(), comparator(0, 0));

  int64_t lhs, rhs;
  bool lhsIsConstant = matchPattern(lhsValue, m_TorchConstantInt(&lhs));
  bool rhsIsConstant = matchPattern(rhsValue, m_TorchConstantInt(&rhs));
  if (lhsIsConstant && rhsIsConstant)
    return getI1IntegerAttr(op.getContext(), comparator(lhs, rhs));

  // Ensure that if there is a constant, it is on the right.
  if (lhsIsConstant && !rhsIsConstant) {
    std::swap(lhs, rhs);
    std::swap(lhsValue, rhsValue);
    std::swap(lhsIsConstant, rhsIsConstant);
    auto newComparator = [comparator](int64_t lhs, int64_t rhs) {
      return comparator(rhs, lhs);
    };
    comparator = newComparator;
  }

  // Fold comparisons of AtenSizeIntOp against negative values.
  // AtenSizeIntOp is known to always be non-negative.
  if (rhsIsConstant && rhs < 0) {
    // We can return `comparator(0, -1)` here because of the property:
    // If x >= 0 && y < 0, then:
    // - cmp(x, y) == cmp(x + 1, y)
    // - cmp(x, y) == cmp(x, y - 1)
    // By induction all cases here are covered.
    if (auto size = lhsValue.getDefiningOp<AtenSizeIntOp>())
      return getI1IntegerAttr(op->getContext(), comparator(0, -1));
  }

  // Fold comparisons of AtenSizeIntOp against 0:
  // - torch.aten.size.int >= 0 ==> True.
  // - torch.aten.size.int < 0 ==> False.
  // (and the operand-swapped versions of the above)
  if (rhsIsConstant && rhs == 0) {
    if (auto size = lhsValue.getDefiningOp<AtenSizeIntOp>()) {
      // >= 0 comparison.
      if (comparator(0, 0) && comparator(1, 0))
        return getI1IntegerAttr(op->getContext(), true);
      // < 0 comparison.
      if (!comparator(0, 0) && comparator(-1, 0) && !comparator(1, 0))
        return getI1IntegerAttr(op->getContext(), false);
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenNeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenNeIntOp::fold(ArrayRef<Attribute> operands) {
  return intComparatorFoldHelper(*this,
                                 [](int64_t a, int64_t b) { return a != b; });
}

//===----------------------------------------------------------------------===//
// AtenEqIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenEqIntOp::fold(ArrayRef<Attribute> operands) {
  return intComparatorFoldHelper(*this,
                                 [](int64_t a, int64_t b) { return a == b; });
}

//===----------------------------------------------------------------------===//
// AtenEqStrOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenEqStrOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand(0) == getOperand(1))
    return getI1IntegerAttr(getContext(), true);

  auto aStr = a().getDefiningOp<ConstantStrOp>();
  auto bStr = b().getDefiningOp<ConstantStrOp>();

  if (aStr && bStr)
    return getI1IntegerAttr(getContext(), aStr == bStr);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenLtIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenLtIntOp::fold(ArrayRef<Attribute> operands) {
  return intComparatorFoldHelper(*this,
                                 [](int64_t a, int64_t b) { return a < b; });
}

//===----------------------------------------------------------------------===//
// AtenLeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenLeIntOp::fold(ArrayRef<Attribute> operands) {
  return intComparatorFoldHelper(*this,
                                 [](int64_t a, int64_t b) { return a <= b; });
}

//===----------------------------------------------------------------------===//
// AtenGtIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenGtIntOp::fold(ArrayRef<Attribute> operands) {
  return intComparatorFoldHelper(*this,
                                 [](int64_t a, int64_t b) { return a > b; });
}

//===----------------------------------------------------------------------===//
// AtenGeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenGeIntOp::fold(ArrayRef<Attribute> operands) {
  return intComparatorFoldHelper(*this,
                                 [](int64_t a, int64_t b) { return a >= b; });
}

//===----------------------------------------------------------------------===//
// AtenBoolFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenBoolFloatOp::fold(ArrayRef<Attribute> operands) {
  double c;
  if (matchPattern(getOperand(), m_TorchConstantFloat(&c)))
    return getI1IntegerAttr(getContext(), c != 0.0);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenBoolIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenBoolIntOp::fold(ArrayRef<Attribute> operands) {
  int64_t c;
  if (matchPattern(getOperand(), m_TorchConstantInt(&c)))
    return getI1IntegerAttr(getContext(), c != 0);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenFloatScalarOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenFloatScalarOp::fold(ArrayRef<Attribute> operands) {
  // Constant fold int -> float conversion.
  if (auto integerAttr = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    return FloatAttr::get(
        mlir::Float64Type::get(getContext()),
        static_cast<double>(integerAttr.getValue().getSExtValue()));
  }
  // If the input is float type already, the op is an identity.
  if (getType() == getOperand().getType())
    return getOperand();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenIntScalarOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenIntScalarOp::fold(ArrayRef<Attribute> operands) {
  // Constant fold float -> int conversion.
  if (auto floatAttr = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return IntegerAttr::get(
        mlir::IntegerType::get(getContext(), 64, IntegerType::Signed),
        static_cast<long>(floatAttr.getValue().convertToDouble()));
  }
  // If the input is int type already, the op is an identity.
  if (getType() == getOperand().getType())
    return getOperand();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// NonValueTensorLiteralOp
//===----------------------------------------------------------------------===//

LogicalResult NonValueTensorLiteralOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto attr = attributes.get("value").dyn_cast_or_null<ElementsAttr>();
  if (!attr)
    return failure();
  RankedTensorType tensorType = attr.getType().cast<RankedTensorType>();
  NonValueTensorType returnType =
      NonValueTensorType::get(tensorType.getContext(), tensorType.getShape(),
                              tensorType.getElementType());
  inferredReturnTypes.push_back(returnType);
  return success();
}

static bool areSizesAndDtypesCompatible(BaseTensorType a, BaseTensorType b) {
  if (a.hasSizes() && b.hasSizes()) {
    if (failed(verifyCompatibleShape(a.getSizes(), b.getSizes())))
      return false;
  }
  if (a.hasDtype() && b.hasDtype()) {
    if (a.getDtype() != b.getDtype())
      return false;
  }
  return true;
}

bool NonValueTensorLiteralOp::isCompatibleReturnTypes(TypeRange inferred,
                                                      TypeRange actual) {
  if (!actual[0].isa<BaseTensorType>())
    return false;
  return areSizesAndDtypesCompatible(inferred[0].cast<BaseTensorType>(),
                                     actual[0].cast<BaseTensorType>());
}

//===----------------------------------------------------------------------===//
// ValueTensorLiteralOp
//===----------------------------------------------------------------------===//

LogicalResult ValueTensorLiteralOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto attr = attributes.get("value").dyn_cast_or_null<ElementsAttr>();
  if (!attr)
    return failure();
  RankedTensorType tensorType = attr.getType().cast<RankedTensorType>();
  ValueTensorType returnType =
      ValueTensorType::get(tensorType.getContext(), tensorType.getShape(),
                           tensorType.getElementType());
  inferredReturnTypes.push_back(returnType);
  return success();
}

OpFoldResult ValueTensorLiteralOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

//----------------------------------------------------------------------------//
// TensorStaticInfoCast
//----------------------------------------------------------------------------//

bool TensorStaticInfoCastOp::areCastCompatible(mlir::TypeRange inputs,
                                               mlir::TypeRange outputs) {
  return areSizesAndDtypesCompatible(inputs[0].cast<BaseTensorType>(),
                                     outputs[0].cast<BaseTensorType>());
}

void TensorStaticInfoCastOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add(+[](TensorStaticInfoCastOp op, PatternRewriter &rewriter) {
    auto reverseCast =
        op.operand().getDefiningOp<Torch::TensorStaticInfoCastOp>();
    if (!reverseCast || reverseCast.operand().getType() != op.getType())
      return failure();

    rewriter.replaceOp(op, reverseCast.operand());
    return success();
  });
  patterns.add(+[](TensorStaticInfoCastOp op, PatternRewriter &rewriter) {
    if (isValidSubtype(op.getOperand().getType(), op.getType())) {
      SmallVector<std::reference_wrapper<OpOperand>> usesToChange(
          llvm::make_filter_range(op->getUses(), [](OpOperand &operand) {
            return operand.getOwner()
                ->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>();
          }));

      if (usesToChange.empty())
        return failure();

      for (OpOperand &use : usesToChange) {
        Operation *user = use.getOwner();
        user->setOperand(use.getOperandNumber(), op.operand());
      }

      return success();
    }
    return failure();
  });
}

//===----------------------------------------------------------------------===//
// CopyToNonValueTensorOp
//===----------------------------------------------------------------------===//

LogicalResult CopyToNonValueTensorOp::verify() {
  auto resultType = getResult().getType().cast<BaseTensorType>();
  auto operandType = getOperand().getType().cast<BaseTensorType>();
  if (!resultType.hasSameSizesAndDtype(operandType))
    return emitError() << "operand and result must have same sizes and dtype";
  return success();
}

LogicalResult CopyToNonValueTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto resultType = operands[0].getType().cast<ValueTensorType>();
  inferredReturnTypes.push_back(resultType.getWithoutValueSemantics());
  return success();
}

void CopyToNonValueTensorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(), getResult());
}

//===----------------------------------------------------------------------===//
// CopyToValueTensorOp
//===----------------------------------------------------------------------===//

LogicalResult CopyToValueTensorOp::verify() {
  auto resultType = getResult().getType().cast<BaseTensorType>();
  auto operandType = getOperand().getType().cast<BaseTensorType>();
  if (!resultType.hasSameSizesAndDtype(operandType))
    return emitError() << "operand and result must have same sizes and dtype";
  return success();
}

LogicalResult CopyToValueTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto resultType = operands[0].getType().cast<NonValueTensorType>();
  inferredReturnTypes.push_back(resultType.getWithValueSemantics());
  return success();
}

void CopyToValueTensorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getOperand());
}

//===----------------------------------------------------------------------===//
// ConstantNoneOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantNoneOp::fold(ArrayRef<Attribute> operands) {
  return TypeAttr::get(Torch::NoneType::get(getContext()));
}

void ConstantNoneOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "none");
}

//===----------------------------------------------------------------------===//
// ConstantStrOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantStrOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void ConstantStrOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "str");
}

//===----------------------------------------------------------------------===//
// ConstantDeviceOp
//===----------------------------------------------------------------------===//

void ConstantDeviceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), value());
}

//===----------------------------------------------------------------------===//
// ConstantIntOp
//===----------------------------------------------------------------------===//

ParseResult ConstantIntOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder builder(result.getContext());
  result.addTypes(builder.getType<Torch::IntType>());
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  int64_t value;
  if (parser.parseInteger(value))
    return failure();
  result.addAttribute("value", builder.getI64IntegerAttr(value));
  return success();
}

void ConstantIntOp::print(OpAsmPrinter &p) {
  p << " ";
  p << value().getSExtValue();
  p.printOptionalAttrDict((*this)->getAttrs(), {"value"});
}

OpFoldResult Torch::ConstantIntOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void Torch::ConstantIntOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char> buf;
  llvm::raw_svector_ostream os(buf);
  os << "int" << value();
  setNameFn(getResult(), os.str());
}

//===----------------------------------------------------------------------===//
// ConstantFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult Torch::ConstantFloatOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void Torch::ConstantFloatOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // Calculate a stringified version of the number, compatible with MLIR
  // identifier syntax. (in practice, this just removes the '+' from 'e+' in
  // float string representation).
  SmallVector<char> buf;
  value().toString(buf, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                   /*TruncateZero=*/false);
  auto isValidMLIRIdentifierChar = [](char c) {
    return isalpha(c) || isdigit(c) || c == '_' || c == '$' || c == '.' ||
           c == '-';
  };
  auto numberStr = llvm::to_vector<16>(
      llvm::make_filter_range(buf, isValidMLIRIdentifierChar));

  // Construct the identifier string.
  buf.clear();
  llvm::append_range(buf, StringRef("float"));
  llvm::append_range(buf, numberStr);
  setNameFn(getResult(), StringRef(buf.data(), buf.size()));
}

//===----------------------------------------------------------------------===//
// ConstantBoolOp
//===----------------------------------------------------------------------===//

OpFoldResult Torch::ConstantBoolOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void Torch::ConstantBoolOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), value() ? "true" : "false");
}

//===----------------------------------------------------------------------===//
// PrimUncheckedCastOp
//===----------------------------------------------------------------------===//

bool PrimUncheckedCastOp::areCastCompatible(mlir::TypeRange inputs,
                                            mlir::TypeRange outputs) {
  return isValidSubtype(outputs[0], inputs[0]);
}

OpFoldResult PrimUncheckedCastOp::fold(ArrayRef<Attribute> operands) {
  if (auto derefineOp = x().getDefiningOp<Torch::DerefineOp>()) {
    if (derefineOp.operand().getType() == getType())
      return derefineOp.operand();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Aten__Getitem__TOp
//===----------------------------------------------------------------------===//

void Aten__Getitem__TOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add(+[](Aten__Getitem__TOp op, PatternRewriter &rewriter) {
    auto torchList = op.getOperand(0);
    if (isListPotentiallyMutated(torchList))
      return failure();

    auto listConstruct = torchList.getDefiningOp<Torch::PrimListConstructOp>();
    if (!listConstruct)
      return failure();

    // Get the index, but be careful because it might be statically invalid.
    llvm::Optional<int64_t> indexOpt = matchLegalConstantIndexIntoListOfSize(
        op.getOperand(1), listConstruct.getNumOperands());
    if (!indexOpt)
      return rewriter.notifyMatchFailure(op, "statically invalid index");

    rewriter.replaceOp(op, {listConstruct.getOperand(*indexOpt)});
    return success();
  });
  patterns.add(+[](Aten__Getitem__TOp op, PatternRewriter &rewriter) {
    auto sizeOp = op.list().getDefiningOp<AtenSizeOp>();
    if (!sizeOp)
      return failure();
    // This assumes tht the size doesn't change between the
    // AtenSizeOp and the Aten__Getitem__TOp.
    // `t_` is the only op I can find that changes the shape in-place. It seems
    // like otherwise we can treat the size of a tensor as having value
    // semantics. The other view-like ops don't have in-place variants --
    // they always return a new SSA value that is aliased to the input.
    // Can we have a pass to normalize the `t_` case and then elsewhere in the
    // compiler treat the size as having value semantics?
    // There's a small number of such ops, and they are marked as `inplace_view`
    // in PyTorch's `native_functions.yaml` file.
    rewriter.replaceOpWithNewOp<AtenSizeIntOp>(op, sizeOp.self(), op.idx());
    return success();
  });
}

//===----------------------------------------------------------------------===//
// AtenEqIntListOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenEqIntListOp::fold(ArrayRef<Attribute> operands) {
  auto lhsLiteral = a().getDefiningOp<Torch::PrimListConstructOp>();
  if (!lhsLiteral)
    return nullptr;
  auto rhsLiteral = b().getDefiningOp<Torch::PrimListConstructOp>();
  if (!rhsLiteral)
    return nullptr;

  // If the sizes don't match, then we know the lists aren't equal.
  if (lhsLiteral.getNumOperands() != rhsLiteral.getNumOperands())
    return getI1IntegerAttr(getContext(), false);

  // If the sizes match and all corresponding list elements are the same Value,
  // then we know the lists are equal.
  // Note that we can't prove that the lists are not-equal with this method,
  // since two different Value's might dynamically be equal.
  if (llvm::all_of(
          llvm::zip(lhsLiteral.getOperands(), rhsLiteral.getOperands()),
          [](const auto &pair) {
            return std::get<0>(pair) == std::get<1>(pair);
          }))
    return getI1IntegerAttr(getContext(), true);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// PrimTupleIndexOp
//===----------------------------------------------------------------------===//

void PrimTupleIndexOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add(+[](PrimTupleIndexOp op, PatternRewriter &rewriter) {
    auto tupleConstruct = op.tup().getDefiningOp<Torch::PrimTupleConstructOp>();
    if (!tupleConstruct)
      return failure();

    int64_t i;
    if (!matchPattern(op.i(), m_TorchConstantInt(&i)))
      return failure();

    if (i >= (int64_t)tupleConstruct.elements().size())
      return failure();

    // TODO: We should have a clear picture of whether we want to consistently
    // allow refinement, and where. It seems desirable to require precise
    // type equality for TupleConstruct / TupleIndex, but that might break
    // things.
    Value replacement = tupleConstruct.elements()[i];
    if (replacement.getType() != op.getType()) {
      if (op.getType().isa<BaseTensorType>()) {
        replacement = rewriter.create<Torch::TensorStaticInfoCastOp>(
            op.getLoc(), op.getType(), replacement);
      } else {
        return failure();
      }
    }
    rewriter.replaceOp(op, replacement);
    return success();
  });
}

//===----------------------------------------------------------------------===//
// PrimUninitializedOp
//===----------------------------------------------------------------------===//

void PrimUninitializedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add(+[](PrimUninitializedOp op, PatternRewriter &rewriter) {
    if (!op.use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  });
}

//===----------------------------------------------------------------------===//
// PrimTupleUnpackOp
//===----------------------------------------------------------------------===//

void PrimTupleUnpackOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add(+[](PrimTupleUnpackOp op, PatternRewriter &rewriter) {
    auto tupleConstruct = op.tup().getDefiningOp<Torch::PrimTupleConstructOp>();
    if (!tupleConstruct)
      return failure();

    rewriter.replaceOp(op, tupleConstruct.elements());
    return success();
  });
}

static PrimDictConstructOp getDictConstructIfNotModified(Value torchDict) {
  if (!llvm::all_of(torchDict.getUsers(), [](Operation *op) {
        return isa<Aten__Getitem__DictStrOp, Aten__Contains__StrOp,
                   AtenKeysStrOp, AtenGetDefaultStrOp, PrimDictConstructOp>(op);
      }))
    return nullptr;

  return torchDict.getDefiningOp<Torch::PrimDictConstructOp>();
}

//===----------------------------------------------------------------------===//
// Aten__Getitem__DictStrOp
//===----------------------------------------------------------------------===//

OpFoldResult Aten__Getitem__DictStrOp::fold(ArrayRef<Attribute> operands) {
  auto dictConstruct = getDictConstructIfNotModified(self());
  if (!dictConstruct)
    return nullptr;

  auto targetKey = key();
  for (auto i : llvm::zip(dictConstruct.keys(), dictConstruct.values())) {
    auto k = std::get<0>(i);
    if (k == targetKey)
      return std::get<1>(i);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Aten__Contains__StrOp
//===----------------------------------------------------------------------===//

OpFoldResult Aten__Contains__StrOp::fold(ArrayRef<Attribute> operands) {
  auto dictConstruct = getDictConstructIfNotModified(dict());
  if (!dictConstruct)
    return nullptr;

  auto targetKey = key();
  for (auto key : dictConstruct.keys()) {
    if (key == targetKey)
      return getI1IntegerAttr(getContext(), true);
  }
  return nullptr;
}

using BinaryIntOperatorFn = std::function<int64_t(int64_t, int64_t)>;
template <typename OpTy>
static OpFoldResult atenBinaryIntOperatorFoldHelper(OpTy op,
                                                    BinaryIntOperatorFn f) {
  int64_t lhs, rhs;
  if (!matchPattern(op.getOperand(0), m_TorchConstantInt(&lhs)) ||
      !matchPattern(op.getOperand(1), m_TorchConstantInt(&rhs)))
    return nullptr;

  return getI64IntegerAttr(op.getContext(), f(lhs, rhs));
}

//===----------------------------------------------------------------------===//
// AtenFloordivIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenFloordivIntOp::fold(ArrayRef<Attribute> operands) {
  return atenBinaryIntOperatorFoldHelper(
      *this, [](int64_t a, int64_t b) { return std::floor(a / (double)b); });
}

//===----------------------------------------------------------------------===//
// AtenRemainderIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenRemainderIntOp::fold(ArrayRef<Attribute> operands) {
  return atenBinaryIntOperatorFoldHelper(
      *this, [](int64_t a, int64_t b) { return a % b; });
}

//===----------------------------------------------------------------------===//
// AtenAddIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenAddIntOp::fold(ArrayRef<Attribute> operands) {
  return atenBinaryIntOperatorFoldHelper(
      *this, [](int64_t a, int64_t b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// AtenSubIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenSubIntOp::fold(ArrayRef<Attribute> operands) {
  return atenBinaryIntOperatorFoldHelper(
      *this, [](int64_t a, int64_t b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// AtenMulIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenMulIntOp::fold(ArrayRef<Attribute> operands) {
  int64_t lhs, rhs;
  bool lConstant = matchPattern(getOperand(0), m_TorchConstantInt(&lhs));
  bool rConstant = matchPattern(getOperand(1), m_TorchConstantInt(&rhs));
  if ((lConstant && lhs == 0) || (rConstant && rhs == 0))
    return getI64IntegerAttr(getContext(), 0);
  if (lConstant && rConstant)
    return getI64IntegerAttr(getContext(), lhs * rhs);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenNegIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenNegIntOp::fold(ArrayRef<Attribute> operands) {
  int64_t c;
  if (matchPattern(getOperand(), m_TorchConstantInt(&c)))
    return getI64IntegerAttr(getContext(), -c);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenSqrtIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenSqrtIntOp::fold(ArrayRef<Attribute> operands) {
  int64_t c;
  if (matchPattern(getOperand(), m_TorchConstantInt(&c)))
    return getF64FloatAttr(getContext(), std::sqrt(c));
  return nullptr;
}

//===----------------------------------------------------------------------===//
// PrimDtypeOp
//===----------------------------------------------------------------------===//

OpFoldResult PrimDtypeOp::fold(ArrayRef<Attribute> operands) {
  BaseTensorType tensorType = a().getType().cast<BaseTensorType>();
  if (tensorType.hasDtype()) {
    int64_t dtypeInt = getDtypeIntegerFromMlirType(tensorType.getDtype());
    if (dtypeInt != -1)
      return getI64IntegerAttr(getContext(), dtypeInt);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenIntTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenIntTensorOp::fold(ArrayRef<Attribute> operands) {
  // If a scalar number is converted to a 0-d tensor and passed on to
  // aten.Int.Tensor, fold to the scalar number.
  if (auto numToTensorScalar = a().getDefiningOp<PrimNumToTensorScalarOp>())
    return numToTensorScalar.a();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenFloatTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenFloatTensorOp::fold(ArrayRef<Attribute> operands) {
  // If a scalar number is converted to a 0-d tensor and passed on to
  // aten.Float.Tensor, fold to the scalar number.
  if (auto numToTensorScalar = a().getDefiningOp<PrimNumToTensorScalarOp>())
    return numToTensorScalar.a();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenDivFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenDivFloatOp::fold(ArrayRef<Attribute> operands) {
  double lhs, rhs;
  bool lConstant = matchPattern(getOperand(0), m_TorchConstantFloat(&lhs));
  bool rConstant = matchPattern(getOperand(1), m_TorchConstantFloat(&rhs));
  if (lConstant && lhs == 0.0)
    return getF64FloatAttr(getContext(), 0.0);
  if (lConstant && rConstant && rhs == 1.0)
    return getF64FloatAttr(getContext(), lhs);
  if (lConstant && rConstant)
    return getF64FloatAttr(getContext(), lhs / rhs);
  return nullptr;
}

// AtenCeilFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenCeilFloatOp::fold(ArrayRef<Attribute> operands) {
  double c;
  if (matchPattern(getOperand(), m_TorchConstantFloat(&c)))
    return getI64IntegerAttr(getContext(), std::ceil(c));
  return nullptr;
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PrimMaxIntOp
//===----------------------------------------------------------------------===//

OpFoldResult PrimMaxIntOp::fold(ArrayRef<Attribute> operands) {
  // If both operands are the same, then the operation is an identity.
  if (a() == b())
    return a();

  auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return nullptr;
  // Torch semantics are that !torch.int is 64-bit signed.
  return IntegerAttr::get(
      lhs.getType(),
      std::max(lhs.getValue().getSExtValue(), rhs.getValue().getSExtValue()));
}

//===----------------------------------------------------------------------===//
// PrimMinSelfIntOp
//===----------------------------------------------------------------------===//

OpFoldResult PrimMinSelfIntOp::fold(ArrayRef<Attribute> operands) {
  auto list = getOperand().getDefiningOp<PrimListConstructOp>();
  if (!list)
    return nullptr;
  // TODO: What does it return for an empty list?
  if (list->getNumOperands() == 0)
    return nullptr;

  SmallVector<int64_t> values;
  for (auto operand : list->getOperands()) {
    int64_t value;
    if (!matchPattern(operand, m_TorchConstantInt(&value)))
      return nullptr;
    values.push_back(value);
  }
  return getI64IntegerAttr(getContext(),
                           *std::min_element(values.begin(), values.end()));
}

//===----------------------------------------------------------------------===//
// ShapeCalculateOp
//===----------------------------------------------------------------------===//

void ShapeCalculateOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  (void)operands;

  if (!index.hasValue()) {
    // First thing the op does is branch into the shape calculation.
    regions.emplace_back(&shapeCalculation());
    return;
  }
  if (*index == 0) {
    // Body returns control to the outer op, passing through results.
    regions.emplace_back(getResults());
    return;
  }
  assert(*index == 1);
  // Shape calculation branches to the body.
  regions.emplace_back(&body());
}

//===----------------------------------------------------------------------===//
// ShapeCalculateYieldShapesOp
//===----------------------------------------------------------------------===//

MutableOperandRange ShapeCalculateYieldShapesOp::getMutableSuccessorOperands(
    Optional<unsigned> index) {
  // The shape operands don't get forwarded to the body.
  // MutableOperandRange always has an owning operation, even if empty, so
  // create a 0-length range.
  return MutableOperandRange(*this, /*start=*/0, /*length=*/0);
}

LogicalResult ShapeCalculateYieldShapesOp::verify() {
  auto parent = cast<ShapeCalculateOp>(getOperation()->getParentOp());
  if (parent.getNumResults() != getNumOperands())
    return emitOpError("expected number of shapes to match number of results");
  return success();
}
