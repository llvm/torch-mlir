//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

Value mlir::NPCOMP::Torch::copyTensorToType(OpBuilder &builder, Location loc,
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

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

LogicalResult MethodOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto func = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, function());
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
  if (func.getType().getNumInputs() == 0 ||
      func.getType().getInput(0) != expectedReceiverArgType) {
    return emitError() << "the referenced function '" << function()
                       << "' must have a first argument of type "
                       << expectedReceiverArgType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NnModuleOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(NnModuleOp op) {
  for (Operation &child : *op.getBody())
    if (!isa<SlotOp, NnModuleTerminatorOp>(&child))
      return child.emitOpError() << "is not allowed inside 'torch.nn_module'";
  return success();
}

// PyTorch has a well-developed notion of subtyping.
//
// This is a restricted subset of it.
//
// TODO: Flesh this out.
// TODO: Decide / properly model the distinction between PEP 483 / Python
// subtyping vs "more static information".
bool isValidSubtype(Type subtype, Type type) {
  if (subtype == type)
    return true;
  if (auto optional = type.dyn_cast<OptionalType>())
    return subtype == optional.getContainedType() ||
           subtype.isa<Torch::NoneType>();
  // TODO: This is not subtyping according to PEP 483. See description
  // of NonValueTensorType.
  if (subtype.isa<NonValueTensorType>() && type.isa<NonValueTensorType>() &&
      type ==
          NonValueTensorType::getWithLeastStaticInformation(type.getContext()))
    return true;
  return false;
}

LogicalResult NnModuleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto classType =
      symbolTable.lookupNearestSymbolFrom<ClassTypeOp>(*this, getClassName());
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

static LogicalResult verify(PrimListConstructOp op) {
  auto resultType = op.getResult().getType();
  auto resultElementType = resultType.dyn_cast<ListType>().getContainedType();
  auto matchResultElementType = [&](Type type) {
    return type.getTypeID() == resultElementType.getTypeID();
  };
  if (!llvm::all_of(op->getOperandTypes(), matchResultElementType)) {
    return op.emitError() << "operand types should have the same type as the "
                             "list contained type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClassTypeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ClassTypeOp op) {
  llvm::StringMap<Operation *> namesToOps;
  for (Operation &child : op.getBody()->without_terminator()) {
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
      auto diag = op.emitOpError().append(
          "has duplicate attr/method with name '", name, "'");
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

//===----------------------------------------------------------------------===//
// PrimIfOp
//===----------------------------------------------------------------------===//

static ParseResult parsePrimIfOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
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

static void print(OpAsmPrinter &p, PrimIfOp op) {
  p << PrimIfOp::getOperationName() << " " << op.condition();
  p << " -> (" << op.getResultTypes() << ")";
  p.printRegion(op.thenRegion(), /*printEntryBlockArgs=*/false);
  p << " else";
  p.printRegion(op.elseRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict(op->getAttrs());
}

void PrimIfOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
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
}

//===----------------------------------------------------------------------===//
// DerefineOp
//===----------------------------------------------------------------------===//

bool DerefineOp::areCastCompatible(mlir::TypeRange inputs,
                                   mlir::TypeRange outputs) {
  return isValidSubtype(inputs[0], outputs[0]);
}

void DerefineOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add(+[](DerefineOp op, PatternRewriter &rewriter) {
    // TODO: Extend RefineTypes for this case and delete this canonicalization,
    // since we don't want control flow or calls to randomly block this fold
    // (this canonicalization pattern makes the compiler brittle to control flow
    // and calls).
    bool allAllowRefinement =
        llvm::all_of(op.getResult().getUsers(), allowsTypeRefinement);
    if (!allAllowRefinement)
      return failure();
    rewriter.replaceOp(op, op.getOperand());
    return success();
  });
}

//===----------------------------------------------------------------------===//
// Aten__Is__Op
//===----------------------------------------------------------------------===//

OpFoldResult Aten__Is__Op::fold(ArrayRef<Attribute> operands) {
  auto lhsType = self().getType();
  auto rhsType = obj().getType();
  // If either type is a NoneType, make it be the lhsType.
  if (rhsType.isa<Torch::NoneType>())
    std::swap(lhsType, rhsType);
  // TODO: Implement and use subtype infra for this.
  // If neither type is a subtype of the other, then the result is false.
  if (lhsType.isa<Torch::NoneType>() && !rhsType.isa<Torch::OptionalType>())
    return IntegerAttr::get(IntegerType::get(getContext(), 1), 0);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenLenTOp
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
  // `len([1,1,1])` -> `3`
  if (auto listConstruct = getOperand().getDefiningOp<Torch::PrimListConstructOp>()) {
    return IntegerAttr::get(IntegerType::get(getContext(), 64),
                            listConstruct.getNumOperands());
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
// AtenGtIntOp
//===----------------------------------------------------------------------===//

static IntegerAttr getI1IntegerAttr(MLIRContext *context, bool value) {
  return IntegerAttr::get(IntegerType::get(context, 1),
                          static_cast<int64_t>(value));
}

OpFoldResult AtenGtIntOp::fold(ArrayRef<Attribute> operands) {
  auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (lhs && rhs) {
    if (lhs.getValue().getSExtValue() > rhs.getValue().getSExtValue())
      return getI1IntegerAttr(getContext(), true);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AtenNeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenNeIntOp::fold(ArrayRef<Attribute> operands) {
  // `torch.aten.ne.int %x, %x` -> `false`
  if (getOperand(0) == getOperand(1))
    return getI1IntegerAttr(getContext(), false);
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
  auto tensorType = attr.getType().cast<RankedTensorType>();
  inferredReturnTypes.push_back(NonValueTensorType::getFromShaped(tensorType));
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
  auto tensorType = attr.getType().cast<RankedTensorType>();
  inferredReturnTypes.push_back(ValueTensorType::getFromShaped(tensorType));
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

//===----------------------------------------------------------------------===//
// CopyToNonValueTensorOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(CopyToNonValueTensorOp op) {
  auto resultType = op.getResult().getType().cast<BaseTensorType>();
  auto operandType = op.getOperand().getType().cast<BaseTensorType>();
  if (!resultType.hasSameSizesAndDtype(operandType)) {
    return op.emitError()
           << "operand and result must have same sizes and dtype";
  }
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

static LogicalResult verify(CopyToValueTensorOp op) {
  auto resultType = op.getResult().getType().cast<BaseTensorType>();
  auto operandType = op.getOperand().getType().cast<BaseTensorType>();
  if (!resultType.hasSameSizesAndDtype(operandType)) {
    return op.emitError()
           << "operand and result must have same sizes and dtype";
  }
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
// ToBuiltinTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ToBuiltinTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto resultType =
      operands[0].getType().cast<ValueTensorType>().toBuiltinTensor();
  if (!resultType)
    return failure();
  inferredReturnTypes.push_back(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// FromBuiltinTensorOp
//===----------------------------------------------------------------------===//

LogicalResult FromBuiltinTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      ValueTensorType::getFromShaped(operands[0].getType().cast<TensorType>()));
  return success();
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
// ConstantIntOp
//===----------------------------------------------------------------------===//

static ParseResult parseConstantIntOp(OpAsmParser &parser,
                                      OperationState &result) {
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

static void print(OpAsmPrinter &p, Torch::ConstantIntOp op) {
  p << Torch::ConstantIntOp::getOperationName() << " ";
  p << op.value().getSExtValue();
  p.printOptionalAttrDict(op->getAttrs(), {"value"});
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
  setNameFn(getResult(), "float");
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
// Aten__Getitem__TOp
//===----------------------------------------------------------------------===//

void Aten__Getitem__TOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                     MLIRContext *context) {
  patterns.add(+[](Aten__Getitem__TOp op, PatternRewriter &rewriter) {
    auto torchList = op.getOperand(0);
    if(!torchList.hasOneUse())
      return failure();

    auto listConstruct = torchList.getDefiningOp<Torch::PrimListConstructOp>();
    if (!listConstruct)
      return failure();

    int64_t index;
    if (!matchPattern(op.getOperand(1), m_TorchConstantInt(&index)))
      return failure();

    rewriter.replaceOp(op, {listConstruct.getOperand(index)});
    return success();
  });
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchOps.cpp.inc"
