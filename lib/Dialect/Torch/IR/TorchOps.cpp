//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
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

static IntegerAttr getI64IntegerAttr(MLIRContext *context, int64_t value) {
  return IntegerAttr::get(IntegerType::get(context, 64), value);
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

LogicalResult MethodOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto func =
      symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, functionAttr());
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

  if (auto any = type.dyn_cast<AnyType>())
    return true;

  if (auto number = type.dyn_cast<NumberType>())
    return subtype.isa<IntType>() || subtype.isa<Torch::FloatType>();

  if (auto optional = type.dyn_cast<OptionalType>())
    return isValidSubtype(subtype, optional.getContainedType()) ||
           subtype.isa<Torch::NoneType>();

  if (auto tuple = type.dyn_cast<Torch::TupleType>()) {
    if (!subtype.isa<Torch::TupleType>())
      return false;
    auto subtypes = subtype.cast<Torch::TupleType>().getContainedTypes();
    auto types = tuple.getContainedTypes();
    if (subtypes.size() != types.size())
      return false;
    for (auto t : llvm::zip(subtypes, types)) {
      if (!isValidSubtype(std::get<0>(t), std::get<1>(t)))
        return false;
    }
    return true;
  }

  // TODO: This is not subtyping according to PEP 483. See description
  // of NonValueTensorType.
  if (subtype.isa<NonValueTensorType>() && type.isa<NonValueTensorType>() &&
      type ==
          NonValueTensorType::getWithLeastStaticInformation(type.getContext()))
    return true;

  if (subtype.isa<ValueTensorType>() && type.isa<ValueTensorType>() &&
      type == ValueTensorType::getWithLeastStaticInformation(type.getContext()))
    return true;
  return false;
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

static LogicalResult verify(PrimListConstructOp op) {
  auto resultType = op.getResult().getType();
  auto resultElementType = resultType.dyn_cast<ListType>().getContainedType();
  auto matchResultElementType = [&](Type type) {
    return isValidSubtype(type, resultElementType);
  };
  if (!llvm::all_of(op->getOperandTypes(), matchResultElementType)) {
    return op.emitError() << "operand types should have the same type as the "
                             "list contained type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrimDictConstructOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(PrimDictConstructOp op) {
  auto isValidSubTypeOf = [](Type expectedType) {
    return [=](Type type) { return isValidSubtype(type, expectedType); };
  };

  Type keyType = op.getKeyType();
  if (!llvm::all_of(op.keys().getTypes(), isValidSubTypeOf(keyType)))
    return op.emitError() << "keys should be of Dict key type";

  Type valueType = op.getValueType();
  if (!llvm::all_of(op.values().getTypes(), isValidSubTypeOf(valueType)))
    return op.emitError() << "values  should be of Dict value type";

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
  p << " " << op.condition();
  p << " -> (" << op.getResultTypes() << ")";
  p.printRegion(op.thenRegion(), /*printEntryBlockArgs=*/false);
  p << " else";
  p.printRegion(op.elseRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict(op->getAttrs());
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
}

//===----------------------------------------------------------------------===//
// DerefineOp
//===----------------------------------------------------------------------===//

bool DerefineOp::areCastCompatible(mlir::TypeRange inputs,
                                   mlir::TypeRange outputs) {
  return isValidSubtype(inputs[0], outputs[0]);
}

template <typename OpTy>
static OpFoldResult atenIsOrIsNotFoldHelper(OpTy op, bool equalIsTrue) {
  Type lhsType = op.self().getType();
  Type rhsType = op.obj().getType();

  // If either type is a NoneType, make it be the lhsType.
  if (rhsType.template isa<Torch::NoneType>())
    std::swap(lhsType, rhsType);
  // TODO: Implement and use subtype infra for this.
  // If neither type is a subtype of the other, then the result is false.
  if (lhsType.template isa<Torch::NoneType>() &&
      rhsType.template isa<Torch::NoneType>())
    return IntegerAttr::get(IntegerType::get(op.getContext(), 1), equalIsTrue);

  if (lhsType.template isa<Torch::NoneType>() &&
      !rhsType.template isa<Torch::OptionalType>())
    return IntegerAttr::get(IntegerType::get(op.getContext(), 1), !equalIsTrue);

  return nullptr;
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

  auto inputType = getOperand(0).getType().dyn_cast<BaseTensorType>();
  if (!inputType || !inputType.hasSizes())
    return nullptr;
  auto resType = getType().dyn_cast<BaseTensorType>();
  if (!resType || !resType.hasSizes() || inputType != resType)
    return nullptr;
  // Fold when both the input tensor and result are of the same type.
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
  // `len([1,1,1])` -> `3`
  if (auto listConstruct =
          getOperand().getDefiningOp<Torch::PrimListConstructOp>()) {
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
// AtenSizeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenSizeIntOp::fold(ArrayRef<Attribute> operands) {
  auto type = getOperand(0).getType().dyn_cast<BaseTensorType>();
  if (!type || !type.hasSizes())
    return nullptr;

  int64_t inputRank = type.getSizes().size();
  int64_t dim;
  if (!matchPattern(this->dim(), m_TorchConstantInt(&dim)))
    return nullptr;
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return nullptr;

  if (type.getSizes()[dim] == kUnknownSize)
    return nullptr;
  return IntegerAttr::get(IntegerType::get(getContext(), 64),
                          type.getSizes()[dim]);
}

//===----------------------------------------------------------------------===//
// AtenGtIntOp
//===----------------------------------------------------------------------===//

static IntegerAttr getI1IntegerAttr(MLIRContext *context, bool value) {
  return IntegerAttr::get(IntegerType::get(context, 1),
                          static_cast<int64_t>(value));
}

using ConstantIntComparator = std::function<bool(int64_t, int64_t)>;
template <typename OpTy>
static OpFoldResult comparatorFoldHelper(OpTy op,
                                         ConstantIntComparator comparator) {
  if (op.getOperand(0) == op.getOperand(1))
    return getI1IntegerAttr(op.getContext(), comparator(0, 0));

  int64_t lhs, rhs;
  if (!matchPattern(op.getOperand(0), m_TorchConstantInt(&lhs)) ||
      !matchPattern(op.getOperand(1), m_TorchConstantInt(&rhs)))
    return nullptr;

  return getI1IntegerAttr(op.getContext(), comparator(lhs, rhs));
}

//===----------------------------------------------------------------------===//
// AtenNeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenNeIntOp::fold(ArrayRef<Attribute> operands) {
  return comparatorFoldHelper(*this,
                              [](int64_t a, int64_t b) { return a != b; });
}

//===----------------------------------------------------------------------===//
// AtenEqIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenEqIntOp::fold(ArrayRef<Attribute> operands) {
  return comparatorFoldHelper(*this,
                              [](int64_t a, int64_t b) { return a == b; });
}

//===----------------------------------------------------------------------===//
// AtenEqFloatOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenEqFloatOp::fold(ArrayRef<Attribute> operands) {
  double lhs, rhs;

  if (!matchPattern(getOperand(0), m_TorchConstantFloat(&lhs)) ||
      !matchPattern(getOperand(1), m_TorchConstantFloat(&rhs)))
    return nullptr;

  return getI1IntegerAttr(getContext(), lhs == rhs);
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
  return comparatorFoldHelper(*this,
                              [](int64_t a, int64_t b) { return a < b; });
}

//===----------------------------------------------------------------------===//
// AtenLeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenLeIntOp::fold(ArrayRef<Attribute> operands) {
  return comparatorFoldHelper(*this,
                              [](int64_t a, int64_t b) { return a <= b; });
}

//===----------------------------------------------------------------------===//
// AtenGtIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenGtIntOp::fold(ArrayRef<Attribute> operands) {
  return comparatorFoldHelper(*this,
                              [](int64_t a, int64_t b) { return a > b; });
}

//===----------------------------------------------------------------------===//
// AtenGeIntOp
//===----------------------------------------------------------------------===//

OpFoldResult AtenGeIntOp::fold(ArrayRef<Attribute> operands) {
  return comparatorFoldHelper(*this,
                              [](int64_t a, int64_t b) { return a >= b; });
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
  p << " ";
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

//===----------------------------------------------------------------------===//
// Aten__Getitem__TOp
//===----------------------------------------------------------------------===//

void Aten__Getitem__TOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add(+[](Aten__Getitem__TOp op, PatternRewriter &rewriter) {
    auto torchList = op.getOperand(0);
    // TODO: Use a proper effects interface when more operands taking a list
    // are implemented.
    if (!llvm::all_of(torchList.getUsers(), [](Operation *op) {
          return isa<Aten__Getitem__TOp, AtenLenTOp>(op);
        }))
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

    rewriter.replaceOp(op, tupleConstruct.elements()[i]);
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

OpFoldResult AtenIntTensorOp::fold(ArrayRef<Attribute> operands) {
  // If an scalar number is converted to a 0-d tensor and passed on to
  // aten.Int.Tensor, fold to the scalar number.
  if (auto numToTensorScalar = a().getDefiningOp<PrimNumToTensorScalarOp>())
    return numToTensorScalar.a();
  return nullptr;
}

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/Torch/IR/TorchOps.cpp.inc"
