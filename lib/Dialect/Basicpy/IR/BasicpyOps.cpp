//===- BasicpyOps.cpp - Core numpy dialect ops -------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::NPCOMP::Basicpy;

// Fallback verifier for ops that don't have a dedicated one.
template <typename T> static LogicalResult verify(T op) { return success(); }

//===----------------------------------------------------------------------===//
// BoolCastOp
//===----------------------------------------------------------------------===//

OpFoldResult BoolCastOp::fold(ArrayRef<Attribute> operands) {
  return operands[0];
}

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult BoolConstantOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void BoolConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  if (value())
    setNameFn(getResult(), "bool_true");
  else
    setNameFn(getResult(), "bool_false");
}

//===----------------------------------------------------------------------===//
// BytesConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult BytesConstantOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void BytesConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "bytes");
}

//===----------------------------------------------------------------------===//
// NumericConstantOp
//===----------------------------------------------------------------------===//

static ParseResult parseNumericConstantOp(OpAsmParser &parser,
                                          OperationState *result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result->attributes) ||
      parser.parseAttribute(valueAttr, "value", result->attributes))
    return failure();

  // If not an Integer or Float attr (which carry the type in the attr),
  // expect a trailing type.
  Type type;
  if (valueAttr.isa<IntegerAttr>() || valueAttr.isa<FloatAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();
  return parser.addTypeToList(type, result->types);
}

static void print(OpAsmPrinter &p, NumericConstantOp op) {
  p << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1)
    p << ' ';
  p << op.value();

  // If not an Integer or Float attr, expect a trailing type.
  if (!op.value().isa<IntegerAttr>() && !op.value().isa<FloatAttr>())
    p << " : " << op.getType();
}

static LogicalResult verify(NumericConstantOp &op) {
  auto value = op.value();
  if (!value)
    return op.emitOpError("requires a 'value' attribute");
  auto type = op.getType();

  if (type.isa<FloatType>()) {
    if (!value.isa<FloatAttr>())
      return op.emitOpError("requires 'value' to be a floating point constant");
    return success();
  }

  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (!value.isa<IntegerAttr>())
      return op.emitOpError("requires 'value' to be an integer constant");
    if (intType.getWidth() == 1)
      return op.emitOpError("cannot have an i1 type");
    return success();
  }

  if (type.isa<ComplexType>()) {
    if (auto complexComps = value.dyn_cast<ArrayAttr>()) {
      if (complexComps.size() == 2) {
        auto realValue = complexComps[0].dyn_cast<FloatAttr>();
        auto imagValue = complexComps[1].dyn_cast<FloatAttr>();
        if (realValue && imagValue &&
            realValue.getType() == imagValue.getType())
          return success();
      }
    }
    return op.emitOpError("requires 'value' to be a two element array of "
                          "floating point complex number components");
  }

  return op.emitOpError("unsupported basicpy.numeric_constant type");
}

OpFoldResult NumericConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "numeric_constant has no operands");
  return value();
}

void NumericConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  Type type = getType();
  if (auto intCst = value().dyn_cast<IntegerAttr>()) {
    IntegerType intTy = type.dyn_cast<IntegerType>();
    APInt intValue = intCst.getValue();

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << "num";
    if (intTy.isSigned())
      specialName << intValue.getSExtValue();
    else
      specialName << intValue.getZExtValue();
    if (intTy)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else {
    setNameFn(getResult(), "num");
  }
}

//===----------------------------------------------------------------------===//
// ExecOp
//===----------------------------------------------------------------------===//

void ExecOp::build(OpBuilder &builder, OperationState &result) {
  OpBuilder::InsertionGuard guard(builder);
  Region *body = result.addRegion();
  builder.createBlock(body);
}

static ParseResult parseExecOp(OpAsmParser &parser, OperationState *result) {
  Region *bodyRegion = result->addRegion();
  if (parser.parseOptionalAttrDictWithKeyword(result->attributes) ||
      parser.parseRegion(*bodyRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, ExecOp op) {
  p.printOptionalAttrDictWithKeyword(op->getAttrs());
  p.printRegion(op.body());
}

//===----------------------------------------------------------------------===//
// FuncTemplateCallOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(FuncTemplateCallOp op) {
  auto argNames = op.arg_names();
  if (argNames.size() > op.args().size()) {
    return op.emitOpError() << "expected <= kw arg names vs args";
  }

  for (auto it : llvm::enumerate(argNames)) {
    auto argName = it.value().cast<StringAttr>().getValue();
    if (argName == "*" && it.index() != 0) {
      return op.emitOpError() << "positional arg pack must be the first kw arg";
    }
    if (argName == "**" && it.index() != argNames.size() - 1) {
      return op.emitOpError() << "kw arg pack must be the last kw arg";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FuncTemplateOp
//===----------------------------------------------------------------------===//

void FuncTemplateOp::build(OpBuilder &builder, OperationState &result) {
  OpBuilder::InsertionGuard guard(builder);
  ensureTerminator(*result.addRegion(), builder, result.location);
}

static ParseResult parseFuncTemplateOp(OpAsmParser &parser,
                                       OperationState *result) {
  Region *bodyRegion = result->addRegion();
  StringAttr symbolName;

  if (parser.parseSymbolName(symbolName, SymbolTable::getSymbolAttrName(),
                             result->attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result->attributes) ||
      parser.parseRegion(*bodyRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  FuncTemplateOp::ensureTerminator(*bodyRegion, parser.getBuilder(),
                                   result->location);

  return success();
}

static void print(OpAsmPrinter &p, FuncTemplateOp op) {
  p << " ";
  p.printSymbolName(op.getName());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {SymbolTable::getSymbolAttrName()});
  p.printRegion(op.body());
}

static LogicalResult verify(FuncTemplateOp op) {
  Block *body = op.getBody();
  for (auto &childOp : body->getOperations()) {
    if (!llvm::isa<FuncOp>(childOp) &&
        !llvm::isa<FuncTemplateTerminatorOp>(childOp)) {
      return childOp.emitOpError() << "illegal operation in func_template";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SlotObjectMakeOp
//===----------------------------------------------------------------------===//

static ParseResult parseSlotObjectMakeOp(OpAsmParser &parser,
                                         OperationState *result) {
  llvm::SmallVector<OpAsmParser::OperandType, 4> operandTypes;
  if (parser.parseOperandList(operandTypes, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result->attributes) ||
      parser.parseArrowTypeList(result->types)) {
    return failure();
  }

  if (result->types.size() != 1 ||
      !result->types.front().isa<SlotObjectType>()) {
    return parser.emitError(parser.getNameLoc(),
                            "custom assembly form requires SlotObject result");
  }
  auto slotObjectType = result->types.front().cast<SlotObjectType>();
  result->addAttribute("className", slotObjectType.getClassName());
  return parser.resolveOperands(operandTypes, slotObjectType.getSlotTypes(),
                                parser.getNameLoc(), result->operands);
}

static void print(OpAsmPrinter &p, SlotObjectMakeOp op) {
  // If the argument types do not match the result type slots, then
  // print the generic form.
  auto canCustomPrint = ([&]() -> bool {
    auto type = op.result().getType().dyn_cast<SlotObjectType>();
    if (!type)
      return false;
    auto args = op.slots();
    auto slotTypes = type.getSlotTypes();
    if (args.size() != slotTypes.size())
      return false;
    for (unsigned i = 0, e = args.size(); i < e; ++i) {
      if (args[i].getType() != slotTypes[i])
        return false;
    }
    return true;
  })();
  if (!canCustomPrint) {
    p.printGenericOp(op);
    return;
  }

  p << "(";
  p.printOperands(op.slots());
  p << ")";
  p.printOptionalAttrDict(op->getAttrs(), {"className"});

  // Not really a symbol but satisfies same rules.
  p.printArrowTypeList(op.getOperation()->getResultTypes());
}

//===----------------------------------------------------------------------===//
// SlotObjectGetOp
//===----------------------------------------------------------------------===//

static ParseResult parseSlotObjectGetOp(OpAsmParser &parser,
                                        OperationState *result) {
  OpAsmParser::OperandType object;
  IntegerAttr indexAttr;
  Type indexType = parser.getBuilder().getIndexType();
  if (parser.parseOperand(object) || parser.parseLSquare() ||
      parser.parseAttribute(indexAttr, indexType, "index",
                            result->attributes) ||
      parser.parseRSquare()) {
    return failure();
  }
  Type objectType;
  if (parser.parseColonType(objectType) ||
      parser.resolveOperand(object, objectType, result->operands)) {
    return failure();
  }

  auto castObjectType = objectType.dyn_cast<SlotObjectType>();
  if (!castObjectType) {
    return parser.emitError(parser.getNameLoc(),
                            "illegal object type on custom assembly form");
  }
  auto index = indexAttr.getValue().getZExtValue();
  auto slotTypes = castObjectType.getSlotTypes();
  if (index >= slotTypes.size()) {
    return parser.emitError(parser.getNameLoc(),
                            "out of bound index on custom assembly form");
  }
  result->addTypes({slotTypes[index]});
  return success();
}

static void print(OpAsmPrinter &p, SlotObjectGetOp op) {
  // If the argument types do not match the result type slots, then
  // print the generic form.
  auto canCustomPrint = ([&]() -> bool {
    auto type = op.object().getType().dyn_cast<SlotObjectType>();
    if (!type)
      return false;
    auto index = op.index().getZExtValue();
    if (index >= type.getSlotCount())
      return false;
    if (op.result().getType() != type.getSlotTypes()[index])
      return false;
    return true;
  })();
  if (!canCustomPrint) {
    p.printGenericOp(op);
    return;
  }

  p << " ";
  p.printOperand(op.object());
  p << "[" << op.index() << "]";
  p.printOptionalAttrDict(op->getAttrs(), {"index"});
  p << " : ";
  p.printType(op.object().getType());
}

//===----------------------------------------------------------------------===//
// SingletonOp
//===----------------------------------------------------------------------===//

OpFoldResult SingletonOp::fold(ArrayRef<Attribute> operands) {
  auto resultType = getResult().getType();
  return TypeAttr::get(resultType);
}

//===----------------------------------------------------------------------===//
// StrConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult StrConstantOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}

void StrConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "str");
}

//===----------------------------------------------------------------------===//
// UnknownCastOp
//===----------------------------------------------------------------------===//

namespace {

class ElideIdentityUnknownCast : public OpRewritePattern<UnknownCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnknownCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.operand().getType() != op.result().getType())
      return failure();
    rewriter.replaceOp(op, op.operand());
    return success();
  }
};

} // namespace

void UnknownCastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<ElideIdentityUnknownCast>(context);
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.cpp.inc"
