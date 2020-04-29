//===- NumpyOps.cpp - Core numpy dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/NumpyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "npcomp/Dialect/Numpy/NumpyDialect.h"

namespace mlir {
namespace npcomp {
namespace NUMPY {

//===----------------------------------------------------------------------===//
// BuildinUfuncOp
//===----------------------------------------------------------------------===//

static ParseResult parseBuiltinUfuncOp(OpAsmParser &parser,
                                       OperationState *result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result->attributes)) {
    return failure();
  }
  if (failed(parser.parseOptionalAttrDict(result->attributes))) {
    return failure();
  }
  return success();
}

static void printBuiltinUfuncOp(OpAsmPrinter &p, BuiltinUfuncOp op) {
  p << op.getOperationName() << " ";
  p.printSymbolName(op.getName());
  p.printOptionalAttrDict(op.getAttrs(), {SymbolTable::getSymbolAttrName()});
}

//===----------------------------------------------------------------------===//
// GenericUfuncOp
//===----------------------------------------------------------------------===//

static ParseResult parseGenericUfuncOp(OpAsmParser &parser,
                                       OperationState *result) {
  Builder b(result->getContext());

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result->attributes))
    return failure();

  // Parse the body of overloads.
  if (parser.parseLParen())
    return failure();

  SmallVector<Attribute, 4> overloadTypes;
  for (bool first = true;; first = false) {
    if (first) {
      if (parser.parseOptionalKeyword("overload"))
        break;
    }
    if (!first) {
      if (parser.parseOptionalComma())
        break;
      if (parser.parseKeyword("overload"))
        return failure();
    }
    SmallVector<OpAsmParser::OperandType, 2> argNames;
    SmallVector<Type, 2> argTypes;
    SmallVector<Type, 1> resultTypes;
    SmallVector<SmallVector<NamedAttribute, 2>, 1> unusedAttrs;
    bool isVariadic = false;
    if (::mlir::impl::parseFunctionSignature(parser, false, argNames, argTypes,
                                             unusedAttrs, isVariadic,
                                             resultTypes, unusedAttrs))
      return failure();
    overloadTypes.push_back(TypeAttr::get(
        FunctionType::get(argTypes, resultTypes, result->getContext())));
    auto *region = result->addRegion();
    if (parser.parseRegion(*region, argNames, argTypes))
      return failure();
  }

  if (parser.parseRParen())
    return failure();

  // Parse 'attributes {...}'
  if (parser.parseOptionalAttrDictWithKeyword(result->attributes))
    return failure();
  result->addAttribute(b.getIdentifier("overload_types"),
                       b.getArrayAttr(overloadTypes));

  return success();
}

static void printGenericUfuncOp(OpAsmPrinter &p, GenericUfuncOp op) {
  p << op.getOperationName() << " @" << op.getName() << "(";
  bool first = true;
  for (auto it : llvm::enumerate(op.getRegions())) {
    auto *region = it.value();
    if (first)
      first = false;
    else
      p << ", ";
    if (region->empty()) {
      p << "<<OVERLOAD_ERROR>>";
      continue;
    }

    Block &entryBlock = region->front();
    p << "overload(";
    if (it.index() >= op.overload_types().size()) {
      p << "<<MISSING OVERLOAD TYPE>>";
      continue;
    }
    TypeAttr tattr = op.overload_types()[it.index()].cast<TypeAttr>();
    FunctionType overloadType = tattr.getValue().dyn_cast<FunctionType>();
    if (!overloadType) {
      p << "<<ILLEGAL OVERLOAD TYPE>>";
      continue;
    }
    if (overloadType.getNumInputs() != entryBlock.getNumArguments()) {
      p << "<<OVERLOAD ARG MISMATCH>>";
      continue;
    }

    auto argTypes = entryBlock.getArgumentTypes();
    for (unsigned i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
      auto arg = entryBlock.getArgument(i);
      if (i > 0)
        p << ", ";
      p.printOperand(arg);
      p << ": ";
      p.printType(overloadType.getInputs()[i]);
    }
    p << ")";
    p.printArrowTypeList(overloadType.getResults());
    p.printRegion(*region, false, true);
  }
  p << ")";
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/Numpy/NumpyOps.cpp.inc"
} // namespace NUMPY
} // namespace npcomp
} // namespace mlir
