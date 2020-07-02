//===- CPASupport.h - Support types and utilities for CPA -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Typing/CPA/Support.h"
#include "npcomp/Typing/CPA/Interfaces.h"

#include "mlir/IR/Operation.h"

using namespace mlir::NPCOMP::Typing::CPA;

ObjectBase::~ObjectBase() = default;

//===----------------------------------------------------------------------===//
// Environment
//===----------------------------------------------------------------------===//

Environment::Environment(Context &context)
    : context(context), constraints(context.newConstraintSet()),
      typeVars(context.newTypeVarSet()) {}

TypeBase *Environment::mapValueToType(Value value) {
  TypeBase *&cpaType = valueTypeMap[value];
  if (cpaType)
    return cpaType;

  cpaType = context.mapIrType(value.getType());
  assert(cpaType && "currently every IR type must map to a CPA type");

  // Do accounting for type vars.
  if (auto *tv = llvm::dyn_cast<TypeVar>(cpaType)) {
    typeVars->getTypeVars().push_back(*tv);
    tv->setAnchorValue(value);
  }

  return cpaType;
}

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

TypeBase *Context::mapIrType(::mlir::Type irType) {
  // First, see if the type knows how to map itself.
  assert(irType);
  if (auto mapper = irType.dyn_cast<CPA::TypeMapInterface>()) {
    auto *cpaType = mapper.mapToCPAType(*this);
    if (cpaType)
      return cpaType;
  }

  // Fallback to an IR type.
  return getIRValueType(irType);
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void Identifier::print(raw_ostream &os, bool brief) {
  os << "'" << value << "'";
}

void TypeVar::print(raw_ostream &os, bool brief) {
  os << "TypeVar(" << ordinal;
  if (!brief) {
    if (auto anchorValue = getAnchorValue()) {
      os << ", ";
      auto blockArg = anchorValue.dyn_cast<BlockArgument>();
      if (blockArg) {
        os << "BlockArgument(" << blockArg.getArgNumber();
        auto *definingOp = blockArg.getDefiningOp();
        if (definingOp) {
          os << ", " << blockArg.getDefiningOp()->getName();
        }
        os << ")";
      } else {
        os << "{" << anchorValue << "}";
      }
    }
  }
  os << ")";
}

void CastType::print(raw_ostream &os, bool brief) {
  os << "cast(" << *typeIdentifier << ", ";
  typeVar->print(os, true);
  os << ")";
}

void ReadType::print(raw_ostream &os, bool brief) {
  os << "read(";
  type->print(os, true);
  os << ")";
}

void WriteType::print(raw_ostream &os, bool brief) {
  os << "write(";
  type->print(os, true);
  os << ")";
}

void IRValueType::print(raw_ostream &os, bool brief) {
  os << "irtype(" << irType << ")";
}

void ObjectValueType::print(raw_ostream &os, bool brief) {
  os << "object(" << *typeIdentifier;
  bool first = true;
  for (auto it : llvm::zip(getFieldIdentifiers(), getFieldTypes())) {
    if (!first)
      os << ", ";
    else
      first = false;
    os << *std::get<0>(it) << ":";
    auto *ft = std::get<1>(it);
    if (ft)
      ft->print(os, true);
    else
      os << "NULL";
  }
  os << ")";
}

void Constraint::print(raw_ostream &os, bool brief) {
  t1->print(os, brief);
  os << " <: ";
  t2->print(os, brief);
  if (!brief && contextOp) {
    os << "\n      " << *contextOp;
  }
}

void ConstraintSet::print(raw_ostream &os, bool brief) {
  for (auto it : llvm::enumerate(constraints)) {
    os << it.index() << ":  ";
    it.value().print(os, brief);
    os << "\n";
  }
}

void TypeVarSet::print(raw_ostream &os, bool brief) {
  for (auto it : typeVars) {
    os << it.getOrdinal() << ":  ";
    it.print(os, brief);
    os << "\n";
  }
}
