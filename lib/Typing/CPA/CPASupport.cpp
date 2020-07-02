//===- CPASupport.h - Support types and utilities for CPA -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Typing/CPA/CPASupport.h"

#include "mlir/IR/Operation.h"

using namespace mlir::npcomp::typing::CPA;

ObjectBase::~ObjectBase() = default;

void Identifier::print(raw_ostream &os, bool brief) {
  os << "'" << value << "'";
}

void TypeVar::print(raw_ostream &os, bool brief) {
  os << "TypeVar(" << ordinal;
  if (!brief) {
    os << ", ";
    auto blockArg = anchor.dyn_cast<BlockArgument>();
    if (blockArg) {
      os << "BlockArgument(" << blockArg.getArgNumber();
      auto *definingOp = blockArg.getDefiningOp();
      if (definingOp) {
        os << ", " << blockArg.getDefiningOp()->getName();
      }
      os << ")";
    } else {
      os << "{" << anchor << "}";
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
