//===- NumpyOps.cpp - Core numpy dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Numpy;

//----------------------------------------------------------------------------//
// Type inference
//----------------------------------------------------------------------------//

/// Adds constraints to relating a unary op that accepts and returns either
/// tensor or ndarray types where the dtype should be the same.
/// Type constraints are added on the dtype, not the outer object type.
static void constrainUnaryDtypeInvariantOp(Typing::CPA::Context &context,
                                           Value source, Value dest,
                                           Operation *op) {
  auto &env = context.getCurrentEnvironment();
  auto *sourceTn =
      llvm::dyn_cast<Typing::CPA::ObjectValueType>(env.mapValueToType(source));
  auto *destTn =
      llvm::dyn_cast<Typing::CPA::ObjectValueType>(env.mapValueToType(dest));
  if (sourceTn && destTn && sourceTn->getFieldCount() == 1 &&
      destTn->getFieldCount() == 1) {
    context.getConstraint(sourceTn->getFieldTypes().front(),
                          destTn->getFieldTypes().front());
  }
}

void CreateArrayFromTensorOp::addCPAConstraints(Typing::CPA::Context &context) {
  constrainUnaryDtypeInvariantOp(context, source(), dest(), *this);
}

void CopyToTensorOp::addCPAConstraints(Typing::CPA::Context &context) {
  constrainUnaryDtypeInvariantOp(context, source(), dest(), *this);
}

void BuiltinUfuncCallOp::addCPAConstraints(Typing::CPA::Context &context) {
  // TODO: This should really be a function call chosen so as to promote
  // arguments. For now, though, we just say that the result is constrained
  // to the inputs. Note that not all ufuncs transfer types like this.
  // We just pretend this is two unary functions that write into the output.
  for (auto input : inputs()) {
    constrainUnaryDtypeInvariantOp(context, input, output(), *this);
  }
}

namespace mlir {
namespace NPCOMP {
namespace Numpy {
#define GET_OP_CLASSES
#include "npcomp/Dialect/Numpy/IR/NumpyOps.cpp.inc"
} // namespace Numpy
} // namespace NPCOMP
} // namespace mlir
