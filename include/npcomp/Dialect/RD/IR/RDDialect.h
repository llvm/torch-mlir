//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_RD_IR_RDDIALECT_H
#define NPCOMP_DIALECT_RD_IR_RDDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace NPCOMP {
namespace rd {

namespace detail {
struct SlotObjectTypeStorage;
} // namespace detail

/// An immutable collection of values.
class DatasetType : public Type::TypeBase<DatasetType, Type, TypeStorage> {
public:
    using Base::Base;
    static DatasetType get(MLIRContext *context) { return Base::get(context); }
};

class IteratorType : public Type::TypeBase<IteratorType, Type, TypeStorage> {
public:
  using Base::Base;
  static IteratorType get(MLIRContext *context) { return Base::get(context); }
};

} // namespace rd
} // namespace NPCOMP
} // namespace mlir

#include "npcomp/Dialect/RD/IR/RDOpsDialect.h.inc"

#endif // NPCOMP_DIALECT_RD_IR_RDDIALECT_H
