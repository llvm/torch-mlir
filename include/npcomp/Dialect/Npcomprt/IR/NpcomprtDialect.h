//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NPCOMPRT_IR_NPCOMPRTDIALECT_H
#define NPCOMP_DIALECT_NPCOMPRT_IR_NPCOMPRTDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace NPCOMP {
namespace npcomprt {

class TensorType : public Type::TypeBase<TensorType, Type, TypeStorage> {
public:
  using Base::Base;

  static TensorType get(MLIRContext *context) { return Base::get(context); }
};

#include "npcomp/Dialect/Npcomprt/IR/NpcomprtOpsDialect.h.inc"

} // namespace npcomprt
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_NPCOMPRT_IR_NPCOMPRTDIALECT_H
