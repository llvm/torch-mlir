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
#include "npcomp/Dialect/Common.h"

namespace mlir {
namespace NPCOMP {
namespace npcomp_rt {

namespace NpcompRtTypes {
enum Kind { BufferViewType = TypeRanges::NpcompRt };
} // namespace NpcompRtTypes

class BufferViewType
    : public Type::TypeBase<BufferViewType, Type, TypeStorage> {
public:
  using Base::Base;

  static BufferViewType get(MLIRContext *context) {
    return Base::get(context, NpcompRtTypes::Kind::BufferViewType);
  }

  static bool kindof(unsigned kind) {
    return kind == NpcompRtTypes::Kind::BufferViewType;
  }
};

#include "npcomp/Dialect/NpcompRt/IR/NpcompRtOpsDialect.h.inc"

} // namespace npcomp_rt
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_NPCOMPRT_IR_NPCOMPRTDIALECT_H
