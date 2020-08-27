//===- ATenDialect.h --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_DIALECT_H
#define NPCOMP_DIALECT_ATEN_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <map>

namespace mlir {
namespace NPCOMP {
namespace aten {

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace detail {
struct ATenListTypeStorage;
}

/// A variadic list of arguments in ATen
class ATenListType : public mlir::Type::TypeBase<ATenListType, mlir::Type,
                                                 detail::ATenListTypeStorage> {
public:
  using Base::Base;

  /// Return the type of individual elements in the array.
  mlir::Type getElementType();

  /// Get the unique instance of this Type from the context.
  static ATenListType get(mlir::Type elementType);
};

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace {

// Return the tensor volume (i.e., the number of elements) of the given shaped
// type.  If the type does not have a rank, return 1.  If the type doesn't
// have a static shape, return 0.
uint64_t getTensorVolume(const ShapedType ty) {
  if (!ty.hasRank())
    return 1;

  if (!ty.hasStaticShape())
    return 0;

  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

// Return the tensor volume (i.e., the number of elements) of the given type.
// If the type doesn't have a shape, return 1.  If the type is shaped, but
// does not have a rank, return 1.  If the type is shaped, but doesn't have a
// static shape, return 0.
uint64_t getTensorVolume(const Type ty) {
  if (auto t = ty.dyn_cast<ShapedType>()) {
    return getTensorVolume(t);
  } else {
    return 1;
  }
}

} // namespace
} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#include "npcomp/Dialect/ATen/ATenOpInterfaces.h"

namespace mlir {
namespace NPCOMP {
namespace aten {
// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "npcomp/Dialect/ATen/ATen.h.inc"

#include "npcomp/Dialect/ATen/ATenDialect.h.inc"

} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#endif
