//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_INTERFACES_TRAITS_H
#define NPCOMP_INTERFACES_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace NPCOMP {
namespace OpTrait {

template <typename ConcreteType>
class AllowsTypeRefinement
    : public ::mlir::OpTrait::TraitBase<ConcreteType, AllowsTypeRefinement> {};

} // namespace OpTrait

// Check if an operation has the AllowsTypeRefinement trait.
//
// This function should be used in preference to
// `op->hasTrait<AllowsTypeRefinement>()` because this function has knowledge of
// some upstream ops that have this property, but which we cannot annotate with
// this trait.
bool allowsTypeRefinement(Operation *op);

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_INTERFACES_TRAITS_H
