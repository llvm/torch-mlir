//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Torch-specific traits.
//
//===----------------------------------------------------------------------===//


#ifndef NPCOMP_DIALECT_TORCH_IR_TORCHTRAITS_H
#define NPCOMP_DIALECT_TORCH_IR_TORCHTRAITS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace NPCOMP {
namespace Torch {
namespace OpTrait {

// If a Torch op has this trait, it means that the op does not exploit the
// mutability / aliasing properties of torch tensors. This enables a
// transformation which locally replaces mutable arrays with immutable tensors.
template <typename ConcreteType>
class HasValueSemantics
    : public ::mlir::OpTrait::TraitBase<ConcreteType, HasValueSemantics> {};

// If a Torch op has this trait, it means that the op is a "trailing underscore"
// op variant that performs an in-place operation on its first argument. These
// operations can be transformed into their value-semantic analog by removing
// the underscore, such as converting `torch.aten.mul_` to `torch.aten.mul` with
// a few surrounding ops to model the inplace semantics.
template <typename ConcreteType>
class IsTrailingUnderscoreInplaceVariant
    : public ::mlir::OpTrait::TraitBase<ConcreteType,
                                        IsTrailingUnderscoreInplaceVariant> {};

} // namespace OpTrait
} // namespace Torch
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TORCH_IR_TORCHTRAITS_H
