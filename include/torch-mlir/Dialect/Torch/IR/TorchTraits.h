//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// Torch-specific traits.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCH_IR_TORCHTRAITS_H
#define TORCHMLIR_DIALECT_TORCH_IR_TORCHTRAITS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace torch {
namespace Torch {
namespace OpTrait {

// If a Torch op has this trait, it means that the op does not exploit the
// mutability / aliasing properties of torch tensors, lists, or dictionaries.
// This enables transformations to locally reason about immutability for those
// types.
template <typename ConcreteType>
class HasValueSemantics
    : public ::mlir::OpTrait::TraitBase<ConcreteType, HasValueSemantics> {};

// If a Torch op has this trait, it means that the op does not mutate any
// operands.
//
// This is a weaker form of HasValueSemantics, since that trait also requires no
// aliasing. That is, HasValueSemantics implies this trait.
template <typename ConcreteType>
class ReadOnly : public ::mlir::OpTrait::TraitBase<ConcreteType, ReadOnly> {};

// If a Torch op has this trait, it means that the op is a "trailing underscore"
// op variant that performs an in-place operation on its first argument. These
// operations can be transformed into their value-semantic analog by removing
// the underscore, such as converting `torch.aten.mul_` to `torch.aten.mul` with
// a few surrounding ops to model the inplace semantics.
template <typename ConcreteType>
class IsTrailingUnderscoreInplaceVariant
    : public ::mlir::OpTrait::TraitBase<ConcreteType,
                                        IsTrailingUnderscoreInplaceVariant> {};

// If a Torch op has this trait, it means that the op allows all of its operand
// and result types to be refined. That is, a less specific type is allowed to
// be replaced by a more specific type, according to PEP 483 subtyping rules.
template <typename ConcreteType>
class AllowsTypeRefinement
    : public ::mlir::OpTrait::TraitBase<ConcreteType, AllowsTypeRefinement> {};

// If a Torch op has this trait, it means that the op is allowed to be used
// in the module initializer. Only a small set of ops are permitted in the
// module initializer. These ops are essentially those which can be produced
// by the IValue importer.
template <typename ConcreteType>
class AllowedInModuleInitializer
    : public ::mlir::OpTrait::TraitBase<ConcreteType,
                                        AllowedInModuleInitializer> {};

} // namespace OpTrait
} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_IR_TORCHTRAITS_H
