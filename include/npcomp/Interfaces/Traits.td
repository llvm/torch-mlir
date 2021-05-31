//===-------------------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Traits for cross-cutting concerns in the npcomp project.
//
//===----------------------------------------------------------------------===//
#ifndef NPCOMP_INTERFACES_TRAITS
#define NPCOMP_INTERFACES_TRAITS

include "mlir/IR/OpBase.td"

class NpcompOpTrait<string name> : OpTrait, NativeTrait<"", ""> {
  let trait = name;
  let cppNamespace = "::mlir::NPCOMP::OpTrait";
}

// Op allows operand and result types to be refined.
// For example a `tensor<?xf32>` can be refined to `tensor<4xf32>`.
//
// TODO: Implement RefinableTypeInterface that allows actually modeling
// which types are refinements of other types.
// See the design in:
// https://llvm.discourse.group/t/allow-shape-concretization-or-type-concretization-in-rewrites/3327/3
def AllowsTypeRefinement : NpcompOpTrait<"AllowsTypeRefinement">;

#endif // NPCOMP_INTERFACES_TRAITS
