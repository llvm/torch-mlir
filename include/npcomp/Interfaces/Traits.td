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

// Empty for now. Kept as boilerplate placeholder.

#endif // NPCOMP_INTERFACES_TRAITS
