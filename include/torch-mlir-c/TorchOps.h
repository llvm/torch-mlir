//===-- torch-mlir-c/TorchOps.h - C API for torch ops -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_C_TORCHOPS_H
#define TORCHMLIR_C_TORCHOPS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

/// Adjusts the static information in the type of `value` to `desiredType`.
///
/// Returns null if such an adjustment is not possible.
///
/// If `userAllowsRefinement` is true, then the original value will be returned
/// if it is a subtype of `desiredType`.
MLIR_CAPI_EXPORTED MlirValue torchMlirAdjustStaticInformation(
    MlirBlock block, MlirOperation insertBefore, MlirValue value,
    MlirType desiredType, bool userAllowsRefinement);

#ifdef __cplusplus
}
#endif

#endif // TORCHMLIR_C_TORCHOPS_H
