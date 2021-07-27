/*===-- npcomp-c/InitLLVM.h - C API for initializing LLVM  --------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef NPCOMP_C_INITLLVM_H
#define NPCOMP_C_INITLLVM_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Initializes LLVM codegen infrastructure and related MLIR bridge components.
 */
MLIR_CAPI_EXPORTED void npcompInitializeLLVMCodegen();

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_INITLLVM_H
