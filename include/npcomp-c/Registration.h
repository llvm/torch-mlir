/*===-- npcomp-c/Registration.h - Registration functions  ---------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef NPCOMP_C_REGISTRATION_H
#define NPCOMP_C_REGISTRATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all NPComp dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
void npcompRegisterAllDialects(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_REGISTRATION_H
