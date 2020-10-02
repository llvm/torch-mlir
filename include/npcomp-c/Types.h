/*===-- npcomp-c/Types.h - NPComp custom types --------------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef NPCOMP_C_TYPES_H
#define NPCOMP_C_TYPES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================*/
/* Bool type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the Python "bool" type. */
int npcompTypeIsABool(MlirType t);

/** Gets the Python "bool" type. */
MlirType npcompBoolTypeGet(MlirContext context);

/*============================================================================*/
/* Any dtype type.                                                            */
/*============================================================================*/

/** Checks whether the given type is the special "any dtype" type that is used
 * to signal an NDArray or tensor of unknown type. */
int npcompTypeIsAAnyDtype(MlirType t);

/** Gets the "any dtype" type. */
MlirType npcompAnyDtypeTypeGet(MlirContext context);

/*============================================================================*/
/* NDArray type.                                                              */
/*============================================================================*/

/** Checks whether the given type is an NdArray type. */
int npcompTypeIsANdArray(MlirType t);

/** Gets a numpy.NdArray type that is ranked. Any dimensions that are -1 are
 * unknown. */
MlirType npcompNdArrayTypeGetRanked(intptr_t rank, const int64_t *shape,
                                    MlirType elementType);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_TYPES_H
