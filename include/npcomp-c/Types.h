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
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================*/
/* Any dtype type.                                                            */
/*============================================================================*/

/** Checks whether the given type is the special "any dtype" type that is used
 * to signal an NDArray or tensor of unknown type. */
int npcompTypeIsAAnyDtype(MlirType t);

/** Gets the "any dtype" type. */
MlirType npcompAnyDtypeTypeGet(MlirContext context);

/*============================================================================*/
/* Bool type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the Python "bool" type. */
int npcompTypeIsABool(MlirType t);

/** Gets the Python "bool" type. */
MlirType npcompBoolTypeGet(MlirContext context);

/*============================================================================*/
/* Dict type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the Python "dict" type. */
int npcompTypeIsADict(MlirType t);

/** Gets the generic Python "dict" type. */
MlirType npcompDictTypeGet(MlirContext context);

/*============================================================================*/
/* List type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the Python "list" type. */
int npcompTypeIsAList(MlirType t);

/** Gets the generic Python "list" type. */
MlirType npcompListTypeGet(MlirContext context);

/*============================================================================*/
/* NDArray type.                                                              */
/*============================================================================*/

/** Checks whether the given type is an NdArray type. */
int npcompTypeIsANdArray(MlirType t);

/** Gets a numpy.NdArray type that is unranked. */
MlirType npcompNdArrayTypeGetUnranked(MlirType elementType);

/** Gets a numpy.NdArray type that is ranked. Any dimensions that are -1 are
 * unknown. */
MlirType npcompNdArrayTypeGetRanked(intptr_t rank, const int64_t *shape,
                                    MlirType elementType);

/** Helper that gets an equivalent NdArrayType from a ShapedType. */
MlirType npcompNdArrayTypeGetFromShaped(MlirType shapedType);

/** Gets the element type of an NdArray type. */
MlirType npcompNdArrayTypeGetDType(MlirType t);

/** Checks whether the given shaped type is ranked. */
int npcompNdArrayTypeHasRank(MlirType t);

/** Returns the rank of the given ranked shaped type. */
int64_t npcompNdArrayTypeGetRank(MlirType t);

/** Returns the dim-th dimension of the given ranked shaped type.
 * Returns -1 if dynamic. */
int64_t npcompNdArrayTypeGetDimSize(MlirType t, intptr_t dim);

/*============================================================================*/
/* None type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the type of the singleton 'None' value. */
int npcompTypeIsANone(MlirType t);

/** Gets the type of the singleton 'None'. */
MlirType npcompNoneTypeGet(MlirContext context);

/*============================================================================*/
/* Tuple type.                                                                */
/*============================================================================*/

/** Checks whether the given type is the special "any dtype" type that is used
 * to signal an NDArray or tensor of unknown type. */
int npcompTypeIsATuple(MlirType t);

/** Gets the generic Python "tuple" type. */
MlirType npcompTupleTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_TYPES_H
