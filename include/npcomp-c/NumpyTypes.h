//===-- npcomp-c/NumpyTypes.h - C API for numpy types -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_C_NUMPYTYPES_H
#define NPCOMP_C_NUMPYTYPES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// !numpy.any_dtype type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the special "any dtype" type that is used
// to signal an NDArray or tensor of unknown type.
bool npcompTypeIsANumpyAnyDtype(MlirType t);

/// Gets the "any dtype" type.
MlirType npcompAnyDtypeTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// NDArray type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is an NdArray type.
bool npcompTypeIsANumpyNdArray(MlirType t);

/// Gets a numpy.NdArray type that is unranked.
MlirType npcompNumpyNdArrayTypeGetUnranked(MlirType elementType);

/// Gets a numpy.NdArray type that is ranked. Any dimensions that are -1 are
/// unknown.
MlirType npcompNumpyNdArrayTypeGetRanked(intptr_t rank, const int64_t *shape,
                                         MlirType elementType);

/// Helper that gets an equivalent NdArrayType from a ShapedType.
MlirType npcompNumpyNdArrayTypeGetFromShaped(MlirType shapedType);

/// Helper that converts an NdArrayType to a TensorType.
MlirType npcompNumpyNdArrayTypeToTensor(MlirType ndarrayType);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_NUMPYTYPES_H
