/*===- ir.c - Simple test of C APIs ---------------------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: npcomp-capi-ir-test 2>&1 | FileCheck %s
 */

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "npcomp-c/BasicpyTypes.h"
#include "npcomp-c/NumpyTypes.h"
#include "npcomp-c/Registration.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Dumps an instance of all NPComp types.
static int printStandardTypes(MlirContext ctx) {
  // Bool type.
  MlirType boolType = npcompBasicpyBoolTypeGet(ctx);
  if (!npcompTypeIsABasicpyBool(boolType))
    return 1;
  mlirTypeDump(boolType);
  fprintf(stderr, "\n");

  // Bytes type.
  MlirType bytesType = npcompBasicpyBytesTypeGet(ctx);
  if (!npcompTypeIsABasicpyBytes(bytesType))
    return 1;
  mlirTypeDump(bytesType);
  fprintf(stderr, "\n");

  // Any dtype.
  MlirType anyDtype = npcompAnyDtypeTypeGet(ctx);
  if (!npcompTypeIsANumpyAnyDtype(anyDtype))
    return 2;
  mlirTypeDump(anyDtype);
  fprintf(stderr, "\n");

  // Ranked NdArray.
  int64_t fourDim = 4;
  MlirType rankedNdArray =
      npcompNumpyNdArrayTypeGetRanked(1, &fourDim, boolType);
  if (!npcompTypeIsANumpyNdArray(rankedNdArray))
    return 3;
  mlirTypeDump(rankedNdArray);
  fprintf(stderr, "\n");

  return 0;
}

int main() {
  MlirContext ctx = mlirContextCreate();
  mlirRegisterAllDialects(ctx);
  npcompRegisterAllDialects(ctx);

  // clang-format off
  // CHECK-LABEL: @types
  // CHECK: !basicpy.BoolType
  // CHECK: !basicpy.BytesType
  // CHECK: !numpy.any_dtype
  // CHECK: !numpy.ndarray<[4]:!basicpy.BoolType>
  // CHECK: 0
  // clang-format on
  fprintf(stderr, "@types\n");
  int errcode = printStandardTypes(ctx);
  fprintf(stderr, "%d\n", errcode);

  mlirContextDestroy(ctx);

  return 0;
}
