//===- torch.c - Test of Torch dialect C API ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: torch-mlir-capi-torch-test 2>&1 | FileCheck %s

#include "mlir-c/BuiltinTypes.h"
#include "torch-mlir-c/Registration.h"
#include "torch-mlir-c/TorchTypes.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static void printToStderr(MlirStringRef str, void *userData) {
  (void)userData;
  fwrite(str.data, 1, str.length, stderr);
}

static void testTensor(MlirContext ctx, intptr_t numSizes, int64_t *sizes,
                       MlirType dType, const char *testName) {
#define DEFINE_CHECK(TTT)                                                      \
  MlirType TTT##Type =                                                         \
      torchMlirTorch##TTT##TypeGet(ctx, numSizes, sizes, dType);               \
                                                                               \
  bool TTT##hasSizes = torchMlirTorch##TTT##TypeHasSizes(TTT##Type);           \
  fprintf(stderr, #TTT "Type %s hasSizes: %d\n", testName, TTT##hasSizes);     \
  bool TTT##hasDtype = torchMlirTorch##TTT##TypeHasDtype(TTT##Type);           \
  fprintf(stderr, #TTT "Type %s hasDtype: %d\n", testName, TTT##hasDtype);     \
  if (TTT##hasSizes) {                                                         \
    fprintf(stderr, #TTT "Type %s rank: %zu\n", testName,                      \
            torchMlirTorch##TTT##TypeGetRank(TTT##Type));                      \
    int64_t *TTT##Sizes = malloc(sizeof(int64_t) * numSizes);                  \
    torchMlirTorch##TTT##TypeGetSizes(TTT##Type, TTT##Sizes);                  \
    for (int i = 0; i < numSizes; ++i) {                                       \
      fprintf(stderr, #TTT "Type %s pos %d size: %ld\n", testName, i,          \
              TTT##Sizes[i]);                                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  if (TTT##hasDtype) {                                                         \
    MlirType TTT##Dtype = torchMlirTorch##TTT##TypeGetDtype(TTT##Type);        \
    fprintf(stderr, #TTT "Type %s dtype: ", testName);                         \
    mlirTypePrint(TTT##Dtype, printToStderr, NULL);                            \
    fprintf(stderr, "\n");                                                     \
  }
  DEFINE_CHECK(NonValueTensor)
  DEFINE_CHECK(ValueTensor)
#undef DEFINE_CHECK
}

// CHECK-LABEL: testTypeMetaDataAccessors
static void testTypeMetaDataAccessors(MlirContext ctx) {
  fprintf(stderr, "testTypeMetaDataAccessors\n");

  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType optionalI8 = torchMlirTorchOptionalTypeGet(i8);

  fprintf(stderr, "optionalI8 isa TorchOptional: %d\n",
          torchMlirTypeIsATorchOptional(optionalI8));
  // CHECK: optionalI8 isa TorchOptional: 1

  MlirType containedType = torchMlirTorchOptionalTypeGetContained(optionalI8);
  fprintf(stderr, "optionalI8 containedType: ");
  mlirTypePrint(containedType, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: optionalI8 containedType: i8

  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType f32 = mlirF32TypeGet(ctx);
  MlirType _tupleI8[3] = {i8, f16, f32};
#define DEFINE_CHECK(TTT)                                                      \
  MlirType TTT##I8 = torchMlirTorch##TTT##TypeGet(ctx, 3, _tupleI8);           \
                                                                               \
  fprintf(stderr, #TTT "I8 isa " #TTT ": %d\n",                                \
          torchMlirTypeIsATorch##TTT(TTT##I8));                                \
                                                                               \
  fprintf(stderr, #TTT "I8 NumTypes: %zu\n",                                   \
          torchMlirTorch##TTT##TypeGetNumTypes(TTT##I8));                      \
                                                                               \
  for (int i = 0; i < 3; ++i) {                                                \
    fprintf(stderr, #TTT "I8 pos %d type: ", i);                               \
    mlirTypePrint(torchMlirTorch##TTT##TypeGetType(TTT##I8, i), printToStderr, \
                  NULL);                                                       \
    fprintf(stderr, "\n");                                                     \
  }
  DEFINE_CHECK(Tuple)
  DEFINE_CHECK(Union)
#undef DEFINE_CHECK
  // CHECK: TupleI8 isa Tuple: 1
  // CHECK: TupleI8 NumTypes: 3
  // CHECK: TupleI8 pos 0 type: i8
  // CHECK: TupleI8 pos 1 type: f16
  // CHECK: TupleI8 pos 2 type: f32
  // CHECK: UnionI8 isa Union: 1
  // CHECK: UnionI8 NumTypes: 3
  // CHECK: UnionI8 pos 0 type: i8
  // CHECK: UnionI8 pos 1 type: f16
  // CHECK: UnionI8 pos 2 type: f32

  int64_t sizes[3] = {1, 2, 3};
  testTensor(ctx, 3, sizes, f32, "has-sizes-dtype");
  // CHECK: NonValueTensorType has-sizes-dtype hasSizes: 1
  // CHECK: NonValueTensorType has-sizes-dtype hasDtype: 1
  // CHECK: NonValueTensorType has-sizes-dtype rank: 3
  // CHECK: NonValueTensorType has-sizes-dtype pos 0 size: 1
  // CHECK: NonValueTensorType has-sizes-dtype pos 1 size: 2
  // CHECK: NonValueTensorType has-sizes-dtype pos 2 size: 3
  // CHECK: NonValueTensorType has-sizes-dtype dtype: f32
  // CHECK: ValueTensorType has-sizes-dtype hasSizes: 1
  // CHECK: ValueTensorType has-sizes-dtype hasDtype: 1
  // CHECK: ValueTensorType has-sizes-dtype rank: 3
  // CHECK: ValueTensorType has-sizes-dtype pos 0 size: 1
  // CHECK: ValueTensorType has-sizes-dtype pos 1 size: 2
  // CHECK: ValueTensorType has-sizes-dtype pos 2 size: 3
  // CHECK: ValueTensorType has-sizes-dtype dtype: f32

  MlirType nullType = {NULL};
  testTensor(ctx, 3, sizes, nullType, "has-sizes-no-dtype");
  // CHECK: NonValueTensorType has-sizes-no-dtype hasSizes: 1
  // CHECK: NonValueTensorType has-sizes-no-dtype hasDtype: 0
  // CHECK: NonValueTensorType has-sizes-no-dtype rank: 3
  // CHECK: NonValueTensorType has-sizes-no-dtype pos 0 size: 1
  // CHECK: NonValueTensorType has-sizes-no-dtype pos 1 size: 2
  // CHECK: NonValueTensorType has-sizes-no-dtype pos 2 size: 3
  // CHECK: ValueTensorType has-sizes-no-dtype hasSizes: 1
  // CHECK: ValueTensorType has-sizes-no-dtype hasDtype: 0
  // CHECK: ValueTensorType has-sizes-no-dtype rank: 3
  // CHECK: ValueTensorType has-sizes-no-dtype pos 0 size: 1
  // CHECK: ValueTensorType has-sizes-no-dtype pos 1 size: 2
  // CHECK: ValueTensorType has-sizes-no-dtype pos 2 size: 3
  testTensor(ctx, -1, sizes, f32, "no-sizes-has-dtype");
  // CHECK: NonValueTensorType no-sizes-has-dtype hasSizes: 0
  // CHECK: NonValueTensorType no-sizes-has-dtype hasDtype: 1
  // CHECK: NonValueTensorType no-sizes-has-dtype dtype: f32
  // CHECK: ValueTensorType no-sizes-has-dtype hasSizes: 0
  // CHECK: ValueTensorType no-sizes-has-dtype hasDtype: 1
  // CHECK: ValueTensorType no-sizes-has-dtype dtype: f32

  MlirType floatType = torchMlirTorchFloatTypeGet(ctx);
  torchMlirTorchDictTypeGetChecked(ctx, f16, floatType);
  // CHECK: error: invalid 'f16' for !torch.dict key type
  torchMlirTorchDictTypeGetChecked(ctx, i8, floatType);
  // CHECK: error: invalid 'i8' for !torch.dict key type
  torchMlirTorchDictTypeGetChecked(ctx, floatType, f16);
  // CHECK: error: invalid 'f16' for !torch.dict value type
  torchMlirTorchDictTypeGetChecked(ctx, floatType, i8);
  // CHECK: error: invalid 'i8' for !torch.dict value type

  MlirType strType = torchMlirTorchStringTypeGet(ctx);

  MlirType dictType1 = torchMlirTorchDictTypeGet(strType, floatType);

  fprintf(stderr, "dict keyType: ");
  mlirTypePrint(torchMlirTorchDictTypeGetKeyType(dictType1), printToStderr,
                NULL);
  fprintf(stderr, "\n");
  // CHECK: dict keyType: !torch.str
  fprintf(stderr, "dict valueType: ");
  mlirTypePrint(torchMlirTorchDictTypeGetValueType(dictType1), printToStderr,
                NULL);
  fprintf(stderr, "\n");
  // CHECK: dict valueType: !torch.float

  MlirType dictType2 = torchMlirTorchDictTypeGet(floatType, strType);

  fprintf(stderr, "dict keyType: ");
  mlirTypePrint(torchMlirTorchDictTypeGetKeyType(dictType2), printToStderr,
                NULL);
  fprintf(stderr, "\n");
  // CHECK: dict keyType: !torch.float
  fprintf(stderr, "dict valueType: ");
  mlirTypePrint(torchMlirTorchDictTypeGetValueType(dictType2), printToStderr,
                NULL);
  fprintf(stderr, "\n");
  // CHECK: dict valueType: !torch.str
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  torchMlirRegisterAllDialects(ctx);
  testTypeMetaDataAccessors(ctx);
  mlirContextDestroy(ctx);
  return EXIT_SUCCESS;
}
