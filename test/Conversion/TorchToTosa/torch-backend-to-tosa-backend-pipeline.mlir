// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-backend-to-tosa-backend-pipeline)' -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.mul.Scalar$mixed_type(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<5xbf16>) -> tensor<5xbf16> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
// CHECK:           %[[VAL_3:.*]] = tosa.mul %[[VAL_0]], %[[VAL_2]], %[[VAL_1]] : (tensor<5xbf16>, tensor<1xbf16>, tensor<1xi8>) -> tensor<5xbf16>
// CHECK:           return %[[VAL_3]] : tensor<5xbf16>
// CHECK:         }
func.func @torch.aten.mul.Scalar$mixed_type(%arg0: !torch.vtensor<[5],bf16>) -> !torch.vtensor<[5],bf16> {
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.aten.mul.Scalar %arg0, %float2.000000e00 : !torch.vtensor<[5],bf16>, !torch.float -> !torch.vtensor<[5],bf16>
  return %0 : !torch.vtensor<[5],bf16>
}

// -----

// CHECK-LABEL: torch.aten.add.Tensor$mixed_type_fp
// CHECK-SAME: %[[VAL_0:.*]]: tensor<6xbf16>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<6xf32>
// CHECK: %[[VAL_3:.*]] = tosa.cast %[[VAL_1]] : (tensor<6xf32>) -> tensor<6xbf16>
// CHECK: %[[VAL_4:.*]] = tosa.add %[[VAL_0]], %[[VAL_3]] : (tensor<6xbf16>, tensor<6xbf16>) -> tensor<6xbf16>
func.func @torch.aten.add.Tensor$mixed_type_fp(%arg0: !torch.vtensor<[6],bf16>, %arg1: !torch.vtensor<[6],f32>, %arg2: !torch.float) -> !torch.vtensor<[6],bf16> {
  %float1 = torch.constant.float 1.000000e+00
  %0 = torch.aten.add.Tensor %arg0, %arg1, %float1 : !torch.vtensor<[6],bf16>, !torch.vtensor<[6],f32>, !torch.float -> !torch.vtensor<[6],bf16>
  return %0 : !torch.vtensor<[6],bf16>
}

// -----

// CHECK-LABEL: torch.aten.add.Tensor$mixed_type_int
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<5xbf16>
// CHECK: %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<5xbf16>) -> tensor<5xf32>
// CHECK: %[[VAL_3:.*]] = tosa.add %[[VAL_0]], %[[VAL_2]] : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
func.func @torch.aten.add.Tensor$mixed_type_int(%arg0: !torch.vtensor<[5],f32>, %arg1: !torch.vtensor<[5],bf16>) -> !torch.vtensor<[5],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],bf16>, !torch.int -> !torch.vtensor<[5],f32>
  return %0 : !torch.vtensor<[5],f32>
}

// -----

// CHECK-LABEL: torch.aten.Scalar$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x1x32x64xi16>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<256> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
// CHECK: %[[VAL_2:.*]] = tosa.cast %[[VAL_0]] : (tensor<1x1x32x64xi16>) -> tensor<1x1x32x64xi32>
// CHECK: %[[VAL_3:.*]] = tosa.add %[[VAL_2]], %[[VAL_1]] : (tensor<1x1x32x64xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x32x64xi32>
func.func @torch.aten.Scalar$mixed_type(%arg0: !torch.vtensor<[1,1,32,64],si16>) -> !torch.vtensor<[1,1,32,64],si32> {
  %int1 = torch.constant.int 1
  %int256 = torch.constant.int 256
  %0 = torch.aten.add.Scalar %arg0, %int256, %int1 : !torch.vtensor<[1,1,32,64],si16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,64],si32>
  return %0 : !torch.vtensor<[1,1,32,64],si32>
}

// -----

// CHECK-LABEL: torch.aten.sub.Scalar$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<bf16>,
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
// CHECK: %[[VAL_3:.*]] = tosa.sub %[[VAL_0]], %[[VAL_2]] : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
func.func @torch.aten.sub.Scalar$mixed_type(%arg0: !torch.vtensor<[],bf16>, %arg1: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Scalar %arg0, %int1, %int1 : !torch.vtensor<[],bf16>, !torch.int,  !torch.int -> !torch.vtensor<[],bf16>
  return %0 : !torch.vtensor<[],bf16>
}

// -----

// CHECK-LABEL: torch.aten.maximum$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x3x1xi32>,
// CHECK-SAME: %[[VAL_1:.*]]: tensor<1x3x1xf32>
// CHECK: %[[VAL_2:.*]] = tosa.cast %[[VAL_0]] : (tensor<1x3x1xi32>) -> tensor<1x3x1xf32>
// CHECK: %[[VAL_3:.*]] = tosa.maximum %[[VAL_2]], %[[VAL_1]] : (tensor<1x3x1xf32>, tensor<1x3x1xf32>) -> tensor<1x3x1xf32>
func.func @torch.aten.maximum$mixed_type(%arg0: !torch.vtensor<[1,3,1],si32>, %arg1: !torch.vtensor<[1,3,1],f32>) -> !torch.vtensor<[1,3,1],f32> {
  %0 = torch.aten.maximum %arg0, %arg1 : !torch.vtensor<[1,3,1],si32>, !torch.vtensor<[1,3,1],f32> -> !torch.vtensor<[1,3,1],f32>
  return %0 : !torch.vtensor<[1,3,1],f32>
}

// -----

// CHECK-LABEL: torch.aten.bitwise_and.Tensor$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<?x?xi16>,
// CHECK-SAME: %[[VAL_1:.*]]: tensor<?x?xi32>
// CHECK: %[[VAL_2:.*]] = tosa.cast %[[VAL_0]] : (tensor<?x?xi16>) -> tensor<?x?xi32>
// CHECK: %[[VAL_3:.*]] = tosa.bitwise_and %[[VAL_2]], %[[VAL_1]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
func.func @torch.aten.bitwise_and.Tensor$mixed_type(%arg0: !torch.vtensor<[?,?],si16>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
  %0 = torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],si16>, !torch.vtensor<[?,?],si32> -> !torch.vtensor<[?,?],si32>
  return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor$mixed_type_fp(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: tensor<?x?xi32>) -> tensor<?x?xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_3:.*]] = tosa.cast %[[VAL_1]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_3]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_0]], %[[VAL_4]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           return %[[VAL_5]] : tensor<?x?xf32>
// CHECK:         }
func.func @torch.aten.div.Tensor$mixed_type_fp(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],si32>) -> !torch.vtensor<[?, ?],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],si32> -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL: torch.aten.div.Tensor$mixed_type_int
// CHECK-SAME: %[[VAL_0:.*]]: tensor<?x?xi16>,
// CHECK-SAME: %[[VAL_1:.*]]: tensor<?x?xi32>
// CHECK: %[[VAL_2:.*]] = tosa.cast %[[VAL_0]] : (tensor<?x?xi16>) -> tensor<?x?xi32>
// CHECK: %[[VAL_3:.*]] = tosa.int_div %[[VAL_2]], %[[VAL_1]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
func.func @torch.aten.div.Tensor$mixed_type_int(%arg0: !torch.vtensor<[?, ?],si16>, %arg1: !torch.vtensor<[?, ?],si32>) -> !torch.vtensor<[?, ?],si32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],si16>, !torch.vtensor<[?, ?],si32> -> !torch.vtensor<[?, ?],si32>
  return %0 : !torch.vtensor<[?, ?],si32>
}

// -----

// CHECK-LABEL: torch.aten.pow.Tensor$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<?x?xf16>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<3.123400e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK: %[[VAL_2:.*]] = tosa.cast %[[VAL_0]] : (tensor<?x?xf16>) -> tensor<?x?xf32>
// CHECK: %[[VAL_3:.*]] = tosa.pow %[[VAL_2]], %[[VAL_1]] : (tensor<?x?xf32>, tensor<1x1xf32>) -> tensor<?x?xf32>
func.func @torch.aten.pow.Tensor$mixed_type(%arg0: !torch.vtensor<[?,?],f16>) -> !torch.vtensor<[?,?],f32> {
  %fp0 = torch.constant.float 3.123400e+00
  %0 = torch.aten.pow.Tensor_Scalar %arg0, %fp0 : !torch.vtensor<[?,?],f16>, !torch.float -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
func.func @torch.prim.TupleConstruct() {
  %int128 = torch.constant.int 128
  %0 = torch.prim.TupleConstruct %int128 : !torch.int -> !torch.tuple<int>
  // expected-error @below {{failed to legalize operation 'torch.prim.Print' that was explicitly marked illegal}}
  torch.prim.Print(%0) : !torch.tuple<int>
  return
}
