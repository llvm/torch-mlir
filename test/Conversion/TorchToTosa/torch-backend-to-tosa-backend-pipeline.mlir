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
// CHECK: %[[VAL_3:.*]] = tosa.intdiv %[[VAL_2]], %[[VAL_1]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
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

// -----
// CHECK-LABEL: conv_within_dequant_quant
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x4x7x8xi8>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<3x4x3x2xi8>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<3xi32>) -> tensor<2x3x5x7xi32>
// CHECK: tosa.conv2d
// CHECK-SAME: {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x7x8x4xi8>, tensor<3x3x2x4xi8>, tensor<3xi32>, tensor<1xi8>, tensor<1xi8>)
// COM: this test verified that dequant->conv->quant is indeed fused into quant_conv in tosa pipeline
func.func @conv_within_dequant_quant(%arg0: !torch.vtensor<[2,4,7,8],si8>, %arg1: !torch.vtensor<[3,4,3,2],si8>, %arg2: !torch.vtensor<[3],si32>) -> !torch.vtensor<[2,3,5,7],si32> {
    %false = torch.constant.bool false
    %int2147483647 = torch.constant.int 2147483647
    %int-2147483648 = torch.constant.int -2147483648
    %int1000 = torch.constant.int 1000
    %int-1000 = torch.constant.int -1000
    %int0 = torch.constant.int 0
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %int3 = torch.constant.int 3
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int7 = torch.constant.int 7
    %int-128 = torch.constant.int -128
    %int127 = torch.constant.int 127
    %int1 = torch.constant.int 1
    %0 = torch.aten.clamp %arg0, %int-128, %int127 : !torch.vtensor<[2,4,7,8],si8>, !torch.int, !torch.int -> !torch.vtensor<[2,4,7,8],si8>
    %1 = torch.aten._make_per_tensor_quantized_tensor %0, %float1.000000e-02, %int7 : !torch.vtensor<[2,4,7,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[2,4,7,8],!torch.qint8>
    %2 = torch.aten.dequantize.tensor %1 : !torch.vtensor<[2,4,7,8],!torch.qint8> -> !torch.vtensor<[2,4,7,8],f32>
    %3 = torch.aten.clamp %arg1, %int-128, %int127 : !torch.vtensor<[3,4,3,2],si8>, !torch.int, !torch.int -> !torch.vtensor<[3,4,3,2],si8>
    %4 = torch.aten._make_per_tensor_quantized_tensor %3, %float1.000000e-02, %int3 : !torch.vtensor<[3,4,3,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,4,3,2],!torch.qint8>
    %5 = torch.aten.dequantize.tensor %4 : !torch.vtensor<[3,4,3,2],!torch.qint8> -> !torch.vtensor<[3,4,3,2],f32>
    %6 = torch.aten.clamp %arg2, %int-1000, %int1000 : !torch.vtensor<[3],si32>, !torch.int, !torch.int -> !torch.vtensor<[3],si32>
    %7 = torch.aten._make_per_tensor_quantized_tensor %6, %float1.000000e00, %int0 : !torch.vtensor<[3],si32>, !torch.float, !torch.int -> !torch.vtensor<[3],!torch.qint32>
    %8 = torch.aten.dequantize.tensor %7 : !torch.vtensor<[3],!torch.qint32> -> !torch.vtensor<[3],f32>
    %9 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %10 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %11 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %13 = torch.aten.convolution %2, %5, %8, %9, %10, %11, %false, %12, %int1 : !torch.vtensor<[2,4,7,8],f32>, !torch.vtensor<[3,4,3,2],f32>, !torch.vtensor<[3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[2,3,5,7],f32>
    %14 = torch.aten.quantize_per_tensor %13, %float1.000000e00, %int0, %int3 : !torch.vtensor<[2,3,5,7],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[2,3,5,7],!torch.qint32>
    %15 = torch.aten.int_repr %14 : !torch.vtensor<[2,3,5,7],!torch.qint32> -> !torch.vtensor<[2,3,5,7],si32>
    %16 = torch.aten.clamp %15, %int-2147483648, %int2147483647 : !torch.vtensor<[2,3,5,7],si32>, !torch.int, !torch.int -> !torch.vtensor<[2,3,5,7],si32>
    return %16 : !torch.vtensor<[2,3,5,7],si32>
  }
