// RUN: torch-mlir-opt %s -convert-torch-to-tosa -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.convolution$3d_basic(
// CHECK-SAME:     %[[ARG:.*]]: !torch.vtensor<[2,3,5,6,7],f32>) -> !torch.vtensor<[2,4,5,6,7],f32> {
// CHECK:         %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[2,3,5,6,7],f32> -> tensor<2x3x5x6x7xf32>
// CHECK:         %[[USE_BIAS:.*]] = torch.constant.bool false
// CHECK:         %[[ONE:.*]] = torch.constant.int 1
// CHECK:         %[[WEIGHT:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_4_3_3_3_3_torch.float32> : tensor<4x3x3x3x3xf32>}> : () -> tensor<4x3x3x3x3xf32>
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[STRIDE:.*]] = torch.prim.ListConstruct %[[ONE]], %[[ONE]], %[[ONE]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[PADDING:.*]] = torch.prim.ListConstruct %[[ONE]], %[[ONE]], %[[ONE]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[DILATION:.*]] = torch.prim.ListConstruct %[[ONE]], %[[ONE]], %[[ONE]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[OUTPUT_PAD:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:         %[[NHWC_INPUT:.*]] = tosa.transpose %[[INPUT]] {perms = array<i32: 0, 2, 3, 4, 1>} : (tensor<2x3x5x6x7xf32>) -> tensor<2x5x6x7x3xf32>
// CHECK:         %[[NHWC_WEIGHT:.*]] = tosa.transpose %[[WEIGHT]] {perms = array<i32: 0, 2, 3, 4, 1>} : (tensor<4x3x3x3x3xf32>) -> tensor<4x3x3x3x3xf32>
// CHECK:         %[[INPUT_ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:         %[[WEIGHT_ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:         %[[BIAS_CONST:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:         %[[CONV:.*]] = tosa.conv3d %[[NHWC_INPUT]], %[[NHWC_WEIGHT]], %[[BIAS_CONST]], %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 1, 1, 1, 1, 1, 1>, stride = array<i64: 1, 1, 1>} : (tensor<2x5x6x7x3xf32>, tensor<4x3x3x3x3xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x5x6x7x4xf32>
// CHECK:         %[[RESULT_NCHW:.*]] = tosa.transpose %[[CONV]] {perms = array<i32: 0, 4, 1, 2, 3>} : (tensor<2x5x6x7x4xf32>) -> tensor<2x4x5x6x7xf32>
// CHECK:         %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_NCHW]] : tensor<2x4x5x6x7xf32> -> !torch.vtensor<[2,4,5,6,7],f32>
// CHECK:         return %[[RESULT]] : !torch.vtensor<[2,4,5,6,7],f32>
func.func @torch.aten.convolution$3d_basic(%arg0: !torch.vtensor<[2,3,5,6,7],f32>) -> !torch.vtensor<[2,4,5,6,7],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %weight = torch.vtensor.literal(dense_resource<torch_tensor_4_3_3_3_3_torch.float32> : tensor<4x3x3x3x3xf32>) : !torch.vtensor<[4,3,3,3,3],f32>
  %none = torch.constant.none
  %stride = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %output_padding = torch.prim.ListConstruct  : () -> !torch.list<int>
  %result = torch.aten.convolution %arg0, %weight, %none, %stride, %padding, %dilation, %false, %output_padding, %int1 : !torch.vtensor<[2,3,5,6,7],f32>, !torch.vtensor<[4,3,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[2,4,5,6,7],f32>
  return %result : !torch.vtensor<[2,4,5,6,7],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.convolution$3d_transpose(
// CHECK-SAME:     %[[ARG:.*]]: !torch.vtensor<[1,1,2,2,2],f32>) -> !torch.vtensor<[1,1,4,4,4],f32> {
// CHECK:         %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[1,1,2,2,2],f32> -> tensor<1x1x2x2x2xf32>
// CHECK:         %[[WEIGHT:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x3x3x3xf32>}> : () -> tensor<1x1x3x3x3xf32>
// CHECK:         %[[BIAS:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:         %[[NHWC_INPUT:.*]] = tosa.transpose %[[INPUT]] {perms = array<i32: 0, 2, 3, 4, 1>} : (tensor<1x1x2x2x2xf32>) -> tensor<1x2x2x2x1xf32>
// CHECK:         %[[NHWC_WEIGHT:.*]] = tosa.transpose %[[WEIGHT]] {perms = array<i32: 1, 2, 3, 4, 0>} : (tensor<1x1x3x3x3xf32>) -> tensor<1x3x3x3x1xf32>
// CHECK:         %[[REV_D:.*]] = tosa.reverse %[[NHWC_WEIGHT]] {axis = 1 : i32}
// CHECK:         %[[REV_H:.*]] = tosa.reverse %[[REV_D]] {axis = 2 : i32}
// CHECK:         %[[REV_W:.*]] = tosa.reverse %[[REV_H]] {axis = 3 : i32}
// CHECK:         %[[INPUT_ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:         %[[WEIGHT_ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:         %[[CONV:.*]] = tosa.conv3d {{.*}}, %[[REV_W]], {{.*}}, {{.*}}, {{.*}} {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x6x6x6x1xf32>, tensor<1x3x3x3x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x4x1xf32>
// CHECK:         %[[RESULT_NCHW:.*]] = tosa.transpose %[[CONV]] {perms = array<i32: 0, 4, 1, 2, 3>} : (tensor<1x4x4x4x1xf32>) -> tensor<1x1x4x4x4xf32>
// CHECK:         %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_NCHW]] : tensor<1x1x4x4x4xf32> -> !torch.vtensor<[1,1,4,4,4],f32>
// CHECK:         return %[[RESULT]] : !torch.vtensor<[1,1,4,4,4],f32>
// CHECK:       }
func.func @torch.aten.convolution$3d_transpose(%arg0: !torch.vtensor<[1,1,2,2,2],f32>) -> !torch.vtensor<[1,1,4,4,4],f32> {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %weight = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x1x3x3x3xf32>) : !torch.vtensor<[1,1,3,3,3],f32>
  %bias = torch.vtensor.literal(dense<0.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
  %stride = torch.prim.ListConstruct %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %out_padding = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %groups = torch.constant.int 1
  %result = torch.aten.convolution %arg0, %weight, %bias, %stride, %padding, %dilation, %true, %out_padding, %groups : !torch.vtensor<[1,1,2,2,2],f32>, !torch.vtensor<[1,1,3,3,3],f32>, !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,4,4,4],f32>
  return %result : !torch.vtensor<[1,1,4,4,4],f32>
}
