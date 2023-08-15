// RUN: torch-mlir-opt <%s -convert-torch-to-tosa -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.convolution(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, 
// CHECK-SAME:                                 %[[ARG_1:.*]]: !torch.vtensor<[16,8,3,3],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[16,8,3,3],f32> -> tensor<16x8x3x3xf32>
// CHECK:           %[[T_2:.*]] = torch.constant.bool false
// CHECK:           %[[T_3:.*]] = torch.constant.none
// CHECK:           %[[T_4:.*]] = torch.constant.int 0
// CHECK:           %[[T_5:.*]] = torch.constant.int 1
// CHECK:           %[[T_6:.*]] = torch_c.to_i64 %[[T_5]]
// CHECK:           %[[T_7:.*]] = torch.prim.ListConstruct %[[T_5]], %[[T_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_8:.*]] = torch.prim.ListConstruct %[[T_4]], %[[T_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_9:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
// CHECK:           %[[T_10:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[T_11:.*]] = "tosa.transpose"(%[[T_0]], %[[T_10]]) : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_12:.*]] = "tosa.transpose"(%[[T_1]], %[[T_10]]) : (tensor<16x8x3x3xf32>, tensor<4xi32>) -> tensor<16x3x3x8xf32>
// CHECK:           %[[T_13:.*]] = "tosa.conv2d"(%[[T_11]], %[[T_12]], %[[T_9]]) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<?x?x?x?xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<?x?x?x16xf32>
// CHECK:           %[[T_14:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[T_15:.*]] = "tosa.transpose"(%[[T_13]], %[[T_14]]) : (tensor<?x?x?x16xf32>, tensor<4xi32>) -> tensor<?x16x?x?xf32>
// CHECK:           %[[T_16:.*]] = tensor.cast %[[T_15]] : tensor<?x16x?x?xf32> to tensor<?x?x?x?xf32>
// CHECK:           %[[T_17:.*]] = torch_c.from_builtin_tensor %[[T_16]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[T_17]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.convolution(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[16,8,3,3],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
%false = torch.constant.bool false
%none = torch.constant.none
%int0 = torch.constant.int 0
%int1 = torch.constant.int 1
%1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
%2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
%3 = torch.prim.ListConstruct  : () -> !torch.list<int>
%4 = torch.aten.convolution %arg0, %arg1, %none, %1, %2, %1, %false, %3, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[16,8,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$depthwise(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, 
// CHECK-SAME:                                 %[[ARG_1:.*]]: !torch.vtensor<[16,1,3,3],f32>) -> !torch.vtensor<[?,16,?,?],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[16,1,3,3],f32> -> tensor<16x1x3x3xf32>
// CHECK:           %[[T_2:.*]] = torch.constant.bool false
// CHECK:           %[[T_3:.*]] = torch.constant.none
// CHECK:           %[[T_4:.*]] = torch.constant.int 0
// CHECK:           %[[T_5:.*]] = torch.constant.int 1
// CHECK:           %[[T_6:.*]] = torch.constant.int 8
// CHECK:           %[[T_7:.*]] = torch.prim.ListConstruct %[[T_5]], %[[T_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_8:.*]] = torch.prim.ListConstruct %[[T_4]], %[[T_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_9:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
// CHECK:           %[[T_10:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[T_11:.*]] = "tosa.transpose"(%[[T_0]], %[[T_10]]) : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_12:.*]] = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[T_13:.*]] = "tosa.transpose"(%[[T_1]], %[[T_12]]) : (tensor<16x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x16x1xf32>
// CHECK:           %[[T_14:.*]] = "tosa.reshape"(%[[T_13]]) <{new_shape = array<i64: 3, 3, 8, 2>}> : (tensor<3x3x16x1xf32>) -> tensor<3x3x8x2xf32>
// CHECK:           %[[T_15:.*]] = "tosa.depthwise_conv2d"(%[[T_11]], %[[T_14]], %[[T_9]]) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<?x?x?x?xf32>, tensor<3x3x8x2xf32>, tensor<16xf32>) -> tensor<?x?x?x16xf32>
// CHECK:           %[[T_16:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[T_17:.*]] = "tosa.transpose"(%[[T_15]], %[[T_16]]) : (tensor<?x?x?x16xf32>, tensor<4xi32>) -> tensor<?x16x?x?xf32>
// CHECK:           %[[T_18:.*]] = tensor.cast %[[T_17]] : tensor<?x16x?x?xf32> to tensor<?x16x?x?xf32>
// CHECK:           %[[T_19:.*]] = torch_c.from_builtin_tensor %[[T_18]] : tensor<?x16x?x?xf32> -> !torch.vtensor<[?,16,?,?],f32>
// CHECK:           return %[[T_19]] : !torch.vtensor<[?,16,?,?],f32>
func.func @torch.aten.convolution$depthwise(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[16,1,3,3],f32>) -> !torch.vtensor<[?,16,?,?],f32> {
%false = torch.constant.bool false
%none = torch.constant.none
%int0 = torch.constant.int 0
%int1 = torch.constant.int 1
%int8 = torch.constant.int 8
%1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
%2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
%3 = torch.prim.ListConstruct  : () -> !torch.list<int>
%4 = torch.aten.convolution %arg0, %arg1, %none, %1, %2, %1, %false, %3, %int8 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[16,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,16,?,?],f32>
return %4 : !torch.vtensor<[?,16,?,?],f32>
}
