// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

func.func @torch.aten.prod.intdim(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  // CHECK-LABEL: @torch.aten.prod.intdim(
  // CHECK: %[[VAL_0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %none = torch.constant.none
  // CHECK: %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[VAL_2:.*]] = stablehlo.reduce(%[[VAL_0]] init: %[[VAL_1]]) applies stablehlo.multiply across dimensions = [1] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xf32>
  %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
  // CHECK: return %[[VAL_3]] : !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

func.func @torch.aten.prod.intdim_negative_dim(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  // CHECK-LABEL: @torch.aten.prod.intdim_negative_dim(
  // CHECK: %[[VAL_0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %none = torch.constant.none
  // CHECK: %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[VAL_2:.*]] = stablehlo.reduce(%[[VAL_0]] init: %[[VAL_1]]) applies stablehlo.multiply across dimensions = [3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xf32>
  %0 = torch.aten.prod.dim_int %arg0, %int-1, %false, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
  // CHECK: return %[[VAL_3]] : !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL: @torch.prims.xor_sum(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[4],si32>) -> !torch.vtensor<[],si32> {
// CHECK:   %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4],si32> -> tensor<4xi32>
// CHECK:   %[[INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK:   %[[REDUCE:.*]] = stablehlo.reduce(%[[INPUT]] init: %[[INIT]]) applies stablehlo.xor across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
// CHECK:   %[[OUT:.*]] = torch_c.from_builtin_tensor %[[REDUCE]] : tensor<i32> -> !torch.vtensor<[],si32>
// CHECK:   return %[[OUT]] : !torch.vtensor<[],si32>
func.func @torch.prims.xor_sum(%arg0: !torch.vtensor<[4],si32>) -> !torch.vtensor<[],si32> {
  %int0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %0 = torch.prims.xor_sum %arg0, %dims, %none : !torch.vtensor<[4],si32>, !torch.list<int>, !torch.none -> !torch.vtensor<[],si32>
  return %0 : !torch.vtensor<[],si32>
}
