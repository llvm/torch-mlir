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
