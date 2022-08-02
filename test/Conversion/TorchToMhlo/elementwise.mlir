// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:  func.func @torch.aten.gelu(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = "chlo.constant_like"(%[[T0]]) {value = 1.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = "chlo.constant_like"(%[[T0]]) {value = 2.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = "chlo.constant_like"(%[[T0]]) {value = 5.000000e-01 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = mhlo.rsqrt %[[T2]] : tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = mhlo.multiply %[[T0]], %[[T4]] : tensor<?x?xf32>
// CHECK:         %[[T6:.*]] = chlo.erf %[[T5]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[T7:.*]] = mhlo.add %[[T6]], %[[T1]] : tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = mhlo.multiply %[[T7]], %[[T3]] : tensor<?x?xf32>
// CHECK:         %[[T9:.*]] = mhlo.multiply %[[T0]], %[[T8]] : tensor<?x?xf32>
// CHECK:         %[[T10:.*]] = torch_c.from_builtin_tensor %[[T9]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T10]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.gelu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %str = torch.constant.str "none"
    %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[?,?],f32>, !torch.str -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}


// CHECK-LABEL:   func.func @torch.aten.tanh$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.tanh %[[VAL_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.tanh$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.log %[[VAL_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.log$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.log %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.exp$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.exponential %[[VAL_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.exp$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.exp %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.neg$basic(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.negate %[[VAL_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.neg$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.addscalar$basic(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = chlo.broadcast_add %[[VAL_1]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addscalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %int9, %int1 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addscalar$alpha(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_multiply %[[VAL_2]], %[[VAL_3]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_add %[[VAL_1]], %[[VAL_4]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addscalar$alpha(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %int2 = torch.constant.int 2
  %0 = torch.aten.add.Scalar %arg0, %int9, %int2 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addtensor$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_add %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addtensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addtensor$alpha(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_multiply %[[VAL_3]], %[[VAL_4]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = chlo.broadcast_add %[[VAL_2]], %[[VAL_5]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addtensor$alpha(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int2 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addtensor$promote(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[?,?],si32>, 
// CHECK-SAME:                                           %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = mhlo.convert(%[[VAL_2]]) : (tensor<?x?xi32>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_add %[[VAL_4]], %[[VAL_3]] : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],si64>
func.func @torch.aten.addtensor$promote(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],si32>, !torch.vtensor<[?,?],si64>, !torch.int -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subscalar$basic(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = chlo.broadcast_subtract %[[VAL_1]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.subscalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Scalar %arg0, %int9, %int1 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subscalar$alpha(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_multiply %[[VAL_2]], %[[VAL_3]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_subtract %[[VAL_1]], %[[VAL_4]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.subscalar$alpha(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %int2 = torch.constant.int 2
  %0 = torch.aten.sub.Scalar %arg0, %int9, %int2 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subtensor$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_subtract %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.subtensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subtensor$alpha(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_multiply %[[VAL_3]], %[[VAL_4]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = chlo.broadcast_subtract %[[VAL_2]], %[[VAL_5]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.subtensor$alpha(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int2 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subtensor$promote(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[?,?],si32>, 
// CHECK-SAME:                                           %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = mhlo.convert(%[[VAL_2]]) : (tensor<?x?xi32>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_subtract %[[VAL_4]], %[[VAL_3]] : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],si64>
func.func @torch.aten.subtensor$promote(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],si32>, !torch.vtensor<[?,?],si64>, !torch.int -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.mulscalar$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = chlo.broadcast_multiply %[[VAL_1]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.mulscalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %0 = torch.aten.mul.Scalar %arg0, %int9 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.multensor$basic(
// CHECK-SAME:                                          %[[VLA_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                          %[[VLA_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VLA_2:.*]] = torch_c.to_builtin_tensor %[[VLA_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VLA_3:.*]] = torch_c.to_builtin_tensor %[[VLA_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VLA_4:.*]] = chlo.broadcast_multiply %[[VLA_2]], %[[VLA_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VLA_5:.*]] = torch_c.from_builtin_tensor %[[VLA_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VLA_5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.multensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divscalar$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = chlo.broadcast_divide %[[VAL_1]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.divscalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %0 = torch.aten.div.Scalar %arg0, %int9 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divtensor$basic(
// CHECK-SAME:                                          %[[VLA_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                          %[[VLA_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VLA_2:.*]] = torch_c.to_builtin_tensor %[[VLA_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VLA_3:.*]] = torch_c.to_builtin_tensor %[[VLA_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VLA_4:.*]] = chlo.broadcast_divide %[[VLA_2]], %[[VLA_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VLA_5:.*]] = torch_c.from_builtin_tensor %[[VLA_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VLA_5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.divtensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gt.scalar(
// CHECK-SAME:                                   %arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %1 = mhlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK:           %2 = chlo.broadcast_compare %0, %1 {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
// CHECK:           %3 = torch_c.from_builtin_tensor %2 : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %3 : !torch.vtensor<[?,?],i1>
func.func @torch.aten.gt.scalar(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.gt.Scalar %arg0, %int3 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gt.tensor(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_compare %[[VAL_2]], %[[VAL_3]] {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<?x?xf32>, tensor<64xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
func.func @torch.aten.gt.tensor(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.lt.tensor(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_compare %[[VAL_2]], %[[VAL_3]] {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?x?xf32>, tensor<64xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
func.func @torch.aten.lt.tensor(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.lt.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.eq.tensor(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_compare %[[VAL_2]], %[[VAL_3]] {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<?x?xf32>, tensor<64xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
func.func @torch.aten.eq.tensor(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ne.tensor(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:           %[[VAL_4:.*]] = chlo.broadcast_compare %[[VAL_2]], %[[VAL_3]] {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction NE>} : (tensor<?x?xf32>, tensor<64xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
func.func @torch.aten.ne.tensor(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.ne.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.permute$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[64,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = "mhlo.transpose"(%[[VAL_1]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x64xf32>) -> tensor<64x4xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<64x4xf32> -> !torch.vtensor<[64,4],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[64,4],f32>
func.func @torch.aten.permute$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[64,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[4,64],f32>, !torch.list<int> -> !torch.vtensor<[64,4],f32>
  return %1 : !torch.vtensor<[64,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.relu(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = "chlo.constant_like"(%[[VAL_1]]) {value = 0.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = mhlo.maximum %[[VAL_1]], %[[VAL_2]] : tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.relu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
