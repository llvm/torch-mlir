// RUN: torch-mlir-opt <%s -convert-torch-to-tosa -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.tanh$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.tanh %[[ARG_BUILTIN]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.tanh$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sigmoid$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.sigmoid %[[ARG_BUILTIN]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.sigmoid$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.sigmoid %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.relu$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.clamp %[[ARG_BUILTIN]] {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.relu$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.leaky_relu$basic(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<1.000000e-01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_4]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_1]], %[[VAL_3]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.select %[[VAL_5]], %[[VAL_1]], %[[VAL_6]] : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.leaky_relu$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %fp0 = torch.constant.float 1.000000e-01
  %0 = torch.aten.leaky_relu %arg0, %fp0 : !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.log$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.log %[[ARG_BUILTIN]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.log$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.log %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.exp$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.exp %[[ARG_BUILTIN]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.exp$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.exp %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.neg$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.negate %[[ARG_BUILTIN]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.neg$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.floor$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.floor %[[ARG_BUILTIN]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.floor$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.floor %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_not$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.bitwise_not %[[ARG_BUILTIN]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.bitwise_not$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.bitwise_not %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ceil$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.ceil %[[VAL_1]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.ceil$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.ceil %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.reciprocal$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reciprocal %[[VAL_1]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.reciprocal$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.reciprocal %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.add$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.add %[[VAL_2]], %[[VAL_6]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.add$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32>, !torch.int -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sub$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.sub %[[VAL_2]], %[[VAL_6]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.sub$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32>, !torch.int -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.mul$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.mul %[[VAL_2]], %[[VAL_3]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.mul$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32> -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_3]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_2]], %[[VAL_4]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.div$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32> -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

func.func @test_reduce_mean_dim$basic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %dim0 = torch.constant.int 0
  %reducedims = torch.prim.ListConstruct %dim0 : (!torch.int) -> !torch.list<int>
  %keepdims = torch.constant.bool false
  %dtype = torch.constant.none
  // expected-error @+1 {{Failed convertReduceMean: support for dynamic input shape not implemented}}
  %0 = torch.aten.mean.dim %arg0, %reducedims, %keepdims, %dtype : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_sum_dims$basic(
// CHECK-SAME:                          %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[ARG1_BUILTIN:.*]] = torch.constant.none
// CHECK:           %[[ARG2_BUILTIN:.*]] = torch.constant.bool false
// CHECK:           %[[ARG3:.*]] = torch.constant.int 0
// CHECK:           %[[ARG3_BUILTIN:.*]] = torch.prim.ListConstruct %[[ARG3]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[SUM:.*]] = tosa.reduce_sum %[[ARG0_BUILTIN]] {axis = 0 : i32} : (tensor<?x?x?x?xf32>) -> tensor<1x?x?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.reshape %[[SUM]] {new_shape = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>} : (tensor<1x?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?,?],f32>
func.func @test_reduce_sum_dims$basic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
    return %1 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @test_linalg_vector_norm$basic(
// CHECK-SAME:                          %[[ARG0:.*]]: !torch.vtensor<[3,151,64],f32>) -> !torch.vtensor<[3,151,1],f32> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[3,151,64],f32> -> tensor<3x151x64xf32>
// CHECK:           %[[ARG1:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[ARG2:.*]] = torch.constant.int -1
// CHECK:           %[[ARG3:.*]] = torch.constant.bool true
// CHECK:           %[[ARG4:.*]] = torch.constant.none
// CHECK:           %[[ARG5:.*]] = torch.prim.ListConstruct %[[ARG2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[ARG6:.*]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[ARG7:.*]] = tosa.abs %[[ARG0_BUILTIN]] : (tensor<3x151x64xf32>) -> tensor<3x151x64xf32>
// CHECK:           %[[ARG8:.*]] = tosa.pow %[[ARG7]], %[[ARG6]] : (tensor<3x151x64xf32>, tensor<f32>) -> tensor<3x151x64xf32>
// CHECK:           %[[ARG9:.*]] = tosa.reduce_sum %[[ARG8]] {axis = 2 : i32} : (tensor<3x151x64xf32>) -> tensor<3x151x1xf32>
// CHECK:           %[[ARG10:.*]] = tosa.reciprocal %[[ARG6]] : (tensor<f32>) -> tensor<f32>
// CHECK:           %[[ARG11:.*]] = tosa.pow %[[ARG9]], %[[ARG10]] : (tensor<3x151x1xf32>, tensor<f32>) -> tensor<3x151x1xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[ARG11]] : tensor<3x151x1xf32> -> !torch.vtensor<[3,151,1],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[3,151,1],f32>
func.func @test_linalg_vector_norm$basic(%arg0: !torch.vtensor<[3,151,64],f32>) -> (!torch.vtensor<[3,151,1],f32>) {
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true
  %none = torch.constant.none
  %1 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.linalg_vector_norm %arg0, %float2.000000e00, %1, %true, %none : !torch.vtensor<[3,151,64],f32>, !torch.float, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,151,1],f32>
  return %2 : !torch.vtensor<[3,151,1],f32>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_sum$basic(
// CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[ARG1_BUILTIN:.*]] = torch.constant.none
// CHECK:           %[[REDUCE1:.*]] = tosa.reduce_sum %[[ARG0_BUILTIN]] {axis = 0 : i32} : (tensor<?x?x?x?xf32>) -> tensor<1x?x?x?xf32>
// CHECK:           %[[REDUCE2:.*]] = tosa.reduce_sum %[[REDUCE1]] {axis = 1 : i32} : (tensor<1x?x?x?xf32>) -> tensor<1x1x?x?xf32>
// CHECK:           %[[REDUCE3:.*]] = tosa.reduce_sum %[[REDUCE2]] {axis = 2 : i32} : (tensor<1x1x?x?xf32>) -> tensor<1x1x1x?xf32>
// CHECK:           %[[REDUCE4:.*]] = tosa.reduce_sum %[[REDUCE3]] {axis = 3 : i32} : (tensor<1x1x1x?xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.reshape %[[REDUCE4]] {new_shape = array<i64: 1>} : (tensor<1x1x1x1xf32>) -> tensor<1xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @test_reduce_sum$basic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.none -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_all$basic(
// CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[1],i1> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],i1> -> tensor<?x?x?x?xi1>
// CHECK:           %[[REDUCE1:.*]] = tosa.reduce_all %[[ARG0_BUILTIN]] {axis = 0 : i32} : (tensor<?x?x?x?xi1>) -> tensor<1x?x?x?xi1>
// CHECK:           %[[REDUCE2:.*]] = tosa.reduce_all %[[REDUCE1]] {axis = 1 : i32} : (tensor<1x?x?x?xi1>) -> tensor<1x1x?x?xi1>
// CHECK:           %[[REDUCE3:.*]] = tosa.reduce_all %[[REDUCE2]] {axis = 2 : i32} : (tensor<1x1x?x?xi1>) -> tensor<1x1x1x?xi1>
// CHECK:           %[[REDUCE4:.*]] = tosa.reduce_all %[[REDUCE3]] {axis = 3 : i32} : (tensor<1x1x1x?xi1>) -> tensor<1x1x1x1xi1>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.reshape %[[REDUCE4]] {new_shape = array<i64: 1>} : (tensor<1x1x1x1xi1>) -> tensor<1xi1>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<1xi1> -> !torch.vtensor<[1],i1>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1],i1>
func.func @test_reduce_all$basic(%arg0: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[1],i1> {
  %0 = torch.aten.all %arg0 : !torch.vtensor<[?,?,?,?],i1> -> !torch.vtensor<[1],i1>
  return %0 : !torch.vtensor<[1],i1>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_any_dim$basic(
// CHECK-SAME:                                    %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[?,?,?],i1> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],i1> -> tensor<?x?x?x?xi1>
// CHECK:           %[[ARG1:.*]] = torch.constant.int 0
// CHECK:           %[[ARG2:.*]] = torch.constant.bool false
// CHECK:           %[[REDUCE:.*]] = tosa.reduce_any %[[ARG0_BUILTIN]] {axis = 0 : i32} : (tensor<?x?x?x?xi1>) -> tensor<1x?x?x?xi1>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.reshape %[[REDUCE]] {new_shape = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>} : (tensor<1x?x?x?xi1>) -> tensor<?x?x?xi1>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?x?xi1> -> !torch.vtensor<[?,?,?],i1>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?,?],i1>
func.func @test_reduce_any_dim$basic(%arg0: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[?,?,?],i1> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.any.dim %arg0, %int0, %false : !torch.vtensor<[?,?,?,?],i1>, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?],i1>
  return %0 : !torch.vtensor<[?,?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_any$basic(
// CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[1],i1> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],i1> -> tensor<?x?x?x?xi1>
// CHECK:           %[[REDUCE1:.*]] = tosa.reduce_any %[[ARG0_BUILTIN]] {axis = 0 : i32} : (tensor<?x?x?x?xi1>) -> tensor<1x?x?x?xi1>
// CHECK:           %[[REDUCE2:.*]] = tosa.reduce_any %[[REDUCE1]] {axis = 1 : i32} : (tensor<1x?x?x?xi1>) -> tensor<1x1x?x?xi1>
// CHECK:           %[[REDUCE3:.*]] = tosa.reduce_any %[[REDUCE2]] {axis = 2 : i32} : (tensor<1x1x?x?xi1>) -> tensor<1x1x1x?xi1>
// CHECK:           %[[REDUCE4:.*]] = tosa.reduce_any %[[REDUCE3]] {axis = 3 : i32} : (tensor<1x1x1x?xi1>) -> tensor<1x1x1x1xi1>
// CHECK:           %[[RESULT_BUILTIN:.*]] = tosa.reshape %[[REDUCE4]] {new_shape = array<i64: 1>} : (tensor<1x1x1x1xi1>) -> tensor<1xi1>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<1xi1> -> !torch.vtensor<[1],i1>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1],i1>
func.func @test_reduce_any$basic(%arg0: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[1],i1> {
  %0 = torch.aten.any %arg0 : !torch.vtensor<[?,?,?,?],i1> -> !torch.vtensor<[1],i1>
  return %0 : !torch.vtensor<[1],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.rsqrt$basic(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.rsqrt %[[VAL_1]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.rsqrt$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.rsqrt %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.maximum$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.maximum %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.maximum$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.maximum %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.minimum$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.minimum %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.minimum$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.minimum %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.pow.Tensor_Scalar$basic(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<3.123400e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.pow %[[VAL_1]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.pow.Tensor_Scalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %fp0 = torch.constant.float 3.123400e+00
  %0 = torch.aten.pow.Tensor_Scalar %arg0, %fp0 : !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.rsub.Scalar$basic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.float 6.432100e+00
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<3.123400e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<6.432100e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_1]], %[[VAL_5]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.sub %[[VAL_4]], %[[VAL_6]] : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.rsub.Scalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %other = torch.constant.float 3.123400e+00
  %alpha = torch.constant.float 6.432100e+00
  %0 = torch.aten.rsub.Scalar %arg0, %other, %alpha : !torch.vtensor<[?,?],f32>, !torch.float, !torch.float -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.rsub.Scalar$basic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<3.123400e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_1]], %[[VAL_5]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.sub %[[VAL_4]], %[[VAL_6]] : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.rsub.Scalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %other = torch.constant.float 3.123400e+00
  %alpha = torch.constant.int 1
  %0 = torch.aten.rsub.Scalar %arg0, %other, %alpha : !torch.vtensor<[?,?],f32>, !torch.float, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gt.Tensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.greater %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.gt.Tensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.lt.Tensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.greater %[[VAL_3]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.lt.Tensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.lt.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.eq.Tensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.equal %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.eq.Tensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.reshape$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: -1>} : (tensor<?x?x?x?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?],f32>
// CHECK:         }
func.func @torch.aten.reshape$basic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?],f32> {
  %dim0 = torch.constant.int -1
  %shape = torch.prim.ListConstruct %dim0 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.reshape %arg0, %shape : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.native_batch_norm$basic(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[10,4,3],f32>) -> !torch.vtensor<[10,4,3],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,4,3],f32> -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<[3.000000e+00, 2.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[VAL_5:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_2]] {new_shape = array<i64: 4, 1>} : (tensor<4xf32>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_3]] {new_shape = array<i64: 4, 1>} : (tensor<4xf32>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_3]] {new_shape = array<i64: 4, 1>} : (tensor<4xf32>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_2]] {new_shape = array<i64: 4, 1>} : (tensor<4xf32>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_13:.*]] = tosa.sub %[[VAL_1]], %[[VAL_8]] : (tensor<10x4x3xf32>, tensor<4x1xf32>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.add %[[VAL_9]], %[[VAL_12]] : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.rsqrt %[[VAL_14]] : (tensor<4x1xf32>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.mul %[[VAL_13]], %[[VAL_15]] {shift = 0 : i8} : (tensor<10x4x3xf32>, tensor<4x1xf32>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.mul %[[VAL_16]], %[[VAL_10]] {shift = 0 : i8} : (tensor<10x4x3xf32>, tensor<4x1xf32>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.add %[[VAL_17]], %[[VAL_11]] : (tensor<10x4x3xf32>, tensor<4x1xf32>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<10x4x3xf32> -> !torch.vtensor<[10,4,3],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[10,4,3],f32>
// CHECK:         }
func.func @torch.aten.native_batch_norm$basic(%arg0: !torch.vtensor<[10,4,3],f32> ) -> !torch.vtensor<[10,4,3],f32> {
  %0 = torch.vtensor.literal(dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %1 = torch.vtensor.literal(dense<[3.000000e+00, 2.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %2 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %false, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[10,4,3],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[10,4,3],f32>
  return %2 : !torch.vtensor<[10,4,3],f32>
}

// -----

// CHECK-LABEL:   func.func @forward(
// CHECK-SAME:                       %[[VAL_0:.*]]: !torch.vtensor<[10,3,8,9,3,4],f32>) -> !torch.vtensor<[10,3,?,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,3,8,9,3,4],f32> -> tensor<10x3x8x9x3x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: 10, 3, 216, 4>} : (tensor<10x3x8x9x3x4xf32>) -> tensor<10x3x216x4xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.cast %[[VAL_4]] : tensor<10x3x216x4xf32> to tensor<10x3x?x4xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<10x3x?x4xf32> -> !torch.vtensor<[10,3,?,4],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[10,3,?,4],f32>
// CHECK:         }
func.func @forward(%arg0: !torch.vtensor<[10,3,8,9,3,4],f32> ) -> !torch.vtensor<[10,3,?,4],f32> {
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %0 = torch.aten.flatten.using_ints %arg0, %int2, %int4 : !torch.vtensor<[10,3,8,9,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[10,3,?,4],f32>
  return %0 : !torch.vtensor<[10,3,?,4],f32>
}

// -----

// CHECK-LABEL:   func.func @forward(
// CHECK-SAME:                  %[[VAL_0:.*]]: !torch.vtensor<[1,6,4],f32>) -> !torch.vtensor<[1,2,3,4],f32> {
// CHECK:           %[[VAL:.*]]   = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,6,4],f32> -> tensor<1x6x4xf32>
// CHECK:           %[[VAL_1:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL]] {new_shape = array<i64: 1, 2, 3, 4>} : (tensor<1x6x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<1x2x3x4xf32> -> !torch.vtensor<[1,2,3,4],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[1,2,3,4],f32>
// CHECK:         }
func.func @forward(%arg0: !torch.vtensor<[1,6,4],f32> ) -> !torch.vtensor<[1,2,3,4],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.unflatten.int %arg0, %int1, %0 : !torch.vtensor<[1,6,4],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,3,4],f32>
  return %1 : !torch.vtensor<[1,2,3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @forward(
// CHECK-SAME:                       %[[VAL_0:.*]]: !torch.vtensor<[5,2,2,3],f32>,
// CHECK-SAME:                       %[[VAL_1:.*]]: !torch.vtensor<[2,2,3],f32>,
// CHECK-SAME:                       %[[VAL_2:.*]]: !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[5,2,2,3],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,2,2,3],f32> -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,2,3],f32> -> tensor<2x2x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[2,2,3],f32> -> tensor<2x2x3xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.float 5.000000e-01
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]], %[[VAL_7]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{value = dense<1.200000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reciprocal %[[VAL_10]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.reduce_sum %[[VAL_3]] {axis = 3 : i32} : (tensor<5x2x2x3xf32>) -> tensor<5x2x2x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.reduce_sum %[[VAL_12]] {axis = 2 : i32} : (tensor<5x2x2x1xf32>) -> tensor<5x2x1x1xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.reduce_sum %[[VAL_13]] {axis = 1 : i32} : (tensor<5x2x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_14]] {new_shape = array<i64: 5, 1, 1, 1>} : (tensor<5x1x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.mul %[[VAL_15]], %[[VAL_11]] {shift = 0 : i8} : (tensor<5x1x1x1xf32>, tensor<1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.sub %[[VAL_3]], %[[VAL_16]] : (tensor<5x2x2x3xf32>, tensor<5x1x1x1xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.mul %[[VAL_17]], %[[VAL_17]] {shift = 0 : i8} : (tensor<5x2x2x3xf32>, tensor<5x2x2x3xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.reduce_sum %[[VAL_18]] {axis = 3 : i32} : (tensor<5x2x2x3xf32>) -> tensor<5x2x2x1xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.reduce_sum %[[VAL_19]] {axis = 2 : i32} : (tensor<5x2x2x1xf32>) -> tensor<5x2x1x1xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.reduce_sum %[[VAL_20]] {axis = 1 : i32} : (tensor<5x2x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.reshape %[[VAL_21]] {new_shape = array<i64: 5, 1, 1, 1>} : (tensor<5x1x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.mul %[[VAL_22]], %[[VAL_11]] {shift = 0 : i8} : (tensor<5x1x1x1xf32>, tensor<1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_24:.*]] = tosa.reshape %[[VAL_4]] {new_shape = array<i64: 1, 2, 2, 3>} : (tensor<2x2x3xf32>) -> tensor<1x2x2x3xf32>
// CHECK:           %[[VAL_25:.*]] = tosa.reshape %[[VAL_5]] {new_shape = array<i64: 1, 2, 2, 3>} : (tensor<2x2x3xf32>) -> tensor<1x2x2x3xf32>
// CHECK:           %[[VAL_26:.*]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_27:.*]] = tosa.sub %[[VAL_3]], %[[VAL_16]] : (tensor<5x2x2x3xf32>, tensor<5x1x1x1xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_28:.*]] = tosa.add %[[VAL_23]], %[[VAL_26]] : (tensor<5x1x1x1xf32>, tensor<f32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_29:.*]] = tosa.rsqrt %[[VAL_28]] : (tensor<5x1x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_30:.*]] = tosa.mul %[[VAL_27]], %[[VAL_29]] {shift = 0 : i8} : (tensor<5x2x2x3xf32>, tensor<5x1x1x1xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_31:.*]] = tosa.mul %[[VAL_30]], %[[VAL_24]] {shift = 0 : i8} : (tensor<5x2x2x3xf32>, tensor<1x2x2x3xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_32:.*]] = tosa.add %[[VAL_31]], %[[VAL_25]] : (tensor<5x2x2x3xf32>, tensor<1x2x2x3xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_33:.*]] = torch_c.from_builtin_tensor %[[VAL_32]] : tensor<5x2x2x3xf32> -> !torch.vtensor<[5,2,2,3],f32>
// CHECK:           return %[[VAL_33]] : !torch.vtensor<[5,2,2,3],f32>
// CHECK:         }
func.func @forward(%arg0: !torch.vtensor<[5,2,2,3],f32> , %arg1: !torch.vtensor<[2,2,3],f32> , %arg2: !torch.vtensor<[2,2,3],f32> ) -> !torch.vtensor<[5,2,2,3],f32> {
  %float5.000000e-01 = torch.constant.float 5.000000e-01
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int2, %int2, %int3 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %0, %arg1, %arg2, %float5.000000e-01 : !torch.vtensor<[5,2,2,3],f32>, !torch.list<int>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.float -> !torch.vtensor<[5,2,2,3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>
  return %result0 : !torch.vtensor<[5,2,2,3],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ne.Tensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.equal %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = tosa.logical_not %[[VAL_4]] : (tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.ne.Tensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.ne.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.logical_or$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],i1>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK:           %[[VAL_4:.*]] = tosa.logical_or %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.logical_or$basic(%arg0: !torch.vtensor<[?,?],i1>, %arg1: !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.logical_or %arg0, %arg1 : !torch.vtensor<[?,?],i1>, !torch.vtensor<[?,?],i1> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @forward(
// CHECK-SAME:                  %[[VAL_0:.*]]: !torch.vtensor<[3,4,2],f32>) -> !torch.vtensor<[3,2,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,2],f32> -> tensor<3x4x2xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_3]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK:           %[[VAL_7:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_6]] : (tensor<3x4x2xf32>, tensor<3xi64>) -> tensor<3x2x4xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<3x2x4xf32> -> !torch.vtensor<[3,2,4],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[3,2,4],f32>
// CHECK:         }
func.func @forward(%arg0: !torch.vtensor<[3,4,2],f32> ) -> !torch.vtensor<[3,2,4],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[3,4,2],f32>, !torch.list<int> -> !torch.vtensor<[3,2,4],f32>
  return %1 : !torch.vtensor<[3,2,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_and.Tensor$basic(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[?,?],si32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.bitwise_and %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],si32>
// CHECK:         }
func.func @torch.aten.bitwise_and.Tensor$basic(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
  %0 = torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],si32>, !torch.vtensor<[?,?],si32> -> !torch.vtensor<[?,?],si32>
  return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log2$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<0.693147182> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.log %[[VAL_1]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_4]], %[[VAL_3]] {shift = 0 : i8} : (tensor<?x?xf32>, tensor<1x1xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.log2$basic(%arg0: !torch.vtensor<[?,?],f32> ) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.log2 %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.zeros$basic() -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_0:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_1:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_1]], %[[VAL_0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_4]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.zeros$basic() -> !torch.vtensor<[3,4],f32> {
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.zeros %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4],f32>
  return %1 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[4,3],si32>) -> !torch.vtensor<[4,3,1],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,3],si32> -> tensor<4x3xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: 4, 3, 1>} : (tensor<4x3xi32>) -> tensor<4x3x1xi32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<4x3x1xi32> -> !torch.vtensor<[4,3,1],si32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[4,3,1],si32>
// CHECK:         }

func.func @torch.aten.unsqueeze$basic(%arg0: !torch.vtensor<[4,3],si32> ) -> !torch.vtensor<[4,3,1],si32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.vtensor<[4,3],si32>, !torch.int -> !torch.vtensor<[4,3,1],si32>
  return %0 : !torch.vtensor<[4,3,1],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$negative_dim(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[4,3],si32>) -> !torch.vtensor<[4,3,1],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,3],si32> -> tensor<4x3xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: 4, 3, 1>} : (tensor<4x3xi32>) -> tensor<4x3x1xi32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<4x3x1xi32> -> !torch.vtensor<[4,3,1],si32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[4,3,1],si32>
// CHECK:         }
func.func @torch.aten.unsqueeze$negative_dim(%arg0: !torch.vtensor<[4,3],si32> ) -> !torch.vtensor<[4,3,1],si32> {
  %int2 = torch.constant.int -1
  %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.vtensor<[4,3],si32>, !torch.int -> !torch.vtensor<[4,3,1],si32>
  return %0 : !torch.vtensor<[4,3,1],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.contiguous$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.contiguous$basic(%arg0: !torch.vtensor<[?,?],f32> ) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.contiguous %arg0, %int0 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ones$basic() -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_0:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_1:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_1]], %[[VAL_0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<1> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_4]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.ones$basic() -> !torch.vtensor<[3,4],f32> {
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.ones %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4],f32>
  return %1 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.dropout$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = tosa.cast %[[VAL_1]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.dropout$basic(%arg0: !torch.vtensor<[?,?],f32> ) -> !torch.vtensor<[?,?],f32> {
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %0 = torch.aten.dropout %arg0, %float0.000000e00, %false : !torch.vtensor<[?,?],f32>, !torch.float, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 7
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_7:.*]] = torch.constant.none
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_11]] : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.avg_pool2d %[[VAL_12]] {acc_type = f32, kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>) -> tensor<1x1x1x512xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.transpose %[[VAL_13]], %[[VAL_14]] : (tensor<1x1x1x512xf32>, tensor<4xi32>) -> tensor<1x512x1x1xf32>
// CHECK:           %[[VAL_16:.*]] = tensor.cast %[[VAL_15]] : tensor<1x512x1x1xf32> to tensor<1x512x1x1xf32>
// CHECK:           %[[VAL_17:.*]] = torch_c.from_builtin_tensor %[[VAL_16]] : tensor<1x512x1x1xf32> -> !torch.vtensor<[1,512,1,1],f32>
// CHECK:           return %[[VAL_17]] : !torch.vtensor<[1,512,1,1],f32>
// CHECK:         }
func.func @torch.aten.avg_pool2d$basic(%arg0: !torch.vtensor<[1,512,7,7],f32> ) -> !torch.vtensor<[1,512,1,1],f32> {
  %int7 = torch.constant.int 7
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %kernel = torch.prim.ListConstruct %int7, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.avg_pool2d %arg0, %kernel, %stride, %padding, %false, %true, %none : !torch.vtensor<[1,512,7,7],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,1,1],f32>
  return %0 : !torch.vtensor<[1,512,1,1],f32>
}

// -----

// CHECK-LABEL: @torch.aten.max.dim$basic(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2x3xf32>)
// CHECK: %[[VAL_0:.*]] = torch_c.from_builtin_tensor %[[ARG0]] : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
// CHECK: %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,2,3],f32> -> tensor<3x2x3xf32>
// CHECK: %[[VAL_TRUE:.*]] = torch.constant.bool true
// CHECK: %[[VAL_I2:.*]] = torch.constant.int 2
// CHECK: %[[VAL_2:.*]] = tosa.reduce_max %[[VAL_1]] {axis = 2 : i32} : (tensor<3x2x3xf32>) -> tensor<3x2x1xf32>
// CHECK: %[[VAL_3:.*]] = tosa.argmax %[[VAL_1]] {axis = 2 : i32} : (tensor<3x2x3xf32>) -> tensor<3x2xi64>
// CHECK: %[[VAL_4:.*]] = tosa.reshape %[[VAL_3]] {new_shape = array<i64: 3, 2, 1>} : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK: %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<3x2x1xf32> -> !torch.vtensor<[3,2,1],f32>
// CHECK: %[[VAL_6:.*]] = torch_c.to_builtin_tensor %[[VAL_5]] : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
// CHECK: return %[[VAL_6]] : tensor<3x2x1xf32>
func.func @torch.aten.max.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.max.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}

// -----

// CHECK-LABEL: @torch.vtensor.literal_si64$basic(
// CHECK: %[[VAL_0:.*]] = "tosa.const"() <{value = dense<-1> : tensor<1x512xi64>}> : () -> tensor<1x512xi64>
// CHECK: %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x512xi64> -> !torch.vtensor<[1,512],si64>
// CHECK: return %[[VAL_1]] : !torch.vtensor<[1,512],si64>
func.func @torch.vtensor.literal_si64$basic() -> !torch.vtensor<[1,512],si64> {
  %0 = torch.vtensor.literal(dense<-1> : tensor<1x512xsi64>) : !torch.vtensor<[1,512],si64>
  return %0 : !torch.vtensor<[1,512],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.arange.start_step() -> !torch.vtensor<[5],si64> {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST5:.*]] = torch.constant.int 5
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4]> : tensor<5xi64>}> : () -> tensor<5xi64>
// CHECK:           %[[VAL_1:.*]] = tosa.cast %[[VAL_0]] : (tensor<5xi64>) -> tensor<5xi64>
// CHECK:           %[[VAL_2:.*]] = torch_c.from_builtin_tensor %1 : tensor<5xi64> -> !torch.vtensor<[5],si64>
// CHECK:           return %[[VAL_2]] : !torch.vtensor<[5],si64>
func.func @torch.aten.arange.start_step() -> !torch.vtensor<[5],si64> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int5 = torch.constant.int 5
  %int1 = torch.constant.int 1
  %0 = torch.aten.arange.start_step %int0, %int5, %int1, %none, %none, %none, %none : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[5],si64>
  return %0 : !torch.vtensor<[5],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.prim.NumToTensor.Scalar() -> !torch.vtensor<[],si64> {
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<[],si64>
func.func @torch.prim.NumToTensor.Scalar() -> !torch.vtensor<[],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.copy(
// CHECK-SAME:                                   %[[ARG_0:.*]]: !torch.vtensor<[1,1,5,5],ui8>) -> !torch.vtensor<[1,1,5,5],i1> {
// CHECK:           %[[INP:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[1,1,5,5],ui8> -> tensor<1x1x5x5xi8>
// CHECK:           %[[CST5:.*]] = torch.constant.int 5
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[CST11:.*]] = torch.constant.int 11
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_2:.*]] = tosa.equal %[[VAL_0]], %[[VAL_1]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           %[[VAL_3:.*]] = tosa.logical_not %[[VAL_2]] : (tensor<i1>) -> tensor<i1>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1x5x5xi8>}> : () -> tensor<1x1x5x5xi8>
// CHECK:           %[[VAL_5:.*]] = tosa.equal %[[INP]], %[[VAL_4]] : (tensor<1x1x5x5xi8>, tensor<1x1x5x5xi8>) -> tensor<1x1x5x5xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.logical_not %[[VAL_5]] : (tensor<1x1x5x5xi1>) -> tensor<1x1x5x5xi1>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1x1x5x5xi1> -> !torch.vtensor<[1,1,5,5],i1>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1,1,5,5],i1>
func.func @torch.aten.copy(%arg0: !torch.vtensor<[1,1,5,5],ui8>) -> !torch.vtensor<[1,1,5,5],i1> {
  %int5 = torch.constant.int 5
  %int1 = torch.constant.int 1
  %int11 = torch.constant.int 11
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %0 = torch.prim.NumToTensor.Scalar %int0 : !torch.int -> !torch.vtensor<[],si64>
  %1 = torch.aten.to.dtype %0, %int11, %false, %false, %none : !torch.vtensor<[],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],i1>
  %2 = torch.prim.ListConstruct %int1, %int1, %int5, %int5 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.broadcast_to %1, %2 : !torch.vtensor<[],i1>, !torch.list<int> -> !torch.vtensor<[1,1,5,5],i1>
  %4 = torch.aten.copy %3, %arg0, %false : !torch.vtensor<[1,1,5,5],i1>, !torch.vtensor<[1,1,5,5],ui8>, !torch.bool -> !torch.vtensor<[1,1,5,5],i1>
  return %4 : !torch.vtensor<[1,1,5,5],i1>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.to.dtype(
// CHECK-SAME:                                   %[[ARG_0:.*]]: !torch.vtensor<[3,5],si64>) -> !torch.vtensor<[3,5],i1> {
// CHECK:           %[[INP:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[3,5],si64> -> tensor<3x5xi64>
// CHECK:           %[[CST11:.*]] = torch.constant.int 11
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{value = dense<0> : tensor<3x5xi64>}> : () -> tensor<3x5xi64>
// CHECK:           %[[VAL_1:.*]] = tosa.equal %[[INP]], %[[VAL_0]] : (tensor<3x5xi64>, tensor<3x5xi64>) -> tensor<3x5xi1>
// CHECK:           %[[VAL_2:.*]] = tosa.logical_not %[[VAL_1]] : (tensor<3x5xi1>) -> tensor<3x5xi1>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<3x5xi1> -> !torch.vtensor<[3,5],i1>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[3,5],i1>
func.func @torch.aten.to.dtype(%arg0: !torch.vtensor<[3,5],si64>) -> !torch.vtensor<[3,5],i1> {
  %int11 = torch.constant.int 11
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.to.dtype %arg0, %int11, %false, %false, %none : !torch.vtensor<[3,5],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,5],i1>
  return %0 : !torch.vtensor<[3,5],i1>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.to.dtype(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[1,128],i1>) -> !torch.vtensor<[1,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,128],i1> -> tensor<1x128xi1>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_1]] : (tensor<1x128xi1>) -> tensor<1x128xi64>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<1x128xi64> -> !torch.vtensor<[1,128],si64>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[1,128],si64>
// CHECK:         }
func.func @torch.aten.to.dtype(%arg0: !torch.vtensor<[1,128],i1>) -> !torch.vtensor<[1,128],si64> {
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.to.dtype %arg0, %int4, %false, %false, %none : !torch.vtensor<[1,128],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,128],si64>
  return %0 : !torch.vtensor<[1,128],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.gather(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[1,4,3],f32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,4,3],f32> -> tensor<1x4x3xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,4,2],si64> -> tensor<1x4x2xi64>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_3]] : (tensor<1x4x2xi64>) -> tensor<1x4x2xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_6]] {new_shape = array<i64: 1, 4, 2, 1>} : (tensor<1x4x2xi32>) -> tensor<1x4x2x1xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x4x2x1xi32>}> : () -> tensor<1x4x2x1xi32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{value = dense<{{\[\[}}{{\[\[}}0], [0]], {{\[\[}}1], [1]], {{\[\[}}2], [2]], {{\[\[}}3], [3]]]]> : tensor<1x4x2x1xi32>}> : () -> tensor<1x4x2x1xi32>
// CHECK:           %[[VAL_10:.*]] = tosa.concat %[[VAL_8]], %[[VAL_9]], %[[VAL_7]] {axis = 3 : i32} : (tensor<1x4x2x1xi32>, tensor<1x4x2x1xi32>, tensor<1x4x2x1xi32>) -> tensor<1x4x2x3xi32>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_2]] {new_shape = array<i64: 1, 12, 1>} : (tensor<1x4x3xf32>) -> tensor<1x12x1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_10]] {new_shape = array<i64: 8, 3>} : (tensor<1x4x2x3xi32>) -> tensor<8x3xi32>
// CHECK:           %[[VAL_13:.*]] = "tosa.const"() <{value = dense<[12, 3, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.mul %[[VAL_12]], %[[VAL_13]] {shift = 0 : i8} : (tensor<8x3xi32>, tensor<3xi32>) -> tensor<8x3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.reduce_sum %[[VAL_14]] {axis = 1 : i32} : (tensor<8x3xi32>) -> tensor<8x1xi32>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_15]] {new_shape = array<i64: 1, 8>} : (tensor<8x1xi32>) -> tensor<1x8xi32>
// CHECK:           %[[VAL_17:.*]] = tosa.gather %[[VAL_11]], %[[VAL_16]] : (tensor<1x12x1xf32>, tensor<1x8xi32>) -> tensor<1x8x1xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_17]] {new_shape = array<i64: 1, 4, 2>} : (tensor<1x8x1xf32>) -> tensor<1x4x2xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<1x4x2xf32> -> !torch.vtensor<[1,4,2],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,4,2],f32>
// CHECK:         }
func.func @torch.aten.gather(%arg0: !torch.vtensor<[1,4,3],f32>, %arg1: !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %0 = torch.aten.gather %arg0, %int-1, %arg1, %false : !torch.vtensor<[1,4,3],f32>, !torch.int, !torch.vtensor<[1,4,2],si64>, !torch.bool -> !torch.vtensor<[1,4,2],f32>
  return %0 : !torch.vtensor<[1,4,2],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.add$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[2,2],si32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[2,2],si32>) -> !torch.vtensor<[2,2],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,2],si32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,2],si32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]] {shift = 0 : i8} : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.add %[[VAL_2]], %[[VAL_6]] : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.cast %[[VAL_7]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<2x2xi64> -> !torch.vtensor<[2,2],si64>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[2,2],si64>
// CHECK:         }
func.func @torch.aten.add$basic(%arg0: !torch.vtensor<[2, 2],si32>, %arg1: !torch.vtensor<[2, 2],si32>) -> !torch.vtensor<[2, 2],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[2, 2],si32>, !torch.vtensor<[2, 2],si32>, !torch.int -> !torch.vtensor<[2, 2],si64>
  return %0 : !torch.vtensor<[2, 2],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.Scalar$basic(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],si64> -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 256
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{value = dense<256> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_4]], %[[VAL_5]] {shift = 0 : i8} : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_1]] : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.add %[[VAL_7]], %[[VAL_6]] : (tensor<1x1x128x128xi32>, tensor<i32>) -> tensor<1x1x128x128xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.cast %[[VAL_8]] : (tensor<1x1x128x128xi32>) -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
// CHECK:           return %[[VAL_10]] : !torch.vtensor<[1,1,128,128],si64>
// CHECK:         }
func.func @torch.aten.Scalar$basic(%arg0: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
  %int1 = torch.constant.int 1
  %int256 = torch.constant.int 256
  %0 = torch.aten.add.Scalar %arg0, %int256, %int1 : !torch.vtensor<[1,1,128,128],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],si64>
  return %0 : !torch.vtensor<[1,1,128,128],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.slice.negative_start(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,16,256],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 100
// CHECK:           %[[VAL_5:.*]] = torch.constant.int -16
// CHECK:           %[[VAL_4:.*]] = tosa.slice %[[VAL_1]] {size = array<i64: 4, 16, 256>, start = array<i64: 0, 49, 0>} : (tensor<4x65x256xf32>) -> tensor<4x16x256xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x16x256xf32> -> !torch.vtensor<[4,16,256],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,16,256],f32>
// CHECK:         }
func.func @torch.aten.slice.negative_start(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,16,256],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int100 = torch.constant.int 100
  %int-16 = torch.constant.int -16
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int-16, %int100, %int1 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16,256],f32>
  return %0 : !torch.vtensor<[4,16,256],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.clamp.min_none(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],si64> -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_fp = 0.000000e+00 : f32, max_int = 0 : i64, min_fp = -3.40282347E+38 : f32, min_int = -9223372036854775808 : i64} : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[1,1,128,128],si64>
// CHECK:         }
func.func @torch.aten.clamp.min_none(%arg0: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.clamp %arg0, %none, %int0 : !torch.vtensor<[1,1,128,128],si64>, !torch.none, !torch.int -> !torch.vtensor<[1,1,128,128],si64>
  return %0 : !torch.vtensor<[1,1,128,128],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.clamp.max_none(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],si64> -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_fp = 3.40282347E+38 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[1,1,128,128],si64>
// CHECK:         }
func.func @torch.aten.clamp.max_none(%arg0: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.clamp %arg0, %int0, %none : !torch.vtensor<[1,1,128,128],si64>, !torch.int, !torch.none -> !torch.vtensor<[1,1,128,128],si64>
  return %0 : !torch.vtensor<[1,1,128,128],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.clamp(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],si64> -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 511
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_fp = 5.110000e+02 : f32, max_int = 511 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[1,1,128,128],si64>
// CHECK:         }
func.func @torch.aten.clamp(%arg0: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
  %int0 = torch.constant.int 0
  %int511 = torch.constant.int 511
  %0 = torch.aten.clamp %arg0, %int0, %int511 : !torch.vtensor<[1,1,128,128],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],si64>
  return %0 : !torch.vtensor<[1,1,128,128],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.clamp.float(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],f32>) -> !torch.vtensor<[1,1,128,128],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],f32> -> tensor<1x1x128x128xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.float 6.432100e+00
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_fp = 6.432100e+00 : f32, max_int = 6 : i64, min_fp = 3.123400e+00 : f32, min_int = 3 : i64} : (tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<1x1x128x128xf32> -> !torch.vtensor<[1,1,128,128],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[1,1,128,128],f32>
// CHECK:         }
func.func @torch.aten.clamp.float(%arg0: !torch.vtensor<[1,1,128,128],f32>) -> !torch.vtensor<[1,1,128,128],f32> {
  %fp_min = torch.constant.float 3.123400e+00
  %fp_max = torch.constant.float 6.432100e+00
  %0 = torch.aten.clamp %arg0, %fp_min, %fp_max : !torch.vtensor<[1,1,128,128],f32>, !torch.float, !torch.float -> !torch.vtensor<[1,1,128,128],f32>
  return %0 : !torch.vtensor<[1,1,128,128],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.masked_fill.Scalar(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[1,12,128,128],f32>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: !torch.vtensor<[1,1,128,128],i1>) -> !torch.vtensor<[1,12,128,128],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,12,128,128],f32> -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[VAL_7:.*]] = tosa.select %[[VAL_3]], %[[VAL_6]], %[[VAL_2]] : (tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<1x12x128x128xf32> -> !torch.vtensor<[1,12,128,128],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[1,12,128,128],f32>
// CHECK:         }
func.func @torch.aten.masked_fill.Scalar(%arg0: !torch.vtensor<[1,12,128,128],f32>, %arg1: !torch.vtensor<[1,1,128,128],i1>) -> !torch.vtensor<[1,12,128,128],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.masked_fill.Scalar %arg0, %arg1, %int0 : !torch.vtensor<[1,12,128,128],f32>, !torch.vtensor<[1,1,128,128],i1>, !torch.int -> !torch.vtensor<[1,12,128,128],f32>
  return %0 : !torch.vtensor<[1,12,128,128],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.masked_fill.Tensor(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[1,12,128,128],f32>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: !torch.vtensor<[1,1,128,128],i1>,
// CHECK-SAME:                                             %[[VAL_2:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,12,128,128],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,12,128,128],f32> -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_4]], %[[VAL_5]], %[[VAL_3]] : (tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1x12x128x128xf32> -> !torch.vtensor<[1,12,128,128],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1,12,128,128],f32>
// CHECK:         }
func.func @torch.aten.masked_fill.Tensor(%arg0: !torch.vtensor<[1,12,128,128],f32>, %arg1: !torch.vtensor<[1,1,128,128],i1>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,12,128,128],f32> {
  %0 = torch.aten.masked_fill.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[1,12,128,128],f32>, !torch.vtensor<[1,1,128,128],i1>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,12,128,128],f32>
  return %0 : !torch.vtensor<[1,12,128,128],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.abs(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.vtensor<[15,15],si64>) -> !torch.vtensor<[15,15],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[15,15],si64> -> tensor<15x15xi64>
// CHECK:           %[[VAL_2:.*]] = tosa.abs %[[VAL_1]] : (tensor<15x15xi64>) -> tensor<15x15xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<15x15xi64> -> !torch.vtensor<[15,15],si64>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[15,15],si64>
// CHECK:         }
func.func @torch.aten.abs(%arg0: !torch.vtensor<[15,15],si64>) -> !torch.vtensor<[15,15],si64>{
  %0 = torch.aten.abs %arg0 : !torch.vtensor<[15,15],si64> -> !torch.vtensor<[15,15],si64>
  return %0 : !torch.vtensor<[15,15],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.where.self(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[1,1,5,5],i1>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[1,12,5,5],f32>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,12,5,5],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,5,5],i1> -> tensor<1x1x5x5xi1>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,12,5,5],f32> -> tensor<1x12x5x5xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) -> tensor<1x12x5x5xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1x12x5x5xf32> -> !torch.vtensor<[1,12,5,5],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1,12,5,5],f32>
// CHECK:         }
func.func @torch.aten.where.self(%arg0: !torch.vtensor<[1,1,5,5],i1>, %arg1: !torch.vtensor<[1,12,5,5],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,12,5,5],f32> {
  %0 = torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[1,1,5,5],i1>, !torch.vtensor<[1,12,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,12,5,5],f32>
  return %0 : !torch.vtensor<[1,12,5,5],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.remainder.Scalar(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_3]] : (tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_1]], %[[VAL_4]] {shift = 0 : i8} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.floor %[[VAL_5]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.mul %[[VAL_3]], %[[VAL_6]] {shift = 0 : i8} : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.sub %[[VAL_1]], %[[VAL_7]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<2x4xf32> -> !torch.vtensor<[2,4],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[2,4],f32>
// CHECK:         }
func.func @torch.aten.remainder.Scalar(%arg0: !torch.vtensor<[2, 4],f32>) -> !torch.vtensor<[2, 4],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.remainder.Scalar %arg0, %int2 : !torch.vtensor<[2, 4],f32>, !torch.int -> !torch.vtensor<[2, 4],f32>
  return %0 : !torch.vtensor<[2, 4],f32>
}

// -----

// CHECK-LABEL:   func.func @forward(
// CHECK-SAME:                       %[[VAL_0:.*]]: !torch.vtensor<[5,5],f32>,
// CHECK-SAME:                       %[[VAL_1:.*]]: !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[5,5],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,5],f32> -> tensor<5x5xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[5,5],f32> -> tensor<5x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.float 1.000000e-08
// CHECK:           %[[VAL_5:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = tosa.sub %[[VAL_2]], %[[VAL_3]] : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.abs %[[VAL_7]] : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.abs %[[VAL_3]] : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_10]], %[[VAL_9]] {shift = 0 : i8} : (tensor<f32>, tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{value = dense<9.99999993E-9> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_13:.*]] = tosa.add %[[VAL_12]], %[[VAL_11]] : (tensor<f32>, tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.greater_equal %[[VAL_13]], %[[VAL_8]] : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xi1>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_14]] : tensor<5x5xi1> -> !torch.vtensor<[5,5],i1>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[5,5],i1>
// CHECK:         }
func.func @forward(%arg0: !torch.vtensor<[5,5],f32>, %arg1: !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[5,5],i1> {
  %float1.000000e-08 = torch.constant.float 1.000000e-08
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %false = torch.constant.bool false
  %0 = torch.aten.isclose %arg0, %arg1, %float1.000000e-05, %float1.000000e-08, %false : !torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[5,5],i1>
  return %0 : !torch.vtensor<[5,5],i1>
}
