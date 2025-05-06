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
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.clamp %[[VAL_1]] {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.relu$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.leaky_relu$basic(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1.000000e-01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_7]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_8]] : (tensor<?x?xf32>, tensor<1x1xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_1]], %[[VAL_5]], %[[VAL_10]] : (tensor<?x?xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.select %[[VAL_9]], %[[VAL_1]], %[[VAL_11]] : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[?,?],f32>
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
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.negate %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
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
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_9:.*]] = tosa.mul %[[VAL_2]], %[[VAL_7]], %[[VAL_8]] : (tensor<?x?xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.add %[[VAL_3]], %[[VAL_9]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[?,?],f32>
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
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_9:.*]] = tosa.mul %[[VAL_2]], %[[VAL_7]], %[[VAL_8]] : (tensor<?x?xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.sub %[[VAL_3]], %[[VAL_9]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[?,?],f32>
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
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_3]], %[[VAL_2]], %[[VAL_4]] : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.mul$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32> -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.div$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32> -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
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

// CHECK-LABEL:   func.func @test_reduce_mean_dim$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[3,4,5,6],f32>) -> !torch.vtensor<[4,5,6],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5,6],f32> -> tensor<3x4x5x6xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 0 : i32} : (tensor<3x4x5x6xf32>) -> tensor<1x4x5x6xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[4, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_7]] : (tensor<1x4x5x6xf32>, !tosa.shape<3>) -> tensor<4x5x6xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0.333333343> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_10]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_13:.*]] = tosa.mul %[[VAL_8]], %[[VAL_11]], %[[VAL_12]] : (tensor<4x5x6xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<4x5x6xf32>
// CHECK:           %[[VAL_14:.*]] = torch_c.from_builtin_tensor %[[VAL_13]] : tensor<4x5x6xf32> -> !torch.vtensor<[4,5,6],f32>
// CHECK:           return %[[VAL_14]] : !torch.vtensor<[4,5,6],f32>
// CHECK:         }
func.func @test_reduce_mean_dim$basic(%arg0: !torch.vtensor<[3,4,5,6],f32>) -> !torch.vtensor<[4,5,6],f32> {
  %dim0 = torch.constant.int 0
  %reducedims = torch.prim.ListConstruct %dim0 : (!torch.int) -> !torch.list<int>
  %keepdims = torch.constant.bool false
  %dtype = torch.constant.none
  %0 = torch.aten.mean.dim %arg0, %reducedims, %keepdims, %dtype : !torch.vtensor<[3,4,5,6],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,5,6],f32>
  return %0 : !torch.vtensor<[4,5,6],f32>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_sum_dims$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[3,4,5,6],f32>) -> !torch.vtensor<[4,5,6],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5,6],f32> -> tensor<3x4x5x6xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 0 : i32} : (tensor<3x4x5x6xf32>) -> tensor<1x4x5x6xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[4, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_7]] : (tensor<1x4x5x6xf32>, !tosa.shape<3>) -> tensor<4x5x6xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<4x5x6xf32> -> !torch.vtensor<[4,5,6],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[4,5,6],f32>
// CHECK:         }
func.func @test_reduce_sum_dims$basic(%arg0: !torch.vtensor<[3,4,5,6],f32>) -> !torch.vtensor<[4,5,6],f32> {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[3,4,5,6],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,5,6],f32>
    return %1 : !torch.vtensor<[4,5,6],f32>
}

// -----

// CHECK-LABEL:   func.func @test_linalg_vector_norm$basic(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[3,151,64],f32>) -> !torch.vtensor<[3,151,1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,151,64],f32> -> tensor<3x151x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_3]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_8]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.abs %[[VAL_1]] : (tensor<3x151x64xf32>) -> tensor<3x151x64xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.pow %[[VAL_10]], %[[VAL_9]] : (tensor<3x151x64xf32>, tensor<1x1x1xf32>) -> tensor<3x151x64xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.reduce_sum %[[VAL_11]] {axis = 2 : i32} : (tensor<3x151x64xf32>) -> tensor<3x151x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.reciprocal %[[VAL_7]] : (tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_13]], %[[VAL_14]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.pow %[[VAL_12]], %[[VAL_15]] : (tensor<3x151x1xf32>, tensor<1x1x1xf32>) -> tensor<3x151x1xf32>
// CHECK:           %[[VAL_17:.*]] = torch_c.from_builtin_tensor %[[VAL_16]] : tensor<3x151x1xf32> -> !torch.vtensor<[3,151,1],f32>
// CHECK:           return %[[VAL_17]] : !torch.vtensor<[3,151,1],f32>
// CHECK:         }
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
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 0 : i32} : (tensor<?x?x?x?xf32>) -> tensor<1x?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_sum %[[VAL_3]] {axis = 1 : i32} : (tensor<1x?x?x?xf32>) -> tensor<1x1x?x?xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.reduce_sum %[[VAL_4]] {axis = 2 : i32} : (tensor<1x1x?x?xf32>) -> tensor<1x1x1x?xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.reduce_sum %[[VAL_5]] {axis = 3 : i32} : (tensor<1x1x1x?xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_7]] : (tensor<1x1x1x1xf32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[1],f32>
// CHECK:         }
func.func @test_reduce_sum$basic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.none -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_all$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[1],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],i1> -> tensor<?x?x?x?xi1>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_all %[[VAL_1]] {axis = 0 : i32} : (tensor<?x?x?x?xi1>) -> tensor<1x?x?x?xi1>
// CHECK:           %[[VAL_3:.*]] = tosa.reduce_all %[[VAL_2]] {axis = 1 : i32} : (tensor<1x?x?x?xi1>) -> tensor<1x1x?x?xi1>
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_all %[[VAL_3]] {axis = 2 : i32} : (tensor<1x1x?x?xi1>) -> tensor<1x1x1x?xi1>
// CHECK:           %[[VAL_5:.*]] = tosa.reduce_all %[[VAL_4]] {axis = 3 : i32} : (tensor<1x1x1x?xi1>) -> tensor<1x1x1x1xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<1x1x1x1xi1>, !tosa.shape<1>) -> tensor<1xi1>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<1xi1> -> !torch.vtensor<[1],i1>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[1],i1>
// CHECK:         }
func.func @test_reduce_all$basic(%arg0: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[1],i1> {
  %0 = torch.aten.all %arg0 : !torch.vtensor<[?,?,?,?],i1> -> !torch.vtensor<[1],i1>
  return %0 : !torch.vtensor<[1],i1>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_any_dim$basic(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[3,4,5,6],i1>) -> !torch.vtensor<[4,5,6],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5,6],i1> -> tensor<3x4x5x6xi1>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_any %[[VAL_1]] {axis = 0 : i32} : (tensor<3x4x5x6xi1>) -> tensor<1x4x5x6xi1>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[4, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_5]] : (tensor<1x4x5x6xi1>, !tosa.shape<3>) -> tensor<4x5x6xi1>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x5x6xi1> -> !torch.vtensor<[4,5,6],i1>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[4,5,6],i1>
// CHECK:         }
func.func @test_reduce_any_dim$basic(%arg0: !torch.vtensor<[3,4,5,6],i1>) -> !torch.vtensor<[4,5,6],i1> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.any.dim %arg0, %int0, %false : !torch.vtensor<[3,4,5,6],i1>, !torch.int, !torch.bool -> !torch.vtensor<[4,5,6],i1>
  return %0 : !torch.vtensor<[4,5,6],i1>
}

// -----

// CHECK-LABEL:   func.func @test_reduce_any$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],i1>) -> !torch.vtensor<[1],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],i1> -> tensor<?x?x?x?xi1>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_any %[[VAL_1]] {axis = 0 : i32} : (tensor<?x?x?x?xi1>) -> tensor<1x?x?x?xi1>
// CHECK:           %[[VAL_3:.*]] = tosa.reduce_any %[[VAL_2]] {axis = 1 : i32} : (tensor<1x?x?x?xi1>) -> tensor<1x1x?x?xi1>
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_any %[[VAL_3]] {axis = 2 : i32} : (tensor<1x1x?x?xi1>) -> tensor<1x1x1x?xi1>
// CHECK:           %[[VAL_5:.*]] = tosa.reduce_any %[[VAL_4]] {axis = 3 : i32} : (tensor<1x1x1x?xi1>) -> tensor<1x1x1x1xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<1x1x1x1xi1>, !tosa.shape<1>) -> tensor<1xi1>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<1xi1> -> !torch.vtensor<[1],i1>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[1],i1>
// CHECK:         }
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
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
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
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
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
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<3.123400e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.pow %[[VAL_1]], %[[VAL_5]] : (tensor<?x?xf32>, tensor<1x1xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?],f32>
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
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<3.123400e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<6.432100e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_8]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_1]], %[[VAL_9]], %[[VAL_10]] : (tensor<?x?xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.sub %[[VAL_7]], %[[VAL_11]] : (tensor<1x1xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.rsub.Scalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %other = torch.constant.float 3.123400e+00
  %alpha = torch.constant.float 6.432100e+00
  %0 = torch.aten.rsub.Scalar %arg0, %other, %alpha : !torch.vtensor<[?,?],f32>, !torch.float, !torch.float -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.rsub.Scalar$float_int(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<3.123400e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_8]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_1]], %[[VAL_9]], %[[VAL_10]] : (tensor<?x?xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.sub %[[VAL_7]], %[[VAL_11]] : (tensor<1x1xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.rsub.Scalar$float_int(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %other = torch.constant.float 3.123400e+00
  %alpha = torch.constant.int 1
  %0 = torch.aten.rsub.Scalar %arg0, %other, %alpha : !torch.vtensor<[?,?],f32>, !torch.float, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gt.Tensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
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
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
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
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
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
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<-1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_4]] : (tensor<?x?x?x?xf32>, !tosa.shape<1>) -> tensor<?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?],f32>
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
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<[3.000000e+00, 2.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[VAL_5:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_7:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[4, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_8]] : (tensor<4xf32>, !tosa.shape<2>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[4, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_10]] : (tensor<4xf32>, !tosa.shape<2>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.const_shape  {values = dense<[4, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_12]] : (tensor<4xf32>, !tosa.shape<2>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[4, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_14]] : (tensor<4xf32>, !tosa.shape<2>) -> tensor<4x1xf32>
// CHECK:           %[[VAL_16:.*]] = "tosa.const"() <{values = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[1, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_17]] : (tensor<4x1xf32>, !tosa.shape<3>) -> tensor<1x4x1xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<[1, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_20:.*]] = tosa.reshape %[[VAL_11]], %[[VAL_19]] : (tensor<4x1xf32>, !tosa.shape<3>) -> tensor<1x4x1xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_22:.*]] = tosa.reshape %[[VAL_16]], %[[VAL_21]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.const_shape  {values = dense<[1, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_24:.*]] = tosa.reshape %[[VAL_13]], %[[VAL_23]] : (tensor<4x1xf32>, !tosa.shape<3>) -> tensor<1x4x1xf32>
// CHECK:           %[[VAL_25:.*]] = tosa.const_shape  {values = dense<[1, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_26:.*]] = tosa.reshape %[[VAL_15]], %[[VAL_25]] : (tensor<4x1xf32>, !tosa.shape<3>) -> tensor<1x4x1xf32>
// CHECK:           %[[VAL_27:.*]] = tosa.sub %[[VAL_1]], %[[VAL_18]] : (tensor<10x4x3xf32>, tensor<1x4x1xf32>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_28:.*]] = tosa.add %[[VAL_20]], %[[VAL_22]] : (tensor<1x4x1xf32>, tensor<1x1x1xf32>) -> tensor<1x4x1xf32>
// CHECK:           %[[VAL_29:.*]] = tosa.rsqrt %[[VAL_28]] : (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>
// CHECK:           %[[VAL_30:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_31:.*]] = tosa.mul %[[VAL_27]], %[[VAL_29]], %[[VAL_30]] : (tensor<10x4x3xf32>, tensor<1x4x1xf32>, tensor<1xi8>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_32:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_33:.*]] = tosa.mul %[[VAL_31]], %[[VAL_24]], %[[VAL_32]] : (tensor<10x4x3xf32>, tensor<1x4x1xf32>, tensor<1xi8>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_34:.*]] = tosa.add %[[VAL_33]], %[[VAL_26]] : (tensor<10x4x3xf32>, tensor<1x4x1xf32>) -> tensor<10x4x3xf32>
// CHECK:           %[[VAL_35:.*]] = torch_c.from_builtin_tensor %[[VAL_34]] : tensor<10x4x3xf32> -> !torch.vtensor<[10,4,3],f32>
// CHECK:           return %[[VAL_35]] : !torch.vtensor<[10,4,3],f32>
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

// CHECK-LABEL:   func.func @torch.aten.flatten.using_ints$basic(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !torch.vtensor<[10,3,8,9,3,4],f32>) -> !torch.vtensor<[10,3,?,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,3,8,9,3,4],f32> -> tensor<10x3x8x9x3x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[10, 3, 216, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_4]] : (tensor<10x3x8x9x3x4xf32>, !tosa.shape<4>) -> tensor<10x3x216x4xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.cast %[[VAL_5]] : tensor<10x3x216x4xf32> to tensor<10x3x?x4xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<10x3x?x4xf32> -> !torch.vtensor<[10,3,?,4],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[10,3,?,4],f32>
// CHECK:         }
func.func @torch.aten.flatten.using_ints$basic(%arg0: !torch.vtensor<[10,3,8,9,3,4],f32> ) -> !torch.vtensor<[10,3,?,4],f32> {
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %0 = torch.aten.flatten.using_ints %arg0, %int2, %int4 : !torch.vtensor<[10,3,8,9,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[10,3,?,4],f32>
  return %0 : !torch.vtensor<[10,3,?,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unflatten.int$basic(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[1,6,4],f32>) -> !torch.vtensor<[1,2,3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,6,4],f32> -> tensor<1x6x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_6]] : (tensor<1x6x4xf32>, !tosa.shape<4>) -> tensor<1x2x3x4xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<1x2x3x4xf32> -> !torch.vtensor<[1,2,3,4],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[1,2,3,4],f32>
// CHECK:         }
func.func @torch.aten.unflatten.int$basic(%arg0: !torch.vtensor<[1,6,4],f32> ) -> !torch.vtensor<[1,2,3,4],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.unflatten.int %arg0, %int1, %0 : !torch.vtensor<[1,6,4],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,3,4],f32>
  return %1 : !torch.vtensor<[1,2,3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.native_layer_norm$basic(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[5,2,2,3],f32>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: !torch.vtensor<[2,2,3],f32>,
// CHECK-SAME:                                                  %[[VAL_2:.*]]: !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[5,2,2,3],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[2,2,3],f32> -> tensor<2x2x3xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,2,3],f32> -> tensor<2x2x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,2,2,3],f32> -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.float 5.000000e-01
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_8]], %[[VAL_7]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<1.200000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reciprocal %[[VAL_10]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_11]], %[[VAL_12]] : (tensor<1xf32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.reduce_sum %[[VAL_5]] {axis = 3 : i32} : (tensor<5x2x2x3xf32>) -> tensor<5x2x2x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.reduce_sum %[[VAL_14]] {axis = 2 : i32} : (tensor<5x2x2x1xf32>) -> tensor<5x2x1x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.reduce_sum %[[VAL_15]] {axis = 1 : i32} : (tensor<5x2x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[5, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_16]], %[[VAL_17]] : (tensor<5x1x1x1xf32>, !tosa.shape<4>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_19:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_20:.*]] = tosa.mul %[[VAL_18]], %[[VAL_13]], %[[VAL_19]] : (tensor<5x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.sub %[[VAL_5]], %[[VAL_20]] : (tensor<5x2x2x3xf32>, tensor<5x1x1x1xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_23:.*]] = tosa.mul %[[VAL_21]], %[[VAL_21]], %[[VAL_22]] : (tensor<5x2x2x3xf32>, tensor<5x2x2x3xf32>, tensor<1xi8>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_24:.*]] = tosa.reduce_sum %[[VAL_23]] {axis = 3 : i32} : (tensor<5x2x2x3xf32>) -> tensor<5x2x2x1xf32>
// CHECK:           %[[VAL_25:.*]] = tosa.reduce_sum %[[VAL_24]] {axis = 2 : i32} : (tensor<5x2x2x1xf32>) -> tensor<5x2x1x1xf32>
// CHECK:           %[[VAL_26:.*]] = tosa.reduce_sum %[[VAL_25]] {axis = 1 : i32} : (tensor<5x2x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_27:.*]] = tosa.const_shape  {values = dense<[5, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_28:.*]] = tosa.reshape %[[VAL_26]], %[[VAL_27]] : (tensor<5x1x1x1xf32>, !tosa.shape<4>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_29:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_30:.*]] = tosa.mul %[[VAL_28]], %[[VAL_13]], %[[VAL_29]] : (tensor<5x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_31:.*]] = tosa.const_shape  {values = dense<[1, 2, 2, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_32:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_31]] : (tensor<2x2x3xf32>, !tosa.shape<4>) -> tensor<1x2x2x3xf32>
// CHECK:           %[[VAL_33:.*]] = tosa.const_shape  {values = dense<[1, 2, 2, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_34:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_33]] : (tensor<2x2x3xf32>, !tosa.shape<4>) -> tensor<1x2x2x3xf32>
// CHECK:           %[[VAL_35:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_36:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_37:.*]] = tosa.reshape %[[VAL_35]], %[[VAL_36]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_38:.*]] = tosa.sub %[[VAL_5]], %[[VAL_20]] : (tensor<5x2x2x3xf32>, tensor<5x1x1x1xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_39:.*]] = tosa.add %[[VAL_30]], %[[VAL_37]] : (tensor<5x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_40:.*]] = tosa.rsqrt %[[VAL_39]] : (tensor<5x1x1x1xf32>) -> tensor<5x1x1x1xf32>
// CHECK:           %[[VAL_41:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_42:.*]] = tosa.mul %[[VAL_38]], %[[VAL_40]], %[[VAL_41]] : (tensor<5x2x2x3xf32>, tensor<5x1x1x1xf32>, tensor<1xi8>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_43:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_44:.*]] = tosa.mul %[[VAL_42]], %[[VAL_32]], %[[VAL_43]] : (tensor<5x2x2x3xf32>, tensor<1x2x2x3xf32>, tensor<1xi8>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_45:.*]] = tosa.add %[[VAL_44]], %[[VAL_34]] : (tensor<5x2x2x3xf32>, tensor<1x2x2x3xf32>) -> tensor<5x2x2x3xf32>
// CHECK:           %[[VAL_46:.*]] = torch_c.from_builtin_tensor %[[VAL_45]] : tensor<5x2x2x3xf32> -> !torch.vtensor<[5,2,2,3],f32>
// CHECK:           return %[[VAL_46]] : !torch.vtensor<[5,2,2,3],f32>
// CHECK:         }
func.func @torch.aten.native_layer_norm$basic(%arg0: !torch.vtensor<[5,2,2,3],f32> , %arg1: !torch.vtensor<[2,2,3],f32> , %arg2: !torch.vtensor<[2,2,3],f32> ) -> !torch.vtensor<[5,2,2,3],f32> {
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
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
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
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK:           %[[VAL_4:.*]] = tosa.logical_or %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.logical_or$basic(%arg0: !torch.vtensor<[?,?],i1>, %arg1: !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.logical_or %arg0, %arg1 : !torch.vtensor<[?,?],i1>, !torch.vtensor<[?,?],i1> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.permute$basic(
// CHECK-SAME:                                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[3,4,2],f32>) -> !torch.vtensor<[3,2,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,2],f32> -> tensor<3x4x2xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_3]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 1>} : (tensor<3x4x2xf32>) -> tensor<3x2x4xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x2x4xf32> -> !torch.vtensor<[3,2,4],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,2,4],f32>
// CHECK:         }
func.func @torch.aten.permute$basic(%arg0: !torch.vtensor<[3,4,2],f32> ) -> !torch.vtensor<[3,2,4],f32> {
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
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
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
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0.693147182> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.log %[[VAL_1]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_4]], %[[VAL_3]], %[[VAL_5]] : (tensor<?x?xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?],f32>
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
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
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
// CHECK:           %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[4, 3, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_3]] : (tensor<4x3xi32>, !tosa.shape<3>) -> tensor<4x3x1xi32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x3x1xi32> -> !torch.vtensor<[4,3,1],si32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,3,1],si32>
// CHECK:         }
func.func @torch.aten.unsqueeze$basic(%arg0: !torch.vtensor<[4,3],si32> ) -> !torch.vtensor<[4,3,1],si32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.vtensor<[4,3],si32>, !torch.int -> !torch.vtensor<[4,3,1],si32>
  return %0 : !torch.vtensor<[4,3,1],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$negative_dim(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !torch.vtensor<[4,3],si32>) -> !torch.vtensor<[4,3,1],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,3],si32> -> tensor<4x3xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[4, 3, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_3]] : (tensor<4x3xi32>, !tosa.shape<3>) -> tensor<4x3x1xi32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x3x1xi32> -> !torch.vtensor<[4,3,1],si32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,3,1],si32>
// CHECK:         }
func.func @torch.aten.unsqueeze$negative_dim(%arg0: !torch.vtensor<[4,3],si32> ) -> !torch.vtensor<[4,3,1],si32> {
  %int2 = torch.constant.int -1
  %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.vtensor<[4,3],si32>, !torch.int -> !torch.vtensor<[4,3,1],si32>
  return %0 : !torch.vtensor<[4,3,1],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.contiguous$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[?,?],f32>
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
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
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
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.dropout$basic(%arg0: !torch.vtensor<[?,?],f32> ) -> !torch.vtensor<[?,?],f32> {
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %0 = torch.aten.dropout %arg0, %float0.000000e00, %false : !torch.vtensor<[?,?],f32>, !torch.float, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d$basic(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 7
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = torch.constant.none
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x7x7xf32>) -> tensor<1x7x7x512xf32>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.avg_pool2d %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] {acc_type = f32, kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1x512xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.transpose %[[VAL_13]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x1x1x512xf32>) -> tensor<1x512x1x1xf32>
// CHECK:           %[[VAL_15:.*]] = tensor.cast %[[VAL_14]] : tensor<1x512x1x1xf32> to tensor<1x512x1x1xf32>
// CHECK:           %[[VAL_16:.*]] = torch_c.from_builtin_tensor %[[VAL_15]] : tensor<1x512x1x1xf32> -> !torch.vtensor<[1,512,1,1],f32>
// CHECK:           return %[[VAL_16]] : !torch.vtensor<[1,512,1,1],f32>
// CHECK:         }
func.func @torch.aten.avg_pool2d$basic(%arg0: !torch.vtensor<[1,512,7,7],f32> ) -> !torch.vtensor<[1,512,1,1],f32> {
  %int7 = torch.constant.int 7
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %none = torch.constant.none
  %kernel = torch.prim.ListConstruct %int7, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.avg_pool2d %arg0, %kernel, %stride, %padding, %false, %false, %none : !torch.vtensor<[1,512,7,7],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,1,1],f32>
  return %0 : !torch.vtensor<[1,512,1,1],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max.dim$basic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,2,3],f32> -> tensor<3x2x3xf32>
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_6:.*]] = tosa.reduce_max %[[VAL_2]] {axis = 2 : i32} : (tensor<3x2x3xf32>) -> tensor<3x2x1xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x2x1xf32> -> !torch.vtensor<[3,2,1],f32>
// CHECK:           %[[VAL_8:.*]] = tosa.argmax %[[VAL_2]] {axis = 2 : i32} : (tensor<3x2x3xf32>) -> tensor<3x2xi64>
// CHECK:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[3, 2, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_9]] : (tensor<3x2xi64>, !tosa.shape<3>) -> tensor<3x2x1xi64>
// CHECK:           %[[VAL_11:.*]] = torch_c.to_builtin_tensor %[[VAL_7]] : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
// CHECK:           return %[[VAL_11]] : tensor<3x2x1xf32>
// CHECK:         }
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
// CHECK: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1x512xi64>}> : () -> tensor<1x512xi64>
// CHECK: %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x512xi64> -> !torch.vtensor<[1,512],si64>
// CHECK: return %[[VAL_1]] : !torch.vtensor<[1,512],si64>
func.func @torch.vtensor.literal_si64$basic() -> !torch.vtensor<[1,512],si64> {
  %0 = torch.vtensor.literal(dense<-1> : tensor<1x512xsi64>) : !torch.vtensor<[1,512],si64>
  return %0 : !torch.vtensor<[1,512],si64>
}

// -----

// CHECK-LABEL: @torch.vtensor.literal_si32$basic(
// CHECK: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1x512xi32>}> : () -> tensor<1x512xi32>
// CHECK: %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<1x512xi32> -> !torch.vtensor<[1,512],si32>
// CHECK: return %[[VAL_1]] : !torch.vtensor<[1,512],si32>
func.func @torch.vtensor.literal_si32$basic() -> !torch.vtensor<[1,512],si32> {
  %0 = torch.vtensor.literal(dense<-1> : tensor<1x512xsi32>) : !torch.vtensor<[1,512],si32>
  return %0 : !torch.vtensor<[1,512],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.arange.start_step() -> !torch.vtensor<[5],si64> {
// CHECK:           %[[VAL_0:.*]] = torch.constant.none
// CHECK:           %[[VAL_1:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4]> : tensor<5xi64>}> : () -> tensor<5xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<5xi64> -> !torch.vtensor<[5],si64>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[5],si64>
// CHECK:         }
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
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<1> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<[],si64>
func.func @torch.prim.NumToTensor.Scalar() -> !torch.vtensor<[],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.copy(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[1,1,5,5],ui8>) -> !torch.vtensor<[1,1,5,5],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,5,5],ui8> -> tensor<1x1x5x5xi8>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 11
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_9:.*]] = tosa.cast %[[VAL_8]] : (tensor<i64>) -> tensor<i1>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_11]] : (tensor<i1>, !tosa.shape<4>) -> tensor<1x1x1x1xi1>
// CHECK:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[1, 1, 5, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_14:.*]] = tosa.tile %[[VAL_12]], %[[VAL_13]] : (tensor<1x1x1x1xi1>, !tosa.shape<4>) -> tensor<1x1x5x5xi1>
// CHECK:           %[[VAL_15:.*]] = tosa.cast %[[VAL_1]] : (tensor<1x1x5x5xi8>) -> tensor<1x1x5x5xi1>
// CHECK:           %[[VAL_16:.*]] = torch_c.from_builtin_tensor %[[VAL_15]] : tensor<1x1x5x5xi1> -> !torch.vtensor<[1,1,5,5],i1>
// CHECK:           return %[[VAL_16]] : !torch.vtensor<[1,1,5,5],i1>
// CHECK:         }
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
// CHECK-LABEL:   func.func @torch.aten.to.dtype$toBool(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[3,5],si64>) -> !torch.vtensor<[3,5],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5],si64> -> tensor<3x5xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 11
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x5xi64>) -> tensor<3x5xi1>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<3x5xi1> -> !torch.vtensor<[3,5],i1>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[3,5],i1>
// CHECK:         }
func.func @torch.aten.to.dtype$toBool(%arg0: !torch.vtensor<[3,5],si64>) -> !torch.vtensor<[3,5],i1> {
  %int11 = torch.constant.int 11
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.to.dtype %arg0, %int11, %false, %false, %none : !torch.vtensor<[3,5],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,5],i1>
  return %0 : !torch.vtensor<[3,5],i1>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.to.dtype$fromBool(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[1,128],i1>) -> !torch.vtensor<[1,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,128],i1> -> tensor<1x128xi1>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_1]] : (tensor<1x128xi1>) -> tensor<1x128xi64>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<1x128xi64> -> !torch.vtensor<[1,128],si64>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[1,128],si64>
// CHECK:         }
func.func @torch.aten.to.dtype$fromBool(%arg0: !torch.vtensor<[1,128],i1>) -> !torch.vtensor<[1,128],si64> {
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.to.dtype %arg0, %int4, %false, %false, %none : !torch.vtensor<[1,128],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,128],si64>
  return %0 : !torch.vtensor<[1,128],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.to.dtype$floatToInt(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3,5],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5],f32> -> tensor<3x5xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.none
// CHECK:           %[[VAL_5:.*]] = tosa.floor %[[VAL_1]] : (tensor<3x5xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.ceil %[[VAL_1]] : (tensor<3x5xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_8]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.greater %[[VAL_9]], %[[VAL_1]] : (tensor<1x1xf32>, tensor<3x5xf32>) -> tensor<3x5xi1>
// CHECK:           %[[VAL_11:.*]] = tosa.select %[[VAL_10]], %[[VAL_6]], %[[VAL_5]] : (tensor<3x5xi1>, tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.cast %[[VAL_11]] : (tensor<3x5xf32>) -> tensor<3x5xi64>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<3x5xi64> -> !torch.vtensor<[3,5],si64>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[3,5],si64>
// CHECK:         }
func.func @torch.aten.to.dtype$floatToInt(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3,5],si64> {
    %int4 = torch.constant.int 4
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.to.dtype %arg0, %int4, %false, %false, %none : !torch.vtensor<[3,5],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,5],si64>
    return %0 : !torch.vtensor<[3,5],si64>
  }

// -----
// CHECK-LABEL:   func.func @torch.aten.gather(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[1,4,3],f32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,4,2],si64> -> tensor<1x4x2xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,4,3],f32> -> tensor<1x4x3xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_5:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_2]] : (tensor<1x4x2xi64>) -> tensor<1x4x2xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[1, 4, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_7]] : (tensor<1x4x2xi32>, !tosa.shape<4>) -> tensor<1x4x2x1xi32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x2x1xi32>}> : () -> tensor<1x4x2x1xi32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}0], [0]], {{\[\[}}1], [1]], {{\[\[}}2], [2]], {{\[\[}}3], [3]]]]> : tensor<1x4x2x1xi32>}> : () -> tensor<1x4x2x1xi32>
// CHECK:           %[[VAL_11:.*]] = tosa.concat %[[VAL_9]], %[[VAL_10]], %[[VAL_8]] {axis = 3 : i32} : (tensor<1x4x2x1xi32>, tensor<1x4x2x1xi32>, tensor<1x4x2x1xi32>) -> tensor<1x4x2x3xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.const_shape  {values = dense<[1, 12, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_12]] : (tensor<1x4x3xf32>, !tosa.shape<3>) -> tensor<1x12x1xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[8, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_11]], %[[VAL_14]] : (tensor<1x4x2x3xi32>, !tosa.shape<2>) -> tensor<8x3xi32>
// CHECK:           %[[VAL_16:.*]] = "tosa.const"() <{values = dense<[12, 3, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_16]], %[[VAL_17]] : (tensor<3xi32>, !tosa.shape<2>) -> tensor<1x3xi32>
// CHECK:           %[[VAL_19:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_20:.*]] = tosa.mul %[[VAL_15]], %[[VAL_18]], %[[VAL_19]] : (tensor<8x3xi32>, tensor<1x3xi32>, tensor<1xi8>) -> tensor<8x3xi32>
// CHECK:           %[[VAL_21:.*]] = tosa.reduce_sum %[[VAL_20]] {axis = 1 : i32} : (tensor<8x3xi32>) -> tensor<8x1xi32>
// CHECK:           %[[VAL_22:.*]] = tosa.const_shape  {values = dense<[1, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_23:.*]] = tosa.reshape %[[VAL_21]], %[[VAL_22]] : (tensor<8x1xi32>, !tosa.shape<2>) -> tensor<1x8xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.gather %[[VAL_13]], %[[VAL_23]] : (tensor<1x12x1xf32>, tensor<1x8xi32>) -> tensor<1x8x1xf32>
// CHECK:           %[[VAL_25:.*]] = tosa.const_shape  {values = dense<[1, 4, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_26:.*]] = tosa.reshape %[[VAL_24]], %[[VAL_25]] : (tensor<1x8x1xf32>, !tosa.shape<3>) -> tensor<1x4x2xf32>
// CHECK:           %[[VAL_27:.*]] = torch_c.from_builtin_tensor %[[VAL_26]] : tensor<1x4x2xf32> -> !torch.vtensor<[1,4,2],f32>
// CHECK:           return %[[VAL_27]] : !torch.vtensor<[1,4,2],f32>
// CHECK:         }
func.func @torch.aten.gather(%arg0: !torch.vtensor<[1,4,3],f32>, %arg1: !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %0 = torch.aten.gather %arg0, %int-1, %arg1, %false : !torch.vtensor<[1,4,3],f32>, !torch.int, !torch.vtensor<[1,4,2],si64>, !torch.bool -> !torch.vtensor<[1,4,2],f32>
  return %0 : !torch.vtensor<[1,4,2],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.add$int(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[2,2],si32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !torch.vtensor<[2,2],si32>) -> !torch.vtensor<[2,2],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,2],si32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,2],si32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_9:.*]] = tosa.mul %[[VAL_2]], %[[VAL_7]], %[[VAL_8]] : (tensor<2x2xi32>, tensor<1x1xi32>, tensor<1xi8>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_10:.*]] = tosa.add %[[VAL_3]], %[[VAL_9]] : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK:           %[[VAL_11:.*]] = tosa.cast %[[VAL_10]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
// CHECK:           %[[VAL_12:.*]] = torch_c.from_builtin_tensor %[[VAL_11]] : tensor<2x2xi64> -> !torch.vtensor<[2,2],si64>
// CHECK:           return %[[VAL_12]] : !torch.vtensor<[2,2],si64>
// CHECK:         }
func.func @torch.aten.add$int(%arg0: !torch.vtensor<[2, 2],si32>, %arg1: !torch.vtensor<[2, 2],si32>) -> !torch.vtensor<[2, 2],si64> {
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
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<256> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_5]] : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_8]] : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_6]], %[[VAL_9]], %[[VAL_10]] : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>, tensor<1xi8>) -> tensor<1x1x1x1xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.cast %[[VAL_1]] : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi32>
// CHECK:           %[[VAL_13:.*]] = tosa.add %[[VAL_12]], %[[VAL_11]] : (tensor<1x1x128x128xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x128x128xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.cast %[[VAL_13]] : (tensor<1x1x128x128xi32>) -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_14]] : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[1,1,128,128],si64>
// CHECK:         }
func.func @torch.aten.Scalar$basic(%arg0: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
  %int1 = torch.constant.int 1
  %int256 = torch.constant.int 256
  %0 = torch.aten.add.Scalar %arg0, %int256, %int1 : !torch.vtensor<[1,1,128,128],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],si64>
  return %0 : !torch.vtensor<[1,1,128,128],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.slice.negative_start(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,16,256],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 100
// CHECK:           %[[VAL_5:.*]] = torch.constant.int -16
// CHECK-DAG:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[0, 49, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[4, 16, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_8:.*]] = tosa.slice %[[VAL_1]], %[[VAL_6]], %[[VAL_7]] : (tensor<4x65x256xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x16x256xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<4x16x256xf32> -> !torch.vtensor<[4,16,256],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[4,16,256],f32>
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
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],si64> -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_val = 0 : i64, min_val = -9223372036854775808 : i64} : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi64>
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
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],si64>) -> !torch.vtensor<[1,1,128,128],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],si64> -> tensor<1x1x128x128xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi64>
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
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_val = 511 : i64, min_val = 0 : i64} : (tensor<1x1x128x128xi64>) -> tensor<1x1x128x128xi64>
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
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[1,1,128,128],f32>) -> !torch.vtensor<[1,1,128,128],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,128,128],f32> -> tensor<1x1x128x128xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.float 6.432100e+00
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_1]] {max_val = 6.432100e+00 : f32, min_val = 3.123400e+00 : f32} : (tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32>
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
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,12,128,128],f32> -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_7]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.select %[[VAL_2]], %[[VAL_8]], %[[VAL_3]] : (tensor<1x1x128x128xi1>, tensor<1x1x1x1xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<1x12x128x128xf32> -> !torch.vtensor<[1,12,128,128],f32>
// CHECK:           return %[[VAL_10]] : !torch.vtensor<[1,12,128,128],f32>
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
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,12,128,128],f32> -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.select %[[VAL_4]], %[[VAL_7]], %[[VAL_5]] : (tensor<1x1x128x128xi1>, tensor<1x1x1x1xf32>, tensor<1x12x128x128xf32>) -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<1x12x128x128xf32> -> !torch.vtensor<[1,12,128,128],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[1,12,128,128],f32>
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
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,12,5,5],f32> -> tensor<1x12x5x5xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,5,5],i1> -> tensor<1x1x5x5xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.select %[[VAL_5]], %[[VAL_4]], %[[VAL_7]] : (tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x5x5xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<1x12x5x5xf32> -> !torch.vtensor<[1,12,5,5],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[1,12,5,5],f32>
// CHECK:         }
func.func @torch.aten.where.self(%arg0: !torch.vtensor<[1,1,5,5],i1>, %arg1: !torch.vtensor<[1,12,5,5],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,12,5,5],f32> {
  %0 = torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[1,1,5,5],i1>, !torch.vtensor<[1,12,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,12,5,5],f32>
  return %0 : !torch.vtensor<[1,12,5,5],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.where.self_differing_rank_inputs(
// CHECK-SAME:                    %[[VAL_0:.*]]: !torch.vtensor<[5,4],i1>,
// CHECK-SAME:                    %[[VAL_1:.*]]: !torch.vtensor<[],f32>,
// CHECK-SAME:                    %[[VAL_2:.*]]: !torch.vtensor<[1,3,1,1,5,4],f32>) -> !torch.vtensor<[1,3,1,1,5,4],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1,3,1,1,5,4],f32> -> tensor<1x3x1x1x5x4xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,4],i1> -> tensor<5x4xi1>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[1, 1, 1, 1, 5, 4]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_8]] : (tensor<5x4xi1>, !tosa.shape<6>) -> tensor<1x1x1x1x5x4xi1>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<1> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_10]] : (tensor<1x1xf32>, !tosa.shape<6>) -> tensor<1x1x1x1x1x1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.select %[[VAL_9]], %[[VAL_11]], %[[VAL_3]] : (tensor<1x1x1x1x5x4xi1>, tensor<1x1x1x1x1x1xf32>, tensor<1x3x1x1x5x4xf32>) -> tensor<1x3x1x1x5x4xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<1x3x1x1x5x4xf32> -> !torch.vtensor<[1,3,1,1,5,4],f32>
// CHECK:           return %[[VAL_13]]
func.func @torch.aten.where.self_differing_rank_inputs(%40: !torch.vtensor<[5,4],i1>, %41: !torch.vtensor<[],f32>, %38 : !torch.vtensor<[1,3,1,1,5,4],f32>) -> (!torch.vtensor<[1,3,1,1,5,4],f32>) {
    %42 = torch.aten.where.self %40, %41, %38 : !torch.vtensor<[5,4],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[1,3,1,1,5,4],f32> -> !torch.vtensor<[1,3,1,1,5,4],f32>
    return %42: !torch.vtensor<[1,3,1,1,5,4],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.remainder.Scalar(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.reciprocal %[[VAL_5]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_8:.*]] = tosa.mul %[[VAL_1]], %[[VAL_6]], %[[VAL_7]] : (tensor<2x4xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.floor %[[VAL_8]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_5]], %[[VAL_9]], %[[VAL_10]] : (tensor<1x1xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.sub %[[VAL_1]], %[[VAL_11]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<2x4xf32> -> !torch.vtensor<[2,4],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[2,4],f32>
// CHECK:         }
func.func @torch.aten.remainder.Scalar(%arg0: !torch.vtensor<[2, 4],f32>) -> !torch.vtensor<[2, 4],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.remainder.Scalar %arg0, %int2 : !torch.vtensor<[2, 4],f32>, !torch.int -> !torch.vtensor<[2, 4],f32>
  return %0 : !torch.vtensor<[2, 4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.isclose$basic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[5,5],f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[5,5],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[5,5],f32> -> tensor<5x5xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,5],f32> -> tensor<5x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.float 1.000000e-08
// CHECK:           %[[VAL_5:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<9.99999993E-9> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_9]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_11]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.sub %[[VAL_3]], %[[VAL_2]] : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.abs %[[VAL_13]] : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.abs %[[VAL_2]] : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_16:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_17:.*]] = tosa.mul %[[VAL_10]], %[[VAL_15]], %[[VAL_16]] : (tensor<1x1xf32>, tensor<5x5xf32>, tensor<1xi8>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.add %[[VAL_12]], %[[VAL_17]] : (tensor<1x1xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.greater_equal %[[VAL_18]], %[[VAL_14]] : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xi1>
// CHECK:           %[[VAL_20:.*]] = torch_c.from_builtin_tensor %[[VAL_19]] : tensor<5x5xi1> -> !torch.vtensor<[5,5],i1>
// CHECK:           return %[[VAL_20]] : !torch.vtensor<[5,5],i1>
// CHECK:         }
func.func @torch.aten.isclose$basic(%arg0: !torch.vtensor<[5,5],f32>, %arg1: !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[5,5],i1> {
  %float1.000000e-08 = torch.constant.float 1.000000e-08
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %false = torch.constant.bool false
  %0 = torch.aten.isclose %arg0, %arg1, %float1.000000e-05, %float1.000000e-08, %false : !torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[5,5],i1>
  return %0 : !torch.vtensor<[5,5],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.__interpolate.size_list_scale_list.bilinear(
// CHECK-SAME:                                                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,16,135,240],f32>) -> !torch.vtensor<[1,16,270,480],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,16,135,240],f32> -> tensor<1x16x135x240xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.str "bilinear"
// CHECK:           %[[VAL_5:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_5]] : (!torch.float, !torch.float) -> !torch.list<float>
// CHECK:           %[[VAL_7:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x16x135x240xf32>) -> tensor<1x135x240x16xf32>
// CHECK-DAG:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[4, 2, 4, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<2> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_11:.*]] = tosa.resize %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] {mode = "BILINEAR"} : (tensor<1x135x240x16xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x270x480x16xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_11]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x270x480x16xf32>) -> tensor<1x16x270x480xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<1x16x270x480xf32> -> !torch.vtensor<[1,16,270,480],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[1,16,270,480],f32>
// CHECK:         }
func.func @torch.aten.__interpolate.size_list_scale_list.bilinear(%arg0: !torch.vtensor<[1,16,135,240],f32>) -> !torch.vtensor<[1,16,270,480],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %str = torch.constant.str "bilinear"
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.prim.ListConstruct %float2.000000e00, %float2.000000e00 : (!torch.float, !torch.float) -> !torch.list<float>
  %1 = torch.aten.__interpolate.size_list_scale_list %arg0, %none, %0, %str, %false, %none, %false : !torch.vtensor<[1,16,135,240],f32>, !torch.none, !torch.list<float>, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,16,270,480],f32>
  return %1 : !torch.vtensor<[1,16,270,480],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.__interpolate.size_list_scale_list.nearest(
// CHECK-SAME:                                                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,16,135,240],f32>) -> !torch.vtensor<[1,16,270,480],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,16,135,240],f32> -> tensor<1x16x135x240xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = torch.constant.str "nearest"
// CHECK:           %[[VAL_5:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_5]] : (!torch.float, !torch.float) -> !torch.list<float>
// CHECK:           %[[VAL_7:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x16x135x240xf32>) -> tensor<1x135x240x16xf32>
// CHECK-DAG:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[4, 2, 4, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<2> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_11:.*]] = tosa.resize %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] {mode = "NEAREST_NEIGHBOR"} : (tensor<1x135x240x16xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x270x480x16xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_11]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x270x480x16xf32>) -> tensor<1x16x270x480xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<1x16x270x480xf32> -> !torch.vtensor<[1,16,270,480],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[1,16,270,480],f32>
// CHECK:         }
func.func @torch.aten.__interpolate.size_list_scale_list.nearest(%arg0: !torch.vtensor<[1,16,135,240],f32>) -> !torch.vtensor<[1,16,270,480],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %str = torch.constant.str "nearest"
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.prim.ListConstruct %float2.000000e00, %float2.000000e00 : (!torch.float, !torch.float) -> !torch.list<float>
  %1 = torch.aten.__interpolate.size_list_scale_list %arg0, %none, %0, %str, %false, %none, %false : !torch.vtensor<[1,16,135,240],f32>, !torch.none, !torch.list<float>, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,16,270,480],f32>
  return %1 : !torch.vtensor<[1,16,270,480],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.tril$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[2,4],si32>) -> !torch.vtensor<[2,4],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,4],si32> -> tensor<2x4xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<{{\[\[}}1, 1, 0, 0], [1, 1, 1, 0]]> : tensor<2x4xi32>}> : () -> tensor<2x4xi32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_1]], %[[VAL_3]], %[[VAL_4]] : (tensor<2x4xi32>, tensor<2x4xi32>, tensor<1xi8>) -> tensor<2x4xi32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<2x4xi32> -> !torch.vtensor<[2,4],si32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[2,4],si32>
// CHECK:         }
func.func @torch.aten.tril$basic(%arg0: !torch.vtensor<[2,4], si32>) -> !torch.vtensor<[2,4], si32> {
  %int0 = torch.constant.int 1
  %0 = torch.aten.tril %arg0, %int0 : !torch.vtensor<[2,4],si32>, !torch.int -> !torch.vtensor<[2,4],si32>
  return %0 : !torch.vtensor<[2,4],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.min.dim$basic(
// CHECK-SAME:                                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,2,3],f32> -> tensor<3x2x3xf32>
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_6:.*]] = tosa.reduce_min %[[VAL_2]] {axis = 2 : i32} : (tensor<3x2x3xf32>) -> tensor<3x2x1xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x2x1xf32> -> !torch.vtensor<[3,2,1],f32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.negate %[[VAL_2]], %[[VAL_8]], %[[VAL_9]] : (tensor<3x2x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x2x3xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.argmax %[[VAL_10]] {axis = 2 : i32} : (tensor<3x2x3xf32>) -> tensor<3x2xi64>
// CHECK:           %[[VAL_12:.*]] = tosa.const_shape  {values = dense<[3, 2, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_11]], %[[VAL_12]] : (tensor<3x2xi64>, !tosa.shape<3>) -> tensor<3x2x1xi64>
// CHECK:           %[[VAL_14:.*]] = torch_c.to_builtin_tensor %[[VAL_7]] : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
// CHECK:           return %[[VAL_14]] : tensor<3x2x1xf32>
// CHECK:         }
func.func @torch.aten.min.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.min.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.min$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3,2,3],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,2,3],f32> -> tensor<3x2x3xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_min %[[VAL_1]] {axis = 0 : i32} : (tensor<3x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reduce_min %[[VAL_2]] {axis = 1 : i32} : (tensor<1x2x3xf32>) -> tensor<1x1x3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_min %[[VAL_3]] {axis = 2 : i32} : (tensor<1x1x3xf32>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_5]] : (tensor<1x1x1xf32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1],f32>
// CHECK:         }
func.func @torch.aten.min$basic(%arg0: !torch.vtensor<[3,2,3],f32>) -> !torch.vtensor<[1],f32> {
  %0 = torch.aten.min %arg0: !torch.vtensor<[3,2,3],f32> -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3,2,3],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,2,3],f32> -> tensor<3x2x3xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_max %[[VAL_1]] {axis = 0 : i32} : (tensor<3x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reduce_max %[[VAL_2]] {axis = 1 : i32} : (tensor<1x2x3xf32>) -> tensor<1x1x3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_max %[[VAL_3]] {axis = 2 : i32} : (tensor<1x1x3xf32>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_5]] : (tensor<1x1x1xf32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1],f32>
// CHECK:         }
func.func @torch.aten.max$basic(%arg0: !torch.vtensor<[3,2,3],f32>) -> !torch.vtensor<[1],f32> {
  %0 = torch.aten.max %arg0: !torch.vtensor<[3,2,3],f32> -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.prod.dim_int$basic(
// CHECK-SAME:                                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[3,2,3],f32>) -> !torch.vtensor<[3,2,1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,2,3],f32> -> tensor<3x2x3xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.constant.none
// CHECK:           %[[VAL_5:.*]] = tosa.reduce_product %[[VAL_1]] {axis = 2 : i32} : (tensor<3x2x3xf32>) -> tensor<3x2x1xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<3x2x1xf32> -> !torch.vtensor<[3,2,1],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[3,2,1],f32>
// CHECK:         }
func.func @torch.aten.prod.dim_int$basic(%arg0: !torch.vtensor<[3,2,3],f32>) -> !torch.vtensor<[3,2,1],f32> {
  %dim = torch.constant.int 2
  %keepdims = torch.constant.bool true
  %dtype = torch.constant.none
  %0 = torch.aten.prod.dim_int %arg0, %dim, %keepdims, %dtype: !torch.vtensor<[3,2,3],f32> , !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  return %0 : !torch.vtensor<[3,2,1],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.all.dim$basic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[3,2,3],i1>) -> !torch.vtensor<[3,2,1],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,2,3],i1> -> tensor<3x2x3xi1>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_all %[[VAL_1]] {axis = 2 : i32} : (tensor<3x2x3xi1>) -> tensor<3x2x1xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<3x2x1xi1> -> !torch.vtensor<[3,2,1],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[3,2,1],i1>
// CHECK:         }
func.func @torch.aten.all.dim$basic(%arg0: !torch.vtensor<[3,2,3],i1>) -> !torch.vtensor<[3,2,1],i1> {
  %dim = torch.constant.int 2
  %keepdims = torch.constant.bool true
  %0 = torch.aten.all.dim %arg0, %dim, %keepdims: !torch.vtensor<[3,2,3],i1> , !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],i1>
  return %0 : !torch.vtensor<[3,2,1],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$float_trunc(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.str "trunc"
// CHECK:           %[[VAL_5:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_7:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]], %[[VAL_6]] : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_11]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_14:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_13]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_10]], %[[VAL_15]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.greater_equal %[[VAL_7]], %[[VAL_14]] : (tensor<?x?xf32>, tensor<1x1xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_18:.*]] = tosa.select %[[VAL_17]], %[[VAL_12]], %[[VAL_16]] : (tensor<?x?xi1>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.abs %[[VAL_7]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.floor %[[VAL_19]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_22:.*]] = tosa.mul %[[VAL_20]], %[[VAL_18]], %[[VAL_21]] : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_23:.*]] = torch_c.from_builtin_tensor %[[VAL_22]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_23]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$float_trunc(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %str = torch.constant.str "trunc"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32>, !torch.str -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$int_trunc(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],si64>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %[[VAL_4:.*]] = torch.constant.str "trunc"
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_3]] : (tensor<?x?xi64>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_2]] : (tensor<?x?xi64>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.intdiv %[[VAL_5]], %[[VAL_6]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.cast %[[VAL_7]] : (tensor<?x?xi32>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[?,?],si64>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$int_trunc(%arg0: !torch.vtensor<[?, ?],si64>, %arg1: !torch.vtensor<[?, ?],si64>) -> !torch.vtensor<[?, ?],si64> {
  %str = torch.constant.str "trunc"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?, ?],si64>, !torch.vtensor<[?, ?],si64>, !torch.str -> !torch.vtensor<[?, ?],si64>
  return %0 : !torch.vtensor<[?, ?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$float_floor(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.str "floor"
// CHECK:           %[[VAL_5:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_7:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]], %[[VAL_6]] : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.floor %[[VAL_7]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$float_floor(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %str = torch.constant.str "floor"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32>, !torch.str -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$int_floor(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],si64>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %[[VAL_4:.*]] = torch.constant.str "floor"
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_3]] : (tensor<?x?xi64>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_2]] : (tensor<?x?xi64>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.intdiv %[[VAL_5]], %[[VAL_6]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_10]] : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_12]] : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_15:.*]] = tosa.mul %[[VAL_5]], %[[VAL_6]], %[[VAL_14]] : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<1xi8>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_16:.*]] = tosa.greater %[[VAL_13]], %[[VAL_15]] : (tensor<1x1xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_17:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_18:.*]] = tosa.mul %[[VAL_7]], %[[VAL_6]], %[[VAL_17]] : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<1xi8>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_19:.*]] = tosa.equal %[[VAL_18]], %[[VAL_5]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_20:.*]] = tosa.logical_not %[[VAL_19]] : (tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_21:.*]] = tosa.sub %[[VAL_7]], %[[VAL_11]] : (tensor<?x?xi32>, tensor<1x1xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_22:.*]] = tosa.logical_and %[[VAL_16]], %[[VAL_20]] : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_23:.*]] = tosa.select %[[VAL_22]], %[[VAL_21]], %[[VAL_7]] : (tensor<?x?xi1>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.cast %[[VAL_23]] : (tensor<?x?xi32>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:           return %[[VAL_25]] : !torch.vtensor<[?,?],si64>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$int_floor(%arg0: !torch.vtensor<[?, ?],si64>, %arg1: !torch.vtensor<[?, ?],si64>) -> !torch.vtensor<[?, ?],si64> {
  %str = torch.constant.str "floor"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?, ?],si64>, !torch.vtensor<[?, ?],si64>, !torch.str -> !torch.vtensor<[?, ?],si64>
  return %0 : !torch.vtensor<[?, ?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$float_basic(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.str ""
// CHECK:           %[[VAL_5:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_7:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]], %[[VAL_6]] : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<1xi8>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$float_basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %str = torch.constant.str ""
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32>, !torch.str -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$int_basic(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],si64>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:           %[[VAL_4:.*]] = torch.constant.str ""
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_3]] : (tensor<?x?xi64>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_2]] : (tensor<?x?xi64>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.intdiv %[[VAL_5]], %[[VAL_6]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.cast %[[VAL_7]] : (tensor<?x?xi32>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[?,?],si64>
// CHECK:         }
func.func @torch.aten.div.Tensor_mode$int_basic(%arg0: !torch.vtensor<[?, ?],si64>, %arg1: !torch.vtensor<[?, ?],si64>) -> !torch.vtensor<[?, ?],si64> {
  %str = torch.constant.str ""
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?, ?],si64>, !torch.vtensor<[?, ?],si64>, !torch.str -> !torch.vtensor<[?, ?],si64>
  return %0 : !torch.vtensor<[?, ?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ge.Tensor$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.greater_equal %[[VAL_3]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.ge.Tensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.ge.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.remainder.Tensor(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[2,4],f32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.floor %[[VAL_6]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_9:.*]] = tosa.mul %[[VAL_2]], %[[VAL_7]], %[[VAL_8]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.sub %[[VAL_3]], %[[VAL_9]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<2x4xf32> -> !torch.vtensor<[2,4],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[2,4],f32>
// CHECK:         }
func.func @torch.aten.remainder.Tensor(%arg0: !torch.vtensor<[2, 4],f32>, %arg1: !torch.vtensor<[2, 4],f32>) -> !torch.vtensor<[2, 4],f32> {
  %0 = torch.aten.remainder.Tensor %arg0, %arg1 : !torch.vtensor<[2, 4],f32>, !torch.vtensor<[2, 4],f32> -> !torch.vtensor<[2, 4],f32>
  return %0 : !torch.vtensor<[2, 4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fmod.Tensor(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[2,4],f32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,4],f32> -> tensor<2x4xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_10]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_13:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_12]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_14]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.greater_equal %[[VAL_6]], %[[VAL_13]] : (tensor<2x4xf32>, tensor<1x1xf32>) -> tensor<2x4xi1>
// CHECK:           %[[VAL_17:.*]] = tosa.select %[[VAL_16]], %[[VAL_11]], %[[VAL_15]] : (tensor<2x4xi1>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.abs %[[VAL_6]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.floor %[[VAL_18]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_20:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_21:.*]] = tosa.mul %[[VAL_19]], %[[VAL_17]], %[[VAL_20]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_23:.*]] = tosa.mul %[[VAL_2]], %[[VAL_21]], %[[VAL_22]] : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_24:.*]] = tosa.sub %[[VAL_3]], %[[VAL_23]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<2x4xf32> -> !torch.vtensor<[2,4],f32>
// CHECK:           return %[[VAL_25]] : !torch.vtensor<[2,4],f32>
// CHECK:         }
func.func @torch.aten.fmod.Tensor(%arg0: !torch.vtensor<[2, 4],f32>, %arg1: !torch.vtensor<[2, 4],f32>) -> !torch.vtensor<[2, 4],f32> {
  %0 = torch.aten.fmod.Tensor %arg0, %arg1 : !torch.vtensor<[2, 4],f32>, !torch.vtensor<[2, 4],f32> -> !torch.vtensor<[2, 4],f32>
  return %0 : !torch.vtensor<[2, 4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.logical_not(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[4,5],i1>) -> !torch.vtensor<[4,5],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5],i1> -> tensor<4x5xi1>
// CHECK:           %[[VAL_2:.*]] = tosa.logical_not %[[VAL_1]] : (tensor<4x5xi1>) -> tensor<4x5xi1>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<4x5xi1> -> !torch.vtensor<[4,5],i1>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[4,5],i1>
// CHECK:         }
func.func @torch.aten.logical_not(%arg0: !torch.vtensor<[4,5],i1>) -> !torch.vtensor<[4,5],i1> {
  %0 = torch.aten.logical_not %arg0 : !torch.vtensor<[4,5],i1> -> !torch.vtensor<[4,5],i1>
  return %0 : !torch.vtensor<[4,5],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cos(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.cos %[[VAL_1]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.cos(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.cos %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sin(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.sin %[[VAL_1]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.sin(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.sin %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.pow.Scalar(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.pow %[[VAL_5]], %[[VAL_1]] : (tensor<1x1xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.pow.Scalar(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.aten.pow.Scalar %float2.000000e00, %arg0 : !torch.float, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.pow.Tensor_Tensor$basic(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.pow %[[VAL_3]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.pow.Tensor_Tensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.erf$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.erf %[[VAL_1]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.erf$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.erf %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_and.Scalar$basic(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<2> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<i64>, !tosa.shape<2>) -> tensor<1x1xi64>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<1x1xi64>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.bitwise_and %[[VAL_1]], %[[VAL_6]] : (tensor<?x?xi32>, tensor<1x1xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],si32>
// CHECK:         }
func.func @torch.aten.bitwise_and.Scalar$basic(%arg0: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.bitwise_and.Scalar %arg0, %int2 : !torch.vtensor<[?,?],si32>, !torch.int -> !torch.vtensor<[?,?],si32>
  return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.le.Tensor$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.greater_equal %[[VAL_2]], %[[VAL_3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.le.Tensor$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.le.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.le.Scalar$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<2> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<i64>, !tosa.shape<2>) -> tensor<1x1xi64>
// CHECK:           %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<1x1xi64>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.greater_equal %[[VAL_6]], %[[VAL_1]] : (tensor<1x1xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.le.Scalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.le.Scalar %arg0, %int2 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.logical_xor$basic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[?,?],i1>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK:           %[[VAL_4:.*]] = tosa.logical_xor %[[VAL_3]], %[[VAL_2]] : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],i1>
// CHECK:         }
func.func @torch.aten.logical_xor$basic(%arg0: !torch.vtensor<[?,?],i1>, %arg1: !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[?,?],i1>, !torch.vtensor<[?,?],i1> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_left_shift.Tensor$basic(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],si32>,
// CHECK-SAME:                                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.logical_left_shift %[[VAL_3]], %[[VAL_2]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],si32>
// CHECK:         }
func.func @torch.aten.bitwise_left_shift.Tensor$basic(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
  %0 = torch.aten.bitwise_left_shift.Tensor %arg0, %arg1: !torch.vtensor<[?,?],si32>, !torch.vtensor<[?,?],si32> -> !torch.vtensor<[?,?],si32>
  return %0: !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bitwise_right_shift.Tensor$basic(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !torch.vtensor<[?,?],si32>,
// CHECK-SAME:                                                           %[[VAL_1:.*]]: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.arithmetic_right_shift %[[VAL_3]], %[[VAL_2]] {round = false} : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],si32>
// CHECK:         }
func.func @torch.aten.bitwise_right_shift.Tensor$basic(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
  %0 = torch.aten.bitwise_right_shift.Tensor %arg0, %arg1: !torch.vtensor<[?,?],si32>, !torch.vtensor<[?,?],si32> -> !torch.vtensor<[?,?],si32>
  return %0: !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.diagonal$basic(
// CHECK-SAME:                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[3,4,5,6],si32>) -> !torch.vtensor<[5,6,2],si32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5,6],si32> -> tensor<3x4x5x6xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.int -2
// CHECK:           %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 2, 3, 1, 0>} : (tensor<3x4x5x6xi32>) -> tensor<5x6x4x3xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]]]> : tensor<1x1x4x3xi32>}> : () -> tensor<1x1x4x3xi32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_8:.*]] = tosa.mul %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : (tensor<5x6x4x3xi32>, tensor<1x1x4x3xi32>, tensor<1xi8>) -> tensor<5x6x4x3xi32>
// CHECK-DAG:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[0, 0, 2, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[5, 6, 2, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_11:.*]] = tosa.slice %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : (tensor<5x6x4x3xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x6x2x3xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.reduce_sum %[[VAL_11]] {axis = 3 : i32} : (tensor<5x6x2x3xi32>) -> tensor<5x6x2x1xi32>
// CHECK:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[5, 6, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_14:.*]] = tosa.reshape %[[VAL_12]], %[[VAL_13]] : (tensor<5x6x2x1xi32>, !tosa.shape<3>) -> tensor<5x6x2xi32>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_14]] : tensor<5x6x2xi32> -> !torch.vtensor<[5,6,2],si32>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[5,6,2],si32>
// CHECK:         }
func.func @torch.aten.diagonal$basic(%arg0: !torch.vtensor<[3,4,5,6], si32>) -> !torch.vtensor<[5,6,2], si32> {
  %dim1 = torch.constant.int 1
  %dim2 = torch.constant.int 0
  %offset = torch.constant.int -2
  %0 = torch.aten.diagonal %arg0, %offset, %dim1, %dim2 : !torch.vtensor<[3,4,5,6],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[5,6,2],si32>
  return %0 : !torch.vtensor<[5,6,2],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.index_select(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[4,5,6],f32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !torch.vtensor<[2],si64>) -> !torch.vtensor<[4,5,2],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2],si64> -> tensor<2xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5,6],f32> -> tensor<4x5x6xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_2]] : (tensor<2xi64>) -> tensor<2xi32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 1, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<2xi32>, !tosa.shape<3>) -> tensor<1x1x2xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[4, 5, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_9:.*]] = tosa.tile %[[VAL_7]], %[[VAL_8]] : (tensor<1x1x2xi32>, !tosa.shape<3>) -> tensor<4x5x2xi32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[4, 5, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_10]] : (tensor<4x5x2xi32>, !tosa.shape<4>) -> tensor<4x5x2x1xi32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}0], [0]], {{\[\[}}0], [0]], {{\[\[}}0], [0]], {{\[\[}}0], [0]], {{\[\[}}0], [0]]], {{\[\[}}[1], [1]], {{\[\[}}1], [1]], {{\[\[}}1], [1]], {{\[\[}}1], [1]], {{\[\[}}1], [1]]], {{\[\[}}[2], [2]], {{\[\[}}2], [2]], {{\[\[}}2], [2]], {{\[\[}}2], [2]], {{\[\[}}2], [2]]], {{\[\[}}[3], [3]], {{\[\[}}3], [3]], {{\[\[}}3], [3]], {{\[\[}}3], [3]], {{\[\[}}3], [3]]]]> : tensor<4x5x2x1xi32>}> : () -> tensor<4x5x2x1xi32>
// CHECK:           %[[VAL_13:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}0], [0]], {{\[\[}}1], [1]], {{\[\[}}2], [2]], {{\[\[}}3], [3]], {{\[\[}}4], [4]]], {{\[\[}}[0], [0]], {{\[\[}}1], [1]], {{\[\[}}2], [2]], {{\[\[}}3], [3]], {{\[\[}}4], [4]]], {{\[\[}}[0], [0]], {{\[\[}}1], [1]], {{\[\[}}2], [2]], {{\[\[}}3], [3]], {{\[\[}}4], [4]]], {{\[\[}}[0], [0]], {{\[\[}}1], [1]], {{\[\[}}2], [2]], {{\[\[}}3], [3]], {{\[\[}}4], [4]]]]> : tensor<4x5x2x1xi32>}> : () -> tensor<4x5x2x1xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.concat %[[VAL_12]], %[[VAL_13]], %[[VAL_11]] {axis = 3 : i32} : (tensor<4x5x2x1xi32>, tensor<4x5x2x1xi32>, tensor<4x5x2x1xi32>) -> tensor<4x5x2x3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 120, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_15]] : (tensor<4x5x6xf32>, !tosa.shape<3>) -> tensor<1x120x1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[40, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_14]], %[[VAL_17]] : (tensor<4x5x2x3xi32>, !tosa.shape<2>) -> tensor<40x3xi32>
// CHECK:           %[[VAL_19:.*]] = "tosa.const"() <{values = dense<[30, 6, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_20:.*]] = tosa.const_shape  {values = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_21:.*]] = tosa.reshape %[[VAL_19]], %[[VAL_20]] : (tensor<3xi32>, !tosa.shape<2>) -> tensor<1x3xi32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_23:.*]] = tosa.mul %[[VAL_18]], %[[VAL_21]], %[[VAL_22]] : (tensor<40x3xi32>, tensor<1x3xi32>, tensor<1xi8>) -> tensor<40x3xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.reduce_sum %[[VAL_23]] {axis = 1 : i32} : (tensor<40x3xi32>) -> tensor<40x1xi32>
// CHECK:           %[[VAL_25:.*]] = tosa.const_shape  {values = dense<[1, 40]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_26:.*]] = tosa.reshape %[[VAL_24]], %[[VAL_25]] : (tensor<40x1xi32>, !tosa.shape<2>) -> tensor<1x40xi32>
// CHECK:           %[[VAL_27:.*]] = tosa.gather %[[VAL_16]], %[[VAL_26]] : (tensor<1x120x1xf32>, tensor<1x40xi32>) -> tensor<1x40x1xf32>
// CHECK:           %[[VAL_28:.*]] = tosa.const_shape  {values = dense<[4, 5, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_29:.*]] = tosa.reshape %[[VAL_27]], %[[VAL_28]] : (tensor<1x40x1xf32>, !tosa.shape<3>) -> tensor<4x5x2xf32>
// CHECK:           %[[VAL_30:.*]] = torch_c.from_builtin_tensor %[[VAL_29]] : tensor<4x5x2xf32> -> !torch.vtensor<[4,5,2],f32>
// CHECK:           return %[[VAL_30]] : !torch.vtensor<[4,5,2],f32>
// CHECK:         }
func.func @torch.aten.index_select(%arg0: !torch.vtensor<[4,5,6],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[4,5,2],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.index_select %arg0, %int2, %arg1 : !torch.vtensor<[4,5,6],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[4,5,2],f32>
  return %0 : !torch.vtensor<[4,5,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fill.Scalar(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[1,12,128,128],f32>) -> !torch.vtensor<[1,12,128,128],f32> {
// CHECK:           %[[VAL_1:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x12x128x128xf32>}> : () -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<1x12x128x128xf32> -> !torch.vtensor<[1,12,128,128],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[1,12,128,128],f32>
// CHECK:         }
func.func @torch.aten.fill.Scalar(%arg0: !torch.vtensor<[1,12,128,128],f32>) -> !torch.vtensor<[1,12,128,128],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.fill.Scalar %arg0, %int0 : !torch.vtensor<[1,12,128,128],f32>, !torch.int -> !torch.vtensor<[1,12,128,128],f32>
  return %0 : !torch.vtensor<[1,12,128,128],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fill.Tensor(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[1,12,128,128],f32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: !torch.vtensor<[1],si32>) -> !torch.vtensor<[1,12,128,128],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1],si32> -> tensor<1xi32>
// CHECK:           %[[VAL_3:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_3]] : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[1, 12, 128, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_6:.*]] = tosa.tile %[[VAL_4]], %[[VAL_5]] : (tensor<1x1x1x1xi32>, !tosa.shape<4>) -> tensor<1x12x128x128xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_6]] : (tensor<1x12x128x128xi32>) -> tensor<1x12x128x128xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<1x12x128x128xf32> -> !torch.vtensor<[1,12,128,128],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[1,12,128,128],f32>
// CHECK:         }
func.func @torch.aten.fill.Tensor(%arg0: !torch.vtensor<[1,12,128,128],f32>, %arg1: !torch.vtensor<[1],si32>) -> !torch.vtensor<[1,12,128,128],f32> {
  %0 = torch.aten.fill.Tensor %arg0, %arg1 : !torch.vtensor<[1,12,128,128],f32>, !torch.vtensor<[1],si32> -> !torch.vtensor<[1,12,128,128],f32>
  return %0 : !torch.vtensor<[1,12,128,128],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.flip(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5],f32> -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = tosa.reverse %[[VAL_1]] {axis = 1 : i32} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.reverse %[[VAL_5]] {axis = 2 : i32} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x4x5xf32> -> !torch.vtensor<[3,4,5],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,4,5],f32>
// CHECK:         }
func.func @torch.aten.flip(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.flip %arg0, %0 : !torch.vtensor<[3,4,5],f32>, !torch.list<int> -> !torch.vtensor<[3,4,5],f32>
  return %1 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.round(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5],f32> -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.floor %[[VAL_1]] : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.sub %[[VAL_1]], %[[VAL_8]] : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.ceil %[[VAL_1]] : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_12:.*]] = tosa.mul %[[VAL_8]], %[[VAL_5]], %[[VAL_11]] : (tensor<3x4x5xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.floor %[[VAL_12]] : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_15:.*]] = tosa.mul %[[VAL_13]], %[[VAL_7]], %[[VAL_14]] : (tensor<3x4x5xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.equal %[[VAL_8]], %[[VAL_15]] : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
// CHECK:           %[[VAL_17:.*]] = tosa.equal %[[VAL_9]], %[[VAL_5]] : (tensor<3x4x5xf32>, tensor<1x1x1xf32>) -> tensor<3x4x5xi1>
// CHECK:           %[[VAL_18:.*]] = tosa.greater %[[VAL_5]], %[[VAL_9]] : (tensor<1x1x1xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
// CHECK:           %[[VAL_19:.*]] = tosa.logical_and %[[VAL_17]], %[[VAL_16]] : (tensor<3x4x5xi1>, tensor<3x4x5xi1>) -> tensor<3x4x5xi1>
// CHECK:           %[[VAL_20:.*]] = tosa.logical_or %[[VAL_18]], %[[VAL_19]] : (tensor<3x4x5xi1>, tensor<3x4x5xi1>) -> tensor<3x4x5xi1>
// CHECK:           %[[VAL_21:.*]] = tosa.select %[[VAL_20]], %[[VAL_8]], %[[VAL_10]] : (tensor<3x4x5xi1>, tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_21]] : tensor<3x4x5xf32> -> !torch.vtensor<[3,4,5],f32>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[3,4,5],f32>
// CHECK:         }
func.func @torch.aten.round(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
  %0 = torch.aten.round %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

func.func @torch.aten.avg_pool2d.count_include_pad_unsupported_value(%arg0: !torch.vtensor<[1,192,35,35],f32>) -> !torch.vtensor<[1,192,35,35],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %false= torch.constant.bool false
  %count_include_pad = torch.constant.bool true
  %divisor_override = torch.constant.none

  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten.avg_pool2d' that was explicitly marked illegal}}
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %count_include_pad, %divisor_override : !torch.vtensor<[1,192,35,35],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,192,35,35],f32>
  return %3 : !torch.vtensor<[1,192,35,35],f32>
}

// -----

func.func @torch.aten.avg_pool2d.divisor_override_unsupported_value(%arg0: !torch.vtensor<[1,192,35,35],f32>) -> !torch.vtensor<[1,192,35,35],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %false= torch.constant.bool false
  %count_include_pad = torch.constant.bool false
  %divisor_override = torch.constant.int 9

  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten.avg_pool2d' that was explicitly marked illegal}}
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %count_include_pad, %divisor_override : !torch.vtensor<[1,192,35,35],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.int -> !torch.vtensor<[1,192,35,35],f32>
  return %3 : !torch.vtensor<[1,192,35,35],f32>
}

// -----
// CHECK-LABEL:   func.func @avgPool2dCHWInput(
// CHECK-SAME:                                 %[[ARG0:.*]]: !torch.vtensor<[1,64,56],f32>) -> !torch.vtensor<[1,59,51],f32> {
// CHECK:           %[[TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,64,56],f32> -> tensor<1x64x56xf32>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[C0:.*]] = torch.constant.int 0
// CHECK:           %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[C6:.*]] = torch.constant.int 6
// CHECK:           %[[L1:.*]] = torch.prim.ListConstruct %[[C6]], %[[C6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[L2:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[L3:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[TRANSPOSE_IN:.*]] = tosa.transpose %[[TENSOR]] {perms = array<i32: 1, 2, 0>} : (tensor<1x64x56xf32>) -> tensor<64x56x1xf32>
// CHECK:           %[[CONST_SHAPE_IN:.*]] = tosa.const_shape  {values = dense<[1, 64, 56, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[RESHAPE_IN:.*]] = tosa.reshape %[[TRANSPOSE_IN]], %[[CONST_SHAPE_IN]] : (tensor<64x56x1xf32>, !tosa.shape<4>) -> tensor<1x64x56x1xf32>
// CHECK:           %[[INPUT_ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[OUTPUT_ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[POOL:.*]] = tosa.avg_pool2d %[[RESHAPE_IN]], %[[INPUT_ZP]], %[[OUTPUT_ZP]] {acc_type = f32, kernel = array<i64: 6, 6>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x64x56x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x59x51x1xf32>
// CHECK:           %[[TRANSPOSE_OUT:.*]] = tosa.transpose %[[POOL]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x59x51x1xf32>) -> tensor<1x1x59x51xf32>
// CHECK:           %[[CONST_SHAPE_OUT:.*]] = tosa.const_shape  {values = dense<[1, 59, 51]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[RESHAPE_OUT:.*]] = tosa.reshape %[[TRANSPOSE_OUT]], %[[CONST_SHAPE_OUT]] : (tensor<1x1x59x51xf32>, !tosa.shape<3>) -> tensor<1x59x51xf32>
// CHECK:           %[[CAST:.*]] = tensor.cast %[[RESHAPE_OUT]] : tensor<1x59x51xf32> to tensor<1x59x51xf32>
// CHECK:           %[[TORCH:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<1x59x51xf32> -> !torch.vtensor<[1,59,51],f32>
// CHECK:           return %[[TORCH]]
func.func @avgPool2dCHWInput(%arg0: !torch.vtensor<[1,64,56],f32>) -> !torch.vtensor<[1,59,51],f32> {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int6 = torch.constant.int 6
    %0 = torch.prim.ListConstruct %int6, %int6 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %true, %false, %none : !torch.vtensor<[1,64,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,59,51],f32>
    return %3 : !torch.vtensor<[1,59,51],f32>
  }

// -----

// CHECK-LABEL:   func.func @torch.aten.empty.memory_format$basic() -> !torch.vtensor<[3,4],si64> {
// CHECK:           %[[VAL_0:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_1:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.constant.device "cpu"
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.cast %[[VAL_7]] : (tensor<3x4xi32>) -> tensor<3x4xi64>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0> : tensor<3x4xi64>}> : () -> tensor<3x4xi64>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<3x4xi64> -> !torch.vtensor<[3,4],si64>
// CHECK:           return %[[VAL_10]] : !torch.vtensor<[3,4],si64>
// CHECK:         }
func.func @torch.aten.empty.memory_format$basic() -> !torch.vtensor<[3,4],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %none = torch.constant.none
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.empty.memory_format %0, %int4, %none, %cpu, %false, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[3,4],si64>
  %2 = torch.aten.fill.Scalar %1, %int0 : !torch.vtensor<[3,4],si64>, !torch.int -> !torch.vtensor<[3,4],si64>
  return %2 : !torch.vtensor<[3,4],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.scatter.src$basic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[10,8,6],f32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: !torch.vtensor<[2,4,3],si64>,
// CHECK-SAME:                                            %[[VAL_2:.*]]: !torch.vtensor<[3,4,3],f32>) -> !torch.vtensor<[10,8,6],f32> {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[3,4,3],f32> -> tensor<3x4x3xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,4,3],si64> -> tensor<2x4x3xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[10,8,6],f32> -> tensor<10x8x6xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_4]] : (tensor<2x4x3xi64>) -> tensor<2x4x3xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[2, 4, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_8]] : (tensor<2x4x3xi32>, !tosa.shape<4>) -> tensor<2x4x3x1xi32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}0], [0], [0]], {{\[\[}}0], [0], [0]], {{\[\[}}0], [0], [0]], {{\[\[}}0], [0], [0]]], {{\[\[}}[1], [1], [1]], {{\[\[}}1], [1], [1]], {{\[\[}}1], [1], [1]], {{\[\[}}1], [1], [1]]]]> : tensor<2x4x3x1xi32>}> : () -> tensor<2x4x3x1xi32>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}0], [1], [2]], {{\[\[}}0], [1], [2]], {{\[\[}}0], [1], [2]], {{\[\[}}0], [1], [2]]], {{\[\[}}[0], [1], [2]], {{\[\[}}0], [1], [2]], {{\[\[}}0], [1], [2]], {{\[\[}}0], [1], [2]]]]> : tensor<2x4x3x1xi32>}> : () -> tensor<2x4x3x1xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.concat %[[VAL_10]], %[[VAL_9]], %[[VAL_11]] {axis = 3 : i32} : (tensor<2x4x3x1xi32>, tensor<2x4x3x1xi32>, tensor<2x4x3x1xi32>) -> tensor<2x4x3x3xi32>
// CHECK:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[1, 36, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_14:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_13]] : (tensor<3x4x3xf32>, !tosa.shape<3>) -> tensor<1x36x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 480, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_15]] : (tensor<10x8x6xf32>, !tosa.shape<3>) -> tensor<1x480x1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[24, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_12]], %[[VAL_17]] : (tensor<2x4x3x3xi32>, !tosa.shape<2>) -> tensor<24x3xi32>
// CHECK:           %[[VAL_19:.*]] = "tosa.const"() <{values = dense<[48, 6, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_20:.*]] = tosa.const_shape  {values = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_21:.*]] = tosa.reshape %[[VAL_19]], %[[VAL_20]] : (tensor<3xi32>, !tosa.shape<2>) -> tensor<1x3xi32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_23:.*]] = tosa.mul %[[VAL_18]], %[[VAL_21]], %[[VAL_22]] : (tensor<24x3xi32>, tensor<1x3xi32>, tensor<1xi8>) -> tensor<24x3xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.reduce_sum %[[VAL_23]] {axis = 1 : i32} : (tensor<24x3xi32>) -> tensor<24x1xi32>
// CHECK:           %[[VAL_25:.*]] = tosa.const_shape  {values = dense<[1, 24]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_26:.*]] = tosa.reshape %[[VAL_24]], %[[VAL_25]] : (tensor<24x1xi32>, !tosa.shape<2>) -> tensor<1x24xi32>
// CHECK:           %[[VAL_27:.*]] = tosa.scatter %[[VAL_16]], %[[VAL_26]], %[[VAL_14]] : (tensor<1x480x1xf32>, tensor<1x24xi32>, tensor<1x36x1xf32>) -> tensor<1x480x1xf32>
// CHECK:           %[[VAL_28:.*]] = tosa.const_shape  {values = dense<[10, 8, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_29:.*]] = tosa.reshape %[[VAL_27]], %[[VAL_28]] : (tensor<1x480x1xf32>, !tosa.shape<3>) -> tensor<10x8x6xf32>
// CHECK:           %[[VAL_30:.*]] = torch_c.from_builtin_tensor %[[VAL_29]] : tensor<10x8x6xf32> -> !torch.vtensor<[10,8,6],f32>
// CHECK:           return %[[VAL_30]] : !torch.vtensor<[10,8,6],f32>
// CHECK:         }
func.func @torch.aten.scatter.src$basic(%arg0: !torch.vtensor<[10,8,6],f32>, %arg1: !torch.vtensor<[2,4,3],si64>, %arg2: !torch.vtensor<[3,4,3],f32>) -> !torch.vtensor<[10,8,6],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.scatter.src %arg0, %int1, %arg1, %arg2 : !torch.vtensor<[10,8,6],f32>, !torch.int, !torch.vtensor<[2,4,3],si64>, !torch.vtensor<[3,4,3],f32> -> !torch.vtensor<[10,8,6],f32>
  return %0 : !torch.vtensor<[10,8,6],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.slice_scatter$basic(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[6,8],f32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: !torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,8],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[6,1],f32> -> tensor<6x1xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[6,8],f32> -> tensor<6x8xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<6x1xi32>}> : () -> tensor<6x1xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[6, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_7]] : (tensor<6x1xi32>, !tosa.shape<3>) -> tensor<6x1x1xi32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[0]], {{\[\[}}1]], {{\[\[}}2]], {{\[\[}}3]], {{\[\[}}4]], {{\[\[}}5]]]> : tensor<6x1x1xi32>}> : () -> tensor<6x1x1xi32>
// CHECK:           %[[VAL_10:.*]] = tosa.concat %[[VAL_9]], %[[VAL_8]] {axis = 2 : i32} : (tensor<6x1x1xi32>, tensor<6x1x1xi32>) -> tensor<6x1x2xi32>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<[1, 6, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_11]] : (tensor<6x1xf32>, !tosa.shape<3>) -> tensor<1x6x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[1, 48, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_14:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_13]] : (tensor<6x8xf32>, !tosa.shape<3>) -> tensor<1x48x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[6, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_10]], %[[VAL_15]] : (tensor<6x1x2xi32>, !tosa.shape<2>) -> tensor<6x2xi32>
// CHECK:           %[[VAL_17:.*]] = "tosa.const"() <{values = dense<[8, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[1, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_19:.*]] = tosa.reshape %[[VAL_17]], %[[VAL_18]] : (tensor<2xi32>, !tosa.shape<2>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_20:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_21:.*]] = tosa.mul %[[VAL_16]], %[[VAL_19]], %[[VAL_20]] : (tensor<6x2xi32>, tensor<1x2xi32>, tensor<1xi8>) -> tensor<6x2xi32>
// CHECK:           %[[VAL_22:.*]] = tosa.reduce_sum %[[VAL_21]] {axis = 1 : i32} : (tensor<6x2xi32>) -> tensor<6x1xi32>
// CHECK:           %[[VAL_23:.*]] = tosa.const_shape  {values = dense<[1, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_24:.*]] = tosa.reshape %[[VAL_22]], %[[VAL_23]] : (tensor<6x1xi32>, !tosa.shape<2>) -> tensor<1x6xi32>
// CHECK:           %[[VAL_25:.*]] = tosa.scatter %[[VAL_14]], %[[VAL_24]], %[[VAL_12]] : (tensor<1x48x1xf32>, tensor<1x6xi32>, tensor<1x6x1xf32>) -> tensor<1x48x1xf32>
// CHECK:           %[[VAL_26:.*]] = tosa.const_shape  {values = dense<[6, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_27:.*]] = tosa.reshape %[[VAL_25]], %[[VAL_26]] : (tensor<1x48x1xf32>, !tosa.shape<2>) -> tensor<6x8xf32>
// CHECK:           %[[VAL_28:.*]] = torch_c.from_builtin_tensor %[[VAL_27]] : tensor<6x8xf32> -> !torch.vtensor<[6,8],f32>
// CHECK:           return %[[VAL_28]] : !torch.vtensor<[6,8],f32>
// CHECK:         }
func.func @torch.aten.slice_scatter$basic(%arg0: !torch.vtensor<[6,8],f32>, %arg1: !torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,8],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %0 = torch.aten.slice_scatter %arg0, %arg1, %int1, %int0, %int1, %int1 : !torch.vtensor<[6,8],f32>, !torch.vtensor<[6,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[6,8],f32>
  return %0 : !torch.vtensor<[6,8],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.diag_embed$basic(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,4],f32> -> tensor<2x3x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int -2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]]], {{\[\[}}[0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]]]]> : tensor<2x3x4x1xi32>}> : () -> tensor<2x3x4x1xi32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[2, 3, 4, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_6]] : (tensor<2x3x4xf32>, !tosa.shape<4>) -> tensor<2x3x4x1xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2x3x4x4xf32>}> : () -> tensor<2x3x4x4xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[2, 3, 4, 1, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_9]] : (tensor<2x3x4x1xi32>, !tosa.shape<5>) -> tensor<2x3x4x1x1xi32>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}[0]], {{\[\[}}0]], {{\[\[}}0]], {{\[\[}}0]]], {{\[\[}}[0]], {{\[\[}}0]], {{\[\[}}0]], {{\[\[}}0]]], {{\[\[}}[0]], {{\[\[}}0]], {{\[\[}}0]], {{\[\[}}0]]]], {{\[\[}}{{\[\[}}1]], {{\[\[}}1]], {{\[\[}}1]], {{\[\[}}1]]], {{\[\[}}[1]], {{\[\[}}1]], {{\[\[}}1]], {{\[\[}}1]]], {{\[\[}}[1]], {{\[\[}}1]], {{\[\[}}1]], {{\[\[}}1]]]]]> : tensor<2x3x4x1x1xi32>}> : () -> tensor<2x3x4x1x1xi32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}[0]], {{\[\[}}0]], {{\[\[}}0]], {{\[\[}}0]]], {{\[\[}}[1]], {{\[\[}}1]], {{\[\[}}1]], {{\[\[}}1]]], {{\[\[}}[2]], {{\[\[}}2]], {{\[\[}}2]], {{\[\[}}2]]]], {{\[\[}}{{\[\[}}0]], {{\[\[}}0]], {{\[\[}}0]], {{\[\[}}0]]], {{\[\[}}[1]], {{\[\[}}1]], {{\[\[}}1]], {{\[\[}}1]]], {{\[\[}}[2]], {{\[\[}}2]], {{\[\[}}2]], {{\[\[}}2]]]]]> : tensor<2x3x4x1x1xi32>}> : () -> tensor<2x3x4x1x1xi32>
// CHECK:           %[[VAL_13:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}[0]], {{\[\[}}1]], {{\[\[}}2]], {{\[\[}}3]]], {{\[\[}}[0]], {{\[\[}}1]], {{\[\[}}2]], {{\[\[}}3]]], {{\[\[}}[0]], {{\[\[}}1]], {{\[\[}}2]], {{\[\[}}3]]]], {{\[\[}}{{\[\[}}0]], {{\[\[}}1]], {{\[\[}}2]], {{\[\[}}3]]], {{\[\[}}[0]], {{\[\[}}1]], {{\[\[}}2]], {{\[\[}}3]]], {{\[\[}}[0]], {{\[\[}}1]], {{\[\[}}2]], {{\[\[}}3]]]]]> : tensor<2x3x4x1x1xi32>}> : () -> tensor<2x3x4x1x1xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.concat %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_10]] {axis = 4 : i32} : (tensor<2x3x4x1x1xi32>, tensor<2x3x4x1x1xi32>, tensor<2x3x4x1x1xi32>, tensor<2x3x4x1x1xi32>) -> tensor<2x3x4x1x4xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 24, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_15]] : (tensor<2x3x4x1xf32>, !tosa.shape<3>) -> tensor<1x24x1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[1, 96, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_17]] : (tensor<2x3x4x4xf32>, !tosa.shape<3>) -> tensor<1x96x1xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<[24, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_20:.*]] = tosa.reshape %[[VAL_14]], %[[VAL_19]] : (tensor<2x3x4x1x4xi32>, !tosa.shape<2>) -> tensor<24x4xi32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<[48, 16, 4, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_22:.*]] = tosa.const_shape  {values = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_23:.*]] = tosa.reshape %[[VAL_21]], %[[VAL_22]] : (tensor<4xi32>, !tosa.shape<2>) -> tensor<1x4xi32>
// CHECK:           %[[VAL_24:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_25:.*]] = tosa.mul %[[VAL_20]], %[[VAL_23]], %[[VAL_24]] : (tensor<24x4xi32>, tensor<1x4xi32>, tensor<1xi8>) -> tensor<24x4xi32>
// CHECK:           %[[VAL_26:.*]] = tosa.reduce_sum %[[VAL_25]] {axis = 1 : i32} : (tensor<24x4xi32>) -> tensor<24x1xi32>
// CHECK:           %[[VAL_27:.*]] = tosa.const_shape  {values = dense<[1, 24]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_28:.*]] = tosa.reshape %[[VAL_26]], %[[VAL_27]] : (tensor<24x1xi32>, !tosa.shape<2>) -> tensor<1x24xi32>
// CHECK:           %[[VAL_29:.*]] = tosa.scatter %[[VAL_18]], %[[VAL_28]], %[[VAL_16]] : (tensor<1x96x1xf32>, tensor<1x24xi32>, tensor<1x24x1xf32>) -> tensor<1x96x1xf32>
// CHECK:           %[[VAL_30:.*]] = tosa.const_shape  {values = dense<[2, 3, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_31:.*]] = tosa.reshape %[[VAL_29]], %[[VAL_30]] : (tensor<1x96x1xf32>, !tosa.shape<4>) -> tensor<2x3x4x4xf32>
// CHECK:           %[[VAL_32:.*]] = tosa.transpose %[[VAL_31]] {perms = array<i32: 0, 1, 2, 3>} : (tensor<2x3x4x4xf32>) -> tensor<2x3x4x4xf32>
// CHECK:           %[[VAL_33:.*]] = torch_c.from_builtin_tensor %[[VAL_32]] : tensor<2x3x4x4xf32> -> !torch.vtensor<[2,3,4,4],f32>
// CHECK:           return %[[VAL_33]] : !torch.vtensor<[2,3,4,4],f32>
// CHECK:         }
func.func @torch.aten.diag_embed$basic(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4,4],f32> {
  %int0 = torch.constant.int 0
  %int-2 = torch.constant.int -2
  %int-1 = torch.constant.int -1
  %0 = torch.aten.diag_embed %arg0, %int0, %int-2, %int-1 : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3,4,4],f32>
  return %0 : !torch.vtensor<[2,3,4,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.index.Tensor_hacked_twin(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !torch.vtensor<[2,4,2],si64>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[4,2],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,4,2],si64> -> tensor<2x4x2xi64>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_1]] : (!torch.vtensor<[],si64>) -> !torch.list<vtensor>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[],si64> -> tensor<i64>
// CHECK:           %[[VAL_5:.*]] = tosa.cast %[[VAL_4]] : (tensor<i64>) -> tensor<i32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<2> : tensor<i32>}> : () -> tensor<i32>
// CHECK:           %[[VAL_8:.*]] = tosa.add %[[VAL_7]], %[[VAL_5]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:           %[[VAL_9:.*]] = tosa.greater %[[VAL_6]], %[[VAL_5]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:           %[[VAL_10:.*]] = tosa.select %[[VAL_9]], %[[VAL_8]], %[[VAL_5]] : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_10]], %[[VAL_11]] : (tensor<i32>, !tosa.shape<1>) -> tensor<1xi32>
// CHECK:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[1, 2, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_14:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_13]] : (tensor<2x4x2xi64>, !tosa.shape<3>) -> tensor<1x2x8xi64>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_12]], %[[VAL_15]] : (tensor<1xi32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_17:.*]] = "tosa.const"() <{values = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_19:.*]] = tosa.reshape %[[VAL_17]], %[[VAL_18]] : (tensor<1xi32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_20:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_21:.*]] = tosa.mul %[[VAL_16]], %[[VAL_19]], %[[VAL_20]] : (tensor<1x1xi32>, tensor<1x1xi32>, tensor<1xi8>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_22:.*]] = tosa.reduce_sum %[[VAL_21]] {axis = 1 : i32} : (tensor<1x1xi32>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_23:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_24:.*]] = tosa.reshape %[[VAL_22]], %[[VAL_23]] : (tensor<1x1xi32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_25:.*]] = tosa.gather %[[VAL_14]], %[[VAL_24]] : (tensor<1x2x8xi64>, tensor<1x1xi32>) -> tensor<1x1x8xi64>
// CHECK:           %[[VAL_26:.*]] = tosa.const_shape  {values = dense<[4, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_27:.*]] = tosa.reshape %[[VAL_25]], %[[VAL_26]] : (tensor<1x1x8xi64>, !tosa.shape<2>) -> tensor<4x2xi64>
// CHECK:           %[[VAL_28:.*]] = torch_c.from_builtin_tensor %[[VAL_27]] : tensor<4x2xi64> -> !torch.vtensor<[4,2],si64>
// CHECK:           return %[[VAL_28]] : !torch.vtensor<[4,2],si64>
// CHECK:         }
func.func @torch.aten.index.Tensor_hacked_twin(%arg0: !torch.vtensor<[2,4,2],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[4,2],si64> {
  %0 = torch.prim.ListConstruct %arg1 : (!torch.vtensor<[],si64>) -> !torch.list<vtensor>
  %1 = torch.aten.index.Tensor_hacked_twin %arg0, %0 : !torch.vtensor<[2,4,2],si64>, !torch.list<vtensor> -> !torch.vtensor<[4,2],si64>
  return %1 : !torch.vtensor<[4,2],si64>
}

// -----

func.func @torch.aten.index.Tensor_hacked_twin.dynamic_size(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[?,1],si64>, %arg2: !torch.vtensor<[1,4],si64>) -> !torch.vtensor<[?,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.prim.ListConstruct %arg1, %arg2 : (!torch.vtensor<[?,1],si64>, !torch.vtensor<[1,4],si64>) -> !torch.list<vtensor>
  // expected-error @+1 {{failed to legalize operation 'torch.aten.index.Tensor_hacked_twin' that was explicitly marked illegal}}
  %1 = torch.aten.index.Tensor_hacked_twin %arg0, %0 : !torch.vtensor<[?,4],f32>, !torch.list<vtensor> -> !torch.vtensor<[?,4],f32>
  return %1 : !torch.vtensor<[?,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.threshold_backward$basic(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !torch.vtensor<[4],si64>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4],si64> -> tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4],si64> -> tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           %[[VAL_7:.*]] = tosa.greater_equal %[[VAL_5]], %[[VAL_2]] : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_8]] : (tensor<i64>, !tosa.shape<1>) -> tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = tosa.select %[[VAL_7]], %[[VAL_9]], %[[VAL_3]] : (tensor<4xi1>, tensor<1xi64>, tensor<4xi64>) -> tensor<4xi64>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<4xi64> -> !torch.vtensor<[4],si64>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[4],si64>
// CHECK:         }
func.func @torch.aten.threshold_backward$basic(%arg0: !torch.vtensor<[4],si64>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.threshold_backward %arg0, %arg1, %int1 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.threshold$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5],si64> -> tensor<4x5xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 5.000000e-01
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<2> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           %[[VAL_6:.*]] = tosa.greater %[[VAL_1]], %[[VAL_4]] : (tensor<4x5xi64>, tensor<1x1xi64>) -> tensor<4x5xi1>
// CHECK:           %[[VAL_7:.*]] = tosa.select %[[VAL_6]], %[[VAL_1]], %[[VAL_5]] : (tensor<4x5xi1>, tensor<4x5xi64>, tensor<1x1xi64>) -> tensor<4x5xi64>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<4x5xi64> -> !torch.vtensor<[4,5],si64>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[4,5],si64>
// CHECK:         }
func.func @torch.aten.threshold$basic(%arg0: !torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64> {
  %float5.000000e-01 = torch.constant.float 5.000000e-01
  %int2 = torch.constant.int 2
  %0 = torch.aten.threshold %arg0, %float5.000000e-01, %int2 : !torch.vtensor<[4,5],si64>, !torch.float, !torch.int -> !torch.vtensor<[4,5],si64>
  return %0 : !torch.vtensor<[4,5],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.logical_and$basic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[4,5],i1>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: !torch.vtensor<[4,5],i1>) -> !torch.vtensor<[4,5],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,5],i1> -> tensor<4x5xi1>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5],i1> -> tensor<4x5xi1>
// CHECK:           %[[VAL_4:.*]] = tosa.logical_and %[[VAL_3]], %[[VAL_2]] : (tensor<4x5xi1>, tensor<4x5xi1>) -> tensor<4x5xi1>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x5xi1> -> !torch.vtensor<[4,5],i1>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,5],i1>
// CHECK:         }
func.func @torch.aten.logical_and$basic(%arg0: !torch.vtensor<[4,5],i1>, %arg1: !torch.vtensor<[4,5],i1>) -> !torch.vtensor<[4,5],i1> {
  %0 = torch.aten.logical_and %arg0, %arg1 : !torch.vtensor<[4,5],i1>, !torch.vtensor<[4,5],i1> -> !torch.vtensor<[4,5],i1>
  return %0 : !torch.vtensor<[4,5],i1>
}

// -----

// CHECK-LABEL:     torch.aten.uniform$basic
// CHECK:           tosa.const
func.func @torch.aten.uniform$basic(%arg0: !torch.vtensor<[3,4],f64>) -> (!torch.vtensor<[3,4],f64>, !torch.vtensor<[3,4],f64>) {
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %float1.000000e01 = torch.constant.float 1.000000e+01
  %none = torch.constant.none
  %0 = torch.aten.uniform %arg0, %float1.000000e00, %float1.000000e01, %none : !torch.vtensor<[3,4],f64>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[3,4],f64>
  return %0, %0 : !torch.vtensor<[3,4],f64>, !torch.vtensor<[3,4],f64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.as_strided$basic(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[3,3],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,5],f32> -> tensor<5x5xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<25> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_8]] : (tensor<5x5xf32>, !tosa.shape<1>) -> tensor<25xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<[0, 1, 2, 2, 3, 4, 4, 5, 6]> : tensor<9xi32>}> : () -> tensor<9xi32>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<[9, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_10]], %[[VAL_11]] : (tensor<9xi32>, !tosa.shape<2>) -> tensor<9x1xi32>
// CHECK:           %[[VAL_13:.*]] = tosa.concat %[[VAL_12]] {axis = 1 : i32} : (tensor<9x1xi32>) -> tensor<9x1xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[1, 25, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_14]] : (tensor<25xf32>, !tosa.shape<3>) -> tensor<1x25x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.const_shape  {values = dense<[9, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_17:.*]] = tosa.reshape %[[VAL_13]], %[[VAL_16]] : (tensor<9x1xi32>, !tosa.shape<2>) -> tensor<9x1xi32>
// CHECK:           %[[VAL_18:.*]] = "tosa.const"() <{values = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_20:.*]] = tosa.reshape %[[VAL_18]], %[[VAL_19]] : (tensor<1xi32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_22:.*]] = tosa.mul %[[VAL_17]], %[[VAL_20]], %[[VAL_21]] : (tensor<9x1xi32>, tensor<1x1xi32>, tensor<1xi8>) -> tensor<9x1xi32>
// CHECK:           %[[VAL_23:.*]] = tosa.reduce_sum %[[VAL_22]] {axis = 1 : i32} : (tensor<9x1xi32>) -> tensor<9x1xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.const_shape  {values = dense<[1, 9]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_25:.*]] = tosa.reshape %[[VAL_23]], %[[VAL_24]] : (tensor<9x1xi32>, !tosa.shape<2>) -> tensor<1x9xi32>
// CHECK:           %[[VAL_26:.*]] = tosa.gather %[[VAL_15]], %[[VAL_25]] : (tensor<1x25x1xf32>, tensor<1x9xi32>) -> tensor<1x9x1xf32>
// CHECK:           %[[VAL_27:.*]] = tosa.const_shape  {values = dense<9> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_28:.*]] = tosa.reshape %[[VAL_26]], %[[VAL_27]] : (tensor<1x9x1xf32>, !tosa.shape<1>) -> tensor<9xf32>
// CHECK:           %[[VAL_29:.*]] = tosa.const_shape  {values = dense<3> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_30:.*]] = tosa.reshape %[[VAL_28]], %[[VAL_29]] : (tensor<9xf32>, !tosa.shape<2>) -> tensor<3x3xf32>
// CHECK:           %[[VAL_31:.*]] = torch_c.from_builtin_tensor %[[VAL_30]] : tensor<3x3xf32> -> !torch.vtensor<[3,3],f32>
// CHECK:           return %[[VAL_31]] : !torch.vtensor<[3,3],f32>
// CHECK:         }
func.func @torch.aten.as_strided$basic(%arg0: !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[3,3],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.as_strided %arg0, %0, %1, %none : !torch.vtensor<[5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[3,3],f32>
  return %2 : !torch.vtensor<[3,3],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool1d$basic(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,64,112],f32>) -> !torch.vtensor<[1,64,56],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,64,112],f32> -> tensor<1x64x112xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_5]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_4]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[1, 64, 112, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_10]] : (tensor<1x64x112xf32>, !tosa.shape<4>) -> tensor<1x64x112x1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_11]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x112x1xf32>) -> tensor<1x112x1x64xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.max_pool2d %[[VAL_12]] {kernel = array<i64: 3, 1>, pad = array<i64: 1, 0, 0, 0>, stride = array<i64: 2, 1>} : (tensor<1x112x1x64xf32>) -> tensor<1x56x1x64xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.transpose %[[VAL_13]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x56x1x64xf32>) -> tensor<1x64x56x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 64, 56]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_14]], %[[VAL_15]] : (tensor<1x64x56x1xf32>, !tosa.shape<3>) -> tensor<1x64x56xf32>
// CHECK:           %[[VAL_17:.*]] = tensor.cast %[[VAL_16]] : tensor<1x64x56xf32> to tensor<1x64x56xf32>
// CHECK:           %[[VAL_18:.*]] = torch_c.from_builtin_tensor %[[VAL_17]] : tensor<1x64x56xf32> -> !torch.vtensor<[1,64,56],f32>
// CHECK:           return %[[VAL_18]] : !torch.vtensor<[1,64,56],f32>
// CHECK:         }
func.func @torch.aten.max_pool1d$basic(%arg0: !torch.vtensor<[1,64,112],f32>) -> !torch.vtensor<[1,64,56],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool1d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,64,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,64,56],f32>
  return %4 : !torch.vtensor<[1,64,56],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool1d$basic(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,512,10],f32>) -> !torch.vtensor<[1,512,10],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,512,10],f32> -> tensor<1x512x10xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_3]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[1, 512, 10, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_8]] : (tensor<1x512x10xf32>, !tosa.shape<4>) -> tensor<1x512x10x1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.transpose %[[VAL_9]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x10x1xf32>) -> tensor<1x10x1x512xf32>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.avg_pool2d %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] {acc_type = f32, kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x10x1x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x10x1x512xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.transpose %[[VAL_13]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x10x1x512xf32>) -> tensor<1x512x10x1xf32>
// CHECK-DAG:       %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 512, 10]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_14]], %[[VAL_15]] : (tensor<1x512x10x1xf32>, !tosa.shape<3>) -> tensor<1x512x10xf32>
// CHECK:           %[[VAL_17:.*]] = tensor.cast %[[VAL_16]] : tensor<1x512x10xf32> to tensor<1x512x10xf32>
// CHECK:           %[[VAL_18:.*]] = torch_c.from_builtin_tensor %[[VAL_17]] : tensor<1x512x10xf32> -> !torch.vtensor<[1,512,10],f32>
// CHECK:           return %[[VAL_18]] : !torch.vtensor<[1,512,10],f32>
// CHECK:         }
func.func @torch.aten.avg_pool1d$basic(%arg0: !torch.vtensor<[1,512,10],f32>) -> !torch.vtensor<[1,512,10],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool1d %arg0, %0, %1, %2, %false, %false : !torch.vtensor<[1,512,10],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,512,10],f32>
  return %3 : !torch.vtensor<[1,512,10],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.clamp.Tensor$basic(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[3,5],f32>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                             %[[VAL_2:.*]]: !torch.vtensor<[1],f32>) -> (!torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>) {
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[1],f32> -> tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1],f32> -> tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5],f32> -> tensor<3x5xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.none
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_8]] : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_10]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.maximum %[[VAL_5]], %[[VAL_9]] : (tensor<3x5xf32>, tensor<1x1xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.minimum %[[VAL_12]], %[[VAL_11]] : (tensor<3x5xf32>, tensor<1x1xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_14:.*]] = torch_c.from_builtin_tensor %[[VAL_13]] : tensor<3x5xf32> -> !torch.vtensor<[3,5],f32>
// CHECK:           %[[VAL_15:.*]] = "tosa.const"() <{values = dense<-3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_16:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_17:.*]] = tosa.reshape %[[VAL_15]], %[[VAL_16]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_19:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_18]] : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.maximum %[[VAL_5]], %[[VAL_17]] : (tensor<3x5xf32>, tensor<1x1xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.minimum %[[VAL_20]], %[[VAL_19]] : (tensor<3x5xf32>, tensor<1x1xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_21]] : tensor<3x5xf32> -> !torch.vtensor<[3,5],f32>
// CHECK:           %[[VAL_23:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_24:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_23]] : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_25:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_26:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_25]] : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_27:.*]] = tosa.maximum %[[VAL_5]], %[[VAL_24]] : (tensor<3x5xf32>, tensor<1x1xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_28:.*]] = tosa.minimum %[[VAL_27]], %[[VAL_26]] : (tensor<3x5xf32>, tensor<1x1xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_29:.*]] = torch_c.from_builtin_tensor %[[VAL_28]] : tensor<3x5xf32> -> !torch.vtensor<[3,5],f32>
// CHECK:           return %[[VAL_14]], %[[VAL_22]], %[[VAL_29]] : !torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>
// CHECK:         }
func.func @torch.aten.clamp.Tensor$basic(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.vtensor<[1],f32>) -> (!torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>) {
  %none = torch.constant.none
  %0 = torch.aten.clamp.Tensor %arg0, %arg1, %none : !torch.vtensor<[3,5],f32>, !torch.vtensor<[1],f32>, !torch.none -> !torch.vtensor<[3,5],f32>
  %1 = torch.aten.clamp.Tensor %arg0, %none, %arg2 : !torch.vtensor<[3,5],f32>, !torch.none, !torch.vtensor<[1],f32> -> !torch.vtensor<[3,5],f32>
  %2 = torch.aten.clamp.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32> -> !torch.vtensor<[3,5],f32>
  return %0, %1, %2 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.prims.collapse$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,12],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,4],f32> -> tensor<2x3x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[2, 12]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_4]] : (tensor<2x3x4xf32>, !tosa.shape<2>) -> tensor<2x12xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<2x12xf32> -> !torch.vtensor<[2,12],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[2,12],f32>
// CHECK:         }
func.func @torch.prims.collapse$basic(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,12],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prims.collapse %arg0, %int1, %int2 : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,12],f32>
  return %0 : !torch.vtensor<[2,12],f32>
}

// -----

func.func @torch.aten.avg_pool1d.count_include_pad_unsupported_value(%arg0: !torch.vtensor<[1,512,10],f32>) -> !torch.vtensor<[1,512,10],f32> {
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %count_include_pad = torch.constant.bool true
  %0 = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten.avg_pool1d' that was explicitly marked illegal}}
  %3 = torch.aten.avg_pool1d %arg0, %0, %1, %2, %false, %count_include_pad : !torch.vtensor<[1,512,10],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,512,10],f32>
  return %3 : !torch.vtensor<[1,512,10],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.reflection_pad1d$basic(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !torch.vtensor<[1,2,4],f32>) -> !torch.vtensor<[1,2,8],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,2,4],f32> -> tensor<1x2x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[0, 0, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 2, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_7:.*]] = tosa.slice %[[VAL_1]], %[[VAL_5]], %[[VAL_6]] : (tensor<1x2x4xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.reverse %[[VAL_7]] {axis = 2 : i32} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK-DAG:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[0, 0, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[1, 2, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_11:.*]] = tosa.slice %[[VAL_1]], %[[VAL_9]], %[[VAL_10]] : (tensor<1x2x4xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x2x1xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.reverse %[[VAL_11]] {axis = 2 : i32} : (tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.concat %[[VAL_8]], %[[VAL_1]], %[[VAL_12]] {axis = 2 : i32} : (tensor<1x2x3xf32>, tensor<1x2x4xf32>, tensor<1x2x1xf32>) -> tensor<1x2x8xf32>
// CHECK:           %[[VAL_14:.*]] = torch_c.from_builtin_tensor %[[VAL_13]] : tensor<1x2x8xf32> -> !torch.vtensor<[1,2,8],f32>
// CHECK:           return %[[VAL_14]] : !torch.vtensor<[1,2,8],f32>
// CHECK:         }
func.func @torch.aten.reflection_pad1d$basic(%arg0: !torch.vtensor<[1,2,4],f32>) -> !torch.vtensor<[1,2,8],f32> {
  %int3 = torch.constant.int 3
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[1,2,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,8],f32>
  return %1 : !torch.vtensor<[1,2,8],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.reflection_pad2d$basic(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !torch.vtensor<[1,20,20],f32>) -> !torch.vtensor<[1,40,40],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,20,20],f32> -> tensor<1x20x20xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 10
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[0, 0, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[1, 20, 10]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_6:.*]] = tosa.slice %[[VAL_1]], %[[VAL_4]], %[[VAL_5]] : (tensor<1x20x20xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x20x10xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.reverse %[[VAL_6]] {axis = 2 : i32} : (tensor<1x20x10xf32>) -> tensor<1x20x10xf32>
// CHECK-DAG:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[0, 0, 9]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[1, 20, 10]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_10:.*]] = tosa.slice %[[VAL_1]], %[[VAL_8]], %[[VAL_9]] : (tensor<1x20x20xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x20x10xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reverse %[[VAL_10]] {axis = 2 : i32} : (tensor<1x20x10xf32>) -> tensor<1x20x10xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.concat %[[VAL_7]], %[[VAL_1]], %[[VAL_11]] {axis = 2 : i32} : (tensor<1x20x10xf32>, tensor<1x20x20xf32>, tensor<1x20x10xf32>) -> tensor<1x20x40xf32>
// CHECK-DAG:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[0, 1, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[1, 10, 40]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_15:.*]] = tosa.slice %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : (tensor<1x20x40xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x10x40xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.reverse %[[VAL_15]] {axis = 1 : i32} : (tensor<1x10x40xf32>) -> tensor<1x10x40xf32>
// CHECK-DAG:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[0, 9, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[1, 10, 40]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_12]], %[[VAL_17]], %[[VAL_18]] : (tensor<1x20x40xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x10x40xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.reverse %[[VAL_19]] {axis = 1 : i32} : (tensor<1x10x40xf32>) -> tensor<1x10x40xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.concat %[[VAL_16]], %[[VAL_12]], %[[VAL_20]] {axis = 1 : i32} : (tensor<1x10x40xf32>, tensor<1x20x40xf32>, tensor<1x10x40xf32>) -> tensor<1x40x40xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_21]] : tensor<1x40x40xf32> -> !torch.vtensor<[1,40,40],f32>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[1,40,40],f32>
// CHECK:         }
func.func @torch.aten.reflection_pad2d$basic(%arg0: !torch.vtensor<[1,20,20],f32>) -> !torch.vtensor<[1,40,40],f32> {
  %int10 = torch.constant.int 10
  %0 = torch.prim.ListConstruct %int10, %int10, %int10, %int10 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad2d %arg0, %0 : !torch.vtensor<[1,20,20],f32>, !torch.list<int> -> !torch.vtensor<[1,40,40],f32>
  return %1 : !torch.vtensor<[1,40,40],f32>
}


// -----
// CHECK-LABEL:   func.func @torch.aten.reflection_pad3d$basic(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !torch.vtensor<[4,5,7,3,4],f32>) -> !torch.vtensor<[4,5,11,7,8],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5,7,3,4],f32> -> tensor<4x5x7x3x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 0, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[4, 5, 7, 3, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_6:.*]] = tosa.slice %[[VAL_1]], %[[VAL_4]], %[[VAL_5]] : (tensor<4x5x7x3x4xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x5x7x3x2xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.reverse %[[VAL_6]] {axis = 4 : i32} : (tensor<4x5x7x3x2xf32>) -> tensor<4x5x7x3x2xf32>
// CHECK-DAG:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 0, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[4, 5, 7, 3, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_10:.*]] = tosa.slice %[[VAL_1]], %[[VAL_8]], %[[VAL_9]] : (tensor<4x5x7x3x4xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x5x7x3x2xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.reverse %[[VAL_10]] {axis = 4 : i32} : (tensor<4x5x7x3x2xf32>) -> tensor<4x5x7x3x2xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.concat %[[VAL_7]], %[[VAL_1]], %[[VAL_11]] {axis = 4 : i32} : (tensor<4x5x7x3x2xf32>, tensor<4x5x7x3x4xf32>, tensor<4x5x7x3x2xf32>) -> tensor<4x5x7x3x8xf32>
// CHECK-DAG:           %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 1, 0]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[4, 5, 7, 2, 8]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_15:.*]] = tosa.slice %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : (tensor<4x5x7x3x8xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x5x7x2x8xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.reverse %[[VAL_15]] {axis = 3 : i32} : (tensor<4x5x7x2x8xf32>) -> tensor<4x5x7x2x8xf32>
// CHECK-DAG:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<0> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[4, 5, 7, 2, 8]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_12]], %[[VAL_17]], %[[VAL_18]] : (tensor<4x5x7x3x8xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x5x7x2x8xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.reverse %[[VAL_19]] {axis = 3 : i32} : (tensor<4x5x7x2x8xf32>) -> tensor<4x5x7x2x8xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.concat %[[VAL_16]], %[[VAL_12]], %[[VAL_20]] {axis = 3 : i32} : (tensor<4x5x7x2x8xf32>, tensor<4x5x7x3x8xf32>, tensor<4x5x7x2x8xf32>) -> tensor<4x5x7x7x8xf32>
// CHECK-DAG:           %[[VAL_22:.*]] = tosa.const_shape  {values = dense<[0, 0, 1, 0, 0]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG:           %[[VAL_23:.*]] = tosa.const_shape  {values = dense<[4, 5, 2, 7, 8]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_24:.*]] = tosa.slice %[[VAL_21]], %[[VAL_22]], %[[VAL_23]] : (tensor<4x5x7x7x8xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x5x2x7x8xf32>
// CHECK:           %[[VAL_25:.*]] = tosa.reverse %[[VAL_24]] {axis = 2 : i32} : (tensor<4x5x2x7x8xf32>) -> tensor<4x5x2x7x8xf32>
// CHECK-DAG:           %[[VAL_26:.*]] = tosa.const_shape  {values = dense<[0, 0, 4, 0, 0]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG:           %[[VAL_27:.*]] = tosa.const_shape  {values = dense<[4, 5, 2, 7, 8]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_28:.*]] = tosa.slice %[[VAL_21]], %[[VAL_26]], %[[VAL_27]] : (tensor<4x5x7x7x8xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x5x2x7x8xf32>
// CHECK:           %[[VAL_29:.*]] = tosa.reverse %[[VAL_28]] {axis = 2 : i32} : (tensor<4x5x2x7x8xf32>) -> tensor<4x5x2x7x8xf32>
// CHECK:           %[[VAL_30:.*]] = tosa.concat %[[VAL_25]], %[[VAL_21]], %[[VAL_29]] {axis = 2 : i32} : (tensor<4x5x2x7x8xf32>, tensor<4x5x7x7x8xf32>, tensor<4x5x2x7x8xf32>) -> tensor<4x5x11x7x8xf32>
// CHECK:           %[[VAL_31:.*]] = torch_c.from_builtin_tensor %[[VAL_30]] : tensor<4x5x11x7x8xf32> -> !torch.vtensor<[4,5,11,7,8],f32>
// CHECK:           return %[[VAL_31]] : !torch.vtensor<[4,5,11,7,8],f32>
// CHECK:         }
func.func @torch.aten.reflection_pad3d$basic(%arg0: !torch.vtensor<[4,5,7,3,4],f32>) -> !torch.vtensor<[4,5,11,7,8],f32> {
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int2, %int2, %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.reflection_pad3d %arg0, %0 : !torch.vtensor<[4,5,7,3,4],f32>, !torch.list<int> -> !torch.vtensor<[4,5,11,7,8],f32>
    return %1 : !torch.vtensor<[4,5,11,7,8],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.replication_pad2d$basic(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,10,6],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,3,3],f32> -> tensor<1x1x3x3xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[1, 1, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_9:.*]] = tosa.slice %[[VAL_1]], %[[VAL_7]], %[[VAL_8]] : (tensor<1x1x3x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x3x1xf32>
// CHECK-DAG:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<[1, 1, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_12:.*]] = tosa.slice %[[VAL_1]], %[[VAL_10]], %[[VAL_11]] : (tensor<1x1x3x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x3x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.concat %[[VAL_9]], %[[VAL_1]], %[[VAL_12]], %[[VAL_12]] {axis = 3 : i32} : (tensor<1x1x3x1xf32>, tensor<1x1x3x3xf32>, tensor<1x1x3x1xf32>, tensor<1x1x3x1xf32>) -> tensor<1x1x3x6xf32>
// CHECK-DAG:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 1, 1, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_16:.*]] = tosa.slice %[[VAL_13]], %[[VAL_14]], %[[VAL_15]] : (tensor<1x1x3x6xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x1x6xf32>
// CHECK-DAG:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[0, 0, 2, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[1, 1, 1, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_13]], %[[VAL_17]], %[[VAL_18]] : (tensor<1x1x3x6xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x1x6xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.concat %[[VAL_16]], %[[VAL_16]], %[[VAL_16]], %[[VAL_13]], %[[VAL_19]], %[[VAL_19]], %[[VAL_19]], %[[VAL_19]] {axis = 2 : i32} : (tensor<1x1x1x6xf32>, tensor<1x1x1x6xf32>, tensor<1x1x1x6xf32>, tensor<1x1x3x6xf32>, tensor<1x1x1x6xf32>, tensor<1x1x1x6xf32>, tensor<1x1x1x6xf32>, tensor<1x1x1x6xf32>) -> tensor<1x1x10x6xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_20]] : tensor<1x1x10x6xf32> -> !torch.vtensor<[1,1,10,6],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[1,1,10,6],f32>
// CHECK:         }
func.func @torch.aten.replication_pad2d$basic(%arg0: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,10,6],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int1, %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.replication_pad2d %arg0, %0 : !torch.vtensor<[1,1,3,3],f32>, !torch.list<int> -> !torch.vtensor<[1,1,10,6],f32>
  return %1 : !torch.vtensor<[1,1,10,6],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.outer$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[3],f32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4],f32> -> tensor<4xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[3, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<3xf32>, !tosa.shape<2>) -> tensor<3x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.tile %[[VAL_5]], %[[VAL_6]] : (tensor<3x1xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_8]] : (tensor<4xf32>, !tosa.shape<2>) -> tensor<1x4xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[3, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_11:.*]] = tosa.tile %[[VAL_9]], %[[VAL_10]] : (tensor<1x4xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_13:.*]] = tosa.mul %[[VAL_7]], %[[VAL_11]], %[[VAL_12]] : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_14:.*]] = torch_c.from_builtin_tensor %[[VAL_13]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_14]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.outer$basic(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.outer %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.prims.split_dim$basic(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[1,8,3,3],si64>) -> !torch.vtensor<[1,2,2,2,3,3],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,8,3,3],si64> -> tensor<1x8x3x3xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[1, 2, 4, 3, 3]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_4]] : (tensor<1x8x3x3xi64>, !tosa.shape<5>) -> tensor<1x2x4x3x3xi64>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 2, 2, 2, 3, 3]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<1x2x4x3x3xi64>, !tosa.shape<6>) -> tensor<1x2x2x2x3x3xi64>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<1x2x2x2x3x3xi64> -> !torch.vtensor<[1,2,2,2,3,3],si64>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[1,2,2,2,3,3],si64>
// CHECK:         }
func.func @torch.prims.split_dim$basic(%arg0: !torch.vtensor<[1,8,3,3],si64>) -> !torch.vtensor<[1,2,2,2,3,3],si64> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prims.split_dim %arg0, %int1, %int2 : !torch.vtensor<[1,8,3,3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,3,3],si64>
  %1 = torch.prims.split_dim %0, %int2, %int2 : !torch.vtensor<[1,2,4,3,3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2,3,3],si64>
  return %1 : !torch.vtensor<[1,2,2,2,3,3],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.upsample_nearest2d$basic(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !torch.vtensor<[1,1,2,3],f64>) -> !torch.vtensor<[1,1,8,9],f64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,2,3],f64> -> tensor<1x1x2x3xf64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 4.000000e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.float 3.000000e+00
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[1, 1, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_7]] : (tensor<1x1x2x3xf64>, !tosa.shape<3>) -> tensor<1x1x6xf64>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3, 4, 4, 4, 5, 5, 5]]]> : tensor<1x1x72xi32>}> : () -> tensor<1x1x72xi32>
// CHECK:           %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[1, 1, 72, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_11:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_10]] : (tensor<1x1x72xi32>, !tosa.shape<4>) -> tensor<1x1x72x1xi32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x72x1xi32>}> : () -> tensor<1x1x72x1xi32>
// CHECK:           %[[VAL_13:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x72x1xi32>}> : () -> tensor<1x1x72x1xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.concat %[[VAL_12]], %[[VAL_13]], %[[VAL_11]] {axis = 3 : i32} : (tensor<1x1x72x1xi32>, tensor<1x1x72x1xi32>, tensor<1x1x72x1xi32>) -> tensor<1x1x72x3xi32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 6, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_15]] : (tensor<1x1x6xf64>, !tosa.shape<3>) -> tensor<1x6x1xf64>
// CHECK:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<[72, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_18:.*]] = tosa.reshape %[[VAL_14]], %[[VAL_17]] : (tensor<1x1x72x3xi32>, !tosa.shape<2>) -> tensor<72x3xi32>
// CHECK:           %[[VAL_19:.*]] = "tosa.const"() <{values = dense<[6, 6, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_20:.*]] = tosa.const_shape  {values = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_21:.*]] = tosa.reshape %[[VAL_19]], %[[VAL_20]] : (tensor<3xi32>, !tosa.shape<2>) -> tensor<1x3xi32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_23:.*]] = tosa.mul %[[VAL_18]], %[[VAL_21]], %[[VAL_22]] : (tensor<72x3xi32>, tensor<1x3xi32>, tensor<1xi8>) -> tensor<72x3xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.reduce_sum %[[VAL_23]] {axis = 1 : i32} : (tensor<72x3xi32>) -> tensor<72x1xi32>
// CHECK:           %[[VAL_25:.*]] = tosa.const_shape  {values = dense<[1, 72]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_26:.*]] = tosa.reshape %[[VAL_24]], %[[VAL_25]] : (tensor<72x1xi32>, !tosa.shape<2>) -> tensor<1x72xi32>
// CHECK:           %[[VAL_27:.*]] = tosa.gather %[[VAL_16]], %[[VAL_26]] : (tensor<1x6x1xf64>, tensor<1x72xi32>) -> tensor<1x72x1xf64>
// CHECK:           %[[VAL_28:.*]] = tosa.const_shape  {values = dense<[1, 1, 72]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_29:.*]] = tosa.reshape %[[VAL_27]], %[[VAL_28]] : (tensor<1x72x1xf64>, !tosa.shape<3>) -> tensor<1x1x72xf64>
// CHECK:           %[[VAL_30:.*]] = tosa.const_shape  {values = dense<[1, 1, 8, 9]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_31:.*]] = tosa.reshape %[[VAL_29]], %[[VAL_30]] : (tensor<1x1x72xf64>, !tosa.shape<4>) -> tensor<1x1x8x9xf64>
// CHECK:           %[[VAL_32:.*]] = torch_c.from_builtin_tensor %[[VAL_31]] : tensor<1x1x8x9xf64> -> !torch.vtensor<[1,1,8,9],f64>
// CHECK:           return %[[VAL_32]] : !torch.vtensor<[1,1,8,9],f64>
// CHECK:         }
func.func @torch.aten.upsample_nearest2d$basic(%arg0: !torch.vtensor<[1,1,2,3],f64>) -> !torch.vtensor<[1,1,8,9],f64> {
  %float4.000000e00 = torch.constant.float 4.000000e+00
  %float3.000000e00 = torch.constant.float 3.000000e+00
  %int8 = torch.constant.int 8
  %int9 = torch.constant.int 9
  %0 = torch.prim.ListConstruct %int8, %int9 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.upsample_nearest2d %arg0, %0, %float4.000000e00, %float3.000000e00 : !torch.vtensor<[1,1,2,3],f64>, !torch.list<int>, !torch.float, !torch.float -> !torch.vtensor<[1,1,8,9],f64>
  return %1 : !torch.vtensor<[1,1,8,9],f64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.upsample_nearest2d.vec$basic(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: !torch.vtensor<[1,1,4,5],f32>) -> !torch.vtensor<[1,1,2,7],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,4,5],f32> -> tensor<1x1x4x5xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 7
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 1, 20]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_6]] : (tensor<1x1x4x5xf32>, !tosa.shape<3>) -> tensor<1x1x20xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[0, 0, 1, 2, 2, 3, 4, 10, 10, 11, 12, 12, 13, 14]]]> : tensor<1x1x14xi32>}> : () -> tensor<1x1x14xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[1, 1, 14, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_9]] : (tensor<1x1x14xi32>, !tosa.shape<4>) -> tensor<1x1x14x1xi32>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x14x1xi32>}> : () -> tensor<1x1x14x1xi32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x14x1xi32>}> : () -> tensor<1x1x14x1xi32>
// CHECK:           %[[VAL_13:.*]] = tosa.concat %[[VAL_11]], %[[VAL_12]], %[[VAL_10]] {axis = 3 : i32} : (tensor<1x1x14x1xi32>, tensor<1x1x14x1xi32>, tensor<1x1x14x1xi32>) -> tensor<1x1x14x3xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[1, 20, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_14]] : (tensor<1x1x20xf32>, !tosa.shape<3>) -> tensor<1x20x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.const_shape  {values = dense<[14, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_17:.*]] = tosa.reshape %[[VAL_13]], %[[VAL_16]] : (tensor<1x1x14x3xi32>, !tosa.shape<2>) -> tensor<14x3xi32>
// CHECK:           %[[VAL_18:.*]] = "tosa.const"() <{values = dense<[20, 20, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_20:.*]] = tosa.reshape %[[VAL_18]], %[[VAL_19]] : (tensor<3xi32>, !tosa.shape<2>) -> tensor<1x3xi32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_22:.*]] = tosa.mul %[[VAL_17]], %[[VAL_20]], %[[VAL_21]] : (tensor<14x3xi32>, tensor<1x3xi32>, tensor<1xi8>) -> tensor<14x3xi32>
// CHECK:           %[[VAL_23:.*]] = tosa.reduce_sum %[[VAL_22]] {axis = 1 : i32} : (tensor<14x3xi32>) -> tensor<14x1xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.const_shape  {values = dense<[1, 14]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_25:.*]] = tosa.reshape %[[VAL_23]], %[[VAL_24]] : (tensor<14x1xi32>, !tosa.shape<2>) -> tensor<1x14xi32>
// CHECK:           %[[VAL_26:.*]] = tosa.gather %[[VAL_15]], %[[VAL_25]] : (tensor<1x20x1xf32>, tensor<1x14xi32>) -> tensor<1x14x1xf32>
// CHECK:           %[[VAL_27:.*]] = tosa.const_shape  {values = dense<[1, 1, 14]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_28:.*]] = tosa.reshape %[[VAL_26]], %[[VAL_27]] : (tensor<1x14x1xf32>, !tosa.shape<3>) -> tensor<1x1x14xf32>
// CHECK:           %[[VAL_29:.*]] = tosa.const_shape  {values = dense<[1, 1, 2, 7]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_30:.*]] = tosa.reshape %[[VAL_28]], %[[VAL_29]] : (tensor<1x1x14xf32>, !tosa.shape<4>) -> tensor<1x1x2x7xf32>
// CHECK:           %[[VAL_31:.*]] = torch_c.from_builtin_tensor %[[VAL_30]] : tensor<1x1x2x7xf32> -> !torch.vtensor<[1,1,2,7],f32>
// CHECK:           return %[[VAL_31]] : !torch.vtensor<[1,1,2,7],f32>
// CHECK:         }
func.func @torch.aten.upsample_nearest2d.vec$basic(%arg0: !torch.vtensor<[1,1,4,5],f32>) -> !torch.vtensor<[1,1,2,7],f32> {
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %int7 = torch.constant.int 7
  %0 = torch.prim.ListConstruct %int2, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.upsample_nearest2d.vec %arg0, %0, %none : !torch.vtensor<[1,1,4,5],f32>, !torch.list<int>, !torch.none -> !torch.vtensor<[1,1,2,7],f32>
  return %1 : !torch.vtensor<[1,1,2,7],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gelu$none(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[1,1500,1536],f32>) -> !torch.vtensor<[1,1500,1536],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1500,1536],f32> -> tensor<1x1500x1536xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.str "none"
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0.707106769> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_7:.*]] = tosa.const_shape {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_7]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.const_shape {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_9]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_11]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.const_shape {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_14:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_13]] : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.sub %[[VAL_1]], %[[VAL_8]] : (tensor<1x1500x1536xf32>, tensor<1x1x1xf32>) -> tensor<1x1500x1536xf32>
// CHECK:           %[[VAL_16:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_17:.*]] = tosa.mul %[[VAL_15]], %[[VAL_14]], %[[VAL_16]] : (tensor<1x1500x1536xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1500x1536xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.erf %[[VAL_17]] : (tensor<1x1500x1536xf32>) -> tensor<1x1500x1536xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.add %[[VAL_10]], %[[VAL_18]] : (tensor<1x1x1xf32>, tensor<1x1500x1536xf32>) -> tensor<1x1500x1536xf32>
// CHECK:           %[[VAL_20:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_21:.*]] = tosa.mul %[[VAL_12]], %[[VAL_19]], %[[VAL_20]] : (tensor<1x1x1xf32>, tensor<1x1500x1536xf32>, tensor<1xi8>) -> tensor<1x1500x1536xf32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_23:.*]] = tosa.mul %[[VAL_1]], %[[VAL_21]], %[[VAL_22]] : (tensor<1x1500x1536xf32>, tensor<1x1500x1536xf32>, tensor<1xi8>) -> tensor<1x1500x1536xf32>
// CHECK:           %[[VAL_24:.*]] = torch_c.from_builtin_tensor %[[VAL_23]] : tensor<1x1500x1536xf32> -> !torch.vtensor<[1,1500,1536],f32>
// CHECK:           return %[[VAL_24]] : !torch.vtensor<[1,1500,1536],f32>
// CHECK:         }
func.func @torch.aten.gelu$none(%arg0: !torch.vtensor<[1,1500,1536],f32>) -> !torch.vtensor<[1,1500,1536],f32> {
  %str = torch.constant.str "none"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[1,1500,1536],f32>, !torch.str -> !torch.vtensor<[1,1500,1536],f32>
  return %0 : !torch.vtensor<[1,1500,1536],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gelu$tanh(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[5,3],f32>) -> !torch.vtensor<[5,3],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,3],f32> -> tensor<5x3xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.str "tanh"
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<3.000000e+00> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<4.471500e-02> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0.636619746> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_9:.*]] = tosa.mul %[[VAL_3]], %[[VAL_1]], %[[VAL_8]] : (tensor<5x3xf32>, tensor<5x3xf32>, tensor<1xi8>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.pow %[[VAL_7]], %[[VAL_3]] : (tensor<5x3xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.pow %[[VAL_1]], %[[VAL_5]] : (tensor<5x3xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_13:.*]] = tosa.mul %[[VAL_6]], %[[VAL_11]], %[[VAL_12]] : (tensor<5x3xf32>, tensor<5x3xf32>, tensor<1xi8>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.add %[[VAL_1]], %[[VAL_13]] : (tensor<5x3xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_15:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_16:.*]] = tosa.mul %[[VAL_10]], %[[VAL_14]], %[[VAL_15]] : (tensor<5x3xf32>, tensor<5x3xf32>, tensor<1xi8>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.tanh %[[VAL_16]] : (tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.add %[[VAL_4]], %[[VAL_17]] : (tensor<5x3xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_19:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_20:.*]] = tosa.mul %[[VAL_9]], %[[VAL_18]], %[[VAL_19]] : (tensor<5x3xf32>, tensor<5x3xf32>, tensor<1xi8>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_20]] : tensor<5x3xf32> -> !torch.vtensor<[5,3],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[5,3],f32>
// CHECK:         }
func.func @torch.aten.gelu$tanh(%arg0: !torch.vtensor<[5,3],f32>) -> !torch.vtensor<[5,3],f32> {
  %str = torch.constant.str "tanh"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[5,3],f32>, !torch.str -> !torch.vtensor<[5,3],f32>
  return %0 : !torch.vtensor<[5,3],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.exp$int(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.exp %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.exp$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.exp %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log10$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1.000000e+01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_3]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.log %[[VAL_1]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.log %[[VAL_4]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.reciprocal %[[VAL_6]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_9:.*]] = tosa.mul %[[VAL_5]], %[[VAL_7]], %[[VAL_8]] : (tensor<3x4xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_10]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.log10$basic(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.log10 %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log10$int(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1.000000e+01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.log %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.log %[[VAL_5]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.reciprocal %[[VAL_7]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_10:.*]] = tosa.mul %[[VAL_6]], %[[VAL_8]], %[[VAL_9]] : (tensor<3x4xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.log10$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.log10 %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log1p$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_3]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.add %[[VAL_1]], %[[VAL_4]] : (tensor<3x4xf32>, tensor<1x1xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.log %[[VAL_5]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.log1p$basic(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.log1p %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log1p$int(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.add %[[VAL_2]], %[[VAL_5]] : (tensor<3x4xf32>, tensor<1x1xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.log %[[VAL_6]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.log1p$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.log1p %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.logit$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 9.9999999999999995E-8
// CHECK:           %[[VAL_3:.*]] = tosa.clamp %[[VAL_1]] {max_val = 0.99999988 : f32, min_val = 1.000000e-07 : f32} : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_5]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.sub %[[VAL_6]], %[[VAL_3]] : (tensor<1x1xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.reciprocal %[[VAL_7]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_10:.*]] = tosa.mul %[[VAL_3]], %[[VAL_8]], %[[VAL_9]] : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.log %[[VAL_10]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_12:.*]] = torch_c.from_builtin_tensor %[[VAL_11]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_12]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.logit$basic(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %float9.999990e-08 = torch.constant.float 9.9999999999999995E-8
  %0 = torch.aten.logit %arg0, %float9.999990e-08 : !torch.vtensor<[3,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.logit$int(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 9.9999999999999995E-8
// CHECK:           %[[VAL_3:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.clamp %[[VAL_3]] {max_val = 0.99999988 : f32, min_val = 1.000000e-07 : f32} : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_6]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.sub %[[VAL_7]], %[[VAL_4]] : (tensor<1x1xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.reciprocal %[[VAL_8]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_4]], %[[VAL_9]], %[[VAL_10]] : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.log %[[VAL_11]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.logit$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %float9.999990e-08 = torch.constant.float 9.9999999999999995E-8
  %0 = torch.aten.logit %arg0, %float9.999990e-08 : !torch.vtensor<[3,4],si32>, !torch.float -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log$int(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.log %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.log$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.log %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log2$int(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.693147182> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_3]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.log %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_7:.*]] = tosa.mul %[[VAL_5]], %[[VAL_4]], %[[VAL_6]] : (tensor<3x4xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.log2$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.log2 %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.erf$int(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.erf %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.erf$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.erf %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.lt.Scalar$intfloat(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4],si64> -> tensor<4xi64>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 1.100000e+00
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1.100000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.cast %[[VAL_3]] : (tensor<f32>) -> tensor<f64>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_5]] : (tensor<f64>, !tosa.shape<1>) -> tensor<1xf64>
// CHECK:           %[[VAL_7:.*]] = tosa.cast %[[VAL_1]] : (tensor<4xi64>) -> tensor<4xf64>
// CHECK:           %[[VAL_8:.*]] = tosa.greater %[[VAL_6]], %[[VAL_7]] : (tensor<1xf64>, tensor<4xf64>) -> tensor<4xi1>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<4xi1> -> !torch.vtensor<[4],i1>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[4],i1>
// CHECK:         }
func.func @torch.aten.lt.Scalar$intfloat(%arg0: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],i1> {
  %float1.100000e00 = torch.constant.float 1.100000e+00
  %0 = torch.aten.lt.Scalar %arg0, %float1.100000e00 : !torch.vtensor<[4],si64>, !torch.float -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sigmoid$int(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[3,5],si32>) -> !torch.vtensor<[3,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5],si32> -> tensor<3x5xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x5xi32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.sigmoid %[[VAL_2]] : (tensor<3x5xf32>) -> tensor<3x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<3x5xf32> -> !torch.vtensor<[3,5],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[3,5],f32>
// CHECK:         }
func.func @torch.aten.sigmoid$int(%arg0: !torch.vtensor<[3,5],si32>) -> !torch.vtensor<[3,5],f32> {
  %0 = torch.aten.sigmoid %arg0 : !torch.vtensor<[3,5],si32> -> !torch.vtensor<[3,5],f32>
  return %0 : !torch.vtensor<[3,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.tan$basic(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.sin %[[VAL_1]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.cos %[[VAL_1]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reciprocal %[[VAL_3]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_6:.*]] = tosa.mul %[[VAL_2]], %[[VAL_4]], %[[VAL_5]] : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.tan$basic(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.tan %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.tan$int(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.sin %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.cos %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.reciprocal %[[VAL_4]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_7:.*]] = tosa.mul %[[VAL_3]], %[[VAL_5]], %[[VAL_6]] : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<1xi8>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.tan$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.tan %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.tanh$int(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.tanh %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.tanh$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.pow.Tensor_Tensor$intfloat(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: !torch.vtensor<[3,4,5],si32>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,4,5],f32> -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4,5],si32> -> tensor<3x4x5xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.cast %[[VAL_3]] : (tensor<3x4x5xi32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.pow %[[VAL_4]], %[[VAL_2]] : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<3x4x5xf32> -> !torch.vtensor<[3,4,5],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[3,4,5],f32>
// CHECK:         }
func.func @torch.aten.pow.Tensor_Tensor$intfloat(%arg0: !torch.vtensor<[3,4,5],si32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
  %0 = torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unfold$basic(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[6,4],f32>) -> !torch.vtensor<[3,4,2],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[6,4],f32> -> tensor<6x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<{{\[\[}}0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]> : tensor<6x4xi32>}> : () -> tensor<6x4xi32>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[6, 4, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_5]] : (tensor<6x4xi32>, !tosa.shape<3>) -> tensor<6x4x1xi32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]], {{\[\[}}0], [1], [2], [3]]]> : tensor<6x4x1xi32>}> : () -> tensor<6x4x1xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.concat %[[VAL_6]], %[[VAL_7]] {axis = 2 : i32} : (tensor<6x4x1xi32>, tensor<6x4x1xi32>) -> tensor<6x4x2xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[1, 24, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_9]] : (tensor<6x4xf32>, !tosa.shape<3>) -> tensor<1x24x1xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.const_shape  {values = dense<[24, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_12:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_11]] : (tensor<6x4x2xi32>, !tosa.shape<2>) -> tensor<24x2xi32>
// CHECK:           %[[VAL_13:.*]] = "tosa.const"() <{values = dense<[4, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[1, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_15:.*]] = tosa.reshape %[[VAL_13]], %[[VAL_14]] : (tensor<2xi32>, !tosa.shape<2>) -> tensor<1x2xi32>
// CHECK:           %[[VAL_16:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_17:.*]] = tosa.mul %[[VAL_12]], %[[VAL_15]], %[[VAL_16]] : (tensor<24x2xi32>, tensor<1x2xi32>, tensor<1xi8>) -> tensor<24x2xi32>
// CHECK:           %[[VAL_18:.*]] = tosa.reduce_sum %[[VAL_17]] {axis = 1 : i32} : (tensor<24x2xi32>) -> tensor<24x1xi32>
// CHECK:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<[1, 24]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_20:.*]] = tosa.reshape %[[VAL_18]], %[[VAL_19]] : (tensor<24x1xi32>, !tosa.shape<2>) -> tensor<1x24xi32>
// CHECK:           %[[VAL_21:.*]] = tosa.gather %[[VAL_10]], %[[VAL_20]] : (tensor<1x24x1xf32>, tensor<1x24xi32>) -> tensor<1x24x1xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.const_shape  {values = dense<[6, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_23:.*]] = tosa.reshape %[[VAL_21]], %[[VAL_22]] : (tensor<1x24x1xf32>, !tosa.shape<2>) -> tensor<6x4xf32>
// CHECK:           %[[VAL_24:.*]] = tosa.const_shape  {values = dense<[3, 2, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_25:.*]] = tosa.reshape %[[VAL_23]], %[[VAL_24]] : (tensor<6x4xf32>, !tosa.shape<3>) -> tensor<3x2x4xf32>
// CHECK:           %[[VAL_26:.*]] = tosa.transpose %[[VAL_25]] {perms = array<i32: 0, 2, 1>} : (tensor<3x2x4xf32>) -> tensor<3x4x2xf32>
// CHECK:           %[[VAL_27:.*]] = torch_c.from_builtin_tensor %[[VAL_26]] : tensor<3x4x2xf32> -> !torch.vtensor<[3,4,2],f32>
// CHECK:           return %[[VAL_27]] : !torch.vtensor<[3,4,2],f32>
// CHECK:         }
func.func @torch.aten.unfold$basic(%arg0: !torch.vtensor<[6,4],f32>) -> !torch.vtensor<[3,4,2],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %0 = torch.aten.unfold %arg0, %int0, %int2, %int2 : !torch.vtensor<[6,4],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,4,2],f32>
  return %0 : !torch.vtensor<[3,4,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unfold$rank_zero(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[1],f32>
// CHECK:         }
func.func @torch.aten.unfold$rank_zero(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.unfold %arg0, %int0, %int1, %int1 : !torch.vtensor<[],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.expm1$basic(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_3:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_3]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.exp %[[VAL_1]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.sub %[[VAL_5]], %[[VAL_4]] : (tensor<3x4xf32>, tensor<1x1xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.expm1$basic(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.expm1 %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.expm1$int(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,4],si32> -> tensor<3x4xi32>
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_1]] : (tensor<3x4xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           %[[VAL_5:.*]] = tosa.reshape %[[VAL_3]], %[[VAL_4]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.exp %[[VAL_2]] : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.sub %[[VAL_6]], %[[VAL_5]] : (tensor<3x4xf32>, tensor<1x1xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[3,4],f32>
// CHECK:         }
func.func @torch.aten.expm1$int(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.expm1 %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd$basic(
// CHECK-SAME:                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,1,20,20,4,4],f32>) -> !torch.vtensor<[1,1,20,20,4,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,20,20,4,4],f32> -> tensor<1x1x20x20x4x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 0xFFF0000000000000
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<12xindex>} : () -> !tosa.shape<12>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = tosa.const_shape  {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           %[[VAL_9:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_8]] : (tensor<f32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.pad %[[VAL_1]], %[[VAL_6]], %[[VAL_9]] : (tensor<1x1x20x20x4x4xf32>, !tosa.shape<12>, tensor<1xf32>) -> tensor<1x1x20x20x4x5xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<1x1x20x20x4x5xf32> -> !torch.vtensor<[1,1,20,20,4,5],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[1,1,20,20,4,5],f32>
// CHECK:         }
func.func @torch.aten.constant_pad_nd$basic(%arg0: !torch.vtensor<[1,1,20,20,4,4],f32>) -> !torch.vtensor<[1,1,20,20,4,5],f32> {
  %float-Inf = torch.constant.float 0xFFF0000000000000
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.constant_pad_nd %arg0, %0, %float-Inf : !torch.vtensor<[1,1,20,20,4,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,1,20,20,4,5],f32>
  return %1 : !torch.vtensor<[1,1,20,20,4,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$basic(
// CHECK-SAME:                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[5,2,10,20],f32>) -> !torch.vtensor<[5,10,14,24],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,2,10,20],f32> -> tensor<5x2x10x20xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_10_2_3_3_torch.float32> : tensor<10x2x3x3xf32>}> : () -> tensor<10x2x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<10xf32>}> : () -> tensor<10xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x2x10x20xf32>) -> tensor<5x10x20x2xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.transpose %[[VAL_4]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<10x2x3x3xf32>) -> tensor<10x3x3x2xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_15:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.conv2d %[[VAL_12]], %[[VAL_13]], %[[VAL_11]], %[[VAL_14]], %[[VAL_15]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 1, 1>} : (tensor<5x10x20x2xf32>, tensor<10x3x3x2xf32>, tensor<10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x14x24x10xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.transpose %[[VAL_16]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x14x24x10xf32>) -> tensor<5x10x14x24xf32>
// CHECK:           %[[VAL_18:.*]] = tensor.cast %[[VAL_17]] : tensor<5x10x14x24xf32> to tensor<5x10x14x24xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<5x10x14x24xf32> -> !torch.vtensor<[5,10,14,24],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[5,10,14,24],f32>
// CHECK:         }
func.func @torch.aten.convolution$basic(%arg0: !torch.vtensor<[5,2,10,20],f32>) -> !torch.vtensor<[5,10,14,24],f32> {
  %false = torch.constant.bool false
  %int3 = torch.constant.int 3
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_10_2_3_3_torch.float32> : tensor<10x2x3x3xf32>) : !torch.vtensor<[10,2,3,3],f32>
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int1 : !torch.vtensor<[5,2,10,20],f32>, !torch.vtensor<[10,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[5,10,14,24],f32>
  return %5 : !torch.vtensor<[5,10,14,24],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$depthwise(
// CHECK-SAME:                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[5,4,10,20],f32>) -> !torch.vtensor<[5,4,5,10],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,4,10,20],f32> -> tensor<5x4x10x20xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_4_1_3_3_torch.float32> : tensor<4x1x3x3xf32>}> : () -> tensor<4x1x3x3xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.none
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x4x10x20xf32>) -> tensor<5x10x20x4xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.transpose %[[VAL_5]] {perms = array<i32: 2, 3, 0, 1>} : (tensor<4x1x3x3xf32>) -> tensor<3x3x4x1xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[3, 3, 4, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_16:.*]] = tosa.reshape %[[VAL_14]], %[[VAL_15]] : (tensor<3x3x4x1xf32>, !tosa.shape<4>) -> tensor<3x3x4x1xf32>
// CHECK:           %[[VAL_17:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_18:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.depthwise_conv2d %[[VAL_13]], %[[VAL_16]], %[[VAL_12]], %[[VAL_17]], %[[VAL_18]] {acc_type = f32, dilation = array<i64: 3, 3>, pad = array<i64: 3, 2, 3, 2>, stride = array<i64: 2, 2>} : (tensor<5x10x20x4xf32>, tensor<3x3x4x1xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x5x10x4xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.transpose %[[VAL_19]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x5x10x4xf32>) -> tensor<5x4x5x10xf32>
// CHECK:           %[[VAL_21:.*]] = tensor.cast %[[VAL_20]] : tensor<5x4x5x10xf32> to tensor<5x4x5x10xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_21]] : tensor<5x4x5x10xf32> -> !torch.vtensor<[5,4,5,10],f32>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[5,4,5,10],f32>
// CHECK:         }
func.func @torch.aten.convolution$depthwise(%arg0: !torch.vtensor<[5,4,10,20],f32>) -> !torch.vtensor<[5,4,5,10],f32> {
  %false = torch.constant.bool false
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_4_1_3_3_torch.float32> : tensor<4x1x3x3xf32>) : !torch.vtensor<[4,1,3,3],f32>
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int4 : !torch.vtensor<[5,4,10,20],f32>, !torch.vtensor<[4,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[5,4,5,10],f32>
  return %5 : !torch.vtensor<[5,4,5,10],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$zero_pad_with_sliced_input(
// CHECK-SAME:                                                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,128,28,28],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_128_64_1_1_torch.float32> : tensor<128x64x1x1xf32>}> : () -> tensor<128x64x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = torch.constant.none
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.transpose %[[VAL_5]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<128x64x1x1xf32>) -> tensor<128x1x1x64xf32>
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_16:.*]] = tosa.const_shape  {values = dense<[1, 55, 56, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_17:.*]] = tosa.slice %[[VAL_13]], %[[VAL_15]], %[[VAL_16]] : (tensor<1x56x56x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x55x56x64xf32>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<[1, 55, 55, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_20:.*]] = tosa.slice %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : (tensor<1x55x56x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x55x55x64xf32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.conv2d %[[VAL_20]], %[[VAL_14]], %[[VAL_12]], %[[VAL_21]], %[[VAL_22]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x55x55x64xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x28x28x128xf32>
// CHECK:           %[[VAL_24:.*]] = tosa.transpose %[[VAL_23]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x28x28x128xf32>) -> tensor<1x128x28x28xf32>
// CHECK:           %[[VAL_25:.*]] = tensor.cast %[[VAL_24]] : tensor<1x128x28x28xf32> to tensor<1x128x28x28xf32>
// CHECK:           %[[VAL_26:.*]] = torch_c.from_builtin_tensor %[[VAL_25]] : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
// CHECK:           return %[[VAL_26]] : !torch.vtensor<[1,128,28,28],f32>
// CHECK:         }
func.func @torch.aten.convolution$zero_pad_with_sliced_input(%arg0: !torch.vtensor<[1,64,56,56],f32>) -> !torch.vtensor<[1,128,28,28],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_128_64_1_1_torch.float32> : tensor<128x64x1x1xf32>) : !torch.vtensor<[128,64,1,1],f32>
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int1 : !torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[128,64,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,128,28,28],f32>
  return %5 : !torch.vtensor<[1,128,28,28],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$full_dim_indivisible_by_stride_without_sliced_input(
// CHECK-SAME:                                                                                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,32,112,112],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,3,224,224],f32> -> tensor<1x3x224x224xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>}> : () -> tensor<32x3x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.transpose %[[VAL_4]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_15:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.conv2d %[[VAL_12]], %[[VAL_13]], %[[VAL_11]], %[[VAL_14]], %[[VAL_15]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x112x112x32xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.transpose %[[VAL_16]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x112x112x32xf32>) -> tensor<1x32x112x112xf32>
// CHECK:           %[[VAL_18:.*]] = tensor.cast %[[VAL_17]] : tensor<1x32x112x112xf32> to tensor<1x32x112x112xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<1x32x112x112xf32> -> !torch.vtensor<[1,32,112,112],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,32,112,112],f32>
// CHECK:         }
func.func @torch.aten.convolution$full_dim_indivisible_by_stride_without_sliced_input(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,32,112,112],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>) : !torch.vtensor<[32,3,3,3],f32>
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int1 : !torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[32,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,112,112],f32>
  return %5 : !torch.vtensor<[1,32,112,112],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$full_dim_indivisible_by_stride_with_sliced_input(
// CHECK-SAME:                                                                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,3,225,225],f32>) -> !torch.vtensor<[1,32,75,75],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,3,225,225],f32> -> tensor<1x3x225x225xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>}> : () -> tensor<32x3x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x3x225x225xf32>) -> tensor<1x225x225x3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.transpose %[[VAL_4]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3xf32>
// CHECK-DAG:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 224, 225, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_16:.*]] = tosa.slice %[[VAL_12]], %[[VAL_14]], %[[VAL_15]] : (tensor<1x225x225x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x224x225x3xf32>
// CHECK-DAG:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[1, 224, 224, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : (tensor<1x224x225x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x224x224x3xf32>
// CHECK:           %[[VAL_20:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.conv2d %[[VAL_19]], %[[VAL_13]], %[[VAL_11]], %[[VAL_20]], %[[VAL_21]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 3, 3>} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x75x75x32xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.transpose %[[VAL_22]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x75x75x32xf32>) -> tensor<1x32x75x75xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.cast %[[VAL_23]] : tensor<1x32x75x75xf32> to tensor<1x32x75x75xf32>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<1x32x75x75xf32> -> !torch.vtensor<[1,32,75,75],f32>
// CHECK:           return %[[VAL_25]] : !torch.vtensor<[1,32,75,75],f32>
// CHECK:         }
func.func @torch.aten.convolution$full_dim_indivisible_by_stride_with_sliced_input(%arg0: !torch.vtensor<[1,3,225,225],f32>) -> !torch.vtensor<[1,32,75,75],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>) : !torch.vtensor<[32,3,3,3],f32>
  %none = torch.constant.none
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int1 : !torch.vtensor<[1,3,225,225],f32>, !torch.vtensor<[32,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,75,75],f32>
  return %5 : !torch.vtensor<[1,32,75,75],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$full_dim_indivisible_by_stride_without_sliced_input_dynamic_batch(
// CHECK-SAME:     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[?,3,224,224],f32>) -> !torch.vtensor<[?,32,112,112],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,224,224],f32> -> tensor<?x3x224x224xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>}> : () -> tensor<32x3x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<?x3x224x224xf32>) -> tensor<?x224x224x3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.transpose %[[VAL_4]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_15:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.conv2d %[[VAL_12]], %[[VAL_13]], %[[VAL_11]], %[[VAL_14]], %[[VAL_15]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<?x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x112x112x32xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.transpose %[[VAL_16]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<?x112x112x32xf32>) -> tensor<?x32x112x112xf32>
// CHECK:           %[[VAL_18:.*]] = tensor.cast %[[VAL_17]] : tensor<?x32x112x112xf32> to tensor<?x32x112x112xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<?x32x112x112xf32> -> !torch.vtensor<[?,32,112,112],f32>
// CHECK:           return %[[VAL_19]]

func.func @torch.aten.convolution$full_dim_indivisible_by_stride_without_sliced_input_dynamic_batch(%arg0: !torch.vtensor<[?,3,224,224],f32>) -> !torch.vtensor<[?,32,112,112],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>) : !torch.vtensor<[32,3,3,3],f32>
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int1 : !torch.vtensor<[?,3,224,224],f32>, !torch.vtensor<[32,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,32,112,112],f32>
  return %5 : !torch.vtensor<[?,32,112,112],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$full_dim_indivisible_by_stride_with_sliced_input_dynamic_batch(
// CHECK-SAME:    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[?,3,225,225],f32>) -> !torch.vtensor<[?,32,75,75],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,225,225],f32> -> tensor<?x3x225x225xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>}> : () -> tensor<32x3x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<?x3x225x225xf32>) -> tensor<?x225x225x3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.transpose %[[VAL_4]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3xf32>
// CHECK-DAG:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[-1, 224, 225, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_16:.*]] = tosa.slice %[[VAL_12]], %[[VAL_14]], %[[VAL_15]] : (tensor<?x225x225x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x224x225x3xf32>
// CHECK-DAG:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[-1, 224, 224, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : (tensor<?x224x225x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x224x224x3xf32>
// CHECK:           %[[VAL_20:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.conv2d %[[VAL_19]], %[[VAL_13]], %[[VAL_11]], %[[VAL_20]], %[[VAL_21]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 3, 3>} : (tensor<?x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x75x75x32xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.transpose %[[VAL_22]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<?x75x75x32xf32>) -> tensor<?x32x75x75xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.cast %[[VAL_23]] : tensor<?x32x75x75xf32> to tensor<?x32x75x75xf32>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<?x32x75x75xf32> -> !torch.vtensor<[?,32,75,75],f32>
// CHECK:           return %[[VAL_25]]
func.func @torch.aten.convolution$full_dim_indivisible_by_stride_with_sliced_input_dynamic_batch(%arg0: !torch.vtensor<[?,3,225,225],f32>) -> !torch.vtensor<[?,32,75,75],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_32_3_3_3_torch.float32> : tensor<32x3x3x3xf32>) : !torch.vtensor<[32,3,3,3],f32>
  %none = torch.constant.none
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int1 : !torch.vtensor<[?,3,225,225],f32>, !torch.vtensor<[32,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,32,75,75],f32>
  return %5 : !torch.vtensor<[?,32,75,75],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$zero_pad_with_sliced_input(
// CHECK-SAME:                                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,1,56,56],f32>) -> !torch.vtensor<[1,1,27,27],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,56,56],f32> -> tensor<1x1x56x56xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_11]], %[[VAL_12]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = torch.constant.bool false
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_16:.*]] = tosa.const_shape  {values = dense<[1, 1, 55, 56]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_17:.*]] = tosa.slice %[[VAL_1]], %[[VAL_15]], %[[VAL_16]] : (tensor<1x1x56x56xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x55x56xf32>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<[1, 1, 55, 55]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_20:.*]] = tosa.slice %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : (tensor<1x1x55x56xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x55x55xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.transpose %[[VAL_20]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x55x55xf32>) -> tensor<1x55x55x1xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.max_pool2d %[[VAL_21]] {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x55x55x1xf32>) -> tensor<1x27x27x1xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.transpose %[[VAL_22]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x27x27x1xf32>) -> tensor<1x1x27x27xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.cast %[[VAL_23]] : tensor<1x1x27x27xf32> to tensor<1x1x27x27xf32>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<1x1x27x27xf32> -> !torch.vtensor<[1,1,27,27],f32>
// CHECK:           return %[[VAL_25]] : !torch.vtensor<[1,1,27,27],f32>
// CHECK:         }
func.func @torch.aten.max_pool2d$zero_pad_with_sliced_input(%arg0: !torch.vtensor<[1,1,56,56],f32>) -> !torch.vtensor<[1,1,27,27],f32> {
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int2 = torch.constant.int 2
  %int2_1 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %int0 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %2 = torch.prim.ListConstruct %int0, %int0_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %int1_3 = torch.constant.int 1
  %3 = torch.prim.ListConstruct %int1, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,1,56,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,27,27],f32>
  return %4 : !torch.vtensor<[1,1,27,27],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$full_dim_indivisible_by_stride_without_sliced_input(
// CHECK-SAME:                                                                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,1,112,112],f32>) -> !torch.vtensor<[1,1,56,56],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,112,112],f32> -> tensor<1x1x112x112xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_11]], %[[VAL_12]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_15:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x112x112xf32>) -> tensor<1x112x112x1xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.max_pool2d %[[VAL_15]] {kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<1x112x112x1xf32>) -> tensor<1x56x56x1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.transpose %[[VAL_16]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x56x56x1xf32>) -> tensor<1x1x56x56xf32>
// CHECK:           %[[VAL_18:.*]] = tensor.cast %[[VAL_17]] : tensor<1x1x56x56xf32> to tensor<1x1x56x56xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<1x1x56x56xf32> -> !torch.vtensor<[1,1,56,56],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[1,1,56,56],f32>
// CHECK:         }
func.func @torch.aten.max_pool2d$full_dim_indivisible_by_stride_without_sliced_input(%arg0: !torch.vtensor<[1,1,112,112],f32>) -> !torch.vtensor<[1,1,56,56],f32> {
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int2 = torch.constant.int 2
  %int2_1 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %int1_2 = torch.constant.int 1
  %2 = torch.prim.ListConstruct %int1, %int1_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1_3 = torch.constant.int 1
  %int1_4 = torch.constant.int 1
  %3 = torch.prim.ListConstruct %int1_3, %int1_4 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,1,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,56,56],f32>
  return %4 : !torch.vtensor<[1,1,56,56],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$full_dim_indivisible_by_stride_with_sliced_input(
// CHECK-SAME:                                                                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,1,75,75],f32>) -> !torch.vtensor<[1,1,25,25],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,75,75],f32> -> tensor<1x1x75x75xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_13:.*]] = torch.prim.ListConstruct %[[VAL_11]], %[[VAL_12]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_14:.*]] = torch.constant.bool false
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_16:.*]] = tosa.const_shape  {values = dense<[1, 1, 74, 75]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_17:.*]] = tosa.slice %[[VAL_1]], %[[VAL_15]], %[[VAL_16]] : (tensor<1x1x75x75xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x74x75xf32>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_19:.*]] = tosa.const_shape  {values = dense<[1, 1, 74, 74]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_20:.*]] = tosa.slice %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : (tensor<1x1x74x75xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x74x74xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.transpose %[[VAL_20]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x74x74xf32>) -> tensor<1x74x74x1xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.max_pool2d %[[VAL_21]] {kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 3, 3>} : (tensor<1x74x74x1xf32>) -> tensor<1x25x25x1xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.transpose %[[VAL_22]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x25x25x1xf32>) -> tensor<1x1x25x25xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.cast %[[VAL_23]] : tensor<1x1x25x25xf32> to tensor<1x1x25x25xf32>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<1x1x25x25xf32> -> !torch.vtensor<[1,1,25,25],f32>
// CHECK:           return %[[VAL_25]] : !torch.vtensor<[1,1,25,25],f32>
// CHECK:         }
func.func @torch.aten.max_pool2d$full_dim_indivisible_by_stride_with_sliced_input(%arg0: !torch.vtensor<[1,1,75,75],f32>) -> !torch.vtensor<[1,1,25,25],f32> {
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int3_1 = torch.constant.int 3
  %int3_2 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int3_1, %int3_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %int1_3 = torch.constant.int 1
  %2 = torch.prim.ListConstruct %int1, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1_4 = torch.constant.int 1
  %int1_5 = torch.constant.int 1
  %3 = torch.prim.ListConstruct %int1_4, %int1_5 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,1,75,75],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,25,25],f32>
  return %4 : !torch.vtensor<[1,1,25,25],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d$zero_pad_with_sliced_input(
// CHECK-SAME:                                                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,1,56,56],f32>) -> !torch.vtensor<[1,1,27,27],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,56,56],f32> -> tensor<1x1x56x56xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_12:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_13:.*]] = torch.constant.none
// CHECK-DAG:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 1, 55, 56]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_16:.*]] = tosa.slice %[[VAL_1]], %[[VAL_14]], %[[VAL_15]] : (tensor<1x1x56x56xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x55x56xf32>
// CHECK-DAG:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[1, 1, 55, 55]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : (tensor<1x1x55x56xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x55x55xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.transpose %[[VAL_19]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x55x55xf32>) -> tensor<1x55x55x1xf32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.avg_pool2d %[[VAL_20]], %[[VAL_21]], %[[VAL_22]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x55x55x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x27x27x1xf32>
// CHECK:           %[[VAL_24:.*]] = tosa.transpose %[[VAL_23]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x27x27x1xf32>) -> tensor<1x1x27x27xf32>
// CHECK:           %[[VAL_25:.*]] = tensor.cast %[[VAL_24]] : tensor<1x1x27x27xf32> to tensor<1x1x27x27xf32>
// CHECK:           %[[VAL_26:.*]] = torch_c.from_builtin_tensor %[[VAL_25]] : tensor<1x1x27x27xf32> -> !torch.vtensor<[1,1,27,27],f32>
// CHECK:           return %[[VAL_26]] : !torch.vtensor<[1,1,27,27],f32>
// CHECK:         }
func.func @torch.aten.avg_pool2d$zero_pad_with_sliced_input(%arg0: !torch.vtensor<[1,1,56,56],f32>) -> !torch.vtensor<[1,1,27,27],f32> {
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int2 = torch.constant.int 2
  %int2_1 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %int0 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %2 = torch.prim.ListConstruct %int0, %int0_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[1,1,56,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,27,27],f32>
  return %3 : !torch.vtensor<[1,1,27,27],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d$full_dim_indivisible_by_stride_without_sliced_input(
// CHECK-SAME:                                                                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,1,112,112],f32>) -> !torch.vtensor<[1,1,56,56],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,112,112],f32> -> tensor<1x1x112x112xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_12:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_13:.*]] = torch.constant.none
// CHECK:           %[[VAL_14:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x112x112xf32>) -> tensor<1x112x112x1xf32>
// CHECK:           %[[VAL_15:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_16:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.avg_pool2d %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<1x112x112x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x56x56x1xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.transpose %[[VAL_17]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x56x56x1xf32>) -> tensor<1x1x56x56xf32>
// CHECK:           %[[VAL_19:.*]] = tensor.cast %[[VAL_18]] : tensor<1x1x56x56xf32> to tensor<1x1x56x56xf32>
// CHECK:           %[[VAL_20:.*]] = torch_c.from_builtin_tensor %[[VAL_19]] : tensor<1x1x56x56xf32> -> !torch.vtensor<[1,1,56,56],f32>
// CHECK:           return %[[VAL_20]] : !torch.vtensor<[1,1,56,56],f32>
// CHECK:         }
func.func @torch.aten.avg_pool2d$full_dim_indivisible_by_stride_without_sliced_input(%arg0: !torch.vtensor<[1,1,112,112],f32>) -> !torch.vtensor<[1,1,56,56],f32> {
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int2 = torch.constant.int 2
  %int2_1 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %int1_2 = torch.constant.int 1
  %2 = torch.prim.ListConstruct %int1, %int1_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %false_3 = torch.constant.bool false
  %none = torch.constant.none
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %false_3, %none : !torch.vtensor<[1,1,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,56,56],f32>
  return %3 : !torch.vtensor<[1,1,56,56],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d$full_dim_indivisible_by_stride_with_sliced_input(
// CHECK-SAME:                                                                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !torch.vtensor<[1,1,75,75],f32>) -> !torch.vtensor<[1,1,25,25],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,75,75],f32> -> tensor<1x1x75x75xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_8]], %[[VAL_9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_12:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_13:.*]] = torch.constant.none
// CHECK-DAG:           %[[VAL_14:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_15:.*]] = tosa.const_shape  {values = dense<[1, 1, 74, 75]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_16:.*]] = tosa.slice %[[VAL_1]], %[[VAL_14]], %[[VAL_15]] : (tensor<1x1x75x75xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x74x75xf32>
// CHECK-DAG:           %[[VAL_17:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:           %[[VAL_18:.*]] = tosa.const_shape  {values = dense<[1, 1, 74, 74]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : (tensor<1x1x74x75xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x74x74xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.transpose %[[VAL_19]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1x74x74xf32>) -> tensor<1x74x74x1xf32>
// CHECK:           %[[VAL_21:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_22:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_23:.*]] = tosa.avg_pool2d %[[VAL_20]], %[[VAL_21]], %[[VAL_22]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 3, 3>} : (tensor<1x74x74x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x25x25x1xf32>
// CHECK:           %[[VAL_24:.*]] = tosa.transpose %[[VAL_23]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x25x25x1xf32>) -> tensor<1x1x25x25xf32>
// CHECK:           %[[VAL_25:.*]] = tensor.cast %[[VAL_24]] : tensor<1x1x25x25xf32> to tensor<1x1x25x25xf32>
// CHECK:           %[[VAL_26:.*]] = torch_c.from_builtin_tensor %[[VAL_25]] : tensor<1x1x25x25xf32> -> !torch.vtensor<[1,1,25,25],f32>
// CHECK:           return %[[VAL_26]] : !torch.vtensor<[1,1,25,25],f32>
// CHECK:         }
func.func @torch.aten.avg_pool2d$full_dim_indivisible_by_stride_with_sliced_input(%arg0: !torch.vtensor<[1,1,75,75],f32>) -> !torch.vtensor<[1,1,25,25],f32> {
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int3_1 = torch.constant.int 3
  %int3_2 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int3_1, %int3_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %int1_3 = torch.constant.int 1
  %2 = torch.prim.ListConstruct %int1, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %false_4 = torch.constant.bool false
  %none = torch.constant.none
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %false_4, %none : !torch.vtensor<[1,1,75,75],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,25,25],f32>
  return %3 : !torch.vtensor<[1,1,25,25],f32>
}

// -----
