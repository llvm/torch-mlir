// RUN: torch-mlir-opt < %s --torch-function-to-torch-backend-pipeline --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:     func.func @torch.aten.native_dropout.train(
// CHECK-SAME:                                      %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: f64) -> (tensor<?x?xf32>, tensor<?x?xi1>) {
// CHECK:             %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:             %[[CST_0:.*]] = arith.constant 1 : index
// CHECK:             %[[CST_1:.*]] = arith.constant 0 : index
// CHECK:             %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK:             %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:             %[[CST_2:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:             %[[CST_3:.*]] = arith.subf %[[CST_2]], %[[ARG1]] : f64
// CHECK:             %[[T3:.*]] = tensor.from_elements %[[CST_3]] : tensor<1xf64>
// CHECK:             %[[T4:.*]] = "mhlo.reshape"(%[[T3]]) : (tensor<1xf64>) -> tensor<f64>
// CHECK:             %[[T5:.*]] = mhlo.convert(%[[ARG0]]) : (tensor<?x?xf32>) -> tensor<?x?xf64>
// CHECK:             %[[DIM_0:.*]] = tensor.dim %[[T5]], %[[CST_1]] : tensor<?x?xf64>
// CHECK:             %[[CST_I64_0:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:             %[[DIM_1:.*]] = tensor.dim %[[T5]], %[[CST_0]] : tensor<?x?xf64>
// CHECK:             %[[CST_I64_1:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:             %[[T6:.*]] = tensor.from_elements %[[CST_I64_0]], %[[CST_I64_1]] : tensor<2xi64>
// CHECK:             %[[T7:.*]] = "mhlo.rng"(%[[T2]], %[[T1]], %[[T6]]) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f64>, tensor<f64>, tensor<2xi64>) -> tensor<?x?xf64>
// CHECK:             %[[T8:.*]] = shape.shape_of %[[T7]] : tensor<?x?xf64> -> tensor<2xindex>
// CHECK:             %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T4]], %[[T8]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>, tensor<2xindex>) -> tensor<?x?xf64>
// CHECK:             %[[T10:.*]] = mhlo.compare  LT, %[[T7]], %[[T9]],  FLOAT : (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xi1>
// CHECK:             %[[T11:.*]] = mhlo.convert(%[[T10]]) : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:             %[[T12:.*]] = shape.shape_of %[[T11]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:             %[[T13:.*]] = shape.shape_of %[[ARG0]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:             %[[T14:.*]] = shape.cstr_broadcastable %[[T12]], %[[T13]] : tensor<2xindex>, tensor<2xindex>
// CHECK:             %[[T15:.*]] = shape.assuming %[[T14]] -> (tensor<?x?xf32>) {
// CHECK:               %[[T16:.*]] = shape.broadcast %[[T12]], %[[T13]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
// CHECK:               %[[T17:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T11]], %[[T16]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:               %[[T18:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[T16]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:               %[[T19:.*]] = mhlo.multiply %[[T17]], %[[T18]] : tensor<?x?xf32>
// CHECK:               shape.assuming_yield %[[T19]] : tensor<?x?xf32>
// CHECK:             }
// CHECK:             %[[T20:.*]] = mhlo.convert(%[[T3]]) : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:             %[[T21:.*]] = "mhlo.reshape"(%[[T20]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:             %[[T22:.*]] = shape.shape_of %[[T15]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:             %[[T23:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T21]], %[[T22]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[T24:.*]] = mhlo.multiply %[[T15]], %[[T23]] : tensor<?x?xf32>
// CHECK:             %[[T25:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T0]], %[[T12]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[T26:.*]] = mhlo.compare  GE, %[[T11]], %[[T25]],  FLOAT : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:             return %[[T24]], %[[T26]] : tensor<?x?xf32>, tensor<?x?xi1>
func.func @torch.aten.native_dropout.train(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.float) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>) {
  %bool_true = torch.constant.bool true
  %result0, %result1 = torch.aten.native_dropout %arg0, %arg1, %bool_true: !torch.vtensor<[?,?],f32>, !torch.float, !torch.bool -> !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>
  return %result0, %result1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>
}