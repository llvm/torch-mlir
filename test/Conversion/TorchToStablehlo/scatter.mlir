// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @forward(
// CHECK-SAME:                      %[[ARG_0:.*]]: !torch.vtensor<[?,?],si64>, %[[ARG_1:.*]]: !torch.vtensor<[?,?],si64>, %[[ARG_2:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK-DAG:      %[[VAR_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK-DAG:      %[[VAR_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK-DAG:      %[[VAR_2:.*]] = torch_c.to_builtin_tensor %[[ARG_2]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:          %int0 = torch.constant.int 0
// CHECK:          %[[INDEX_0:.*]] = arith.constant 0 : index
// CHECK:          %[[DIM_0:.*]] = tensor.dim %[[VAR_1]], %[[INDEX_0]] : tensor<?x?xi64>
// CHECK:          %[[VAR_3:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:          %[[INDEX_1:.*]] = arith.constant 1 : index
// CHECK:          %[[DIM_1:.*]] = tensor.dim %1, %[[INDEX_1]] : tensor<?x?xi64>
// CHECK:          %[[VAR_4:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:          %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:          %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:          %[[FE_:.*]] = tensor.from_elements %[[CONSTANT_0]], %[[CONSTANT_0]] : tensor<2xi64>
// CHECK:          %[[FE_1:.*]] = tensor.from_elements %[[CONSTANT_1]], %[[CONSTANT_1]] : tensor<2xi64>
// CHECK:          %[[FE_2:.*]] = tensor.from_elements %[[VAR_3]], %[[VAR_4]] : tensor<2xi64>
// CHECK:          %[[VAR_5:.*]] = stablehlo.real_dynamic_slice %[[VAR_2]], %[[FE_]], %[[FE_2]], %[[FE_1]] : (tensor<?x?xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:          %[[FE_3:.*]] = tensor.from_elements %[[VAR_3]], %[[VAR_4]], %[[CONSTANT_1]] : tensor<3xi64>
// CHECK:          %[[VAR_6:.*]] = stablehlo.dynamic_reshape %1, %[[FE_3]] : (tensor<?x?xi64>, tensor<3xi64>) -> tensor<?x?x1xi64>
// CHECK:          %[[VAR_7:.*]] = stablehlo.dynamic_iota %[[FE_3]], dim = 1 : (tensor<3xi64>) -> tensor<?x?x1xi64>
// CHECK:          %[[VAR_8:.*]] = stablehlo.concatenate %[[VAR_6]], %[[VAR_7]], dim = 2 : (tensor<?x?x1xi64>, tensor<?x?x1xi64>) -> tensor<?x?x2xi64>
// CHECK:          %[[VAR_9:.*]] = "stablehlo.scatter"(%[[VAR_0]], %[[VAR_8]], %[[VAR_5]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK:          ^bb0(%arg3: tensor<i64>, %[[ARG_4:.*]]: tensor<i64>):
// CHECK:            stablehlo.return %[[ARG_4]] : tensor<i64>
// CHECK:          }) : (tensor<?x?xi64>, tensor<?x?x2xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
// CHECK:          %[[VAR_10:.*]] = torch_c.from_builtin_tensor %[[VAR_9]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:          return %[[VAR_10]] : !torch.vtensor<[?,?],si64>
func.func @forward(%arg0: !torch.vtensor<[?,?],si64>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[?,?],si64>, !torch.int, !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64> -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}
