// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @forward(
// CHECK-SAME:                      %[[ARG_0:.*]]: !torch.vtensor<[?,?],si64>, %[[ARG_1:.*]]: !torch.vtensor<[?,?],si64>, %[[ARG_2:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
// CHECK:          %[[VAR_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:          %[[VAR_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:          %[[VAR_2:.*]] = torch_c.to_builtin_tensor %[[ARG_2]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
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
// CHECK:          %[[VAR_9:.*]] = "stablehlo.scatter"(%[[VAR_0]], %[[VAR_8]], %[[VAR_5]]) ({
// CHECK:          ^bb0(%arg3: tensor<i64>, %[[ARG_4:.*]]: tensor<i64>):
// CHECK:            stablehlo.return %[[ARG_4]] : tensor<i64>
// CHECK:          }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false} : (tensor<?x?xi64>, tensor<?x?x2xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
// CHECK:          %[[VAR_10:.*]] = torch_c.from_builtin_tensor %[[VAR_9]] : tensor<?x?xi64> -> !torch.vtensor<[?,?],si64>
// CHECK:          return %[[VAR_10]] : !torch.vtensor<[?,?],si64>
func.func @forward(%arg0: !torch.vtensor<[?,?],si64>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[?,?],si64>, !torch.int, !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64> -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.index_put.hacked_twin(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[1,4],si64>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: !torch.vtensor<[1,1],si64>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[3],si64>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[1,3],si64>) -> !torch.vtensor<[1,4],si64> {
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,4],si64> -> tensor<1x4xi64>
// CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_3]] : !torch.vtensor<[1,3],si64> -> tensor<1x3xi64>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_1]], %[[VAL_2]] : (!torch.vtensor<[1,1],si64>, !torch.vtensor<[3],si64>) -> !torch.list<vtensor>
// CHECK:           %[[VAL_8:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,1],si64> -> tensor<1x1xi64>
// CHECK:           %[[VAL_9:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[3],si64> -> tensor<3xi64>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %[[VAL_8]] : (tensor<1x1xi64>) -> tensor<1x1x1xi64>
// CHECK:           %[[VAL_11:.*]] = stablehlo.broadcast_in_dim %[[VAL_10]], dims = [0, 1, 2] : (tensor<1x1x1xi64>) -> tensor<1x3x1xi64>
// CHECK:           %[[VAL_12:.*]] = stablehlo.reshape %[[VAL_9]] : (tensor<3xi64>) -> tensor<3x1xi64>
// CHECK:           %[[VAL_13:.*]] = stablehlo.broadcast_in_dim %[[VAL_12]], dims = [1, 2] : (tensor<3x1xi64>) -> tensor<1x3x1xi64>
// CHECK:           %[[VAL_14:.*]] = stablehlo.concatenate %[[VAL_11]], %[[VAL_13]], dim = 2 : (tensor<1x3x1xi64>, tensor<1x3x1xi64>) -> tensor<1x3x2xi64>
// CHECK:           %[[VAL_15:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1x3xi64>) -> tensor<3x1xi64>
// CHECK:           %[[VAL_16:.*]] = stablehlo.reshape %[[VAL_14]] : (tensor<1x3x2xi64>) -> tensor<3x2xi64>
// CHECK:           %[[VAL_17:.*]] = "stablehlo.scatter"(%[[VAL_4]], %[[VAL_16]], %[[VAL_15]]) ({
// CHECK:           ^bb0(%[[VAL_18:.*]]: tensor<i64>, %[[VAL_19:.*]]: tensor<i64>):
// CHECK:             stablehlo.return %[[VAL_19]] : tensor<i64>
// CHECK:           }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x4xi64>, tensor<3x2xi64>, tensor<3x1xi64>) -> tensor<1x4xi64>
// CHECK:           %[[VAL_20:.*]] = torch_c.from_builtin_tensor %[[VAL_17]] : tensor<1x4xi64> -> !torch.vtensor<[1,4],si64>
// CHECK:           return %[[VAL_20]] : !torch.vtensor<[1,4],si64>
// CHECK:         }
func.func @torch.aten.index_put.hacked_twin(%input: !torch.vtensor<[1,4],si64>, %index1: !torch.vtensor<[1,1],si64>, %index2: !torch.vtensor<[3],si64>, %fillValues: !torch.vtensor<[1,3],si64>) -> !torch.vtensor<[1,4],si64>{
  %false = torch.constant.bool false
  %indices = torch.prim.ListConstruct %index1, %index2 : (!torch.vtensor<[1,1],si64>, !torch.vtensor<[3],si64>) -> !torch.list<vtensor>
  %out = torch.aten.index_put.hacked_twin %input, %indices, %fillValues, %false: !torch.vtensor<[1,4],si64>, !torch.list<vtensor>, !torch.vtensor<[1,3],si64>, !torch.bool -> !torch.vtensor<[1,4],si64>
  return %out : !torch.vtensor<[1,4],si64>
}