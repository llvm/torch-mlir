// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.index_select$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,4],f32>, %[[ARG1:.*]]: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
// CHECK-DAG:     %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,4],f32> -> tensor<?x4xf32>
// CHECK-DAG:     %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2],si64> -> tensor<2xi64>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x4xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "stablehlo.dynamic_gather"(%[[T0]], %[[T1]], %[[T4]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false}> : (tensor<?x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4xf32>
// CHECK:         %[[T6:.*]] = stablehlo.convert %[[T5]] : tensor<2x4xf32>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]] : tensor<2x4xf32> -> !torch.vtensor<[2,4],f32>
// CHECK:         return %[[T7]] : !torch.vtensor<[2,4],f32>
func.func @torch.aten.index_select$basic(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.index_select %arg0, %int0, %arg1 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL:  func.func @torch.aten.embedding$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?,?],f32> {
// CHECK-DAG:     %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:     %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?],si64> -> tensor<?xi64>
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "stablehlo.dynamic_gather"(%[[T0]], %[[T1]], %[[T4]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false}> : (tensor<?x?xf32>, tensor<?xi64>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:         %[[T6:.*]] = stablehlo.convert %[[T5]] : tensor<?x?xf32>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T7]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.embedding$basic(%weight: !torch.vtensor<[?,?],f32>, %indices: !torch.vtensor<[?], si64>) -> !torch.vtensor<[?,?],f32> {
  %false = torch.constant.bool false
  %int-1 = torch.constant.int -1
  %ret = torch.aten.embedding %weight, %indices, %int-1, %false, %false : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?], si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %ret: !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.embedding$rank_two_indices(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,1],si64>) -> !torch.vtensor<[?,1,?],f32> {
// CHECK-DAG:     %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:     %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,1],si64> -> tensor<?x1xi64>
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "stablehlo.dynamic_gather"(%[[T0]], %[[T1]], %[[T4]]) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false}> : (tensor<?x?xf32>, tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x1x?xf32>
// CHECK:         %[[T6:.*]] = stablehlo.convert %[[T5]] : tensor<?x1x?xf32>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]] : tensor<?x1x?xf32> -> !torch.vtensor<[?,1,?],f32>
// CHECK:         return %[[T7]] : !torch.vtensor<[?,1,?],f32>
func.func @torch.aten.embedding$rank_two_indices(%weight: !torch.vtensor<[?,?],f32>, %indices: !torch.vtensor<[?,1], si64>) -> !torch.vtensor<[?,1,?],f32> {
  %false = torch.constant.bool false
  %int-1 = torch.constant.int -1
  %ret = torch.aten.embedding %weight, %indices, %int-1, %false, %false : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,1], si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[?,1,?],f32>
  return %ret: !torch.vtensor<[?,1,?],f32>
}
