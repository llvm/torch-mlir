// RUN: torch-mlir-opt %s \
// RUN: --convert-torch-to-tmtensor --canonicalize --split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @test_topk_static_dims
// CHECK-SAME:      %[[t:.*]]: !torch.vtensor<[2,5],f32>)
// CHECK-DAG:     %[[positive_inf:.*]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:     %[[t_as_tensor:.*]] = torch_c.to_builtin_tensor %[[t]] : !torch.vtensor<[2,5],f32> -> tensor<2x5xf32>
// CHECK:         %[[empty_as_t:.*]] = tensor.empty() : tensor<2x3xf32>
// CHECK:         %[[pos_inf_like_t:.*]] = linalg.fill ins(%[[positive_inf]] : f32)
// CHECK-SAME:        outs(%[[empty_as_t]] : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[empty_indices:.*]] = tensor.empty() : tensor<2x3xi64>
// CHECK:         %[[topk_result:.*]]:2 = tm_tensor.topk dimension(1) ins(%[[t_as_tensor]] : tensor<2x5xf32>)
// CHECK-SAME:        outs(%[[pos_inf_like_t]], %[[empty_indices]] : tensor<2x3xf32>, tensor<2x3xi64>) {
// CHECK:         ^bb0(%[[lhs:.*]]: f32, %[[rhs:.*]]: f32):
// CHECK:           %[[cmpf_res:.*]] = arith.cmpf olt, %[[lhs]], %[[rhs]] : f32
// CHECK:           tm_tensor.yield %[[cmpf_res]] : i1
// CHECK:         } -> tensor<2x3xf32>, tensor<2x3xi64>
// CHECK-DAG:     %[[indices:.*]] = torch_c.from_builtin_tensor %[[topk_result]]#1 : tensor<2x3xi64> -> !torch.vtensor<[2,3],si64>
// CHECK-DAG:     %[[values:.*]] = torch_c.from_builtin_tensor %[[topk_result]]#0 : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:         return %[[values]], %[[indices]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],si64>
func.func @test_topk_static_dims(%t: !torch.vtensor<[2,5],f32>) ->
    (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],si64>) {
  %k = torch.constant.int 3
  %dim = torch.constant.int 1
  %largest = torch.constant.bool false
  %sorted = torch.constant.bool false
  %values, %indices = torch.aten.topk %t, %k, %dim, %largest, %sorted :
      !torch.vtensor<[2,5],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool ->
      !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],si64>
  return %values, %indices : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],si64>
}

// -----

// CHECK-LABEL: func.func @test_topk_with_dynamic_dim_and_k
// CHECK-SAME:      %[[t:.*]]: !torch.vtensor<[?,8,32],f32>,
// CHECK-SAME:      %[[k:.*]]: !torch.int
// CHECK-DAG:     %[[negative_inf:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK:         %[[t_as_tensor:.*]] = torch_c.to_builtin_tensor %[[t]] : !torch.vtensor<[?,8,32],f32> -> tensor<?x8x32xf32>
// CHECK:         %[[dim_0_size:.*]] = tensor.dim %[[t_as_tensor]], %[[c0]] : tensor<?x8x32xf32>
// CHECK:         %[[k_as_i64:.*]] = torch_c.to_i64 %[[k]]
// CHECK:         %[[k_as_index:.*]] = arith.index_cast %[[k_as_i64]] : i64 to index
// CHECK:         %[[empty_like_t:.*]] = tensor.empty(%[[dim_0_size]], %[[k_as_index]]) : tensor<?x8x?xf32>
// CHECK:         %[[neg_inf_like_t:.*]] = linalg.fill ins(%[[negative_inf]] : f32) outs(%[[empty_like_t]] : tensor<?x8x?xf32>) -> tensor<?x8x?xf32>
// CHECK:         %[[empty_indices:.]] = tensor.empty(%[[dim_0_size]], %[[k_as_index]]) : tensor<?x8x?xi64>
// CHECK:         %[[topk_result:.*]]:2 = tm_tensor.topk dimension(2) ins(%[[t_as_tensor]] : tensor<?x8x32xf32>) outs(%[[neg_inf_like_t]], %[[empty_indices]] : tensor<?x8x?xf32>, tensor<?x8x?xi64>) {
// CHECK:         ^bb0(%[[lhs:.*]]: f32, %[[rhs:.*]]: f32):
// CHECK:           %[[cmp_res:.*]] = arith.cmpf ogt, %[[lhs]], %[[rhs]] : f32
// CHECK:           tm_tensor.yield %[[cmp_res]] : i1
// CHECK:         } -> tensor<?x8x?xf32>, tensor<?x8x?xi64>
// CHECK-DAG:     %[[indices:.*]] = torch_c.from_builtin_tensor %[[topk_result]]#1 : tensor<?x8x?xi64> -> !torch.vtensor<[?,8,?],si64>
// CHECK-DAG:     %[[values:.*]] = torch_c.from_builtin_tensor %[[topk_result]]#0 : tensor<?x8x?xf32> -> !torch.vtensor<[?,8,?],f32>
// CHECK:         return %[[values]], %[[indices]] : !torch.vtensor<[?,8,?],f32>, !torch.vtensor<[?,8,?],si64>
func.func @test_topk_with_dynamic_dim_and_k(%t: !torch.vtensor<[?,8,32],f32>, %k: !torch.int) ->
    (!torch.vtensor<[?,8,?],f32>, !torch.vtensor<[?,8,?],si64>) {
  %dim = torch.constant.int -1
  %largest = torch.constant.bool true
  %sorted = torch.constant.bool false
  %values, %indices = torch.aten.topk %t, %k, %dim, %largest, %sorted :
      !torch.vtensor<[?,8,32],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool ->
      !torch.vtensor<[?,8,?],f32>, !torch.vtensor<[?,8,?],si64>
  return %values, %indices : !torch.vtensor<[?,8,?],f32>, !torch.vtensor<[?,8,?],si64>
}
