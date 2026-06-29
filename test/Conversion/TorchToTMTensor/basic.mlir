// RUN: torch-mlir-opt <%s -convert-torch-to-tmtensor -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: @sdpa_scale_none
// CHECK: tm_tensor.attention
func.func @sdpa_scale_none(%query: !torch.vtensor<[1,4,8,64],f32>, %key: !torch.vtensor<[1,4,8,64],f32>, %value: !torch.vtensor<[1,4,8,64],f32>) -> !torch.vtensor<[1,4,8,64],f32> {
  %float0 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %none, %false : !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,4,8,64],f32>
  return %0 : !torch.vtensor<[1,4,8,64],f32>
}

// -----

// Test scale = 1/sqrt(64) = 0.125 which is the default PyTorch scale for headDim=64
// CHECK-LABEL: @sdpa_scale_rsqrt_head_dim
// CHECK: tm_tensor.attention {scale = {{(1\.250000e-01|0\.125)}} : f64}
func.func @sdpa_scale_rsqrt_head_dim(%query: !torch.vtensor<[1,4,8,64],f32>, %key: !torch.vtensor<[1,4,8,64],f32>, %value: !torch.vtensor<[1,4,8,64],f32>) -> !torch.vtensor<[1,4,8,64],f32> {
  %float0 = torch.constant.float 0.000000e+00
  // 1/sqrt(64) = 1/8 = 0.125
  %scale = torch.constant.float 1.250000e-01
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %scale, %false : !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[1,4,8,64],f32>
  return %0 : !torch.vtensor<[1,4,8,64],f32>
}

// -----

// Test scale = 1/sqrt(128) ≈ 0.0883883 for headDim=128
// CHECK-LABEL: @sdpa_scale_rsqrt_head_dim_128
// CHECK: tm_tensor.attention {scale = 0.0883883{{[0-9]*}} : f64}
func.func @sdpa_scale_rsqrt_head_dim_128(%query: !torch.vtensor<[1,4,8,128],f32>, %key: !torch.vtensor<[1,4,8,128],f32>, %value: !torch.vtensor<[1,4,8,128],f32>) -> !torch.vtensor<[1,4,8,128],f32> {
  %float0 = torch.constant.float 0.000000e+00
  // 1/sqrt(128) ≈ 0.0883883476483184
  %scale = torch.constant.float 0.0883883476483184
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %scale, %false : !torch.vtensor<[1,4,8,128],f32>, !torch.vtensor<[1,4,8,128],f32>, !torch.vtensor<[1,4,8,128],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[1,4,8,128],f32>
  return %0 : !torch.vtensor<[1,4,8,128],f32>
}

// -----

// Test GQA with different sequence lengths between query and key/value.
// Query has 32 heads with seq_len=10, key/value have 8 heads with seq_len=20.
// The repeated key/value should keep seq_len=20, not inherit seq_len=10 from query.
// CHECK-LABEL: @sdpa_gqa_different_seq_len
// CHECK: tm_tensor.attention
// CHECK-SAME: tensor<32x10x64xf32>, tensor<32x20x64xf32>, tensor<32x20x64xf32>
// CHECK-SAME: outs({{.*}} : tensor<32x10x64xf32>)
func.func @sdpa_gqa_different_seq_len(%query: !torch.vtensor<[1,32,10,64],f32>, %key: !torch.vtensor<[1,8,20,64],f32>, %value: !torch.vtensor<[1,8,20,64],f32>) -> !torch.vtensor<[1,32,10,64],f32> {
  %float0 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %none, %true : !torch.vtensor<[1,32,10,64],f32>, !torch.vtensor<[1,8,20,64],f32>, !torch.vtensor<[1,8,20,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,32,10,64],f32>
  return %0 : !torch.vtensor<[1,32,10,64],f32>
}

// -----

// Test GQA with independent key/value head counts (H_k != H_v).
// Q=8 heads, K=4 heads, V=2 heads. Key should be repeated 2x, value 4x.
// Verifies the repeat factors are applied to the correct operands.
// CHECK-LABEL: @sdpa_gqa_independent_kv_heads
// key: 4 heads repeated 2x -> broadcast to [1,4,2,20,64]
// CHECK: torch.aten.broadcast_to {{.*}} : !torch.vtensor<[1,4,1,20,64],f32>, !torch.list<int> -> !torch.vtensor<[1,4,2,20,64],f32>
// value: 2 heads repeated 4x -> broadcast to [1,2,4,20,64]
// CHECK: torch.aten.broadcast_to {{.*}} : !torch.vtensor<[1,2,1,20,64],f32>, !torch.list<int> -> !torch.vtensor<[1,2,4,20,64],f32>
// CHECK: tm_tensor.attention
// CHECK-SAME: tensor<8x10x64xf32>, tensor<8x20x64xf32>, tensor<8x20x64xf32>
func.func @sdpa_gqa_independent_kv_heads(%query: !torch.vtensor<[1,8,10,64],f32>, %key: !torch.vtensor<[1,4,20,64],f32>, %value: !torch.vtensor<[1,2,20,64],f32>) -> !torch.vtensor<[1,8,10,64],f32> {
  %float0 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %none, %true : !torch.vtensor<[1,8,10,64],f32>, !torch.vtensor<[1,4,20,64],f32>, !torch.vtensor<[1,2,20,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,8,10,64],f32>
  return %0 : !torch.vtensor<[1,8,10,64],f32>
}

// -----

// Test that a constant scale with dynamic head dimension is propagated.
// CHECK-LABEL: @sdpa_scale_dynamic_head_dim
// CHECK: tm_tensor.attention {scale = {{(1\.250000e-01|0\.125)}} : f64}
func.func @sdpa_scale_dynamic_head_dim(%query: !torch.vtensor<[1,4,8,?],f32>, %key: !torch.vtensor<[1,4,8,?],f32>, %value: !torch.vtensor<[1,4,8,?],f32>) -> !torch.vtensor<[1,4,8,?],f32> {
  %float0 = torch.constant.float 0.000000e+00
  %scale = torch.constant.float 1.250000e-01
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %scale, %false : !torch.vtensor<[1,4,8,?],f32>, !torch.vtensor<[1,4,8,?],f32>, !torch.vtensor<[1,4,8,?],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[1,4,8,?],f32>
  return %0 : !torch.vtensor<[1,4,8,?],f32>
}

// -----

// CHECK-LABEL: @scatter_src_i64_index
// CHECK: tm_tensor.scatter {dimension_map = array<i64: 0, 1, 2>} unique_indices(false) ins(%{{.*}}, %{{.*}} : tensor<?xf32>, tensor<?x3xi64>) outs(%{{.*}} : tensor<10x8x6xf32>) {
// CHECK:      ^bb0(%arg3: f32, %arg4: f32):
// CHECK:        tm_tensor.yield %arg3 : f32
// CHECK:      } -> tensor<10x8x6xf32>
func.func @scatter_src_i64_index(%arg0: !torch.vtensor<[10,8,6],f32>, %arg1: !torch.vtensor<[2,4,3],si64>, %arg2: !torch.vtensor<[5,8,6],f32>) -> !torch.vtensor<[10,8,6],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[10,8,6],f32>, !torch.int, !torch.vtensor<[2,4,3],si64>, !torch.vtensor<[5,8,6],f32> -> !torch.vtensor<[10,8,6],f32>
  return %0 : !torch.vtensor<[10,8,6],f32>
}


// -----

// CHECK-LABEL: @scatter_src_i32_index
// CHECK: tm_tensor.scatter {dimension_map = array<i64: 0, 1, 2>} unique_indices(false) ins(%{{.*}}, %{{.*}} : tensor<?xf32>, tensor<?x3xi32>) outs(%{{.*}} : tensor<10x8x6xf32>) {
// CHECK:      ^bb0(%arg3: f32, %arg4: f32):
// CHECK:        tm_tensor.yield %arg3 : f32
// CHECK:      } -> tensor<10x8x6xf32>
func.func @scatter_src_i32_index(%arg0: !torch.vtensor<[10,8,6],f32>, %arg1: !torch.vtensor<[2,4,3],si32>, %arg2: !torch.vtensor<[5,8,6],f32>) -> !torch.vtensor<[10,8,6],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[10,8,6],f32>, !torch.int, !torch.vtensor<[2,4,3],si32>, !torch.vtensor<[5,8,6],f32> -> !torch.vtensor<[10,8,6],f32>
  return %0 : !torch.vtensor<[10,8,6],f32>
}

// -----

// CHECK-LABEL: @scatter_src_dim2
// CHECK: tm_tensor.scatter {dimension_map = array<i64: 0, 1, 2, 3>} unique_indices(false) ins(%{{.*}}, %{{.*}} : tensor<?xf32>, tensor<?x4xi64>) outs(%{{.*}} : tensor<2x2x5x8xf32>) {
// CHECK:      ^bb0(%arg3: f32, %arg4: f32):
// CHECK:        tm_tensor.yield %arg3 : f32
// CHECK:      } -> tensor<2x2x5x8xf32>
func.func @scatter_src_dim2(%arg0: !torch.vtensor<[2,2,5,8],f32>, %arg1: !torch.vtensor<[2,2,1,8],si64>, %arg2: !torch.vtensor<[2,2,1,8],f32>) -> !torch.vtensor<[2,2,5,8],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.scatter.src %arg0, %int2, %arg1, %arg2 : !torch.vtensor<[2,2,5,8],f32>, !torch.int, !torch.vtensor<[2,2,1,8],si64>, !torch.vtensor<[2,2,1,8],f32> -> !torch.vtensor<[2,2,5,8],f32>
  return %0 : !torch.vtensor<[2,2,5,8],f32>
}

// -----

// CHECK-LABEL: @sort_float_ascending_nan_aware
// CHECK: tm_tensor.sort
// CHECK: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: i64, %{{.*}}: i64):
// CHECK-DAG: %[[RHS_NAN:.*]] = arith.cmpf une, %[[RHS]], %[[RHS]] : f32
// CHECK-DAG: %[[NUMERIC_LE:.*]] = arith.cmpf ole, %[[LHS]], %[[RHS]] : f32
// CHECK: %[[KEEP:.*]] = arith.ori %[[RHS_NAN]], %[[NUMERIC_LE]] : i1
// CHECK: tm_tensor.yield %[[KEEP]] : i1
func.func @sort_float_ascending_nan_aware(%arg0: !torch.vtensor<[6],f32>) -> (!torch.vtensor<[6],f32>, !torch.vtensor<[6],si64>) {
  %dim = torch.constant.int -1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.sort %arg0, %dim, %false : !torch.vtensor<[6],f32>, !torch.int, !torch.bool -> !torch.vtensor<[6],f32>, !torch.vtensor<[6],si64>
  return %values, %indices : !torch.vtensor<[6],f32>, !torch.vtensor<[6],si64>
}

// -----

// CHECK-LABEL: @sort_float_descending_nan_aware
// CHECK: tm_tensor.sort
// CHECK: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: i64, %{{.*}}: i64):
// CHECK-DAG: %[[LHS_NAN:.*]] = arith.cmpf une, %[[LHS]], %[[LHS]] : f32
// CHECK-DAG: %[[NUMERIC_GE:.*]] = arith.cmpf oge, %[[LHS]], %[[RHS]] : f32
// CHECK: %[[KEEP:.*]] = arith.ori %[[LHS_NAN]], %[[NUMERIC_GE]] : i1
// CHECK: tm_tensor.yield %[[KEEP]] : i1
func.func @sort_float_descending_nan_aware(%arg0: !torch.vtensor<[6],f32>) -> (!torch.vtensor<[6],f32>, !torch.vtensor<[6],si64>) {
  %dim = torch.constant.int -1
  %true = torch.constant.bool true
  %values, %indices = torch.aten.sort %arg0, %dim, %true : !torch.vtensor<[6],f32>, !torch.int, !torch.bool -> !torch.vtensor<[6],f32>, !torch.vtensor<[6],si64>
  return %values, %indices : !torch.vtensor<[6],f32>, !torch.vtensor<[6],si64>
}

// -----

// CHECK-LABEL: @kthvalue_float_nan_aware
// CHECK: tm_tensor.topk
// CHECK: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
// CHECK-DAG: %[[RHS_NAN:.*]] = arith.cmpf une, %[[RHS]], %[[RHS]] : f32
// CHECK-DAG: %[[LHS_NOT_NAN:.*]] = arith.cmpf oeq, %[[LHS]], %[[LHS]] : f32
// CHECK-DAG: %[[NUMERIC_LT:.*]] = arith.cmpf olt, %[[LHS]], %[[RHS]] : f32
// CHECK: %[[RHS_NAN_ONLY:.*]] = arith.andi %[[RHS_NAN]], %[[LHS_NOT_NAN]] : i1
// CHECK: %[[IS_BETTER:.*]] = arith.ori %[[RHS_NAN_ONLY]], %[[NUMERIC_LT]] : i1
// CHECK: tm_tensor.yield %[[IS_BETTER]] : i1
func.func @kthvalue_float_nan_aware(%arg0: !torch.vtensor<[6],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
  %k = torch.constant.int 2
  %dim = torch.constant.int 0
  %false = torch.constant.bool false
  %values, %indices = torch.aten.kthvalue %arg0, %k, %dim, %false : !torch.vtensor<[6],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
  return %values, %indices : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
}
