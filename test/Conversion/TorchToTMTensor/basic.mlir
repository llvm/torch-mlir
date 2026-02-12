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
// CHECK: tm_tensor.attention
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
// CHECK: tm_tensor.attention
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

// Test that an invalid scale (not 1/sqrt(headDim)) is rejected
func.func @sdpa_scale_invalid(%query: !torch.vtensor<[1,4,8,64],f32>, %key: !torch.vtensor<[1,4,8,64],f32>, %value: !torch.vtensor<[1,4,8,64],f32>) -> !torch.vtensor<[1,4,8,64],f32> {
  %float0 = torch.constant.float 0.000000e+00
  // 0.5 is not 1/sqrt(64)=0.125
  %scale = torch.constant.float 5.000000e-01
  %false = torch.constant.bool false
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.aten.scaled_dot_product_attention'}}
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %scale, %false : !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[1,4,8,64],f32>
  return %0 : !torch.vtensor<[1,4,8,64],f32>
}

// -----

// Test that a scale just over 1e-6 relative error from 1/sqrt(headDim) is rejected.
// For headDim=64: expected = 0.125, we use 0.12500025 which is 2e-6 relative error.
func.func @sdpa_scale_just_outside_tolerance(%query: !torch.vtensor<[1,4,8,64],f32>, %key: !torch.vtensor<[1,4,8,64],f32>, %value: !torch.vtensor<[1,4,8,64],f32>) -> !torch.vtensor<[1,4,8,64],f32> {
  %float0 = torch.constant.float 0.000000e+00
  // 0.12500025 = 0.125 * (1 + 2e-6), which is 2e-6 relative error (> 1e-6 tolerance)
  %scale = torch.constant.float 0.12500025
  %false = torch.constant.bool false
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.aten.scaled_dot_product_attention'}}
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %scale, %false : !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.vtensor<[1,4,8,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[1,4,8,64],f32>
  return %0 : !torch.vtensor<[1,4,8,64],f32>
}

// -----

// Test that any scale with dynamic head dimension is rejected
// (we cannot verify scale matches 1/sqrt(headDim) without knowing headDim)
func.func @sdpa_scale_dynamic_head_dim(%query: !torch.vtensor<[1,4,8,?],f32>, %key: !torch.vtensor<[1,4,8,?],f32>, %value: !torch.vtensor<[1,4,8,?],f32>) -> !torch.vtensor<[1,4,8,?],f32> {
  %float0 = torch.constant.float 0.000000e+00
  %scale = torch.constant.float 1.250000e-01
  %false = torch.constant.bool false
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.aten.scaled_dot_product_attention'}}
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %float0, %false, %scale, %false : !torch.vtensor<[1,4,8,?],f32>, !torch.vtensor<[1,4,8,?],f32>, !torch.vtensor<[1,4,8,?],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[1,4,8,?],f32>
  return %0 : !torch.vtensor<[1,4,8,?],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @sdpa_bool_mask_key_seq_dynamic
// CHECK: %[[MASK_IN:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[1,1,?],i1> -> tensor<1x1x?xi1>
// CHECK: %[[KEY:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,?,128],f16> -> tensor<16x?x128xf16>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[KEY_SEQ:.*]] = tensor.dim %[[KEY]], %[[C1]] : tensor<16x?x128xf16>
// CHECK: %[[EMPTY_MASK:.*]] = tensor.empty(%[[KEY_SEQ]]) : tensor<16x1x?xi1>
// CHECK: %[[BCAST_MASK:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[MASK_IN]] : tensor<1x1x?xi1>) outs(%[[EMPTY_MASK]] : tensor<16x1x?xi1>)
// CHECK: tm_tensor.attention ins(%{{.*}}, %{{.*}}, %{{.*}}, %[[BCAST_MASK]] : tensor<16x1x128xf16>, tensor<16x?x128xf16>, tensor<16x?x128xf16>, tensor<16x1x?xi1>)
func.func @sdpa_bool_mask_key_seq_dynamic(%query: !torch.vtensor<[16,1,128],f16>, %key: !torch.vtensor<[16,?,128],f16>, %value: !torch.vtensor<[16,?,128],f16>, %mask: !torch.vtensor<[1,1,?],i1>) -> !torch.vtensor<[16,1,128],f16> {
  %float0 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %mask, %float0, %false, %none, %false : !torch.vtensor<[16,1,128],f16>, !torch.vtensor<[16,?,128],f16>, !torch.vtensor<[16,?,128],f16>, !torch.vtensor<[1,1,?],i1>, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[16,1,128],f16>
  return %0 : !torch.vtensor<[16,1,128],f16>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @sdpa_bool_mask_both_seq_dynamic
// CHECK: %[[MASK_IN:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[1,?,?],i1> -> tensor<1x?x?xi1>
// CHECK: %[[KEY:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,?,128],f16> -> tensor<16x?x128xf16>
// CHECK: %[[QUERY:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,?,128],f16> -> tensor<16x?x128xf16>
// CHECK: %[[C1_A:.*]] = arith.constant 1 : index
// CHECK: %[[QSEQ:.*]] = tensor.dim %[[QUERY]], %[[C1_A]] : tensor<16x?x128xf16>
// CHECK: %[[C1_B:.*]] = arith.constant 1 : index
// CHECK: %[[KSEQ:.*]] = tensor.dim %[[KEY]], %[[C1_B]] : tensor<16x?x128xf16>
// CHECK: %[[EMPTY_MASK:.*]] = tensor.empty(%[[QSEQ]], %[[KSEQ]]) : tensor<16x?x?xi1>
// CHECK: %[[BCAST_MASK:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[MASK_IN]] : tensor<1x?x?xi1>) outs(%[[EMPTY_MASK]] : tensor<16x?x?xi1>)
// CHECK: tm_tensor.attention ins(%{{.*}}, %{{.*}}, %{{.*}}, %[[BCAST_MASK]] : tensor<16x?x128xf16>, tensor<16x?x128xf16>, tensor<16x?x128xf16>, tensor<16x?x?xi1>)
func.func @sdpa_bool_mask_both_seq_dynamic(%query: !torch.vtensor<[16,?,128],f16>, %key: !torch.vtensor<[16,?,128],f16>, %value: !torch.vtensor<[16,?,128],f16>, %mask: !torch.vtensor<[1,?,?],i1>) -> !torch.vtensor<[16,?,128],f16> {
  %float0 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %mask, %float0, %false, %none, %false : !torch.vtensor<[16,?,128],f16>, !torch.vtensor<[16,?,128],f16>, !torch.vtensor<[16,?,128],f16>, !torch.vtensor<[1,?,?],i1>, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[16,?,128],f16>
  return %0 : !torch.vtensor<[16,?,128],f16>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// CHECK-LABEL: @sdpa_bool_mask_4d_static_ones
// CHECK: %[[MASK_IN:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[1,1,1,1],i1> -> tensor<1x1x1x1xi1>
// CHECK: %[[KEY:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,16,?,128],f16> -> tensor<1x16x?x128xf16>
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[KSEQ:.*]] = tensor.dim %[[KEY]], %[[C2]] : tensor<1x16x?x128xf16>
// CHECK: %[[EMPTY_MASK:.*]] = tensor.empty(%[[KSEQ]]) : tensor<1x16x1x?xi1>
// CHECK: %[[BCAST_MASK:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[MASK_IN]] : tensor<1x1x1x1xi1>) outs(%[[EMPTY_MASK]] : tensor<1x16x1x?xi1>)
// CHECK: %[[COLLAPSED_MASK:.*]] = tensor.collapse_shape %[[BCAST_MASK]] {{.*}} : tensor<1x16x1x?xi1> into tensor<16x1x?xi1>
// CHECK: tm_tensor.attention ins(%{{.*}}, %{{.*}}, %{{.*}}, %[[COLLAPSED_MASK]] : tensor<16x1x128xf16>, tensor<16x?x128xf16>, tensor<16x?x128xf16>, tensor<16x1x?xi1>)
func.func @sdpa_bool_mask_4d_static_ones(%query: !torch.vtensor<[1,16,1,128],f16>, %key: !torch.vtensor<[1,16,?,128],f16>, %value: !torch.vtensor<[1,16,?,128],f16>, %mask: !torch.vtensor<[1,1,1,1],i1>) -> !torch.vtensor<[1,16,1,128],f16> {
  %float0 = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.scaled_dot_product_attention %query, %key, %value, %mask, %float0, %false, %none, %false : !torch.vtensor<[1,16,1,128],f16>, !torch.vtensor<[1,16,?,128],f16>, !torch.vtensor<[1,16,?,128],f16>, !torch.vtensor<[1,1,1,1],i1>, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,16,1,128],f16>
  return %0 : !torch.vtensor<[1,16,1,128],f16>
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
