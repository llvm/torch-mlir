// RUN: torch-mlir-opt <%s -convert-torch-to-tmtensor -split-input-file -verify-diagnostics | FileCheck %s

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
