// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -mlir-print-local-scope -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$simple_static(
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty() : tensor<3x4x2xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:        ins({{.*}} : tensor<4x2xf32>) outs({{.*}} : tensor<3x4x2xf32>) {
// CHECK:           ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK:             linalg.yield %[[IN]] : f32
// CHECK:           } -> tensor<3x4x2xf32>
func.func @torch.aten.broadcast_to$simple_static(%arg0: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[3,4,2],f32> {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %list = torch.prim.ListConstruct %int3, %int4, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.broadcast_to %arg0, %list : !torch.vtensor<[4,2],f32>, !torch.list<int> -> !torch.vtensor<[3,4,2],f32>
  return %0 : !torch.vtensor<[3,4,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$static_numpy_broadcast(
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty() : tensor<1x4x2xf32>
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]] : tensor<1x1x2xf32> into tensor<1x2xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:        ins(%[[COLLAPSE]] : tensor<1x2xf32>) outs({{.*}} : tensor<1x4x2xf32>) {
// CHECK:           ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK:             linalg.yield %[[IN]] : f32
// CHECK:           } -> tensor<1x4x2xf32>
func.func @torch.aten.broadcast_to$static_numpy_broadcast(%arg0: !torch.vtensor<[1,1,2],f32>) -> !torch.vtensor<[1,4,2],f32> {
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %list = torch.prim.ListConstruct %int1, %int4, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.broadcast_to %arg0, %list : !torch.vtensor<[1,1,2],f32>, !torch.list<int> -> !torch.vtensor<[1,4,2],f32>
  return %0 : !torch.vtensor<[1,4,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$empty_input(
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty({{.*}}) : tensor<?xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>]
// CHECK-SAME:        iterator_types = ["parallel"]}
// CHECK-SAME:        ins({{.*}} : tensor<f32>) outs({{.*}} : tensor<?xf32>) {
// CHECK:           ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK:             linalg.yield %[[IN]] : f32
// CHECK:           } -> tensor<?xf32>
func.func @torch.aten.broadcast_to$empty_input(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.int) -> !torch.vtensor<[?],f32> {
  %list = torch.prim.ListConstruct %arg1 : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.broadcast_to %arg0, %list : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$strict_dynamic_broadcast(
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty({{.*}}) : tensor<?x?xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]
// CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:        ins({{.*}} : tensor<?xf32>) outs({{.*}} : tensor<?x?xf32>) {
// CHECK:           ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK:             linalg.yield %[[IN]] : f32
// CHECK:           } -> tensor<?x?xf32>
func.func @torch.aten.broadcast_to$strict_dynamic_broadcast(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %list = torch.prim.ListConstruct %arg1, %arg2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.broadcast_to %arg0, %list : !torch.vtensor<[?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

/// Nothing we can do; verify we hit the fall back path.
// CHECK-LABEL:   func.func @torch.aten.broadcast_to$pure_dynamic_broadcast(
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty({{.*}}) : tensor<?x?xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>]
// CHECK-SAME:        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:        outs({{.*}} : tensor<?x?xf32>) {
// CHECK:           ^bb0(%[[OUT:.+]]: f32):
// CHECK:             tensor.extract
func.func @torch.aten.broadcast_to$pure_dynamic_broadcast(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %list = torch.prim.ListConstruct %arg1, %arg2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.broadcast_to %arg0, %list : !torch.vtensor<[?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
