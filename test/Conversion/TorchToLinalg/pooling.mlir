// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward_max_pool2d
func.func @forward_max_pool2d(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %int6 = torch.constant.int 6
  %int7 = torch.constant.int 7
  %int8 = torch.constant.int 8
  %false = torch.constant.bool false
  // CHECK: %[[C1:.*]] = torch_c.to_i64 %int1
  // CHECK: %[[C2:.*]] = torch_c.to_i64 %int2
  // CHECK: %[[NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: %[[PADDED:.*]] = tensor.pad %{{.*}} low[0, 0, 5, 6] high[0, 0, 5, 6]
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[T1:.*]] = arith.index_cast %[[C1]] : i64 to index
  // CHECK: %[[T2:.*]] = arith.index_cast %[[C2]] : i64 to index
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[T1]], %[[T2]]) : tensor<?x?xf32>
  // CHECK: linalg.pooling_nchw_max {dilations = dense<[7, 8]> : vector<2xi64>, strides = dense<[3, 4]> : vector<2xi64>} ins(%[[PADDED]], %[[INIT]] : tensor<?x?x?x?xf32>, tensor<?x?xf32>) outs(%[[OUT]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %kernel_size = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int5, %int6 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int7, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %kernel_size, %stride, %padding, %dilation, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2 * 2 + d5 * 3, d3 * 2 + d6 * 3, d4 * 2 + d7 * 3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @forward_max_pool3d
func.func @forward_max_pool3d(%arg0: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?,?],f32> {
  %kernel_size1 = torch.constant.int 8
  %kernel_size2 = torch.constant.int 8
  %kernel_size3 = torch.constant.int 8

  %stride1 = torch.constant.int 2
  %stride2 = torch.constant.int 2
  %stride3 = torch.constant.int 2

  %padding1 = torch.constant.int 4
  %padding2 = torch.constant.int 4
  %padding3 = torch.constant.int 4

  %dilation1 = torch.constant.int 3
  %dilation2 = torch.constant.int 3
  %dilation3 = torch.constant.int 3

  %false = torch.constant.bool false
  %kernel_size = torch.prim.ListConstruct %kernel_size1, %kernel_size2, %kernel_size3 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %stride1, %stride2, %stride3 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %padding1, %padding2, %padding3 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %dilation1, %dilation2, %dilation3 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>

  %4 = torch.aten.max_pool3d %arg0, %kernel_size, %stride, %padding, %dilation, %false : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?,?],f32>

  // CHECK: %[[MIN_VALUE:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: %[[PADDED_INPUT_TENSOR:.*]] = tensor.pad %{{.*}} low[0, 0, 4, 4, 4] high[0, 0, 4, 4, 4] {
  // CHECK-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
  // CHECK-NEXT:  tensor.yield %[[MIN_VALUE:.*]] : f32
  // CHECK: } : tensor<?x?x?x?x?xf32> to tensor<?x?x?x?x?xf32>

  // CHECK: %[[OUTPUT_TENSOR:.*]] = linalg.fill ins(%[[MIN_VALUE:.*]] : f32) outs(%{{.*}} : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  // CHECK: %[[MAX_3D_POOL:.*]] = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[PADDED_INPUT_TENSOR:.*]], %{{.*}} : tensor<?x?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[OUTPUT_TENSOR:.*]] : tensor<?x?x?x?x?xf32>) {
  // CHECK-NEXT:  ^bb0(%[[CURRENT_VALUE:.*]]: f32, %[[KERNEL:.*]]: f32, %[[ACC_OUT:.*]]: f32):
  // CHECK-NEXT:    %[[MAXF:.*]] = arith.maximumf %[[CURRENT_VALUE:.*]], %[[ACC_OUT:.*]] : f32
  // CHECK-NEXT:    linalg.yield %[[MAXF:.*]] : f32
  // CHECK:  } -> tensor<?x?x?x?x?xf32>
  return %4 : !torch.vtensor<[?,?,?,?,?],f32>
}
