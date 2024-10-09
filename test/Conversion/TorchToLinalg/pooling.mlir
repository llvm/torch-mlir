// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward_max_pool1d
func.func @forward_max_pool1d(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %false = torch.constant.bool false
  // CHECK: %[[C1:.*]] = arith.constant 1 : i64
  // CHECK: %[[NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: %[[PADDED:.*]] = tensor.pad %{{.*}} low[0, 0, 3] high[0, 0, 3]
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK: %[[T1:.*]] = arith.constant 1 : index
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1xf32>
  // CHECK: linalg.pooling_ncw_max {dilations = dense<4> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%[[PADDED]], %[[INIT]] : tensor<?x?x?xf32>, tensor<1xf32>) outs(%[[OUT]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %kernel_size = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool1d %arg0, %kernel_size, %stride, %padding, %dilation, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?],f32>
}

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
  // CHECK: %[[C1:.*]] = arith.constant 1 : i64
  // CHECK: %[[C2:.*]] = arith.constant 2 : i64
  // CHECK: %[[NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: %[[PADDED:.*]] = tensor.pad %{{.*}} low[0, 0, 5, 6] high[0, 0, 5, 6]
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[T1:.*]] = arith.constant 1 : index
  // CHECK: %[[T2:.*]] = arith.constant 2 : index
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x2xf32>
  // CHECK: linalg.pooling_nchw_max {dilations = dense<[7, 8]> : vector<2xi64>, strides = dense<[3, 4]> : vector<2xi64>} ins(%[[PADDED]], %[[INIT]] : tensor<?x?x?x?xf32>, tensor<1x2xf32>) outs(%[[OUT]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
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
  // CHECK: %[[MAX_3D_POOL:.*]] = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[PADDED_INPUT_TENSOR:.*]], %{{.*}} : tensor<?x?x?x?x?xf32>, tensor<8x8x8xf32>) outs(%[[OUTPUT_TENSOR:.*]] : tensor<?x?x?x?x?xf32>) {
  // CHECK-NEXT:  ^bb0(%[[CURRENT_VALUE:.*]]: f32, %[[KERNEL:.*]]: f32, %[[ACC_OUT:.*]]: f32):
  // CHECK-NEXT:    %[[MAXF:.*]] = arith.maximumf %[[CURRENT_VALUE:.*]], %[[ACC_OUT:.*]] : f32
  // CHECK-NEXT:    linalg.yield %[[MAXF:.*]] : f32
  // CHECK:  } -> tensor<?x?x?x?x?xf32>
  return %4 : !torch.vtensor<[?,?,?,?,?],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 2, d3 floordiv 2)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @forward_max_unpool2d
func.func @forward_max_unpool2d(%arg0: !torch.vtensor<[1,1,2,2],f32>, %arg1: !torch.vtensor<[1,1,2,2],si64>) -> !torch.vtensor<[1,1,4,4],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %int1 = torch.constant.int 1
  %int1_0 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int4_1 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int1, %int1_0, %int4, %int4_1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int0 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %1 = torch.prim.ListConstruct %int0, %int0_2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int2 = torch.constant.int 2
  %int2_3 = torch.constant.int 2
  %2 = torch.prim.ListConstruct %int2, %int2_3 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.max_unpool2d %arg0, %arg1, %0, %2, %1 : !torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,1,2,2],si64>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,1,4,4],f32>

  // CHECK: %[[INDICES:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,2,2],si64> -> tensor<1x1x2x2xi64>
  // CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,1,2,2],f32> -> tensor<1x1x2x2xf32>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[INPUT]], %[[C0]] : tensor<1x1x2x2xf32>
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[INPUT]], %[[C1]] : tensor<1x1x2x2xf32>
  // CHECK: %[[SHAPE:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?x4x4xf32>
  // CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[INPUT]], %[[INDICES]] : tensor<1x1x2x2xf32>, tensor<1x1x2x2xi64>) outs(%[[SHAPE]] : tensor<?x?x4x4xf32>) {
  // CHECK-NEXT:  ^bb0(%[[CURRENT_VALUE:.*]]: f32, %[[CURRENT_INDEX:.*]]: i64, %[[OUT:.*]]: f32):
  // CHECK-NEXT:    %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT:    %[[INDEX_CAST:.*]] = arith.index_cast %[[CURRENT_INDEX:.*]] : i64 to index
  // CHECK-NEXT:    %[[INDEX2:.*]] = linalg.index 2 : index
  // CHECK-NEXT:    %[[INDEX3:.*]] = linalg.index 3 : index
  // CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : index
  // CHECK-NEXT:    %[[MULI:.*]] = arith.muli %[[INDEX2:.*]], %[[C4:.*]] : index
  // CHECK-NEXT:    %[[ADDI:.*]] = arith.addi %[[MULI:.*]], %[[INDEX3:.*]] : index
  // CHECK-NEXT:    %[[CMPI:.*]] = arith.cmpi eq, %[[INDEX_CAST:.*]], %[[ADDI:.*]] : index
  // CHECK-NEXT:    %[[SELECT:.*]] = arith.select %[[CMPI:.*]], %[[CURRENT_VALUE:.*]], %[[CST:.*]] : f32
  // CHECK-NEXT:    linalg.yield %[[SELECT:.*]] : f32
  // CHECK:  } -> tensor<?x?x4x4xf32>
  return %3 : !torch.vtensor<[1,1,4,4],f32>
}
