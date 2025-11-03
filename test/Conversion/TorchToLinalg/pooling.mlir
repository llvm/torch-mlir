// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// RUN: torch-mlir-opt <%s -convert-torch-to-linalg=supports-non-finites=false -split-input-file -verify-diagnostics | FileCheck %s -check-prefix=UNSUPPORTED-NON-FINITES

// CHECK-LABEL: func @forward_max_pool1d
func.func @forward_max_pool1d(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %false = torch.constant.bool false
  // CHECK: %[[NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // CHECK-NOT: arith.constant -3.40282347E+38 : f32
  // UNSUPPORTED-NON-FINITES: %[[NEUTRAL:.*]] = arith.constant -3.40282347E+38 : f32
  // UNSUPPORTED-NON-FINITES-NOT: arith.constant 0xFF800000 : f32
  // CHECK: %[[PADDED:.*]] = tensor.pad %{{.*}} low[0, 0, 3] high[0, 0, 3]
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1xf32>
  // CHECK: linalg.pooling_ncw_max {dilations = dense<4> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%[[PADDED]], %[[INIT]] : tensor<?x?x?xf32>, tensor<1xf32>) outs(%[[OUT]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %kernel_size = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool1d %arg0, %kernel_size, %stride, %padding, %dilation, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?],f32>
}

// -----

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
  // CHECK: %[[NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // UNSUPPORTED-NON-FINITES: %[[NEUTRAL:.*]] = arith.constant -3.40282347E+38 : f32
  // CHECK: %[[PADDED:.*]] = tensor.pad %{{.*}} low[0, 0, 5, 6] high[0, 0, 5, 6]
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
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

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 2, d3 floordiv 2)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @forward_max_unpool2d
func.func @forward_max_unpool2d(%arg0: !torch.vtensor<[2,2,2,4],f32>, %arg1: !torch.vtensor<[2,2,2,4],si64>) -> !torch.vtensor<[2,2,4,8],f32> {
  %int8 = torch.constant.int 8
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int4, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK: ins(
  // CHECK: outs(
  // CHECK: ^bb0(
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[CAST:.*]] = arith.index_cast %{{.*}} : i64 to index
  // CHECK:   %[[IDX2:.*]] = linalg.index 2 : index
  // CHECK:   %[[IDX3:.*]] = linalg.index 3 : index
  // CHECK:   %[[C8_2:.*]] = arith.constant 8 : index
  // CHECK:   %[[MUL:.*]] = arith.muli %[[IDX2]], %[[C8_2]] : index
  // CHECK:   %[[ADD:.*]] = arith.addi %[[MUL]], %[[IDX3]] : index
  // CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[CAST]], %[[ADD]] : index
  // CHECK:   %[[SEL:.*]] = arith.select %[[CMP]], %{{.*}}, %[[CST]] : f32
  // CHECK:   linalg.yield %[[SEL]] : f32
  %1 = torch.aten.max_unpool2d %arg0, %arg1, %0 : !torch.vtensor<[2,2,2,4],f32>, !torch.vtensor<[2,2,2,4],si64>, !torch.list<int> -> !torch.vtensor<[2,2,4,8],f32>
  return %1 : !torch.vtensor<[2,2,4,8],f32>
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

  // CHECK: %[[NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // UNSUPPORTED-NON-FINITES: %[[NEUTRAL:.*]] = arith.constant -3.40282347E+38 : f32
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

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @forward_avg_pool2d
func.func @forward_avg_pool2d(%arg0: !torch.vtensor<[1,3,64,56],f32>) -> !torch.vtensor<[1,3, 61,27],f32> {
  // CHECK: linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 2]> : vector<2xi64>} ins(%[[IN1:.*]], %[[KSIZE1:.*]] : tensor<1x3x64x58xf32>, tensor<4x5xf32>) outs(%[[OUT1:.*]] : tensor<1x3x61x27xf32>) -> tensor<1x3x61x27xf32>
  // CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[IN2:.*]] : tensor<1x3x61x27xf32>) outs(%[[OUT2:.*]] : tensor<1x3x61x27xf32>)
  // CHECK-NEXT:  ^bb0(%[[BIIN1:.*]]: f32, %[[BOUT1:.*]]: f32):
  // CHECK-NEXT:  %[[TMP1:.*]] = arith.divf %[[BIIN1:.*]], %[[CONST1:.*]] : f32
  // CHECK-NEXT:  linalg.yield %[[TMP1:.*]] : f32
  // CHECK-NEXT:  } -> tensor<1x3x61x27xf32>
  %none = torch.constant.none
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %0 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[1,3,64,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,61,27],f32>
  return %3 : !torch.vtensor<[1,3,61,27],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @forward_avg_pool2d_countincludepad_false
func.func @forward_avg_pool2d_countincludepad_false(%arg0: !torch.vtensor<[1,3,64,56],f32>) -> !torch.vtensor<[1,3, 61,27],f32> {
  // CHECK: linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 2]> : vector<2xi64>} ins(%[[IN1:.*]], %[[KSIZE1:.*]] : tensor<1x3x64x58xf32>, tensor<4x5xf32>) outs(%[[OUT1:.*]] : tensor<1x3x61x27xf32>) -> tensor<1x3x61x27xf32>
  // CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[IN2:.*]] : tensor<1x3x61x27xf32>) outs(%[[OUT2:.*]] : tensor<1x3x61x27xf32>)
  // CHECK-NEXT:  ^bb0(%[[BIIN1:.*]]: f32, %[[BOUT1:.*]]: f32):
  // CHECK-COUNT-1: arith.minsi
  // CHECK-COUNT-1: arith.divf
  // CHECK:  linalg.yield %[[TMP1:.*]] : f32
  // CHECK-NEXT:  } -> tensor<1x3x61x27xf32>
  %none = torch.constant.none
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %0 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %false, %none : !torch.vtensor<[1,3,64,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,61,27],f32>
  return %3 : !torch.vtensor<[1,3,61,27],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @forward_avg_pool3d
func.func @forward_avg_pool3d(%arg0: !torch.vtensor<[1,3,7,64,56],f32>) -> !torch.vtensor<[1,3,4,31,54],f32> {
  // CHECK: linalg.pooling_ndhwc_sum {dilations = dense<1> : vector<3xi64>, strides = dense<[1, 2, 1]> : vector<3xi64>} ins(%[[IN1:.*]], %[[KSIZE1:.*]] : tensor<1x7x66x58x3xf32>, tensor<4x5x5xf32>) outs(%[[OUT1:.*]] : tensor<1x4x31x54x3xf32>) -> tensor<1x4x31x54x3xf32>
  // CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[IN2:.*]] : tensor<1x3x4x31x54xf32>) outs(%[[OUT2:.*]] : tensor<1x3x4x31x54xf32>)
  // CHECK-NEXT:  ^bb0(%[[BIN1:.*]]: f32, %[[BOUT1:.*]]: f32):
  // CHECK-NEXT:  %[[TMP1:.*]] = arith.divf %[[BIN1:.*]], %[[CONST1:.*]] : f32
  // CHECK-NEXT:  linalg.yield %[[TMP1:.*]] : f32
  // CHECK-NEXT:  } -> tensor<1x3x4x31x54xf32>
  %none = torch.constant.none
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %0 = torch.prim.ListConstruct %int4, %int5, %int5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool3d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[1,3,7,64,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,4,31,54],f32>
  return %3 : !torch.vtensor<[1,3,4,31,54],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @forward_avg_pool3dd_countincludepad_false
func.func @forward_avg_pool3dd_countincludepad_false(%arg0: !torch.vtensor<[1,3,7,64,56],f32>) -> !torch.vtensor<[1,3,4,31,54],f32> {
  // CHECK: linalg.pooling_ndhwc_sum {dilations = dense<1> : vector<3xi64>, strides = dense<[1, 2, 1]> : vector<3xi64>} ins(%[[IN1:.*]], %[[KSIZE1:.*]] : tensor<1x7x66x58x3xf32>, tensor<4x5x5xf32>) outs(%[[OUT1:.*]] : tensor<1x4x31x54x3xf32>) -> tensor<1x4x31x54x3xf32>
  // CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[IN2:.*]] : tensor<1x3x4x31x54xf32>) outs(%[[OUT2:.*]] : tensor<1x3x4x31x54xf32>)
  // CHECK-NEXT:  ^bb0(%[[BIN1:.*]]: f32, %[[BOUT1:.*]]: f32):
  // CHECK-COUNT-3: arith.minsi
  // CHECK-COUNT-1: arith.divf
  // CHECK-NEXT:  linalg.yield %[[TMP1:.*]] : f32
  // CHECK-NEXT:  } -> tensor<1x3x4x31x54xf32>
  %none = torch.constant.none
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %0 = torch.prim.ListConstruct %int4, %int5, %int5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool3d %arg0, %0, %1, %2, %false, %false, %none : !torch.vtensor<[1,3,7,64,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,4,31,54],f32>
  return %3 : !torch.vtensor<[1,3,4,31,54],f32>
}

// -----

// CHECK-LABEL: func @forward_avg_pool1d
func.func @forward_avg_pool1d(%arg0: !torch.vtensor<[1,512,10],f32>) -> !torch.vtensor<[1,512,12],f32> {
  // CHECK: linalg.pooling_ncw_sum {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%[[IN1:.*]], %[[IN2:.*]] : tensor<1x512x12xf32>, tensor<1xf32>) outs(%[[OUT1:.*]] : tensor<1x512x12xf32>) -> tensor<1x512x12xf32>
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool1d %arg0, %0, %1, %2, %false, %true : !torch.vtensor<[1,512,10],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,512,12],f32>
  return %3 : !torch.vtensor<[1,512,12],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @forward_avg_pool1d_countincludepad_false
func.func @forward_avg_pool1d_countincludepad_false(%arg0: !torch.vtensor<[1,512,10],f32>) -> !torch.vtensor<[1,512,12],f32> {
  // CHECK: linalg.pooling_ncw_sum {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%[[IN1:.*]], %[[IN2:.*]] : tensor<1x512x12xf32>, tensor<1xf32>) outs(%[[OUT1:.*]] : tensor<1x512x12xf32>) -> tensor<1x512x12xf32>
  // CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[IN3:.*]] : tensor<1x512x12xf32>) outs(%[[OUT2:.*]] : tensor<1x512x12xf32>
  // CHECK-NEXT:  ^bb0(%[[BIN1:.*]]: f32, %[[BOUT1:.*]]: f32):
  // CHECK-COUNT-1: arith.minsi
  // CHECK-COUNT-1: arith.divf
  // CHECK-NEXT:  linalg.yield %[[TMP1:.*]] : f32
  // CHECK-NEXT:  } -> tensor<1x512x12xf32>
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool1d %arg0, %0, %1, %2, %false, %false : !torch.vtensor<[1,512,10],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,512,12],f32>
  return %3 : !torch.vtensor<[1,512,12],f32>
}

// CHECK-LABEL: func @forward_avgpool_2d_ceil
func.func @forward_avgpool_2d_ceil(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32> {
  // CHECK: %[[POOL_OUT:.*]] = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%[[PADDED_IN:.*]], %[[KERNEL_IN:.*]] : tensor<1x1x6x6xf32>, tensor<3x3xf32>) outs(%[[OUT1:.*]] : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  // CHECK: linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[POOL_OUT]] : tensor<1x1x2x2xf32>) outs(%[[GEN_OUT:.*]] : tensor<1x1x2x2xf32>) {
  // CHECK-NEXT:  ^bb0(%[[BIN1:.*]]: f32, %[[BOUT1:.*]]: f32):
  // CHECK-COUNT-3: arith.muli
  // CHECK-COUNT-1: arith.sitofp
  // CHECK-COUNT-1: arith.divf
  // CHECK-NEXT:  linalg.yield %[[TMP1:.*]] : f32
  // CHECK-NEXT:  } -> tensor<1x1x2x2xf32>
  %int3 = torch.constant.int 3
  %int3_0 = torch.constant.int 3
  %int0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int2_2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int1_3 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int3, %int3_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int0, %int0_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int2_2, %int1, %int1_3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %none = torch.constant.none
  %3 = torch.aten.avg_pool2d %arg0, %0, %2, %1, %true, %false, %none : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,2,2],f32>
  return %3 : !torch.vtensor<[1,1,2,2],f32>
}
