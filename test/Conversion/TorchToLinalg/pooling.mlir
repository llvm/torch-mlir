// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward
func.func @forward(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
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
