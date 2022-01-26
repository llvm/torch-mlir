// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward
builtin.func @forward(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %int6 = torch.constant.int 6
  %int7 = torch.constant.int 7
  %int8 = torch.constant.int 8
  %false = torch.constant.bool false
  // CHECK: %[[PADDED:.*]] = tensor.pad %{{.*}} low[0, 0, 5, 6] high[0, 0, 5, 6]
  // CHECK:  %[[NEUTRAL:.*]] = arith.constant -1.401300e-45 : f32
  // CHECK: %[[OUT:.*]] = linalg.fill(%[[NEUTRAL]], %{{.*}}) : f32, tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[C1]], %[[C2]]] : tensor<?x?xf32>
  // CHECK: linalg.pooling_nchw_max {dilations = dense<[7, 8]> : vector<2xi64>, strides = dense<[3, 4]> : vector<2xi64>} ins(%[[PADDED]], %[[INIT]] : tensor<?x?x?x?xf32>, tensor<?x?xf32>) outs(%[[OUT]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %kernel_size = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %stride = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %padding = torch.prim.ListConstruct %int5, %int6 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %dilation = torch.prim.ListConstruct %int7, %int8 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %4 = torch.aten.max_pool2d %arg0, %kernel_size, %stride, %padding, %dilation, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}
