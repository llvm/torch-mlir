// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo=allow-non-finites=false -split-input-file -verify-diagnostics | FileCheck %s

// COM: the tests in this file locks down the behavior for allow-non-finites=false that replaces non-finites with the closest finite value for a given dtype.

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:       stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_2]])
func.func @torch.aten.max_pool2d(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$padding(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:       stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_2]])
func.func @torch.aten.max_pool2d$padding(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.max_pool2d %arg0, %0, %1, %2, %2, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$ceiloff(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,27,27],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,256,56,56],f32> -> tensor<1x256x56x56xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:       stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_2]])
func.func @torch.aten.max_pool2d$ceiloff(%arg0: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,27,27],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[1,256,56,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,256,27,27],f32>
  return %4 : !torch.vtensor<[1,256,27,27],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$ceilon(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,28,28],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,256,56,56],f32> -> tensor<1x256x56x56xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:       stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = "stablehlo.reduce_window"(%[[VAL_1]], %[[VAL_2]])
func.func @torch.aten.max_pool2d$ceilon(%arg0: !torch.vtensor<[1,256,56,56],f32>) -> !torch.vtensor<[1,256,28,28],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %true : !torch.vtensor<[1,256,56,56],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,256,28,28],f32>
  return %4 : !torch.vtensor<[1,256,28,28],f32>
}


// -----

// CHECK-LABEL:  func.func @torch.aten.max_pool2d_with_indices(
// CHECK          stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:     stablehlo.constant dense<0xFF800000> : tensor<f32>
func.func @torch.aten.max_pool2d_with_indices(%arg0: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>) {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
  return %result0, %result1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
}
