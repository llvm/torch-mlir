// RUN: torch-mlir-opt <%s -convert-torch-to-linalg=allow-non-finites=false -split-input-file -verify-diagnostics | FileCheck %s

// COM: the tests in this file locks down the behavior for allow-non-finites=false that replaces non-finites with the closest finite value for a given dtype.

// -----

// CHECK-LABEL: func @forward_max_pool1d
func.func @forward_max_pool1d(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %false = torch.constant.bool false
  // CHECK-NOT:  arith.constant 0xFF800000 : f32
  // CHECK: %[[NEUTRAL:.*]] = arith.constant -3.40282347E+38 : f32
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
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
  // CHECK-NOT: arith.constant 0xFF800000 : f32
  // CHECK: %[[NEUTRAL:.*]] = arith.constant -3.40282347E+38 : f32
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[NEUTRAL]] : f32) outs(%{{.*}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %kernel_size = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %stride = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int5, %int6 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int7, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %kernel_size, %stride, %padding, %dilation, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}


// -----

// CHECK-LABEL: func @forward_max_pool3d
// CHECK: arith.constant -3.40282347E+38 : f32
// CHECK-NOT: arith.constant 0xFF800000 : f32
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

  return %4 : !torch.vtensor<[?,?,?,?,?],f32>
}
