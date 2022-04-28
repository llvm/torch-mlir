// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

func @torch.aten.mean.dim$invalid_dtype(%arg0: !torch.vtensor<[?,?,5],si32>) -> !torch.vtensor<[?,?,1],si32> {
  %neg_1 = torch.constant.int -1
  %true = torch.constant.bool true
  %none = torch.constant.none
  %dimList = torch.prim.ListConstruct %neg_1 : (!torch.int) -> !torch.list<int>
  // expected-error@+2 {{only float types are valid for mean operation}}
  // expected-error@+1 {{failed to legalize operation 'torch.aten.mean.dim' that was explicitly marked illegal}}
  %1 = torch.aten.mean.dim %arg0, %dimList, %true, %none : !torch.vtensor<[?,?,5],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?,?,1],si32>
  return %1 : !torch.vtensor<[?,?,1],si32>
}

// -----

func @torch.aten.mean.dim$invalid_dim(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %neg_1 = torch.constant.int -1
  %true = torch.constant.bool true
  %none = torch.constant.none
  %dimList = torch.prim.ListConstruct %neg_1 : (!torch.int) -> !torch.list<int>
  // expected-error@+2 {{insufficient type information to determine tensor size; check input tensor annotations}}
  // expected-error@+1 {{failed to legalize operation 'torch.aten.mean.dim' that was explicitly marked illegal}}
  %1 = torch.aten.mean.dim %arg0, %dimList, %true, %none : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  return %1 : !torch.vtensor<[?,?,?],f32>
}

// -----

func @torch.aten.mean.dim$valid(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,1],f32> {
  %neg_1 = torch.constant.int -1
  %true = torch.constant.bool true
  %none = torch.constant.none
  %dimList = torch.prim.ListConstruct %neg_1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.mean.dim %arg0, %dimList, %true, %none : !torch.vtensor<[3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f32>
  return %1 : !torch.vtensor<[3,4,1],f32>

  // CHECK: %[[sumOp:.*]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins({{.*}} : tensor<3x4x5xf32>) outs({{.*}} : tensor<?x?x?xf32>) {
  // CHECK: ^bb0(%[[arg1:.*]]: f32, %[[arg2:.*]]: f32):
  // CHECK:   %[[add:.*]] = arith.addf %[[arg1]], %[[arg2]] : f32
  // CHECK:   linalg.yield %[[add]] : f32
  // CHECK: } -> tensor<?x?x?xf32>

  // CHECK: %[[inverseSize:.*]] = arith.constant 2.000000e-01 : f32

  // CHECK: linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[sumOp]] : tensor<?x?x?xf32>) outs({{.*}} : tensor<?x?x?xf32>) {
  // CHECK: ^bb0(%[[arg1:.*]]: f32, %[[arg2:.*]]: f32):
  // CHECK:   %[[mul:.*]] = arith.mulf %[[arg1]], %[[inverseSize]] : f32
  // CHECK:   linalg.yield %[[mul]] : f32
  // CHECK: } -> tensor<?x?x?xf32>
}
