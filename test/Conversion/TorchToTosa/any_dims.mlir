// RUN: torch-mlir-opt %s -convert-torch-to-tosa -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @any_dims_none(
// CHECK: %[[REDUCE0:.*]] = tosa.reduce_any %{{.*}} axis(0) : (tensor<512x4xi1>) -> tensor<1x4xi1>
// CHECK: %[[REDUCE1:.*]] = tosa.reduce_any %[[REDUCE0]] axis(1) : (tensor<1x4xi1>) -> tensor<1x1xi1>
// CHECK: %[[RESULT:.*]] = tosa.reshape %[[REDUCE1]], %{{.*}} : (tensor<1x1xi1>, !tosa.shape<0>) -> tensor<i1>
func.func @any_dims_none(%arg0: !torch.vtensor<[512,4],i1>) -> !torch.vtensor<[],i1> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.any.dims %arg0, %none, %false : !torch.vtensor<[512,4],i1>, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: func.func @any_dims_explicit_keepdim(
// CHECK: %[[REDUCE0:.*]] = tosa.reduce_any %{{.*}} axis(0) : (tensor<2x3x4xi1>) -> tensor<1x3x4xi1>
// CHECK: %[[REDUCE2:.*]] = tosa.reduce_any %[[REDUCE0]] axis(2) : (tensor<1x3x4xi1>) -> tensor<1x3x1xi1>
func.func @any_dims_explicit_keepdim(%arg0: !torch.vtensor<[2,3,4],i1>) -> !torch.vtensor<[1,3,1],i1> {
  %int0 = torch.constant.int 0
  %int_neg1 = torch.constant.int -1
  %dims = torch.prim.ListConstruct %int0, %int_neg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %true = torch.constant.bool true
  %0 = torch.aten.any.dims %arg0, %dims, %true : !torch.vtensor<[2,3,4],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,1],i1>
  return %0 : !torch.vtensor<[1,3,1],i1>
}

// -----

// CHECK-LABEL: func.func @any_dims_empty(
// CHECK: %[[CAST:.*]] = tosa.cast %{{.*}} : (tensor<2x3xi64>) -> tensor<2x3xi1>
// CHECK-NOT: tosa.reduce_any
// CHECK: tosa.identity %[[CAST]] : (tensor<2x3xi1>) -> tensor<2x3xi1>
func.func @any_dims_empty(%arg0: !torch.vtensor<[2,3],si64>) -> !torch.vtensor<[2,3],i1> {
  %dims = torch.prim.ListConstruct : () -> !torch.list<int>
  %false = torch.constant.bool false
  %0 = torch.aten.any.dims %arg0, %dims, %false : !torch.vtensor<[2,3],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,3],i1>
  return %0 : !torch.vtensor<[2,3],i1>
}

// -----

// CHECK-LABEL: func.func @any_dims_scalar_dim(
// CHECK-NOT: torch.aten.any.dims
// CHECK-NOT: tosa.reduce_any
// CHECK: return %arg0 : !torch.vtensor<[],i1>
func.func @any_dims_scalar_dim(%arg0: !torch.vtensor<[],i1>) -> !torch.vtensor<[],i1> {
  %int0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %0 = torch.aten.any.dims %arg0, %dims, %false : !torch.vtensor<[],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: func.func @any_dims_scalar_negative_dim(
// CHECK-NOT: torch.aten.any.dims
// CHECK-NOT: tosa.reduce_any
// CHECK: return %arg0 : !torch.vtensor<[],i1>
func.func @any_dims_scalar_negative_dim(%arg0: !torch.vtensor<[],i1>) -> !torch.vtensor<[],i1> {
  %int_neg1 = torch.constant.int -1
  %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %0 = torch.aten.any.dims %arg0, %dims, %false : !torch.vtensor<[],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}
