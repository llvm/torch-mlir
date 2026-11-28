// RUN: torch-mlir-opt -torch-restructure-non-constant-axes -split-input-file %s | FileCheck %s

// A reduction over a runtime (non-constant) axis is rewritten to reduce a
// constant axis. With keepdim=false the result rank is restored through a
// statically shaped [result_sizes..., 1] intermediate.

// CHECK-LABEL: func.func @sum_reduce_keepdim_false
func.func @sum_reduce_keepdim_false(
    %arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>)
    -> !torch.vtensor<[3,2],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.sum.dim_IntList %arg0, %1, %false, %none
      : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none
      -> !torch.vtensor<[3,2],f32>
  return %2 : !torch.vtensor<[3,2],f32>

  // CHECK-NOT: torch.constant.bool false
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %{{.*}}, %[[TRUE]]
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2,1],f32>
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2],f32>
}

// -----

// The same rewriting works when the reduced runtime axis is the leading one.
// CHECK-LABEL: func.func @sum_reduce_keepdim_false_leading_axis
func.func @sum_reduce_keepdim_false_leading_axis(
    %arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>)
    -> !torch.vtensor<[2,2],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.sum.dim_IntList %arg0, %1, %false, %none
      : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none
      -> !torch.vtensor<[2,2],f32>
  return %2 : !torch.vtensor<[2,2],f32>

  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %{{.*}}, %[[TRUE]]
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[2,2,1],f32>
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[2,2],f32>
}

// -----

// The rewriting also handles reductions whose runtime axis is a scalar rather
// than a one-element list.
// CHECK-LABEL: func.func @all_reduce_keepdim_false_scalar_axis
func.func @all_reduce_keepdim_false_scalar_axis(
    %arg0: !torch.vtensor<[3,2,2],i1>, %arg1: !torch.vtensor<[1],si64>)
    -> !torch.vtensor<[3,2],i1> {
  %false = torch.constant.bool false
  %0 = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  %1 = torch.aten.all.dim %arg0, %0, %false
      : !torch.vtensor<[3,2,2],i1>, !torch.int, !torch.bool
      -> !torch.vtensor<[3,2],i1>
  return %1 : !torch.vtensor<[3,2],i1>

  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.all.dim %{{.*}}, %{{.*}}, %[[TRUE]]
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2,1],i1>
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2],i1>
}

// -----

// The rewriting is not limited to sum; it applies to the other supported
// reduction ops as well.
// CHECK-LABEL: func.func @frobenius_reduce_keepdim_false
func.func @frobenius_reduce_keepdim_false(
    %arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>)
    -> !torch.vtensor<[3,2],f32> {
  %false = torch.constant.bool false
  %0 = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.frobenius_norm.dim %arg0, %1, %false
      : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool
      -> !torch.vtensor<[3,2],f32>
  return %2 : !torch.vtensor<[3,2],f32>

  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.frobenius_norm.dim %{{.*}}, %{{.*}}, %[[TRUE]]
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2,1],f32>
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2],f32>
}

// -----

// With keepdim=true the reduction keeps its rank and is only rewritten to
// reduce a constant axis.
// CHECK-LABEL: func.func @sum_reduce_keepdim_true
func.func @sum_reduce_keepdim_true(
    %arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>)
    -> !torch.vtensor<[3,1,2],f32> {
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  %1 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.sum.dim_IntList %arg0, %1, %true, %none
      : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none
      -> !torch.vtensor<[3,1,2],f32>
  return %2 : !torch.vtensor<[3,1,2],f32>

  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %{{.*}}, %[[TRUE]]
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,1,2],f32>
}
