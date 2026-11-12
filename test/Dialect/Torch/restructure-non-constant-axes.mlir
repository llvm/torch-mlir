// RUN: torch-mlir-opt -torch-restructure-non-constant-axes -split-input-file %s | FileCheck %s

// keepdim=false with a runtime single-element axis must be rewritten:
// force a keepdim=1 sum on the fixed axis, then restore rank via a two-step
// view through a static [result..., 1] intermediate.
// CHECK-LABEL: func.func @reduce_keepdim_false_runtime_axis
func.func @reduce_keepdim_false_runtime_axis(
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

  // Reduction is rewritten to a keepdim=true sum on the fixed axis:
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %{{.*}}, %[[TRUE]]
  // Two-step view: first to full-rank [.,.,1], then down to [3,2]:
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2,1],f32>
  // CHECK: torch.aten.view %{{.*}} -> !torch.vtensor<[3,2],f32>
  // No residual runtime-dim keepdim=false sum survives:
  // CHECK-NOT: torch.aten.sum.dim_IntList %{{.*}}, %{{.*}}, %false
}
