// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: func.func @and_tensor_i1_scalar_broadcast
// CHECK: linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK: ^bb0(%[[LHS:.*]]: i1, %[[RHS:.*]]: i1, %{{.*}}: i1):
// CHECK: %[[AND:.*]] = arith.andi %[[LHS]], %[[RHS]] : i1
// CHECK: linalg.yield %[[AND]] : i1

func.func @and_tensor_i1_scalar_broadcast(
    %arg0: !torch.vtensor<[],i1>,
    %arg1: !torch.vtensor<[1,1,3,3],i1>)
    -> !torch.vtensor<[1,1,3,3],i1> {
  %0 = torch.aten.__and__.Tensor %arg0, %arg1
      : !torch.vtensor<[],i1>, !torch.vtensor<[1,1,3,3],i1>
      -> !torch.vtensor<[1,1,3,3],i1>
  return %0 : !torch.vtensor<[1,1,3,3],i1>
}
