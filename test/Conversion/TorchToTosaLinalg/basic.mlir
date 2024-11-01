// RUN: torch-mlir-opt <%s -convert-torch-to-tosa-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d.divisor_override
// CHECK: linalg.pooling_nchw_sum
// CHECK-NOT: torch.aten.avg_pool2d
func.func @torch.aten.avg_pool2d.divisor_override(%arg0: !torch.vtensor<[1,192,35,35],f32>) -> !torch.vtensor<[1,192,35,35],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %false= torch.constant.bool false
  %count_include_pad = torch.constant.bool false
  %divisor_override = torch.constant.int 9

  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %count_include_pad, %divisor_override : !torch.vtensor<[1,192,35,35],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.int -> !torch.vtensor<[1,192,35,35],f32>
  return %3 : !torch.vtensor<[1,192,35,35],f32>
}
