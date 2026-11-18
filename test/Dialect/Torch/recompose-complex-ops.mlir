// RUN: torch-mlir-opt -torch-recompose-complex-ops -split-input-file %s | FileCheck %s

// The last slice may end past the dim size, which the slice clamps.
// CHECK-LABEL:  func.func @split_tensor_symbolic_list_unpack(
// CHECK-SAME:      %[[SELF:[^:]+]]: !torch.vtensor<[?,4],f32>, %[[SPLIT:[^:]+]]: !torch.int)
// CHECK-DAG:      %[[C0:.*]] = torch.constant.int 0
// CHECK-DAG:      %[[C1:.*]] = torch.constant.int 1
// CHECK-DAG:      %[[C2:.*]] = torch.constant.int 2
// CHECK-DAG:      %[[C3:.*]] = torch.constant.int 3
// CHECK:          torch.runtime.assert
// CHECK:          %[[S0:.*]] = torch.aten.slice.Tensor %[[SELF]], %[[C0]], %[[C0]], %[[SPLIT]], %[[C1]]
// CHECK:          %[[END1:.*]] = torch.aten.mul.int %[[C2]], %[[SPLIT]]
// CHECK:          %[[S1:.*]] = torch.aten.slice.Tensor %[[SELF]], %[[C0]], %[[SPLIT]], %[[END1]], %[[C1]]
// CHECK:          %[[END2:.*]] = torch.aten.mul.int %[[C3]], %[[SPLIT]]
// CHECK:          %[[S2:.*]] = torch.aten.slice.Tensor %[[SELF]], %[[C0]], %[[END1]], %[[END2]], %[[C1]]
// CHECK:          return %[[S0]], %[[S1]], %[[S2]]
func.func @split_tensor_symbolic_list_unpack(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.int) -> (!torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>) {
  %int0 = torch.constant.int 0
  %0 = torch.aten.split.Tensor %arg0, %arg1, %int0 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.int -> !torch.list<vtensor>
  %1:3 = torch.prim.ListUnpack %0 : !torch.list<vtensor> -> !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>
  return %1#0, %1#1, %1#2 : !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>
}

// -----

// CHECK-LABEL:  func.func @split_tensor_symbolic_getitem(
// CHECK-SAME:      %[[SELF:[^:]+]]: !torch.vtensor<[?,4],f32>, %[[SPLIT:[^:]+]]: !torch.int)
// CHECK-DAG:      %[[C0:.*]] = torch.constant.int 0
// CHECK-DAG:      %[[C1:.*]] = torch.constant.int 1
// CHECK:          torch.runtime.assert
// CHECK:          %[[END:.*]] = torch.aten.add.int %[[SPLIT]], %[[SPLIT]]
// CHECK:          %[[SLICE:.*]] = torch.aten.slice.Tensor %[[SELF]], %[[C0]], %[[SPLIT]], %[[END]], %[[C1]]
// CHECK:          return %[[SLICE]]
func.func @split_tensor_symbolic_getitem(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.split.Tensor %arg0, %arg1, %int0 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.int -> !torch.list<vtensor>
  %1 = torch.aten.__getitem__.t %0, %int1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,4],f32>
  return %1 : !torch.vtensor<[?,4],f32>
}
