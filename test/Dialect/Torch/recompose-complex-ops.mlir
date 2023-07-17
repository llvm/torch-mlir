// RUN: torch-mlir-opt %s -torch-recompose-complex-ops | FileCheck %s

// CHECK-LABEL:   func.func @index_tensor_with_nonzero_numpy(
// CHECK-SAME:        %[[ARG_0:.*]]: !torch.vtensor<[2,3],f32>, 
// CHECK-SAME:        %[[ARG_1:.*]]: !torch.vtensor<[2,3],si64>) -> !torch.vtensor<*,f32> {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT_11:.*]] = torch.constant.int 11
// CHECK:           %[[VAL_0:.*]] = torch.aten.to.dtype %[[ARG_1]], %[[INT_11]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[2,3],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,3],i1>
// CHECK:           %[[VAL_1:.*]] = torch.aten.masked_select %[[ARG_0]], %[[VAL_0]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],i1> -> !torch.vtensor<*,f32>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<*,f32>
func.func @index_tensor_with_nonzero_numpy(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],si64>) -> !torch.vtensor<*,f32> {
  %0 = torch.aten.nonzero_numpy %arg1 : !torch.vtensor<[2,3],si64> -> !torch.list<tensor>
  %1 = torch.aten.index.Tensor %arg0, %0 : !torch.vtensor<[2,3],f32>, !torch.list<tensor> -> !torch.vtensor<*,f32>
  return %1 : !torch.vtensor<*,f32>
}