// RUN: npcomp-opt -torch-adjust-calling-conventions -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[NONVAL_TENSOR:.*]] = torch.copy.to_tensor %[[ERASED]] : !torch.tensor
// CHECK:           return %[[NONVAL_TENSOR]] : !torch.tensor
func @basic(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func @no_type_bound(
// CHECK-SAME:                        %[[ARG:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[ARG]] : !torch.tensor
func @no_type_bound(%arg0: !torch.tensor) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func @call(
// CHECK-SAME:               %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ARG_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[ARG_NONVAL:.*]] = torch.copy.to_tensor %[[ARG_ERASED]] : !torch.tensor
// CHECK:           %[[INFO_ADDED:.*]] = torch.tensor_static_info_cast %[[ARG_NONVAL]] : !torch.tensor to !torch.tensor<[2,3,?],f32>
// CHECK:           %[[CALL_ARG:.*]] = torch.copy.to_vtensor %[[INFO_ADDED]] : !torch.vtensor<[2,3,?],f32>
// CHECK:           %[[CALL_RES:.*]] = call @call(%[[CALL_ARG]]) : (!torch.vtensor<[2,3,?],f32>) -> !torch.tensor
// CHECK:           return %[[ARG_NONVAL]] : !torch.tensor
func @call(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  %0 = call @call(%arg0) : (!torch.tensor) -> !torch.tensor
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func @none_return() {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           return
func @none_return() -> !torch.none {
  %1 = torch.constant.none
  return %1 : !torch.none
}

// CHECK-LABEL:   func @none_call_return() {
// CHECK:           call @none_return() : () -> ()
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           "test.use"(%[[NONE]]) : (!torch.none) -> ()
// CHECK:           return
func @none_call_return() {
  %0 = call @none_return() : () -> !torch.none
  "test.use"(%0) : (!torch.none) -> ()
  return
}
