// RUN: torch-mlir-opt -split-input-file -verify-diagnostics %s -torch-refine-public-return | FileCheck %s

// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
// CHECK:           return %[[ARG]] : !torch.vtensor<[2,3,?],f32>
func.func @basic(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
  %1 = torch.copy.to_tensor %arg0 : !torch.tensor<[2,3,?],f32>
  %2 = torch.tensor_static_info_cast %1 : !torch.tensor<[2,3,?],f32> to !torch.tensor
  return %2 : !torch.tensor
}

// CHECK-LABEL:   func.func @multiple_use_non_value_tensor(
// CHECK-SAME:                                        %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                        %[[ARG1:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[NON_VALUE_TENSOR:.*]] = torch.copy.to_tensor %[[ARG0]] : !torch.tensor
// CHECK:           torch.overwrite.tensor.contents %[[ARG1]] overwrites %[[NON_VALUE_TENSOR]] : !torch.vtensor, !torch.tensor
// CHECK:           %[[RESULT:.*]] = torch.copy.to_vtensor %[[NON_VALUE_TENSOR]] : !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func.func @multiple_use_non_value_tensor(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.tensor {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  torch.overwrite.tensor.contents %arg1 overwrites %0 : !torch.vtensor, !torch.tensor
  return %0 : !torch.tensor
}

// No conversion on private function.
// CHECK-LABEL:   func.func private @basic_private(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[COPIED:.*]] = torch.copy.to_tensor %[[ARG]] : !torch.tensor<[2,3,?],f32>
// CHECK:           %[[CASTED:.*]] = torch.tensor_static_info_cast %[[COPIED]] : !torch.tensor<[2,3,?],f32> to !torch.tensor
// CHECK:           return %[[CASTED]] : !torch.tensor
func.func private @basic_private(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
  %1 = torch.copy.to_tensor %arg0 : !torch.tensor<[2,3,?],f32>
  %2 = torch.tensor_static_info_cast %1 : !torch.tensor<[2,3,?],f32> to !torch.tensor
  return %2 : !torch.tensor
}

// -----

// Call to public function.
// expected-error @+1 {{unimplemented}}
func.func @called(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  return %arg0 : tensor<*xf32>
}

func.func private @caller(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = call @called(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Multiple returns.
// expected-error @+1 {{unimplemented}}
func.func @called(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %ctrue = arith.constant true
  cf.cond_br %ctrue, ^bb1, ^bb2
^bb1:
  return %arg0 : tensor<*xf32>
^bb2:
  return %arg0 : tensor<*xf32>
}
