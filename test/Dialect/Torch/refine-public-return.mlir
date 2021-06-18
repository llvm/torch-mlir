// RUN: npcomp-opt -split-input-file -verify-diagnostics %s -torch-refine-public-return | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
// CHECK:           %[[COPIED_NONVAL:.*]] = torch.copy.to_tensor %[[ARG]] : !torch.tensor<[2,3,?],f32>
// CHECK:           %[[COPIED_VALUE:.*]] = torch.copy.to_vtensor %[[COPIED_NONVAL]] : !torch.vtensor<[2,3,?],f32>
// CHECK:           return %[[COPIED_VALUE]] : !torch.vtensor<[2,3,?],f32>
func @basic(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
  %1 = torch.copy.to_tensor %arg0 : !torch.tensor<[2,3,?],f32>
  %2 = torch.tensor_static_info_cast %1 : !torch.tensor<[2,3,?],f32> to !torch.tensor
  return %2 : !torch.tensor
}

// No conversion on private function.
// CHECK-LABEL:   func private @basic_private(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[COPIED:.*]] = torch.copy.to_tensor %[[ARG]] : !torch.tensor<[2,3,?],f32>
// CHECK:           %[[CASTED:.*]] = torch.tensor_static_info_cast %[[COPIED]] : !torch.tensor<[2,3,?],f32> to !torch.tensor
// CHECK:           return %[[CASTED]] : !torch.tensor
func private @basic_private(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
  %1 = torch.copy.to_tensor %arg0 : !torch.tensor<[2,3,?],f32>
  %2 = torch.tensor_static_info_cast %1 : !torch.tensor<[2,3,?],f32> to !torch.tensor
  return %2 : !torch.tensor
}



// -----

// Call to public function.
// expected-error @+1 {{unimplemented}}
func @called(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  return %arg0 : tensor<*xf32>
}

func private @caller(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = call @called(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Multiple returns.
// expected-error @+1 {{unimplemented}}
func @called(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %ctrue = constant true
  cond_br %ctrue, ^bb1, ^bb2
^bb1:
  return %arg0 : tensor<*xf32>
^bb2:
  return %arg0 : tensor<*xf32>
}
