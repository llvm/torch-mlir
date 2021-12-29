// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL:   func @tensor_tensor$same_category_different_width(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[1],f64>,
// CHECK-SAME:                                                      %[[VAL_2:.*]]: !torch.float) {
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Tensor %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] :
// CHECK-SAME:         !torch.vtensor<[1],f32>, !torch.vtensor<[1],f64>, !torch.float -> !torch.vtensor<[1],f64>
// CHECK:           return
builtin.func @tensor_tensor$same_category_different_width(%t0: !torch.vtensor<[1],f32>,
                                            %t1: !torch.vtensor<[1],f64>,
                                            %alpha: !torch.float) {
  %1 = torch.aten.add.Tensor %t0, %t1, %alpha: !torch.vtensor<[1],f32>, !torch.vtensor<[1],f64>, !torch.float -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func @tensor_tensor$different_category(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[1],si32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: !torch.vtensor<[1],f64>,
// CHECK-SAME:                                           %[[VAL_2:.*]]: !torch.float) {
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Tensor %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] :
// CHECK-SAME:         !torch.vtensor<[1],si32>, !torch.vtensor<[1],f64>, !torch.float -> !torch.vtensor<[1],f64>
// CHECK:           return
builtin.func @tensor_tensor$different_category(%t0: !torch.vtensor<[1],si32>,
                                 %t1: !torch.vtensor<[1],f64>,
                                 %alpha: !torch.float) {
  %1 = torch.aten.add.Tensor %t0, %t1, %alpha: !torch.vtensor<[1],si32>, !torch.vtensor<[1],f64>, !torch.float -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func @tensor_tensor$same_category_zero_rank_wider(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[],f64>,
// CHECK-SAME:                                                      %[[VAL_2:.*]]: !torch.int) {
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Tensor %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] :
// CHECK-SAME:         !torch.vtensor<[1],f32>, !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor<[1],f32>
// CHECK:           return
builtin.func @tensor_tensor$same_category_zero_rank_wider(
                                                  %t0: !torch.vtensor<[1],f32>,
                                                  %t1: !torch.vtensor<[],f64>,
                                                  %alpha: !torch.int) {
  %1 = torch.aten.add.Tensor %t0, %t1, %alpha: !torch.vtensor<[1],f32>, !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func @tensor_tensor$zero_rank_higher_category(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[1],si64>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: !torch.vtensor<[],f32>,
// CHECK-SAME:                                                  %[[VAL_2:.*]]: !torch.int) {
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Tensor %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] :
// CHECK-SAME:         !torch.vtensor<[1],si64>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[1],f32>
// CHECK:           return
builtin.func @tensor_tensor$zero_rank_higher_category(%t0: !torch.vtensor<[1],si64>,
                                        %t1: !torch.vtensor<[],f32>,
                                        %alpha: !torch.int) {
  %1 = torch.aten.add.Tensor %t0, %t1, %alpha: !torch.vtensor<[1],si64>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func @tensor_tensor$alpha_wider_no_contribution(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[1],f32>, %[[VAL_1:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: !torch.float) {
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Tensor %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] :
// CHECK-SAME:        !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.float -> !torch.vtensor<[1],f32>
// CHECK:           return
builtin.func @tensor_tensor$alpha_wider_no_contribution(%t0: !torch.vtensor<[1],f32>,
                               %t1: !torch.vtensor<[1],f32>,
                               %alpha: !torch.float) {
  %1 = torch.aten.add.Tensor %t0, %t1, %alpha: !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.float -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func @tensor_scalar$scalar_higher_category(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !torch.vtensor<[1],si64>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: !torch.float,
// CHECK-SAME:                                               %[[VAL_2:.*]]: !torch.int) {
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Scalar %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] :
// CHECK-SAME:        !torch.vtensor<[1],si64>, !torch.float, !torch.int -> !torch.vtensor<[1],f32>
// CHECK:           return
builtin.func @tensor_scalar$scalar_higher_category(%t0: !torch.vtensor<[1],si64>, %scalar: !torch.float, %alpha: !torch.int) {
  %1 = torch.aten.add.Scalar %t0, %scalar, %alpha: !torch.vtensor<[1], si64>, !torch.float, !torch.int -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func @tensor_scalar$scalar_same_category_wider(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !torch.vtensor<[1],si32>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !torch.int,
// CHECK-SAME:                                                   %[[VAL_2:.*]]: !torch.int) {
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Scalar %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] :
// CHECK-SAME:        !torch.vtensor<[1],si32>, !torch.int, !torch.int -> !torch.vtensor<[1],si32>
// CHECK:           return
builtin.func @tensor_scalar$scalar_same_category_wider(%t0: !torch.vtensor<[1],si32>, %scalar: !torch.int, %alpha: !torch.int) {
  %1 = torch.aten.add.Scalar %t0, %scalar, %alpha: !torch.vtensor<[1], si32>, !torch.int, !torch.int -> !torch.vtensor
  return
}
