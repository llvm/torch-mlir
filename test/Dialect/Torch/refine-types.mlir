// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// This file tests the structural logic of the pass. This is for testing logic
// that does not scale with the number of ops supported, such as the core
// propagation logic, rewriting, etc.
// Code for testing transfer functions for new ops (which is most changes)
// should go in refine-types-ops.mlir.

// -----
// CHECK-LABEL:   func @basic(
// CHECK-SAME:                      %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor {
// CHECK:           %[[TANH:.*]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[TANH]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func @basic(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %1 = torch.aten.tanh %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func @keep_existing_shape_information(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor<[2],f32> {
// CHECK:           %[[TANH:.*]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<[2],f32>
// CHECK:           return %[[TANH]] : !torch.vtensor<[2],f32>
func @keep_existing_shape_information(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor<[2],f32> {
  %1 = torch.aten.tanh %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor<[2], f32>
  return %1 : !torch.vtensor<[2],f32>
}

// -----
// CHECK-LABEL:   func @propagate_through_multiple_ops(
// CHECK-SAME:                                         %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor {
// CHECK:           %[[TANH0:.*]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[TANH1:.*]] = torch.aten.tanh %[[TANH0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[TANH2:.*]] = torch.aten.tanh %[[TANH1]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[TANH3:.*]] = torch.tensor_static_info_cast %[[TANH2]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[TANH3]] : !torch.vtensor
func @propagate_through_multiple_ops(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %1 = torch.aten.tanh %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor
  %2 = torch.aten.tanh %1 : !torch.vtensor -> !torch.vtensor
  %3 = torch.aten.tanh %2 : !torch.vtensor -> !torch.vtensor
  return %3 : !torch.vtensor
}

// -----
// Check rewriting logic in case of mixes of users that do/don't allow type
// refinement.
// CHECK-LABEL:   func @mixed_allowing_not_allowing_type_refinement(
// CHECK-SAME:                                                      %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           %[[TANH0:.*]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[TANH0]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           %[[TANH1:.*]] = torch.aten.tanh %[[TANH0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           return %[[ERASED]], %[[ERASED]] : !torch.vtensor, !torch.vtensor
func @mixed_allowing_not_allowing_type_refinement(%arg0: !torch.vtensor<*,f32>) -> (!torch.vtensor, !torch.vtensor) {
  %1 = torch.aten.tanh %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor
  %3 = torch.aten.tanh %1 : !torch.vtensor -> !torch.vtensor
  return %1, %1 : !torch.vtensor, !torch.vtensor
}

// -----
// CHECK-LABEL:   func @type_promotion$same_category_different_width(
// CHECK-SAME:                                                       %[[ARG0:.*]]: !torch.vtensor<[?],si32>,
// CHECK-SAME:                                                       %[[ARG1:.*]]: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],unk> {
// CHECK:           %[[ALPHA:.*]] = torch.constant.int 3
// CHECK:           %[[ADD:.*]] = torch.aten.add.Tensor %[[ARG0]], %[[ARG1]], %[[ALPHA]] : !torch.vtensor<[?],si32>, !torch.vtensor<[?],si64>, !torch.int -> !torch.vtensor<[?],si64>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[ADD]] : !torch.vtensor<[?],si64> to !torch.vtensor<[?],unk>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],unk>
func @type_promotion$same_category_different_width(%arg0: !torch.vtensor<[?],si32>, %arg1: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],unk> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int3 : !torch.vtensor<[?],si32>, !torch.vtensor<[?],si64>, !torch.int -> !torch.vtensor<[?],unk>
  return %0 : !torch.vtensor<[?],unk>
}

// -----
// CHECK-LABEL:   func @type_promotion$different_category(
// CHECK-SAME:                                            %[[ARG0:.*]]: !torch.vtensor<[?],si64>,
// CHECK-SAME:                                            %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],unk> {
// CHECK:           %[[ALPHA:.*]] = torch.constant.int 3
// CHECK:           %[[ADD:.*]] = torch.aten.add.Tensor %[[ARG0]], %[[ARG1]], %[[ALPHA]] : !torch.vtensor<[?],si64>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[ADD]] : !torch.vtensor<[?],f32> to !torch.vtensor<[?],unk>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],unk>
func @type_promotion$different_category(%arg0: !torch.vtensor<[?],si64>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],unk> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int3 : !torch.vtensor<[?],si64>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],unk>
  return %0 : !torch.vtensor<[?],unk>
}

// -----
// CHECK-LABEL:   func @type_promotion$same_category_zero_rank_wider(
// CHECK-SAME:                                                       %[[ARG0:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                                                       %[[ARG1:.*]]: !torch.vtensor<[],f64>) -> !torch.vtensor<[?],unk> {
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 2.300000e+00
// CHECK:           %[[ADD:.*]] = torch.aten.add.Tensor %[[ARG0]], %[[ARG1]], %[[ALPHA]] : !torch.vtensor<[?],f32>, !torch.vtensor<[],f64>, !torch.float -> !torch.vtensor<[?],f32>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[ADD]] : !torch.vtensor<[?],f32> to !torch.vtensor<[?],unk>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],unk>
func @type_promotion$same_category_zero_rank_wider(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[],f64>) -> !torch.vtensor<[?],unk> {
  %float2.300000e00 = torch.constant.float 2.300000e+00
  %0 = torch.aten.add.Tensor %arg0, %arg1, %float2.300000e00 : !torch.vtensor<[?],f32>, !torch.vtensor<[],f64>, !torch.float -> !torch.vtensor<[?],unk>
  return %0 : !torch.vtensor<[?],unk>
}

// -----
// CHECK-LABEL:   func @type_promotion$zero_rank_higher_category(
// CHECK-SAME:                                                   %[[ARG0:.*]]: !torch.vtensor<[?],si64>,
// CHECK-SAME:                                                   %[[ARG1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[?],unk> {
// CHECK:           %[[ALPHA:.*]] = torch.constant.int 2
// CHECK:           %[[ADD:.*]] = torch.aten.add.Tensor %[[ARG0]], %[[ARG1]], %[[ALPHA]] : !torch.vtensor<[?],si64>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[ADD]] : !torch.vtensor<[?],f32> to !torch.vtensor<[?],unk>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],unk>
func @type_promotion$zero_rank_higher_category(%arg0: !torch.vtensor<[?],si64>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[?],unk> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int2 : !torch.vtensor<[?],si64>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[?],unk>
  return %0 : !torch.vtensor<[?],unk>
}

// -----
// CHECK-LABEL:   func @type_promotion$alpha_wider(
// CHECK-SAME:                                     %[[ARG0:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                                     %[[ARG1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[?],unk> {
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 2.300000e+00
// CHECK:           %[[ADD:.*]] = torch.aten.add.Tensor %[[ARG0]], %[[ARG1]], %[[ALPHA]] : !torch.vtensor<[?],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[?],f32>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[ADD]] : !torch.vtensor<[?],f32> to !torch.vtensor<[?],unk>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],unk>
func @type_promotion$alpha_wider(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[?],unk> {
  %float2.300000e00 = torch.constant.float 2.300000e+00
  %0 = torch.aten.add.Tensor %arg0, %arg1, %float2.300000e00 : !torch.vtensor<[?],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[?],unk>
  return %0 : !torch.vtensor<[?],unk>
}

// -----
// CHECK-LABEL:   func @type_promotion_scalar_operation(
// CHECK-SAME:                         %[[FLOAT:.*]]: !torch.float,
// CHECK-SAME:                         %[[INT:.*]]: !torch.int) -> !torch.number {
// CHECK:           %[[ADD:.*]] = torch.aten.add %[[FLOAT]], %[[INT]] : !torch.float, !torch.int -> !torch.float
// CHECK:           %[[RET:.*]] = torch.derefine %[[ADD]] : !torch.float to !torch.number
// CHECK:           return %[[RET]] : !torch.number
func @type_promotion_scalar_operation(%float: !torch.float, %int: !torch.int) -> !torch.number {
  %ret = torch.aten.add %float, %int : !torch.float, !torch.int -> !torch.number
  return %ret : !torch.number
}

// -----
// CHECK-LABEL:   func @torch.overwrite.tensor.contents$dynamic_overwrites_static(
// CHECK-SAME:                                                           %[[STATIC:.*]]: !torch.vtensor<[2],f32>,
// CHECK-SAME:                                                           %[[DYNAMIC:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[2],f32> {
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[DYNAMIC_COPY:.*]] : !torch.vtensor<[?],f32> to !torch.vtensor<*,f32>
// CHECK:           %[[CAST2:.*]] = torch.tensor_static_info_cast %[[CAST:.*]] : !torch.vtensor<*,f32> to !torch.vtensor<*,f32>
// CHECK:           torch.overwrite.tensor.contents %[[CAST2]] overwrites %[[STATIC_COPY:.*]] : !torch.vtensor<*,f32>, !torch.tensor<*,f32>
func @torch.overwrite.tensor.contents$dynamic_overwrites_static(%static: !torch.vtensor<[2],f32>, %dynamic: !torch.vtensor<[?],f32>) -> !torch.vtensor<[2],f32> {
  %static_no_type = torch.tensor_static_info_cast %static : !torch.vtensor<[2],f32> to !torch.vtensor
  %static_copy = torch.copy.to_tensor %static_no_type : !torch.tensor
  %dynamic_no_type = torch.tensor_static_info_cast %dynamic : !torch.vtensor<[?],f32> to !torch.vtensor
  torch.overwrite.tensor.contents %dynamic_no_type overwrites %static_copy : !torch.vtensor, !torch.tensor
  %static_value_copy = torch.copy.to_vtensor %static_copy : !torch.vtensor
  %result = torch.tensor_static_info_cast %static_value_copy : !torch.vtensor to !torch.vtensor<[2],f32>
  return %result : !torch.vtensor<[2],f32>
}

// -----
// CHECK-LABEL:   func @torch.overwrite.tensor.contents$static_overwrites_dynamic(
// CHECK-SAME:                                                                    %[[ARG0:.*]]: !torch.vtensor<[2],f32>,
// CHECK-SAME:                                                                    %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[ARG0_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[2],f32> to !torch.vtensor<*,f32>
// CHECK:           %[[ARG1_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG1]] : !torch.vtensor<[?],f32> to !torch.vtensor<*,f32>
// CHECK:           %[[MUTABLE_COPY:.*]] = torch.copy.to_tensor %[[ARG1_ERASED]] : !torch.tensor<*,f32>
// CHECK:           torch.overwrite.tensor.contents %[[ARG0_ERASED]] overwrites %[[MUTABLE_COPY]] : !torch.vtensor<*,f32>, !torch.tensor<*,f32>
func @torch.overwrite.tensor.contents$static_overwrites_dynamic(%static: !torch.vtensor<[2],f32>, %dynamic: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  %static_no_type = torch.tensor_static_info_cast %static : !torch.vtensor<[2],f32> to !torch.vtensor
  %dynamic_no_type = torch.tensor_static_info_cast %dynamic : !torch.vtensor<[?],f32> to !torch.vtensor
  %dynamic_copy = torch.copy.to_tensor %dynamic_no_type : !torch.tensor
  torch.overwrite.tensor.contents %static_no_type overwrites %dynamic_copy : !torch.vtensor, !torch.tensor
  %dynamic_value_copy = torch.copy.to_vtensor %dynamic_copy : !torch.vtensor
  %result = torch.tensor_static_info_cast %dynamic_value_copy : !torch.vtensor to !torch.vtensor<[?],f32>
  return %result : !torch.vtensor<[?],f32>
}

// -----
// CHECK-LABEL:   func @bf16_result_type(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.vtensor<*,bf16>) -> !torch.vtensor<[2],bf16> {
// CHECK:           %[[SQRT:.*]] = torch.aten.sqrt %[[ARG0]] : !torch.vtensor<*,bf16> -> !torch.vtensor<[2],bf16>
// CHECK:           return %[[SQRT]] : !torch.vtensor<[2],bf16>
func @bf16_result_type(%arg0: !torch.vtensor<*,bf16>) -> !torch.vtensor<[2],bf16> {
  %1 = torch.aten.sqrt %arg0 : !torch.vtensor<*,bf16> -> !torch.vtensor<[2], bf16>
  return %1 : !torch.vtensor<[2],bf16>
}

// -----
// CHECK-LABEL:   func @propagate_scalar_type(
// CHECK-SAME:                                %[[INT:.*]]: !torch.int) -> !torch.number {
// CHECK:           %[[NUM:.*]] = torch.derefine %[[INT]] : !torch.int to !torch.number
// CHECK:           %[[ABS:.*]] = torch.prim.abs.Scalar %[[INT]] : !torch.int -> !torch.int
// CHECK:           %[[RET:.*]] = torch.derefine %[[ABS]] : !torch.int to !torch.number
// CHECK:           return %[[RET]] : !torch.number
func @propagate_scalar_type(%arg0: !torch.int) -> !torch.number {
  %num = torch.derefine %arg0 : !torch.int to !torch.number
  %1 = torch.prim.abs.Scalar %num: !torch.number -> !torch.number
  return %1 : !torch.number
}
