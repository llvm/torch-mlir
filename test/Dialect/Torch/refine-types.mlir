// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// This file tests the structural logic of the pass. This is for testing logic
// that does not scale with the number of ops supported, such as the core
// propagation logic, rewriting, etc.
// Code for testing transfer functions for new ops (which is most changes)
// should go in refine-types-ops.mlir.

// -----
// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                      %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor {
// CHECK:           %[[COS:.*]] = torch.aten.cos %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[COS]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func.func @basic(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %1 = torch.aten.cos %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @keep_existing_shape_information(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor<[2],f32> {
// CHECK:           %[[COS:.*]] = torch.aten.cos %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<[2],f32>
// CHECK:           return %[[COS]] : !torch.vtensor<[2],f32>
func.func @keep_existing_shape_information(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor<[2],f32> {
  %1 = torch.aten.cos %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor<[2], f32>
  return %1 : !torch.vtensor<[2],f32>
}

// -----
// CHECK-LABEL:   func.func @propagate_through_multiple_ops(
// CHECK-SAME:                                         %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor {
// CHECK:           %[[COS0:.*]] = torch.aten.cos %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[COS1:.*]] = torch.aten.cos %[[COS0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[COS2:.*]] = torch.aten.cos %[[COS1]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[COS3:.*]] = torch.tensor_static_info_cast %[[COS2]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[COS3]] : !torch.vtensor
func.func @propagate_through_multiple_ops(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %1 = torch.aten.cos %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor
  %2 = torch.aten.cos %1 : !torch.vtensor -> !torch.vtensor
  %3 = torch.aten.cos %2 : !torch.vtensor -> !torch.vtensor
  return %3 : !torch.vtensor
}

// -----
// Check rewriting logic in case of mixes of users that do/don't allow type
// refinement.
// CHECK-LABEL:   func.func @mixed_allowing_not_allowing_type_refinement(
// CHECK-SAME:                                                      %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           %[[COS0:.*]] = torch.aten.cos %[[ARG0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[COS0]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           %[[COS1:.*]] = torch.aten.cos %[[COS0]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           return %[[ERASED]], %[[ERASED]] : !torch.vtensor, !torch.vtensor
func.func @mixed_allowing_not_allowing_type_refinement(%arg0: !torch.vtensor<*,f32>) -> (!torch.vtensor, !torch.vtensor) {
  %1 = torch.aten.cos %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor
  %3 = torch.aten.cos %1 : !torch.vtensor -> !torch.vtensor
  return %1, %1 : !torch.vtensor, !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func @torch.overwrite.tensor.contents$dynamic_overwrites_static(
// CHECK-SAME:                                                           %[[STATIC:.*]]: !torch.vtensor<[2],f32>,
// CHECK-SAME:                                                           %[[DYNAMIC:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[2],f32> {
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[DYNAMIC_COPY:.*]] : !torch.vtensor<[?],f32> to !torch.vtensor<*,f32>
// CHECK:           %[[CAST2:.*]] = torch.tensor_static_info_cast %[[CAST:.*]] : !torch.vtensor<*,f32> to !torch.vtensor<*,f32>
// CHECK:           torch.overwrite.tensor.contents %[[CAST2]] overwrites %[[STATIC_COPY:.*]] : !torch.vtensor<*,f32>, !torch.tensor<*,f32>
func.func @torch.overwrite.tensor.contents$dynamic_overwrites_static(%static: !torch.vtensor<[2],f32>, %dynamic: !torch.vtensor<[?],f32>) -> !torch.vtensor<[2],f32> {
  %static_no_type = torch.tensor_static_info_cast %static : !torch.vtensor<[2],f32> to !torch.vtensor
  %static_copy = torch.copy.to_tensor %static_no_type : !torch.tensor
  %dynamic_no_type = torch.tensor_static_info_cast %dynamic : !torch.vtensor<[?],f32> to !torch.vtensor
  torch.overwrite.tensor.contents %dynamic_no_type overwrites %static_copy : !torch.vtensor, !torch.tensor
  %static_value_copy = torch.copy.to_vtensor %static_copy : !torch.vtensor
  %result = torch.tensor_static_info_cast %static_value_copy : !torch.vtensor to !torch.vtensor<[2],f32>
  return %result : !torch.vtensor<[2],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.overwrite.tensor.contents$static_overwrites_dynamic(
// CHECK-SAME:                                                                    %[[ARG0:.*]]: !torch.vtensor<[2],f32>,
// CHECK-SAME:                                                                    %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[ARG0_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[2],f32> to !torch.vtensor<*,f32>
// CHECK:           %[[ARG1_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG1]] : !torch.vtensor<[?],f32> to !torch.vtensor<*,f32>
// CHECK:           %[[MUTABLE_COPY:.*]] = torch.copy.to_tensor %[[ARG1_ERASED]] : !torch.tensor<*,f32>
// CHECK:           torch.overwrite.tensor.contents %[[ARG0_ERASED]] overwrites %[[MUTABLE_COPY]] : !torch.vtensor<*,f32>, !torch.tensor<*,f32>
func.func @torch.overwrite.tensor.contents$static_overwrites_dynamic(%static: !torch.vtensor<[2],f32>, %dynamic: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  %static_no_type = torch.tensor_static_info_cast %static : !torch.vtensor<[2],f32> to !torch.vtensor
  %dynamic_no_type = torch.tensor_static_info_cast %dynamic : !torch.vtensor<[?],f32> to !torch.vtensor
  %dynamic_copy = torch.copy.to_tensor %dynamic_no_type : !torch.tensor
  torch.overwrite.tensor.contents %static_no_type overwrites %dynamic_copy : !torch.vtensor, !torch.tensor
  %dynamic_value_copy = torch.copy.to_vtensor %dynamic_copy : !torch.vtensor
  %result = torch.tensor_static_info_cast %dynamic_value_copy : !torch.vtensor to !torch.vtensor<[?],f32>
  return %result : !torch.vtensor<[?],f32>
}

// -----
// CHECK-LABEL:   func.func @bf16_result_type(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.vtensor<*,bf16>) -> !torch.vtensor<[2],bf16> {
// CHECK:           %[[SQRT:.*]] = torch.aten.sqrt %[[ARG0]] : !torch.vtensor<*,bf16> -> !torch.vtensor<[2],bf16>
// CHECK:           return %[[SQRT]] : !torch.vtensor<[2],bf16>
func.func @bf16_result_type(%arg0: !torch.vtensor<*,bf16>) -> !torch.vtensor<[2],bf16> {
  %1 = torch.aten.sqrt %arg0 : !torch.vtensor<*,bf16> -> !torch.vtensor<[2], bf16>
  return %1 : !torch.vtensor<[2],bf16>
}

// -----
// CHECK-LABEL:   func.func @propagate_scalar_type(
// CHECK-SAME:                                %[[INT:.*]]: !torch.int) -> !torch.number {
// CHECK:           %[[NUM:.*]] = torch.derefine %[[INT]] : !torch.int to !torch.number
// CHECK:           %[[ABS:.*]] = torch.prim.abs.Scalar %[[INT]] : !torch.int -> !torch.int
// CHECK:           %[[RET:.*]] = torch.derefine %[[ABS]] : !torch.int to !torch.number
// CHECK:           return %[[RET]] : !torch.number
func.func @propagate_scalar_type(%arg0: !torch.int) -> !torch.number {
  %num = torch.derefine %arg0 : !torch.int to !torch.number
  %1 = torch.prim.abs.Scalar %num: !torch.number -> !torch.number
  return %1 : !torch.number
}

// -----
// CHECK-LABEL:   func.func @prim.dtype(
// CHECK-SAME:        %[[arg:.*]]: !torch.vtensor<*,bf16>) -> !torch.vtensor {

// CHECK:           %[[zero:.*]] = torch.constant.int 0
// CHECK:           %[[false:.*]] = torch.constant.bool false

// CHECK:           %[[neg:.*]] = torch.aten.neg %[[arg]] : !torch.vtensor<*,bf16> -> !torch.vtensor<*,bf16>
// CHECK:           %[[dtype0:.*]] = torch.prim.dtype %[[neg]] : !torch.vtensor<*,bf16> -> !torch.int
// CHECK:           %[[device0:.*]] = torch.prim.device %[[neg]] : !torch.vtensor<*,bf16> -> !torch.Device
// CHECK:           %[[tensor:.*]] = torch.aten.tensor.int %[[zero]], %[[dtype0]], %[[device0]], %[[false]] : !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<*,bf16>

// CHECK:           %[[dtype1:.*]] = torch.prim.dtype %[[tensor]] : !torch.vtensor<*,bf16> -> !torch.int
// CHECK:           %[[device1:.*]] = torch.prim.device %[[tensor]] : !torch.vtensor<*,bf16> -> !torch.Device
// CHECK:           %[[result:.*]] = torch.aten.tensor.int %[[zero]], %[[dtype1]], %[[device1]], %[[false]] : !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<*,bf16>

// CHECK:           %[[cast:.*]] = torch.tensor_static_info_cast %[[result]] : !torch.vtensor<*,bf16> to !torch.vtensor
// CHECK:           return %[[cast]] : !torch.vtensor
// CHECK:         }

func.func @prim.dtype(%arg: !torch.vtensor<*,bf16>) -> !torch.vtensor<*,unk> {
  %zero = torch.constant.int 0
  %false = torch.constant.bool false

  // Op that requires type refinement
  %neg = torch.aten.neg %arg : !torch.vtensor<*,bf16> -> !torch.vtensor<*,unk>

  // Op whose processing requires type refinement on its source argument.
  %dtype = torch.prim.dtype %neg : !torch.vtensor<*,unk> -> !torch.int
  %device = torch.prim.device %neg : !torch.vtensor<*,unk> -> !torch.Device

  // Another op that requires type refinement
  %result = torch.aten.tensor.int %zero, %dtype, %device, %false : !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<*,unk>

  // Repeat the above three ops a second time to ensure that the type refinement
  // code works regardless of the number of alternating refinement+prim.dtype
  // sequences.
  %dtype2 = torch.prim.dtype %result : !torch.vtensor<*,unk> -> !torch.int
  %device2 = torch.prim.device %result : !torch.vtensor<*,unk> -> !torch.Device
  %result2 = torch.aten.tensor.int %zero, %dtype2, %device2, %false : !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<*,unk>

  return %result2 : !torch.vtensor<*,unk>
}

// -----

// Check that we don't crash on this input.

// CHECK-LABEL: func.func @forward
func.func @forward() -> !torch.vtensor {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct  : () -> !torch.list<tensor>
  // CHECK: torch.aten.tensor
  %1 = torch.aten.tensor %0, %none, %none, %false : !torch.list<tensor>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// Check that we don't crash on this input.
// TODO: This appears to result in aten.mul.Tensor not being visited.
// We should investigate why that happens.

// CHECK-LABEL: func.func @forward
func.func @forward(%arg0: !torch.bool, %arg1: !torch.tensor) {
  %0 = torch.prim.If %arg0 -> (!torch.tensor) {
    torch.prim.If.yield %arg1 : !torch.tensor
  } else {
    torch.prim.If.yield %arg1 : !torch.tensor
  }
  %1 = torch.copy.to_vtensor %0 : !torch.vtensor
  %2 = torch.aten.mul.Tensor %1, %1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func.func @torch.aten.zeros_like(
// CHECK-SAME:        %[[arg:.*]]: !torch.vtensor) {
// CHECK:           %[[INT6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[CPU:.*]] = torch.constant.device "cpu"
// CHECK:           %[[ZEROS:.*]] = torch.aten.zeros_like %[[arg]], %[[INT6]], %[[INT0]], %[[CPU]], %[[FALSE]], %[[INT1]] : !torch.vtensor, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.int -> !torch.vtensor<*,f32>
// CHECK:           return
func.func @torch.aten.zeros_like(%arg: !torch.vtensor) {
  %int6 = torch.constant.int 6
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %cpu = torch.constant.device "cpu"
  %2 = torch.aten.zeros_like %arg, %int6, %int0, %cpu, %false, %int1 : !torch.vtensor, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.int -> !torch.vtensor
  return
}

// -----

// The data-flow analysis does not always propagate information to the entire graph.
// This results in some lattice elements being uninitialized, which must be properly
// handled when using the lattice elements to rewrite the graph.
// In this particular case, the presence of the loop causes `torch.copy.to_vtensor`
// to end up with an uninitialized lattice element. This is the simplest graph I was
// able to come up with that reproduces such behavior.

// CHECK-LABEL:   func.func @uninitialized_lattice_elements(
// CHECK:           %{{.*}} = torch.copy.to_vtensor %{{.*}} : !torch.vtensor<*,f32>

func.func @uninitialized_lattice_elements(%arg0: !torch.vtensor<*,f32>, %arg3: !torch.tensor) -> !torch.vtensor<*,f32> {
  %true = torch.constant.bool true
  %1 = torch.constant.int 0
  %2 = torch.prim.Loop %1, %true, init(%arg3) {
  ^bb0(%arg1: !torch.int, %arg2: !torch.tensor):
    torch.prim.Loop.condition %true, iter(%arg2 : !torch.tensor)
  } : (!torch.int, !torch.bool, !torch.tensor) -> !torch.tensor
  %3 = torch.tensor_static_info_cast %2 : !torch.tensor to !torch.tensor<*,f32>
  %4 = torch.copy.to_vtensor %3 : !torch.vtensor<*,f32>
  return %4 : !torch.vtensor<*,f32>
}
