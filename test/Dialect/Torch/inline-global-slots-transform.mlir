// RUN: torch-mlir-opt -torch-inline-global-slots -split-input-file %s | FileCheck %s

// Transform aspect of the pass.

// Test case: Most basic case that can be inlined.

// CHECK-NOT: @slot0
torch.global_slot "private" @slot0 : !torch.int

// CHECK-LABEL:   torch.global_slot.module_initializer {
// CHECK:           torch.initialize.global_slots [
// CHECK-NEXT       ]
torch.global_slot.module_initializer {
  %0 = torch.constant.int 1
  torch.initialize.global_slots [
    @slot0(%0 : !torch.int)
  ]
}

// CHECK-LABEL:   func.func @forward() {
// CHECK:           %[[C1:.*]] = torch.constant.int 1
// CHECK:           return
func.func @forward() {
  %0 = torch.global_slot.get @slot0 : !torch.int
  return
}

// -----

// Test case: Shared objects in object graph shared between two initial values.

torch.global_slot "private" @tensor : !torch.tensor
torch.global_slot "private" @list_of_tensor : !torch.list<tensor>

// CHECK-LABEL:   torch.global_slot.module_initializer {
// CHECK:           torch.initialize.global_slots [
// CHECK-NEXT       ]
torch.global_slot.module_initializer {
  %0 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor
  %1 = torch.prim.ListConstruct %0 : (!torch.tensor) -> !torch.list<tensor>
  torch.initialize.global_slots [
    @tensor(%0 : !torch.tensor)
    @list_of_tensor(%1 : !torch.list<tensor>)
  ]
}

// CHECK-LABEL:   func.func @forward() {
// CHECK:           %[[T0:.*]] = torch.tensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.tensor
// CHECK:           %[[T1:.*]] = torch.tensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.tensor
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[T1]] : (!torch.tensor) -> !torch.list<tensor>
// CHECK:           return
func.func @forward() {
  %0 = torch.global_slot.get @tensor : !torch.tensor
  %1 = torch.global_slot.get @list_of_tensor : !torch.tensor
  return
}

// -----

// Test case: Adjusting static info.

// CHECK-NOT: @tensor
torch.global_slot "private" @tensor : !torch.tensor

// CHECK-LABEL:   torch.global_slot.module_initializer {
// CHECK:           torch.initialize.global_slots [
// CHECK-NEXT       ]
torch.global_slot.module_initializer {
  %0 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor<[],f32>
  torch.initialize.global_slots [
    @tensor(%0 : !torch.tensor<[],f32>)
  ]
}

// CHECK-LABEL:   func.func @forward() {
// CHECK:           %[[T:.*]] = torch.tensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.tensor<[],f32>
// CHECK:           %[[CASTED:.*]] = torch.tensor_static_info_cast %[[T]] : !torch.tensor<[],f32> to !torch.tensor
func.func @forward() {
  %0 = torch.global_slot.get @tensor : !torch.tensor
  return
}
