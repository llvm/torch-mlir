// RUN: npcomp-opt -torch-inline-global-slots -split-input-file %s | FileCheck %s

// CHECK-NOT: @readonly
torch.global_slot "private" @readonly : !torch.tensor  {
  %0 = torch.tensor.literal(dense<0.0> : tensor<1xf32>) : !torch.tensor
  torch.global_slot.init %0 : !torch.tensor
}
// CHECK-LABEL: torch.global_slot @public
torch.global_slot @public : !torch.tensor  {
  %0 = torch.tensor.literal(dense<0.0> : tensor<2xf32>) : !torch.tensor
  torch.global_slot.init %0 : !torch.tensor
}
// CHECK-LABEL: torch.global_slot "private" @mutated
torch.global_slot "private" @mutated : !torch.tensor  {
  %0 = torch.tensor.literal(dense<0.0> : tensor<3xf32>) : !torch.tensor
  torch.global_slot.init %0 : !torch.tensor
}

// CHECK-LABEL:   func @forward() -> (!torch.tensor, !torch.tensor, !torch.tensor) {
func @forward() -> (!torch.tensor, !torch.tensor, !torch.tensor) {
  // Inlined.
  // CHECK:           %[[READONLY:.*]] = torch.tensor.literal(dense<0.000000e+00> : tensor<1xf32>) : !torch.tensor
  %0 = torch.global_slot.get @readonly : !torch.tensor

  // Not inlined: potentially mutated by externals.
  // CHECK:           %[[PUBLIC:.*]] = torch.global_slot.get @public : !torch.tensor
  %1 = torch.global_slot.get @public : !torch.tensor

  // Not inlined: potentially mutated internally.
  // CHECK:           torch.global_slot.set @mutated = %[[READONLY]] : !torch.tensor
  // CHECK:           %[[MUTATED:.*]] = torch.global_slot.get @mutated : !torch.tensor
  torch.global_slot.set @mutated = %0 : !torch.tensor
  %2 = torch.global_slot.get @mutated : !torch.tensor

  // CHECK:           return %[[READONLY]], %[[PUBLIC]], %[[MUTATED]] : !torch.tensor, !torch.tensor, !torch.tensor
  return %0, %1, %2 : !torch.tensor, !torch.tensor, !torch.tensor
}
