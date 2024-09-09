// RUN: torch-mlir-opt -torch-inline-global-slots -split-input-file %s | FileCheck %s

// Safety analysis aspect of the pass.




// Test case: An unsafe subobject (@tensor) blocks inlining of the containing object (@list).
// Note that we can check just the initializer -- if we inlined the slot, then
// we would have eliminated the slot from the initializer.
// Also, the initializer is verified to match the list of global slots in the
// module. So it is a nice one-stop-shop.

torch.global_slot "private" @tensor : !torch.tensor
torch.global_slot "private" @list : !torch.list<tensor>

// CHECK-LABEL:   torch.global_slot.module_initializer {
// CHECK:           torch.initialize.global_slots [
// CHECK-NEXT:        @tensor(%{{.*}} : !torch.tensor)
// CHECK-NEXT:        @list(%{{.*}} : !torch.list<tensor>)
// CHECK-NEXT:      ]
torch.global_slot.module_initializer {
  %0 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor
  %1 = torch.prim.ListConstruct %0 : (!torch.tensor) -> !torch.list<tensor>
  torch.initialize.global_slots [
    @tensor(%0 : !torch.tensor)
    @list(%1 : !torch.list<tensor>)
  ]
}

func.func @forward() {
  %int0 = torch.constant.int 0
  %0 = torch.global_slot.get @list : !torch.list<tensor>
  %tensor = torch.global_slot.get @tensor : !torch.tensor
  torch.aten.relu_ %tensor : !torch.tensor -> !torch.tensor
  return
}
