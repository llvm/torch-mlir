// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Check that multiple nested initialization ops are properly handled.

// CHECK-LABEL:   torch.global_slot.module_initializer {
// CHECK:           %[[L0:.*]] = torch.prim.ListConstruct  : () -> !torch.list<tensor>
// CHECK:           %[[L1:.*]] = torch.prim.ListConstruct %[[L0]], %[[L0]] : (!torch.list<tensor>, !torch.list<tensor>) -> !torch.list<list<tensor>>
// CHECK:           %[[L2:.*]] = torch.prim.ListConstruct %[[L1]], %[[L1]] : (!torch.list<list<tensor>>, !torch.list<list<tensor>>) -> !torch.list<list<list<tensor>>>
// CHECK:           torch.initialize.global_slots [
// CHECK:             @l(%[[L2]] : !torch.list<list<list<tensor>>>)
// CHECK:           ]
// CHECK:         }
// CHECK-LABEL:   torch.global_slot @l : !torch.list<list<list<tensor>>>

torch.class_type @c {
  torch.attr "l" : !torch.list<list<list<tensor>>>
}

%l0 = torch.prim.ListConstruct : () -> !torch.list<tensor>
%l1 = torch.prim.ListConstruct %l0, %l0 : (!torch.list<tensor>, !torch.list<tensor>) -> !torch.list<list<tensor>>
%l2 = torch.prim.ListConstruct %l1, %l1 : (!torch.list<list<tensor>>, !torch.list<list<tensor>>) -> !torch.list<list<list<tensor>>>
torch.nn_module {
  torch.slot "l", %l2 : !torch.list<list<list<tensor>>>
} : !torch.nn.Module<"c">

func.func private @ensure_all_slots_are_used(%arg0: !torch.nn.Module<"c">) {
  %0 = torch.prim.GetAttr %arg0["l"] : !torch.nn.Module<"c"> -> !torch.list<list<list<tensor>>>
  return
}
