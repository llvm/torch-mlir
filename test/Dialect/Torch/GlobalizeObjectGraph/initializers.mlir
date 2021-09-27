// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// CHECK that multiple nested initialization ops are properly handled.

// CHECK-LABEL:   torch.global_slot @l : !torch.list<!torch.list<!torch.list<!torch.tensor>>> {
// CHECK:           %[[L0:.*]] = torch.prim.ListConstruct  : () -> !torch.list<!torch.tensor>
// CHECK:           %[[L1:.*]] = torch.prim.ListConstruct %[[L0]], %[[L0]] : (!torch.list<!torch.tensor>, !torch.list<!torch.tensor>) -> !torch.list<!torch.list<!torch.tensor>>
// CHECK:           %[[L2:.*]] = torch.prim.ListConstruct %[[L1]], %[[L1]] : (!torch.list<!torch.list<!torch.tensor>>, !torch.list<!torch.list<!torch.tensor>>) -> !torch.list<!torch.list<!torch.list<!torch.tensor>>>
// CHECK:           torch.global_slot.init %[[L2]] : !torch.list<!torch.list<!torch.list<!torch.tensor>>>
// CHECK:         }

torch.class_type @c {
  torch.attr "l" : !torch.list<!torch.list<!torch.list<!torch.tensor>>>
}

%l0 = torch.prim.ListConstruct : () -> !torch.list<!torch.tensor>
%l1 = torch.prim.ListConstruct %l0, %l0 : (!torch.list<!torch.tensor>, !torch.list<!torch.tensor>) -> !torch.list<!torch.list<!torch.tensor>>
%l2 = torch.prim.ListConstruct %l1, %l1 : (!torch.list<!torch.list<!torch.tensor>>, !torch.list<!torch.list<!torch.tensor>>) -> !torch.list<!torch.list<!torch.list<!torch.tensor>>>
torch.nn_module {
  torch.slot "l", %l2 : !torch.list<!torch.list<!torch.list<!torch.tensor>>>
} : !torch.nn.Module<"c">
