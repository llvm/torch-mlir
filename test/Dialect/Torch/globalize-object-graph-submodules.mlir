// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Check that linkage names consist of the dotted path from the root. 

// CHECK-LABEL:   torch.global_slot @m.float : f64

// CHECK-LABEL:   func @__torch_global_slot_initializer() {
// CHECK:           %[[C42:.*]] = constant 4.200000e+01 : f64
// CHECK:           torch.global_slot.set @m.float = %[[C42]] : f64
// CHECK:           return

torch.class_type @child {
  torch.attr "float" : f64
}
torch.class_type @parent {
  torch.attr "m" : !torch.nn.Module<"child">
}

%c42 = std.constant 42.0 : f64
%child = torch.nn_module {
  torch.slot "float", %c42 : f64
} : !torch.nn.Module<"child">
%parent = torch.nn_module {
  torch.slot "m", %child : !torch.nn.Module<"child">
} : !torch.nn.Module<"parent">
