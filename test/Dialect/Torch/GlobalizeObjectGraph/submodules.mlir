// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Check that linkage names consist of the dotted path from the root. 

// CHECK-LABEL:   torch.global_slot @m.float : f64  {
// CHECK:           %[[INIT:.*]] = constant 4.200000e+01 : f64
// CHECK:           torch.global_slot.init %[[INIT]] : f64
// CHECK:         }


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
