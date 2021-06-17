// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Check that linkage names consist of the dotted path from the root. 

// CHECK-LABEL:   torch.global_slot @m.float : !torch.float  {
// CHECK:           %[[INIT:.*]] = torch.constant.float 4.200000e+01
// CHECK:           torch.global_slot.init %[[INIT]] : !torch.float
// CHECK:         }


torch.class_type @child {
  torch.attr "float" : !torch.float
}
torch.class_type @parent {
  torch.attr "m" : !torch.nn.Module<"child">
}

%c42 = torch.constant.float 42.0
%child = torch.nn_module {
  torch.slot "float", %c42 : !torch.float
} : !torch.nn.Module<"child">
%parent = torch.nn_module {
  torch.slot "m", %child : !torch.nn.Module<"child">
} : !torch.nn.Module<"parent">
