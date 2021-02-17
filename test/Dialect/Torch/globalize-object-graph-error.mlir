// RUN: npcomp-opt -torch-globalize-object-graph -verify-diagnostics -split-input-file %s

torch.class_type @c1 {}
torch.class_type @c2 {}

// expected-note @+1 {{see other root module here}}
torch.nn_module {} : !torch.nn.Module<"c1">
// expected-error @+1 {{found more than one root module (module that is not a child of any other module)}}
torch.nn_module {} : !torch.nn.Module<"c2">

// -----

// expected-error @+1 {{class type has more than one instance: the current TorchScript supported subset only allows single instances}}
torch.class_type @child {}
torch.class_type @parent {
    torch.attr "m1" : !torch.nn.Module<"child">
    torch.attr "m2" : !torch.nn.Module<"child">
}

// expected-note @+1 {{see instance here}}
%0 = torch.nn_module {} : !torch.nn.Module<"child">
// expected-note @+1 {{see instance here}}
%1 = torch.nn_module {} : !torch.nn.Module<"child">

%root = torch.nn_module {
    torch.slot "m1", %0 : !torch.nn.Module<"child">
    torch.slot "m2", %1 : !torch.nn.Module<"child">
} : !torch.nn.Module<"parent">

// -----

torch.class_type @c {
  torch.method "f", @f
}
// expected-error @+1 {{func argument at index 1 has uses that were not able to be converted}}
func private @f(%arg0: !torch.nn.Module<"c">, %arg1: !torch.nn.Module<"c">) {
  // expected-note @+1 {{see user here}}
  %0 = basicpy.build_list %arg1 : (!torch.nn.Module<"c">) -> !basicpy.ListType
  return
}

torch.nn_module {} : !torch.nn.Module<"c">

// -----

// expected-error @+1 {{reachable by multiple paths from root object: '<root>.m' and '<root>.m2'}}
torch.class_type @child {
  torch.attr "float" : f64
}
torch.class_type @parent {
  torch.attr "m" : !torch.nn.Module<"child">
  torch.attr "m2" : !torch.nn.Module<"child">

}

%c42 = std.constant 42.0 : f64
%child = torch.nn_module {
  torch.slot "float", %c42 : f64
} : !torch.nn.Module<"child">
%parent = torch.nn_module {
  torch.slot "m", %child : !torch.nn.Module<"child">
  torch.slot "m2", %child : !torch.nn.Module<"child">
} : !torch.nn.Module<"parent">
