// RUN: torch-mlir-opt -torch-globalize-object-graph -verify-diagnostics -split-input-file %s

torch.class_type @c1 {}
torch.class_type @c2 {}

// expected-note @+1 {{see other root module here}}
torch.nn_module {} : !torch.nn.Module<"c1">
// expected-error @+1 {{found more than one root module (module that is not a child of any other module)}}
torch.nn_module {} : !torch.nn.Module<"c2">

// -----

torch.class_type @child {
  torch.attr "float" : !torch.float
}
torch.class_type @parent {
  torch.attr "m" : !torch.nn.Module<"child">
  torch.attr "m2" : !torch.nn.Module<"child">

}

%c42 = torch.constant.float 42.0
// expected-error @+1 {{reachable by multiple paths from root object: '<root>.m' and '<root>.m2'}}
%child = torch.nn_module {
  torch.slot "float", %c42 : !torch.float
} : !torch.nn.Module<"child">
%parent = torch.nn_module {
  torch.slot "m", %child : !torch.nn.Module<"child">
  torch.slot "m2", %child : !torch.nn.Module<"child">
} : !torch.nn.Module<"parent">

func.func private @ensure_all_slots_are_used(%arg0: !torch.nn.Module<"parent">, %arg1: !torch.nn.Module<"child">) {
  %0 = torch.prim.GetAttr %arg0["m"] : !torch.nn.Module<"parent"> -> !torch.nn.Module<"child">
  %1 = torch.prim.GetAttr %arg0["m2"] : !torch.nn.Module<"parent"> -> !torch.nn.Module<"child">
  %2 = torch.prim.GetAttr %arg1["float"] : !torch.nn.Module<"child"> -> !torch.float
  return
}
