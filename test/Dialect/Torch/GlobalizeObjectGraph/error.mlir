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

// -----


func.func private @forward(%arg0: !torch.nn.Module<"__torch__.Model">, %arg1: !torch.tensor) -> !torch.tensor {
  %tmp0 = torch.prim.GetAttr %arg0["slot0"] : !torch.nn.Module<"__torch__.Model"> -> !torch.tensor
  %tmp1 = torch.prim.GetAttr %arg0["slot1"] : !torch.nn.Module<"__torch__.Model"> -> !torch.tensor
  %0 = torch.operator "foo"(%tmp0, %tmp1) : (!torch.tensor, !torch.tensor) -> !torch.tensor
  return %0 : !torch.tensor
}

torch.class_type @__torch__.Model {
  torch.attr private "slot0" : !torch.tensor
  torch.attr private "slot1" : !torch.tensor
  torch.method "forward", @forward
}

// expected-error @+1 {{unsafe initializer used to initialize multiple slots}}
%tensor = torch.tensor.literal(dense<0.000000e+00> : tensor<2xf32>) : !torch.tensor

%module = torch.nn_module {
  torch.slot "slot0", %tensor : !torch.tensor
  torch.slot "slot1", %tensor : !torch.tensor
} : !torch.nn.Module<"__torch__.Model">
