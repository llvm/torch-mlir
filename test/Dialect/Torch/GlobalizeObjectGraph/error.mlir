// RUN: npcomp-opt -torch-globalize-object-graph -verify-diagnostics -split-input-file %s

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

torch.class_type @c {
  torch.attr "t1" : !torch.tensor
  torch.attr "t2" : !torch.tensor
}

// expected-error @+1 {{potentially-aliased value used to initialize multiple slots}}
%t = torch.tensor(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
torch.nn_module {
  torch.slot "t1", %t : !torch.tensor
  torch.slot "t2", %t : !torch.tensor
} : !torch.nn.Module<"c">
