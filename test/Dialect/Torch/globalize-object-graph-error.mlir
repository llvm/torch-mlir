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

// -----

torch.class_type @c {
  torch.attr "a1" : !numpy.ndarray<*:!numpy.any_dtype>
  torch.attr "a2" : !numpy.ndarray<*:!numpy.any_dtype>
}

%cst = constant dense<1.000000e+00> : tensor<1xf32>
// expected-error @+1 {{potentially-aliased value used to initialize multiple slots}}
%a = numpy.create_array_from_tensor %cst : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
torch.nn_module {
  torch.slot "a1", %a : !numpy.ndarray<*:!numpy.any_dtype>
  torch.slot "a2", %a : !numpy.ndarray<*:!numpy.any_dtype>
} : !torch.nn.Module<"c">
