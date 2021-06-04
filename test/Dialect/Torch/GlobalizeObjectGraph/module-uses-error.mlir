// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file -verify-diagnostics %s

torch.class_type @parent {
  torch.method "module_type_return", @module_type_return
}

func private @module_type_return(%arg0: !torch.nn.Module<"parent">) {
  // expected-error @+1 {{unsupported use of a torch.nn.Module. Expected only method calls or attribute get/set}}
  torch.prim.ListConstruct %arg0 : (!torch.nn.Module<"parent">) -> !torch.list<!torch.nn.Module<"parent">>
  return
}

torch.nn_module {} : !torch.nn.Module<"parent">
