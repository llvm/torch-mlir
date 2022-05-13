// RUN: torch-mlir-opt -torch-globalize-object-graph -verify-diagnostics -split-input-file %s

func.func private @__torch__.Submodule.forward(%arg0: !torch.nn.Module<"__torch__.Submodule">, %arg1: !torch.nn.Module<"__torch__.Submodule">) {
  return
}
func.func private @__torch__.TestModule.forward(%arg0: !torch.nn.Module<"__torch__.TestModule">) {
  %5 = torch.prim.GetAttr %arg0["s1"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  %6 = torch.prim.GetAttr %arg0["s2"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  call @__torch__.Submodule.forward(%5, %6) : (!torch.nn.Module<"__torch__.Submodule">, !torch.nn.Module<"__torch__.Submodule">) -> ()
  call @__torch__.Submodule.forward(%5, %5) : (!torch.nn.Module<"__torch__.Submodule">, !torch.nn.Module<"__torch__.Submodule">) -> ()
  return
}
torch.class_type @__torch__.TestModule  {
  torch.attr private "s1" : !torch.nn.Module<"__torch__.Submodule">
  torch.attr private "s2" : !torch.nn.Module<"__torch__.Submodule">
  torch.method "forward", @__torch__.TestModule.forward
}
%bool_true = torch.constant.bool true
%0 = torch.constant.none
torch.class_type @__torch__.Submodule  {
  torch.attr private "n" : !torch.int
  // expected-error @+1 {{public function with multiple monomorphizations}}
  torch.method "forward", @__torch__.Submodule.forward
}
%int1 = torch.constant.int 1
%1 = torch.nn_module  {
  torch.slot "n", %int1 : !torch.int
} : !torch.nn.Module<"__torch__.Submodule">
%int2 = torch.constant.int 2
%2 = torch.nn_module  {
  torch.slot "n", %int2 : !torch.int
} : !torch.nn.Module<"__torch__.Submodule">
%3 = torch.nn_module  {
  torch.slot "s1", %1 : !torch.nn.Module<"__torch__.Submodule">
  torch.slot "s2", %2 : !torch.nn.Module<"__torch__.Submodule">
} : !torch.nn.Module<"__torch__.TestModule">
