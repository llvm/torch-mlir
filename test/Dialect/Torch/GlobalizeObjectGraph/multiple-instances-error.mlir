// RUN: npcomp-opt -torch-globalize-object-graph -verify-diagnostics -split-input-file %s

func private @__torch__.Submodule.forward(%arg0: !torch.nn.Module<"__torch__.Submodule">, %arg1: !torch.nn.Module<"__torch__.Submodule">) {
  return
}
func private @__torch__.TestModule.forward(%arg0: !torch.nn.Module<"__torch__.TestModule">) {
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
%bool_true = basicpy.bool_constant true
%0 = basicpy.singleton : !basicpy.NoneType
torch.class_type @__torch__.Submodule  {
  torch.attr private "n" : i64
  // expected-error @+1 {{public function with multiple monomorphizations}}
  torch.method "forward", @__torch__.Submodule.forward
}
%num1_i64 = basicpy.numeric_constant 1 : i64
%1 = torch.nn_module  {
  torch.slot "n", %num1_i64 : i64
} : !torch.nn.Module<"__torch__.Submodule">
%num2_i64 = basicpy.numeric_constant 2 : i64
%2 = torch.nn_module  {
  torch.slot "n", %num2_i64 : i64
} : !torch.nn.Module<"__torch__.Submodule">
%3 = torch.nn_module  {
  torch.slot "s1", %1 : !torch.nn.Module<"__torch__.Submodule">
  torch.slot "s2", %2 : !torch.nn.Module<"__torch__.Submodule">
} : !torch.nn.Module<"__torch__.TestModule">
