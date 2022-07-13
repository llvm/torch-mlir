// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @__torch__.TestModule  {
  torch.attr private "s1" : !torch.nn.Module<"__torch__.Submodule">
  torch.attr private "s2" : !torch.nn.Module<"__torch__.Submodule">
  torch.method "forward", @__torch__.TestModule.forward
}
torch.class_type @__torch__.Submodule  {
  torch.attr private "n" : !torch.int
  torch.method private "forward", @__torch__.Submodule.forward
}

// CHECK-LABEL:   torch.global_slot.module_initializer {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           torch.initialize.global_slots [
// CHECK:             @s1.n(%[[INT1]] : !torch.int)
// CHECK:             @s2.n(%[[INT2]] : !torch.int)
// CHECK:           ]
// CHECK:         }
// CHECK-LABEL:   torch.global_slot "private" @s1.n : !torch.int
// CHECK-LABEL:   torch.global_slot "private" @s2.n : !torch.int

%int1 = torch.constant.int 1
%s1 = torch.nn_module  {
  torch.slot "n", %int1 : !torch.int
} : !torch.nn.Module<"__torch__.Submodule">
%int2 = torch.constant.int 2
%s2 = torch.nn_module  {
  torch.slot "n", %int2 : !torch.int
} : !torch.nn.Module<"__torch__.Submodule">
%3 = torch.nn_module  {
  torch.slot "s1", %s1 : !torch.nn.Module<"__torch__.Submodule">
  torch.slot "s2", %s2 : !torch.nn.Module<"__torch__.Submodule">
} : !torch.nn.Module<"__torch__.TestModule">


// CHECK-LABEL:   func.func @forward() {
// CHECK:           call @s1.forward() : () -> ()
// CHECK:           call @s2.forward() : () -> ()
// CHECK:           return
func.func private @__torch__.TestModule.forward(%arg0: !torch.nn.Module<"__torch__.TestModule">) {
  %4 = torch.prim.GetAttr %arg0["s1"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  %5 = torch.prim.GetAttr %arg0["s2"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  call @__torch__.Submodule.forward(%4) : (!torch.nn.Module<"__torch__.Submodule">) -> ()
  call @__torch__.Submodule.forward(%5) : (!torch.nn.Module<"__torch__.Submodule">) -> ()
  return
}
// CHECK-LABEL:   func.func private @s1.forward() {
// CHECK:           %[[C3:.*]] = torch.constant.int 3
// CHECK:           %[[N:.*]] = torch.global_slot.get @s1.n : !torch.int
// CHECK:           %[[NEWVAL:.*]] = torch.aten.add.int %[[N]], %[[C3]] : !torch.int, !torch.int -> !torch.int
// CHECK:           torch.global_slot.set @s1.n = %[[NEWVAL]] : !torch.int
// CHECK:           return

// CHECK-LABEL:   func.func private @s2.forward() {
// CHECK:           %[[C3:.*]] = torch.constant.int 3
// CHECK:           %[[N:.*]] = torch.global_slot.get @s2.n : !torch.int
// CHECK:           %[[NEWVAL:.*]] = torch.aten.add.int %[[N]], %[[C3]] : !torch.int, !torch.int -> !torch.int
// CHECK:           torch.global_slot.set @s2.n = %[[NEWVAL]] : !torch.int
// CHECK:           return
func.func private @__torch__.Submodule.forward(%arg0: !torch.nn.Module<"__torch__.Submodule">) {
  %int3 = torch.constant.int 3
  %5 = torch.prim.GetAttr %arg0["n"] : !torch.nn.Module<"__torch__.Submodule"> -> !torch.int
  %6 = torch.aten.add.int %5, %int3 : !torch.int, !torch.int -> !torch.int
  torch.prim.SetAttr %arg0["n"] = %6 : !torch.nn.Module<"__torch__.Submodule">, !torch.int
  return
}
