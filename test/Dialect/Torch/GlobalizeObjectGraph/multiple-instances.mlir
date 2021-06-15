// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @__torch__.TestModule  {
  torch.attr private "s1" : !torch.nn.Module<"__torch__.Submodule">
  torch.attr private "s2" : !torch.nn.Module<"__torch__.Submodule">
  torch.method "forward", @__torch__.TestModule.forward
}
torch.class_type @__torch__.Submodule  {
  torch.attr private "n" : i64
  torch.method private "forward", @__torch__.Submodule.forward
}

%num1_i64 = torch.constant.int 1 : i64
%s1 = torch.nn_module  {
  // CHECK-LABEL:   torch.global_slot "private" @s1.n : i64  {
  // CHECK:           %[[C1:.*]] = torch.constant.int 1 : i64
  // CHECK:           torch.global_slot.init %[[C1]] : i64
  // CHECK:         }
  torch.slot "n", %num1_i64 : i64
} : !torch.nn.Module<"__torch__.Submodule">
%num2_i64 = torch.constant.int 2 : i64
%s2 = torch.nn_module  {
  // CHECK-LABEL:   torch.global_slot "private" @s2.n : i64  {
  // CHECK:           %[[C2:.*]] = torch.constant.int 2 : i64
  // CHECK:           torch.global_slot.init %[[C2]] : i64
  // CHECK:         }
  torch.slot "n", %num2_i64 : i64
} : !torch.nn.Module<"__torch__.Submodule">
%3 = torch.nn_module  {
  torch.slot "s1", %s1 : !torch.nn.Module<"__torch__.Submodule">
  torch.slot "s2", %s2 : !torch.nn.Module<"__torch__.Submodule">
} : !torch.nn.Module<"__torch__.TestModule">


// CHECK-LABEL:   func @forward() {
// CHECK:           call @s1.forward() : () -> ()
// CHECK:           call @s2.forward() : () -> ()
// CHECK:           return
func private @__torch__.TestModule.forward(%arg0: !torch.nn.Module<"__torch__.TestModule">) {
  %4 = torch.prim.GetAttr %arg0["s1"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  %5 = torch.prim.GetAttr %arg0["s2"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  call @__torch__.Submodule.forward(%4) : (!torch.nn.Module<"__torch__.Submodule">) -> ()
  call @__torch__.Submodule.forward(%5) : (!torch.nn.Module<"__torch__.Submodule">) -> ()
  return
}
// CHECK-LABEL:   func private @s1.forward() {
// CHECK:           %[[C1:.*]] = torch.constant.int 1 : i64
// CHECK:           %[[N:.*]] = torch.global_slot.get @s1.n : i64
// CHECK:           %[[NEWVAL:.*]] = addi %[[N]], %[[C1]] : i64
// CHECK:           torch.global_slot.set @s1.n = %[[NEWVAL]] : i64
// CHECK:           return

// CHECK-LABEL:   func private @s2.forward() {
// CHECK:           %[[C1:.*]] = torch.constant.int 1 : i64
// CHECK:           %[[N:.*]] = torch.global_slot.get @s2.n : i64
// CHECK:           %[[NEWVAL:.*]] = addi %[[N]], %[[C1]] : i64
// CHECK:           torch.global_slot.set @s2.n = %[[NEWVAL]] : i64
// CHECK:           return
func private @__torch__.Submodule.forward(%arg0: !torch.nn.Module<"__torch__.Submodule">) {
  %c1_i64 = torch.constant.int 1 : i64
  %5 = torch.prim.GetAttr %arg0["n"] : !torch.nn.Module<"__torch__.Submodule"> -> i64
  %6 = addi %5, %c1_i64 : i64
  torch.prim.SetAttr %arg0["n"] = %6 : !torch.nn.Module<"__torch__.Submodule">, i64
  return
}
