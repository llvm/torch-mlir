// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Tests monomorphization of same function with different instance argument types.

torch.class_type @__torch__.TestModule  {
  torch.attr private "s1" : !torch.nn.Module<"__torch__.Submodule">
  torch.attr private "s2" : !torch.nn.Module<"__torch__.Submodule">
  torch.method "forward", @__torch__.TestModule.forward
}
torch.class_type @__torch__.Submodule  {
  torch.attr private "n" : i64
  torch.method private "forward", @__torch__.Submodule.forward
}

%num1_i64 = basicpy.numeric_constant 1 : i64
%s1 = torch.nn_module  {
  // CHECK-LABEL:   torch.global_slot "private" @s1.n : i64  {
  // CHECK:           %[[C1:.*]] = basicpy.numeric_constant 1 : i64
  // CHECK:           torch.global_slot.init %[[C1]] : i64
  // CHECK:         }
  torch.slot "n", %num1_i64 : i64
} : !torch.nn.Module<"__torch__.Submodule">
%num2_i64 = basicpy.numeric_constant 2 : i64
%s2 = torch.nn_module  {
  // CHECK-LABEL:   torch.global_slot "private" @s2.n : i64  {
  // CHECK:           %[[C2:.*]] = basicpy.numeric_constant 2 : i64                                                                                                                                              
  // CHECK:           torch.global_slot.init %[[C2]] : i64
  // CHECK:         }
  torch.slot "n", %num2_i64 : i64
} : !torch.nn.Module<"__torch__.Submodule">
%3 = torch.nn_module  {
  torch.slot "s1", %s1 : !torch.nn.Module<"__torch__.Submodule">
  torch.slot "s2", %s2 : !torch.nn.Module<"__torch__.Submodule">
} : !torch.nn.Module<"__torch__.TestModule">


// CHECK-LABEL:   func @forward() {
// CHECK:           call @__torch__.free_function$[[$MONOMORPHIZE_TAG0:.*]]() : () -> ()
// CHECK:           call @__torch__.free_function$[[$MONOMORPHIZE_TAG1:.*]]() : () -> ()
func private @__torch__.TestModule.forward(%arg0: !torch.nn.Module<"__torch__.TestModule">) {
  %4 = torch.prim.GetAttr %arg0["s1"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  %5 = torch.prim.GetAttr %arg0["s2"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  call @__torch__.free_function(%4, %5) : (!torch.nn.Module<"__torch__.Submodule">, !torch.nn.Module<"__torch__.Submodule">) -> ()
  %7 = torch.prim.GetAttr %arg0["s2"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  %8 = torch.prim.GetAttr %arg0["s1"] : !torch.nn.Module<"__torch__.TestModule"> -> !torch.nn.Module<"__torch__.Submodule">
  call @__torch__.free_function(%7, %8) : (!torch.nn.Module<"__torch__.Submodule">, !torch.nn.Module<"__torch__.Submodule">) -> ()
  return
}

// s1 called first, then s2
// CHECK-LABEL:   func private
// CHECK-SAME         @__torch__.free_function$[[$MONOMORPHIZE_TAG0]]() {
// CHECK:           call @s1.forward() : () -> ()
// CHECK:           call @s2.forward() : () -> ()

// s2 called first, then s1
// CHECK-LABEL:   func private
// CHECK-SAME:        @__torch__.free_function$[[$MONOMORPHIZE_TAG1]]() {
// CHECK:           call @s2.forward() : () -> ()
// CHECK:           call @s1.forward() : () -> ()
func private @__torch__.free_function(%arg0: !torch.nn.Module<"__torch__.Submodule">, %arg1: !torch.nn.Module<"__torch__.Submodule">) {
  call @__torch__.Submodule.forward(%arg0) : (!torch.nn.Module<"__torch__.Submodule">) -> ()
  call @__torch__.Submodule.forward(%arg1) : (!torch.nn.Module<"__torch__.Submodule">) -> ()
  return
}

// CHECK-LABEL:   func private @s2.forward() {
// CHECK:           return

// CHECK-LABEL:   func private @s1.forward() {
// CHECK:           return
func private @__torch__.Submodule.forward(%arg0: !torch.nn.Module<"__torch__.Submodule">) {
  return
}
