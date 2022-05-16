// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @child {
  torch.attr "float" : !torch.float
}
torch.class_type @parent {
  torch.attr "m" : !torch.nn.Module<"child">
  torch.method "get_attr_returns_module_type", @get_attr_returns_module_type
  torch.method "module_type_argument", @module_type_argument
  torch.method "method_call", @method_call
}

// CHECK-LABEL:   func.func @get_attr_returns_module_type() -> !torch.float {
func.func private @get_attr_returns_module_type(%arg0: !torch.nn.Module<"parent">) -> !torch.float {
  %0 = torch.prim.GetAttr %arg0["m"] : !torch.nn.Module<"parent"> -> !torch.nn.Module<"child">
  // CHECK-NEXT: %[[V:.*]] = torch.global_slot.get @m.float : !torch.float
  %1 = torch.prim.GetAttr %0["float"] : !torch.nn.Module<"child"> -> !torch.float
  // CHECK-NEXT: torch.global_slot.set @m.float = %[[V]] : !torch.float
  torch.prim.SetAttr %0["float"] = %1 : !torch.nn.Module<"child">, !torch.float
  // CHECK-NEXT: return %[[V]] : !torch.float
  return %1 : !torch.float
}

// CHECK-LABEL:   func.func @module_type_argument(
// CHECK-SAME:                               %[[F:.*]]: !torch.float) -> !torch.none {
func.func private @module_type_argument(%arg0: !torch.nn.Module<"parent">, %arg1: !torch.nn.Module<"parent">, %arg2: !torch.float, %arg3: !torch.nn.Module<"parent">) -> !torch.none {
  %0 = torch.constant.none
  return %0 : !torch.none
}

// CHECK-LABEL:   func.func @method_call() -> !torch.none {
func.func private @method_call(%arg0: !torch.nn.Module<"parent">) -> !torch.none {
  // CHECK-NEXT: %[[C:.*]] = torch.constant.float 4.300000e+01
  %c = torch.constant.float 43.0
  // CHECK-NEXT: %[[F:.*]] = call @module_type_argument(%[[C]]) : (!torch.float) -> !torch.none
  %0 = call @module_type_argument(%arg0, %arg0, %c, %arg0) : (!torch.nn.Module<"parent">, !torch.nn.Module<"parent">, !torch.float, !torch.nn.Module<"parent">) -> (!torch.none)
  // CHECK-NEXT: return %[[F]] : !torch.none
  return %0 : !torch.none
}

%c42 = torch.constant.float 42.0
%child = torch.nn_module {
  torch.slot "float", %c42 : !torch.float
} : !torch.nn.Module<"child">

torch.nn_module {
  torch.slot "m", %child : !torch.nn.Module<"child">
} : !torch.nn.Module<"parent">
