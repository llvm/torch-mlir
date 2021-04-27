// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @child {
  torch.attr "float" : f64
}
torch.class_type @parent {
  torch.attr "m" : !torch.nn.Module<"child">
  torch.method "get_attr_returns_module_type", @get_attr_returns_module_type
  torch.method "module_type_argument", @module_type_argument
  torch.method "method_call", @method_call
}

// CHECK-LABEL:   func @get_attr_returns_module_type() -> f64 {
func private @get_attr_returns_module_type(%arg0: !torch.nn.Module<"parent">) -> f64 {
  %0 = torch.prim.GetAttr %arg0["m"] : !torch.nn.Module<"parent"> -> !torch.nn.Module<"child">
  // CHECK-NEXT: %[[V:.*]] = torch.global_slot.get @m.float : f64
  %1 = torch.prim.GetAttr %0["float"] : !torch.nn.Module<"child"> -> f64
  // CHECK-NEXT: torch.global_slot.set @m.float = %[[V]] : f64
  torch.prim.SetAttr %0["float"] = %1 : !torch.nn.Module<"child">, f64
  // CHECK-NEXT: return %[[V]] : f64
  return %1 : f64
}

// CHECK-LABEL:   func @module_type_argument(
// CHECK-SAME:                               %[[F:.*]]: f64) -> !basicpy.NoneType {
func private @module_type_argument(%arg0: !torch.nn.Module<"parent">, %arg1: !torch.nn.Module<"parent">, %arg2: f64, %arg3: !torch.nn.Module<"parent">) -> !basicpy.NoneType {
  %0 = basicpy.singleton : !basicpy.NoneType
  return %0 : !basicpy.NoneType
}

// CHECK-LABEL:   func @method_call() -> !basicpy.NoneType {
func private @method_call(%arg0: !torch.nn.Module<"parent">) -> !basicpy.NoneType {
  // CHECK-NEXT: %[[C:.*]] = constant 4.300000e+01 : f64
  %c = constant 43.0 : f64
  // CHECK-NEXT: %[[F:.*]] = call @module_type_argument(%[[C]]) : (f64) -> !basicpy.NoneType
  %0 = call @module_type_argument(%arg0, %arg0, %c, %arg0) : (!torch.nn.Module<"parent">, !torch.nn.Module<"parent">, f64, !torch.nn.Module<"parent">) -> (!basicpy.NoneType)
  // CHECK-NEXT: return %[[F]] : !basicpy.NoneType
  return %0 : !basicpy.NoneType
}

%c42 = constant 42.0 : f64
%child = torch.nn_module {
  torch.slot "float", %c42 : f64
} : !torch.nn.Module<"child">

torch.nn_module {
  torch.slot "m", %child : !torch.nn.Module<"child">
} : !torch.nn.Module<"parent">
