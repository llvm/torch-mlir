// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  torch.attr "float" : f64
  torch.method "test_get", @test_get
  torch.method "test_set", @test_set
  torch.method "test_call", @test_call
}

// CHECK-LABEL:   func @test_get() -> f64 {
// CHECK:           %[[V:.*]] = torch.global_slot.get @float : f64
// CHECK:           return %[[V]] : f64
func private @test_get(%arg0: !torch.nn.Module<"c">) -> f64 {
  %0 = torch.prim.GetAttr %arg0["float"] : !torch.nn.Module<"c"> -> f64
  return %0 : f64
}

// CHECK-LABEL:   func @test_set(
// CHECK-SAME:                   %[[A:.*]]: f64) {
// CHECK:           torch.global_slot.set @float = %[[A]] : f64
// CHECK:           return
func private @test_set(%arg0: !torch.nn.Module<"c">, %arg1: f64) {
  torch.prim.SetAttr %arg0["float"] = %arg1 : !torch.nn.Module<"c">, f64
  return
}

// CHECK-LABEL:   func @test_call(
// CHECK-SAME:                    %[[A:.*]]: f64) -> f64 {
// CHECK:           %[[V:.*]] = call @test_call(%[[A]]) : (f64) -> f64
// CHECK:           return %[[V]] : f64
func private @test_call(%arg0: !torch.nn.Module<"c">, %arg1: f64) -> f64 {
  %0 = torch.prim.CallMethod %arg0["test_call"] (%arg1) : !torch.nn.Module<"c">, (f64) -> f64
  return %0 : f64
}

%c42 = std.constant 42.0 : f64
torch.nn_module {
  torch.slot "float", %c42 : f64
} : !torch.nn.Module<"c">
