// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  torch.attr "float" : !torch.float
  torch.method "test_get", @test_get
  torch.method "test_set", @test_set
  torch.method "test_call", @test_call
}

// CHECK-LABEL:   func.func @test_get() -> !torch.float {
// CHECK:           %[[V:.*]] = torch.global_slot.get @float : !torch.float
// CHECK:           return %[[V]] : !torch.float
func.func private @test_get(%arg0: !torch.nn.Module<"c">) -> !torch.float {
  %0 = torch.prim.GetAttr %arg0["float"] : !torch.nn.Module<"c"> -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @test_set(
// CHECK-SAME:                   %[[A:.*]]: !torch.float) {
// CHECK:           torch.global_slot.set @float = %[[A]] : !torch.float
// CHECK:           return
func.func private @test_set(%arg0: !torch.nn.Module<"c">, %arg1: !torch.float) {
  torch.prim.SetAttr %arg0["float"] = %arg1 : !torch.nn.Module<"c">, !torch.float
  return
}

// CHECK-LABEL:   func.func @test_call(
// CHECK-SAME:                    %[[A:.*]]: !torch.float) -> !torch.float {
// CHECK:           %[[V:.*]] = call @test_call(%[[A]]) : (!torch.float) -> !torch.float
// CHECK:           return %[[V]] : !torch.float
func.func private @test_call(%arg0: !torch.nn.Module<"c">, %arg1: !torch.float) -> !torch.float {
  %0 = call @test_call(%arg0, %arg1) : (!torch.nn.Module<"c">, !torch.float) -> !torch.float
  return %0 : !torch.float
}

%c42 = torch.constant.float 42.0
torch.nn_module {
  torch.slot "float", %c42 : !torch.float
} : !torch.nn.Module<"c">
