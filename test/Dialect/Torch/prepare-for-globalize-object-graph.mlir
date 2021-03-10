// RUN: npcomp-opt -torch-prepare-for-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  torch.method "test_call_method", @test_call_method
  torch.method "test_call_indirect", @test_call_indirect

}

// CHECK-LABEL:   func private @test_call_method(
// CHECK-SAME:                            %[[RECEIVER:.*]]: !torch.nn.Module<"c">,
// CHECK-SAME:                            %[[F:.*]]: f64) -> f64 {
// CHECK:           %[[RET:.*]] = call @test_call_method(%[[RECEIVER]], %[[F]]) : (!torch.nn.Module<"c">, f64) -> f64
// CHECK:           return %[[RET]] : f64
func private @test_call_method(%arg0: !torch.nn.Module<"c">, %arg1: f64) -> f64 {
  %0 = torch.prim.CallMethod %arg0["test_call_method"] (%arg1) : !torch.nn.Module<"c">, (f64) -> f64
  return %0 : f64
}

// CHECK-LABEL:   func private @test_call_indirect(
// CHECK-SAME:                                     %[[RECEIVER:.*]]: !torch.nn.Module<"c">,
// CHECK-SAME:                                     %[[F:.*]]: f64) -> f64 {
// Ensure no std.constant.
// CHECK-NEXT:      %[[VAL_2:.*]] = call @test_call_method(%[[RECEIVER]], %[[F]]) : (!torch.nn.Module<"c">, f64) -> f64
// CHECK-NEXT:      return %[[VAL_2]] : f64
func private @test_call_indirect(%arg0: !torch.nn.Module<"c">, %arg1: f64) -> f64 {
  %0 = constant @test_call_method : (!torch.nn.Module<"c">, f64) -> f64
  %1 = call_indirect %0(%arg0, %arg1) : (!torch.nn.Module<"c">, f64) -> f64
  return %1 : f64
}

torch.nn_module {
} : !torch.nn.Module<"c">
