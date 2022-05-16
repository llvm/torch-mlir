// RUN: torch-mlir-opt -torch-prepare-for-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  torch.method "test_call_method", @test_call_method
  torch.method "test_call_indirect", @test_call_indirect

}

// CHECK-LABEL:   func.func private @test_call_method(
// CHECK-SAME:                            %[[RECEIVER:.*]]: !torch.nn.Module<"c">,
// CHECK-SAME:                            %[[F:.*]]: !torch.float) -> !torch.float {
// CHECK:           %[[RET:.*]] = call @test_call_method(%[[RECEIVER]], %[[F]]) : (!torch.nn.Module<"c">, !torch.float) -> !torch.float
// CHECK:           return %[[RET]] : !torch.float
func.func private @test_call_method(%arg0: !torch.nn.Module<"c">, %arg1: !torch.float) -> !torch.float {
  %0 = torch.prim.CallMethod %arg0["test_call_method"] (%arg1) : !torch.nn.Module<"c">, (!torch.float) -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func private @test_call_indirect(
// CHECK-SAME:                                     %[[RECEIVER:.*]]: !torch.nn.Module<"c">,
// CHECK-SAME:                                     %[[F:.*]]: !torch.float) -> !torch.float {
// Ensure no func.constant.
// CHECK-NEXT:      %[[VAL_2:.*]] = call @test_call_method(%[[RECEIVER]], %[[F]]) : (!torch.nn.Module<"c">, !torch.float) -> !torch.float
// CHECK-NEXT:      return %[[VAL_2]] : !torch.float
func.func private @test_call_indirect(%arg0: !torch.nn.Module<"c">, %arg1: !torch.float) -> !torch.float {
  %0 = constant @test_call_method : (!torch.nn.Module<"c">, !torch.float) -> !torch.float
  %1 = call_indirect %0(%arg0, %arg1) : (!torch.nn.Module<"c">, !torch.float) -> !torch.float
  return %1 : !torch.float
}

torch.nn_module {
} : !torch.nn.Module<"c">
