// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  torch.attr "float" : !torch.float
  torch.method "calls_free_function", @calls_free_function
}
// CHECK-LABEL:   func.func private
// CHECK-SAME:        @free_function$[[$MONOMORPHIZE_TAG0:.*]](
// CHECK-SAME:                                                 %[[F:.*]]: !torch.float) -> !torch.float {
// CHECK:           return %[[F]] : !torch.float
// CHECK:         }
func.func private @free_function(%arg0: !torch.float, %arg1: !torch.nn.Module<"c">) -> !torch.float {
  return %arg0 : !torch.float
}

// CHECK-LABEL:   func.func private
// CHECK-SAME:        @free_function_no_module_args$[[$MONOMORPHIZE_TAG1:.*]](
// CHECK-SAME:                                                 %[[F:.*]]: !torch.float) -> !torch.float {
// CHECK:           return %[[F]] : !torch.float
// CHECK:         }
func.func private @free_function_no_module_args(%arg0: !torch.float) -> !torch.float {
  return %arg0 : !torch.float
}

// CHECK-LABEL:   func.func @calls_free_function() -> !torch.float {
// CHECK:           %[[F1:.*]] = torch.global_slot.get @float : !torch.float
// CHECK:           %[[F2:.*]] = call @free_function$[[$MONOMORPHIZE_TAG0]](%[[F1]]) : (!torch.float) -> !torch.float
// CHECK:           %[[RET:.*]] = call @free_function_no_module_args$[[$MONOMORPHIZE_TAG1]](%[[F2]]) : (!torch.float) -> !torch.float
// CHECK:           return %[[RET]] : !torch.float
// CHECK:         }
func.func private @calls_free_function(%arg0: !torch.nn.Module<"c">) -> !torch.float {
  %0 = torch.prim.GetAttr %arg0["float"] : !torch.nn.Module<"c"> -> !torch.float
  %1 = call @free_function(%0, %arg0) : (!torch.float, !torch.nn.Module<"c">) -> !torch.float
  %2 = call @free_function_no_module_args(%1) : (!torch.float) -> !torch.float
  return %2 : !torch.float
}

%c42 = torch.constant.float 42.0
torch.nn_module {
  torch.slot "float", %c42 : !torch.float
} : !torch.nn.Module<"c">
