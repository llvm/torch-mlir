// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  torch.attr "float" : f64
  torch.method "calls_free_function", @calls_free_function
}
// CHECK-LABEL:   func private
// CHECK-SAME:        @free_function$[[$MONOMORPHIZE_TAG0:.*]](
// CHECK-SAME:                                                 %[[F:.*]]: f64) -> f64 {
// CHECK:           return %[[F]] : f64
// CHECK:         }
func private @free_function(%arg0: f64, %arg1: !torch.nn.Module<"c">) -> f64 {
  return %arg0 : f64
}

// CHECK-LABEL:   func private
// CHECK-SAME:        @free_function_no_module_args$[[$MONOMORPHIZE_TAG1:.*]](
// CHECK-SAME:                                                 %[[F:.*]]: f64) -> f64 {
// CHECK:           return %[[F]] : f64
// CHECK:         }
func private @free_function_no_module_args(%arg0: f64) -> f64 {
  return %arg0 : f64
}

// CHECK-LABEL:   func @calls_free_function() -> f64 {
// CHECK:           %[[F1:.*]] = torch.global_slot.get @float : f64
// CHECK:           %[[F2:.*]] = call @free_function$[[$MONOMORPHIZE_TAG0]](%[[F1]]) : (f64) -> f64
// CHECK:           %[[RET:.*]] = call @free_function_no_module_args$[[$MONOMORPHIZE_TAG1]](%[[F2]]) : (f64) -> f64
// CHECK:           return %[[RET]] : f64
// CHECK:         }
func private @calls_free_function(%arg0: !torch.nn.Module<"c">) -> f64 {
  %0 = torch.prim.GetAttr %arg0["float"] : !torch.nn.Module<"c"> -> f64
  %1 = call @free_function(%0, %arg0) : (f64, !torch.nn.Module<"c">) -> f64
  %2 = call @free_function_no_module_args(%1) : (f64) -> f64
  return %2 : f64
}

%c42 = std.constant 42.0 : f64
torch.nn_module {
  torch.slot "float", %c42 : f64
} : !torch.nn.Module<"c">
