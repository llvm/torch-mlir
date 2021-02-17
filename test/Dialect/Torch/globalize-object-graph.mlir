// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Basic case.

// CHECK-LABEL:   torch.global_slot @b : !basicpy.BoolType
// CHECK:         torch.global_slot @i : i64
// CHECK:         torch.global_slot @f : f64
// CHECK:         torch.global_slot @a : !numpy.ndarray<*:!numpy.any_dtype>

// CHECK-LABEL:   func @__torch_global_slot_initializer() {
// CHECK:           %[[CB:.*]] = basicpy.bool_constant true
// CHECK:           torch.global_slot.set @b = %[[CB]] : !basicpy.BoolType
// CHECK:           %[[CI:.*]] = basicpy.numeric_constant 3 : i64
// CHECK:           torch.global_slot.set @i = %[[CI]] : i64
// CHECK:           %[[CF:.*]] = basicpy.numeric_constant 4.250000e+01 : f64
// CHECK:           torch.global_slot.set @f = %[[CF]] : f64
// CHECK:           %[[C:.*]] = constant dense<1.000000e+00> : tensor<1xf32>
// CHECK:           %[[CA:.*]] = numpy.create_array_from_tensor %[[C]] : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           torch.global_slot.set @a = %[[CA]] : !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           return

torch.class_type @c {
  torch.attr "b" : !basicpy.BoolType
  torch.attr "i" : i64
  torch.attr "f" : f64
  torch.attr "a" : !numpy.ndarray<*:!numpy.any_dtype>
}

%bool_true = basicpy.bool_constant true
%i = basicpy.numeric_constant 3 : i64
%f = basicpy.numeric_constant 4.250000e+01 : f64
%cst = constant dense<1.000000e+00> : tensor<1xf32>
%a = numpy.create_array_from_tensor %cst : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
torch.nn_module {
  torch.slot "b", %bool_true : !basicpy.BoolType
  torch.slot "i", %i : i64
  torch.slot "f", %f : f64
  torch.slot "a", %a : !numpy.ndarray<*:!numpy.any_dtype>
} : !torch.nn.Module<"c">

// -----

// Same SSA value used as initializer for multiple slots.

// CHECK-LABEL:   torch.global_slot @b1 : !basicpy.BoolType
// CHECK-LABEL:   torch.global_slot @b2 : !basicpy.BoolType
// CHECK-LABEL:   func @__torch_global_slot_initializer() {
// CHECK:           %[[TRUE:.*]] = basicpy.bool_constant true
// CHECK:           torch.global_slot.set @b1 = %[[TRUE]] : !basicpy.BoolType
// CHECK:           torch.global_slot.set @b2 = %[[TRUE]] : !basicpy.BoolType
// CHECK:           return
// CHECK:         }

torch.class_type @c {
  torch.attr "b1" : !basicpy.BoolType
  torch.attr "b2" : !basicpy.BoolType
}

%bool_true = basicpy.bool_constant true
torch.nn_module {
  torch.slot "b1", %bool_true : !basicpy.BoolType
  torch.slot "b2", %bool_true : !basicpy.BoolType
} : !torch.nn.Module<"c">
