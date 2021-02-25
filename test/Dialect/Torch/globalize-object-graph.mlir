// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Basic case.

// CHECK-LABEL:   torch.global_slot @b : !basicpy.BoolType  {
// CHECK:           %[[INIT:.*]] = basicpy.bool_constant true
// CHECK:           torch.global_slot.init %[[INIT]] : !basicpy.BoolType
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @i : i64  {
// CHECK:           %[[INIT:.*]] = basicpy.numeric_constant 3 : i64
// CHECK:           torch.global_slot.init %[[INIT]] : i64
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @f : f64  {
// CHECK:           %[[INIT:.*]] = basicpy.numeric_constant 4.250000e+01 : f64
// CHECK:           torch.global_slot.init %[[INIT]] : f64
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @a : !numpy.ndarray<*:!numpy.any_dtype>  {
// CHECK:           %[[C:.*]] = constant dense<1.000000e+00> : tensor<1xf32>
// CHECK:           %[[A:.*]] = numpy.create_array_from_tensor %[[C]] : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           torch.global_slot.init %[[A]] : !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:         }

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
