// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Basic case.

// CHECK-LABEL:   torch.global_slot @b : !torch.bool  {
// CHECK:           %[[INIT:.*]] = torch.constant.bool true
// CHECK:           torch.global_slot.init %[[INIT]] : !torch.bool
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @i : i64  {
// CHECK:           %[[INIT:.*]] = torch.constant.int 3 : i64
// CHECK:           torch.global_slot.init %[[INIT]] : i64
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @f : f64  {
// CHECK:           %[[INIT:.*]] = torch.constant.float 4.250000e+01
// CHECK:           torch.global_slot.init %[[INIT]] : f64
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @t : !torch.tensor  {
// CHECK:           %[[T:.*]] = torch.tensor(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
// CHECK:           torch.global_slot.init %[[T]] : !torch.tensor
// CHECK:         }

torch.class_type @c {
  torch.attr "b" : !torch.bool
  torch.attr "i" : i64
  torch.attr "f" : f64
  torch.attr "t" : !torch.tensor
}

%bool_true = torch.constant.bool true
%i = torch.constant.int 3 : i64
%f = torch.constant.float 4.250000e+01
%t = torch.tensor(dense<1.0> : tensor<1xf32>) : !torch.tensor
torch.nn_module {
  torch.slot "b", %bool_true : !torch.bool
  torch.slot "i", %i : i64
  torch.slot "f", %f : f64
  torch.slot "t", %t : !torch.tensor
} : !torch.nn.Module<"c">
