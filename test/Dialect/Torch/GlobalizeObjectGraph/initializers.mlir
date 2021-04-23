// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// CHECK that multiple nested initialization ops are properly handled.

// CHECK-LABEL:   torch.global_slot @l : !basicpy.ListType  {
// CHECK:           %[[L0:.*]] = basicpy.build_list  : () -> !basicpy.ListType
// CHECK:           %[[L1:.*]] = basicpy.build_list %[[L0]], %[[L0]] : (!basicpy.ListType, !basicpy.ListType) -> !basicpy.ListType
// CHECK:           %[[L2:.*]] = basicpy.build_list %[[L1]], %[[L1]] : (!basicpy.ListType, !basicpy.ListType) -> !basicpy.ListType
// CHECK:           torch.global_slot.init %[[L2]] : !basicpy.ListType
// CHECK:         }

torch.class_type @c {
  torch.attr "l" : !basicpy.ListType
}

%l0 = basicpy.build_list : () -> !basicpy.ListType
%l1 = basicpy.build_list %l0, %l0 : (!basicpy.ListType, !basicpy.ListType) -> !basicpy.ListType
%l2 = basicpy.build_list %l1, %l1 : (!basicpy.ListType, !basicpy.ListType) -> !basicpy.ListType
torch.nn_module {
  torch.slot "l", %l2 : !basicpy.ListType
} : !torch.nn.Module<"c">
