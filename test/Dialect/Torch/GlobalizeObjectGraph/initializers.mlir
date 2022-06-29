// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// CHECK that multiple nested initialization ops are properly handled.

// CHECK-LABEL:   torch.global_slot @l : !torch.list<list<list<tensor>>> {
// CHECK:           %[[L0:.*]] = torch.prim.ListConstruct  : () -> !torch.list<tensor>
// CHECK:           %[[L1:.*]] = torch.prim.ListConstruct %[[L0]], %[[L0]] : (!torch.list<tensor>, !torch.list<tensor>) -> !torch.list<list<tensor>>
// CHECK:           %[[L2:.*]] = torch.prim.ListConstruct %[[L1]], %[[L1]] : (!torch.list<list<tensor>>, !torch.list<list<tensor>>) -> !torch.list<list<list<tensor>>>
// CHECK:           torch.global_slot.init %[[L2]] : !torch.list<list<list<tensor>>>
// CHECK:         }

torch.class_type @c {
  torch.attr "l" : !torch.list<list<list<tensor>>>
}

%l0 = torch.prim.ListConstruct : () -> !torch.list<tensor>
%l1 = torch.prim.ListConstruct %l0, %l0 : (!torch.list<tensor>, !torch.list<tensor>) -> !torch.list<list<tensor>>
%l2 = torch.prim.ListConstruct %l1, %l1 : (!torch.list<list<tensor>>, !torch.list<list<tensor>>) -> !torch.list<list<list<tensor>>>
torch.nn_module {
  torch.slot "l", %l2 : !torch.list<list<list<tensor>>>
} : !torch.nn.Module<"c">

// -----

torch.class_type @c {
  torch.attr "t1" : !torch.tensor
  torch.attr "t2" : !torch.tensor
}

%t = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
%t_list = torch.prim.ListConstruct %t, %t : (!torch.tensor, !torch.tensor) -> !torch.list<tensor>

// CHECK-LABEL: torch.global_slot @t1 : !torch.tensor {
// CHECK:         %[[T1:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
// CHECK:         torch.global_slot.init %[[T1]] : !torch.tensor
// CHECK:       }

// CHECK-LABEL: torch.global_slot @t2 : !torch.tensor {
// CHECK:         %[[T2:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
// CHECK:         torch.global_slot.init %[[T2]] : !torch.tensor
// CHECK:       }

torch.nn_module {
  torch.slot "t1", %t : !torch.tensor
  torch.slot "t2", %t : !torch.tensor
} : !torch.nn.Module<"c">

// CHECK-LABEL: func.func private @use_slot
// CHECK:         %[[t1:.*]] = torch.global_slot.get @t1 : !torch.tensor
// CHECK:         %[[t2:.*]] = torch.global_slot.get @t2 : !torch.tensor
// CHECK:         %[[tuple:.*]] = torch.prim.TupleConstruct %[[t1]], %[[t2]] : !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
// CHECK:         return %[[tuple]] : !torch.tuple<tensor, tensor>
// CHECK:       }

func.func private @use_slot(%arg0 : !torch.nn.Module<"c">) -> !torch.tuple<tensor, tensor> {
  %t1 = torch.prim.GetAttr %arg0["t1"] : !torch.nn.Module<"c"> -> !torch.tensor
  %t2 = torch.prim.GetAttr %arg0["t2"] : !torch.nn.Module<"c"> -> !torch.tensor
  %tuple = torch.prim.TupleConstruct %t1, %t2 : !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
  return %tuple : !torch.tuple<tensor, tensor>
}

// CHECK-LABEL: func.func private @set_slot
// CHECK-SAME:      (%[[ARG:.*]]: !torch.tensor)
// CHECK:         torch.global_slot.set @t1 = %[[ARG]] : !torch.tensor
// CHECK:         torch.global_slot.set @t2 = %[[ARG]] : !torch.tensor
// CHECK:         return
// CHECK:       }

func.func private @set_slot(%arg0 : !torch.nn.Module<"c">, %arg1 : !torch.tensor) {
  torch.prim.SetAttr %arg0["t1"] = %arg1: !torch.nn.Module<"c">, !torch.tensor
  torch.prim.SetAttr %arg0["t2"] = %arg1: !torch.nn.Module<"c">, !torch.tensor
  return
}
