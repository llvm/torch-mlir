
// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// CHECK-LABEL:   torch.global_slot "private" @slot0 : !torch.tensor {
// CHECK:           %[[VAL_0:.*]] = torch.tensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.tensor<[],f32>
// CHECK:           torch.global_slot.init %[[VAL_0]] : !torch.tensor<[],f32>
// CHECK:         }

// CHECK-LABEL:   torch.global_slot "private" @slot1 : !torch.list<tensor> {
// CHECK:           %[[VAL_0:.*]] = torch.global_slot.get @slot0 : !torch.tensor<[],f32>
// CHECK:           %[[VAL_1:.*]] = torch.prim.ListConstruct %[[VAL_0]] : (!torch.tensor<[],f32>) -> !torch.list<tensor>
// CHECK:           torch.global_slot.init %[[VAL_1]] : !torch.list<tensor>
// CHECK:         }

// CHECK-LABEL:   func.func @forward() -> !torch.tuple<tensor, list<tensor>> {
// CHECK:           %[[VAL_0:.*]] = torch.global_slot.get @slot0 : !torch.tensor
// CHECK:           %[[VAL_1:.*]] = torch.global_slot.get @slot1 : !torch.list<tensor>
// CHECK:           %[[VAL_2:.*]] = torch.prim.TupleConstruct %[[VAL_0]], %[[VAL_1]] : !torch.tensor, !torch.list<tensor> -> !torch.tuple<tensor, list<tensor>>
// CHECK:           return %[[VAL_2]] : !torch.tuple<tensor, list<tensor>>

func.func private @__torch__.Model.forward(%arg0: !torch.nn.Module<"__torch__.Model">) -> !torch.tuple<tensor, list<tensor>> {
  %3 = torch.prim.GetAttr %arg0["slot0"] : !torch.nn.Module<"__torch__.Model"> -> !torch.tensor
  %4 = torch.prim.GetAttr %arg0["slot1"] : !torch.nn.Module<"__torch__.Model"> -> !torch.list<tensor>
  %5 = torch.prim.TupleConstruct %3, %4 : !torch.tensor, !torch.list<tensor> -> !torch.tuple<tensor, list<tensor>>
  return %5 : !torch.tuple<tensor, list<tensor>>
}
torch.class_type @__torch__.Model {
  torch.attr private "slot0" : !torch.tensor
  torch.attr private "slot1" : !torch.list<tensor>
  torch.method "forward", @__torch__.Model.forward
}
%0 = torch.tensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.tensor<[],f32>
%1 = torch.prim.ListConstruct %0 : (!torch.tensor<[],f32>) -> !torch.list<tensor>
%2 = torch.nn_module {
  torch.slot "slot0", %0 : !torch.tensor<[],f32>
  torch.slot "slot1", %1 : !torch.list<tensor>
} : !torch.nn.Module<"__torch__.Model">
