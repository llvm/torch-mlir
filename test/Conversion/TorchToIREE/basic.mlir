
// RUN: npcomp-opt <%s -convert-torch-to-iree -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward(
// CHECK-SAME:                          %[[ARG_TORCH:.*]]: !torch.float) -> !torch.list<!torch.float> {
// CHECK:           %[[ARG:.*]] = torch_c.to_f64 %[[ARG_TORCH]]
// CHECK:           %[[ALSO_ARG:.*]] = torch_c.to_f64 %[[ARG_TORCH]]
// CHECK:           %[[C2:.*]] = constant 2 : index
// CHECK:           %[[LIST:.*]] = iree.list.create %[[C2]] : !iree.list<f64>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           iree.list.set %[[LIST]][%[[C0]]], %[[ARG]] : !iree.list<f64>, f64
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           iree.list.set %[[LIST]][%[[C1]]], %[[ALSO_ARG]] : !iree.list<f64>, f64
// CHECK:           %[[LIST_TORCH:.*]] = torch_c.from_iree_list %[[LIST]] : !iree.list<f64> -> !torch.list<!torch.float>
// CHECK:           return %[[LIST_TORCH]] : !torch.list<!torch.float>
builtin.func @forward(%arg0: !torch.float) -> !torch.list<!torch.float> {
  %0 = torch.prim.ListConstruct %arg0, %arg0 : (!torch.float, !torch.float) -> !torch.list<!torch.float>
  return %0 : !torch.list<!torch.float>
}
