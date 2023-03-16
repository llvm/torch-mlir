// RUN: torch-mlir-opt <%s -convert-torch-to-tcp -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @torch.aten.cat
//   CHECK-SAME:   %[[ARG0:.+]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.+]]: !torch.vtensor<[?,?],f32>
//        CHECK:   %[[V1:.+]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
//        CHECK:   %[[V2:.+]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
//        CHECK:   %[[V3:.+]] = tcp.concat %[[V1]], %[[V2]] attributes {axis = 0 : i64} : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
func.func @torch.aten.cat(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}
