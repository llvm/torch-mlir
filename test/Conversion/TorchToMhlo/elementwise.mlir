// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.gelu(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = "chlo.constant_like"(%[[T0]]) {value = 1.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = "chlo.constant_like"(%[[T0]]) {value = 2.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = "chlo.constant_like"(%[[T0]]) {value = 5.000000e-01 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = mhlo.rsqrt %[[T2]] : tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = mhlo.multiply %[[T0]], %[[T4]] : tensor<?x?xf32>
// CHECK:         %[[T6:.*]] = chlo.erf %[[T5]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[T7:.*]] = mhlo.add %[[T6]], %[[T1]] : tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = mhlo.multiply %[[T7]], %[[T3]] : tensor<?x?xf32>
// CHECK:         %[[T9:.*]] = mhlo.multiply %[[T0]], %[[T8]] : tensor<?x?xf32>
// CHECK:         %[[T10:.*]] = torch_c.from_builtin_tensor %[[T9]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T10]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.gelu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %str = torch.constant.str "none"
    %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[?,?],f32>, !torch.str -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}