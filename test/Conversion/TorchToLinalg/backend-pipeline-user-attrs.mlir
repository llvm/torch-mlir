// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)' -split-input-file | FileCheck %s

// End-to-end check that `mlir.user` annotations survive the full
// linalg-on-tensors backend pipeline. See docs/user_annotations.md.

// Function argument annotations survive the backend type conversion, and an
// annotated op whose result only feeds other converted ops keeps its
// annotation while the downstream op does not inherit it.
// CHECK-LABEL:  func.func @linear_relu_user_attrs(
// CHECK-SAME:       %{{.*}}: tensor<1x4xf32> {mlir.user = {role = "input"}}
// CHECK:          linalg.matmul
// CHECK-NOT:      mlir.user
// CHECK:          linalg.generic {{.*}} attrs =  {mlir.user = [{lo = -1.000000e+00 : f64}]}
// CHECK:            arith.addf
// CHECK-NOT:      mlir.user
// CHECK:          return
func.func @linear_relu_user_attrs(%arg0: !torch.vtensor<[1,4],f32> {mlir.user = {role = "input"}}, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[1,4],f32>
  %1 = torch.aten.add.Tensor %0, %arg2, %int1 {mlir.user = [{lo = -1.0 : f64}]} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[1,4],f32>
  %2 = torch.aten.relu %1 : !torch.vtensor<[1,4],f32> -> !torch.vtensor<[1,4],f32>
  return %2 : !torch.vtensor<[1,4],f32>
}

// -----

// Multi-result op lowered through tm_tensor: each result keeps its own
// dictionary.
// CHECK-LABEL:  func.func @sort_user_attrs(
// CHECK:          tm_tensor.sort {mlir.user = [{tag = "values"}, {tag = "indices"}]}
func.func @sort_user_attrs(%arg0: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>) {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.sort %arg0, %int-1, %false {mlir.user = [{tag = "values"}, {tag = "indices"}]} : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
  return %values, %indices : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
}

// -----

// CHECK-LABEL:  func.func @cumsum_user_attrs(
// CHECK:          tm_tensor.scan {{.*}} {mlir.user = [{tag = "cumsum"}]}
func.func @cumsum_user_attrs(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.aten.cumsum %arg0, %int1, %none {mlir.user = [{tag = "cumsum"}]} : !torch.vtensor<[3,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}
