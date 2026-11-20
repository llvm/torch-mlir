// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torch-backend-to-tosa-backend-pipeline)' -split-input-file | FileCheck %s

// End-to-end check that `mlir.user` annotations survive the full TOSA backend
// pipeline. See docs/user_annotations.md.

// Function argument annotations survive the backend type conversion, and an
// annotated op whose result only feeds other converted ops keeps its
// annotation while the downstream op does not inherit it.
// CHECK-LABEL:  func.func @linear_relu_user_attrs(
// CHECK-SAME:       %{{.*}}: tensor<1x4xf32> {mlir.user = {role = "input"}}
// CHECK:          tosa.add {{.*}} {mlir.user = [{lo = -1.000000e+00 : f64}]}
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

// The matmul lowering ends in a reshape back to the original rank; the
// annotation lands on that reshape, which produces the replacement value.
// CHECK-LABEL:  func.func @mm_user_attrs(
// CHECK:          tosa.matmul
// CHECK:          tosa.reshape {{.*}} {mlir.user = [{tag = "mm_tag"}]}
func.func @mm_user_attrs(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mm %arg0, %arg1 {mlir.user = [{tag = "mm_tag"}]} : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}
