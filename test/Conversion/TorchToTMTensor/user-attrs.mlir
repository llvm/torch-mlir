// RUN: torch-mlir-opt <%s -convert-torch-to-tmtensor -split-input-file | FileCheck %s

// Verify that `mlir.user` annotations are forwarded from Torch ops onto the
// ops producing their replacement values when lowering to TMTensor.

// CHECK-LABEL:  func.func @sort_user_attrs(
// CHECK:          tm_tensor.sort {mlir.user = [{tag = "sort_values"}, {tag = "sort_indices"}]} dimension(1)
func.func @sort_user_attrs(%arg0: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>) {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.sort %arg0, %int-1, %false {mlir.user = [{tag = "sort_values"}, {tag = "sort_indices"}]} : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
  return %values, %indices : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
}

// -----

// The cumsum lowering returns a `tensor.cast` of the scan result; the
// annotation is forwarded through the cast onto `tm_tensor.scan`.
// CHECK-LABEL:  func.func @cumsum_user_attrs(
// CHECK:          tm_tensor.scan dimension(1) inclusive(true) {mlir.user = [{tag = "cumsum_tag"}]}
func.func @cumsum_user_attrs(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.aten.cumsum %arg0, %int1, %none {mlir.user = [{tag = "cumsum_tag"}]} : !torch.vtensor<[3,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:  func.func @cumprod_user_attrs(
// CHECK:          tm_tensor.scan dimension(1) inclusive(true) {mlir.user = [{tag = "cumprod_tag"}]}
func.func @cumprod_user_attrs(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.aten.cumprod %arg0, %int1, %none {mlir.user = [{tag = "cumprod_tag"}]} : !torch.vtensor<[3,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:  func.func @scatter_add_user_attrs(
// CHECK:          tm_tensor.scatter <dimension_map = [0, 1]> {mlir.user = [{tag = "scatter_add_tag"}]}
func.func @scatter_add_user_attrs(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[3,4],si64>, %arg2: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.scatter_add %arg0, %int1, %arg1, %arg2 {mlir.user = [{tag = "scatter_add_tag"}]} : !torch.vtensor<[3,4],f32>, !torch.int, !torch.vtensor<[3,4],si64>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}
