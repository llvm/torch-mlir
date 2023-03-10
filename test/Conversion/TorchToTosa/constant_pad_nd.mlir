// RUN: torch-mlir-opt <%s -convert-torch-to-tosa -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd_2d(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[1,20,4,4],f32>) -> !torch.vtensor<[1,20,9,5],f32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"()
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = "tosa.pad"
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<1x20x9x5xf32> -> !torch.vtensor<[1,20,9,5],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1,20,9,5],f32>
func.func @torch.aten.constant_pad_nd_2d(%arg0: !torch.vtensor<[1,20,4,4],f32>) -> !torch.vtensor<[1,20,9,5],f32> {
  %float-Inf = torch.constant.float 0xFFF0000000000000
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.constant_pad_nd %arg0, %1, %float-Inf : !torch.vtensor<[1,20,4,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,20,9,5],f32>
  return %2 : !torch.vtensor<[1,20,9,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.constant_pad_nd_1d(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[1,20,4,4],f32>) -> !torch.vtensor<[1,20,4,5],f32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"()
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = "tosa.pad"
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<1x20x4x5xf32> -> !torch.vtensor<[1,20,4,5],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1,20,4,5],f32>
func.func @torch.aten.constant_pad_nd_1d(%arg0: !torch.vtensor<[1,20,4,4],f32>) -> !torch.vtensor<[1,20,4,5],f32> {
  %float-Inf = torch.constant.float 0xFFF0000000000000
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %1 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.constant_pad_nd %arg0, %1, %float-Inf : !torch.vtensor<[1,20,4,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,20,4,5],f32>
  return %2 : !torch.vtensor<[1,20,4,5],f32>
}
