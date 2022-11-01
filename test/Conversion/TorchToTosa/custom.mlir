// RUN: torch-mlir-opt <%s -convert-torch-to-tosa-custom="custom-ops=torch.aten.softmax.int,torch.aten.rsub.Scalar" -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.softmax.int$cst_dim(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = "tosa.custom"(%[[VAL_1]], %[[VAL_4]], %[[VAL_5]]) {identifier = "torch.aten.softmax.int"} : (tensor<2x3xf32>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[2,3],f32>
// CHECK:         }
func.func @torch.aten.softmax.int$cst_dim(%t: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %dtype = torch.constant.int 1
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %dtype : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
  return %ret : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.rsub.Scalar$basic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 3.123400e+00
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() {value = dense<3.123400e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = "tosa.custom"(%[[VAL_1]], %[[VAL_4]], %[[VAL_5]]) {identifier = "torch.aten.rsub.Scalar"} : (tensor<?x?xf32>, tensor<1xf32>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.rsub.Scalar$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %other = torch.constant.float 3.123400e+00
  %alpha = torch.constant.int 1
  %0 = torch.aten.rsub.Scalar %arg0, %other, %alpha : !torch.vtensor<[?,?],f32>, !torch.float, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}