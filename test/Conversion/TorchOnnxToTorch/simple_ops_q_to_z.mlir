// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.

// CHECK-LABEL: func.func @test_selu
func.func @test_selu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 6 : si64} {
  // CHECK-DAG: %[[F1:.+]] = torch.constant.float 1
  // CHECK-DAG: %[[F2:.+]] = torch.constant.float 2
  // CHECK-DAG: %[[F3:.+]] = torch.constant.float 3
  // CHECK: %[[ELU:.+]] = torch.aten.elu %arg0, %[[F2]], %[[F3]], %[[F1]]
  %0 = torch.operator "onnx.Selu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32, torch.onnx.gamma = 3.000000e+00 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_transpose_default
func.func @test_transpose_default(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,3,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[I2:.+]] = torch.constant.int 2
  // CHECK: %[[TRANSPOSE:.+]] = torch.aten.transpose.int %arg0, %[[I0]], %[[I2]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,3,2],f32>
  %0 = torch.operator "onnx.Transpose"(%arg0) : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,3,2],f32>

  // CHECK: return %[[TRANSPOSE]]
  return %0 : !torch.vtensor<[4,3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_transpose_all_permutations_4
func.func @test_transpose_all_permutations_4(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,2,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG:  %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG:  %[[I2:.+]] = torch.constant.int 2
  // CHECK:  %[[TRANSPOSE0:.+]] = torch.aten.transpose.int %arg0, %[[I0]], %[[I2]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,3,2],f32>
  // CHECK-DAG:  %[[I1:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[I2:.+]] = torch.constant.int 2
  // CHECK:  %[[TRANSPOSE1:.+]] = torch.aten.transpose.int %[[TRANSPOSE0]], %[[I1]], %[[I2]] : !torch.vtensor<[4,3,2],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,2,3],f32>
  %0 = torch.operator "onnx.Transpose"(%arg0) {torch.onnx.perm = [2 : si64, 0 : si64, 1 : si64]} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,2,3],f32>

  // CHECK:  return %[[TRANSPOSE1]]
  return %0 : !torch.vtensor<[4,2,3],f32>
}
