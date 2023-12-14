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

// CHECK-LABEL: func.func @test_sinh
func.func @test_sinh_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 9 : si64} {
  // CHECK: torch.aten.sinh %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sinh"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}
