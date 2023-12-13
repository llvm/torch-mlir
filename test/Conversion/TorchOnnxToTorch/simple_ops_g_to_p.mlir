// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.

// CHECK-LABEL: @test_matmul_2d
func.func @test_matmul_2d(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,3],f32> -> !torch.vtensor<[3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,3],f32>
  return %0 : !torch.vtensor<[3,3],f32>
}

// CHECK-LABEL: @test_matmul_3d
func.func @test_matmul_3d(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,4,3],f32> -> !torch.vtensor<[2,3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,3,3],f32>
  return %0 : !torch.vtensor<[2,3,3],f32>
}

// CHECK-LABEL: @test_matmul_4d
func.func @test_matmul_4d(%arg0: !torch.vtensor<[1,2,3,4],f32>, %arg1: !torch.vtensor<[1,2,4,3],f32>) -> !torch.vtensor<[1,2,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,4,3],f32> -> !torch.vtensor<[1,2,3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,4,3],f32>) -> !torch.vtensor<[1,2,3,3],f32>
  return %0 : !torch.vtensor<[1,2,3,3],f32>
}

// CHECK-LABEL: func.func @test_less_or_equal
func.func @test_less_or_equal(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.le.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.LessOrEqual"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}