// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch --split-input-file | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.

// CHECK-LABEL: func.func @test_layer_norm
  func.func @test_layer_norm(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[3,4],f32>, 
     %arg2: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, 
     torch.onnx_meta.opset_version = 17 : si64, 
     torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    //CHECK-SAME: %1 = torch.aten.layer_norm %arg0, %0, %arg1, %arg2, %float9.999990e-06, %false : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4],f32>
    //CHECK-SAME: return %1 : !torch.vtensor<[3,4],f32>
    %0:3 = torch.operator "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>)
    return %0 : !torch.vtensor<[3,4],f32>
  }
