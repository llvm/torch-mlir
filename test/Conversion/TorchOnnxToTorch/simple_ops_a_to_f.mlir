// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.

// CHECK-LABEL: func.func @test_abs
func.func @test_abs(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.abs %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Abs"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_add
func.func @test_add(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %[[INT1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_add_bcast
func.func @test_add_bcast(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %[[INT1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_add_uint8
func.func @test_add_uint8(%arg0: !torch.vtensor<[3,4,5],ui8>, %arg1: !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %[[INT1]] : !torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>, !torch.int -> !torch.vtensor<[3,4,5],ui8>
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8>
  return %0 : !torch.vtensor<[3,4,5],ui8>
}

// CHECK-LABEL: @test_and_bcast3v1d
func.func @test_and_bcast3v1d(%arg0: !torch.vtensor<[3,4,5],i1>, %arg1: !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_and %arg0, %arg1 : !torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.And"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// CHECK-LABEL: @test_argmax_default_axis_example
func.func @test_argmax_default_axis_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 0
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmax %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,2],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64>
  return %0 : !torch.vtensor<[1,2],si64>
}

// CHECK-LABEL: @test_argmax_negative_axis_keepdims_example
func.func @test_argmax_negative_axis_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmax %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,1],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = -1 : si64, torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64>
  return %0 : !torch.vtensor<[2,1],si64>
}

// CHECK-LABEL: @test_argmax_no_keepdims_example
func.func @test_argmax_no_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool false
  // CHECK: torch.aten.argmax %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// CHECK-LABEL: @test_argmin_default_axis_example
func.func @test_argmin_default_axis_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 0
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmin %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,2],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64>
  return %0 : !torch.vtensor<[1,2],si64>
}

// CHECK-LABEL: @test_argmin_negative_axis_keepdims_example
func.func @test_argmin_negative_axis_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmin %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,1],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.axis = -1 : si64, torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64>
  return %0 : !torch.vtensor<[2,1],si64>
}

// CHECK-LABEL: @test_argmin_no_keepdims_example
func.func @test_argmin_no_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool false
  // CHECK: torch.aten.argmin %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}
