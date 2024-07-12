// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s
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

// -----

// CHECK-LABEL: @test_add
func.func @test_add(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %[[INT1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_add_bcast
func.func @test_add_bcast(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %[[INT1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_add_uint8
func.func @test_add_uint8(%arg0: !torch.vtensor<[3,4,5],ui8>, %arg1: !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %[[INT1]] : !torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>, !torch.int -> !torch.vtensor<[3,4,5],ui8>
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8>
  return %0 : !torch.vtensor<[3,4,5],ui8>
}

// -----

// CHECK-LABEL: @test_and_bcast3v1d
func.func @test_and_bcast3v1d(%arg0: !torch.vtensor<[3,4,5],i1>, %arg1: !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_and %arg0, %arg1 : !torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.And"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: @test_argmax_default_axis_example
func.func @test_argmax_default_axis_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 0
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmax %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,2],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64>
  return %0 : !torch.vtensor<[1,2],si64>
}

// -----

// CHECK-LABEL: @test_argmax_negative_axis_keepdims_example
func.func @test_argmax_negative_axis_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmax %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,1],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = -1 : si64, torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64>
  return %0 : !torch.vtensor<[2,1],si64>
}

// -----

// CHECK-LABEL: @test_argmax_negative_axis_keepdims_random_select_last_index
func.func @test_argmax_negative_axis_keepdims_random_select_last_index(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,1],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[C2_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %arg0, %[[DIMS]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,3,4],f32>
  // CHECK: %[[ARGMAX:.*]] = torch.aten.argmax %[[FLIP]], %[[C2]], %[[TRUE]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,3,1],si64>
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[ARGMAX]], %[[C3]], %[[C1]] : !torch.vtensor<[2,3,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,3,1],si64>
  // CHECK: %[[ABS:.*]] = torch.aten.abs %[[SUB]] : !torch.vtensor<[2,3,1],si64> -> !torch.vtensor<[2,3,1],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = -1 : si64, torch.onnx.keepdims = 1 : si64, torch.onnx.select_last_index = 1 : si64} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,1],si64>
  return %0 : !torch.vtensor<[2,3,1],si64>
}

// -----

// CHECK-LABEL: @test_argmax_no_keepdims_example
func.func @test_argmax_no_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool false
  // CHECK: torch.aten.argmax %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// -----

// CHECK-LABEL: @test_argmax_no_keepdims_random_select_last_index
func.func @test_argmax_no_keepdims_random_select_last_index(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,4],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[C1_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %arg0, %[[DIMS]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,3,4],f32>
  // CHECK: %[[ARGMAX:.*]] = torch.aten.argmax %[[FLIP]], %[[C1]], %[[FALSE]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,4],si64>
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[ARGMAX]], %[[C2]], %[[C1_1]] : !torch.vtensor<[2,4],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4],si64>
  // CHECK: %[[ABS:.*]] = torch.aten.abs %[[SUB]] : !torch.vtensor<[2,4],si64> -> !torch.vtensor<[2,4],si64>
  %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64, torch.onnx.select_last_index = 1 : si64} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,4],si64>
  return %0 : !torch.vtensor<[2,4],si64>
}

// -----

// CHECK-LABEL: @test_argmin_default_axis_example
func.func @test_argmin_default_axis_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 0
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmin %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,2],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64>
  return %0 : !torch.vtensor<[1,2],si64>
}

// -----

// CHECK-LABEL: @test_argmin_negative_axis_keepdims_example
func.func @test_argmin_negative_axis_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool true
  // CHECK: torch.aten.argmin %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,1],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.axis = -1 : si64, torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,1],si64>
  return %0 : !torch.vtensor<[2,1],si64>
}

// -----

// CHECK-LABEL: @test_argmin_negative_axis_keepdims_random_select_last_index
func.func @test_argmin_negative_axis_keepdims_random_select_last_index(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,1],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[C2_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %arg0, %[[DIMS]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,3,4],f32>
  // CHECK: %[[ARGMIN:.*]] = torch.aten.argmin %[[FLIP]], %[[C2]], %[[TRUE]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,3,1],si64>
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[ARGMIN]], %[[C3]], %[[C1]] : !torch.vtensor<[2,3,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,3,1],si64>
  // CHECK: %[[ABS:.*]] = torch.aten.abs %[[SUB]] : !torch.vtensor<[2,3,1],si64> -> !torch.vtensor<[2,3,1],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.axis = -1 : si64, torch.onnx.keepdims = 1 : si64, torch.onnx.select_last_index = 1 : si64} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,1],si64>
  return %0 : !torch.vtensor<[2,3,1],si64>
}

// -----

// CHECK-LABEL: @test_argmin_no_keepdims_example
func.func @test_argmin_no_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 1
  // CHECK: %[[BOOL:.*]] = torch.constant.bool false
  // CHECK: torch.aten.argmin %arg0, %[[INT]], %[[BOOL]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// -----

// CHECK-LABEL: @test_argmin_no_keepdims_example_select_last_index
func.func @test_argmin_no_keepdims_example_select_last_index(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[C1_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %arg0, %[[DIMS]] : !torch.vtensor<[2,2],f32>, !torch.list<int> -> !torch.vtensor<[2,2],f32>
  // CHECK: %[[ARGMIN:.*]] = torch.aten.argmin %[[FLIP]], %[[C1]], %[[FALSE]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2],si64>
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[SUB:.*]] = torch.aten.sub.Scalar %[[ARGMIN]], %[[C1_1]], %[[C1_2]] : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[2],si64>
  // CHECK: %[[ABS:.*]] = torch.aten.abs %[[SUB]] : !torch.vtensor<[2],si64> -> !torch.vtensor<[2],si64>
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64, torch.onnx.select_last_index = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// -----

// CHECK-LABEL: @test_atan
func.func @test_atan(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.atan %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Atan"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_atanh
func.func @test_atanh(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.atanh %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Atanh"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_acos
func.func @test_acos(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.acos %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Acos"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_bernoulli
func.func @test_bernoulli(%arg0: !torch.vtensor<[10],f64>) -> !torch.vtensor<[10],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %0 = torch.aten.bernoulli %arg0, %[[NONE]] : !torch.vtensor<[10],f64>, !torch.none -> !torch.vtensor<[10],f64>
  %0 = torch.operator "onnx.Bernoulli"(%arg0) : (!torch.vtensor<[10],f64>) -> !torch.vtensor<[10],f64>
  return %0 : !torch.vtensor<[10],f64>
}

// -----

// CHECK-LABEL: @test_bernoulli_double
func.func @test_bernoulli_double(%arg0: !torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[BERNOULLI:.*]] = torch.aten.bernoulli %arg0, %[[NONE]] : !torch.vtensor<[10],f32>, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 7
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %[[BERNOULLI]], %[[DTYPE]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f64>
  %0 = torch.operator "onnx.Bernoulli"(%arg0) {torch.onnx.dtype = 11 : si64} : (!torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f64>
  return %0 : !torch.vtensor<[10],f64>
}

// -----

// CHECK-LABEL: @test_bitshift_left_uint8
func.func @test_bitshift_left_uint8(%arg0: !torch.vtensor<[3],ui8>, %arg1: !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8> -> !torch.vtensor<[3],ui8>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8>
  return %0 : !torch.vtensor<[3],ui8>
}

// -----

// CHECK-LABEL: @test_bitshift_left_uint16
func.func @test_bitshift_left_uint16(%arg0: !torch.vtensor<[3],ui16>, %arg1: !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16> -> !torch.vtensor<[3],ui16>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16>
  return %0 : !torch.vtensor<[3],ui16>
}

// -----

// CHECK-LABEL: @test_bitshift_left_uint32
func.func @test_bitshift_left_uint32(%arg0: !torch.vtensor<[3],ui32>, %arg1: !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32> -> !torch.vtensor<[3],ui32>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32>
  return %0 : !torch.vtensor<[3],ui32>
}

// -----

// CHECK-LABEL: @test_bitshift_left_uint64
func.func @test_bitshift_left_uint64(%arg0: !torch.vtensor<[3],ui64>, %arg1: !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64> -> !torch.vtensor<[3],ui64>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64>
  return %0 : !torch.vtensor<[3],ui64>
}

// -----

// CHECK-LABEL: @test_bitshift_right_uint8
func.func @test_bitshift_right_uint8(%arg0: !torch.vtensor<[3],ui8>, %arg1: !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8> -> !torch.vtensor<[3],ui8>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8>
  return %0 : !torch.vtensor<[3],ui8>
}

// -----

// CHECK-LABEL: @test_bitshift_right_uint16
func.func @test_bitshift_right_uint16(%arg0: !torch.vtensor<[3],ui16>, %arg1: !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16> -> !torch.vtensor<[3],ui16>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16>
  return %0 : !torch.vtensor<[3],ui16>
}

// -----

// CHECK-LABEL: @test_bitshift_right_uint32
func.func @test_bitshift_right_uint32(%arg0: !torch.vtensor<[3],ui32>, %arg1: !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32> -> !torch.vtensor<[3],ui32>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32>
  return %0 : !torch.vtensor<[3],ui32>
}

// -----

// CHECK-LABEL: @test_bitshift_right_uint64
func.func @test_bitshift_right_uint64(%arg0: !torch.vtensor<[3],ui64>, %arg1: !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64> -> !torch.vtensor<[3],ui64>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64>
  return %0 : !torch.vtensor<[3],ui64>
}

// -----

// CHECK-LABEL: @test_bitwise_and_i16_3d
func.func @test_bitwise_and_i16_3d(%arg0: !torch.vtensor<[3,4,5],si16>, %arg1: !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16> -> !torch.vtensor<[3,4,5],si16>
  %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16>
  return %0 : !torch.vtensor<[3,4,5],si16>
}

// -----

// CHECK-LABEL: @test_bitwise_and_i32_2d
func.func @test_bitwise_and_i32_2d(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// -----

// CHECK-LABEL: @test_bitwise_and_ui8_bcast_4v3d
func.func @test_bitwise_and_ui8_bcast_4v3d(%arg0: !torch.vtensor<[3,4,5,6],ui8>, %arg1: !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// -----

// CHECK-LABEL: @test_bitwise_or_i16_4d
func.func @test_bitwise_or_i16_4d(%arg0: !torch.vtensor<[3,4,5,6],si8>, %arg1: !torch.vtensor<[3,4,5,6],si8>) -> !torch.vtensor<[3,4,5,6],si8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],si8>, !torch.vtensor<[3,4,5,6],si8> -> !torch.vtensor<[3,4,5,6],si8>
  %0 = torch.operator "onnx.BitwiseOr"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],si8>, !torch.vtensor<[3,4,5,6],si8>) -> !torch.vtensor<[3,4,5,6],si8>
  return %0 : !torch.vtensor<[3,4,5,6],si8>
}

// -----

// CHECK-LABEL: @test_bitwise_or_i32_2d
func.func @test_bitwise_or_i32_2d(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseOr"(%arg0, %arg1) : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// -----

// CHECK-LABEL: @test_bitwise_or_ui8_bcast_4v3d
func.func @test_bitwise_or_ui8_bcast_4v3d(%arg0: !torch.vtensor<[3,4,5,6],ui8>, %arg1: !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseOr"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// -----

// CHECK-LABEL: @test_bitwise_not_2d
func.func @test_bitwise_not_2d(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_not %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseNot"(%arg0) : (!torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// -----

// CHECK-LABEL: @test_bitwise_not_4d
func.func @test_bitwise_not_4d(%arg0: !torch.vtensor<[3,4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_not %arg0 : !torch.vtensor<[3,4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseNot"(%arg0) : (!torch.vtensor<[3,4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// -----

// CHECK-LABEL: @test_bitwise_xor_i16_3d
func.func @test_bitwise_xor_i16_3d(%arg0: !torch.vtensor<[3,4,5],si16>, %arg1: !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_xor.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16> -> !torch.vtensor<[3,4,5],si16>
  %0 = torch.operator "onnx.BitwiseXor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16>
  return %0 : !torch.vtensor<[3,4,5],si16>
}

// -----

// CHECK-LABEL: @test_bitwise_xor_i32_2d
func.func @test_bitwise_xor_i32_2d(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_xor.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseXor"(%arg0, %arg1) : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// -----

// CHECK-LABEL: @test_bitwise_xor_ui8_bcast_4v3d
func.func @test_bitwise_xor_ui8_bcast_4v3d(%arg0: !torch.vtensor<[3,4,5,6],ui8>, %arg1: !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_xor.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseXor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// -----

// CHECK-LABEL: @test_cast_BFLOAT16_to_FLOAT
func.func @test_cast_BFLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],bf16>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],bf16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],bf16>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_cast_DOUBLE_to_FLOAT
func.func @test_cast_DOUBLE_to_FLOAT(%arg0: !torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_cast_DOUBLE_to_FLOAT16
func.func @test_cast_DOUBLE_to_FLOAT16(%arg0: !torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 5
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f16>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 10 : si64} : (!torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f16>
  return %0 : !torch.vtensor<[3,4],f16>
}

// -----

// CHECK-LABEL: @test_cast_FLOAT_to_BFLOAT16
func.func @test_cast_FLOAT_to_BFLOAT16(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],bf16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 15
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],bf16>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 16 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],bf16>
  return %0 : !torch.vtensor<[3,4],bf16>
}

// -----

// CHECK-LABEL: @test_cast_FLOAT_to_DOUBLE
func.func @test_cast_FLOAT_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 7
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f64>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f64>
  return %0 : !torch.vtensor<[3,4],f64>
}

// -----

// CHECK-LABEL: @test_cast_FLOAT_to_FLOAT16
func.func @test_cast_FLOAT_to_FLOAT16(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 5
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f16>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 10 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f16>
  return %0 : !torch.vtensor<[3,4],f16>
}

// -----

// CHECK-LABEL: @test_cast_FLOAT16_to_DOUBLE
func.func @test_cast_FLOAT16_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 7
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f64>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f64>
  return %0 : !torch.vtensor<[3,4],f64>
}

// -----

// CHECK-LABEL: @test_cast_FLOAT_to_BOOL
func.func @test_cast_FLOAT_to_BOOL(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 11
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],i1>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 9 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],i1>
  return %0 : !torch.vtensor<[3,4],i1>
}

// -----

// CHECK-LABEL: @test_cast_FLOAT16_to_FLOAT
func.func @test_cast_FLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_castlike_BFLOAT16_to_FLOAT
func.func @test_castlike_BFLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],bf16>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],bf16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],bf16>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_castlike_DOUBLE_to_FLOAT
func.func @test_castlike_DOUBLE_to_FLOAT(%arg0: !torch.vtensor<[3,4],f64>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f64>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_castlike_FLOAT_to_DOUBLE
func.func @test_castlike_FLOAT_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],f64>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 7
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f64>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],f64>) -> !torch.vtensor<[3,4],f64>
  return %0 : !torch.vtensor<[3,4],f64>
}

// -----

// CHECK-LABEL: @test_castlike_FLOAT16_to_FLOAT
func.func @test_castlike_FLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],f16>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f16>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_ceil_example
func.func @test_ceil_example(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.ceil %arg0 : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  %0 = torch.operator "onnx.Ceil"(%arg0) : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: @test_ceil
func.func @test_ceil(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.ceil %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Ceil"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_clip_default_int8_min
func.func @test_clip_default_int8_min(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp_min.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[3,4,5],si8>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8>
  return %0 : !torch.vtensor<[3,4,5],si8>
}

// -----

// CHECK-LABEL: @test_clip_default_int8_max
func.func @test_clip_default_int8_max(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  // CHECK: torch.aten.clamp.Tensor %arg0, %none, %arg1 : !torch.vtensor<[3,4,5],si8>, !torch.none, !torch.vtensor<[],si8> -> !torch.vtensor<[3,4,5],si8>
  %0 = torch.operator "onnx.Clip"(%arg0, %none, %arg1) : (!torch.vtensor<[3,4,5],si8>, !torch.none, !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8>
  return %0 : !torch.vtensor<[3,4,5],si8>
}

// -----

// CHECK-LABEL: @test_clip_default_min
func.func @test_clip_default_min(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp_min.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_clip_example
func.func @test_clip_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_clip
func.func @test_clip(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

module {
  func.func @test_clip_attrs(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64} {
    %none = torch.constant.none

    // CHECK: %[[MIN:.+]]  = torch.vtensor.literal(dense<-5.000000e-01> : tensor<3x4xf32>) : !torch.vtensor<[3,4],f32>
    // CHECK: %[[MAX:.+]] = torch.vtensor.literal(dense<5.000000e-01> : tensor<3x4xf32>) : !torch.vtensor<[3,4],f32>
    // CHECK: %[[CLAMP:.+]] = torch.aten.clamp.Tensor %arg0, %[[MIN]], %[[MAX]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    %0 = torch.operator "onnx.Clip"(%arg0) {torch.onnx.max = 5.000000e-01 : f32, torch.onnx.min = -5.000000e-01 : f32} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
    return %0 : !torch.vtensor<[3,4],f32>
  }
}

// -----

// CHECK-LABEL: @test_cos_example
func.func @test_cos_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.cos %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Cos"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_cos
func.func @test_cos(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.cos %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Cos"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_cosh_example
func.func @test_cosh_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.cosh %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Cosh"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_cosh
func.func @test_cosh(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.cosh %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Cosh"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_acosh_example
func.func @test_acosh_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.acosh %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Acosh"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_acosh
func.func @test_acosh(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.acosh %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Acosh"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_asin_example
func.func @test_asin_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.asin %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Asin"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_asin
func.func @test_asin(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.asin %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Asin"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_asinh_example
func.func @test_asinh_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.asinh %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Asinh"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_asinh
func.func @test_asinh(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.asinh %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Asinh"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_deform_conv
func.func @test_deform_conv(%arg0: !torch.vtensor<[1,1,7,6],f32>, %arg1: !torch.vtensor<[1,8,6,5],f32>, %arg2: !torch.vtensor<[1,1,2,2],f32>, %arg3: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,1,6,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.4.0"} {
  // CHECK: %[[cstOne:.*]] = torch.constant.float 1.000000e+00
  // CHECK: %[[mask:.*]] = torch.aten.full %[[sizeList:.*]], %[[cstOne]]
  // CHECK-SAME: -> !torch.vtensor<[1,4,6,5],f32>
  // CHECK: torch.torchvision.deform_conv2d %arg0, %arg2, %arg1, %[[mask]], %arg3
  // CHECK-SAME: : !torch.vtensor<[1,1,7,6],f32>, !torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,8,6,5],f32>, !torch.vtensor<[1,4,6,5],f32>, !torch.vtensor<[1],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[1,1,6,5],f32>
  %1 = torch.operator "onnx.DeformConv"(%arg0, %arg2, %arg1, %arg3) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.offset_group = 1 : si64, torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,1,7,6],f32>, !torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,8,6,5],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,1,6,5],f32>
  return %1 : !torch.vtensor<[1,1,6,5],f32>
}

// -----

// CHECK-LABEL: @test_dequantizelinear_si8
func.func @test_dequantizelinear_si8(%arg0: !torch.vtensor<[6],si8>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[6],f32>
  // CHECK: %[[SCALE:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[ZP:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],si8> -> !torch.int
  // CHECK: %[[MAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[ZP]]
  // CHECK: %[[DEQ:.+]] = torch.aten.dequantize.self %[[MAKE]]
  // CHECK: return %[[DEQ]]
  return %0 : !torch.vtensor<[6],f32>
}

// -----

// CHECK-LABEL: @test_dequantizelinear_si16
func.func @test_dequantizelinear_si16(%arg0: !torch.vtensor<[6],si16>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si16>) -> !torch.vtensor<[6],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],si16>, !torch.vtensor<[],f32>, !torch.vtensor<[],si16>) -> !torch.vtensor<[6],f32>
  // CHECK: %[[SCALE:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[ZP:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],si16> -> !torch.int
  // CHECK: %[[MAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[ZP]]
  // CHECK: %[[DEQ:.+]] = torch.aten.dequantize.self %[[MAKE]]
  // CHECK: return %[[DEQ]]
  return %0 : !torch.vtensor<[6],f32>
}

// -----

// CHECK-LABEL: @test_dequantizelinear_ui8
func.func @test_dequantizelinear_ui8(%arg0: !torch.vtensor<[6],ui8>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],ui8>) -> !torch.vtensor<[6],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[6],f32>
  // CHECK: %[[SCALE:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[ZP:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[MAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[ZP]]
  // CHECK: %[[DEQ:.+]] = torch.aten.dequantize.self %[[MAKE]]
  // CHECK: return %[[DEQ]]
  return %0 : !torch.vtensor<[6],f32>
}

// -----

// CHECK-LABEL: @test_dequantizelinear_i32
func.func @test_dequantizelinear_i32(%arg0: !torch.vtensor<[6],si32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si32>) -> !torch.vtensor<[6],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],si32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[6],f32>
  // CHECK: %[[SCALE:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[ZP:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK: %[[MAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[ZP]]
  // CHECK: %[[DEQ:.+]] = torch.aten.dequantize.self %[[MAKE]]
  // CHECK: return %[[DEQ]]
  return %0 : !torch.vtensor<[6],f32>
}

// -----


// CHECK-LABEL: @test_div_bcast
func.func @test_div_bcast(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_div_example
func.func @test_div_example(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: @test_div
func.func @test_div(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_div_int32
func.func @test_div_int32(%arg0: !torch.vtensor<[3,4,5],si32>, %arg1: !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],si32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32> -> !torch.vtensor<[3,4,5],si32>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],si32>
  return %0 : !torch.vtensor<[3,4,5],si32>
}

// -----

// CHECK-LABEL: @test_div_uint8
func.func @test_div_uint8(%arg0: !torch.vtensor<[3,4,5],ui8>, %arg1: !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8> -> !torch.vtensor<[3,4,5],ui8>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8>
  return %0 : !torch.vtensor<[3,4,5],ui8>
}

// -----

// CHECK-LABEL: @test_dynamicquantizelinear
func.func @test_dynamicquantizelinear(%arg0: !torch.vtensor<[3,4,5],f32>) -> (!torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MAX0:.*]] = torch.aten.max %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[],f32>
  // CHECK: %[[MIN0:.*]] = torch.aten.min %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[],f32>
  // CHECK: %[[CF0:.*]] = torch.constant.float 0.000000e+00
  // CHECK: %[[CI1:.*]] = torch.constant.int 1
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct : () -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[CI6:.*]] = torch.constant.int 6
  // CHECK: %[[CF0_T:.*]] = torch.aten.full %[[LIST]], %[[CF0]], %[[CI6]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[MAX:.*]] = torch.aten.maximum %[[MAX0]], %[[CF0_T]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // CHECK: %[[MIN:.*]] = torch.aten.minimum %[[MIN0]], %[[CF0_T]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // CHECK: %[[DIAM:.*]] = torch.aten.sub.Tensor %[[MAX]], %[[MIN]], %[[CI1]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  // CHECK: %[[CQMAX:.*]] = torch.constant.float 2.550000e+02
  // CHECK: %[[LIST_1:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[NONE_1:.*]] = torch.constant.none
  // CHECK: %[[CI6_1:.*]] = torch.constant.int 6
  // CHECK: %[[CQMAX_T:.*]] = torch.aten.full %[[LIST_1]], %[[CQMAX]], %[[CI6_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[SCALE_T:.*]] = torch.aten.div.Tensor %[[DIAM]], %[[CQMAX_T]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // CHECK: %[[PZP_0:.*]] = torch.aten.div.Tensor %[[MIN0]], %[[SCALE_T]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // CHECK: %[[PZP_1:.*]] = torch.aten.sub.Tensor %[[CF0_T]], %[[PZP_0]], %[[CI1]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  // CHECK: %[[PZP_2:.*]] = torch.aten.clamp %[[PZP_1]], %[[CF0]], %[[CQMAX]] : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
  // CHECK: %[[PZP_3:.*]] = torch.aten.round %[[PZP_2]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // CHECK: %[[CI13:.*]] = torch.constant.int 13
  // CHECK: %[[NONE_2:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[ZP_T:.*]] = torch.aten.to.dtype %[[PZP_3]], %[[CI13]], %[[FALSE]], %[[FALSE]], %[[NONE_2]] : !torch.vtensor<[],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],ui8>
  // CHECK: %[[ZP:.*]] = torch.aten.item %[[ZP_T]] : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[SCALE:.*]] = torch.aten.item %[[SCALE_T]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[QUANT:.*]] = torch.aten.quantize_per_tensor %arg0, %[[SCALE]], %[[ZP]], %[[CI13]] : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[3,4,5],!torch.quint8>
  // CHECK: %[[INTQUANT:.*]] = torch.aten.int_repr %[[QUANT]] : !torch.vtensor<[3,4,5],!torch.quint8> -> !torch.vtensor<[3,4,5],ui8>
  %0:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> (!torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
  // CHECK: return %[[INTQUANT]], %[[SCALE_T]], %[[ZP_T]] : !torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>
}

// -----

// CHECK-LABEL: @test_equal_bcast
func.func @test_equal_bcast(%arg0: !torch.vtensor<[3,4,5],si32>, %arg1: !torch.vtensor<[5],si32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si32>, !torch.vtensor<[5],si32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Equal"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si32>, !torch.vtensor<[5],si32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: @test_erf
func.func @test_erf(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.erf %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Erf"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_equal
func.func @test_equal(%arg0: !torch.vtensor<[3,4,5],si32>, %arg1: !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Equal"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: @test_floor_example
func.func @test_floor_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.floor %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Floor"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_floor
func.func @test_floor(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.floor %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Floor"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_averagepool_1d_default
func.func @test_averagepool_1d_default(%arg0: !torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.avg_pool1d %arg0, %0, %2, %1, %false, %true : !torch.vtensor<[1,3,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,3,31],f32>
  %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.kernel_shape = [2 : si64], torch.onnx.count_include_pad = 1 : si64} : (!torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32>
  return %0 : !torch.vtensor<[1,3,31],f32>
}

// -----

// CHECK-LABEL: @test_averagepool_2d_ceil
func.func @test_averagepool_2d_ceil(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.avg_pool2d %arg0, %0, %2, %1, %true, %false, %none : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,2,2],f32>
  %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.ceil_mode = 1 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32>
  return %0 : !torch.vtensor<[1,1,2,2],f32>
}

// -----

// CHECK-LABEL: @test_averagepool_3d_default
func.func @test_averagepool_3d_default(%arg0: !torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.avg_pool3d %arg0, %0, %2, %1, %false, %false{{.*}}, %none : !torch.vtensor<[1,3,32,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,31,31,31],f32>
  %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32>
  return %0 : !torch.vtensor<[1,3,31,31,31],f32>
}

// -----

// CHECK-LABEL: @test_averagepool_with_padding
// CHECK-SAME:    %[[ARG:.*]]: !torch.vtensor<[1,20,64,48],f32>
// CHECK: torch.aten.avg_pool2d %[[ARG]], {{.*}} : !torch.vtensor<[1,20,64,48],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,20,32,24],f32>

func.func @test_averagepool_with_padding(%arg0: !torch.vtensor<[1,20,64,48],f32>) -> !torch.vtensor<[1,20,32,24],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 19 : si64} {

  %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,20,64,48],f32>) -> !torch.vtensor<[1,20,32,24],f32>
  return %0 : !torch.vtensor<[1,20,32,24],f32>
}

// -----

// CHECK-LABEL: @test_conv_with_strides_no_padding
func.func @test_conv_with_strides_no_padding(%arg0: !torch.vtensor<[1,1,7,5],f32>, %arg1: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,3,2],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C2]], %[[C2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0_1]], %[[C0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool false
  // CHECK: %[[BIAS:.*]] = torch.constant.none
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: torch.aten.convolution %arg0, %arg1, %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,7,5],f32>, !torch.vtensor<[1,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,3,2],f32>
  %0 = torch.operator "onnx.Conv"(%arg0, %arg1) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,7,5],f32>, !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,3,2],f32>
  return %0 : !torch.vtensor<[1,1,3,2],f32>
}

// -----

// CHECK-LABEL: @test_conv_with_strides_padding
func.func @test_conv_with_strides_padding(%arg0: !torch.vtensor<[1,1,7,5],f32>, %arg1: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,4,3],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1_1]], %[[C1_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C2]], %[[C2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool false
  // CHECK: %[[BIAS:.*]] = torch.constant.none
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: torch.aten.convolution %arg0, %arg1, %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,7,5],f32>, !torch.vtensor<[1,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,4,3],f32>
  %0 = torch.operator "onnx.Conv"(%arg0, %arg1) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,7,5],f32>, !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,4,3],f32>
  return %0 : !torch.vtensor<[1,1,4,3],f32>
}

// -----

// CHECK-LABEL: @test_conv_with_asymmetric_padding
func.func @test_conv_with_asymmetric_padding(%arg0: !torch.vtensor<[1,1,7,5],f32>, %arg1: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,4,3],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[int0:.*]] = torch.constant.int 0
  // CHECK: %[[int2:.*]] = torch.constant.int 2
  // CHECK: %[[int0_0:.*]] = torch.constant.int 0
  // CHECK: %[[int2_1:.*]] = torch.constant.int 2
  // CHECK: %[[int0_2:.*]] = torch.constant.int 0
  // CHECK: %[[int0_3:.*]] = torch.constant.int 0
  // CHECK: %[[FakePADS:.*]] = torch.prim.ListConstruct %[[int0_0]], %[[int0_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OGPADS:.*]] = torch.prim.ListConstruct %[[int0]], %[[int2]], %[[int2_1]], %[[int0_2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[str:.*]] = torch.constant.str "constant"
  // CHECK: %[[float0:.*]] = torch.constant.float 0.000
  // CHECK: %[[PrePad:.*]] = torch.aten.pad %arg0, %[[OGPADS]], %[[str]], %[[float0]] : !torch.vtensor<[1,1,7,5],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[1,1,9,7],f32>
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1_1]], %[[C1_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C2]], %[[C2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool false
  // CHECK: %[[BIAS:.*]] = torch.constant.none
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: %[[Conv:.*]] = torch.aten.convolution %[[PrePad]], %arg1, %[[BIAS]], %[[STRIDE]], %[[FakePADS]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,9,7],f32>, !torch.vtensor<[1,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,4,3],f32>
  // CHECK: return %[[Conv]]
  %0 = torch.operator "onnx.Conv"(%arg0, %arg1) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [2 : si64, 0 : si64, 0 : si64, 2 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,7,5],f32>, !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,4,3],f32>
  return %0 : !torch.vtensor<[1,1,4,3],f32>
}

// -----

// CHECK-LABEL: @test_conv_with_bias_strides_padding
func.func @test_conv_with_bias_strides_padding(%arg0: !torch.vtensor<[?,?,224,224],f32>, %arg1: !torch.vtensor<[64,3,7,7],f32>, %arg2: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,64,112,112],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C3]], %[[C3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C2]], %[[C2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool false
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: torch.aten.convolution %arg0, %arg1, %arg2, %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,64,112,112],f32>
  %0 = torch.operator "onnx.Conv"(%arg0, %arg1, %arg2) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [7 : si64, 7 : si64], torch.onnx.pads = [3 : si64, 3 : si64, 3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,64,112,112],f32>
  return %0 : !torch.vtensor<[?,64,112,112],f32>
}

// -----

// CHECK-LABEL: @test_convinteger_without_padding
func.func @test_convinteger_without_padding(%arg0: !torch.vtensor<[1,1,3,3],ui8>, %arg1: !torch.vtensor<[1,1,2,2],ui8>, %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[1,1,2,2],si32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[SCALE:.*]] = torch.constant.float 1.000000e+00
  // CHECK: %[[INPUT_ZP:.*]] = torch.aten.item %arg2 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[WEIGHT_ZP:.*]] = torch.aten.item %arg3 : !torch.vtensor<[1],ui8> -> !torch.int
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1_0]], %[[C1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[C1_3:.*]] = torch.constant.int 1
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1_2]], %[[C1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[C0_2:.*]] = torch.constant.int 0
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0_1]], %[[C0_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool false
  // CHECK: %[[BIAS:.*]] = torch.constant.none
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: %[[INPUT:.*]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[INPUT_ZP]] : !torch.vtensor<[1,1,3,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,3,3],!torch.quint8>
  // CHECK: %[[WEIGHT:.*]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[SCALE]], %[[WEIGHT_ZP]] : !torch.vtensor<[1,1,2,2],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,2,2],!torch.quint8>
  // CHECK: torch.aten.convolution %[[INPUT]], %[[WEIGHT]], %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,3,3],!torch.quint8>, !torch.vtensor<[1,1,2,2],!torch.quint8>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,2,2],si32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[1,1,3,3],ui8>, !torch.vtensor<[1,1,2,2],ui8>, !torch.vtensor<[],ui8>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[1,1,2,2],si32>
  return %0 : !torch.vtensor<[1,1,2,2],si32>
}

// -----

// CHECK-LABEL: @test_convinteger_with_padding
func.func @test_convinteger_with_padding(%arg0: !torch.vtensor<[1,1,3,3],ui8>, %arg1: !torch.vtensor<[1,1,2,2],ui8>, %arg2: !torch.vtensor<[],ui8>) -> !torch.vtensor<[1,1,4,4],si32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[SCALE:.*]] = torch.constant.float 1.000000e+00
  // CHECK: %[[INPUT_ZP:.*]] = torch.aten.item %arg2 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[WEIGHT_ZP:.*]] = torch.constant.int 0
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C1_0]], %[[C1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[C1_3:.*]] = torch.constant.int 1
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1_2]], %[[C1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C1_4:.*]] = torch.constant.int 1
  // CHECK: %[[C1_5:.*]] = torch.constant.int 1
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1_4]], %[[C1_5]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool false
  // CHECK: %[[BIAS:.*]] = torch.constant.none
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: %[[INPUT:.*]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[INPUT_ZP]] : !torch.vtensor<[1,1,3,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,3,3],!torch.quint8>
  // CHECK: %[[WEIGHT:.*]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[SCALE]], %[[WEIGHT_ZP]] : !torch.vtensor<[1,1,2,2],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,2,2],!torch.quint8>
  // CHECK: torch.aten.convolution %[[INPUT]], %[[WEIGHT]], %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,3,3],!torch.quint8>, !torch.vtensor<[1,1,2,2],!torch.quint8>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,4,4],si32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.ConvInteger"(%arg0, %arg1, %arg2) {torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64]} : (!torch.vtensor<[1,1,3,3],ui8>, !torch.vtensor<[1,1,2,2],ui8>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[1,1,4,4],si32>
  return %0 : !torch.vtensor<[1,1,4,4],si32>
}

// -----

// CHECK-LABEL: @test_convtranspose_dilations
func.func @test_convtranspose_dilations(%arg0: !torch.vtensor<[1,1,3,3],f32>, %arg1: !torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[C0_2:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C2]], %[[C2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0_1]], %[[C0_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool true
  // CHECK: %[[BIAS:.*]] = torch.constant.none
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: torch.aten.convolution %arg0, %arg1, %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,1,2,2],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,5,5],f32>
  %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {torch.onnx.dilations = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,5,5],f32>
  return %0 : !torch.vtensor<[1,1,5,5],f32>
}

// -----

// CHECK-LABEL: @test_convtranspose
func.func @test_convtranspose(%arg0: !torch.vtensor<[1,1,3,3],f32>, %arg1: !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,5,5],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[C0_2:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1_1]], %[[C1_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0_1]], %[[C0_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool true
  // CHECK: %[[BIAS:.*]] = torch.constant.none
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: torch.aten.convolution %arg0, %arg1, %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,5,5],f32>
  %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,5,5],f32>
  return %0 : !torch.vtensor<[1,2,5,5],f32>
}

// -----

// CHECK-LABEL: @test_convtranspose_pad
  func.func @test_convtranspose_pad(%arg0: !torch.vtensor<[1,1,3,3],f32>, %arg1: !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,10,8],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[C0:.*]] = torch.constant.int 0
    // CHECK: %[[C0_0:.*]] = torch.constant.int 0
    // CHECK: %[[C1:.*]] = torch.constant.int 1
    // CHECK: %[[C1_0:.*]] = torch.constant.int 1
    // CHECK: %[[C3:.*]] = torch.constant.int 3
    // CHECK: %[[C2:.*]] = torch.constant.int 2
    // CHECK: %[[C1_1:.*]] = torch.constant.int 1
    // CHECK: %[[C1_2:.*]] = torch.constant.int 1
    // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_0]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C3]], %[[C2]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C1_1]], %[[C1_2]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool true
    // CHECK: %[[BIAS:.*]] = torch.constant.none
    // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
    // CHECK: torch.aten.convolution %arg0, %arg1, %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,10,8],f32>
    %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {torch.onnx.output_padding = [1 : si64, 1 : si64], torch.onnx.strides = [3 : si64, 2 : si64]} : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,10,8],f32>
    return %0 : !torch.vtensor<[1,2,10,8],f32>
  }

// -----

// CHECK-LABEL: @test_convtranspose_pads
  func.func @test_convtranspose_pads(%arg0: !torch.vtensor<[1,1,3,3],f32>, %arg1: !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,7,3],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[C1:.*]] = torch.constant.int 1
    // CHECK: %[[C2:.*]] = torch.constant.int 2
    // CHECK: %[[C1_0:.*]] = torch.constant.int 1
    // CHECK: %[[C1_1:.*]] = torch.constant.int 1
    // CHECK: %[[C3:.*]] = torch.constant.int 3
    // CHECK: %[[C2_0:.*]] = torch.constant.int 2
    // CHECK: %[[C0:.*]] = torch.constant.int 0
    // CHECK: %[[C0_1:.*]] = torch.constant.int 0
    // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C1]], %[[C2]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1_0]], %[[C1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C3]], %[[C2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool true
    // CHECK: %[[BIAS:.*]] = torch.constant.none
    // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
    // CHECK: torch.aten.convolution %arg0, %arg1, %[[BIAS]], %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,7,3],f32>
    %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {torch.onnx.pads = [1 : si64, 2 : si64, 1 : si64, 2 : si64], torch.onnx.strides = [3 : si64, 2 : si64]} : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,7,3],f32>
    return %0 : !torch.vtensor<[1,2,7,3],f32>
  }

// -----

// CHECK-LABEL: @test_batchnorm_epsilon
func.func @test_batchnorm_epsilon(%arg0: !torch.vtensor<[2,3,4,5],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>, %arg4: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[MOMENTUM:.*]] = torch.constant.float 0.89999997615814208
  // CHECK: %[[EPS:.*]] = torch.constant.float 0.0099999997764825821
  // CHECK: torch.aten.batch_norm %arg0, %arg1, %arg2, %arg3, %arg4, %[[FALSE]], %[[MOMENTUM]], %[[EPS]], %[[FALSE]] : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[2,3,4,5],f32>
  %0 = torch.operator "onnx.BatchNormalization"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.epsilon = 0.00999999977 : f32} : (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32>
  return %0 : !torch.vtensor<[2,3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_batchnorm_example
func.func @test_batchnorm_example(%arg0: !torch.vtensor<[2,3,4,5],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>, %arg4: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[MOMENTUM:.*]] = torch.constant.float 0.89999997615814208
  // CHECK: %[[EPS:.*]] = torch.constant.float 9.9999997473787516E-6
  // CHECK: torch.aten.batch_norm %arg0, %arg1, %arg2, %arg3, %arg4, %[[FALSE]], %[[MOMENTUM]], %[[EPS]], %[[FALSE]] : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[2,3,4,5],f32>
  %0 = torch.operator "onnx.BatchNormalization"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32>
  return %0 : !torch.vtensor<[2,3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_concat_1d_axis_0
func.func @test_concat_1d_axis_0(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 0
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @test_concat_1d_axis_negative_1
func.func @test_concat_1d_axis_negative_1(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @test_concat_2d_axis_0
func.func @test_concat_2d_axis_0(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 0
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// -----

// CHECK-LABEL: @test_concat_2d_axis_1
func.func @test_concat_2d_axis_1(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: @test_concat_2d_axis_negative_1
func.func @test_concat_2d_axis_negative_1(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: @test_concat_2d_axis_negative_2
func.func @test_concat_2d_axis_negative_2(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -2
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// -----

// CHECK-LABEL: @test_concat_3d_axis_0
func.func @test_concat_3d_axis_0(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 0
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32>
  return %0 : !torch.vtensor<[4,2,2],f32>
}

// -----

// CHECK-LABEL: @test_concat_3d_axis_1
func.func @test_concat_3d_axis_1(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32>
  return %0 : !torch.vtensor<[2,4,2],f32>
}

// -----

// CHECK-LABEL: @test_concat_3d_axis_2
func.func @test_concat_3d_axis_2(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 2
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32>
  return %0 : !torch.vtensor<[2,2,4],f32>
}

// -----

// CHECK-LABEL: @test_concat_3d_axis_negative_1
func.func @test_concat_3d_axis_negative_1(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32>
  return %0 : !torch.vtensor<[2,2,4],f32>
}

// -----

// CHECK-LABEL: @test_concat_3d_axis_negative_2
func.func @test_concat_3d_axis_negative_2(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -2
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32>
  return %0 : !torch.vtensor<[2,4,2],f32>
}

// -----

// CHECK-LABEL: @test_concat_3d_axis_negative_3
func.func @test_concat_3d_axis_negative_3(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -3
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -3 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32>
  return %0 : !torch.vtensor<[4,2,2],f32>
}

// -----

// CHECK-LABEL: @test_cumsum_1d_exclusive
func.func @test_cumsum_1d_exclusive(%arg0: !torch.vtensor<[5],f64>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[RANK:.*]] = torch.constant.int 1
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[AXIS:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK: %[[BOOL:.*]] = torch.aten.lt.int %[[AXIS]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[INT:.*]] = torch.aten.Int.bool %[[BOOL]] : !torch.bool -> !torch.int
  // CHECK: %[[OTHER:.*]] = torch.aten.mul.int %[[INT]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[AXIS]], %[[OTHER]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %arg0, %[[ADD]], %[[NONE]] : !torch.vtensor<[5],f64>, !torch.int, !torch.none -> !torch.vtensor<[5],f64>
  // CHECK: torch.aten.sub.Tensor %[[CUMSUM]], %arg0, %[[C1]] : !torch.vtensor<[5],f64>, !torch.vtensor<[5],f64>, !torch.int -> !torch.vtensor<[5],f64>
  %0 = torch.operator "onnx.CumSum"(%arg0, %arg1) {torch.onnx.exclusive = 1 : si64} : (!torch.vtensor<[5],f64>, !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64>
  return %0 : !torch.vtensor<[5],f64>
}

// -----

// CHECK-LABEL: @test_cumsum_1d_reverse
func.func @test_cumsum_1d_reverse(%arg0: !torch.vtensor<[5],f64>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[RANK:.*]] = torch.constant.int 1
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[AXIS:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK: %[[BOOL:.*]] = torch.aten.lt.int %[[AXIS]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[INT:.*]] = torch.aten.Int.bool %[[BOOL]] : !torch.bool -> !torch.int
  // CHECK: %[[OTHER:.*]] = torch.aten.mul.int %[[INT]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[AXIS]], %[[OTHER]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[ADD]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %arg0, %[[DIMS]] : !torch.vtensor<[5],f64>, !torch.list<int> -> !torch.vtensor<[5],f64>
  // CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[FLIP]], %[[ADD]], %[[NONE]] : !torch.vtensor<[5],f64>, !torch.int, !torch.none -> !torch.vtensor<[5],f64>
  // CHECK: torch.aten.flip %[[CUMSUM]], %[[DIMS]] : !torch.vtensor<[5],f64>, !torch.list<int> -> !torch.vtensor<[5],f64>
  %0 = torch.operator "onnx.CumSum"(%arg0, %arg1) {torch.onnx.reverse = 1 : si64} : (!torch.vtensor<[5],f64>, !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64>
  return %0 : !torch.vtensor<[5],f64>
}

// -----

// CHECK-LABEL: @test_cumsum_1d_reverse_exclusive
func.func @test_cumsum_1d_reverse_exclusive(%arg0: !torch.vtensor<[5],f64>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[RANK:.*]] = torch.constant.int 1
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[AXIS:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK: %[[BOOL:.*]] = torch.aten.lt.int %[[AXIS]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[INT:.*]] = torch.aten.Int.bool %[[BOOL]] : !torch.bool -> !torch.int
  // CHECK: %[[OTHER:.*]] = torch.aten.mul.int %[[INT]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[AXIS]], %[[OTHER]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[ADD]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %arg0, %[[DIMS]] : !torch.vtensor<[5],f64>, !torch.list<int> -> !torch.vtensor<[5],f64>
  // CHECK: %[[CUMSUM:.*]] = torch.aten.cumsum %[[FLIP]], %[[ADD]], %[[NONE]] : !torch.vtensor<[5],f64>, !torch.int, !torch.none -> !torch.vtensor<[5],f64>
  // CHECK: %[[FLIP_0:.*]] = torch.aten.flip %[[CUMSUM]], %[[DIMS]] : !torch.vtensor<[5],f64>, !torch.list<int> -> !torch.vtensor<[5],f64>
  // CHECK: torch.aten.sub.Tensor %[[FLIP_0]], %arg0, %[[C1]] : !torch.vtensor<[5],f64>, !torch.vtensor<[5],f64>, !torch.int -> !torch.vtensor<[5],f64>
  %0 = torch.operator "onnx.CumSum"(%arg0, %arg1) {torch.onnx.exclusive = 1 : si64, torch.onnx.reverse = 1 : si64} : (!torch.vtensor<[5],f64>, !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64>
  return %0 : !torch.vtensor<[5],f64>
}

// -----

// CHECK-LABEL: @test_cumsum_2d
func.func @test_cumsum_2d(%arg0: !torch.vtensor<[2,3],f64>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[2,3],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[RANK:.*]] = torch.constant.int 2
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[AXIS:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK: %[[BOOL:.*]] = torch.aten.lt.int %[[AXIS]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[INT:.*]] = torch.aten.Int.bool %[[BOOL]] : !torch.bool -> !torch.int
  // CHECK: %[[OTHER:.*]] = torch.aten.mul.int %[[INT]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[AXIS]], %[[OTHER]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // torch.aten.cumsum %arg0, %[[ADD]], %[[NONE]] : !torch.vtensor<[2,3],f64>, !torch.int, !torch.none -> !torch.vtensor<[2,3],f64>
  %0 = torch.operator "onnx.CumSum"(%arg0, %arg1) : (!torch.vtensor<[2,3],f64>, !torch.vtensor<[],si32>) -> !torch.vtensor<[2,3],f64>
  return %0 : !torch.vtensor<[2,3],f64>
}

// -----

// CHECK-LABEL:   func.func @test_exp
func.func @test_exp(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64} {
  // CHECK:  torch.aten.exp %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Exp"(%arg0) : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_expand_dim2_shape2
func.func @test_expand_dim2_shape2(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[2],si32>)
              -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INT0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[SEL0:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[2],si32>, !torch.int, !torch.int -> !torch.vtensor<[],si32>
  // CHECK-DAG: %[[ITEM0:.+]] = torch.aten.item %[[SEL0]] : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[SZ0:.+]] = torch.aten.size.int %arg0, %[[I0]] : !torch.vtensor<[1,4],f32>, !torch.int -> !torch.int
  // CHECK-DAG: %[[MX0:.+]] = torch.prim.max.int %[[ITEM0]], %[[SZ0]] : !torch.int, !torch.int -> !torch.int
  // CHECK-DAG: %[[INT1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[SEL1:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT1]] : !torch.vtensor<[2],si32>, !torch.int, !torch.int -> !torch.vtensor<[],si32>
  // CHECK-DAG: %[[ITEM1:.+]] = torch.aten.item %[[SEL1]] : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[I1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[SZ1:.+]] = torch.aten.size.int %arg0, %[[I1]] : !torch.vtensor<[1,4],f32>, !torch.int -> !torch.int
  // CHECK-DAG: %[[MX1:.+]] = torch.prim.max.int %[[ITEM1]], %[[SZ1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[MX0]], %[[MX1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.broadcast_to %arg0, %[[LIST]] : !torch.vtensor<[1,4],f32>, !torch.list<int> -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Expand"(%arg0, %arg1) : (!torch.vtensor<[1,4],f32>, !torch.vtensor<[2],si32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @test_expand_dim2_shape3
func.func @test_expand_dim2_shape3(%arg0: !torch.vtensor<[3,1],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,6],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:  %[[I0:.+]] = torch.constant.int 0
  // CHECK-NEXT:  %[[I0_0:.+]] = torch.constant.int 0
  // CHECK-NEXT:  %[[SEL0:.+]] = torch.aten.select.int %arg1, %[[I0]], %[[I0_0]]
  // CHECK-NEXT:  %[[ITEM0:.+]] = torch.aten.item %[[SEL0]]
  // CHECK-NEXT:  %[[I1:.+]] = torch.constant.int 1
  // CHECK-NEXT:  %[[SEL1:.+]] = torch.aten.select.int %arg1, %[[I0]], %[[I1]]
  // CHECK-NEXT:  %[[ITEM1:.+]] = torch.aten.item %[[SEL1]]
  // CHECK-NEXT:  %[[D1:.+]] = torch.constant.int 0
  // CHECK-NEXT:  %[[SZ1:.+]] = torch.aten.size.int %arg0, %[[D1]]
  // CHECK-NEXT:  %[[MX1:.+]] = torch.prim.max.int %[[ITEM1]], %[[SZ1]] : !torch.int, !torch.int -> !torch.int
  // CHECK-NEXT:  %[[I2:.+]] = torch.constant.int 2
  // CHECK-NEXT:  %[[SEL2:.+]] = torch.aten.select.int %arg1, %[[I0]], %[[I2]]
  // CHECK-NEXT:  %[[ITEM2:.+]] = torch.aten.item %[[SEL2]]
  // CHECK-NEXT:  %[[D2:.+]] = torch.constant.int 1
  // CHECK-NEXT:  %[[SZ2:.+]] = torch.aten.size.int %arg0, %[[D2]]
  // CHECK-NEXT:  %[[MX2:.+]] = torch.prim.max.int %[[ITEM2]], %[[SZ2]]
  // CHECK-NEXT:  %[[LIST:.+]] = torch.prim.ListConstruct %[[ITEM0]], %[[MX1]], %[[MX2]]
  // CHECK-NEXT:  %[[EXPAND:.+]] = torch.aten.broadcast_to %arg0, %[[LIST]]
  // CHECK:  return %[[EXPAND]]
  %0 = torch.operator "onnx.Expand"(%arg0, %arg1) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,6],f32>
  return %0 : !torch.vtensor<[2,3,6],f32>
}

// -----

// CHECK-LABEL: @test_dropout
func.func @test_dropout(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %float5.000000e-01, %false : !torch.vtensor<[3],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3],f32
  %0 = torch.operator "onnx.Dropout"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_dropout_default
func.func @test_dropout_default(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %float5.000000e-01, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Dropout"(%arg0) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_dropout_default_mask
func.func @test_dropout_default_mask(%arg0: !torch.vtensor<[3,4,5],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %float5.000000e-01, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.ones_like %arg0, %int11, %none, %none, %none, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4,5],i1>
  %0:2 = torch.operator "onnx.Dropout"(%arg0) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>)
  return %0#0, %0#1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: @test_dropout_default_mask_ratio
func.func @test_dropout_default_mask_ratio(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %0, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.ones_like %arg0, %int11, %none, %none, %none, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4,5],i1>
  %0:2 = torch.operator "onnx.Dropout"(%arg0, %arg1) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>)
  return %0#0, %0#1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: @test_dropout_default_ratio
func.func @test_dropout_default_ratio(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %0, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Dropout"(%arg0, %arg1) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_training_dropout_zero_ratio
func.func @test_training_dropout_zero_ratio(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],i1>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %0, %2 : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Dropout"(%arg0, %arg1, %arg2) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],i1>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_elu_default
func.func @test_elu_default(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.elu %arg0, %float0.000000e00, %float1.000000e00, %float1.000000e00 : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.float, !torch.float -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Elu"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_elu_example
func.func @test_elu_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.elu %arg0, %float2.000000e00, %float1.000000e00, %float1.000000e00 : !torch.vtensor<[3],f32>, !torch.float, !torch.float, !torch.float -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Elu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_depthtospace_example
func.func @test_depthtospace_example(%arg0: !torch.vtensor<[1,8,2,3],f32>) -> !torch.vtensor<[1,2,4,6],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[SIZE:.*]] = torch.aten.size.int %arg0, %[[C0]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[SIZE_0:.*]] = torch.aten.size.int %arg0, %[[C1]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[SIZE_1:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[SIZE_2:.*]] = torch.aten.size.int %arg0, %[[C3]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C4:.*]] = torch.constant.int 4
  // CHECK: %[[DIV:.*]] = torch.aten.div.int %[[SIZE_0]], %[[C4]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[INT:.*]] = torch.aten.Int.float %[[DIV]] : !torch.float -> !torch.int
  // CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[SIZE]], %[[C2_0]], %[[C2_0]], %[[INT]], %[[SIZE_1]], %[[SIZE_2]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESHAPE:.*]] = torch.aten.reshape %arg0, %[[RESHAPE_LIST]] : !torch.vtensor<[1,8,2,3],f32>, !torch.list<int> -> !torch.vtensor<[1,2,2,2,2,3],f32>
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[RESHAPE]], %[[C1_0]], %[[C3_0]] : !torch.vtensor<[1,2,2,2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2,2,3],f32>
  // CHECK: %[[C2_1:.*]] = torch.constant.int 2
  // CHECK: %[[C4_0:.*]] = torch.constant.int 4
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[TRANSPOSE]], %[[C2_1]], %[[C4_0]] : !torch.vtensor<[1,2,2,2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2,2,3],f32>
  // CHECK: %[[C4_1:.*]] = torch.constant.int 4
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[C4_1]], %[[C5]] : !torch.vtensor<[1,2,2,2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2,3,2],f32>
  // CHECK: %[[MUL:.*]] = torch.aten.mul.int %[[SIZE_1]], %[[C2_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[MUL_0:.*]] = torch.aten.mul.int %[[SIZE_2]], %[[C2_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[RESHAPE_LIST_0:.*]] = torch.prim.ListConstruct %[[SIZE]], %5, %[[MUL]], %[[MUL_0]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.reshape %[[TRANSPOSE_2]], %[[RESHAPE_LIST_0]] : !torch.vtensor<[1,2,2,2,3,2],f32>, !torch.list<int> -> !torch.vtensor<[1,2,4,6],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[1,2,4,6],f32
  %0 = torch.operator "onnx.DepthToSpace"(%arg0) {torch.onnx.blocksize = 2 : si64, torch.onnx.mode = "DCR"} : (!torch.vtensor<[1,8,2,3],f32>) -> !torch.vtensor<[1,2,4,6],f32>
  return %0 : !torch.vtensor<[1,2,4,6],f32>
}

// -----

// CHECK-LABEL: @test_depthtospace_crd_mode_example
func.func @test_depthtospace_crd_mode_example(%arg0: !torch.vtensor<[1,8,2,3],f32>) -> !torch.vtensor<[1,2,4,6],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[SIZE:.*]] = torch.aten.size.int %arg0, %[[C0]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[SIZE_0:.*]] = torch.aten.size.int %arg0, %[[C1]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[SIZE_1:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[SIZE_2:.*]] = torch.aten.size.int %arg0, %[[C3]] : !torch.vtensor<[1,8,2,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C4:.*]] = torch.constant.int 4
  // CHECK: %[[DIV:.*]] = torch.aten.div.int %[[SIZE_0]], %[[C4]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[INT:.*]] = torch.aten.Int.float %[[DIV]] : !torch.float -> !torch.int
  // CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[SIZE]], %[[C2_0]], %[[C2_0]], %[[INT]], %[[SIZE_1]], %[[SIZE_2]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESHAPE:.*]] = torch.aten.reshape %arg0, %[[RESHAPE_LIST]] : !torch.vtensor<[1,8,2,3],f32>, !torch.list<int> -> !torch.vtensor<[1,2,2,2,2,3],f32>
  // CHECK: %[[C2_1:.*]] = torch.constant.int 2
  // CHECK: %[[C4_0:.*]] = torch.constant.int 4
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[RESHAPE]], %[[C2_1]], %[[C4_0]] : !torch.vtensor<[1,2,2,2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2,2,3],f32>
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[C4_1:.*]] = torch.constant.int 4
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[TRANSPOSE]], %[[C3_0]], %[[C4_1]] : !torch.vtensor<[1,2,2,2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2,2,3],f32>
  // CHECK: %[[C4_1:.*]] = torch.constant.int 4
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[C4_1]], %[[C5]] : !torch.vtensor<[1,2,2,2,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2,3,2],f32>
  // CHECK: %[[MUL:.*]] = torch.aten.mul.int %[[SIZE_1]], %[[C2_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[MUL_0:.*]] = torch.aten.mul.int %[[SIZE_2]], %[[C2_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[RESHAPE_LIST_0:.*]] = torch.prim.ListConstruct %[[SIZE]], %5, %[[MUL]], %[[MUL_0]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.reshape %[[TRANSPOSE_2]], %[[RESHAPE_LIST_0]] : !torch.vtensor<[1,2,2,2,3,2],f32>, !torch.list<int> -> !torch.vtensor<[1,2,4,6],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[1,2,4,6],f32
  %0 = torch.operator "onnx.DepthToSpace"(%arg0) {torch.onnx.blocksize = 2 : si64, torch.onnx.mode = "CRD"} : (!torch.vtensor<[1,8,2,3],f32>) -> !torch.vtensor<[1,2,4,6],f32>
  return %0 : !torch.vtensor<[1,2,4,6],f32>
}

// -----

// CHECK-LABEL: @float_constant
func.func @float_constant() -> !torch.vtensor<[], f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK: %[[CST:.+]] = torch.vtensor.literal(dense<2.500000e-01> : tensor<f32>) : !torch.vtensor<[],f32>
  // CHECK: return %[[CST]]
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value_float = 0.25 : f32} : () -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: @int_constant
func.func @int_constant() -> !torch.vtensor<[], si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK: %[[CST:.+]] = torch.vtensor.literal(dense<79> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: return %[[CST]]
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value_int = 79 : si64} : () -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL: @dense_constant
func.func @dense_constant() -> !torch.vtensor<[1], si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK: %[[CST:.+]] = torch.vtensor.literal(dense<13> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // CHECK: return %[[CST]]
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<13> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64>
  return %0 : !torch.vtensor<[1],si64>
}

// -----

// CHECK-LABEL: @ints_constant
func.func @ints_constant() -> !torch.vtensor<[2], si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK: %[[CST:.+]] = torch.vtensor.literal(dense<[7, 9]> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
  // CHECK: return %[[CST]]
  %0 = "torch.operator"() <{name = "onnx.Constant"}> {torch.onnx.value_ints = [7 : si64, 9 : si64]} : () -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// -----

// CHECK-LABEL: @dense_resource_constant
func.func @dense_resource_constant() -> () attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK: torch.vtensor.literal(dense<[0, 10, 128, 17000]> : tensor<4xsi32>) : !torch.vtensor<[4],si32>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_int32> : tensor<4xsi32>} : () -> !torch.vtensor<[4],si32>
  // CHECK: torch.vtensor.literal(dense<[0.000000e+00, 1.000000e+01, 1.280000e+02, 1.700000e+04]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_float32> : tensor<4xf32>} : () -> !torch.vtensor<[4],f32>
  // CHECK: torch.vtensor.literal(dense<[-128, -1, 50, 127]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_int8> : tensor<4xsi8>} : () -> !torch.vtensor<[4],si8>
  // CHECK: torch.vtensor.literal(dense<[128, 255, 50, 127]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_int8> : tensor<4xui8>} : () -> !torch.vtensor<[4],ui8>
  return
}

{-#
  dialect_resources: {
    builtin: {
      _int8: "0x0800000080FF327F",
      _int32: "0x08000000000000000a0000008000000068420000",
      _float32: "0x0800000000000000000020410000004300d08446"
    }
  }
#-}

// -----

// CHECK-LABEL: @dense_constant_i1
func.func @dense_constant_i1() -> !torch.vtensor<[5],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK: %[[CST:.+]] = torch.vtensor.literal(dense<[true, false, false, true, true]> : tensor<5xi1>) : !torch.vtensor<[5],i1>
  // CHECK: return %[[CST]] : !torch.vtensor<[5],i1>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_> : tensor<5xi1>} : () -> !torch.vtensor<[5],i1>
  return %0 : !torch.vtensor<[5],i1>
}

{-#
  dialect_resources: {
    builtin: {
      _: "0x080000000100000101"
    }
  }
#-}

// -----


// CHECK-LABEL: @test_flatten_4d_axis_2
func.func @test_flatten_4d_axis_2(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3,20],f32>
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 1
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor<[2,3,20],f32>, !torch.int, !torch.int -> !torch.vtensor<[6,20],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32>
  return %0 : !torch.vtensor<[6,20],f32>
}

// -----

// // CHECK-LABEL: @test_flatten_4d_axis_0
func.func @test_flatten_4d_axis_0(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[120],f32>
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor<[120],f32>, !torch.int -> !torch.vtensor<[1,120],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32>
  return %0 : !torch.vtensor<[1,120],f32>
}

// -----

// CHECK-LABEL: @test_flatten_4d_axis_4
func.func @test_flatten_4d_axis_4(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[120,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_INDEX:.*]] = torch.constant.int 4
  // CHECK-DAG: %[[CR:.*]] = torch.aten.unsqueeze %arg0, %[[RIGHT_INDEX]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int -> !torch.vtensor<[2,3,4,5,1],f32>
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 3
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor<[2,3,4,5,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[120,1],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 4 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[120,1],f32>
  return %0 : !torch.vtensor<[120,1],f32>
}

// -----

// CHECK-LABEL: @test_flatten_4d_axis_negative_2
func.func @test_flatten_4d_axis_negative_2(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3,20],f32>
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 1
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor<[2,3,20],f32>, !torch.int, !torch.int -> !torch.vtensor<[6,20],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32>
  return %0 : !torch.vtensor<[6,20],f32>
}

// -----

// CHECK-LABEL: @test_flatten_4d_axis_negative_1
func.func @test_flatten_4d_axis_negative_1(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[24,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3,4,5],f32>
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 2
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[24,5],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[24,5],f32>
  return %0 : !torch.vtensor<[24,5],f32>
}

// -----

// CHECK-LABEL: @test_flatten_4d_axis_negative_4
func.func @test_flatten_4d_axis_negative_4(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[120],f32>
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor<[120],f32>, !torch.int -> !torch.vtensor<[1,120],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -4 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32>
  return %0 : !torch.vtensor<[1,120],f32>
}

// -----

// CHECK-LABEL: @test_flatten_2d_axis_1
func.func @test_flatten_2d_axis_1(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 0
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL: @test_flatten_1d_axis_0
func.func @test_flatten_1d_axis_0(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2],f32>, !torch.int, !torch.int -> !torch.vtensor<[2],f32>
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32>
  return %0 : !torch.vtensor<[1,2],f32>
}

// -----

// CHECK-LABEL: @test_flatten_1d_axis_negative_1
func.func @test_flatten_1d_axis_negative_1(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2],f32>, !torch.int, !torch.int -> !torch.vtensor<[2],f32>
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32>
  return %0 : !torch.vtensor<[1,2],f32>
}

// -----

// COM: CHECK-LABEL: @test_flatten_1d_axis_1
func.func @test_flatten_1d_axis_1(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_INDEX:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[CR:.*]] = torch.aten.unsqueeze %arg0, %[[RIGHT_INDEX]] : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[2,1],f32>
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 0
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor<[2,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,1],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2,1],f32>
  return %0 : !torch.vtensor<[2,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_constant_of_shape_dense_float_default
func.func @test_constant_of_shape_dense_float_default() -> !torch.vtensor<[2,3,4], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[SHAPE_CST:.*]] = torch.vtensor.literal(dense<[2, 3, 4]> : tensor<3xsi64>) : !torch.vtensor<[3],si64>
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: %[[EXTRACT_0:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT0_0]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_0:.*]] = torch.aten.item %[[EXTRACT_0]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[EXTRACT_1:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT1]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_1:.*]] = torch.aten.item %[[EXTRACT_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT2:.*]]  = torch.constant.int 2
  // CHECK: %[[EXTRACT_2:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT2]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_2:.*]] = torch.aten.item %[[EXTRACT_2]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[ELE_0]], %[[ELE_1]], %[[ELE_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FILL_VAL:.*]] = torch.constant.float 0.000000e+00
  // CHECK: %[[ATEN_FULL:.*]] = torch.aten.full %[[DIM_LIST]], %[[FILL_VAL]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3,4],f32>
  %cst = torch.vtensor.literal(dense<[2,3,4]> : tensor<3xsi64>) : !torch.vtensor<[3], si64>
  %0 = "torch.operator"(%cst) <{name = "onnx.ConstantOfShape"}> : (!torch.vtensor<[3], si64>) -> !torch.vtensor<[2,3,4], f32>
  return %0 : !torch.vtensor<[2,3,4], f32>
}

// -----

// CHECK-LABEL: func.func @test_constant_of_shape_dense_float_cst
func.func @test_constant_of_shape_dense_float_cst() -> !torch.vtensor<[2,3,4], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[SHAPE_CST:.*]] = torch.vtensor.literal(dense<[2, 3, 4]> : tensor<3xsi64>) : !torch.vtensor<[3],si64>
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: %[[EXTRACT_0:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT0_0]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_0:.*]] = torch.aten.item %[[EXTRACT_0]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[EXTRACT_1:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT1]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_1:.*]] = torch.aten.item %[[EXTRACT_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT2:.*]]  = torch.constant.int 2
  // CHECK: %[[EXTRACT_2:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT2]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_2:.*]] = torch.aten.item %[[EXTRACT_2]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[ELE_0]], %[[ELE_1]], %[[ELE_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FILL_VAL:.*]] = torch.constant.float 3.4000000953674316
  // CHECK: %[[ATEN_FULL:.*]] = torch.aten.full %[[DIM_LIST]], %[[FILL_VAL]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3,4],f32>
  %cst = torch.vtensor.literal(dense<[2,3,4]> : tensor<3xsi64>) : !torch.vtensor<[3], si64>
  %0 = "torch.operator"(%cst) <{name = "onnx.ConstantOfShape"}> {torch.onnx.value = dense<3.4> : tensor<1xf32>}: (!torch.vtensor<[3], si64>) -> !torch.vtensor<[2,3,4], f32>
  return %0 : !torch.vtensor<[2,3,4], f32>
}

// -----

// CHECK-LABEL: func.func @test_constant_of_shape_dense_int_cst
func.func @test_constant_of_shape_dense_int_cst() -> !torch.vtensor<[2,3,4], si64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[SHAPE_CST:.*]] = torch.vtensor.literal(dense<[2, 3, 4]> : tensor<3xsi64>) : !torch.vtensor<[3],si64>
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: %[[EXTRACT_0:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT0_0]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_0:.*]] = torch.aten.item %[[EXTRACT_0]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[EXTRACT_1:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT1]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_1:.*]] = torch.aten.item %[[EXTRACT_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT2:.*]]  = torch.constant.int 2
  // CHECK: %[[EXTRACT_2:.*]] = torch.aten.select.int %[[SHAPE_CST]], %[[INT0]], %[[INT2]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ELE_2:.*]] = torch.aten.item %[[EXTRACT_2]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[ELE_0]], %[[ELE_1]], %[[ELE_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FILL_VAL:.*]] = torch.constant.int 3
  // CHECK: %[[ATEN_FULL:.*]] = torch.aten.full %[[DIM_LIST]], %[[FILL_VAL]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3,4],si64>
  %cst = torch.vtensor.literal(dense<[2,3,4]> : tensor<3xsi64>) : !torch.vtensor<[3], si64>
  %0 = "torch.operator"(%cst) <{name = "onnx.ConstantOfShape"}> {torch.onnx.value = dense<3> : tensor<1xsi64>}: (!torch.vtensor<[3], si64>) -> !torch.vtensor<[2,3,4], si64>
  return %0 : !torch.vtensor<[2,3,4], si64>
}

// -----

// CHECK-LABEL: func.func @test_celu
func.func @test_celu(%arg0: !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:  %[[ALPHA:.*]] = torch.constant.float 2.000000e+00
  // CHECK:  %0 = torch.aten.div.Scalar %arg0, %[[ALPHA]] : !torch.vtensor<[3,3,3,1],f32>, !torch.float -> !torch.vtensor<[3,3,3,1],f32>
  // CHECK:   %1 = torch.aten.exp %0 : !torch.vtensor<[3,3,3,1],f32> -> !torch.vtensor<[3,3,3,1],f32>
  // CHECK:    %int1 = torch.constant.int 1
  // CHECK:    %2 = torch.aten.sub.Scalar %1, %int1, %int1 : !torch.vtensor<[3,3,3,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,3,3,1],f32>
  // CHECK:    %3 = torch.aten.mul.Scalar %2, %[[ALPHA]] : !torch.vtensor<[3,3,3,1],f32>, !torch.float -> !torch.vtensor<[3,3,3,1],f32>
  // CHECK:    %int0 = torch.constant.int 0
  // CHECK:    %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK:    %none = torch.constant.none
  // CHECK:    %int6 = torch.constant.int 6
  // CHECK:    %[[ZERO:.*]] = torch.aten.full %4, %int0, %int6, %none, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK:    %[[MIN:.*]] = torch.aten.minimum %[[ZERO]], %3 : !torch.vtensor<[],f32>, !torch.vtensor<[3,3,3,1],f32> -> !torch.vtensor<[3,3,3,1],f32>
  // CHECK:    %[[MAX:.*]] = torch.aten.maximum %[[ZERO]], %arg0 : !torch.vtensor<[],f32>, !torch.vtensor<[3,3,3,1],f32> -> !torch.vtensor<[3,3,3,1],f32>
  // CHECK:    %8 = torch.aten.add.Tensor %[[MAX]], %[[MIN]], %int1 : !torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[3,3,3,1],f32>, !torch.int -> !torch.vtensor<[3,3,3,1],f32>
  %0 = torch.operator "onnx.Celu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32>
  return %0 : !torch.vtensor<[3,3,3,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_compress
func.func @test_compress(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[3], si64>) -> !torch.vtensor<[2,3,2], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INDEX:.*]] = torch.aten.nonzero %arg1 : !torch.vtensor<[3],si64> -> !torch.vtensor<[2],si64>
  // CHECK: %[[DIM:.*]] = torch.constant.int 2
  // CHECK: torch.aten.index_select %arg0, %[[DIM]], %[[INDEX]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[2,3,2],f32>
  %0 = torch.operator "onnx.Compress"(%arg0, %arg1) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,2],f32>
  return %0 : !torch.vtensor<[2,3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_compress_default_axis
func.func @test_compress_default_axis(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[CST:.*]] = torch.vtensor.literal(dense<[0, 1, 0, 1, 0, 1]> : tensor<6xsi64>) : !torch.vtensor<[6],si64>
  // CHECK: %[[INDEX:.*]] = torch.aten.nonzero %[[CST]] : !torch.vtensor<[6],si64> -> !torch.vtensor<[3],si64>
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[END_DIM:.*]] = torch.constant.int -1
  // CHECK: %[[ATEN_FLATTEN:.*]] = torch.aten.flatten.using_ints %arg0, %[[INT0]], %[[END_DIM]] : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[6],f32>
  // CHECK: torch.aten.index_select %[[ATEN_FLATTEN]], %[[INT0]], %[[INDEX]] : !torch.vtensor<[6],f32>, !torch.int, !torch.vtensor<[3],si64> -> !torch.vtensor<[3],f32>
  %cst = torch.vtensor.literal(dense<[0,1,0,1,0,1]> : tensor<6xsi64>) : !torch.vtensor<[6], si64>
  %0 = torch.operator "onnx.Compress"(%arg0, %cst) : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[6],si64>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_compress_neg_axis
func.func @test_compress_neg_axis(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,2,4], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[CST:.*]] = torch.vtensor.literal(dense<[0, 1, 1]> : tensor<3xsi64>) : !torch.vtensor<[3],si64>
  // CHECK: %[[INDEX:.*]] = torch.aten.nonzero %[[CST]] : !torch.vtensor<[3],si64> -> !torch.vtensor<[2],si64>
  // CHECK: %[[DIM:.*]] = torch.constant.int 1
  // CHECK: %[[ATEN_INDEX_SELECT:.*]] = torch.aten.index_select %arg0, %[[DIM]], %[[INDEX]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[2,2,4],f32>
  %cst = torch.vtensor.literal(dense<[0,1,1]> : tensor<3xsi64>) : !torch.vtensor<[3], si64>
  %0 = torch.operator "onnx.Compress"(%arg0, %cst) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,2,4],f32>
  return %0 : !torch.vtensor<[2,2,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_einsum_batch_diagonal
func.func @test_einsum_batch_diagonal(%arg0: !torch.vtensor<[3,5,5],f64>) -> !torch.vtensor<[3,5],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS:.*]] = torch.prim.ListConstruct %arg0 : (!torch.vtensor<[3,5,5],f64>) -> !torch.list<vtensor>
  // CHECK: %[[EQUATION:.*]] = torch.constant.str "...ii ->...i"
  // CHECK: %[[PATH:.*]] = torch.constant.none
  // CHECK: torch.aten.einsum %[[EQUATION]], %[[TENSORS]], %[[PATH]] : !torch.str, !torch.list<vtensor>, !torch.none -> !torch.vtensor<[3,5],f64>
  %0 = torch.operator "onnx.Einsum"(%arg0) {torch.onnx.equation = "...ii ->...i"} : (!torch.vtensor<[3,5,5],f64>) -> !torch.vtensor<[3,5],f64>
  return %0 : !torch.vtensor<[3,5],f64>
}

// -----

// CHECK-LABEL: func.func @test_einsum_batch_matmul
func.func @test_einsum_batch_matmul(%arg0: !torch.vtensor<[5,2,3],f64>, %arg1: !torch.vtensor<[5,3,4],f64>) -> !torch.vtensor<[5,2,4],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[5,2,3],f64>, !torch.vtensor<[5,3,4],f64>) -> !torch.list<vtensor>
  // CHECK: %[[EQUATION:.*]] = torch.constant.str "bij, bjk -> bik"
  // CHECK: %[[PATH:.*]] = torch.constant.none
  // CHECK: torch.aten.einsum %[[EQUATION]], %[[TENSORS]], %[[PATH]] : !torch.str, !torch.list<vtensor>, !torch.none -> !torch.vtensor<[5,2,4],f64>
  %0 = torch.operator "onnx.Einsum"(%arg0, %arg1) {torch.onnx.equation = "bij, bjk -> bik"} : (!torch.vtensor<[5,2,3],f64>, !torch.vtensor<[5,3,4],f64>) -> !torch.vtensor<[5,2,4],f64>
  return %0 : !torch.vtensor<[5,2,4],f64>
}

// -----

// CHECK-LABEL: func.func @test_einsum_sum
func.func @test_einsum_sum(%arg0: !torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS:.*]] = torch.prim.ListConstruct %arg0 : (!torch.vtensor<[3,4],f64>) -> !torch.list<vtensor>
  // CHECK: %[[EQUATION:.*]] = torch.constant.str "ij->i"
  // CHECK: %[[PATH:.*]] = torch.constant.none
  // CHECK: torch.aten.einsum %[[EQUATION]], %[[TENSORS]], %[[PATH]] : !torch.str, !torch.list<vtensor>, !torch.none -> !torch.vtensor<[3],f64>
  %0 = torch.operator "onnx.Einsum"(%arg0) {torch.onnx.equation = "ij->i"} : (!torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3],f64>
  return %0 : !torch.vtensor<[3],f64>
}

// -----

// CHECK-LABEL: func.func @test_einsum_transpose
func.func @test_einsum_transpose(%arg0: !torch.vtensor<[3,4],f64>) -> !torch.vtensor<[4,3],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS:.*]] = torch.prim.ListConstruct %arg0 : (!torch.vtensor<[3,4],f64>) -> !torch.list<vtensor>
  // CHECK: %[[EQUATION:.*]] = torch.constant.str "ij->ji"
  // CHECK: %[[PATH:.*]] = torch.constant.none
  // CHECK: torch.aten.einsum %[[EQUATION]], %[[TENSORS]], %[[PATH]] : !torch.str, !torch.list<vtensor>, !torch.none -> !torch.vtensor<[4,3],f64>
  %0 = torch.operator "onnx.Einsum"(%arg0) {torch.onnx.equation = "ij->ji"} : (!torch.vtensor<[3,4],f64>) -> !torch.vtensor<[4,3],f64>
  return %0 : !torch.vtensor<[4,3],f64>
}

// -----

// CHECK-LABEL: func.func @test_eyelike_m
func.func @test_eyelike_m(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DIM0:.*]] = torch.aten.size.int %arg0, %[[INT0]] : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.int
  // CHECK: %[[DIM1:.*]] = torch.aten.size.int %arg0, %[[INT1]] : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.eye.m %[[DIM0]], %[[DIM1]], %[[DTYPE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.EyeLike"(%arg0) : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_eyelike_int
func.func @test_eyelike_int(%arg0: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[3,3], si64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DIM0:.*]] = torch.aten.size.int %arg0, %[[INT0]] : !torch.vtensor<[3,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[DIM1:.*]] = torch.aten.size.int %arg0, %[[INT1]] : !torch.vtensor<[3,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 4
  // CHECK: torch.aten.eye.m %[[DIM0]], %[[DIM1]], %[[DTYPE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,3],si64>
  %0 = torch.operator "onnx.EyeLike"(%arg0) {torch.onnx.dtype = 7 : si64} : (!torch.vtensor<[3,3],f32>) -> !torch.vtensor<[3,3],si64>
  return %0 : !torch.vtensor<[3,3],si64>
}

// -----

// CHECK-LABEL: func.func @test_eyelike_diagonal
func.func @test_eyelike_diagonal(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DIM0:.*]] = torch.aten.size.int %arg0, %[[INT0]] : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.int
  // CHECK: %[[DIM1:.*]] = torch.aten.size.int %arg0, %[[INT1]] : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: %[[DIAG:.*]] = torch.constant.int 1
  // CHECK: %[[NEW_DIM:.*]] = torch.aten.sub.int %[[DIM1]], %[[DIAG]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[EYE:.*]] = torch.aten.eye.m %[[DIM0]], %[[NEW_DIM]], %[[DTYPE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,3],f32>
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[ZEROS:.*]] = torch.aten.zeros %[[SHAPE]], %[[DTYPE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.slice_scatter %[[ZEROS]], %[[EYE]], %[[INT1]], %[[DIAG]], %[[DIM1]], %[[INT1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.EyeLike"(%arg0) {torch.onnx.k = 1 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_eyelike_dynamic
func.func @test_eyelike_dynamic(%arg0: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[3,?], f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DIM0:.*]] = torch.aten.size.int %arg0, %[[INT0]] : !torch.vtensor<[3,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[DIM1:.*]] = torch.aten.size.int %arg0, %[[INT1]] : !torch.vtensor<[3,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: %[[DIAG:.*]] = torch.constant.int 1
  // CHECK: %[[NEW_DIM:.*]] = torch.aten.sub.int %[[DIM0]], %[[DIAG]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[EYE:.*]] = torch.aten.eye.m %[[NEW_DIM]], %[[DIM1]], %[[DTYPE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,?],f32>
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[ZEROS:.*]] = torch.aten.zeros %[[SHAPE]], %[[DTYPE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,?],f32>
  // CHECK: torch.aten.slice_scatter %[[ZEROS]], %[[EYE]], %[[INT0]], %[[DIAG]], %[[DIM0]], %[[INT1]] : !torch.vtensor<[3,?],f32>, !torch.vtensor<[2,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?],f32>
  %0 = torch.operator "onnx.EyeLike"(%arg0) {torch.onnx.k = -1 : si64} : (!torch.vtensor<[3,?],f32>) -> !torch.vtensor<[3,?],f32>
  return %0 : !torch.vtensor<[3,?],f32>
}

// -----

// CHECK-LABEL: func.func @test_blackmanwindow_symmetric
func.func @test_blackmanwindow_symmetric(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[A0:.+]] = torch.constant.float 4.200000e-01
  // CHECK-DAG: %[[A1:.+]] = torch.constant.float -5.000000e-01
  // CHECK-DAG: %[[A2:.+]] = torch.constant.float 8.000000e-02
  // CHECK-DAG: %[[FLOAT0_0:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[FLOAT1:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[FLOAT2:.+]] = torch.constant.float 2.000000e+00
  // CHECK-DAG: %[[TWOPI:.+]] = torch.constant.float 6.2831853071795862
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[INT6:.+]] = torch.constant.int 6
  // CHECK-DAG: %[[CAST_0:.+]] = torch.aten.to.dtype %arg0, %[[INT6]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMSIZE:.+]] = torch.aten.sub.Scalar %[[CAST_0]], %[[FLOAT1]], %[[FLOAT1]] : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[PERIODIC:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[SYMMETRIC:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[PERIODICCOMP:.+]] = torch.aten.mul.Scalar %[[CAST_0]], %[[PERIODIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMETRICCOMP:.+]] = torch.aten.mul.Scalar %[[SYMMSIZE]], %[[SYMMETRIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SIZEFP:.+]] = torch.aten.add.Tensor %[[SYMMETRICCOMP]], %[[PERIODICCOMP]], %[[FLOAT1]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RANGELIM:.+]] = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[ARANGE:.+]] = torch.aten.arange.start_step %[[FLOAT0_0]], %[[RANGELIM]], %[[FLOAT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGETIMESTAU:.+]] = torch.aten.mul.Scalar %[[ARANGE]], %[[TWOPI]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGEANGULAR:.+]] = torch.aten.div.Tensor %[[RANGETIMESTAU]], %[[SIZEFP]] : !torch.vtensor<[10],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWORANGEANGULAR:.+]] = torch.aten.mul.Scalar %[[RANGEANGULAR]], %[[FLOAT2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSRANGEANGULAR:.+]] = torch.aten.cos %[[RANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSTWORANGEANGULAR:.+]] = torch.aten.cos %[[TWORANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A1COMP:.+]] = torch.aten.mul.Scalar %[[COSRANGEANGULAR]], %[[A1]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A2COMP:.+]] = torch.aten.mul.Scalar %[[COSTWORANGEANGULAR]], %[[A2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RES:.+]] = torch.aten.add.Scalar %[[A1COMP]], %[[A0]], %[[FLOAT1]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RESULT:.+]] = torch.aten.add.Tensor %[[RES]], %[[A2COMP]], %[[FLOAT1]] : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %[[RESULT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK: return %[[CAST]] : !torch.vtensor<[10],f32>
  %0 = torch.operator "onnx.BlackmanWindow"(%arg0) {torch.onnx.periodic = 0 : si64} : (!torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_blackmanwindow
func.func @test_blackmanwindow(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[A0:.+]] = torch.constant.float 4.200000e-01
  // CHECK-DAG: %[[A1:.+]] = torch.constant.float -5.000000e-01
  // CHECK-DAG: %[[A2:.+]] = torch.constant.float 8.000000e-02
  // CHECK-DAG: %[[FLOAT0_0:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[FLOAT1:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[FLOAT2:.+]] = torch.constant.float 2.000000e+00
  // CHECK-DAG: %[[TWOPI:.+]] = torch.constant.float 6.2831853071795862
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[INT6:.+]] = torch.constant.int 6
  // CHECK-DAG: %[[CAST_0:.+]] = torch.aten.to.dtype %arg0, %[[INT6]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMSIZE:.+]] = torch.aten.sub.Scalar %[[CAST_0]], %[[FLOAT1]], %[[FLOAT1]] : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[PERIODIC:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[SYMMETRIC:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[PERIODICCOMP:.+]] = torch.aten.mul.Scalar %[[CAST_0]], %[[PERIODIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMETRICCOMP:.+]] = torch.aten.mul.Scalar %[[SYMMSIZE]], %[[SYMMETRIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SIZEFP:.+]] = torch.aten.add.Tensor %[[SYMMETRICCOMP]], %[[PERIODICCOMP]], %[[FLOAT1]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RANGELIM:.+]] = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[ARANGE:.+]] = torch.aten.arange.start_step %[[FLOAT0_0]], %[[RANGELIM]], %[[FLOAT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGETIMESTAU:.+]] = torch.aten.mul.Scalar %[[ARANGE]], %[[TWOPI]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGEANGULAR:.+]] = torch.aten.div.Tensor %[[RANGETIMESTAU]], %[[SIZEFP]] : !torch.vtensor<[10],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWORANGEANGULAR:.+]] = torch.aten.mul.Scalar %[[RANGEANGULAR]], %[[FLOAT2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSRANGEANGULAR:.+]] = torch.aten.cos %[[RANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSTWORANGEANGULAR:.+]] = torch.aten.cos %[[TWORANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A1COMP:.+]] = torch.aten.mul.Scalar %[[COSRANGEANGULAR]], %[[A1]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A2COMP:.+]] = torch.aten.mul.Scalar %[[COSTWORANGEANGULAR]], %[[A2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RES:.+]] = torch.aten.add.Scalar %[[A1COMP]], %[[A0]], %[[FLOAT1]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RESULT:.+]] = torch.aten.add.Tensor %[[RES]], %[[A2COMP]], %[[FLOAT1]] : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %[[RESULT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK: return %[[CAST]] : !torch.vtensor<[10],f32>
  %0 = torch.operator "onnx.BlackmanWindow"(%arg0) {torch.onnx.periodic = 1 : si64} : (!torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}
// -----

// CHECK-LABEL: func.func @test_hannwindow
func.func @test_hannwindow(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[A0:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[A1:.+]] = torch.constant.float -5.000000e-01
  // CHECK-DAG: %[[A2:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[TWO:.+]] = torch.constant.float 2.000000e+00
  // CHECK-DAG: %[[TAU:.+]] = torch.constant.float 6.2831853071795862
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK-DAG: %[[CAST_0:.+]] = torch.aten.to.dtype %arg0, %[[INT6_0]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMSIZE:.+]] = torch.aten.sub.Scalar %[[CAST_0]], %[[ONE]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[PERIODIC:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[SYMMETRIC:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[PERIODICCOMP:.+]] = torch.aten.mul.Scalar %[[CAST_0]], %[[PERIODIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMETRICCOMP:.+]] = torch.aten.mul.Scalar %[[SYMMSIZE]], %[[SYMMETRIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SIZEFP:.+]] = torch.aten.add.Tensor %[[SYMMETRICCOMP]], %[[PERIODICCOMP]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RANGELIM:.+]] = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[ARANGE:.+]] = torch.aten.arange.start_step %[[ZERO]], %[[RANGELIM]], %[[ONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGETIMESTAU:.+]] = torch.aten.mul.Scalar %[[ARANGE]], %[[TAU]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGEANGULAR:.+]] = torch.aten.div.Tensor %[[RANGETIMESTAU]], %[[SIZEFP]] : !torch.vtensor<[10],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWORANGEANGULAR:.+]] = torch.aten.mul.Scalar %[[RANGEANGULAR]], %[[TWO]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSRANGEANGULAR:.+]] = torch.aten.cos %[[RANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWOCOSRANGEANGULAR:.+]] = torch.aten.cos %[[TWORANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A1COMP:.+]] = torch.aten.mul.Scalar %[[COSRANGEANGULAR]], %[[A1]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A2COMP:.+]] = torch.aten.mul.Scalar %[[TWOCOSRANGEANGULAR]], %[[A2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RES:.+]] = torch.aten.add.Scalar %[[A1COMP]], %[[A0]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RESULT:.+]] = torch.aten.add.Tensor %[[RES]], %[[A2COMP]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CAST_1:.+]] = torch.aten.to.dtype %[[RESULT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK: return %[[CAST_1]] : !torch.vtensor<[10],f32>

  %0 = torch.operator "onnx.HannWindow"(%arg0) {torch.onnx.periodic = 1 : si64} : (!torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_hannwindow_symmetric
func.func @test_hannwindow_symmetric(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[A0:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[A1:.+]] = torch.constant.float -5.000000e-01
  // CHECK-DAG: %[[A2:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[TWO:.+]] = torch.constant.float 2.000000e+00
  // CHECK-DAG: %[[TAU:.+]] = torch.constant.float 6.2831853071795862
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK-DAG: %[[CAST_0:.+]] = torch.aten.to.dtype %arg0, %[[INT6_0]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMSIZE:.+]] = torch.aten.sub.Scalar %[[CAST_0]], %[[ONE]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[PERIODIC:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[SYMMETRIC:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[PERIODICCOMP:.+]] = torch.aten.mul.Scalar %[[CAST_0]], %[[PERIODIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMETRICCOMP:.+]] = torch.aten.mul.Scalar %[[SYMMSIZE]], %[[SYMMETRIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SIZEFP:.+]] = torch.aten.add.Tensor %[[SYMMETRICCOMP]], %[[PERIODICCOMP]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RANGELIM:.+]] = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[ARANGE:.+]] = torch.aten.arange.start_step %[[ZERO]], %[[RANGELIM]], %[[ONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGETIMESTAU:.+]] = torch.aten.mul.Scalar %[[ARANGE]], %[[TAU]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGEANGULAR:.+]] = torch.aten.div.Tensor %[[RANGETIMESTAU]], %[[SIZEFP]] : !torch.vtensor<[10],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWORANGEANGULAR:.+]] = torch.aten.mul.Scalar %[[RANGEANGULAR]], %[[TWO]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSRANGEANGULAR:.+]] = torch.aten.cos %[[RANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWOCOSRANGEANGULAR:.+]] = torch.aten.cos %[[TWORANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A1COMP:.+]] = torch.aten.mul.Scalar %[[COSRANGEANGULAR]], %[[A1]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A2COMP:.+]] = torch.aten.mul.Scalar %[[TWOCOSRANGEANGULAR]], %[[A2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RES:.+]] = torch.aten.add.Scalar %[[A1COMP]], %[[A0]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RESULT:.+]] = torch.aten.add.Tensor %[[RES]], %[[A2COMP]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CAST_1:.+]] = torch.aten.to.dtype %[[RESULT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK: return %[[CAST_1]] : !torch.vtensor<[10],f32>

  %0 = torch.operator "onnx.HannWindow"(%arg0) {torch.onnx.periodic = 0 : si64} : (!torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_hammingwindow_symmetric
func.func @test_hammingwindow_symmetric(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[A0:.+]] = torch.constant.float 5.434780e-01
  // CHECK-DAG: %[[A1:.+]] = torch.constant.float -4.565220e-01
  // CHECK-DAG: %[[A2:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[TWO:.+]] = torch.constant.float 2.000000e+00
  // CHECK-DAG: %[[TAU:.+]] = torch.constant.float 6.2831853071795862
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK-DAG: %[[CAST_0:.+]] = torch.aten.to.dtype %arg0, %[[INT6_0]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMSIZE:.+]] = torch.aten.sub.Scalar %[[CAST_0]], %[[ONE]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[PERIODIC:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[SYMMETRIC:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[PERIODICCOMP:.+]] = torch.aten.mul.Scalar %[[CAST_0]], %[[PERIODIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMETRICCOMP:.+]] = torch.aten.mul.Scalar %[[SYMMSIZE]], %[[SYMMETRIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SIZEFP:.+]] = torch.aten.add.Tensor %[[SYMMETRICCOMP]], %[[PERIODICCOMP]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RANGELIM:.+]] = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[ARANGE:.+]] = torch.aten.arange.start_step %[[ZERO]], %[[RANGELIM]], %[[ONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGETIMESTAU:.+]] = torch.aten.mul.Scalar %[[ARANGE]], %[[TAU]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGEANGULAR:.+]] = torch.aten.div.Tensor %[[RANGETIMESTAU]], %[[SIZEFP]] : !torch.vtensor<[10],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWORANGEANGULAR:.+]] = torch.aten.mul.Scalar %[[RANGEANGULAR]], %[[TWO]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSRANGEANGULAR:.+]] = torch.aten.cos %[[RANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWOCOSRANGEANGULAR:.+]] = torch.aten.cos %[[TWORANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A1COMP:.+]] = torch.aten.mul.Scalar %[[COSRANGEANGULAR]], %[[A1]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A2COMP:.+]] = torch.aten.mul.Scalar %[[TWOCOSRANGEANGULAR]], %[[A2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RES:.+]] = torch.aten.add.Scalar %[[A1COMP]], %[[A0]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RESULT:.+]] = torch.aten.add.Tensor %[[RES]], %[[A2COMP]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CAST_1:.+]] = torch.aten.to.dtype %[[RESULT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK: return %[[CAST_1]] : !torch.vtensor<[10],f32>

  %0 = torch.operator "onnx.HammingWindow"(%arg0) {torch.onnx.periodic = 0 : si64} : (!torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_hammingwindow
func.func @test_hammingwindow(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[A0:.+]] = torch.constant.float 5.434780e-01
  // CHECK-DAG: %[[A1:.+]] = torch.constant.float -4.565220e-01
  // CHECK-DAG: %[[A2:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[TWO:.+]] = torch.constant.float 2.000000e+00
  // CHECK-DAG: %[[TAU:.+]] = torch.constant.float 6.2831853071795862
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK-DAG: %[[CAST_0:.+]] = torch.aten.to.dtype %arg0, %[[INT6_0]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMSIZE:.+]] = torch.aten.sub.Scalar %[[CAST_0]], %[[ONE]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[PERIODIC:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG: %[[SYMMETRIC:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG: %[[PERIODICCOMP:.+]] = torch.aten.mul.Scalar %[[CAST_0]], %[[PERIODIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SYMMETRICCOMP:.+]] = torch.aten.mul.Scalar %[[SYMMSIZE]], %[[SYMMETRIC]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[SIZEFP:.+]] = torch.aten.add.Tensor %[[SYMMETRICCOMP]], %[[PERIODICCOMP]], %[[ONE]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RANGELIM:.+]] = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK-DAG: %[[ARANGE:.+]] = torch.aten.arange.start_step %[[ZERO]], %[[RANGELIM]], %[[ONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGETIMESTAU:.+]] = torch.aten.mul.Scalar %[[ARANGE]], %[[TAU]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RANGEANGULAR:.+]] = torch.aten.div.Tensor %[[RANGETIMESTAU]], %[[SIZEFP]] : !torch.vtensor<[10],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWORANGEANGULAR:.+]] = torch.aten.mul.Scalar %[[RANGEANGULAR]], %[[TWO]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[COSRANGEANGULAR:.+]] = torch.aten.cos %[[RANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[TWOCOSRANGEANGULAR:.+]] = torch.aten.cos %[[TWORANGEANGULAR]] : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A1COMP:.+]] = torch.aten.mul.Scalar %[[COSRANGEANGULAR]], %[[A1]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[A2COMP:.+]] = torch.aten.mul.Scalar %[[TWOCOSRANGEANGULAR]], %[[A2]] : !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RES:.+]] = torch.aten.add.Scalar %[[A1COMP]], %[[A0]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[RESULT:.+]] = torch.aten.add.Tensor %[[RES]], %[[A2COMP]], %[[ONE]] : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  // CHECK-DAG: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CAST_1:.+]] = torch.aten.to.dtype %[[RESULT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK: return %[[CAST_1]] : !torch.vtensor<[10],f32>

  %0 = torch.operator "onnx.HammingWindow"(%arg0) {torch.onnx.periodic = 1 : si64} : (!torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_col2im
func.func @test_col2im(%arg0: !torch.vtensor<[1,5,5],f32>, %arg1: !torch.vtensor<[2],si64>, %arg2: !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DILATIONSLIST:.*]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[INT1_3:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[STRIDESLIST:.*]] = torch.prim.ListConstruct %[[INT1_2]], %[[INT1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[PADLIST:.*]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_2:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_3:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[NUMTOTENSOR_0:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_0:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_0]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM0:.*]] = torch.aten.item %[[INDEXSELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_1:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_1:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_1:.*]] = torch.aten.item %[[INDEXSELECT_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[INT1_4:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[NUMTOTENSOR_2:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_4]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_2:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_2]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_2:.*]] = torch.aten.item %[[INDEXSELECT_2]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_3:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_4]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_3:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_3]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_3:.*]] = torch.aten.item %[[INDEXSELECT_3]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[IMAGESHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM0]], %[[ITEM_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[BLOCKSHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM_1]], %[[ITEM_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[COL2IM:.*]] = torch.aten.col2im %arg0, %[[IMAGESHAPELIST]], %[[BLOCKSHAPELIST]], %[[DILATIONSLIST]], %[[PADLIST]], %[[STRIDESLIST]] : !torch.vtensor<[1,5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,1,5,5],f32>
  // CHECK-DAG: return %[[COL2IM]] : !torch.vtensor<[1,1,5,5],f32>
  %0 = torch.operator "onnx.Col2Im"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,5,5],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32>
  return %0 : !torch.vtensor<[1,1,5,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_col2im_pads
func.func @test_col2im_pads(%arg0: !torch.vtensor<[1,5,15],f32>, %arg1: !torch.vtensor<[2],si64>, %arg2: !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DILATIONSLIST:.*]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[INT1_3:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[STRIDESLIST:.*]] = torch.prim.ListConstruct %[[INT1_2]], %[[INT1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT1_4:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[PADLIST:.*]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT1_4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_2:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_3:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[NUMTOTENSOR_0:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_0:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_0]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM0:.*]] = torch.aten.item %[[INDEXSELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_1:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_1:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_1:.*]] = torch.aten.item %[[INDEXSELECT_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[INT1_5:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[NUMTOTENSOR_2:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_5]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_2:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_2]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_2:.*]] = torch.aten.item %[[INDEXSELECT_2]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_3:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_5]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_3:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_3]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_3:.*]] = torch.aten.item %[[INDEXSELECT_3]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[IMAGESHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM0]], %[[ITEM_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[BLOCKSHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM_1]], %[[ITEM_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[COL2IM:.*]] = torch.aten.col2im %arg0, %[[IMAGESHAPELIST]], %[[BLOCKSHAPELIST]], %[[DILATIONSLIST]], %[[PADLIST]], %[[STRIDESLIST]] : !torch.vtensor<[1,5,15],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,1,5,5],f32>
  // CHECK: return %[[COL2IM]] : !torch.vtensor<[1,1,5,5],f32>
  %0 = torch.operator "onnx.Col2Im"(%arg0, %arg1, %arg2) {torch.onnx.pads = [0 : si64, 1 : si64, 0 : si64, 1 : si64]} : (!torch.vtensor<[1,5,15],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32>
  return %0 : !torch.vtensor<[1,1,5,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_col2im_dilations
func.func @test_col2im_dilations(%arg0: !torch.vtensor<[1,4,5],f32>, %arg1: !torch.vtensor<[2],si64>, %arg2: !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,6,6],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[INT5_0:.*]] = torch.constant.int 5
  // CHECK-DAG: %[[DILATIONSLIST:.*]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT5_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[STRIDESLIST:.*]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT1_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[PADLIST:.*]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_2:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_3:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[NUMTOTENSOR_0:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_0:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_0]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM0:.*]] = torch.aten.item %[[INDEXSELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_1:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_1:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_1:.*]] = torch.aten.item %[[INDEXSELECT_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[INT1_3:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[NUMTOTENSOR_2:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_2:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_2]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_2:.*]] = torch.aten.item %[[INDEXSELECT_2]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_3:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_3:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_3]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_3:.*]] = torch.aten.item %[[INDEXSELECT_3]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[IMAGESHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM0]], %[[ITEM_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[BLOCKSHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM_1]], %[[ITEM_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[COL2IM:.*]] = torch.aten.col2im %arg0, %[[IMAGESHAPELIST]], %[[BLOCKSHAPELIST]], %[[DILATIONSLIST]], %[[PADLIST]], %[[STRIDESLIST]] : !torch.vtensor<[1,4,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,1,6,6],f32>
  // CHECK-DAG: return %[[COL2IM]] : !torch.vtensor<[1,1,6,6],f32>
  %0 = torch.operator "onnx.Col2Im"(%arg0, %arg1, %arg2) {torch.onnx.dilations = [1 : si64, 5 : si64]} : (!torch.vtensor<[1,4,5],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,6,6],f32>
  return %0 : !torch.vtensor<[1,1,6,6],f32>
}

// -----

// CHECK-LABEL: func.func @test_col2im_strides
func.func @test_col2im_strides(%arg0: !torch.vtensor<[1,9,4],f32>, %arg1: !torch.vtensor<[2],si64>, %arg2: !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DILATIONSLIST:.*]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT2_0:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[INT2_1:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[STRIDESLIST:.*]] = torch.prim.ListConstruct %[[INT2_0]], %[[INT2_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[PADLIST:.*]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[INT0_2:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[INT0_3:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[NUMTOTENSOR_0:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_0:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_0]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM0:.*]] = torch.aten.item %[[INDEXSELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_1:.*]] = torch.prim.NumToTensor.Scalar %[[INT0_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_1:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_1:.*]] = torch.aten.item %[[INDEXSELECT_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[NUMTOTENSOR_2:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_2]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_2:.*]] = torch.aten.index_select %arg1, %[[INT0_2]], %[[NUMTOTENSOR_2]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_2:.*]] = torch.aten.item %[[INDEXSELECT_2]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[NUMTOTENSOR_3:.*]] = torch.prim.NumToTensor.Scalar %[[INT1_2]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[INDEXSELECT_3:.*]] = torch.aten.index_select %arg2, %[[INT0_2]], %[[NUMTOTENSOR_3]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK-DAG: %[[ITEM_3:.*]] = torch.aten.item %[[INDEXSELECT_3]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK-DAG: %[[IMAGESHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM0]], %[[ITEM_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[BLOCKSHAPELIST:.*]] = torch.prim.ListConstruct %[[ITEM_1]], %[[ITEM_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[COL2IM:.*]] = torch.aten.col2im %arg0, %[[IMAGESHAPELIST]], %[[BLOCKSHAPELIST]], %[[DILATIONSLIST]], %[[PADLIST]], %[[STRIDESLIST]] : !torch.vtensor<[1,9,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,1,5,5],f32>
  // CHECK-DAG: return %[[COL2IM]] : !torch.vtensor<[1,1,5,5],f32>
  %0 = torch.operator "onnx.Col2Im"(%arg0, %arg1, %arg2) {torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,9,4],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32>
  return %0 : !torch.vtensor<[1,1,5,5],f32>
}

// -----

// CHECK-LABEL: @test_center_crop_pad_crop_and_pad
func.func @test_center_crop_pad_crop_and_pad(%arg0: !torch.vtensor<[20,8,3],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[10,10,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[STR:.*]] = torch.constant.str "floor"
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[INDEX:.*]] = torch.prim.NumToTensor.Scalar %[[C0_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[C0_2:.*]] = torch.constant.int 0
  // CHECK: %[[SIZE:.*]] = torch.aten.size.int %arg0, %[[C0_2]] : !torch.vtensor<[20,8,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[SUB:.*]] = torch.aten.sub.int %[[SIZE]], %[[ITEM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SUB_TENSOR:.*]] = torch.prim.NumToTensor.Scalar %[[SUB]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIV:.*]] = torch.aten.div.Scalar_mode %[[SUB_TENSOR]], %[[C2]], %[[STR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.str -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[DIV]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[ITEM_0]], %[[ITEM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %arg0, %[[C0_0]], %[[ITEM_0]], %[[ADD]], %[[C1]] : !torch.vtensor<[20,8,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,3],f32>
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[INDEX_0:.*]] = torch.prim.NumToTensor.Scalar %[[C1_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_0:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_0]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_1:.*]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[SIZE_0:.*]] = torch.aten.size.int %[[SLICE]], %[[C1_2]] : !torch.vtensor<[?,?,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[SUB_0:.*]] = torch.aten.sub.int %[[ITEM_1]], %[[SIZE_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SUB_TENSOR_0:.*]] = torch.prim.NumToTensor.Scalar %[[SUB_0]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIV_0:.*]] = torch.aten.div.Scalar_mode %[[SUB_TENSOR_0]], %[[C2]], %[[STR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.str -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_2:.*]] = torch.aten.item %[[DIV_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[ADD_0:.*]] = torch.aten.add.int %[[ITEM_2]], %[[SIZE_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[C0_3:.*]] = torch.constant.int 0
  // CHECK: %[[SIZE_1:.*]] = torch.aten.size.int %[[SLICE]], %[[C0_3]] : !torch.vtensor<[?,?,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[SIZE_2:.*]] = torch.aten.size.int %[[SLICE]], %[[C2_0]] : !torch.vtensor<[?,?,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[SIZE_1]], %[[ITEM_1]], %[[SIZE_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[ZEROS:.*]] = torch.aten.zeros %[[SHAPE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10,10,3],f32>
  // CHECK: torch.aten.slice_scatter %[[ZEROS]], %[[SLICE]], %[[C1_0]], %[[ITEM_2]], %[[ADD_0]], %[[C1]] : !torch.vtensor<[10,10,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[10,10,3],f32>
  %0 = torch.operator "onnx.CenterCropPad"(%arg0, %arg1) : (!torch.vtensor<[20,8,3],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[10,10,3],f32>
  return %0 : !torch.vtensor<[10,10,3],f32>
}

// -----

// CHECK-LABEL: @test_center_crop_pad_crop_axes_chw
func.func @test_center_crop_pad_crop_axes_chw(%arg0: !torch.vtensor<[3,20,8],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,10,9],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[STR:.*]] = torch.constant.str "floor"
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[INDEX:.*]] = torch.prim.NumToTensor.Scalar %[[C0_0]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[SIZE:.*]] = torch.aten.size.int %arg0, %[[C1_1]] : !torch.vtensor<[3,20,8],f32>, !torch.int -> !torch.int
  // CHECK: %[[SUB:.*]] = torch.aten.sub.int %[[SIZE]], %[[ITEM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SUB_TENSOR:.*]] = torch.prim.NumToTensor.Scalar %[[SUB]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIV:.*]] = torch.aten.div.Scalar_mode %[[SUB_TENSOR]], %[[C2]], %[[STR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.str -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[DIV]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[ITEM_0]], %[[ITEM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %arg0, %[[C1_0]], %[[ITEM_0]], %[[ADD]], %[[C1]] : !torch.vtensor<[3,20,8],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?,?],f32>
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[INDEX_0:.*]] = torch.prim.NumToTensor.Scalar %[[C1_2]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_0:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_0]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_1:.*]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[C2_1:.*]] = torch.constant.int 2
  // CHECK: %[[SIZE_0:.*]] = torch.aten.size.int %[[SLICE]], %[[C2_1]] : !torch.vtensor<[3,?,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[SUB_0:.*]] = torch.aten.sub.int %[[ITEM_1]], %[[SIZE_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SUB_TENSOR_0:.*]] = torch.prim.NumToTensor.Scalar %[[SUB_0]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIV_0:.*]] = torch.aten.div.Scalar_mode %[[SUB_TENSOR_0]], %[[C2]], %[[STR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.str -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_2:.*]] = torch.aten.item %[[DIV_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[ADD_0:.*]] = torch.aten.add.int %[[ITEM_2]], %[[SIZE_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[SIZE_1:.*]] = torch.aten.size.int %[[SLICE]], %[[C0_1]] : !torch.vtensor<[3,?,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[C1_3:.*]] = torch.constant.int 1
  // CHECK: %[[SIZE_2:.*]] = torch.aten.size.int %[[SLICE]], %[[C1_3]] : !torch.vtensor<[3,?,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[SIZE_1]], %[[SIZE_2]], %[[ITEM_1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[ZEROS:.*]] = torch.aten.zeros %[[SHAPE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,10,9],f32>
  // CHECK: torch.aten.slice_scatter %[[ZEROS]], %[[SLICE]], %[[C2_0]], %[[ITEM_2]], %[[ADD_0]], %[[C1]] : !torch.vtensor<[3,10,9],f32>, !torch.vtensor<[3,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,10,9],f32>
  %0 = torch.operator "onnx.CenterCropPad"(%arg0, %arg1) {torch.onnx.axes = [1 : si64, 2 : si64]} : (!torch.vtensor<[3,20,8],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,10,9],f32>
  return %0 : !torch.vtensor<[3,10,9],f32>
}

// -----

// CHECK-LABEL: @test_center_crop_pad_crop_negative_axes_hwc
func.func @test_center_crop_pad_crop_negative_axes_hwc(%arg0: !torch.vtensor<[20,8,3],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[10,9,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[STR:.*]] = torch.constant.str "floor"
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[INDEX:.*]] = torch.prim.NumToTensor.Scalar %[[C0_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[C0_2:.*]] = torch.constant.int 0
  // CHECK: %[[SIZE:.*]] = torch.aten.size.int %arg0, %[[C0_2]] : !torch.vtensor<[20,8,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[SUB:.*]] = torch.aten.sub.int %[[SIZE]], %[[ITEM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SUB_TENSOR:.*]] = torch.prim.NumToTensor.Scalar %[[SUB]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIV:.*]] = torch.aten.div.Scalar_mode %[[SUB_TENSOR]], %[[C2]], %[[STR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.str -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[DIV]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[ITEM_0]], %[[ITEM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %arg0, %[[C0_0]], %[[ITEM_0]], %[[ADD]], %[[C1]] : !torch.vtensor<[20,8,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,3],f32>
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[INDEX_0:.*]] = torch.prim.NumToTensor.Scalar %[[C1_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_0:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_0]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_1:.*]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[SIZE_0:.*]] = torch.aten.size.int %[[SLICE]], %[[C1_2]] : !torch.vtensor<[?,?,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[SUB_0:.*]] = torch.aten.sub.int %[[ITEM_1]], %[[SIZE_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SUB_TENSOR_0:.*]] = torch.prim.NumToTensor.Scalar %[[SUB_0]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIV_0:.*]] = torch.aten.div.Scalar_mode %[[SUB_TENSOR_0]], %[[C2]], %[[STR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.str -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_2:.*]] = torch.aten.item %[[DIV_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[ADD_0:.*]] = torch.aten.add.int %[[ITEM_2]], %[[SIZE_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[C0_3:.*]] = torch.constant.int 0
  // CHECK: %[[SIZE_1:.*]] = torch.aten.size.int %[[SLICE]], %[[C0_3]] : !torch.vtensor<[?,?,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[SIZE_2:.*]] = torch.aten.size.int %[[SLICE]], %[[C2_0]] : !torch.vtensor<[?,?,3],f32>, !torch.int -> !torch.int
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[SIZE_1]], %[[ITEM_1]], %[[SIZE_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[ZEROS:.*]] = torch.aten.zeros %[[SHAPE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10,9,3],f32>
  // CHECK: torch.aten.slice_scatter %[[ZEROS]], %[[SLICE]], %[[C1_0]], %[[ITEM_2]], %[[ADD_0]], %[[C1]] : !torch.vtensor<[10,9,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[10,9,3],f32>
  %0 = torch.operator "onnx.CenterCropPad"(%arg0, %arg1) {torch.onnx.axes = [-3 : si64, -2 : si64]} : (!torch.vtensor<[20,8,3],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[10,9,3],f32>
  return %0 : !torch.vtensor<[10,9,3],f32>
}

// -----

// CHECK-LABEL:   func.func @test_dft_fft
func.func @test_dft_fft(%arg0: !torch.vtensor<[10,10,1],f32>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,10,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "dft_example", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME:                            %[[INPUT_SIGNAL:.*]]: !torch.vtensor<[10,10,1],f32>,
  // CHECK-SAME:                            %[[AXIS:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,10,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "dft_example", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[NONE:.*]] = torch.constant.none
  // CHECK:           %[[SIG_LEN:.*]] = torch.constant.none
  // CHECK:           %[[DIM:.*]] = torch.aten.item %[[AXIS]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:           %[[NORM:.*]] = torch.constant.str "backward"
  // CHECK:           %[[FILL_VAL:.*]] = torch.constant.float 0.000000e+00
  // CHECK:           %[[ONE:.*]] = torch.constant.int 1
  // CHECK:           %[[ZERO:.*]] = torch.constant.int 0
  // CHECK:           %[[PAD_DIM_LIST:.*]] = torch.prim.ListConstruct %[[ZERO]], %[[ONE]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[MODE:.*]] = torch.constant.str "constant"
  // CHECK:           %[[INPUT_PADDED:.*]] = torch.aten.pad %[[INPUT_SIGNAL]], %[[PAD_DIM_LIST]], %[[MODE]], %[[FILL_VAL]] : !torch.vtensor<[10,10,1],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[10,10,2],f32>
  // CHECK:           %[[INPUT_T_CMPLX:.*]] = torch.aten.view_as_complex %[[INPUT_PADDED]] : !torch.vtensor<[10,10,2],f32> -> !torch.vtensor<[10,10],complex<f32>>
  // CHECK:           %[[FFT_CMPLX:.*]] = torch.aten.fft_fft %[[INPUT_T_CMPLX]], %[[SIG_LEN]], %[[DIM]], %[[NORM]] : !torch.vtensor<[10,10],complex<f32>>, !torch.none, !torch.int, !torch.str -> !torch.vtensor<[10,10],complex<f32>>
  // CHECK:           %[[FFT_RES_REAL:.*]] = torch.aten.view_as_real %[[FFT_CMPLX]] : !torch.vtensor<[10,10],complex<f32>> -> !torch.vtensor<[10,10,2],f32>
  // CHECK:           return %[[FFT_RES_REAL]] : !torch.vtensor<[10,10,2],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.DFT"(%arg0, %none, %arg1) : (!torch.vtensor<[10,10,1],f32>, !torch.none, !torch.vtensor<[],si64>) -> !torch.vtensor<[10,10,2],f32>
  return %0 : !torch.vtensor<[10,10,2],f32>
}

// -----

// CHECK-LABEL:   func.func @test_dft_inverse_real
func.func @test_dft_inverse_real(%arg0: !torch.vtensor<[10,10,1],f32>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,10,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "dft_example", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME:                                     %[[INPUT_SIGNAL:.*]]: !torch.vtensor<[10,10,1],f32>,
  // CHECK-SAME:                                     %[[AXIS:.*]]: !torch.vtensor<[],si64>) -> !torch.vtensor<[10,10,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "dft_example", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[NONE:.*]] = torch.constant.none
  // CHECK:           %[[SIG_LEN:.*]] = torch.constant.none
  // CHECK:           %[[DIM:.*]] = torch.aten.item %[[AXIS]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:           %[[NORM:.*]] = torch.constant.str "backward"
  // CHECK:           %[[FILL_VAL:.*]] = torch.constant.float 0.000000e+00
  // CHECK:           %[[ONE:.*]] = torch.constant.int 1
  // CHECK:           %[[ZERO:.*]] = torch.constant.int 0
  // CHECK:           %[[PAD_DIM_LIST:.*]] = torch.prim.ListConstruct %[[ZERO]], %[[ONE]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[MODE:.*]] = torch.constant.str "constant"
  // CHECK:           %[[INPUT_PADDED:.*]] = torch.aten.pad %[[INPUT_SIGNAL]], %[[PAD_DIM_LIST]], %[[MODE]], %[[FILL_VAL]] : !torch.vtensor<[10,10,1],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[10,10,2],f32>
  // CHECK:           %[[INPUT_T_CMPLX:.*]] = torch.aten.view_as_complex %[[INPUT_PADDED]] : !torch.vtensor<[10,10,2],f32> -> !torch.vtensor<[10,10],complex<f32>>
  // CHECK:           %[[IFFT_CMPLX:.*]] = torch.aten.fft_ifft %[[INPUT_T_CMPLX]], %[[SIG_LEN]], %[[DIM]], %[[NORM]] : !torch.vtensor<[10,10],complex<f32>>, !torch.none, !torch.int, !torch.str -> !torch.vtensor<[10,10],complex<f32>>
  // CHECK:           %[[IFFT_RES_REAL:.*]] = torch.aten.view_as_real %[[IFFT_CMPLX]] : !torch.vtensor<[10,10],complex<f32>> -> !torch.vtensor<[10,10,2],f32>
  // CHECK:           return %[[IFFT_RES_REAL]] : !torch.vtensor<[10,10,2],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.DFT"(%arg0, %none, %arg1) {torch.onnx.inverse = 1 : si64} : (!torch.vtensor<[10,10,1],f32>, !torch.none, !torch.vtensor<[],si64>) -> !torch.vtensor<[10,10,2],f32>
  return %0 : !torch.vtensor<[10,10,2],f32>
}
