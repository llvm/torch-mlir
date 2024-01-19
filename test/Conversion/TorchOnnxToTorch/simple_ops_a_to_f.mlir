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

// CHECK-LABEL: @test_atan
func.func @test_atan(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.atan %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Atan"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_acos
func.func @test_acos(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.acos %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Acos"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_bernoulli
func.func @test_bernoulli(%arg0: !torch.vtensor<[10],f64>) -> !torch.vtensor<[10],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %0 = torch.aten.bernoulli %arg0, %[[NONE]] : !torch.vtensor<[10],f64>, !torch.none -> !torch.vtensor<[10],f64>
  %0 = torch.operator "onnx.Bernoulli"(%arg0) : (!torch.vtensor<[10],f64>) -> !torch.vtensor<[10],f64>
  return %0 : !torch.vtensor<[10],f64>
}

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

// CHECK-LABEL: @test_bitshift_left_uint8
func.func @test_bitshift_left_uint8(%arg0: !torch.vtensor<[3],ui8>, %arg1: !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8> -> !torch.vtensor<[3],ui8>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8>
  return %0 : !torch.vtensor<[3],ui8>
}

// CHECK-LABEL: @test_bitshift_left_uint16
func.func @test_bitshift_left_uint16(%arg0: !torch.vtensor<[3],ui16>, %arg1: !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16> -> !torch.vtensor<[3],ui16>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16>
  return %0 : !torch.vtensor<[3],ui16>
}

// CHECK-LABEL: @test_bitshift_left_uint32
func.func @test_bitshift_left_uint32(%arg0: !torch.vtensor<[3],ui32>, %arg1: !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32> -> !torch.vtensor<[3],ui32>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32>
  return %0 : !torch.vtensor<[3],ui32>
}

// CHECK-LABEL: @test_bitshift_left_uint64
func.func @test_bitshift_left_uint64(%arg0: !torch.vtensor<[3],ui64>, %arg1: !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_left_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64> -> !torch.vtensor<[3],ui64>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64>
  return %0 : !torch.vtensor<[3],ui64>
}

// CHECK-LABEL: @test_bitshift_right_uint8
func.func @test_bitshift_right_uint8(%arg0: !torch.vtensor<[3],ui8>, %arg1: !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8> -> !torch.vtensor<[3],ui8>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8>
  return %0 : !torch.vtensor<[3],ui8>
}

// CHECK-LABEL: @test_bitshift_right_uint16
func.func @test_bitshift_right_uint16(%arg0: !torch.vtensor<[3],ui16>, %arg1: !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16> -> !torch.vtensor<[3],ui16>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui16>, !torch.vtensor<[3],ui16>) -> !torch.vtensor<[3],ui16>
  return %0 : !torch.vtensor<[3],ui16>
}

// CHECK-LABEL: @test_bitshift_right_uint32
func.func @test_bitshift_right_uint32(%arg0: !torch.vtensor<[3],ui32>, %arg1: !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32> -> !torch.vtensor<[3],ui32>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32>
  return %0 : !torch.vtensor<[3],ui32>
}

// CHECK-LABEL: @test_bitshift_right_uint64
func.func @test_bitshift_right_uint64(%arg0: !torch.vtensor<[3],ui64>, %arg1: !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_right_shift.Tensor %arg0, %arg1 : !torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64> -> !torch.vtensor<[3],ui64>
  %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "RIGHT"} : (!torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64>
  return %0 : !torch.vtensor<[3],ui64>
}

// CHECK-LABEL: @test_bitwise_and_i16_3d
func.func @test_bitwise_and_i16_3d(%arg0: !torch.vtensor<[3,4,5],si16>, %arg1: !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16> -> !torch.vtensor<[3,4,5],si16>
  %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16>
  return %0 : !torch.vtensor<[3,4,5],si16>
}

// CHECK-LABEL: @test_bitwise_and_i32_2d
func.func @test_bitwise_and_i32_2d(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// CHECK-LABEL: @test_bitwise_and_ui8_bcast_4v3d
func.func @test_bitwise_and_ui8_bcast_4v3d(%arg0: !torch.vtensor<[3,4,5,6],ui8>, %arg1: !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// CHECK-LABEL: @test_bitwise_or_i16_4d
func.func @test_bitwise_or_i16_4d(%arg0: !torch.vtensor<[3,4,5,6],si8>, %arg1: !torch.vtensor<[3,4,5,6],si8>) -> !torch.vtensor<[3,4,5,6],si8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],si8>, !torch.vtensor<[3,4,5,6],si8> -> !torch.vtensor<[3,4,5,6],si8>
  %0 = torch.operator "onnx.BitwiseOr"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],si8>, !torch.vtensor<[3,4,5,6],si8>) -> !torch.vtensor<[3,4,5,6],si8>
  return %0 : !torch.vtensor<[3,4,5,6],si8>
}

// CHECK-LABEL: @test_bitwise_or_i32_2d
func.func @test_bitwise_or_i32_2d(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseOr"(%arg0, %arg1) : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// CHECK-LABEL: @test_bitwise_or_ui8_bcast_4v3d
func.func @test_bitwise_or_ui8_bcast_4v3d(%arg0: !torch.vtensor<[3,4,5,6],ui8>, %arg1: !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseOr"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// CHECK-LABEL: @test_bitwise_not_2d
func.func @test_bitwise_not_2d(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_not %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseNot"(%arg0) : (!torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// CHECK-LABEL: @test_bitwise_not_4d
func.func @test_bitwise_not_4d(%arg0: !torch.vtensor<[3,4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_not %arg0 : !torch.vtensor<[3,4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseNot"(%arg0) : (!torch.vtensor<[3,4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// CHECK-LABEL: @test_bitwise_xor_i16_3d
func.func @test_bitwise_xor_i16_3d(%arg0: !torch.vtensor<[3,4,5],si16>, %arg1: !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_xor.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16> -> !torch.vtensor<[3,4,5],si16>
  %0 = torch.operator "onnx.BitwiseXor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16>
  return %0 : !torch.vtensor<[3,4,5],si16>
}

// CHECK-LABEL: @test_bitwise_xor_i32_2d
func.func @test_bitwise_xor_i32_2d(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_xor.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  %0 = torch.operator "onnx.BitwiseXor"(%arg0, %arg1) : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// CHECK-LABEL: @test_bitwise_xor_ui8_bcast_4v3d
func.func @test_bitwise_xor_ui8_bcast_4v3d(%arg0: !torch.vtensor<[3,4,5,6],ui8>, %arg1: !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.bitwise_xor.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8> -> !torch.vtensor<[3,4,5,6],ui8>
  %0 = torch.operator "onnx.BitwiseXor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],ui8>, !torch.vtensor<[4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8>
  return %0 : !torch.vtensor<[3,4,5,6],ui8>
}

// CHECK-LABEL: @test_cast_BFLOAT16_to_FLOAT
func.func @test_cast_BFLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],bf16>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],bf16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],bf16>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// CHECK-LABEL: @test_cast_DOUBLE_to_FLOAT
func.func @test_cast_DOUBLE_to_FLOAT(%arg0: !torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// CHECK-LABEL: @test_cast_DOUBLE_to_FLOAT16
func.func @test_cast_DOUBLE_to_FLOAT16(%arg0: !torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 5
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f16>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 10 : si64} : (!torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f16>
  return %0 : !torch.vtensor<[3,4],f16>
}

// CHECK-LABEL: @test_cast_FLOAT_to_BFLOAT16
func.func @test_cast_FLOAT_to_BFLOAT16(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],bf16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 15
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],bf16>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 16 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],bf16>
  return %0 : !torch.vtensor<[3,4],bf16>
}

// CHECK-LABEL: @test_cast_FLOAT_to_DOUBLE
func.func @test_cast_FLOAT_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 7
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f64>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f64>
  return %0 : !torch.vtensor<[3,4],f64>
}

// CHECK-LABEL: @test_cast_FLOAT_to_FLOAT16
func.func @test_cast_FLOAT_to_FLOAT16(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 5
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f16>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 10 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f16>
  return %0 : !torch.vtensor<[3,4],f16>
}

// CHECK-LABEL: @test_cast_FLOAT16_to_DOUBLE
func.func @test_cast_FLOAT16_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 7
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f64>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f64>
  return %0 : !torch.vtensor<[3,4],f64>
}

// CHECK-LABEL: @test_cast_FLOAT16_to_FLOAT
func.func @test_cast_FLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],f16>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// CHECK-LABEL: @test_castlike_BFLOAT16_to_FLOAT
func.func @test_castlike_BFLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],bf16>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],bf16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],bf16>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// CHECK-LABEL: @test_castlike_DOUBLE_to_FLOAT
func.func @test_castlike_DOUBLE_to_FLOAT(%arg0: !torch.vtensor<[3,4],f64>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f64>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// CHECK-LABEL: @test_castlike_FLOAT_to_DOUBLE
func.func @test_castlike_FLOAT_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],f64>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 7
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f64>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],f64>) -> !torch.vtensor<[3,4],f64>
  return %0 : !torch.vtensor<[3,4],f64>
}

// CHECK-LABEL: @test_castlike_FLOAT16_to_FLOAT
func.func @test_castlike_FLOAT16_to_FLOAT(%arg0: !torch.vtensor<[3,4],f16>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT:.*]] = torch.constant.int 6
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.to.dtype %arg0, %[[INT]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f16>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// CHECK-LABEL: @test_ceil_example
func.func @test_ceil_example(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.ceil %arg0 : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  %0 = torch.operator "onnx.Ceil"(%arg0) : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// CHECK-LABEL: @test_ceil
func.func @test_ceil(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.ceil %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Ceil"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_clip_default_int8_min
func.func @test_clip_default_int8_min(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp_min.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[3,4,5],si8>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8>
  return %0 : !torch.vtensor<[3,4,5],si8>
}

// CHECK-LABEL: @test_clip_default_min
func.func @test_clip_default_min(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp_min.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_clip_example
func.func @test_clip_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// CHECK-LABEL: @test_clip
func.func @test_clip(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.clamp.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Clip"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_cos_example
func.func @test_cos_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.cos %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Cos"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// CHECK-LABEL: @test_cos
func.func @test_cos(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.cos %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Cos"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
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

// CHECK-LABEL: @test_div_example
func.func @test_div_example(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// CHECK-LABEL: @test_div
func.func @test_div(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_div_uint8
func.func @test_div_uint8(%arg0: !torch.vtensor<[3,4,5],ui8>, %arg1: !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8> -> !torch.vtensor<[3,4,5],ui8>
  %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8>
  return %0 : !torch.vtensor<[3,4,5],ui8>
}

// CHECK-LABEL: @test_equal_bcast
func.func @test_equal_bcast(%arg0: !torch.vtensor<[3,4,5],si32>, %arg1: !torch.vtensor<[5],si32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si32>, !torch.vtensor<[5],si32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Equal"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si32>, !torch.vtensor<[5],si32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// CHECK-LABEL: @test_erf
func.func @test_erf(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.erf %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Erf"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_equal
func.func @test_equal(%arg0: !torch.vtensor<[3,4,5],si32>, %arg1: !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Equal"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}


// CHECK-LABEL: @test_floor_example
func.func @test_floor_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.floor %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Floor"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// CHECK-LABEL: @test_floor
func.func @test_floor(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.floor %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Floor"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_averagepool_1d_default
func.func @test_averagepool_1d_default(%arg0: !torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.avg_pool1d %arg0, %0, %2, %1, %false, %true : !torch.vtensor<[1,3,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,3,31],f32>
  %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.kernel_shape = [2 : si64], torch.onnx.count_include_pad = 1 : si64} : (!torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32>
  return %0 : !torch.vtensor<[1,3,31],f32>
}

// CHECK-LABEL: @test_averagepool_2d_ceil
func.func @test_averagepool_2d_ceil(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.avg_pool2d %arg0, %0, %2, %1, %true, %false, %none : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,2,2],f32>
  %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.ceil_mode = 1 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32>
  return %0 : !torch.vtensor<[1,1,2,2],f32>
}

// CHECK-LABEL: @test_averagepool_3d_default
func.func @test_averagepool_3d_default(%arg0: !torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.avg_pool3d %arg0, %0, %2, %1, %false, %false_2, %none : !torch.vtensor<[1,3,32,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,31,31,31],f32>
  %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32>
  return %0 : !torch.vtensor<[1,3,31,31,31],f32>
}

// CHECK-LABEL: @test_conv_with_strides_no_padding
func.func @test_conv_with_strides_no_padding(%arg0: !torch.vtensor<[1,1,7,5],f32>, %arg1: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,3,2],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0_0]] : (!torch.int, !torch.int) -> !torch.list<int>
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

// CHECK-LABEL: @test_conv_with_strides_padding
func.func @test_conv_with_strides_padding(%arg0: !torch.vtensor<[1,1,7,5],f32>, %arg1: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,4,3],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_2:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
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

// CHECK-LABEL: @test_conv_with_bias_strides_padding
func.func @test_conv_with_bias_strides_padding(%arg0: !torch.vtensor<[?,?,224,224],f32>, %arg1: !torch.vtensor<[64,3,7,7],f32>, %arg2: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,64,112,112],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C3]], %[[C3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[DILATIONS:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C2]], %[[C2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[OUTPUT_PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRANSPOSED:.*]] = torch.constant.bool false
  // CHECK: %[[GROUPS:.*]] = torch.constant.int 1
  // CHECK: torch.aten.convolution %arg0, %arg1, %arg2, %[[STRIDE]], %[[PADDING]], %[[DILATIONS]], %[[TRANSPOSED]], %[[OUTPUT_PADDING]], %[[GROUPS]] : !torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,64,112,112],f32>
  %0 = torch.operator "onnx.Conv"(%arg0, %arg1, %arg2) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [7 : si64, 7 : si64], torch.onnx.pads = [3 : si64, 3 : si64, 3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[64,3,7,7],f32>, !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,64,112,112],f32>
  return %0 : !torch.vtensor<[?,64,112,112],f32>
}

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

// CHECK-LABEL: @test_batchnorm_epsilon
func.func @test_batchnorm_epsilon(%arg0: !torch.vtensor<[2,3,4,5],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>, %arg4: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[MOMENTUM:.*]] = torch.constant.float 0.89999997615814208
  // CHECK: %[[EPS:.*]] = torch.constant.float 0.0099999997764825821
  // CHECK: torch.aten.batch_norm %arg0, %arg1, %arg2, %arg3, %arg4, %[[FALSE]], %[[MOMENTUM]], %[[EPS]], %[[FALSE]] : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[2,3,4,5],f32>
  %0 = torch.operator "onnx.BatchNormalization"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.epsilon = 0.00999999977 : f32} : (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32>
  return %0 : !torch.vtensor<[2,3,4,5],f32>
}

// CHECK-LABEL: @test_batchnorm_example
func.func @test_batchnorm_example(%arg0: !torch.vtensor<[2,3,4,5],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>, %arg4: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[MOMENTUM:.*]] = torch.constant.float 0.89999997615814208
  // CHECK: %[[EPS:.*]] = torch.constant.float 9.9999997473787516E-6
  // CHECK: torch.aten.batch_norm %arg0, %arg1, %arg2, %arg3, %arg4, %[[FALSE]], %[[MOMENTUM]], %[[EPS]], %[[FALSE]] : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[2,3,4,5],f32>
  %0 = torch.operator "onnx.BatchNormalization"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3,4,5],f32>
  return %0 : !torch.vtensor<[2,3,4,5],f32>
}

// CHECK-LABEL: @test_concat_1d_axis_0
func.func @test_concat_1d_axis_0(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 0
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// CHECK-LABEL: @test_concat_1d_axis_negative_1
func.func @test_concat_1d_axis_negative_1(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// CHECK-LABEL: @test_concat_2d_axis_0
func.func @test_concat_2d_axis_0(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 0
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// CHECK-LABEL: @test_concat_2d_axis_1
func.func @test_concat_2d_axis_1(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: @test_concat_2d_axis_negative_1
func.func @test_concat_2d_axis_negative_1(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: @test_concat_2d_axis_negative_2
func.func @test_concat_2d_axis_negative_2(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -2
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// CHECK-LABEL: @test_concat_3d_axis_0
func.func @test_concat_3d_axis_0(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 0
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32>
  return %0 : !torch.vtensor<[4,2,2],f32>
}

// CHECK-LABEL: @test_concat_3d_axis_1
func.func @test_concat_3d_axis_1(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32>
  return %0 : !torch.vtensor<[2,4,2],f32>
}

// CHECK-LABEL: @test_concat_3d_axis_2
func.func @test_concat_3d_axis_2(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int 2
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32>
  return %0 : !torch.vtensor<[2,2,4],f32>
}

// CHECK-LABEL: @test_concat_3d_axis_negative_1
func.func @test_concat_3d_axis_negative_1(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -1
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,2,4],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,2,4],f32>
  return %0 : !torch.vtensor<[2,2,4],f32>
}

// CHECK-LABEL: @test_concat_3d_axis_negative_2
func.func @test_concat_3d_axis_negative_2(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -2
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,4,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[2,4,2],f32>
  return %0 : !torch.vtensor<[2,4,2],f32>
}

// CHECK-LABEL: @test_concat_3d_axis_negative_3
func.func @test_concat_3d_axis_negative_3(%arg0: !torch.vtensor<[2,2,2],f32>, %arg1: !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TENSORS_LIST:.*]] = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[DIM:.*]] = torch.constant.int -3
  // CHECK: torch.aten.cat %[[TENSORS_LIST]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,2,2],f32>
  %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -3 : si64} : (!torch.vtensor<[2,2,2],f32>, !torch.vtensor<[2,2,2],f32>) -> !torch.vtensor<[4,2,2],f32>
  return %0 : !torch.vtensor<[4,2,2],f32>
}

// CHECK-LABEL: @test_expand_dim2_shape2
func.func @test_expand_dim2_shape2(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[2],si32>) 
              -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[2],si32>, !torch.int, !torch.int -> !torch.vtensor<[1],si32>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si32> -> !torch.int
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %arg1, %int0, %int1 : !torch.vtensor<[2],si32>, !torch.int, !torch.int -> !torch.vtensor<[1],si32>
  // CHECK: torch.aten.item %2 : !torch.vtensor<[1],si32> -> !torch.int
  // CHECK: torch.prim.ListConstruct %1, %3 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.broadcast_to %arg0, %4 : !torch.vtensor<[1,4],f32>, !torch.list<int> -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Expand"(%arg0, %arg1) : (!torch.vtensor<[1,4],f32>, !torch.vtensor<[2],si32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}
// CHECK-LABEL: @test_expand_dim2_shape3
func.func @test_expand_dim2_shape3(%arg0: !torch.vtensor<[3,1],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,6],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %arg1, %int0, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %2 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %4 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.prim.ListConstruct %1, %3, %5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.broadcast_to %arg0, %6 : !torch.vtensor<[3,1],f32>, !torch.list<int> -> !torch.vtensor<[2,3,6],f32>
  %0 = torch.operator "onnx.Expand"(%arg0, %arg1) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,6],f32>
  return %0 : !torch.vtensor<[2,3,6],f32>
}

// CHECK-LABEL: @test_expand_dim3_shape4
func.func @test_expand_dim3_shape4(%arg0: !torch.vtensor<[1,3,1],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[3,3,3,3],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %arg1, %int0, %int1 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %2 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %4 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: torch.aten.select.int %arg1, %int0, %int3 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %6 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.prim.ListConstruct %1, %3, %5, %7 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %9 = torch.aten.broadcast_to %arg0, %8 : !torch.vtensor<[1,3,1],f32>, !torch.list<int> -> !torch.vtensor<[3,3,3,3],f32>   
  %0 = torch.operator "onnx.Expand"(%arg0, %arg1) : (!torch.vtensor<[1,3,1],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[3,3,3,3],f32>
  return %0 : !torch.vtensor<[3,3,3,3],f32>
} 
// CHECK-LABEL: @test_dropout
func.func @test_dropout(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %float5.000000e-01, %false : !torch.vtensor<[3],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3],f32
  %0 = torch.operator "onnx.Dropout"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// CHECK-LABEL: @test_dropout_default
func.func @test_dropout_default(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %float5.000000e-01, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Dropout"(%arg0) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_dropout_default_mask
func.func @test_dropout_default_mask(%arg0: !torch.vtensor<[3,4,5],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %float5.000000e-01, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.ones_like %arg0, %int11, %none, %none, %none, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4,5],i1>
  %0:2 = torch.operator "onnx.Dropout"(%arg0) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>)
  return %0#0, %0#1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>
}

// CHECK-LABEL: @test_dropout_default_mask_ratio
func.func @test_dropout_default_mask_ratio(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %0, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.ones_like %arg0, %int11, %none, %none, %none, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4,5],i1>
  %0:2 = torch.operator "onnx.Dropout"(%arg0, %arg1) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>)
  return %0#0, %0#1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],i1>
}

// CHECK-LABEL: @test_dropout_default_ratio
func.func @test_dropout_default_ratio(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %0, %false : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Dropout"(%arg0, %arg1) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_training_dropout_zero_ratio
func.func @test_training_dropout_zero_ratio(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],i1>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.dropout %arg0, %0, %2 : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Dropout"(%arg0, %arg1, %arg2) {torch.onnx.seed = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],i1>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_elu_default
func.func @test_elu_default(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.elu %arg0, %float0.000000e00, %float1.000000e00, %float1.000000e00 : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.float, !torch.float -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Elu"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: @test_elu_example
func.func @test_elu_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.elu %arg0, %float2.000000e00, %float1.000000e00, %float1.000000e00 : !torch.vtensor<[3],f32>, !torch.float, !torch.float, !torch.float -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Elu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

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

// CHECK-LABEL: @test_flatten_4d_axis_2
func.func @test_flatten_4d_axis_2(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 1
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor, !torch.int, !torch.int -> !torch.vtensor<[6,20],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32>
  return %0 : !torch.vtensor<[6,20],f32>
}

// CHECK-LABEL: @test_flatten_4d_axis_0
func.func @test_flatten_4d_axis_0(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor, !torch.int -> !torch.vtensor<[1,120],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32>
  return %0 : !torch.vtensor<[1,120],f32>
}

// CHECK-LABEL: @test_flatten_4d_axis_4
func.func @test_flatten_4d_axis_4(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[120,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_INDEX:.*]] = torch.constant.int 4
  // CHECK-DAG: %[[CR:.*]] = torch.aten.unsqueeze %arg0, %[[RIGHT_INDEX]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 3
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor, !torch.int, !torch.int -> !torch.vtensor<[120,1],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 4 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[120,1],f32>
  return %0 : !torch.vtensor<[120,1],f32>
}

// CHECK-LABEL: @test_flatten_4d_axis_negative_2
func.func @test_flatten_4d_axis_negative_2(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 1
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor, !torch.int, !torch.int -> !torch.vtensor<[6,20],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32>
  return %0 : !torch.vtensor<[6,20],f32>
}

// CHECK-LABEL: @test_flatten_4d_axis_negative_1
func.func @test_flatten_4d_axis_negative_1(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[24,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 2
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor, !torch.int, !torch.int -> !torch.vtensor<[24,5],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[24,5],f32>
  return %0 : !torch.vtensor<[24,5],f32>
}

// CHECK-LABEL: @test_flatten_4d_axis_negative_4
func.func @test_flatten_4d_axis_negative_4(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor, !torch.int -> !torch.vtensor<[1,120],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -4 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32>
  return %0 : !torch.vtensor<[1,120],f32>
}

// CHECK-LABEL: @test_flatten_2d_axis_1
func.func @test_flatten_2d_axis_1(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 0
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: @test_flatten_1d_axis_0
func.func @test_flatten_1d_axis_0(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor, !torch.int -> !torch.vtensor<[1,2],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32>
  return %0 : !torch.vtensor<[1,2],f32>
}

// CHECK-LABEL: @test_flatten_1d_axis_negative_1
func.func @test_flatten_1d_axis_negative_1(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[RIGHT_END:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[CR:.*]] = torch.prims.collapse %arg0, %[[RIGHT_START]], %[[RIGHT_END]] : !torch.vtensor<[2],f32>, !torch.int, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_INDEX:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %[[CR]], %[[LEFT_INDEX]] : !torch.vtensor, !torch.int -> !torch.vtensor<[1,2],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32>
  return %0 : !torch.vtensor<[1,2],f32>
}

// COM: CHECK-LABEL: @test_flatten_1d_axis_1
func.func @test_flatten_1d_axis_1(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[RIGHT_INDEX:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[CR:.*]] = torch.aten.unsqueeze %arg0, %[[RIGHT_INDEX]] : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor
  // CHECK-DAG: %[[LEFT_START:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[LEFT_END:.*]] = torch.constant.int 0
  // CHECK: torch.prims.collapse %[[CR]], %[[LEFT_START]], %[[LEFT_END]] : !torch.vtensor, !torch.int, !torch.int -> !torch.vtensor<[2,1],f32>
  %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2,1],f32>
  return %0 : !torch.vtensor<[2,1],f32>
}
