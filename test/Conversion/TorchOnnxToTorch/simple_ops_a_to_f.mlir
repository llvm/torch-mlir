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
