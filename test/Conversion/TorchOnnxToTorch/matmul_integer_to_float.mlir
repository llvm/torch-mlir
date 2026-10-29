// RUN: torch-mlir-opt %s -split-input-file -convert-torch-onnx-to-torch | FileCheck %s --implicit-check-not=torch.operator --implicit-check-not=torch.aten._make_per_tensor_quantized_tensor --implicit-check-not=torch.aten.int_repr --implicit-check-not=torch.quantized_decomposed.quantize_

// MatMulIntegerToFloat consumes integer storage and produces float compute.
// Cover missing, A-only, B-only, and both zero points. Bias is checked once
// independently, with None placeholders preserving the optional input positions.

// CHECK-LABEL: @matmul_integer_to_float_optional_none(
func.func @matmul_integer_to_float_optional_none(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],si8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_optional_a_zp(
func.func @matmul_integer_to_float_optional_a_zp(
    %a: !torch.vtensor<[4,3],ui8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[],f32>,
    %a_zp: !torch.vtensor<[],ui8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[A_MIN:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_MAX:.+]] = torch.constant.int 255{{$}}
  // CHECK: %[[A_DTYPE:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_OUT_DTYPE:.+]] = torch.constant.int 6{{$}}
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.aten.item %arg4 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]], %[[A_MIN]], %[[A_MAX]], %[[A_DTYPE]], %[[A_OUT_DTYPE]]
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_MIN:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[B_MAX:.+]] = torch.constant.int 255{{$}}
  // CHECK: %[[B_DTYPE:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[B_OUT_DTYPE:.+]] = torch.constant.int 6{{$}}
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]], %[[B_MIN]], %[[B_MAX]], %[[B_DTYPE]], %[[B_OUT_DTYPE]]
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp)
      : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_optional_b_zp(
func.func @matmul_integer_to_float_optional_b_zp(
    %a: !torch.vtensor<[4,3],ui8>,
    %b: !torch.vtensor<[3,5],si8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[],f32>,
    %b_zp: !torch.vtensor<[],si8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %none = torch.constant.none
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.aten.item %arg4 : !torch.vtensor<[],si8> -> !torch.int
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %none, %b_zp)
      : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,5],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.vtensor<[],si8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_optional_a_zp_b_zp(
func.func @matmul_integer_to_float_optional_a_zp_b_zp(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[],f32>,
    %a_zp: !torch.vtensor<[],si8>,
    %b_zp: !torch.vtensor<[],ui8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.aten.item %arg4 : !torch.vtensor<[],si8> -> !torch.int
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.aten.item %arg5 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_optional_bias(
func.func @matmul_integer_to_float_optional_bias(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],si8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[],f32>,
    %bias: !torch.vtensor<[5],f32>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %none = torch.constant.none
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NEXT: %[[ONE:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[ADD:.+]] = torch.aten.add.Tensor %[[MM]], %arg4, %[[ONE]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NEXT: return %[[ADD]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %none, %none, %bias)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.vtensor<[5],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_singleton_qparams(
func.func @matmul_integer_to_float_singleton_qparams(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[1],f32>,
    %b_scale: !torch.vtensor<[1],f32>,
    %a_zp: !torch.vtensor<[1],si8>,
    %b_zp: !torch.vtensor<[1],ui8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.aten.item %arg4 : !torch.vtensor<[1],si8> -> !torch.int
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.aten.item %arg5 : !torch.vtensor<[1],ui8> -> !torch.int
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si8>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_per_column_a(
func.func @matmul_integer_to_float_per_column_a(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[3],f32>,
    %b_scale: !torch.vtensor<[],f32>,
    %a_zp: !torch.vtensor<[3],si8>,
    %b_zp: !torch.vtensor<[],ui8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %arg4, %[[A_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.aten.item %arg5 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[3],si8>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_per_column_b(
func.func @matmul_integer_to_float_per_column_b(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[5],f32>,
    %a_zp: !torch.vtensor<[],si8>,
    %b_zp: !torch.vtensor<[5],ui8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.aten.item %arg4 : !torch.vtensor<[],si8> -> !torch.int
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %arg5, %[[B_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[],si8>, !torch.vtensor<[5],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_per_column_both(
func.func @matmul_integer_to_float_per_column_both(
    %a: !torch.vtensor<[4,3],ui8>,
    %b: !torch.vtensor<[3,5],si8>,
    %a_scale: !torch.vtensor<[3],f32>,
    %b_scale: !torch.vtensor<[5],f32>,
    %a_zp: !torch.vtensor<[3],ui8>,
    %b_zp: !torch.vtensor<[5],si8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[A_MIN:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_MAX:.+]] = torch.constant.int 255{{$}}
  // CHECK: %[[A_DTYPE:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_OUT_DTYPE:.+]] = torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %arg4, %[[A_AXIS]], %[[A_MIN]], %[[A_MAX]], %[[A_DTYPE]], %[[A_OUT_DTYPE]]
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: %[[B_MIN:.+]] = torch.constant.int -128{{$}}
  // CHECK: %[[B_MAX:.+]] = torch.constant.int 127{{$}}
  // CHECK: %[[B_DTYPE:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[B_OUT_DTYPE:.+]] = torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %arg5, %[[B_AXIS]], %[[B_MIN]], %[[B_MAX]], %[[B_DTYPE]], %[[B_OUT_DTYPE]]
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,5],si8>, !torch.vtensor<[3],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[3],ui8>, !torch.vtensor<[5],si8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_per_column_default_zp(
func.func @matmul_integer_to_float_per_column_default_zp(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[3],f32>,
    %b_scale: !torch.vtensor<[5],f32>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[A_ZP:.+]] = torch.constant.none
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %[[A_ZP]], %[[A_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[B_ZP:.+]] = torch.constant.none
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %[[B_ZP]], %[[B_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_per_column_scalar_zp(
func.func @matmul_integer_to_float_per_column_scalar_zp(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[3],f32>,
    %b_scale: !torch.vtensor<[5],f32>,
    %a_zp: !torch.vtensor<[],si8>,
    %b_zp: !torch.vtensor<[1],ui8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[A_SIZE_ZP:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK: %[[A_SIZES_ZP:.+]] = torch.prim.ListConstruct %[[A_SIZE_ZP]]
  // CHECK: %[[A_IMPLICIT_ZP:.+]] = torch.constant.bool false
  // CHECK: %[[A_ZP:.+]] = torch.aten.expand %arg4, %[[A_SIZES_ZP]], %[[A_IMPLICIT_ZP]]
  // CHECK-SAME: -> !torch.vtensor<[3],si8>
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %[[A_ZP]], %[[A_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[B_SIZE_ZP:.+]] = torch.aten.size.int %arg1, %[[B_AXIS]]
  // CHECK: %[[B_SIZES_ZP:.+]] = torch.prim.ListConstruct %[[B_SIZE_ZP]]
  // CHECK: %[[B_IMPLICIT_ZP:.+]] = torch.constant.bool false
  // CHECK: %[[B_ZP:.+]] = torch.aten.expand %arg5, %[[B_SIZES_ZP]], %[[B_IMPLICIT_ZP]]
  // CHECK-SAME: -> !torch.vtensor<[5],ui8>
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %[[B_ZP]], %[[B_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[],si8>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_scalar_scale_per_column_zp(
func.func @matmul_integer_to_float_scalar_scale_per_column_zp(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[1],f32>,
    %a_zp: !torch.vtensor<[3],si8>,
    %b_zp: !torch.vtensor<[5],ui8>
    ) -> !torch.vtensor<[4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[A_SIZE:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK: %[[A_SIZES:.+]] = torch.prim.ListConstruct %[[A_SIZE]]
  // CHECK: %[[A_IMPLICIT:.+]] = torch.constant.bool false
  // CHECK: %[[A_SCALE:.+]] = torch.aten.expand %arg2, %[[A_SIZES]], %[[A_IMPLICIT]]
  // CHECK-SAME: -> !torch.vtensor<[3],f32>
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %[[A_SCALE]], %arg4, %[[A_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[4,3],f32>
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[B_SIZE:.+]] = torch.aten.size.int %arg1, %[[B_AXIS]]
  // CHECK: %[[B_SIZES:.+]] = torch.prim.ListConstruct %[[B_SIZE]]
  // CHECK: %[[B_IMPLICIT:.+]] = torch.constant.bool false
  // CHECK: %[[B_SCALE:.+]] = torch.aten.expand %arg3, %[[B_SIZES]], %[[B_IMPLICIT]]
  // CHECK-SAME: -> !torch.vtensor<[5],f32>
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %[[B_SCALE]], %arg5, %[[B_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[4,5],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[3],si8>, !torch.vtensor<[5],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_batched_different_ranks(
func.func @matmul_integer_to_float_batched_different_ranks(
    %a: !torch.vtensor<[2,4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %a_scale: !torch.vtensor<[3],f32>,
    %b_scale: !torch.vtensor<[5],f32>,
    %a_zp: !torch.vtensor<[3],si8>,
    %b_zp: !torch.vtensor<[5],ui8>
    ) -> !torch.vtensor<[2,4,5],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 2{{$}}
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %arg4, %[[A_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[2,4,3],f32>
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %arg5, %[[B_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[2,4,5],f32>
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[2,4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[3],si8>, !torch.vtensor<[5],ui8>) -> !torch.vtensor<[2,4,5],f32>
  return %0 : !torch.vtensor<[2,4,5],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_dynamic_columns(
// Known qparam lengths determine the DQ form; runtime guards constrain the
// otherwise unknown column counts before DQ indexes or expands the qparams.
func.func @matmul_integer_to_float_dynamic_columns(
    %a: !torch.vtensor<[?,?],si8>,
    %b: !torch.vtensor<[?,?],ui8>,
    %a_scale: !torch.vtensor<[3],f32>,
    %b_scale: !torch.vtensor<[5],f32>,
    %a_zp: !torch.vtensor<[],si8>,
    %b_zp: !torch.vtensor<[5],ui8>
    ) -> !torch.vtensor<[?,?],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[A_COLUMNS:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK-NEXT: %[[A_LENGTH:.+]] = torch.constant.int 3{{$}}
  // CHECK-NEXT: %[[A_MATCHES:.+]] = torch.aten.eq.int %[[A_COLUMNS]], %[[A_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[A_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK: %[[A_SIZE_ZP:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK: %[[A_SIZES_ZP:.+]] = torch.prim.ListConstruct %[[A_SIZE_ZP]]
  // CHECK: %[[A_IMPLICIT_ZP:.+]] = torch.constant.bool false
  // CHECK: %[[A_ZP:.+]] = torch.aten.expand %arg4, %[[A_SIZES_ZP]], %[[A_IMPLICIT_ZP]]
  // CHECK-SAME: -> !torch.vtensor<[?],si8>
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %[[A_ZP]], %[[A_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[?,?],f32>
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[B_COLUMNS:.+]] = torch.aten.size.int %arg1, %[[B_AXIS]]
  // CHECK-NEXT: %[[B_LENGTH:.+]] = torch.constant.int 5{{$}}
  // CHECK-NEXT: %[[B_MATCHES:.+]] = torch.aten.eq.int %[[B_COLUMNS]], %[[B_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[B_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK-NEXT: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %arg5, %[[B_AXIS]],
  // CHECK-SAME: -> !torch.vtensor<[?,?],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[?,?],f32>
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[?,?],si8>, !torch.vtensor<[?,?],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[],si8>, !torch.vtensor<[5],ui8>) -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// Scalar qparams apply to any column count, so dynamic matrix dimensions do
// not require a qparam size assertion for per-tensor DQ.
// CHECK-LABEL: @matmul_integer_to_float_dynamic_per_tensor(
func.func @matmul_integer_to_float_dynamic_per_tensor(
    %a: !torch.vtensor<[?,?],si8>,
    %b: !torch.vtensor<[?,?],ui8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[1],f32>
    ) -> !torch.vtensor<[?,?],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK-NOT: torch.runtime.assert
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0,
  // CHECK-NOT: torch.runtime.assert
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1,
  // CHECK-NOT: torch.runtime.assert
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale)
      : (!torch.vtensor<[?,?],si8>, !torch.vtensor<[?,?],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// A's per-column zero points determine the guard even with a scalar scale.
// B's omitted zero point defaults to zero and still needs its scale-size guard.
// CHECK-LABEL: @matmul_integer_to_float_dynamic_column_zero_points(
func.func @matmul_integer_to_float_dynamic_column_zero_points(
    %a: !torch.vtensor<[4,?],si8>,
    %b: !torch.vtensor<[?,?],ui8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[5],f32>,
    %a_zp: !torch.vtensor<[3],si8>
    ) -> !torch.vtensor<[4,?],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[A_COLUMNS:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK-NEXT: %[[A_LENGTH:.+]] = torch.constant.int 3{{$}}
  // CHECK-NEXT: %[[A_MATCHES:.+]] = torch.aten.eq.int %[[A_COLUMNS]], %[[A_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[A_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK: %[[A_SCALE:.+]] = torch.aten.expand %arg2,
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %[[A_SCALE]], %arg4, %[[A_AXIS]],
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[B_COLUMNS:.+]] = torch.aten.size.int %arg1, %[[B_AXIS]]
  // CHECK-NEXT: %[[B_LENGTH:.+]] = torch.constant.int 5{{$}}
  // CHECK-NEXT: %[[B_MATCHES:.+]] = torch.aten.eq.int %[[B_COLUMNS]], %[[B_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[B_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK-NEXT: %[[B_ZP:.+]] = torch.constant.none
  // CHECK-NEXT: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %[[B_ZP]], %[[B_AXIS]],
  // CHECK-NEXT: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp)
      : (!torch.vtensor<[4,?],si8>, !torch.vtensor<[?,?],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[3],si8>) -> !torch.vtensor<[4,?],f32>
  return %0 : !torch.vtensor<[4,?],f32>
}

// -----

// CHECK-LABEL: @matmul_integer_to_float_vector_inputs(
func.func @matmul_integer_to_float_vector_inputs(
    %a: !torch.vtensor<[3],si8>,
    %b: !torch.vtensor<[3],ui8>,
    %a_scale: !torch.vtensor<[],f32>,
    %b_scale: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[A_SCALE:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg0, %[[A_SCALE]], %[[A_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3],f32>
  // CHECK: %[[B_SCALE:.+]] = torch.aten.item %arg3 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[B_ZP:.+]] = torch.constant.int 0{{$}}
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_tensor %arg1, %[[B_SCALE]], %[[B_ZP]],
  // CHECK-SAME: -> !torch.vtensor<[3],f32>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-SAME: -> !torch.vtensor<[],f32>
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale)
      : (!torch.vtensor<[3],si8>, !torch.vtensor<[3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// Both unknown A qparams need separate checks; B has no zero point.
// CHECK-LABEL: @matmul_integer_to_float_dynamic_qparam_lengths(
func.func @matmul_integer_to_float_dynamic_qparam_lengths(
    %a: !torch.vtensor<[4,?],si8>,
    %b: !torch.vtensor<[?,?],ui8>,
    %a_scale: !torch.vtensor<[?],f32>,
    %b_scale: !torch.vtensor<[?],f32>,
    %a_zp: !torch.vtensor<[?],si8>
    ) -> !torch.vtensor<[4,?],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[A_COLUMNS:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK-NEXT: %[[A_SCALE_ZERO:.+]] = torch.constant.int 0{{$}}
  // CHECK-NEXT: %[[A_SCALE_LENGTH:.+]] = torch.aten.size.int %arg2, %[[A_SCALE_ZERO]]
  // CHECK-NEXT: %[[A_SCALE_MATCHES:.+]] = torch.aten.eq.int %[[A_COLUMNS]], %[[A_SCALE_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[A_SCALE_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK-NEXT: %[[A_ZP_ZERO:.+]] = torch.constant.int 0{{$}}
  // CHECK-NEXT: %[[A_ZP_LENGTH:.+]] = torch.aten.size.int %arg4, %[[A_ZP_ZERO]]
  // CHECK-NEXT: %[[A_ZP_MATCHES:.+]] = torch.aten.eq.int %[[A_COLUMNS]], %[[A_ZP_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[A_ZP_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %arg4, %[[A_AXIS]],
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[B_COLUMNS:.+]] = torch.aten.size.int %arg1, %[[B_AXIS]]
  // CHECK-NEXT: %[[B_SCALE_ZERO:.+]] = torch.constant.int 0{{$}}
  // CHECK-NEXT: %[[B_SCALE_LENGTH:.+]] = torch.aten.size.int %arg3, %[[B_SCALE_ZERO]]
  // CHECK-NEXT: %[[B_SCALE_MATCHES:.+]] = torch.aten.eq.int %[[B_COLUMNS]], %[[B_SCALE_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[B_SCALE_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK-NEXT: %[[B_ZP:.+]] = torch.constant.none
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %[[B_ZP]], %[[B_AXIS]],
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp)
      : (!torch.vtensor<[4,?],si8>, !torch.vtensor<[?,?],ui8>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],si8>) -> !torch.vtensor<[4,?],f32>
  return %0 : !torch.vtensor<[4,?],f32>
}

// -----

// A known length must not suppress the check for the other unknown length.
// CHECK-LABEL: @matmul_integer_to_float_mixed_known_dynamic_qparam_lengths(
func.func @matmul_integer_to_float_mixed_known_dynamic_qparam_lengths(
    %a: !torch.vtensor<[4,?],si8>,
    %b: !torch.vtensor<[?,?],ui8>,
    %a_scale: !torch.vtensor<[3],f32>,
    %b_scale: !torch.vtensor<[?],f32>,
    %a_zp: !torch.vtensor<[?],si8>,
    %b_zp: !torch.vtensor<[5],ui8>
    ) -> !torch.vtensor<[4,?],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[A_COLUMNS:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK-NEXT: %[[A_SCALE_LENGTH:.+]] = torch.constant.int 3{{$}}
  // CHECK-NEXT: %[[A_SCALE_MATCHES:.+]] = torch.aten.eq.int %[[A_COLUMNS]], %[[A_SCALE_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[A_SCALE_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK-NEXT: %[[A_ZP_ZERO:.+]] = torch.constant.int 0{{$}}
  // CHECK-NEXT: %[[A_ZP_LENGTH:.+]] = torch.aten.size.int %arg4, %[[A_ZP_ZERO]]
  // CHECK-NEXT: %[[A_ZP_MATCHES:.+]] = torch.aten.eq.int %[[A_COLUMNS]], %[[A_ZP_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[A_ZP_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %arg4, %[[A_AXIS]],
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[B_COLUMNS:.+]] = torch.aten.size.int %arg1, %[[B_AXIS]]
  // CHECK-NEXT: %[[B_SCALE_ZERO:.+]] = torch.constant.int 0{{$}}
  // CHECK-NEXT: %[[B_SCALE_LENGTH:.+]] = torch.aten.size.int %arg3, %[[B_SCALE_ZERO]]
  // CHECK-NEXT: %[[B_SCALE_MATCHES:.+]] = torch.aten.eq.int %[[B_COLUMNS]], %[[B_SCALE_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[B_SCALE_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK-NEXT: %[[B_ZP_LENGTH:.+]] = torch.constant.int 5{{$}}
  // CHECK-NEXT: %[[B_ZP_MATCHES:.+]] = torch.aten.eq.int %[[B_COLUMNS]], %[[B_ZP_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[B_ZP_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %arg3, %arg5, %[[B_AXIS]],
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,?],si8>, !torch.vtensor<[?,?],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],si8>, !torch.vtensor<[5],ui8>) -> !torch.vtensor<[4,?],f32>
  return %0 : !torch.vtensor<[4,?],f32>
}

// -----

// Only statically known scalar qparams are expanded to the runtime columns.
// CHECK-LABEL: @matmul_integer_to_float_scalar_dynamic_qparam_lengths(
func.func @matmul_integer_to_float_scalar_dynamic_qparam_lengths(
    %a: !torch.vtensor<[4,?],si8>,
    %b: !torch.vtensor<[?,?],ui8>,
    %a_scale: !torch.vtensor<[?],f32>,
    %b_scale: !torch.vtensor<[],f32>,
    %a_zp: !torch.vtensor<[],si8>,
    %b_zp: !torch.vtensor<[?],ui8>
    ) -> !torch.vtensor<[4,?],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[A_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[A_COLUMNS:.+]] = torch.aten.size.int %arg0, %[[A_AXIS]]
  // CHECK-NEXT: %[[A_SCALE_ZERO:.+]] = torch.constant.int 0{{$}}
  // CHECK-NEXT: %[[A_SCALE_LENGTH:.+]] = torch.aten.size.int %arg2, %[[A_SCALE_ZERO]]
  // CHECK-NEXT: %[[A_SCALE_MATCHES:.+]] = torch.aten.eq.int %[[A_COLUMNS]], %[[A_SCALE_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[A_SCALE_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK: %[[A_ZP:.+]] = torch.aten.expand %arg4,
  // CHECK: %[[A_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg2, %[[A_ZP]], %[[A_AXIS]],
  // CHECK: torch.constant.int 6{{$}}
  // CHECK: %[[B_AXIS:.+]] = torch.constant.int 1{{$}}
  // CHECK-NEXT: %[[B_COLUMNS:.+]] = torch.aten.size.int %arg1, %[[B_AXIS]]
  // CHECK-NEXT: %[[B_ZP_ZERO:.+]] = torch.constant.int 0{{$}}
  // CHECK-NEXT: %[[B_ZP_LENGTH:.+]] = torch.aten.size.int %arg5, %[[B_ZP_ZERO]]
  // CHECK-NEXT: %[[B_ZP_MATCHES:.+]] = torch.aten.eq.int %[[B_COLUMNS]], %[[B_ZP_LENGTH]]
  // CHECK-NEXT: torch.runtime.assert %[[B_ZP_MATCHES]], "MatMulIntegerToFloat: per-column qparam length must match input's last dimension"
  // CHECK: %[[B_SCALE:.+]] = torch.aten.expand %arg3,
  // CHECK: %[[B_DQ:.+]] = torch.quantized_decomposed.dequantize_per_channel %arg1, %[[B_SCALE]], %arg5, %[[B_AXIS]],
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[A_DQ]], %[[B_DQ]]
  // CHECK-NEXT: return %[[MM]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %a_scale, %b_scale, %a_zp, %b_zp)
      : (!torch.vtensor<[4,?],si8>, !torch.vtensor<[?,?],ui8>, !torch.vtensor<[?],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>, !torch.vtensor<[?],ui8>) -> !torch.vtensor<[4,?],f32>
  return %0 : !torch.vtensor<[4,?],f32>
}
