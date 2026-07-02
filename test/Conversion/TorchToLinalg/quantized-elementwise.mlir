// RUN: torch-mlir-opt -convert-torch-to-linalg -split-input-file %s | FileCheck %s

// Standalone dequantize — no matmul; lowered to linalg.generic by ConvertElementwiseOp.
// CHECK-LABEL: func.func @standalone_dequantize_si8
// CHECK:       linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<4x8xi8>) outs({{.*}} : tensor<4x8xf32>)
// CHECK:         arith.extsi {{.*}} : i8 to i32
// CHECK:         arith.subi {{.*}} : i32
// CHECK:         arith.sitofp {{.*}} : i32 to f32
// CHECK:         arith.mulf {{.*}} : f32
func.func @standalone_dequantize_si8(
    %input: !torch.vtensor<[4,8],si8>) -> !torch.vtensor<[4,8],f32> {
  %scale = torch.constant.float 3.000000e-01
  %zp    = torch.constant.int 0
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none  = torch.constant.none
  %od    = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// Standalone quantize — no matmul; lowered to linalg.generic by ConvertElementwiseOp.
// CHECK-LABEL: func.func @standalone_quantize_si8
// CHECK:       linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<4x8xf32>) outs({{.*}} : tensor<4x8xi8>)
// CHECK:         arith.divf {{.*}} : f32
// CHECK:         math.roundeven {{.*}} : f32
// CHECK:         arith.addf {{.*}} : f32
// CHECK:         arith.fptosi {{.*}} : f32 to i8
func.func @standalone_quantize_si8(
    %input: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],si8> {
  %scale = torch.constant.float 3.000000e-01
  %zp    = torch.constant.int 0
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out = torch.quantized_decomposed.quantize_per_tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,8],si8>
  return %out : !torch.vtensor<[4,8],si8>
}
