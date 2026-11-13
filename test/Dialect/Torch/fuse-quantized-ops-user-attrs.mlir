// RUN: torch-mlir-opt %s -torch-fuse-quantized-ops | FileCheck %s

// Verify that `mlir.user` annotations on a matmul whose operands are fused
// with dequantization are forwarded onto the value that replaces it (the
// final dequantize of the quantized accumulator).

// CHECK-LABEL: @mm_user_attrs
// CHECK:       torch.aten.dequantize.tensor {{.*}} {mlir.user = [{tag = "qmm"}]}
func.func @mm_user_attrs(%arg0: !torch.vtensor<[4, 4],si8>, %arg1: !torch.vtensor<[4, 4],si8>) -> !torch.vtensor<[4, 4],f32> {
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %zp = torch.constant.int -128
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[4, 4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4, 4],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[4, 4],!torch.qint8> -> !torch.vtensor<[4, 4],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[4, 4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4, 4],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[4, 4],!torch.qint8> -> !torch.vtensor<[4, 4],f32>
  %16 = torch.aten.mm %7, %13 {mlir.user = [{tag = "qmm"}]} : !torch.vtensor<[4, 4],f32>, !torch.vtensor<[4, 4],f32> -> !torch.vtensor<[4, 4],f32>
  return %16 : !torch.vtensor<[4, 4],f32>
}
