// RUN: torch-mlir-opt %s '-pass-pipeline=builtin.module(func.func(torch-unpack-quant-tensor))' -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward
func.func @forward(%arg0: !torch.vtensor<[1,1,8],f16>) -> !torch.vtensor<[1,1,8],f16> {
  %q_rhs = torch.vtensor.literal(dense<[[57, 128, 249, 244], [7, 243, 27, 15], [1, 2, 159, 71], [159, 253, 160, 231], [248, 224, 191, 228], [96, 15, 158, 220], [240, 250, 47, 208], [127, 192, 239, 176]]> : tensor<8x4xui8>) : !torch.vtensor<[8,4],ui8>
  // CHECK: %[[C0:.*]] = torch.vtensor.literal(dense<{{\[\[}}9, 3, 0, 8, 9, 15, 4, 15], [7, 0, 3, 15, 11, 1, 15, 0], [1, 0, 2, 0, 15, 9, 7, 4], [15, 9, 13, 15, 0, 10, 7, 14], [8, 15, 0, 14, 15, 11, 4, 14], [0, 6, 15, 0, 14, 9, 12, 13], [0, 15, 10, 15, 15, 2, 0, 13], [15, 7, 0, 12, 15, 14, 0, 11]]> : tensor<8x8xui4>) : !torch.vtensor<[8,8],ui4>
  %scales = torch.vtensor.literal(dense<1.0> : tensor<8x4x1xf16>) : !torch.vtensor<[8,4,1],f16>
  %zps = torch.vtensor.literal(dense<0.0> : tensor<8x4x1xf16>) : !torch.vtensor<[8,4,1],f16>
  %bit_width = torch.constant.int 4
  %group_size = torch.constant.int 2
  %output = torch.operator "quant.matmul_rhs_group_quant"(%arg0, %q_rhs, %scales, %zps, %bit_width, %group_size) : (!torch.vtensor<[1,1,8],f16>, !torch.vtensor<[8,4],ui8>, !torch.vtensor<[8,4,1],f16>, !torch.vtensor<[8,4,1],f16>, !torch.int, !torch.int) -> !torch.vtensor<[1,1,8],f16>
  return %output : !torch.vtensor<[1,1,8],f16>
}
