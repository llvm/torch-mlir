// RUN: torch-mlir-opt %s -torch-convert-custom-quant-op -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward
func.func @forward(%arg0: !torch.vtensor<[1,1,2],f16>) -> !torch.vtensor<[1,1,2],f16> {
  %q_rhs = torch.vtensor.literal(dense<[[0, 1], [2, 3]]> : tensor<2x2xui8>) : !torch.vtensor<[2,2],ui8>
  %scales = torch.vtensor.literal(dense<[[[1.0]], [[1.0]]]> : tensor<2x1x1xf16>) : !torch.vtensor<[2,1,1],f16>
  %zps = torch.vtensor.literal(dense<[[[0.0]], [[0.0]]]> : tensor<2x1x1xf16>) : !torch.vtensor<[2,1,1],f16>
  %bit_width = torch.constant.int 8
  %group_size = torch.constant.int 2
  %output = torch.operator "quant.matmul_rhs_group_quant"(%arg0, %q_rhs, %scales, %zps, %bit_width, %group_size) : (!torch.vtensor<[1,1,2],f16>, !torch.vtensor<[2,2],ui8>, !torch.vtensor<[2,1,1],f16>, !torch.vtensor<[2,1,1],f16>, !torch.int, !torch.int) -> !torch.vtensor<[1,1,2],f16>
  return %output : !torch.vtensor<[1,1,2],f16>
}
