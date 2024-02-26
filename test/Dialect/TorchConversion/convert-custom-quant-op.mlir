// RUN: torch-mlir-opt %s '-pass-pipeline=builtin.module(func.func(torch-convert-custom-quant-op))' -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-LABEL: func @forward
func.func @forward(%arg0: !torch.vtensor<[1,1,2],f16>) -> !torch.vtensor<[1,1,2],f16> {
  %q_rhs = torch.vtensor.literal(dense<[[0, 1], [2, 3]]> : tensor<2x2xui8>) : !torch.vtensor<[2,2],ui8>
  %scales = torch.vtensor.literal(dense<1.0> : tensor<2x1x1xf16>) : !torch.vtensor<[2,1,1],f16>
  %zps = torch.vtensor.literal(dense<0.0> : tensor<2x1x1xf16>) : !torch.vtensor<[2,1,1],f16>
  %bit_width = torch.constant.int 8
  %group_size = torch.constant.int 2
  %output = torch.operator "quant.matmul_rhs_group_quant"(%arg0, %q_rhs, %scales, %zps, %bit_width, %group_size) : (!torch.vtensor<[1,1,2],f16>, !torch.vtensor<[2,2],ui8>, !torch.vtensor<[2,1,1],f16>, !torch.vtensor<[2,1,1],f16>, !torch.int, !torch.int) -> !torch.vtensor<[1,1,2],f16>
  // CHECK: %[[LHS:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,1,2],f16> -> tensor<1x1x2xf16>
  // CHECK: %[[TENSOR1:.*]] = torch.vtensor.literal(dense<{{\[\[}}0, 1], [2, 3]]> : tensor<2x2xui8>) : !torch.vtensor<[2,2],ui8>
  // CHECK: %[[QUANT_RHS:.*]] = torch_c.to_builtin_tensor %[[TENSOR1]] : !torch.vtensor<[2,2],ui8> -> tensor<2x2xi8>
  // CHECK: %[[TENSOR2:.*]] = torch.vtensor.literal(dense<1.000000e+00> : tensor<2x1x1xf16>) : !torch.vtensor<[2,1,1],f16>
  // CHECK: %[[SCALES:.*]] = torch_c.to_builtin_tensor %[[TENSOR2]] : !torch.vtensor<[2,1,1],f16> -> tensor<2x1x1xf16>
  // CHECK: %[[TENSOR3:.*]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<2x1x1xf16>) : !torch.vtensor<[2,1,1],f16>
  // CHECK: %[[ZPS:.*]] = torch_c.to_builtin_tensor %[[TENSOR3]] : !torch.vtensor<[2,1,1],f16> -> tensor<2x1x1xf16>
  // CHECK: %[[EXPANDED_LHS:.*]] = tensor.expand_shape %[[LHS]] {{\[\[}}0], [1], [2, 3]] : tensor<1x1x2xf16> into tensor<1x1x1x2xf16>
  // CHECK: %[[EXPANDED_RHS:.*]] = tensor.expand_shape %[[QUANT_RHS]] {{\[\[}}0], [1, 2]] : tensor<2x2xi8> into tensor<2x1x2xi8>
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<2x1x2xf16>
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1x1x2xf16>
  // CHECK: %[[OUT:.*]] = linalg.fill ins(%[[CST]] : f16) outs(%[[EMPTY2]] : tensor<1x1x2xf16>) -> tensor<1x1x2xf16>
  // CHECK: %[[DEQUANT_RHS:.*]] = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[EXPANDED_RHS]], %[[SCALES]], %[[ZPS]] : tensor<2x1x2xi8>, tensor<2x1x1xf16>, tensor<2x1x1xf16>) outs(%[[EMPTY1]] : tensor<2x1x2xf16>) {
  // CHECK-NEXT: ^bb0(%[[WEIGHTS:.*]]: i8, %[[SCALES:.*]]: f16, %[[ZPS:.*]]: f16, %{{.*}}: f16):
  // CHECK-NEXT:   %[[EXTUI:.*]] = arith.extui %[[WEIGHTS]] : i8 to i32
  // CHECK-NEXT:   %[[UITOFP:.*]] = arith.uitofp %[[EXTUI]] : i32 to f16
  // CHECK-NEXT:   %[[SUBF:.*]] = arith.subf %[[UITOFP]], %[[ZPS]] : f16
  // CHECK-NEXT:   %[[MULF:.*]] = arith.mulf %[[SUBF]], %[[SCALES]] : f16
  // CHECK-NEXT:   linalg.yield %[[MULF]] : f16
  // CHECK-NEXT: } -> tensor<2x1x2xf16>
  // CHECK: %[[MATMUL:.*]] = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%[[EXPANDED_LHS]], %[[DEQUANT_RHS]] : tensor<1x1x1x2xf16>, tensor<2x1x2xf16>) outs(%[[OUT]] : tensor<1x1x2xf16>) {
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: f16, %[[RHS:.*]]: f16, %[[OUT:.*]]: f16):
  // CHECK-NEXT:   %[[MULF:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f16
  // CHECK-NEXT:   %[[ADDF:.*]] = arith.addf %[[MULF]], %[[OUT]] : f16
  // CHECK-NEXT:   linalg.yield %[[ADDF]] : f16
  // CHECK-NEXT: } -> tensor<1x1x2xf16>
  // CHECK: %[[CASTED:.*]] = tensor.cast %[[MATMUL]] : tensor<1x1x2xf16> to tensor<1x1x2xf16>
  return %output : !torch.vtensor<[1,1,2],f16>
}
