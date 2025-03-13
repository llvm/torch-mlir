// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s


// CHECK-LABEL: test_quantization_per_tensor
func.func @test_quantization_per_tensor(%arg0: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[2,4,4],f32> {
  // CHECK: %[[ARG0:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[2,4,4],f32> -> tensor<2x4x4xf32>
  %int12 = torch.constant.int 12
  %float1.000000e-01 = torch.constant.float 0.1
  %zero = torch.constant.int 0
  // CHECK: %[[QUANT:.+]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK-SAME: (tensor<2x4x4xf32>) -> tensor<2x4x4x!quant.uniform<i8:f32, 1.000000e-01>>
  %0 = torch.aten.quantize_per_tensor %arg0, %float1.000000e-01, %zero, %int12 : !torch.vtensor<[2,4,4],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[2,4,4],!torch.qint8>
  %1 = torch.aten.int_repr %0 : !torch.vtensor<[2,4,4],!torch.qint8> -> !torch.vtensor<[2,4,4],si8>
  // CHECK: %[[DEQ:.+]] = stablehlo.uniform_dequantize %[[QUANT]]
  %2 = torch.aten._make_per_tensor_quantized_tensor %1, %float1.000000e-01, %zero : !torch.vtensor<[2,4,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[2,4,4],!torch.qint8>
  %3 = torch.aten.dequantize.self %2 : !torch.vtensor<[2,4,4],!torch.qint8> -> !torch.vtensor<[2,4,4],f32>
  return %3 : !torch.vtensor<[2,4,4],f32>
}

// -----

// CHECK-LABEL: test_quantization_per_channel
func.func @test_quantization_per_channel(%arg0: !torch.vtensor<[4,3,7,7],f32>) -> !torch.vtensor<[4,3,7,7],f32> {
  // CHECK: %[[ARG0:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,3,7,7],f32> -> tensor<4x3x7x7xf32>
  %0 = torch.vtensor.literal(dense<[4.000000e-01, 1.000000e-01, 2.000000e-01, 3.000000e-01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %1 = torch.vtensor.literal(dense<[4, 1, 2, 3]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %int12 = torch.constant.int 12
  %int1 = torch.constant.int 1
  // CHECK: %[[QUANT:.+]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK-SAME: (tensor<4x3x7x7xf32>) -> tensor<4x3x7x7x!quant.uniform<i8:f32:1, {0.4{{.*}}:4,0.1{{.*}}:1,0.2{{.*}}:2,0.3{{.*}}:3}>>
  %2 = torch.aten.quantize_per_channel %arg0, %0, %1, %int1, %int12 : !torch.vtensor<[4,3,7,7],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],si8>, !torch.int, !torch.int -> !torch.vtensor<[4,3,7,7],!torch.qint8>
  %3 = torch.aten.int_repr %2 : !torch.vtensor<[4,3,7,7],!torch.qint8> -> !torch.vtensor<[4,3,7,7],si8>
  // CHECK: %[[DEQ:.+]] = stablehlo.uniform_dequantize %[[QUANT]]
  // CHECK-SAME: (tensor<4x3x7x7x!quant.uniform<i8:f32:1, {0.4{{.*}}:4,0.1{{.*}}:1,0.2{{.*}}:2,0.3{{.*}}:3}>>) -> tensor<4x3x7x7xf32>
  %4 = torch.aten._make_per_channel_quantized_tensor %3, %0, %1, %int1 : !torch.vtensor<[4,3,7,7],si8>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4,3,7,7],!torch.qint8>
  %5 = torch.aten.dequantize.self %4 : !torch.vtensor<[4,3,7,7],!torch.qint8> -> !torch.vtensor<[4,3,7,7],f32>
  return %5 : !torch.vtensor<[4,3,7,7],f32>
}
