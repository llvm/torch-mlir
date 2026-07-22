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
  %zero = torch.constant.int 0
  // CHECK: %[[QUANT:.+]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK-SAME: (tensor<4x3x7x7xf32>) -> tensor<4x3x7x7x!quant.uniform<i8:f32:0, {0.4{{.*}}:4,0.1{{.*}}:1,0.2{{.*}}:2,0.3{{.*}}:3}>>
  %2 = torch.aten.quantize_per_channel %arg0, %0, %1, %zero, %int12 : !torch.vtensor<[4,3,7,7],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],si8>, !torch.int, !torch.int -> !torch.vtensor<[4,3,7,7],!torch.qint8>
  %3 = torch.aten.int_repr %2 : !torch.vtensor<[4,3,7,7],!torch.qint8> -> !torch.vtensor<[4,3,7,7],si8>
  // CHECK: %[[DEQ:.+]] = stablehlo.uniform_dequantize %[[QUANT]]
  // CHECK-SAME: (tensor<4x3x7x7x!quant.uniform<i8:f32:0, {0.4{{.*}}:4,0.1{{.*}}:1,0.2{{.*}}:2,0.3{{.*}}:3}>>) -> tensor<4x3x7x7xf32>
  %4 = torch.aten._make_per_channel_quantized_tensor %3, %0, %1, %zero : !torch.vtensor<[4,3,7,7],si8>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4,3,7,7],!torch.qint8>
  %5 = torch.aten.dequantize.self %4 : !torch.vtensor<[4,3,7,7],!torch.qint8> -> !torch.vtensor<[4,3,7,7],f32>
  return %5 : !torch.vtensor<[4,3,7,7],f32>
}

// -----

// CHECK-LABEL: func.func @quantize_per_tensor_basic
// CHECK-SAME:    %[[ARG0:.+]]: !torch.vtensor<[4,8],f32>
func.func @quantize_per_tensor_basic(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],si8> {
  %float0.03 = torch.constant.float 3.000000e-02
  %int_neg10 = torch.constant.int -10
  %int_neg128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int1 = torch.constant.int 1
  // CHECK: %[[INPUT:.+]] = torch_c.to_builtin_tensor %[[ARG0]]
  // CHECK-SAME: !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
  // CHECK-DAG: %[[INV_SCALE:.+]] = stablehlo.constant dense<33.333{{.+}}> : tensor<f32>
  // CHECK: %[[INV_SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[INV_SCALE]], dims = []
  // CHECK: %[[SCALED:.+]] = stablehlo.multiply %[[INPUT]], %[[INV_SCALE_BCAST]]
  // CHECK: %[[ROUNDED:.+]] = stablehlo.round_nearest_even %[[SCALED]]
  // CHECK-DAG: %[[ZP:.+]] = stablehlo.constant dense<-1.000000e+01> : tensor<f32>
  // CHECK: %[[ZP_BCAST:.+]] = stablehlo.broadcast_in_dim %[[ZP]], dims = []
  // CHECK: %[[SHIFTED:.+]] = stablehlo.add %[[ROUNDED]], %[[ZP_BCAST]]
  // CHECK-DAG: %[[QMIN:.+]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QMAX:.+]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[CLAMPED:.+]] = stablehlo.clamp %[[QMIN]], %[[SHIFTED]], %[[QMAX]]
  // CHECK: %{{.+}} = stablehlo.convert %[[CLAMPED]] : (tensor<4x8xf32>) -> tensor<4x8xi8>
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %float0.03, %int_neg10, %int_neg128, %int127, %int1
    : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
    -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

// Dynamic scale should fail to legalize (no CHECK-LABEL here as conversion fails).
func.func @quantize_per_tensor_dynamic_scale(%arg0: !torch.vtensor<[4,8],f32>, %scale: !torch.float) -> !torch.vtensor<[4,8],si8> {
  %int0 = torch.constant.int 0
  %int_neg128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int1 = torch.constant.int 1
  // expected-error @+1 {{failed to legalize operation 'torch.quantized_decomposed.quantize_per_tensor' that was explicitly marked illegal}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %scale, %int0, %int_neg128, %int127, %int1
    : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
    -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_tensor_basic
// CHECK-SAME:    %[[ARG0:.+]]: !torch.vtensor<[4,8],si8>
func.func @dequantize_per_tensor_basic(%arg0: !torch.vtensor<[4,8],si8>) -> !torch.vtensor<[4,8],f32> {
  %float0.03 = torch.constant.float 3.000000e-02
  %int_neg10 = torch.constant.int -10
  %int_neg128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  // CHECK: %[[INPUT:.+]] = torch_c.to_builtin_tensor %[[ARG0]]
  // CHECK-SAME: !torch.vtensor<[4,8],si8> -> tensor<4x8xi8>
  // CHECK: %[[INT_WIDE:.+]] = stablehlo.convert %[[INPUT]] : (tensor<4x8xi8>) -> tensor<4x8xi32>
  // CHECK-DAG: %[[ZP:.+]] = stablehlo.constant dense<-10> : tensor<i32>
  // CHECK: %[[ZP_BCAST:.+]] = stablehlo.broadcast_in_dim %[[ZP]], dims = []
  // CHECK: %[[SUBTR:.+]] = stablehlo.subtract %[[INT_WIDE]], %[[ZP_BCAST]]
  // CHECK: %[[AS_FLOAT:.+]] = stablehlo.convert %[[SUBTR]] : (tensor<4x8xi32>) -> tensor<4x8xf32>
  // CHECK-DAG: %[[SCALE:.+]] = stablehlo.constant dense<3.000000e-02> : tensor<f32>
  // CHECK: %[[SCALE_BCAST:.+]] = stablehlo.broadcast_in_dim %[[SCALE]], dims = []
  // CHECK: %{{.+}} = stablehlo.multiply %[[AS_FLOAT]], %[[SCALE_BCAST]]
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %float0.03, %int_neg10, %int_neg128, %int127, %int1, %int6
    : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int
    -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

// Dynamic scale should fail to legalize.
func.func @dequantize_per_tensor_dynamic_scale(%arg0: !torch.vtensor<[4,8],si8>, %scale: !torch.float) -> !torch.vtensor<[4,8],f32> {
  %int0 = torch.constant.int 0
  %int_neg128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  // expected-error @+1 {{failed to legalize operation 'torch.quantized_decomposed.dequantize_per_tensor' that was explicitly marked illegal}}
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %scale, %int0, %int_neg128, %int127, %int1, %int6
    : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int
    -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

// Zero scale on quantize should fail to legalize (quantize divides by scale).
func.func @quantize_per_tensor_zero_scale(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],si8> {
  %float0 = torch.constant.float 0.000000e+00
  %int_neg10 = torch.constant.int -10
  %int_neg128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int1 = torch.constant.int 1
  // expected-error @+1 {{failed to legalize operation 'torch.quantized_decomposed.quantize_per_tensor' that was explicitly marked illegal}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %float0, %int_neg10, %int_neg128, %int127, %int1
    : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
    -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

// Dequantize with a dynamic (SSA) quant_min still legalizes -- quant_min /
// quant_max are metadata unused by the dequantize arithmetic.
// CHECK-LABEL: func.func @dequantize_per_tensor_dynamic_quant_min
func.func @dequantize_per_tensor_dynamic_quant_min(%arg0: !torch.vtensor<[4,8],si8>, %qmin: !torch.int) -> !torch.vtensor<[4,8],f32> {
  %float0.03 = torch.constant.float 3.000000e-02
  %int_neg10 = torch.constant.int -10
  %int127 = torch.constant.int 127
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  // CHECK: stablehlo.subtract
  // CHECK: stablehlo.multiply
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %float0.03, %int_neg10, %qmin, %int127, %int1, %int6
    : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int
    -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}
