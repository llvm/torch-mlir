// RUN: torch-mlir-opt --split-input-file --torch-match-quantized-custom-ops %s | FileCheck %s

// CHECK-LABEL: func.func @quantize_per_tensor
func.func @quantize_per_tensor(%arg0: !torch.vtensor<[1,3,8,8],f32>) -> !torch.vtensor<[1,3,8,8],si8> {
  %float = torch.constant.float 0.5
  %zp = torch.constant.int 17
  %min = torch.constant.int -128
  %max = torch.constant.int 127
  %dtype = torch.constant.int 1

  // CHECK-DAG: %[[SCALE:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[ZP:.+]] = torch.constant.int 17
  // CHECK-DAG: %[[MIN:.+]] = torch.constant.int -128
  // CHECK-DAG: %[[MAX:.+]] = torch.constant.int 127
  // CHECK-DAG: %[[DTYPE:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[QUANT:.+]] = torch.aten.quantize_per_tensor %arg0, %[[SCALE]], %[[ZP]], %[[DTYPE]] : !torch.vtensor<[1,3,8,8],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  // CHECK-DAG: %[[REPR:.+]] = torch.aten.int_repr %[[QUANT]] : !torch.vtensor<[1,3,8,8],!torch.qint8> -> !torch.vtensor<[1,3,8,8],si8>
  // CHECK: torch.aten.clamp %[[REPR]], %[[MIN]], %[[MAX]]
  %0 = torch.operator "torch.quantized_decomposed.quantize_per_tensor"(%arg0, %float, %zp, %min, %max, %dtype) : (!torch.vtensor<[1,3,8,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.vtensor<[1,3,8,8],si8>
  return %0 : !torch.vtensor<[1,3,8,8],si8>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_tensor
func.func @dequantize_per_tensor(%arg0: !torch.vtensor<[1,3,8,8],si8>) -> !torch.vtensor<[1,3,8,8],f32> {
  %float = torch.constant.float 0.5
  %zp = torch.constant.int 17
  %min = torch.constant.int -128
  %max = torch.constant.int 127
  %dtype = torch.constant.int 1

  // CHECK-DAG: %[[SCALE:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[ZP:.+]] = torch.constant.int 17
  // CHECK-DAG: %[[MIN:.+]] = torch.constant.int -128
  // CHECK-DAG: %[[MAX:.+]] = torch.constant.int 127
  // CHECK-DAG: %[[CLAMP:.+]] = torch.aten.clamp %arg0, %[[MIN]], %[[MAX]] : !torch.vtensor<[1,3,8,8],si8>, !torch.int, !torch.int -> !torch.vtensor<[1,3,8,8],si8>
  // CHECK-DAG: %[[QINT:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[CLAMP]], %[[SCALE]], %[[ZP]] : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  // CHECK: %[[DEQUANT:.+]] = torch.aten.dequantize.tensor %[[QINT]] : !torch.vtensor<[1,3,8,8],!torch.qint8> -> !torch.vtensor<[1,3,8,8],f32>
  %13 = torch.operator "torch.quantized_decomposed.dequantize_per_tensor"(%arg0, %float, %zp, %min, %max, %dtype) : (!torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.vtensor<[1,3,8,8],f32>
  return %13 : !torch.vtensor<[1,3,8,8],f32>
}
