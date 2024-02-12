// RUN: torch-mlir-opt %s --split-input-file --torch-fuse-quantized-ops | FileCheck %s

// CHECK-LABEL: @mm
func.func @mm(%arg0: !torch.vtensor<[4, 4],si8>, %arg1: !torch.vtensor<[4, 4],si8>) -> !torch.vtensor<[4, 4],f32> {
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %zp = torch.constant.int -128
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[4, 4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4, 4],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[4, 4],!torch.qint8> -> !torch.vtensor<[4, 4],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[4, 4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4, 4],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[4, 4],!torch.qint8> -> !torch.vtensor<[4, 4],f32>
  %16 = torch.aten.mm %7, %13 : !torch.vtensor<[4, 4],f32>, !torch.vtensor<[4, 4],f32> -> !torch.vtensor<[4, 4],f32>

  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[QUARTER:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[HALF:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[QLHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[HALF:.+]], %[[ONE]] : !torch.vtensor<[4,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4,4],!torch.qint8>
  // CHECK-DAG: %[[QRHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[HALF:.+]], %[[ZERO]] : !torch.vtensor<[4,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4,4],!torch.qint8>
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %[[QLHS]], %[[QRHS]] : !torch.vtensor<[4,4],!torch.qint8>, !torch.vtensor<[4,4],!torch.qint8> -> !torch.vtensor<[4,4],!torch.qint32>
  // CHECK-DAG: %[[INT:.+]] = torch.aten.int_repr %[[MM]] : !torch.vtensor<[4,4],!torch.qint32> -> !torch.vtensor<[4,4],si32>
  // CHECK-DAG: %[[QOUT:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[INT]], %[[QUARTER]], %[[ZERO]] : !torch.vtensor<[4,4],si32>, !torch.float, !torch.int -> !torch.vtensor<[4,4],!torch.qint32>
  // CHECK: %[[OUT:.+]] = torch.aten.dequantize.tensor %[[QOUT]] : !torch.vtensor<[4,4],!torch.qint32> -> !torch.vtensor<[4,4],f32>
  return %16 : !torch.vtensor<[4, 4],f32>
}

// -----

// CHECK-LABEL: @convolution_bias
func.func @convolution_bias(%arg0: !torch.vtensor<[1,3,8,8],si8>, %arg1: !torch.vtensor<[3,3,2,2],si8>, %arg2 : !torch.vtensor<[3], f32>) -> !torch.vtensor<[1,3,7,7],f32> {
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %zp = torch.constant.int -128
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[1,3,8,8],!torch.qint8> -> !torch.vtensor<[1,3,8,8],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[3,3,2,2],!torch.qint8> -> !torch.vtensor<[3,3,2,2],f32>
  %14 = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  %15 = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
  %16 = torch.aten.convolution %7, %13, %arg2, %14, %15, %14, %false, %15, %one : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3,3,2,2],f32>, !torch.vtensor<[3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],f32>

  // CHECK: %[[DTYPE:.+]] = torch.constant.int 14
  // CHECK: %[[SCALEO:.+]] = torch.constant.float 2.500000e-01
  // CHECK: %[[HALF:.+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK: %[[QLHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[HALF]], %[[ONE]] : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  // CHECK: %[[QRHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[HALF]], %[[ZERO]] : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  // CHECK: %[[ONES:.+]] = torch.prim.ListConstruct %[[ONE]], %[[ONE]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[ZEROS:.+]] = torch.prim.ListConstruct %[[ZERO]], %[[ZERO]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[QBIAS:.+]] = torch.aten.quantize_per_tensor %arg2, %[[SCALEO]], %[[ZERO]], %[[DTYPE]] : !torch.vtensor<[3],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[3],!torch.qint32>
  // CHECK: %[[INT:.+]] = torch.aten.int_repr %[[QBIAS]] : !torch.vtensor<[3],!torch.qint32> -> !torch.vtensor<[3],si32>
  // CHECK: %[[CONV:.+]] = torch.aten.convolution %[[QLHS]], %[[QRHS]], %[[INT]], %[[ONES]], %[[ZEROS]], %[[ONES]], %[[FALSE]], %[[ZEROS]], %[[ONE]] : !torch.vtensor<[1,3,8,8],!torch.qint8>, !torch.vtensor<[3,3,2,2],!torch.qint8>, !torch.vtensor<[3],si32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],si32>
  // CHECK: %[[QOUT:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[CONV]], %[[SCALEO]], %[[ZERO]] : !torch.vtensor<[1,3,7,7],si32>, !torch.float, !torch.int -> !torch.vtensor<[1,3,7,7],!torch.qint32>
  // CHECK: %[[FOUT:.+]] = torch.aten.dequantize.tensor %[[QOUT]] : !torch.vtensor<[1,3,7,7],!torch.qint32> -> !torch.vtensor<[1,3,7,7],f32>
  return %16 : !torch.vtensor<[1,3,7,7],f32>
}


// -----

// CHECK-LABEL: @convolution_nobias
func.func @convolution_nobias(%arg0: !torch.vtensor<[1,3,8,8],si8>, %arg1: !torch.vtensor<[3,3,2,2],si8>) -> !torch.vtensor<[1,3,7,7],f32> {
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %zp = torch.constant.int -128
  %none = torch.constant.none
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[1,3,8,8],!torch.qint8> -> !torch.vtensor<[1,3,8,8],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[3,3,2,2],!torch.qint8> -> !torch.vtensor<[3,3,2,2],f32>
  %14 = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  %15 = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
  %16 = torch.aten.convolution %7, %13, %none, %14, %15, %14, %false, %15, %one : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3,3,2,2],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],f32>

  // CHECK-DAG: %[[SCALEO:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[HALF:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[QLHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[HALF]], %[[ONE]] : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  // CHECK-DAG: %[[QRHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[HALF]], %[[ZERO]] : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  // CHECK-DAG: %[[ONES:.+]] = torch.prim.ListConstruct %[[ONE]], %[[ONE]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[ZEROS:.+]] = torch.prim.ListConstruct %[[ZERO]], %[[ZERO]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[CONV:.+]] = torch.aten.convolution %[[QLHS]], %[[QRHS]], %[[NONE]], %[[ONES]], %[[ZEROS]], %[[ONES]], %[[FALSE]], %[[ZEROS]], %[[ONE]] : !torch.vtensor<[1,3,8,8],!torch.qint8>, !torch.vtensor<[3,3,2,2],!torch.qint8>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],si32>
  // CHECK-DAG: %[[QOUT:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[CONV]], %[[SCALEO]], %[[ZERO]] : !torch.vtensor<[1,3,7,7],si32>, !torch.float, !torch.int -> !torch.vtensor<[1,3,7,7],!torch.qint32>
  // CHECK: %[[FOUT:.+]] = torch.aten.dequantize.tensor %[[QOUT]] : !torch.vtensor<[1,3,7,7],!torch.qint32> -> !torch.vtensor<[1,3,7,7],f32>
  return %16 : !torch.vtensor<[1,3,7,7],f32>
}
