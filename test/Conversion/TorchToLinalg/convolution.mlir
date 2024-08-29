// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -mlir-print-local-scope -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @torch.aten.convolution$nobias(
// CHECK:           %[[CONSTANT:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[FILL_RESULT:.*]] = linalg.fill ins(%[[CONSTANT]] : f32) outs(%{{.*}} : tensor<1x54x16x128x128xf32>) -> tensor<1x54x16x128x128xf32>
// CHECK:           %[[CONV3D:.*]] = linalg.conv_3d_ncdhw_fcdhw {{.*}} outs(%[[FILL_RESULT]] : tensor<1x54x16x128x128xf32>) -> tensor<1x54x16x128x128xf32>
func.func @torch.aten.convolution$nobias(%arg0: !torch.vtensor<[1,24,16,128,128],f16>, %arg1: !torch.vtensor<[54,24,1,1,1],f16>) -> !torch.vtensor<[1,54,16,128,128],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.convolution %arg0, %arg1, %none, %2, %0, %1, %false, %3, %int1 : !torch.vtensor<[1,24,16,128,128],f16>, !torch.vtensor<[54,24,1,1,1],f16>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,54,16,128,128],f16>
  return %4 : !torch.vtensor<[1,54,16,128,128],f16>
}

// -----

// CHECK-LABEL: func.func @q_conv_test
// CHECK: %[[c3:.*]] = arith.constant 3 : i32
// CHECK: %[[c7:.*]] = arith.constant 7 : i32
// CHECK: %[[input:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,?,?],si8> -> tensor<?x?x?x?xi8>
// CHECK: %[[weight:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[?,?,?,?],si8> -> tensor<?x?x?x?xi8>
// CHECK: %[[TransInput:.*]] = linalg.transpose ins(%[[input]] : tensor<?x?x?x?xi8>)
// CHECK-SAME: permutation = [0, 2, 3, 1]
// CHECK: %[[TransWeight:.*]] = linalg.transpose ins(%[[weight]] : tensor<?x?x?x?xi8>)
// CHECK-SAME: permutation = [2, 3, 1, 0]
// CHECK: %[[conv:.*]] = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME: ins(%[[TransInput]], %[[TransWeight]], %[[c7]], %[[c3]] : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, i32, i32)
// CHECK-SAME: outs(%[[convout:.*]] : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
func.func @q_conv_test(%arg0: !torch.vtensor<[?,?,?,?],si8>, %arg1: !torch.vtensor<[?,?,?,?],si8>, %arg2: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %float1.000000e-04 = torch.constant.float 1.000000e-04
  %int3 = torch.constant.int 3
  %int7 = torch.constant.int 7
  %float1.000000e-02 = torch.constant.float 1.000000e-02
  %int14 = torch.constant.int 14
  %0 = torch.aten.quantize_per_tensor %arg2, %float1.000000e-04, %int0, %int14 : !torch.vtensor<[?],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[?],!torch.qint32>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[?],!torch.qint32> -> !torch.vtensor<[?],f32>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten._make_per_tensor_quantized_tensor %arg0, %float1.000000e-02, %int7 : !torch.vtensor<[?,?,?,?],si8>, !torch.float, !torch.int -> !torch.vtensor<[?,?,?,?],!torch.qint8>
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg1, %float1.000000e-02, %int3 : !torch.vtensor<[?,?,?,?],si8>, !torch.float, !torch.int -> !torch.vtensor<[?,?,?,?],!torch.qint8>
  %7 = torch.aten.quantize_per_tensor %1, %float1.000000e-04, %int0, %int14 : !torch.vtensor<[?],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[?],!torch.qint32>
  %8 = torch.aten.int_repr %7 : !torch.vtensor<[?],!torch.qint32> -> !torch.vtensor<[?],si32>
  %9 = torch.aten.convolution %5, %6, %8, %2, %3, %2, %false, %4, %int1 : !torch.vtensor<[?,?,?,?],!torch.qint8>, !torch.vtensor<[?,?,?,?],!torch.qint8>, !torch.vtensor<[?],si32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?,?,?],si32>
  %10 = torch.aten._make_per_tensor_quantized_tensor %9, %float1.000000e-04, %int0 : !torch.vtensor<[?,?,?,?],si32>, !torch.float, !torch.int -> !torch.vtensor<[?,?,?,?],!torch.qint32>
  %11 = torch.aten.dequantize.tensor %10 : !torch.vtensor<[?,?,?,?],!torch.qint32> -> !torch.vtensor<[?,?,?,?],f32>
  return %11 : !torch.vtensor<[?,?,?,?],f32>
}
