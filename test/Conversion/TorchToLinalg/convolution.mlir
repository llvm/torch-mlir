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
