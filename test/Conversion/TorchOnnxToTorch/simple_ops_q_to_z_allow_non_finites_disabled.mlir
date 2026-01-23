// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch=allow-non-finites=false | FileCheck %s

// COM: the tests in this file locks down the behavior for allow-non-finites=false that replaces non-finites with the closest finite value for a given dtype.

// -----

// CHECK-LABEL: func.func @test_reduce_max_empty_set_fp

func.func @test_reduce_max_empty_set_fp(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-NOT: torch.constant.float 0xFFF0000000000000
  // CHECK-DAG: %[[INF:.+]] = torch.constant.float -3.4028234663852886E+38
  // CHECK-DAG: %[[FULL:.+]] = torch.aten.full
  // CHECK-SAME: %[[INF]]
  %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32>
  return %0 : !torch.vtensor<[2,1,4],f32>
}
