// RUN: torch-mlir-opt %s \
// RUN:   -convert-torch-onnx-to-torch \
// RUN:   -torch-backend-to-linalg-on-tensors-backend-pipeline \
// RUN:   | FileCheck %s

// End-to-end: a runtime single-element axes tensor lowers cleanly to linalg
// with no orphan constant.int ('failed to legalize') remaining.

// CHECK-LABEL: func.func @test_reduce_sum_keepdims_random
// CHECK-NOT: failed to legalize
// CHECK: linalg.generic
func.func @test_reduce_sum_keepdims_random(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}
