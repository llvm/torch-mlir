// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// CHECK-LABEL: func.func @test_ifop_basic
// CHECK: %[[IF:.*]] = torch.prim.If %{{.*}} -> (!torch.vtensor<[1],f32>)
// CHECK-DAG: %[[SUB:.*]] = torch.aten.sub.Tensor %arg1, %arg2, %{{.*}} : !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.int -> !torch.vtensor<[1],f32>
// CHECK-DAG: torch.prim.If.yield %[[SUB]] : !torch.vtensor<[1],f32>
// CHECK-DAG: } else {
// CHECK-DAG: %[[ADD:.*]] = torch.aten.add.Tensor %arg1, %arg2, %{{.*}} : !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.int -> !torch.vtensor<[1],f32>
// CHECK-DAG: torch.prim.If.yield %[[ADD]] : !torch.vtensor<[1],f32>
func.func @test_ifop_basic(%arg0: !torch.vtensor<[1],i1>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "conditional_example", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0 = torch.operator "onnx.If"(%arg0) : (!torch.vtensor<[1],i1>) -> !torch.vtensor<[1],f32> {
    %1 = torch.operator "onnx.Add"(%arg1, %arg2) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32>
    torch.operator_terminator %1 : !torch.vtensor<[1],f32>
  }, {
    %1 = torch.operator "onnx.Sub"(%arg1, %arg2) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32>
    torch.operator_terminator %1 : !torch.vtensor<[1],f32>
  }
  return %0 : !torch.vtensor<[1],f32>
}
