// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s



// CHECK-LABEL:   func.func @test_lstm_basic(
// CHECK-SAME:                               %[[X:.*]]: !torch.vtensor<[15,2,4],f32>,
// CHECK-SAME:                               %[[W:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                               %[[R:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                               %[[B:.*]]: !torch.vtensor<[1,24],f32>)
// CHECK:           %[[LOOP_RESULT:.*]]:3 = torch.prim.Loop %[[MAX_TRIPS:.*]], %[[ENTER_LOOP:.*]], init(%[[Y:.*]], %[[INITIAL_H:.*]], %[[INITIAL_C:.*]]) {
// CHECK:           ^bb0(%[[LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV:.*]]: !torch.vtensor<[15,2,3],f32>, %[[H_PREV:.*]]: !torch.vtensor<[2,3],f32>, %[[C_PREV:.*]]: !torch.vtensor<[2,3],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK-DAG:           }
// CHECK:         }
module {
  func.func @test_lstm_basic(%arg0: !torch.vtensor<[15,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[15,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
    return %0#0, %0#1, %0#2 : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
  }
}
