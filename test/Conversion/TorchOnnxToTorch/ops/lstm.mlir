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

func.func @test_lstm_basic(%arg0: !torch.vtensor<[15,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[15,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL:   func.func @test_lstm_bidirectional_with_initial_bias(
// CHECK-SAME:                                               %[[X:.*]]: !torch.vtensor<[32,32,192],f32>,
// CHECK-SAME:                                               %[[W:.*]]: !torch.vtensor<[2,192,192],f32>,
// CHECK-SAME:                                               %[[R:.*]]: !torch.vtensor<[2,192,48],f32>,
// CHECK-SAME:                                               %[[B:.*]]: !torch.vtensor<[2,384],f32>)
// CHECK:           %[[FORWARD_LOOP_RES:.*]]:3 = torch.prim.Loop %[[MAX_TRIP_FWD:.*]], %[[LOOP_COND_FWD:.*]], init(%[[Y_FWD:.*]], %[[INITIAL_H_FWD:.*]], %[[INITIAL_C_FWD:.*]]) {
// CHECK:           ^bb0(%[[FORWARD_LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV_FWD:.*]]: !torch.vtensor<[32,32,48],f32>, %[[H_PREV_FWD:.*]]: !torch.vtensor<[32,48],f32>, %[[C_PREV_FWD:.*]]: !torch.vtensor<[32,48],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK:           }
// CHECK:           torch.aten.flip
// CHECK:           %[[REVERSE_LOOP_RES:.*]]:3 = torch.prim.Loop %[[MAX_TRIPS_REV:.*]], %[[LOOP_COND_REV:.*]], init(%[[Y_REV:.*]], %[[INITIAL_H_REV:.*]], %[[INITIAL_C_REV:.*]]) {
// CHECK:           ^bb0(%[[REVERSE_LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV_REV:.*]]: !torch.vtensor<[32,32,48],f32>, %[[H_PREV_REV:.*]]: !torch.vtensor<[32,48],f32>, %[[C_PREV_REV:.*]]: !torch.vtensor<[32,48],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK:           }
// CHECK:           torch.aten.flip
// CHECK:           return %[[Y:.*]], %[[Y_H:.*]], %[[Y_C:.*]] : !torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>
// CHECK:         }

func.func @test_lstm_bidirectional_with_initial_bias(%arg0: !torch.vtensor<[32,32,192],f32>, %arg1: !torch.vtensor<[2,192,192],f32>, %arg2: !torch.vtensor<[2,192,48],f32>, %arg3: !torch.vtensor<[2,384],f32>) -> (!torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.direction = "bidirectional", torch.onnx.hidden_size = 48 : si64, torch.onnx.layout = 0 : si64} : (!torch.vtensor<[32,32,192],f32>, !torch.vtensor<[2,192,192],f32>, !torch.vtensor<[2,192,48],f32>, !torch.vtensor<[2,384],f32>) -> (!torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[32,2,32,48],f32>, !torch.vtensor<[2,32,48],f32>, !torch.vtensor<[2,32,48],f32>
}

// -----

// CHECK-LABEL:   func.func @test_lstm_batchwise_two_outputs(
// CHECK-SAME:                                               %[[X_LAYOUT_1:.*]]: !torch.vtensor<[3,1,2],f32>,
// CHECK-SAME:                                               %[[W:.*]]: !torch.vtensor<[1,28,2],f32>,
// CHECK-SAME:                                               %[[R:.*]]: !torch.vtensor<[1,28,7],f32>)
// CHECK:       torch.aten.transpose.int
// CHECK:           %[[LOOP_RES:.*]]:3 = torch.prim.Loop %[[MAX_TRIP:.*]], %[[LOOP_COND_FWD:.*]], init(%[[Y:.*]], %[[INITIAL_H:.*]], %[[INITIAL_C:.*]]) {
// CHECK:           ^bb0(%[[LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV:.*]]: !torch.vtensor<[1,3,7],f32>, %[[H_PREV:.*]]: !torch.vtensor<[3,7],f32>, %[[C_PREV:.*]]: !torch.vtensor<[3,7],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK:           }
// CHECK-DAG:           torch.aten.transpose.int
// CHECK-DAG:           torch.aten.transpose.int
// CHECK-DAG:           torch.aten.transpose.int
// CHECK-DAG:           torch.aten.transpose.int
// CHECK:           return %[[Y:.*]], %[[Y_H:.*]] : !torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>
// CHECK:         }

func.func @test_lstm_batchwise_two_outputs(%arg0: !torch.vtensor<[3,1,2],f32>, %arg1: !torch.vtensor<[1,28,2],f32>, %arg2: !torch.vtensor<[1,28,7],f32>) -> (!torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 7 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[3,1,2],f32>, !torch.vtensor<[1,28,2],f32>, !torch.vtensor<[1,28,7],f32>) -> (!torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,1,7],f32>, !torch.vtensor<[3,1,7],f32>
}

      func.func @test_lstm_dynamic(
    %arg0: !torch.vtensor<[?,?,?],f32>,
    %arg1: !torch.vtensor<[1,12,4],f32>,
    %arg2: !torch.vtensor<[1,12,3],f32>,
    %arg3: !torch.vtensor<[1,24],f32>
  ) -> (
    !torch.vtensor<[?,1,?,3],f32>,
    !torch.vtensor<[1,?,3],f32>,
    !torch.vtensor<[1,?,3],f32>
  ) attributes {
    torch.onnx_meta.ir_version = 9 : si64,
    torch.onnx_meta.opset_version = 20 : si64
  } {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.LSTM"(
      %arg0, %arg1, %arg2, %arg3
    ) { torch.onnx.hidden_size = 3 : si64 }
      : (
        !torch.vtensor<[?,?,?],f32>,
        !torch.vtensor<[1,12,4],f32>,
        !torch.vtensor<[1,12,3],f32>,
        !torch.vtensor<[1,24],f32>
      )
      -> (
        !torch.vtensor<[?,1,?,3],f32>,
        !torch.vtensor<[1,?,3],f32>,
        !torch.vtensor<[1,?,3],f32>
      )
    return %0#0, %0#1, %0#2 :
      !torch.vtensor<[?,1,?,3],f32>,
      !torch.vtensor<[1,?,3],f32>,
      !torch.vtensor<[1,?,3],f32>
  }
