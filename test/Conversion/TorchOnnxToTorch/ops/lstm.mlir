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

// -----

// CHECK-LABEL:   func.func @test_lstm_dynamic(
// CHECK-SAME:                               %[[X:.*]]: !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:                               %[[W:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                               %[[R:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                               %[[B:.*]]: !torch.vtensor<[1,24],f32>)
// CHECK:           torch.runtime.assert %[[EQ:.*]], "The input_size of W must equal X."
// CHECK:           %[[LOOP_RESULT:.*]]:3 = torch.prim.Loop %[[MAX_TRIPS:.*]], %[[ENTER_LOOP:.*]], init(%[[Y:.*]], %[[INITIAL_H:.*]], %[[INITIAL_C:.*]]) {
// CHECK:           ^bb0(%[[LOOP_INDEX:.*]]: !torch.int, %[[Y_PREV:.*]]: !torch.vtensor<[?,?,3],f32>, %[[H_PREV:.*]]: !torch.vtensor<[?,3],f32>, %[[C_PREV:.*]]: !torch.vtensor<[?,3],f32>):
// CHECK-DAG:             torch.aten.select.int
// CHECK-DAG:             torch.aten.linear
// CHECK-DAG:             torch.aten.sigmoid
// CHECK-DAG:             torch.aten.tanh
// CHECK-DAG:             torch.prim.Loop.condition
// CHECK-DAG:           }
// CHECK:         }

 func.func @test_lstm_dynamic(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// When initial_h / initial_c are provided with a dynamic hidden dim (e.g. from
// an onnx.Tile broadcasting the initial state to a dynamic batch dim), the
// expander must recover the static hidden_size from the attribute so the loop
// body carries the static hidden dim instead of propagating an unknown size
// that leaves an unresolvable materialization against the static result type.

// CHECK-LABEL:   func.func @test_lstm_dynamic_initial_state(
// CHECK-SAME:                               %[[X:.*]]: !torch.vtensor<[?,?,4],f32>,
// CHECK-SAME:                               %[[W:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                               %[[R:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                               %[[B:.*]]: !torch.vtensor<[1,24],f32>,
// CHECK-SAME:                               %[[INITIAL_H:.*]]: !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:                               %[[INITIAL_C:.*]]: !torch.vtensor<[?,?,?],f32>)
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[INITIAL_H]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %[[INITIAL_C]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[H0_FWD:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_FWD:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_FWD]], %[[C0_FWD]]) {
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[?,?,3],f32>, %{{.*}}: !torch.vtensor<[?,3],f32>, %{{.*}}: !torch.vtensor<[?,3],f32>):
// CHECK:           }

func.func @test_lstm_dynamic_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[?,?,?],f32>, %arg5: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// When initial_h / initial_c already carry a static hidden dim, no refinement
// cast is emitted: the dim is only refined when it is dynamic. This also guards
// against emitting a verifier-invalid cast for a statically-mismatched dim.

// CHECK-LABEL:   func.func @test_lstm_static_initial_state(
// CHECK-SAME:                               %[[INITIAL_H:.*]]: !torch.vtensor<[1,?,3],f32>,
// CHECK-SAME:                               %[[INITIAL_C:.*]]: !torch.vtensor<[1,?,3],f32>)
// CHECK-NOT:       torch.tensor_static_info_cast %[[INITIAL_H]]
// CHECK-NOT:       torch.tensor_static_info_cast %[[INITIAL_C]]

func.func @test_lstm_static_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[1,?,3],f32>, %arg5: !torch.vtensor<[1,?,3],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}

// -----

// layout=1 with a dynamic initial state: the layout transpose runs before the
// refinement, so the hidden dim (still the trailing dim after the transpose) is
// pinned to the static hidden_size on the transposed value.

// CHECK-LABEL:   func.func @test_lstm_layout1_dynamic_initial_state(
// CHECK:           %[[TH:.*]] = torch.aten.transpose.int %arg4, %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[TH]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[TC:.*]] = torch.aten.transpose.int %arg5, %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %[[TC]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[H0_FWD:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_FWD:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_FWD]], %[[C0_FWD]])

func.func @test_lstm_layout1_dynamic_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[?,?,?],f32>, %arg5: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,1,3],f32>, !torch.vtensor<[?,1,3],f32>, !torch.vtensor<[?,1,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64, torch.onnx.layout = 1 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>)-> (!torch.vtensor<[?,?,1,3],f32>, !torch.vtensor<[?,1,3],f32>, !torch.vtensor<[?,1,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,?,1,3],f32>, !torch.vtensor<[?,1,3],f32>, !torch.vtensor<[?,1,3],f32>
}

// -----

// bidirectional with a dynamic initial state: the refinement runs once on the
// combined [num_directions, batch, hidden] operand before the forward/reverse
// split, pinning the hidden dim to the static hidden_size.

// CHECK-LABEL:   func.func @test_lstm_bidirectional_dynamic_initial_state(
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %arg4 : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %arg5 : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,3],f32>
// CHECK:           %[[H0_FWD:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_FWD:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[H0_REV:.*]] = torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0_REV:.*]] = torch.aten.select.int %[[C0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_FWD]], %[[C0_FWD]])
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0_REV]], %[[C0_REV]])

func.func @test_lstm_bidirectional_dynamic_initial_state(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[2,12,4],f32>, %arg2: !torch.vtensor<[2,12,3],f32>, %arg3: !torch.vtensor<[2,24],f32>, %arg4: !torch.vtensor<[?,?,?],f32>, %arg5: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,2,?,3],f32>, !torch.vtensor<[2,?,3],f32>, !torch.vtensor<[2,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64, torch.onnx.direction = "bidirectional" }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[2,12,4],f32>, !torch.vtensor<[2,12,3],f32>, !torch.vtensor<[2,24],f32>, !torch.none, !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>)-> (!torch.vtensor<[?,2,?,3],f32>, !torch.vtensor<[2,?,3],f32>, !torch.vtensor<[2,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,2,?,3],f32>, !torch.vtensor<[2,?,3],f32>, !torch.vtensor<[2,?,3],f32>
}

// -----

// A statically-batched initial_h/initial_c (batch=1, as emitted by common ONNX
// exporters) combined with a dynamic X batch must not bake the loop-carried
// hidden/cell state to batch 1: the state batch tracks X's batch. The provided
// states are cast to the dynamic-batch state type before the loop, and each
// gate matmul against the dynamic-batch Xt stays dynamic ([?,3], not [1,3]).

// CHECK-LABEL:   func.func @test_lstm_static_batch_dynamic_x(
// CHECK:           %[[H0_DIR:.*]] = torch.aten.select.int %arg4, %{{.*}}, %{{.*}} : !torch.vtensor<[1,1,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,3],f32>
// CHECK:           %[[C0_DIR:.*]] = torch.aten.select.int %arg5, %{{.*}}, %{{.*}} : !torch.vtensor<[1,1,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,3],f32>
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[H0_DIR]] : !torch.vtensor<[1,3],f32> to !torch.vtensor<[?,3],f32>
// CHECK:           %[[C0:.*]] = torch.tensor_static_info_cast %[[C0_DIR]] : !torch.vtensor<[1,3],f32> to !torch.vtensor<[?,3],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0]], %[[C0]])
// CHECK:           torch.aten.linear %{{.*}} : !torch.vtensor<[?,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[?,3],f32>
// CHECK-NOT:       -> !torch.vtensor<[1,3],f32>

func.func @test_lstm_static_batch_dynamic_x(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[1,1,3],f32>, %arg5: !torch.vtensor<[1,1,3],f32>) -> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64} {
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %arg4, %arg5) { torch.onnx.hidden_size = 3 : si64 }: (!torch.vtensor<[?,?,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[1,1,3],f32>, !torch.vtensor<[1,1,3],f32>)-> (!torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,1,?,3],f32>, !torch.vtensor<[1,?,3],f32>, !torch.vtensor<[1,?,3],f32>
}
