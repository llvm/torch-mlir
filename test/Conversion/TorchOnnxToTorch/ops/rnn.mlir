// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch -cse | FileCheck %s
// COM: -cse leaves one torch.constant.int per value (the conversion emits one
// COM: per use), so the checks can capture the dim constant that each
// COM: aten.size.int reads.

// Static shapes: every size folds to a constant.

// CHECK-LABEL:   func.func @test_rnn_static(
// CHECK-NOT:       torch.aten.size.int
// CHECK:           torch.prim.Loop
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[3,2,4],f32>, %{{.*}}: !torch.vtensor<[2,4],f32>):
// CHECK:           return %{{.*}}, %{{.*}} : !torch.vtensor<[3,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>

func.func @test_rnn_static(%arg0: !torch.vtensor<[3,2,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[3,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[3,2,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[3,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>
}

// -----

// Dynamic seq: the size of X dim 0 sizes the Y buffer and is the trip count.

// CHECK-LABEL:   func.func @test_rnn_dynamic_seq(
// CHECK-DAG:       %[[C0:.*]] = torch.constant.int 0
// CHECK:           %[[SEQ:.*]] = torch.aten.size.int %arg0, %[[C0]] : !torch.vtensor<[?,2,5],f32>, !torch.int -> !torch.int
// CHECK:           %[[Y_SHAPE:.*]] = torch.prim.ListConstruct %[[SEQ]], %{{.*}}, %{{.*}} :
// CHECK:           %[[Y0:.*]] = torch.aten.zeros %[[Y_SHAPE]], {{.*}} -> !torch.vtensor<[?,2,4],f32>
// CHECK:           torch.prim.Loop %[[SEQ]], %{{.*}}, init(%[[Y0]], %{{.*}}) {
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[?,2,4],f32>, %{{.*}}: !torch.vtensor<[2,4],f32>):
// CHECK:           return %{{.*}}, %{{.*}} : !torch.vtensor<[?,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>

func.func @test_rnn_dynamic_seq(%arg0: !torch.vtensor<[?,2,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[?,2,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[?,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>
}

// -----

// Dynamic batch, no initial_h: the size of X dim 1 sizes the default initial_h
// and the Y buffer.

// CHECK-LABEL:   func.func @test_rnn_dynamic_batch(
// CHECK-DAG:       %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[BATCH:.*]] = torch.aten.size.int %arg0, %[[C1]] : !torch.vtensor<[3,?,5],f32>, !torch.int -> !torch.int
// CHECK:           %[[H_SHAPE:.*]] = torch.prim.ListConstruct %{{.*}}, %[[BATCH]], %{{.*}} :
// CHECK:           torch.aten.zeros %[[H_SHAPE]], {{.*}} -> !torch.vtensor<[1,?,4],f32>
// CHECK:           %[[BATCH_L:.*]] = torch.aten.size.int %arg0, %[[C1]] : !torch.vtensor<[3,?,5],f32>, !torch.int -> !torch.int
// CHECK:           %[[Y_SHAPE:.*]] = torch.prim.ListConstruct %{{.*}}, %[[BATCH_L]], %{{.*}} :
// CHECK:           torch.aten.zeros %[[Y_SHAPE]], {{.*}} -> !torch.vtensor<[3,?,4],f32>
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[3,?,4],f32>, %{{.*}}: !torch.vtensor<[?,4],f32>):
// CHECK:           return %{{.*}}, %{{.*}} : !torch.vtensor<[3,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>

func.func @test_rnn_dynamic_batch(%arg0: !torch.vtensor<[3,?,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[3,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[3,?,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[3,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>
}

// -----

// layout = 1 (X is [batch, seq, input]) with a dynamic batch: after the layout
// transpose to [seq, batch, input], the default initial_h and the layer read
// batch from dim 1, and seq stays static.

// CHECK-LABEL:   func.func @test_rnn_layout1_dynamic_batch(
// CHECK-DAG:       %[[C0:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[XT:.*]] = torch.aten.transpose.int %arg0, {{.*}} -> !torch.vtensor<[3,?,5],f32>
// CHECK:           %[[BATCH:.*]] = torch.aten.size.int %[[XT]], %[[C1]] : !torch.vtensor<[3,?,5],f32>, !torch.int -> !torch.int
// CHECK:           %[[H_SHAPE:.*]] = torch.prim.ListConstruct %{{.*}}, %[[BATCH]], %{{.*}} :
// CHECK:           torch.aten.zeros %[[H_SHAPE]], {{.*}} -> !torch.vtensor<[1,?,4],f32>
// CHECK-DAG:       %[[C2:.*]] = torch.constant.int 2
// CHECK:           %[[BATCH_L:.*]] = torch.aten.size.int %[[XT]], %[[C1]] : !torch.vtensor<[3,?,5],f32>, !torch.int -> !torch.int
// CHECK:           %[[Y_SHAPE:.*]] = torch.prim.ListConstruct %{{.*}}, %[[BATCH_L]], %{{.*}} :
// CHECK:           %[[LOOP:.*]]:2 = torch.prim.Loop
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[3,?,4],f32>, %{{.*}}: !torch.vtensor<[?,4],f32>):
// CHECK:           %[[YH_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#1, %[[C0]] : !torch.vtensor<[?,4],f32>, !torch.int -> !torch.vtensor<[1,?,4],f32>
// CHECK:           %[[Y_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#0, %[[C1]] : !torch.vtensor<[3,?,4],f32>, !torch.int -> !torch.vtensor<[3,1,?,4],f32>
// CHECK:           %[[YH:.*]] = torch.aten.transpose.int %[[YH_U]], %[[C0]], %[[C1]] : {{.*}} -> !torch.vtensor<[?,1,4],f32>
// CHECK:           %[[Y_T:.*]] = torch.aten.transpose.int %[[Y_U]], %[[C1]], %[[C2]] : {{.*}} -> !torch.vtensor<[3,?,1,4],f32>
// CHECK:           %[[Y:.*]] = torch.aten.transpose.int %[[Y_T]], %[[C0]], %[[C1]] : {{.*}} -> !torch.vtensor<[?,3,1,4],f32>
// CHECK:           return %[[Y]], %[[YH]] : !torch.vtensor<[?,3,1,4],f32>, !torch.vtensor<[?,1,4],f32>

func.func @test_rnn_layout1_dynamic_batch(%arg0: !torch.vtensor<[?,3,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,3,1,4],f32>, !torch.vtensor<[?,1,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[?,3,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,3,1,4],f32>, !torch.vtensor<[?,1,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[?,3,1,4],f32>, !torch.vtensor<[?,1,4],f32>
}

// -----

// layout = 1 with a dynamic seq: after the layout transpose the layer reads seq
// from dim 0 and uses it as the trip count; batch stays static.

// CHECK-LABEL:   func.func @test_rnn_layout1_dynamic_seq(
// CHECK-DAG:       %[[C0:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[XT:.*]] = torch.aten.transpose.int %arg0, {{.*}} -> !torch.vtensor<[?,2,5],f32>
// CHECK-DAG:       %[[C2:.*]] = torch.constant.int 2
// CHECK:           %[[SEQ:.*]] = torch.aten.size.int %[[XT]], %[[C0]] : !torch.vtensor<[?,2,5],f32>, !torch.int -> !torch.int
// CHECK:           %[[Y_SHAPE:.*]] = torch.prim.ListConstruct %[[SEQ]], %{{.*}}, %{{.*}} :
// CHECK:           %[[LOOP:.*]]:2 = torch.prim.Loop %[[SEQ]], %{{.*}}, init(
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[?,2,4],f32>, %{{.*}}: !torch.vtensor<[2,4],f32>):
// CHECK:           %[[YH_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#1, %[[C0]] : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[1,2,4],f32>
// CHECK:           %[[Y_U:.*]] = torch.aten.unsqueeze %[[LOOP]]#0, %[[C1]] : !torch.vtensor<[?,2,4],f32>, !torch.int -> !torch.vtensor<[?,1,2,4],f32>
// CHECK:           %[[YH:.*]] = torch.aten.transpose.int %[[YH_U]], %[[C0]], %[[C1]] : {{.*}} -> !torch.vtensor<[2,1,4],f32>
// CHECK:           %[[Y_T:.*]] = torch.aten.transpose.int %[[Y_U]], %[[C1]], %[[C2]] : {{.*}} -> !torch.vtensor<[?,2,1,4],f32>
// CHECK:           %[[Y:.*]] = torch.aten.transpose.int %[[Y_T]], %[[C0]], %[[C1]] : {{.*}} -> !torch.vtensor<[2,?,1,4],f32>
// CHECK:           return %[[Y]], %[[YH]] : !torch.vtensor<[2,?,1,4],f32>, !torch.vtensor<[2,1,4],f32>

func.func @test_rnn_layout1_dynamic_seq(%arg0: !torch.vtensor<[2,?,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[2,?,1,4],f32>, !torch.vtensor<[2,1,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[2,?,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[2,?,1,4],f32>, !torch.vtensor<[2,1,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[2,?,1,4],f32>, !torch.vtensor<[2,1,4],f32>
}

// -----

// A static batch-1 initial_h with a dynamic X batch: the state is cast to the
// dynamic-batch type before the loop, so the loop-carried state is not baked to
// batch 1.

// CHECK-LABEL:   func.func @test_rnn_static_batch_initial_h_dynamic_x(
// CHECK:           %[[H0_DIR:.*]] = torch.aten.select.int %arg4, %{{.*}}, %{{.*}} : !torch.vtensor<[1,1,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[H0_DIR]] : !torch.vtensor<[1,4],f32> to !torch.vtensor<[?,4],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0]])
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[?,?,4],f32>, %{{.*}}: !torch.vtensor<[?,4],f32>):

func.func @test_rnn_static_batch_initial_h_dynamic_x(%arg0: !torch.vtensor<[?,?,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>, %arg4: !torch.vtensor<[1,1,4],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %none, %arg4) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[?,?,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>, !torch.none, !torch.vtensor<[1,1,4],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>
}

// -----

// Dynamic input size in X with a static W: checked at run time against W's
// input size.

// CHECK-LABEL:   func.func @test_rnn_dynamic_input_size(
// CHECK-DAG:       %[[C2:.*]] = torch.constant.int 2
// CHECK:           %[[IN:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK-DAG:       %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[EQ:.*]] = torch.aten.eq.int %[[IN]], %[[C5]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[EQ]], "The input_size of W must equal X."
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[?,?,4],f32>, %{{.*}}: !torch.vtensor<[?,4],f32>):

func.func @test_rnn_dynamic_input_size(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>
}

// -----

// initial_h with a dynamic hidden dim: refined to the hidden_size attribute.

// CHECK-LABEL:   func.func @test_rnn_dynamic_initial_h(
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %arg4 : !torch.vtensor<[?,?,?],f32> to !torch.vtensor<[?,?,4],f32>
// CHECK:           torch.aten.select.int %[[H0]], %{{.*}}, %{{.*}} : !torch.vtensor<[?,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[?,?,4],f32>, %{{.*}}: !torch.vtensor<[?,4],f32>):

func.func @test_rnn_dynamic_initial_h(%arg0: !torch.vtensor<[?,?,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>, %arg4: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %none, %arg4) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[?,?,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>, !torch.none, !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>
}

// -----

// A declared Y more static than the computed one (a static Y for a dynamic X):
// Y is cast to the declared type; Y_h already matches and is not cast.

// CHECK-LABEL:   func.func @test_rnn_static_result_dynamic_x(
// CHECK:           %[[YH:.*]] = torch.aten.unsqueeze %{{.*}}#1, %{{.*}} : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[1,3,4],f32>
// CHECK:           %[[Y:.*]] = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[?,1,3,4],f32> to !torch.vtensor<[2,1,3,4],f32>
// CHECK:           return %[[Y]], %[[YH]] : !torch.vtensor<[2,1,3,4],f32>, !torch.vtensor<[1,3,4],f32>

func.func @test_rnn_static_result_dynamic_x(%arg0: !torch.vtensor<[?,3,5],f32>, %arg1: !torch.vtensor<[1,4,5],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[2,1,3,4],f32>, !torch.vtensor<[1,3,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[?,3,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[2,1,3,4],f32>, !torch.vtensor<[1,3,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[2,1,3,4],f32>, !torch.vtensor<[1,3,4],f32>
}

// -----

// Dynamic input size in both X and W: the two sizes are checked at run time.

// CHECK-LABEL:   func.func @test_rnn_dynamic_input_size_dynamic_w(
// CHECK-DAG:       %[[C2:.*]] = torch.constant.int 2
// CHECK:           %[[IN:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[W_IN:.*]] = torch.aten.size.int %arg1, %[[C2]] : {{.*}} -> !torch.int
// CHECK:           %[[EQ:.*]] = torch.aten.eq.int %[[IN]], %[[W_IN]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[EQ]], "The input_size of W must equal X."
// CHECK:           torch.prim.Loop

func.func @test_rnn_dynamic_input_size_dynamic_w(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[1,4,?],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1,4,?],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[?,1,?,4],f32>, !torch.vtensor<[1,?,4],f32>
}

// -----

// Static input size in X with a dynamic W: W's input size is checked at run
// time against X's.

// CHECK-LABEL:   func.func @test_rnn_static_input_size_dynamic_w(
// CHECK-DAG:       %[[C2:.*]] = torch.constant.int 2
// CHECK-DAG:       %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[W_IN:.*]] = torch.aten.size.int %arg1, %[[C2]] : {{.*}} -> !torch.int
// CHECK:           %[[EQ:.*]] = torch.aten.eq.int %[[C5]], %[[W_IN]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[EQ]], "The input_size of W must equal X."
// CHECK:           torch.prim.Loop

func.func @test_rnn_static_input_size_dynamic_w(%arg0: !torch.vtensor<[3,2,5],f32>, %arg1: !torch.vtensor<[1,4,?],f32>, %arg2: !torch.vtensor<[1,4,4],f32>, %arg3: !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[3,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 4 : si64} : (!torch.vtensor<[3,2,5],f32>, !torch.vtensor<[1,4,?],f32>, !torch.vtensor<[1,4,4],f32>, !torch.vtensor<[1,8],f32>) -> (!torch.vtensor<[3,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,4],f32>, !torch.vtensor<[1,2,4],f32>
}

// -----

// layout = 1 with a provided initial_h [batch, num_directions, hidden] that has
// a static batch 1 and a dynamic hidden dim: initial_h is transposed to layout
// 0, its hidden dim is refined to hidden_size, and the per-direction state is
// cast to X's dynamic batch.

// CHECK-LABEL:   func.func @test_rnn_layout1_initial_h(
// CHECK-DAG:       %[[C0:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[H0_T:.*]] = torch.aten.transpose.int %arg4, %[[C0]], %[[C1]] : !torch.vtensor<[1,1,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,?],f32>
// CHECK:           %[[H0_R:.*]] = torch.tensor_static_info_cast %[[H0_T]] : !torch.vtensor<[1,1,?],f32> to !torch.vtensor<[1,1,4],f32>
// CHECK:           %[[H0_DIR:.*]] = torch.aten.select.int %[[H0_R]], %[[C0]], %[[C0]] : !torch.vtensor<[1,1,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
// CHECK:           %[[H0:.*]] = torch.tensor_static_info_cast %[[H0_DIR]] : !torch.vtensor<[1,4],f32> to !torch.vtensor<[?,4],f32>
// CHECK:           torch.prim.Loop %{{.*}}, %{{.*}}, init(%{{.*}}, %[[H0]])
// CHECK:           ^bb0(%{{.*}}: !torch.int, %{{.*}}: !torch.vtensor<[3,?,4],f32>, %{{.*}}: !torch.vtensor<[?,4],f32>):

func.func @test_rnn_layout1_initial_h(%arg0: !torch.vtensor<[?,3,5],f32>, %arg1: !torch.vtensor<[1,12,5],f32>, %arg2: !torch.vtensor<[1,12,4],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[1,1,?],f32>) -> (!torch.vtensor<[?,3,1,4],f32>, !torch.vtensor<[?,1,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %none, %arg4) {torch.onnx.hidden_size = 4 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[?,3,5],f32>, !torch.vtensor<[1,12,5],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,24],f32>, !torch.none, !torch.vtensor<[1,1,?],f32>) -> (!torch.vtensor<[?,3,1,4],f32>, !torch.vtensor<[?,1,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[?,3,1,4],f32>, !torch.vtensor<[?,1,4],f32>
}
