// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s
// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch="gru-split-gates-min-elements=2000" | FileCheck %s --check-prefix=SPLIT

// CHECK-LABEL:   func.func @test_gru_forward(
// The input projection of all three gates runs once, before the loop.
// CHECK:           torch.aten.linear {{.*}} -> !torch.vtensor<[8,15],f32>
// CHECK:           torch.prim.Loop
// CHECK:             torch.aten.select.int {{.*}} -> !torch.vtensor<[2,15],f32>
// With linear_before_reset, one matmul gives H.R^T of all three gates.
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[2,15],f32>
// z and r share one sigmoid.
// CHECK:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[2,10],f32>
// CHECK-NOT:         torch.aten.linear
// CHECK-NOT:         torch.aten.sigmoid
// CHECK:             torch.aten.tanh
// CHECK:             torch.prim.Loop.condition

func.func @test_gru_forward(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>, %arg3: !torch.vtensor<[1,30],f32>) -> (!torch.vtensor<[4,1,2,5],f32>, !torch.vtensor<[1,2,5],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 5 : si64, torch.onnx.linear_before_reset = 1 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>, !torch.vtensor<[1,30],f32>) -> (!torch.vtensor<[4,1,2,5],f32>, !torch.vtensor<[1,2,5],f32>)
  return %0#0, %0#1 : !torch.vtensor<[4,1,2,5],f32>, !torch.vtensor<[1,2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @test_gru_reverse(
// CHECK:           torch.aten.linear {{.*}} -> !torch.vtensor<[8,15],f32>
// CHECK:           torch.prim.Loop
// CHECK:           ^bb0(%[[I:.*]]: !torch.int,
// The reverse layer goes from the last timestep to the first.
// CHECK:             %[[T:.*]] = torch.aten.sub.int %{{.*}}, %[[I]]
// CHECK:             torch.aten.select.int %{{.*}}, %{{.*}}, %[[T]]
// Without linear_before_reset, the packed matmul holds z and r only.
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[2,10],f32>
// CHECK:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[2,10],f32>
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[2,5],f32>
// CHECK:             torch.aten.tanh
// CHECK:             torch.aten.slice_scatter %{{.*}}, %{{.*}}, %{{.*}}, %[[T]],

func.func @test_gru_reverse(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[1,15,3],f32>, %arg2: !torch.vtensor<[1,15,5],f32>) -> !torch.vtensor<[1,2,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.direction = "reverse", torch.onnx.hidden_size = 5 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[1,15,3],f32>, !torch.vtensor<[1,15,5],f32>) -> (!torch.none, !torch.vtensor<[1,2,5],f32>)
  return %0#1 : !torch.vtensor<[1,2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @test_gru_bidirectional(
// CHECK:           torch.prim.Loop
// CHECK-NOT:         torch.aten.sub.int
// CHECK:             torch.prim.Loop.condition
// CHECK:           torch.prim.Loop
// CHECK:             torch.aten.sub.int
// CHECK:             torch.prim.Loop.condition
// CHECK:           torch.aten.cat {{.*}} -> !torch.vtensor<[4,2,2,5],f32>
// CHECK:           torch.aten.cat {{.*}} -> !torch.vtensor<[2,2,5],f32>

func.func @test_gru_bidirectional(%arg0: !torch.vtensor<[4,2,3],f32>, %arg1: !torch.vtensor<[2,15,3],f32>, %arg2: !torch.vtensor<[2,15,5],f32>, %arg3: !torch.vtensor<[2,30],f32>) -> (!torch.vtensor<[4,2,2,5],f32>, !torch.vtensor<[2,2,5],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.direction = "bidirectional", torch.onnx.hidden_size = 5 : si64, torch.onnx.linear_before_reset = 1 : si64} : (!torch.vtensor<[4,2,3],f32>, !torch.vtensor<[2,15,3],f32>, !torch.vtensor<[2,15,5],f32>, !torch.vtensor<[2,30],f32>) -> (!torch.vtensor<[4,2,2,5],f32>, !torch.vtensor<[2,2,5],f32>)
  return %0#0, %0#1 : !torch.vtensor<[4,2,2,5],f32>, !torch.vtensor<[2,2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @test_gru_large_batch(
// CHECK:           torch.aten.linear {{.*}} -> !torch.vtensor<[128,48],f32>
// CHECK:           torch.prim.Loop
// CHECK:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,48],f32>
// CHECK:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[64,32],f32>

// The packed input projection has 2 * 64 * 48 elements and the packed recurrent
// matmul 64 * 48. Both reach 2000, so each gate gets its own matmul.
// SPLIT-LABEL:   func.func @test_gru_large_batch(
// SPLIT-COUNT-3:   torch.aten.linear {{.*}} -> !torch.vtensor<[128,16],f32>
// SPLIT:           torch.prim.Loop
// SPLIT:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.sigmoid {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.linear {{.*}} -> !torch.vtensor<[64,16],f32>
// SPLIT:             torch.aten.tanh

func.func @test_gru_large_batch(%arg0: !torch.vtensor<[2,64,8],f32>, %arg1: !torch.vtensor<[1,48,8],f32>, %arg2: !torch.vtensor<[1,48,16],f32>, %arg3: !torch.vtensor<[1,96],f32>) -> (!torch.vtensor<[2,1,64,16],f32>, !torch.vtensor<[1,64,16],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 16 : si64, torch.onnx.linear_before_reset = 1 : si64} : (!torch.vtensor<[2,64,8],f32>, !torch.vtensor<[1,48,8],f32>, !torch.vtensor<[1,48,16],f32>, !torch.vtensor<[1,96],f32>) -> (!torch.vtensor<[2,1,64,16],f32>, !torch.vtensor<[1,64,16],f32>)
  return %0#0, %0#1 : !torch.vtensor<[2,1,64,16],f32>, !torch.vtensor<[1,64,16],f32>
}
