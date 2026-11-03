// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch --split-input-file | FileCheck %s --implicit-check-not=torch.operator

// com.microsoft::MultiHeadAttention lowering to
// aten.scaled_dot_product_attention. Notation: B batch, S query seq, L kv seq,
// P past seq, N num_heads, H / Hv head sizes.

// Cross-length 3D Q/K/V, no bias, default scale (SDPA scale = none).
//   query [1, 3, 8], key/value [1, 5, 8], N = 2, H = 4.

// CHECK-LABEL: func.func @test_mha_basic
// CHECK-SAME: (%[[Q:.*]]: !torch.vtensor<[1,3,8],f32>, %[[K:.*]]: !torch.vtensor<[1,5,8],f32>, %[[V:.*]]: !torch.vtensor<[1,5,8],f32>)
func.func @test_mha_basic(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,8],f32>, %v: !torch.vtensor<[1,5,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[QU:.*]] = torch.aten.unflatten.int %[[Q]], %{{.*}}, %{{.*}} : !torch.vtensor<[1,3,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,3,2,4],f32>
  // CHECK: %[[QH:.*]] = torch.aten.transpose.int %[[QU]], %{{.*}}, %{{.*}} : !torch.vtensor<[1,3,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[KU:.*]] = torch.aten.unflatten.int %[[K]], {{.*}} -> !torch.vtensor<[1,5,2,4],f32>
  // CHECK: %[[KH:.*]] = torch.aten.transpose.int %[[KU]], {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[VU:.*]] = torch.aten.unflatten.int %[[V]], {{.*}} -> !torch.vtensor<[1,5,2,4],f32>
  // CHECK: %[[VH:.*]] = torch.aten.transpose.int %[[VU]], {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[ATT:.*]] = torch.aten.scaled_dot_product_attention %[[QH]], %[[KH]], %[[VH]], %[[NONE]], %{{.*}}, %{{.*}}, %[[NONE]], %{{.*}} : {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[OT:.*]] = torch.aten.transpose.int %[[ATT]], {{.*}} -> !torch.vtensor<[1,3,2,4],f32>
  // CHECK: %[[OUT:.*]] = torch.aten.flatten.using_ints %[[OT]], {{.*}} -> !torch.vtensor<[1,3,8],f32>
  // CHECK: return %[[OUT]]
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[1,5,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// scale = 0 selects the default 1/sqrt(H), as in ORT: SDPA scale is none.

// CHECK-LABEL: func.func @test_mha_scale_zero
func.func @test_mha_scale_zero(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %[[NONE]], %{{.*}}, %{{.*}}, %[[NONE]], %{{.*}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64, torch.onnx.scale = 0.000000e+00 : f32} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Explicit scale, bias, and v_hidden != hidden: Hv = 6, bias = 8 + 8 + 12.

// CHECK-LABEL: func.func @test_mha_bias_vhidden
// CHECK-SAME: (%[[Q:.*]]: !torch.vtensor<[1,3,8],f32>, %[[K:.*]]: !torch.vtensor<[1,5,8],f32>, %[[V:.*]]: !torch.vtensor<[1,5,12],f32>, %[[B:.*]]: !torch.vtensor<[28],f32>)
func.func @test_mha_bias_vhidden(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,8],f32>, %v: !torch.vtensor<[1,5,12],f32>, %b: !torch.vtensor<[28],f32>) -> !torch.vtensor<[1,3,12],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[DIM:.*]] = torch.constant.int 0
  // CHECK: torch.constant.bool false
  // CHECK: %[[S0:.*]] = torch.constant.int 0
  // CHECK: %[[E0:.*]] = torch.constant.int 8
  // CHECK: %[[BQ:.*]] = torch.aten.slice.Tensor %[[B]], %[[DIM]], %[[S0]], %[[E0]], %{{.*}} : {{.*}} -> !torch.vtensor<[8],f32>
  // CHECK: torch.aten.add.Tensor %[[Q]], %[[BQ]], %{{.*}} : !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[1,3,8],f32>
  // CHECK: %[[S1:.*]] = torch.constant.int 8
  // CHECK: %[[E1:.*]] = torch.constant.int 16
  // CHECK: %[[BK:.*]] = torch.aten.slice.Tensor %[[B]], %[[DIM]], %[[S1]], %[[E1]], %{{.*}} : {{.*}} -> !torch.vtensor<[8],f32>
  // CHECK: torch.aten.add.Tensor %[[K]], %[[BK]], %{{.*}} : !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[1,5,8],f32>
  // CHECK: %[[S2:.*]] = torch.constant.int 16
  // CHECK: %[[E2:.*]] = torch.constant.int 28
  // CHECK: %[[BV:.*]] = torch.aten.slice.Tensor %[[B]], %[[DIM]], %[[S2]], %[[E2]], %{{.*}} : {{.*}} -> !torch.vtensor<[12],f32>
  // CHECK: torch.aten.add.Tensor %[[V]], %[[BV]], %{{.*}} : !torch.vtensor<[1,5,12],f32>, !torch.vtensor<[12],f32>, !torch.int -> !torch.vtensor<[1,5,12],f32>
  // CHECK: torch.aten.unflatten.int {{.*}} -> !torch.vtensor<[1,5,2,6],f32>
  // CHECK: %[[SCALE:.*]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SCALE]], %{{.*}} : !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,5,4],f32>, !torch.vtensor<[1,2,5,6],f32>, {{.*}} -> !torch.vtensor<[1,2,3,6],f32>
  // CHECK: torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[1,3,12],f32>
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %b) {torch.onnx.num_heads = 2 : si64, torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[1,5,12],f32>, !torch.vtensor<[28],f32>) -> !torch.vtensor<[1,3,12],f32>
  return %0 : !torch.vtensor<[1,3,12],f32>
}

// -----

// Self-attention without past: unidirectional masks j > i with the lowest
// finite f32 value; no past offset is added to the row index.

// CHECK-LABEL: func.func @test_mha_unidirectional
func.func @test_mha_unidirectional(%q: !torch.vtensor<[1,4,8],f32>, %k: !torch.vtensor<[1,4,8],f32>, %v: !torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[ROWS:.*]] = torch.aten.arange {{.*}} -> !torch.vtensor<[4],si64>
  // CHECK-NOT: torch.aten.add.Scalar
  // CHECK: %[[COLS:.*]] = torch.aten.arange {{.*}} -> !torch.vtensor<[4],si64>
  // CHECK: %[[C2:.*]] = torch.aten.unsqueeze %[[COLS]], {{.*}} -> !torch.vtensor<[1,4],si64>
  // CHECK: %[[R2:.*]] = torch.aten.unsqueeze %[[ROWS]], {{.*}} -> !torch.vtensor<[4,1],si64>
  // CHECK: %[[FUT:.*]] = torch.aten.gt.Tensor %[[C2]], %[[R2]] : !torch.vtensor<[1,4],si64>, !torch.vtensor<[4,1],si64> -> !torch.vtensor<[4,4],i1>
  // CHECK: %[[FF:.*]] = torch.aten.to.dtype %[[FUT]], {{.*}} -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[LOW:.*]] = torch.constant.float -3.4028234663852886E+38
  // CHECK: %[[ADD:.*]] = torch.aten.mul.Scalar %[[FF]], %[[LOW]]
  // CHECK: torch.aten.unsqueeze %[[ADD]], {{.*}} -> !torch.vtensor<[1,4,4],f32>
  // CHECK: %[[MASK:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK]], %{{.*}}, %[[FALSE:.*]], %{{.*}}, %[[FALSE]] :
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// past_key / past_value + unidirectional, three outputs: present = cat(past,
// current) along dim 2; causal row index offset by P = 2.

// CHECK-LABEL: func.func @test_mha_past_unidirectional
// CHECK-SAME: %[[PK:[a-zA-Z0-9]+]]: !torch.vtensor<[1,2,2,4],f32>, %[[PV:[a-zA-Z0-9]+]]: !torch.vtensor<[1,2,2,4],f32>)
func.func @test_mha_past_unidirectional(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %pk: !torch.vtensor<[1,2,2,4],f32>, %pv: !torch.vtensor<[1,2,2,4],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>, !torch.vtensor<[1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QH:.*]] = torch.aten.transpose.int {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[KH:.*]] = torch.aten.transpose.int {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[VH:.*]] = torch.aten.transpose.int {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[KL:.*]] = torch.prim.ListConstruct %[[PK]], %[[KH]]
  // CHECK: %[[PRK:.*]] = torch.aten.cat %[[KL]], %{{.*}} : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[VL:.*]] = torch.prim.ListConstruct %[[PV]], %[[VH]]
  // CHECK: %[[PRV:.*]] = torch.aten.cat %[[VL]], %{{.*}} : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[ROWS:.*]] = torch.aten.arange {{.*}} -> !torch.vtensor<[3],si64>
  // CHECK: %[[P:.*]] = torch.constant.int 2
  // CHECK: %[[ROWSP:.*]] = torch.aten.add.Scalar %[[ROWS]], %[[P]], %{{.*}} : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[3],si64>
  // CHECK: torch.aten.arange {{.*}} -> !torch.vtensor<[5],si64>
  // CHECK: torch.aten.unsqueeze %[[ROWSP]], {{.*}} -> !torch.vtensor<[3,1],si64>
  // CHECK: torch.aten.gt.Tensor {{.*}} -> !torch.vtensor<[3,5],i1>
  // CHECK: %[[MASK:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[1,1,3,5],f32>
  // CHECK: %[[ATT:.*]] = torch.aten.scaled_dot_product_attention %[[QH]], %[[PRK]], %[[PRV]], %[[MASK]], {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[OUT:.*]] = torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[1,3,8],f32>
  // CHECK: return %[[OUT]], %[[PRK]], %[[PRV]]
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>, !torch.vtensor<[1,2,5,4],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>, !torch.vtensor<[1,2,5,4],f32>
}

// -----

// Two outputs (output, present_key); present_value is not materialised.

// CHECK-LABEL: func.func @test_mha_present_key_only
func.func @test_mha_present_key_only(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %pk: !torch.vtensor<[1,2,2,4],f32>, %pv: !torch.vtensor<[1,2,2,4],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[PRK:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[OUT:.*]] = torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[1,3,8],f32>
  // CHECK: return %[[OUT]], %[[PRK]] : !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>
}

// -----

// present outputs without past: present_key / present_value are the
// reshaped current key / value.

// CHECK-LABEL: func.func @test_mha_present_no_past
func.func @test_mha_present_no_past(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,3,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK-NOT: torch.aten.cat
  // CHECK: %[[QH:.*]] = torch.aten.transpose.int {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[KH:.*]] = torch.aten.transpose.int {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[VH:.*]] = torch.aten.transpose.int {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK-NOT: torch.aten.cat
  // CHECK: torch.aten.scaled_dot_product_attention %[[QH]], %[[KH]], %[[VH]],
  // CHECK: return %{{.*}}, %[[KH]], %[[VH]]
  %0:3 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,3,4],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,3,4],f32>
}

// -----

// [B, T] key_padding_mask with mask_filter_value: padded keys (== 0) get the
// filter value, broadcast as [B, 1, 1, T].

// CHECK-LABEL: func.func @test_mha_key_padding_mask_2d
// CHECK-SAME: %[[M:[a-zA-Z0-9]+]]: !torch.vtensor<[2,5],si32>)
func.func @test_mha_key_padding_mask_2d(%q: !torch.vtensor<[2,3,8],f32>, %k: !torch.vtensor<[2,5,8],f32>, %v: !torch.vtensor<[2,5,8],f32>, %m: !torch.vtensor<[2,5],si32>) -> !torch.vtensor<[2,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[PAD:.*]] = torch.aten.eq.Scalar %[[M]], %{{.*}} : !torch.vtensor<[2,5],si32>, !torch.int -> !torch.vtensor<[2,5],i1>
  // CHECK: %[[PADF:.*]] = torch.aten.to.dtype %[[PAD]], {{.*}} -> !torch.vtensor<[2,5],f32>
  // CHECK: %[[FILL:.*]] = torch.constant.float -1.000000e+03
  // CHECK: %[[ADD:.*]] = torch.aten.mul.Scalar %[[PADF]], %[[FILL]]
  // CHECK: %[[U1:.*]] = torch.aten.unsqueeze %[[ADD]], {{.*}} -> !torch.vtensor<[2,1,5],f32>
  // CHECK: %[[MASK:.*]] = torch.aten.unsqueeze %[[U1]], {{.*}} -> !torch.vtensor<[2,1,1,5],f32>
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK]],
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64, torch.onnx.mask_filter_value = -1.000000e+03 : f32} : (!torch.vtensor<[2,3,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.none, !torch.vtensor<[2,5],si32>) -> !torch.vtensor<[2,3,8],f32>
  return %0 : !torch.vtensor<[2,3,8],f32>
}

// -----

// [B, S, T] key_padding_mask, broadcast as [B, 1, S, T].

// CHECK-LABEL: func.func @test_mha_key_padding_mask_3d
func.func @test_mha_key_padding_mask_3d(%q: !torch.vtensor<[2,3,8],f32>, %k: !torch.vtensor<[2,5,8],f32>, %v: !torch.vtensor<[2,5,8],f32>, %m: !torch.vtensor<[2,3,5],si32>) -> !torch.vtensor<[2,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.eq.Scalar {{.*}} -> !torch.vtensor<[2,3,5],i1>
  // CHECK: %[[FILL:.*]] = torch.constant.float -1.000000e+04
  // CHECK: torch.aten.mul.Scalar %{{.*}}, %[[FILL]]
  // CHECK: %[[MASK:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[2,1,3,5],f32>
  // CHECK-NOT: torch.aten.unsqueeze
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK]],
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[2,3,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.none, !torch.vtensor<[2,3,5],si32>) -> !torch.vtensor<[2,3,8],f32>
  return %0 : !torch.vtensor<[2,3,8],f32>
}

// -----

// [B] key lengths (right-side padding), combined with unidirectional and
// attention_bias: the three additive contributions are summed with
// broadcast-derived static types.

// CHECK-LABEL: func.func @test_mha_combined_masks
// CHECK-SAME: %[[M:[a-zA-Z0-9]+]]: !torch.vtensor<[2],si32>, %[[AB:[a-zA-Z0-9]+]]: !torch.vtensor<[1,2,4,4],f32>)
func.func @test_mha_combined_masks(%q: !torch.vtensor<[2,4,8],f32>, %k: !torch.vtensor<[2,4,8],f32>, %v: !torch.vtensor<[2,4,8],f32>, %m: !torch.vtensor<[2],si32>, %ab: !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[2,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[CAUSAL:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[LEN:.*]] = torch.aten.to.dtype %[[M]], {{.*}} -> !torch.vtensor<[2],si64>
  // CHECK: %[[POS:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[1,4],si64>
  // CHECK: %[[LEN2:.*]] = torch.aten.unsqueeze %[[LEN]], {{.*}} -> !torch.vtensor<[2,1],si64>
  // CHECK: %[[PAD:.*]] = torch.aten.ge.Tensor %[[POS]], %[[LEN2]] : {{.*}} -> !torch.vtensor<[2,4],i1>
  // CHECK: %[[KPM:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[2,1,1,4],f32>
  // CHECK: %[[S1:.*]] = torch.aten.add.Tensor %[[CAUSAL]], %[[KPM]], %{{.*}} : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[2,1,1,4],f32>, !torch.int -> !torch.vtensor<[2,1,4,4],f32>
  // CHECK: %[[S2:.*]] = torch.aten.add.Tensor %[[S1]], %[[AB]], %{{.*}} : !torch.vtensor<[2,1,4,4],f32>, !torch.vtensor<[1,2,4,4],f32>, !torch.int -> !torch.vtensor<[2,2,4,4],f32>
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %[[S2]],
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m, %ab) {torch.onnx.num_heads = 2 : si64, torch.onnx.unidirectional = 1 : si64, torch.onnx.mask_filter_value = -5.000000e+00 : f32} : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>, !torch.none, !torch.vtensor<[2],si32>, !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[2,4,8],f32>
  return %0 : !torch.vtensor<[2,4,8],f32>
}

// -----

// attention_bias alone is passed to SDPA unchanged.

// CHECK-LABEL: func.func @test_mha_attention_bias
// CHECK-SAME: %[[AB:[a-zA-Z0-9]+]]: !torch.vtensor<[1,1,3,5],f32>)
func.func @test_mha_attention_bias(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,8],f32>, %v: !torch.vtensor<[1,5,8],f32>, %ab: !torch.vtensor<[1,1,3,5],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %[[AB]],
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %ab) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.none, !torch.none, !torch.vtensor<[1,1,3,5],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Cross-attention with [B, N, L, H] key/value: used as-is, bias applies to
// the query slice only.

// CHECK-LABEL: func.func @test_mha_cross_bnsh
// CHECK-SAME: (%[[Q:.*]]: !torch.vtensor<[1,3,8],f32>, %[[K:.*]]: !torch.vtensor<[1,2,5,4],f32>, %[[V:.*]]: !torch.vtensor<[1,2,5,4],f32>, %[[B:.*]]: !torch.vtensor<[24],f32>)
func.func @test_mha_cross_bnsh(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,2,5,4],f32>, %v: !torch.vtensor<[1,2,5,4],f32>, %b: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.slice.Tensor %[[B]], {{.*}} -> !torch.vtensor<[8],f32>
  // CHECK: torch.aten.add.Tensor %[[Q]],
  // CHECK-NOT: torch.aten.slice.Tensor
  // CHECK-NOT: torch.aten.cat
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %[[K]], %[[V]],
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %b) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,5,4],f32>, !torch.vtensor<[1,2,5,4],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Packed QKV [B, S, N, 3, H] with bias: each slot is selected, flattened to
// [B, S, N*H], biased with its slice, then reshaped to heads.

// CHECK-LABEL: func.func @test_mha_packed_qkv
// CHECK-SAME: (%[[QKV:.*]]: !torch.vtensor<[1,3,2,3,4],f32>, %[[B:.*]]: !torch.vtensor<[24],f32>)
func.func @test_mha_packed_qkv(%q: !torch.vtensor<[1,3,2,3,4],f32>, %b: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QS:.*]] = torch.aten.select.int %[[QKV]], %{{.*}}, %{{.*}} : !torch.vtensor<[1,3,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,3,2,4],f32>
  // CHECK: %[[QF:.*]] = torch.aten.flatten.using_ints %[[QS]], {{.*}} -> !torch.vtensor<[1,3,8],f32>
  // CHECK: %[[QB:.*]] = torch.aten.slice.Tensor %[[B]], {{.*}} -> !torch.vtensor<[8],f32>
  // CHECK: torch.aten.add.Tensor %[[QF]], %[[QB]],
  // CHECK: torch.aten.select.int %[[QKV]], {{.*}} -> !torch.vtensor<[1,3,2,4],f32>
  // CHECK: torch.aten.select.int %[[QKV]], {{.*}} -> !torch.vtensor<[1,3,2,4],f32>
  // CHECK: torch.aten.scaled_dot_product_attention {{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %none, %none, %b) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,2,3,4],f32>, !torch.none, !torch.none, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Packed KV [B, L, N, 2, H].

// CHECK-LABEL: func.func @test_mha_packed_kv
// CHECK-SAME: (%[[Q:.*]]: !torch.vtensor<[1,3,8],f32>, %[[KV:.*]]: !torch.vtensor<[1,5,2,2,4],f32>)
func.func @test_mha_packed_kv(%q: !torch.vtensor<[1,3,8],f32>, %kv: !torch.vtensor<[1,5,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.select.int %[[KV]], {{.*}} -> !torch.vtensor<[1,5,2,4],f32>
  // CHECK: torch.aten.select.int %[[KV]], {{.*}} -> !torch.vtensor<[1,5,2,4],f32>
  // CHECK: torch.aten.scaled_dot_product_attention {{.*}} : !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,5,4],f32>, !torch.vtensor<[1,2,5,4],f32>,
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %kv) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// f16: element type is preserved; the causal fill is the lowest finite f16.

// CHECK-LABEL: func.func @test_mha_f16_unidirectional
func.func @test_mha_f16_unidirectional(%q: !torch.vtensor<[1,3,8],f16>, %k: !torch.vtensor<[1,3,8],f16>, %v: !torch.vtensor<[1,3,8],f16>) -> !torch.vtensor<[1,3,8],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.to.dtype {{.*}} -> !torch.vtensor<[3,3],f16>
  // CHECK: %[[LOW:.*]] = torch.constant.float -6.550400e+04
  // CHECK: torch.aten.mul.Scalar %{{.*}}, %[[LOW]]
  // CHECK: torch.aten.scaled_dot_product_attention {{.*}} -> !torch.vtensor<[1,2,3,4],f16>
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,3,8],f16>, !torch.vtensor<[1,3,8],f16>, !torch.vtensor<[1,3,8],f16>) -> !torch.vtensor<[1,3,8],f16>
  return %0 : !torch.vtensor<[1,3,8],f16>
}

// -----

// Dynamic B / S / P with past, present and unidirectional: causal mask
// extents are read at runtime via aten.size.int.

// CHECK-LABEL: func.func @test_mha_dynamic_past_unidirectional
// CHECK-SAME: %[[PK:[a-zA-Z0-9]+]]: !torch.vtensor<[?,2,?,4],f32>, %[[PV:[a-zA-Z0-9]+]]: !torch.vtensor<[?,2,?,4],f32>)
func.func @test_mha_dynamic_past_unidirectional(%q: !torch.vtensor<[?,?,8],f32>, %k: !torch.vtensor<[?,?,8],f32>, %v: !torch.vtensor<[?,?,8],f32>, %pk: !torch.vtensor<[?,2,?,4],f32>, %pv: !torch.vtensor<[?,2,?,4],f32>) -> (!torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[?,2,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QH:.*]] = torch.aten.transpose.int {{.*}} -> !torch.vtensor<[?,2,?,4],f32>
  // CHECK: %[[PRK:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[?,2,?,4],f32>
  // CHECK: %[[PRV:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[?,2,?,4],f32>
  // With dynamic S / L, causal masking checks L == S at runtime.
  // CHECK: %[[S0:.*]] = torch.aten.size.int %[[QH]], {{.*}} -> !torch.int
  // CHECK: %[[L0:.*]] = torch.aten.size.int {{.*}} -> !torch.int
  // CHECK: %[[EQ:.*]] = torch.aten.eq.int %[[S0]], %[[L0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.runtime.assert %[[EQ]], "MultiHeadAttention: unidirectional requires the key/value sequence length to equal the query's"
  // CHECK: %[[S:.*]] = torch.aten.size.int %[[QH]], %{{.*}} : !torch.vtensor<[?,2,?,4],f32>, !torch.int -> !torch.int
  // CHECK: %[[ROWS:.*]] = torch.aten.arange %[[S]], {{.*}} -> !torch.vtensor<[?],si64>
  // CHECK: %[[P:.*]] = torch.aten.size.int %[[PK]], %{{.*}} : !torch.vtensor<[?,2,?,4],f32>, !torch.int -> !torch.int
  // CHECK: torch.aten.add.Scalar %[[ROWS]], %[[P]], {{.*}} -> !torch.vtensor<[?],si64>
  // CHECK: %[[T:.*]] = torch.aten.size.int %[[PRK]], %{{.*}} : !torch.vtensor<[?,2,?,4],f32>, !torch.int -> !torch.int
  // CHECK: torch.aten.arange %[[T]], {{.*}} -> !torch.vtensor<[?],si64>
  // CHECK: %[[MASK:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[1,1,?,?],f32>
  // CHECK: %[[SCALE:.*]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.scaled_dot_product_attention %[[QH]], %[[PRK]], %[[PRV]], %[[MASK]], %{{.*}}, %{{.*}}, %[[SCALE]], %{{.*}} :
  // CHECK: %[[OUT:.*]] = torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[?,?,8],f32>
  // CHECK: return %[[OUT]], %[[PRK]], %[[PRV]]
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64, torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,?,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[?,2,?,4],f32>) -> (!torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[?,2,?,4],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[?,2,?,4],f32>
}

// -----

// Dynamic B / S / L cross-attention with [B, N, L, H] key/value and a
// [B, T] key_padding_mask: no concat, key/value forwarded unchanged.

// CHECK-LABEL: func.func @test_mha_dynamic_cross_bnsh
// CHECK-SAME: %[[K:[a-zA-Z0-9]+]]: !torch.vtensor<[?,2,?,4],f32>, %[[V:[a-zA-Z0-9]+]]: !torch.vtensor<[?,2,?,4],f32>, %[[M:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?],si32>)
func.func @test_mha_dynamic_cross_bnsh(%q: !torch.vtensor<[?,?,8],f32>, %k: !torch.vtensor<[?,2,?,4],f32>, %v: !torch.vtensor<[?,2,?,4],f32>, %m: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK-NOT: torch.aten.cat
  // CHECK: torch.aten.eq.Scalar %[[M]], {{.*}} -> !torch.vtensor<[?,?],i1>
  // CHECK: %[[MASK:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[?,1,1,?],f32>
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %[[K]], %[[V]], %[[MASK]], {{.*}} -> !torch.vtensor<[?,2,?,4],f32>
  // CHECK: torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[?,?,8],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.none, !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?,8],f32>
  return %0 : !torch.vtensor<[?,?,8],f32>
}

// -----

// Dynamic B with a 1D key_padding_mask: taken as [B] key lengths, with the
// mask length checked against B at runtime.

// CHECK-LABEL: func.func @test_mha_dynamic_key_lengths_1d
// CHECK-SAME: %[[Q:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,8],f32>, %{{.*}}, %{{.*}}, %[[M:[a-zA-Z0-9]+]]: !torch.vtensor<[?],si32>)
func.func @test_mha_dynamic_key_lengths_1d(%q: !torch.vtensor<[?,?,8],f32>, %k: !torch.vtensor<[?,?,8],f32>, %v: !torch.vtensor<[?,?,8],f32>, %m: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?,?,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[ML:.*]] = torch.aten.size.int %[[M]], %{{.*}} : !torch.vtensor<[?],si32>, !torch.int -> !torch.int
  // CHECK: %[[B:.*]] = torch.aten.size.int %[[Q]], %{{.*}} : !torch.vtensor<[?,?,8],f32>, !torch.int -> !torch.int
  // CHECK: %[[EQ:.*]] = torch.aten.eq.int %[[ML]], %[[B]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.runtime.assert %[[EQ]], "MultiHeadAttention: a 1D key_padding_mask must hold one key length per batch entry"
  // CHECK: %[[LEN:.*]] = torch.aten.to.dtype %[[M]], {{.*}} -> !torch.vtensor<[?],si64>
  // CHECK: %[[PAD:.*]] = torch.aten.ge.Tensor {{.*}} -> !torch.vtensor<[?,?],i1>
  // CHECK: %[[MASK:.*]] = torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[?,1,1,?],f32>
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK]],
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,?,8],f32>, !torch.none, !torch.vtensor<[?],si32>) -> !torch.vtensor<[?,?,8],f32>
  return %0 : !torch.vtensor<[?,?,8],f32>
}

// -----

// Larger head count (N = 4, H = 2) with 3D Q/K/V.

// CHECK-LABEL: func.func @test_mha_four_heads
func.func @test_mha_four_heads(%q: !torch.vtensor<[2,3,8],f32>, %k: !torch.vtensor<[2,5,8],f32>, %v: !torch.vtensor<[2,5,8],f32>) -> !torch.vtensor<[2,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.unflatten.int {{.*}} -> !torch.vtensor<[2,3,4,2],f32>
  // CHECK: torch.aten.scaled_dot_product_attention {{.*}} : !torch.vtensor<[2,4,3,2],f32>, !torch.vtensor<[2,4,5,2],f32>, !torch.vtensor<[2,4,5,2],f32>, {{.*}} -> !torch.vtensor<[2,4,3,2],f32>
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[2,3,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.vtensor<[2,5,8],f32>) -> !torch.vtensor<[2,3,8],f32>
  return %0 : !torch.vtensor<[2,3,8],f32>
}

// -----

// Result types less static than the inferred ones get a static_info_cast,
// independently per present output.

// CHECK-LABEL: func.func @test_mha_present_type_relaxed
func.func @test_mha_present_type_relaxed(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %pk: !torch.vtensor<[1,2,2,4],f32>, %pv: !torch.vtensor<[1,2,2,4],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[1,2,?,?],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[PRK:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[PRV:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[CK:.*]] = torch.tensor_static_info_cast %[[PRK]] : !torch.vtensor<[1,2,5,4],f32> to !torch.vtensor<[?,2,?,4],f32>
  // CHECK: %[[CV:.*]] = torch.tensor_static_info_cast %[[PRV]] : !torch.vtensor<[1,2,5,4],f32> to !torch.vtensor<[1,2,?,?],f32>
  // CHECK: return %{{.*}}, %[[CK]], %[[CV]]
  %none = torch.constant.none
  %0:3 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[1,2,?,?],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[1,2,?,?],f32>
}

// -----

// past with only the attention output requested: past ++ current K / V are
// still the SDPA operands (T = P + L), unlike the ORT CPU kernel.

// CHECK-LABEL: func.func @test_mha_past_output_only
func.func @test_mha_past_output_only(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %pk: !torch.vtensor<[1,2,2,4],f32>, %pv: !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[PRK:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[PRV:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: torch.aten.scaled_dot_product_attention %{{.*}}, %[[PRK]], %[[PRV]], %{{.*}} -> !torch.vtensor<[1,2,3,4],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}
