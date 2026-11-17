// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// The reasons passed to notifyMatchFailure are not emitted as diagnostics, so
// each case can only check the generic legalization failure. To keep that
// check meaningful, every case is a valid input layout with exactly one
// input changed, so the intended check is the only one that can fail.

// num_heads is not a multiple of kv_num_heads.
func.func @group_query_attention_num_heads_not_multiple(%qkv: !torch.vtensor<[1,1,56],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 3 : si64} : (!torch.vtensor<[1,1,56],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// causal is neither 0 nor 1.
func.func @group_query_attention_causal_2(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.causal = 2 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A local window of zero keys.
func.func @group_query_attention_local_window_size_zero(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.local_window_size = 0 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A local window without causal attention.
func.func @group_query_attention_local_window_size_non_causal(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.causal = 0 : si64, torch.onnx.local_window_size = 2 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// qk_output is neither 0, 1 nor 2.
func.func @group_query_attention_qk_output_3(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,4,1,1],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:4 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.qk_output = 3 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,4,1,1],f32>)
  return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,4,1,1],f32>
}

// -----

// Only the K cache is quantized.
func.func @group_query_attention_quant_types_differ(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.k_quant_type = "PER_TENSOR", torch.onnx.kv_cache_bit_width = 8 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// kv_cache_bit_width without a quantized cache.
func.func @group_query_attention_kv_cache_bit_width_without_quantization(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.kv_cache_bit_width = 8 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// sliding_window_cache without a local window.
func.func @group_query_attention_sliding_window_cache_without_window(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.sliding_window_cache = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Fewer than the 7 required input positions.
func.func @group_query_attention_too_few_inputs(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// query is absent.
func.func @group_query_attention_query_none(%pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%none, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Only one of key and value is absent.
func.func @group_query_attention_key_none_value_present(%q: !torch.vtensor<[1,1,32],f32>, %v: !torch.vtensor<[1,1,16],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%q, %none, %v, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,32],f32>, !torch.none, !torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Only one of past_key and past_value is absent.
func.func @group_query_attention_past_key_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %none, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// do_rotary=1 without cos_cache.
func.func @group_query_attention_rotary_cos_cache_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %sin: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %sin) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A seqlens_k of unknown rank.
func.func @group_query_attention_seqlens_k_unranked(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<*,si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<*,si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// The fused per-head Q/K norm, which the onnxruntime CPU kernel rejects too.
func.func @group_query_attention_q_norm_weight(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %w: !torch.vtensor<[8],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %none, %none, %none, %none, %none, %none, %w, %w) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A quantized cache without k_scale and v_scale.
func.func @group_query_attention_quantized_without_scales(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],si8>, %pv: !torch.vtensor<[1,2,0,8],si8>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],si8>, !torch.vtensor<[1,2,1,8],si8>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.k_quant_type = "PER_TENSOR", torch.onnx.v_quant_type = "PER_TENSOR", torch.onnx.kv_cache_bit_width = 8 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],si8>, !torch.vtensor<[1,2,0,8],si8>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],si8>, !torch.vtensor<[1,2,1,8],si8>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],si8>, !torch.vtensor<[1,2,1,8],si8>
}

// -----

// A present_key and present_value that are not tensors.
func.func @group_query_attention_result_not_tensor(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.int, !torch.int) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.int, !torch.int)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.int, !torch.int
}

// -----

// An output of 16 values per token instead of num_heads * head_size = 32.
func.func @group_query_attention_output_wrong_hidden(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// An output_qk of rank 3.
func.func @group_query_attention_output_qk_rank_3(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,4,1],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:4 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.qk_output = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,4,1],f32>)
  return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,4,1],f32>
}

// -----

// present_key without present_value.
func.func @group_query_attention_present_key_only(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.none)
  return %0#0, %0#1 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// qk_output without an output_qk output.
func.func @group_query_attention_qk_output_without_output_qk(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.qk_output = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A quantized cache without present outputs.
func.func @group_query_attention_quantized_without_presents(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],si8>, %pv: !torch.vtensor<[1,2,0,8],si8>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %s: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,1,32],f32> attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %none, %none, %none, %none, %s, %s) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.k_quant_type = "PER_TENSOR", torch.onnx.v_quant_type = "PER_TENSOR", torch.onnx.kv_cache_bit_width = 8 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],si8>, !torch.vtensor<[1,2,0,8],si8>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,1,32],f32>
  return %0 : !torch.vtensor<[1,1,32],f32>
}

// -----

// sliding_window_cache without a past cache.
func.func @group_query_attention_sliding_window_cache_without_past(%qkv: !torch.vtensor<[1,1,64],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %none, %none, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.local_window_size = 2 : si64, torch.onnx.sliding_window_cache = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A query with a dynamic hidden size.
func.func @group_query_attention_query_dynamic_hidden(%qkv: !torch.vtensor<[1,1,?],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,?],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A packed query whose hidden size is not a multiple of its 8 heads.
func.func @group_query_attention_query_hidden_not_multiple(%qkv: !torch.vtensor<[1,1,60],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,60],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A key of 12 values per token instead of kv_num_heads * head_size = 16.
func.func @group_query_attention_key_hidden_mismatch(%q: !torch.vtensor<[1,1,32],f32>, %k: !torch.vtensor<[1,1,12],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%q, %k, %k, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,1,12],f32>, !torch.vtensor<[1,1,12],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A key with more tokens than the query.
func.func @group_query_attention_key_sequence_mismatch(%q: !torch.vtensor<[1,1,32],f32>, %k: !torch.vtensor<[1,2,16],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%q, %k, %k, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,16],f32>, !torch.vtensor<[1,2,16],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// An empty key without a past cache to share.
func.func @group_query_attention_empty_key_without_past(%q: !torch.vtensor<[1,1,32],f32>, %k: !torch.vtensor<[1,0,16],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%q, %k, %k, %none, %none, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,0,16],f32>, !torch.vtensor<[1,0,16],f32>, !torch.none, !torch.none, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// An int4 cache with an odd head size, which does not pack into bytes.
func.func @group_query_attention_int4_odd_head_size(%qkv: !torch.vtensor<[1,1,15],f32>, %pk: !torch.vtensor<[1,1,0,1],ui8>, %pv: !torch.vtensor<[1,1,0,1],ui8>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %s: !torch.vtensor<[1],f32>) -> (!torch.vtensor<[1,1,9],f32>, !torch.vtensor<[1,1,1,1],ui8>, !torch.vtensor<[1,1,1,1],ui8>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %none, %none, %none, %none, %s, %s) {torch.onnx.kv_num_heads = 1 : si64, torch.onnx.num_heads = 3 : si64, torch.onnx.k_quant_type = "PER_TENSOR", torch.onnx.v_quant_type = "PER_TENSOR", torch.onnx.kv_cache_bit_width = 4 : si64} : (!torch.vtensor<[1,1,15],f32>, !torch.none, !torch.none, !torch.vtensor<[1,1,0,1],ui8>, !torch.vtensor<[1,1,0,1],ui8>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> (!torch.vtensor<[1,1,9],f32>, !torch.vtensor<[1,1,1,1],ui8>, !torch.vtensor<[1,1,1,1],ui8>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,9],f32>, !torch.vtensor<[1,1,1,1],ui8>, !torch.vtensor<[1,1,1,1],ui8>
}

// -----

// An f16 past cache for an f32 query.
func.func @group_query_attention_past_key_wrong_dtype(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f16>, %pv: !torch.vtensor<[1,2,0,8],f16>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f16>, !torch.vtensor<[1,2,0,8],f16>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A PER_TENSOR k_scale with two elements.
func.func @group_query_attention_k_scale_wrong_size(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],si8>, %pv: !torch.vtensor<[1,2,0,8],si8>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %ks: !torch.vtensor<[2],f32>, %vs: !torch.vtensor<[1],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],si8>, !torch.vtensor<[1,2,1,8],si8>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %none, %none, %none, %none, %ks, %vs) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.k_quant_type = "PER_TENSOR", torch.onnx.v_quant_type = "PER_TENSOR", torch.onnx.kv_cache_bit_width = 8 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],si8>, !torch.vtensor<[1,2,0,8],si8>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[2],f32>, !torch.vtensor<[1],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],si8>, !torch.vtensor<[1,2,1,8],si8>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],si8>, !torch.vtensor<[1,2,1,8],si8>
}

// -----

// An attention_bias with 2 heads, neither 1 nor num_heads.
func.func @group_query_attention_attention_bias_wrong_heads(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %bias: !torch.vtensor<[1,2,1,1],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %none, %none, %bias) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,1,1],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// A head_sink with 3 values for 4 heads.
func.func @group_query_attention_head_sink_wrong_size(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %sink: !torch.vtensor<[3],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %none, %none, %none, %sink) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[3],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// position_ids of rank 1.
func.func @group_query_attention_position_ids_rank_1(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %cos: !torch.vtensor<[2,4],f32>, %ids: !torch.vtensor<[1],si64>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %cos, %cos, %ids) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}
