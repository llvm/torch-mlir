// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// The reasons passed to notifyMatchFailure are not emitted as diagnostics, so
// each case can only check the generic legalization failure. To keep that
// check meaningful, every case is a valid input layout with exactly one
// input changed, so the intended check is the only one that can fail.

// Only one of key/value is absent.
func.func @gqa_key_none_value_present(%q: !torch.vtensor<[1,1,16],f32>, %v: !torch.vtensor<[1,1,16],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%q, %none, %v, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,1,16],f32>, !torch.none, !torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

func.func @gqa_key_present_value_none(%q: !torch.vtensor<[1,1,16],f32>, %k: !torch.vtensor<[1,1,16],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%q, %k, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,1,16],f32>, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,16],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Missing query.
func.func @gqa_query_none(%pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%none, %none, %none, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Missing past_key.
func.func @gqa_past_key_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %none, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Missing past_value.
func.func @gqa_past_value_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %none, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.none, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Missing seqlens_k.
func.func @gqa_seqlens_k_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %none, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.none, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// do_rotary=1 but cos_cache is absent.
func.func @gqa_rotary_cos_cache_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %sin: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %sin) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// do_rotary=1 but sin_cache is absent.
func.func @gqa_rotary_sin_cache_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %cos: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %cos, %none) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.vtensor<[2,4],f32>, !torch.none) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Removed shortened packed layout without rotary (5 operands).
func.func @gqa_short_packed_no_rotary(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %pk, %pv, %sk, %tsl) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Removed shortened packed layout with rotary (7 operands, do_rotary=1).
func.func @gqa_short_packed_rotary(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %cos: !torch.vtensor<[2,4],f32>, %sin: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %pk, %pv, %sk, %tsl, %cos, %sin) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}

// -----

// Non-empty trailing optional input (e.g. position_ids) is not supported.
func.func @gqa_trailing_non_none(%qkv: !torch.vtensor<[1,1,64],f32>, %pk: !torch.vtensor<[1,2,0,8],f32>, %pv: !torch.vtensor<[1,2,0,8],f32>, %sk: !torch.vtensor<[1],si32>, %tsl: !torch.vtensor<[1],si32>, %extra: !torch.vtensor<[1,1],si64>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.GroupQueryAttention"(%qkv, %none, %none, %pk, %pv, %sk, %tsl, %none, %none, %extra) {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64} : (!torch.vtensor<[1,1,64],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1,2,0,8],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.none, !torch.none, !torch.vtensor<[1,1],si64>) -> (!torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[1,1,32],f32>, !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,1,8],f32>
}
