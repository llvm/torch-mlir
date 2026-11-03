// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch -split-input-file -verify-diagnostics

// com.microsoft::MultiHeadAttention diagnostics: each case hits a distinct
// rejection path and must leave the op unconverted (clean match failure).

// num_heads is required.
func.func @test_mha_num_heads_missing(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// num_heads must be strictly positive.
func.func @test_mha_num_heads_zero(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 0 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// num_heads must be an integer attribute.
func.func @test_mha_attribute_bind(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2.0 : f32} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// query (input 0) is required.
func.func @test_mha_missing_query(%k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%none, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.none, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// past_sequence_length (input 8) is only meaningful with buffer sharing.
func.func @test_mha_past_sequence_length(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %pk: !torch.vtensor<[1,2,2,4],f32>, %pv: !torch.vtensor<[1,2,2,4],f32>, %psl: !torch.vtensor<[1],si32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv, %psl) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1],si32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// cache_indirection (input 9) is a beam-search feature.
func.func @test_mha_cache_indirection(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %ci: !torch.vtensor<[1,1,3],si32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %none, %none, %none, %ci) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,1,3],si32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// past_key without past_value.
func.func @test_mha_past_key_only(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %pk: !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// The op has at most 10 inputs.
func.func @test_mha_too_many_inputs(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %x: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %none, %none, %none, %none, %x) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// The attention output (result 0) is required.
func.func @test_mha_no_results(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> ()
  return
}

// -----

// The fourth output (qk) is not supported.
func.func @test_mha_qk_output(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,2,3,3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0:4 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,3,3],f32>)
  return %0#3 : !torch.vtensor<[1,2,3,3],f32>
}

// -----

// Results must be value tensors.
func.func @test_mha_result_not_vtensor(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.tensor attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.tensor
  return %0 : !torch.tensor
}

// -----

// T must be a floating-point type.
func.func @test_mha_integer_query(%q: !torch.vtensor<[1,3,8],si32>, %k: !torch.vtensor<[1,3,8],si32>, %v: !torch.vtensor<[1,3,8],si32>) -> !torch.vtensor<[1,3,8],si32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],si32>, !torch.vtensor<[1,3,8],si32>, !torch.vtensor<[1,3,8],si32>) -> !torch.vtensor<[1,3,8],si32>
  return %0 : !torch.vtensor<[1,3,8],si32>
}

// -----

// All T-typed inputs share one element type.
func.func @test_mha_dtype_mismatch(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f16>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f16>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// A 4D query is not a spec layout.
func.func @test_mha_query_rank4(%q: !torch.vtensor<[1,2,3,4],f32>, %k: !torch.vtensor<[1,2,3,4],f32>, %v: !torch.vtensor<[1,2,3,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,3,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Packed QKV excludes separate key / value.
func.func @test_mha_packed_qkv_with_key(%q: !torch.vtensor<[1,3,2,3,4],f32>, %k: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,2,3,4],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Packed QKV must have 3 slots along dim 3.
func.func @test_mha_packed_qkv_slots(%q: !torch.vtensor<[1,3,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// The query hidden size must be divisible by num_heads.
func.func @test_mha_hidden_not_divisible(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 3 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// The query hidden size must be static to derive the head size.
func.func @test_mha_dynamic_hidden(%q: !torch.vtensor<[1,3,?],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,?],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Packed KV head size must match the query head size.
func.func @test_mha_packed_kv_head_size(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,2,2,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,2,2,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// A 3D key must have the query hidden size.
func.func @test_mha_key_hidden_mismatch(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,12],f32>, %v: !torch.vtensor<[1,5,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,12],f32>, !torch.vtensor<[1,5,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// A 3D value hidden size must be divisible by num_heads.
func.func @test_mha_value_hidden_not_divisible(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,8],f32>, %v: !torch.vtensor<[1,5,7],f32>) -> !torch.vtensor<[1,3,7],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[1,5,7],f32>) -> !torch.vtensor<[1,3,7],f32>
  return %0 : !torch.vtensor<[1,3,7],f32>
}

// -----

// 3D key and value must share the sequence length.
func.func @test_mha_kv_seq_mismatch(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,8],f32>, %v: !torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// A 4D key must have num_heads along dim 1.
func.func @test_mha_bnsh_num_heads_mismatch(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,4,5,2],f32>, %v: !torch.vtensor<[1,4,5,2],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,4,5,2],f32>, !torch.vtensor<[1,4,5,2],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Mixed 3D key / 4D value is not a spec layout.
func.func @test_mha_mixed_kv_layout(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,8],f32>, %v: !torch.vtensor<[1,2,5,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[1,2,5,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// A 3D query requires key and value (packed QKV is 5D).
func.func @test_mha_missing_key_value(%q: !torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// bias must be [2 * N * H + N * Hv].
func.func @test_mha_bias_length(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %b: !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %b) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// past_key head size must match the key head size.
func.func @test_mha_past_head_size(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %pk: !torch.vtensor<[1,2,2,8],f32>, %pv: !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[1,2,2,8],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// unidirectional with kv_seq != seq: ORT leaves the last query row unmasked,
// which is not reproduced.
func.func @test_mha_unidirectional_cross_length(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,5,8],f32>, %v: !torch.vtensor<[1,5,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,5,8],f32>, !torch.vtensor<[1,5,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// The [3 * B + 2] 1D key_padding_mask encoding is not supported.
func.func @test_mha_mask_3b_plus_2(%q: !torch.vtensor<[2,3,8],f32>, %k: !torch.vtensor<[2,5,8],f32>, %v: !torch.vtensor<[2,5,8],f32>, %m: !torch.vtensor<[8],si32>) -> !torch.vtensor<[2,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[2,3,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.none, !torch.vtensor<[8],si32>) -> !torch.vtensor<[2,3,8],f32>
  return %0 : !torch.vtensor<[2,3,8],f32>
}

// -----

// With a dynamic query batch, a 1D key_padding_mask of length 3 conflicts
// with the static key batch (B = 2).
func.func @test_mha_key_lengths_batch_conflict(%q: !torch.vtensor<[?,3,8],f32>, %k: !torch.vtensor<[2,5,8],f32>, %v: !torch.vtensor<[2,5,8],f32>, %m: !torch.vtensor<[3],si32>) -> !torch.vtensor<[?,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[?,3,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.vtensor<[2,5,8],f32>, !torch.none, !torch.vtensor<[3],si32>) -> !torch.vtensor<[?,3,8],f32>
  return %0 : !torch.vtensor<[?,3,8],f32>
}

// -----

// A 2D key_padding_mask must span the total sequence length (P + L).
func.func @test_mha_mask_total_seq_mismatch(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %m: !torch.vtensor<[1,3],si32>, %pk: !torch.vtensor<[1,2,2,4],f32>, %pv: !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.vtensor<[1,3],si32>, !torch.none, !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// A 4D key_padding_mask is not a spec layout.
func.func @test_mha_mask_rank4(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %m: !torch.vtensor<[1,1,3,3],si32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.vtensor<[1,1,3,3],si32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// attention_bias must be 4D.
func.func @test_mha_attention_bias_rank3(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %ab: !torch.vtensor<[2,3,3],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %ab) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.vtensor<[2,3,3],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// attention_bias must be [B|1, N|1, S, T]; here T = 4 != 3.
func.func @test_mha_attention_bias_shape(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %ab: !torch.vtensor<[1,2,3,4],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %ab) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,3,4],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// key_padding_mask must be an integer tensor.
func.func @test_mha_mask_float(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[1,3,8],f32>, %v: !torch.vtensor<[1,3,8],f32>, %m: !torch.vtensor<[1,3],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.vtensor<[1,3,8],f32>, !torch.none, !torch.vtensor<[1,3],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// Batch size must agree across query, key and value.
func.func @test_mha_batch_mismatch(%q: !torch.vtensor<[1,3,8],f32>, %k: !torch.vtensor<[2,3,8],f32>, %v: !torch.vtensor<[2,3,8],f32>) -> !torch.vtensor<[1,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,3,8],f32>, !torch.vtensor<[2,3,8],f32>, !torch.vtensor<[2,3,8],f32>) -> !torch.vtensor<[1,3,8],f32>
  return %0 : !torch.vtensor<[1,3,8],f32>
}

// -----

// With a dynamic query batch, key (B = 2) and past_key / past_value (B = 3)
// are each compatible with the query but conflict with one another.
func.func @test_mha_batch_conflict_dynamic_query(%q: !torch.vtensor<[?,3,8],f32>, %k: !torch.vtensor<[2,3,8],f32>, %v: !torch.vtensor<[2,3,8],f32>, %pk: !torch.vtensor<[3,2,2,4],f32>, %pv: !torch.vtensor<[3,2,2,4],f32>) -> !torch.vtensor<[?,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %none, %none, %pk, %pv) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[?,3,8],f32>, !torch.vtensor<[2,3,8],f32>, !torch.vtensor<[2,3,8],f32>, !torch.none, !torch.none, !torch.none, !torch.vtensor<[3,2,2,4],f32>, !torch.vtensor<[3,2,2,4],f32>) -> !torch.vtensor<[?,3,8],f32>
  return %0 : !torch.vtensor<[?,3,8],f32>
}

// -----

// With a dynamic batch, key_padding_mask (B = 2) and attention_bias (B = 3)
// are each compatible with the query but cannot be broadcast together.
func.func @test_mha_mask_bias_batch_conflict(%q: !torch.vtensor<[?,3,8],f32>, %k: !torch.vtensor<[?,3,8],f32>, %v: !torch.vtensor<[?,3,8],f32>, %m: !torch.vtensor<[2,3],si32>, %ab: !torch.vtensor<[3,2,3,3],f32>) -> !torch.vtensor<[?,3,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.MultiHeadAttention"(%q, %k, %v, %none, %m, %ab) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[?,3,8],f32>, !torch.vtensor<[?,3,8],f32>, !torch.vtensor<[?,3,8],f32>, !torch.none, !torch.vtensor<[2,3],si32>, !torch.vtensor<[3,2,3,3],f32>) -> !torch.vtensor<[?,3,8],f32>
  return %0 : !torch.vtensor<[?,3,8],f32>
}
