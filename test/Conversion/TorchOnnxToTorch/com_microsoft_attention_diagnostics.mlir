// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch -split-input-file -verify-diagnostics

// com.microsoft::Attention diagnostics: these cases must produce a clean
// match failure (the op is left unconverted and torch-mlir emits a
// legalization error) rather than crashing or silently mis-lowering.

// A rank-1 mask_index (one of the right-side-padding encodings ORT emits) is
// not supported by this lowering and must be rejected.
func.func @test_attention_mask_index_1d(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %mask: !torch.vtensor<[1],si32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %mask) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[1],si32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// A 2D mask_index whose last (key) dimension does not equal the total
// sequence length must be rejected: it cannot broadcast against the scores.
func.func @test_attention_mask_index_totalseq_mismatch(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %mask: !torch.vtensor<[1,5],si32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %mask) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[1,5],si32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// past_present_share_buffer is not supported and must be rejected.
func.func @test_attention_share_buffer_unsupported(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.past_present_share_buffer = 1 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// qkv_hidden_sizes that do not sum to the weights hidden dimension must be
// rejected.
func.func @test_attention_qkv_hidden_sum_mismatch(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 4 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// A dynamic weights last dimension (qkv_hidden_sum) must be rejected: the
// per-projection slice bounds depend on a statically known total hidden size.
func.func @test_attention_dynamic_weights_hidden(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,?],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,?],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// A dynamic past sequence-length dimension (past dim 3) must be rejected: the
// total sequence length and the present KV-cache shape are computed statically.
func.func @test_attention_dynamic_past_seq(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,?,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,?,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,?,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,?,4],f32>
}

// -----

// Fewer than two operands (only `input`) must be rejected: `weights` is
// required to form the QKV projection.
func.func @test_attention_too_few_operands(%input: !torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// The optional `past_sequence_length` tensor input (operand index 6) is only
// meaningful with past_present_share_buffer, which is unsupported, so its
// presence must be rejected.
func.func @test_attention_past_seq_length_operand(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>, %pastseqlen: !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past, %none, %pastseqlen) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>, !torch.none, !torch.vtensor<[1],si32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}

// -----

// num_heads = 0 is invalid (head size would divide by zero) and must be
// rejected.
func.func @test_attention_num_heads_zero(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 0 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// A missing num_heads attribute (required) must be rejected.
func.func @test_attention_num_heads_missing(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// The input tensor must be rank 3 (batch, seq, hidden). A rank-2 input must
// be rejected.
func.func @test_attention_input_rank(%input: !torch.vtensor<[4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

// The weights tensor must be rank 2 (input_hidden, qkv_hidden_sum). A rank-3
// weights tensor must be rejected.
func.func @test_attention_weights_rank(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[1,8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// A default-split (qkv_hidden_sizes omitted) weights hidden dimension that is
// not divisible by 3 cannot be split evenly into Q/K/V and must be rejected.
func.func @test_attention_default_split_not_div3(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,25],f32>, %bias: !torch.vtensor<[25],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,25],f32>, !torch.vtensor<[25],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// qkv_hidden_sizes must have exactly three entries (Q, K, V). A different
// length must be rejected.
func.func @test_attention_qkv_hidden_sizes_wrong_length(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [12 : si64, 12 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// The Q and K projection hidden sizes must be equal (the scores matmul
// requires matching head sizes). Unequal Q/K sizes must be rejected.
func.func @test_attention_q_k_hidden_mismatch(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [6 : si64, 10 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Each projection hidden size must be divisible by num_heads. A Q/K/V hidden
// size that is not a multiple of num_heads must be rejected.
func.func @test_attention_hidden_not_div_num_heads(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 3 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// The `past` KV-cache tensor must be rank 5 (2, batch, heads, past_seq, head).
// A rank-4 past tensor must be rejected.
func.func @test_attention_past_rank(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  %none = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}
