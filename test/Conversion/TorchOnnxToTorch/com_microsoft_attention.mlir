// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch --split-input-file | FileCheck %s --implicit-check-not=torch.operator

// Test com.microsoft::Attention decomposition.
//   input   [1, 4, 8]
//   weights [8, 24]  bias [24]
//   num_heads = 2, qkv_hidden_sizes = [8, 8, 8], scale = 0.5
//   output  [1, 4, 8]

// CHECK-LABEL: func.func @test_com_microsoft_attention
func.func @test_com_microsoft_attention(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QKV:.*]] = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: torch.aten.add.Tensor %[[QKV]], %arg2, %{{.*}} : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.unflatten.int
  // CHECK: torch.aten.transpose.int
  // CHECK: torch.aten.matmul
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.matmul
  // CHECK: torch.aten.flatten.using_ints
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention without the optional bias input.

// CHECK-LABEL: func.func @test_com_microsoft_attention_no_bias
func.func @test_com_microsoft_attention_no_bias(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QKV:.*]] = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK-NOT: torch.aten.add.Tensor %[[QKV]]
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints
  %0 = torch.operator "onnx.Attention"(%input, %weights) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with unidirectional (causal) masking enabled. The causal mask is
// built from a strictly-upper-triangular ones tensor scaled by
// mask_filter_value and added to the scores before softmax.

// CHECK-LABEL: func.func @test_com_microsoft_attention_unidirectional
func.func @test_com_microsoft_attention_unidirectional(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.full
  // CHECK: torch.aten.triu
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.softmax.int
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with a 2D raw mask_index (batch, total_seq). The mask is converted
// to an additive bias (keep -> 0, mask -> mask_filter_value), reshaped to
// [batch, 1, 1, total_seq] and added to the scores.

// CHECK-LABEL: func.func @test_com_microsoft_attention_mask_index_2d
func.func @test_com_microsoft_attention_mask_index_2d(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %mask: !torch.vtensor<[1,4],si32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.add.Scalar
  // CHECK: torch.aten.unsqueeze
  // CHECK: torch.aten.unsqueeze
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.softmax.int
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %mask) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[1,4],si32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with an attention_bias input (index 5). mask_index is passed as a
// none value so the bias slot is reached. The bias is added directly to the
// scores prior to softmax.

// CHECK-LABEL: func.func @test_com_microsoft_attention_attention_bias
func.func @test_com_microsoft_attention_attention_bias(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %attn_bias: !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.constant.none
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.softmax.int
  %none = torch.constant.none
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %none, %attn_bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention on f16 tensors. The decomposition preserves the input element type
// throughout, so all intermediate tensors are f16.

// CHECK-LABEL: func.func @test_com_microsoft_attention_f16
func.func @test_com_microsoft_attention_f16(%input: !torch.vtensor<[1,4,8],f16>, %weights: !torch.vtensor<[8,24],f16>, %bias: !torch.vtensor<[24],f16>) -> !torch.vtensor<[1,4,8],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,4,8],f16>, !torch.vtensor<[8,24],f16> -> !torch.vtensor<[1,4,24],f16>
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,2,4],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f16>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f16>, !torch.vtensor<[8,24],f16>, !torch.vtensor<[24],f16>) -> !torch.vtensor<[1,4,8],f16>
  return %0 : !torch.vtensor<[1,4,8],f16>
}

// -----

// Attention with an explicit scale = 0.0. The com.microsoft spec allows a
// user-provided scale of 0.0, so 0.0 must NOT be treated as "unset". This test
// pins that behavior: with num_heads = 1 the head_size is 8, so an "unset"
// scale would produce 1/sqrt(8) = 0.353..., whereas an explicit scale = 0.0
// must be honored and emit a mul by exactly 0.0.

// CHECK-LABEL: func.func @test_com_microsoft_attention_scale_zero
func.func @test_com_microsoft_attention_scale_zero(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[SCALE:.*]] = torch.constant.float 0.000000e+00
  // CHECK: torch.aten.mul.Scalar %{{.*}}, %[[SCALE]]
  // CHECK: torch.aten.softmax.int
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 1 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 0.000000e+00 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with scale omitted. The scale must default to 1/sqrt(head_size).
// With num_heads = 1 and qkv_hidden_sizes = [8, 8, 8] the head_size is 8, so
// the emitted multiplier is 1/sqrt(8) = 0.35355339...

// CHECK-LABEL: func.func @test_com_microsoft_attention_scale_default
func.func @test_com_microsoft_attention_scale_default(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[SCALE:.*]] = torch.constant.float 0.3535533905932737{{.*}}
  // CHECK: torch.aten.mul.Scalar %{{.*}}, %[[SCALE]]
  // CHECK: torch.aten.softmax.int
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 1 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64]} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with a past KV-cache input and a present output.
// past shape: [2, 1, 2, 3, 4]  (2=K/V, batch=1, heads=2, past_seq=3, head=4)
// input shape: [1, 2, 16]  (batch=1, seq=2, input_hidden=16 -> qkv=3*8=24)
// output shape: [1, 2, 8], present shape: [2, 1, 2, 5, 4]
//   (total_seq = past_seq(3) + seq(2) = 5)
//
// The lowering must:
//   - select(past, 0, 0) -> pastK [1, 2, 3, 4]
//   - select(past, 0, 1) -> pastV [1, 2, 3, 4]
//   - cat([pastK, kHeads], dim=2) -> kHeads [1, 2, 5, 4]
//   - cat([pastV, vHeads], dim=2) -> vHeads [1, 2, 5, 4]
//   - scores shape [1, 2, 2, 5]
//   - unsqueeze kHeads/vHeads on dim 0 -> [1, 1, 2, 5, 4]
//   - cat -> present [2, 1, 2, 5, 4]

// CHECK-LABEL: func.func @test_com_microsoft_attention_past_present
func.func @test_com_microsoft_attention_past_present(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.matmul
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.select.int {{.*}} : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: torch.aten.select.int {{.*}} : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: torch.aten.matmul {{.*}} -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[1,2,8],f32>
  // CHECK: torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: torch.aten.unsqueeze {{.*}} -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[2,1,2,5,4],f32>
  // past is operand index 4; pass none for mask_index (index 3).
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}

// -----

// Attention with do_rotary = 1. With head_size = 4 and rotary_embedding_dim = 0
// (default, meaning rotate all 4 elements), the lowering must emit:
//   - arange to build freq indices
//   - exp/cos/sin for the RoPE tables
//   - slice x1/x2 and the rotate-combine (mul, sub, add, cat)
// applied to both Q and K heads before the attention scores.

// CHECK-LABEL: func.func @test_com_microsoft_attention_do_rotary
func.func @test_com_microsoft_attention_do_rotary(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.arange.start_step
  // CHECK: torch.aten.exp
  // CHECK: torch.aten.arange.start_step
  // CHECK: torch.aten.cos
  // CHECK: torch.aten.sin
  // RoPE applied to Q
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.sub.Tensor
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.cat
  // RoPE applied to K (applyRope called a second time)
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.sub.Tensor
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.cat
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>

  return %0 : !torch.vtensor<[1,4,8],f32>
}


// -----

// Attention with do_rotary=1 and a past KV cache. This exercises the combined
// code path: RoPE is applied with position offset pastSeq, and the present
// output must store the POST-RoPE K/V (matching ORT CUDA, which writes present
// from the rotated QKV workspace). Because present stores rotated K, there is
// only one K concat (post-RoPE cat([pastK, kRotated])) and it is reused both
// for attention and for the present output - there is no separate pre-RoPE
// cat. This is the regression guard for the "cached keys never rotated" bug:
// if present were rebuilt from an unrotated kHeadsPreRope, this single-cat
// structure would not hold.
// Model dims: batch=1, seq=2, num_heads=2, head_size=4, past_seq=3.
//   - input [1,2,8], weights [8,24], bias [24], past [2,1,2,3,4]
//   - qkv_hidden_sizes = [8,8,8], rotary_embedding_dim = 0 -> rotate all 4 dims
//   - freqRange arange.start_step with float scalars -> exp -> invFreq
//   - posRange arange.start_step(3.0, 5.0, 1.0) for positions [3,4]
//   - cos/sin computed in f32, cast to elemDtype, applied to Q/K new tokens only
//   - cat([pastK, kRotated], dim=2) -> kHeads [1,2,5,4] for attention and present
//   - output [1,2,8], present [2,1,2,5,4]


// CHECK-LABEL: func.func @test_com_microsoft_attention_rotary_with_past
func.func @test_com_microsoft_attention_rotary_with_past(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // RoPE freq table uses float arange(0.0, 4.0, 2.0) -> exp -> inv_freq
  // CHECK: torch.aten.arange.start_step
  // CHECK: torch.aten.exp
  // Position arange starts at pastSeq=3: arange(3.0, 5.0, 1.0)
  // CHECK: torch.aten.arange.start_step
  // CHECK: torch.aten.cos
  // CHECK: torch.aten.sin
  // RoPE applied to Q new tokens
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.sub.Tensor
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.cat
  // RoPE applied to K new tokens (applyRope called a second time)
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.sub.Tensor
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.cat
  // past K and V extracted and prepended for attention. The post-RoPE K cat
  // SSA value %[[KCAT]] is captured here and must be the SAME value that
  // feeds the present output below (via unsqueeze). This binding is the
  // regression guard: if present were built from an unrotated kHeadsPreRope
  // cat, %[[KCAT]] would not match the unsqueeze operand.
  // CHECK: torch.aten.select.int {{.*}} : !torch.vtensor<[2,1,2,3,4],f32>
  // CHECK: torch.aten.select.int {{.*}} : !torch.vtensor<[2,1,2,3,4],f32>
  // CHECK: %[[KCAT:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[VCAT:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // attention scores, softmax, output
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[1,2,8],f32>
  // present: built from the SAME post-RoPE K/V values used for attention.
  // The unsqueeze that feeds the present cat must consume %[[KCAT]] / %[[VCAT]].
  // CHECK: torch.aten.unsqueeze %[[KCAT]], {{.*}} -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: torch.aten.unsqueeze %[[VCAT]], {{.*}} -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[2,1,2,5,4],f32>

  %none = torch.constant.none
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}

// -----

// Attention with unidirectional=1 and a past KV cache. This exercises the
// causal mask with a non-zero pastSeq so that the mask shape is
// [seq, totalSeqLen] = [2, 5] and the triu diagonal = pastSeq+1 = 4.
//   - batch=1, seq=2, num_heads=2, head_size=4, past_seq=3, total_seq=5
//   - full([2,5]) -> triu(diagonal=4) -> scale by filter_value
//   - scores [1,2,2,5] + causal mask [2,5] broadcast

// CHECK-LABEL: func.func @test_com_microsoft_attention_unidirectional_with_past
func.func @test_com_microsoft_attention_unidirectional_with_past(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // past K and V extracted and prepended (emitted BEFORE the causal mask)
  // CHECK: torch.aten.select.int {{.*}} : !torch.vtensor<[2,1,2,3,4],f32>
  // CHECK: torch.aten.select.int {{.*}} : !torch.vtensor<[2,1,2,3,4],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,5,4],f32>
  // Causal mask must be [2, 5] (not square) with triu diagonal = pastSeq+1 = 4
  // CHECK: torch.aten.full {{.*}} -> !torch.vtensor<[2,5],f32>
  // CHECK: torch.aten.triu
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.add.Tensor
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[1,2,8],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[2,1,2,5,4],f32>
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}

// -----

// Attention where the V projection has a larger head size than Q/K. With
// qkv_hidden_sizes = [8, 8, 16] and num_heads = 2, the Q/K head_size is 4 but
// the V head_size is 8. The scores path stays at head_size 4 while the context
// path carries the v_head_size 8 through to a [1, 4, 16] output. This pins
// that the two head sizes are tracked independently.

// CHECK-LABEL: func.func @test_com_microsoft_attention_vheadsize
func.func @test_com_microsoft_attention_vheadsize(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,32],f32>, %bias: !torch.vtensor<[32],f32>) -> !torch.vtensor<[1,4,16],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QKV:.*]] = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,32],f32> -> !torch.vtensor<[1,4,32],f32>
  // CHECK: %[[QKVB:.*]] = torch.aten.add.Tensor %[[QKV]], %arg2, %{{.*}} : !torch.vtensor<[1,4,32],f32>, !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,4,32],f32>
  // Q and K slices are 8 wide (head_size 4), V slice is 16 wide (v_head_size 8).
  // CHECK: torch.aten.slice.Tensor %[[QKVB]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: torch.aten.slice.Tensor %[[QKVB]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: torch.aten.slice.Tensor %[[QKVB]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,16],f32>
  // V heads carry v_head_size 8.
  // CHECK: torch.aten.unflatten.int %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,16],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,8],f32>
  // CHECK: %[[VHEADS:.*]] = torch.aten.transpose.int %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,2,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,8],f32>
  // Scores stay at head_size 4.
  // CHECK: torch.aten.matmul %{{.*}}, %{{.*}} : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: torch.aten.softmax.int
  // Context matmul mixes probs with the wide V heads -> v_head_size 8.
  // CHECK: torch.aten.matmul %{{.*}}, %[[VHEADS]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,8],f32> -> !torch.vtensor<[1,2,4,8],f32>
  // CHECK: torch.aten.flatten.using_ints %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,2,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,16],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 16 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,32],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[1,4,16],f32>
  return %0 : !torch.vtensor<[1,4,16],f32>
}

// -----

// Attention with qkv_hidden_sizes omitted. The weights hidden dimension (24)
// must be split evenly into three projections of 8, giving head_size 4 for
// num_heads = 2. This pins the default equal-split path.

// CHECK-LABEL: func.func @test_com_microsoft_attention_default_split
func.func @test_com_microsoft_attention_default_split(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QKV:.*]] = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[QKVB:.*]] = torch.aten.add.Tensor %[[QKV]], %arg2, %{{.*}} : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // All three projections are 8 wide (even split of 24).
  // CHECK: torch.aten.slice.Tensor %[[QKVB]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: torch.aten.slice.Tensor %[[QKVB]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: torch.aten.slice.Tensor %[[QKVB]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: torch.aten.matmul %{{.*}}, %{{.*}} : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with a 3D mask_index [batch, seq, total_seq]. The mask is converted
// to an additive bias (keep -> 0, mask -> mask_filter_value) and unsqueezed on
// the head dim to [batch, 1, seq, total_seq] before being added to the scores.

// CHECK-LABEL: func.func @test_com_microsoft_attention_mask_index_3d
func.func @test_com_microsoft_attention_mask_index_3d(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %mask: !torch.vtensor<[1,4,4],si32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[SCORES:.*]] = torch.aten.mul.Scalar %{{.*}}, %{{.*}} : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // Mask is cast to f32, scaled, offset, then unsqueezed on the head dim.
  // CHECK: %[[MASKF:.*]] = torch.aten.to.dtype %arg3, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,4],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,4,4],f32>
  // CHECK: torch.aten.mul.Scalar %[[MASKF]], %{{.*}} : !torch.vtensor<[1,4,4],f32>, !torch.float -> !torch.vtensor<[1,4,4],f32>
  // CHECK: torch.aten.add.Scalar %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,4],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,4,4],f32>
  // CHECK: %[[MASK4D:.*]] = torch.aten.unsqueeze %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,4],f32>, !torch.int -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: torch.aten.add.Tensor %[[SCORES]], %[[MASK4D]], %{{.*}} : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,1,4,4],f32>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %mask) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[1,4,4],si32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Dynamic batch and sequence dimensions are supported: only the hidden
// dimension (dim 2) must be static. The batch (dim 0) and sequence (dim 1)
// sizes flow through as dynamic (`?`) and the lowering still legalizes.

// CHECK-LABEL: func.func @test_com_microsoft_attention_dynamic_batch_seq
func.func @test_com_microsoft_attention_dynamic_batch_seq(%input: !torch.vtensor<[?,?,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[?,?,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[QKV:.*]] = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[?,?,24],f32>
  // CHECK: torch.aten.add.Tensor %[[QKV]], %arg2, %{{.*}} : !torch.vtensor<[?,?,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[?,?,24],f32>
  // CHECK: torch.aten.slice.Tensor
  // CHECK: torch.aten.unflatten.int
  // CHECK: torch.aten.transpose.int
  // CHECK: torch.aten.matmul
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[?,?,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[?,?,8],f32>
  return %0 : !torch.vtensor<[?,?,8],f32>
}

// -----

// Attention with do_rotary = 1 and a partial rotary_embedding_dim = 2 that is
// smaller than head_size = 4. Only the first 2 of each head's 4 elements are
// rotated; the unrotated tail (elements [2, 4)) must be sliced off and
// concatenated back after the rotation. This pins the `rotDim < head_size`
// tail-concat branch of applyRope, which the full-rotation rotary tests
// (rotary_embedding_dim = 0) never exercise.

// CHECK-LABEL: func.func @test_com_microsoft_attention_partial_rotary
func.func @test_com_microsoft_attention_partial_rotary(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: torch.aten.arange.start_step
  // CHECK: torch.aten.exp
  // CHECK: torch.aten.cos
  // CHECK: torch.aten.sin
  // Rotated half (rotDim = 2) produced by the rotate-combine, then the
  // unrotated tail [2, 4) is sliced and concatenated back to width 4.
  // CHECK: %[[ROT:.*]] = torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[TAIL:.*]] = torch.aten.slice.Tensor {{.*}} -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: torch.aten.cat {{.*}} -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: torch.aten.softmax.int
  // CHECK: torch.aten.flatten.using_ints
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.do_rotary = 1 : si64, torch.onnx.rotary_embedding_dim = 2 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}


