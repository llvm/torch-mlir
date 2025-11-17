// RUN: torch-mlir-opt %s -torch-decompose-complex-ops -convert-torch-to-tosa -split-input-file | FileCheck %s

// Verify that lowering a single TransformerEncoderLayer produces the expected
// TOSA building blocks (QKV projection, attention, feed-forward, and layer norm).
// The operands are intentionally small so we can keep static shapes throughout.
module {
  // CHECK-LABEL: func.func @transformer(
  // CHECK: %[[QKV:.*]] = tosa.matmul %{{.*}} : (tensor<1x4x8xf32>, tensor<1x8x24xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x24xf32>
  // CHECK: %[[QKVRESHAPE:.*]] = tosa.reshape %{{.*}} : (tensor<1x4x24xf32>, !tosa.shape<5>) -> tensor<1x4x3x2x4xf32>
  // CHECK: %[[SCORES:.*]] = tosa.matmul %{{.*}} : (tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x4xf32>
  // CHECK: %[[SCALE:.*]] = tosa.mul %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x2x4x4xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x2x4x4xf32>
  // CHECK: tosa.sub %{{.*}} : (tensor<1x4x16xf32>, tensor<1x1x1xf32>) -> tensor<1x4x16xf32>
  // CHECK: tosa.mul %{{.*}} : (tensor<1x4x16xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x4x16xf32>
  // CHECK: tosa.erf %{{.*}} : (tensor<1x4x16xf32>) -> tensor<1x4x16xf32>
  // CHECK: %[[FFN:.*]] = tosa.matmul %{{.*}} : (tensor<1x4x16xf32>, tensor<1x16x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x8xf32>
  // CHECK: %[[NORMINV:.*]] = tosa.rsqrt %{{.*}} : (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>
  // CHECK: return %{{.*}} : !torch.vtensor<[1,4,8],f32>
  func.func @transformer(
      %arg0: !torch.vtensor<[1,4,8],f32>,
      %qkv_weight: !torch.vtensor<[24,8],f32>,
      %qkv_bias: !torch.vtensor<[24],f32>,
      %proj_weight: !torch.vtensor<[8,8],f32>,
      %proj_bias: !torch.vtensor<[8],f32>,
      %norm1_weight: !torch.vtensor<[8],f32>,
      %norm1_bias: !torch.vtensor<[8],f32>,
      %norm2_weight: !torch.vtensor<[8],f32>,
      %norm2_bias: !torch.vtensor<[8],f32>,
      %ffn1_weight: !torch.vtensor<[16,8],f32>,
      %ffn1_bias: !torch.vtensor<[16],f32>,
      %ffn2_weight: !torch.vtensor<[8,16],f32>,
      %ffn2_bias: !torch.vtensor<[8],f32>) -> !torch.vtensor<[1,4,8],f32> {
    %embed_dim = torch.constant.int 8
    %num_heads = torch.constant.int 2
    %use_gelu = torch.constant.bool true
    %norm_first = torch.constant.bool false
    %eps = torch.constant.float 1.000000e-05
    %none = torch.constant.none
    %result = torch.operator "torch.aten._transformer_encoder_layer_fwd.default"(
        %arg0, %embed_dim, %num_heads, %qkv_weight, %qkv_bias, %proj_weight,
        %proj_bias, %use_gelu, %norm_first, %eps, %norm1_weight, %norm1_bias,
        %norm2_weight, %norm2_bias, %ffn1_weight, %ffn1_bias, %ffn2_weight,
        %ffn2_bias, %none, %none
      ) : (!torch.vtensor<[1,4,8],f32>, !torch.int, !torch.int,
           !torch.vtensor<[24,8],f32>, !torch.vtensor<[24],f32>,
           !torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>, !torch.bool,
           !torch.bool, !torch.float, !torch.vtensor<[8],f32>,
           !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32>,
           !torch.vtensor<[8],f32>, !torch.vtensor<[16,8],f32>,
           !torch.vtensor<[16],f32>, !torch.vtensor<[8,16],f32>,
           !torch.vtensor<[8],f32>, !torch.none, !torch.none)
        -> !torch.vtensor<[1,4,8],f32>
    return %result : !torch.vtensor<[1,4,8],f32>
  }
}
