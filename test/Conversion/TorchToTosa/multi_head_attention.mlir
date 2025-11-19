// RUN: torch-mlir-opt %s -torch-decompose-complex-ops -convert-torch-to-tosa -split-input-file | FileCheck %s

// Checks that scaled dot product attention (single-head configuration) lowers
// through the decomposition pass into the expected TOSA matmul + softmax flow.
module {
  // CHECK-LABEL: func.func @scaled_dot_product_attention(
  // CHECK: %[[KEY_T:.*]] = tosa.transpose %{{.*}} {perms = array<i32: 0, 2, 1>} : (tensor<1x4x8xf32>) -> tensor<1x8x4xf32>
  // CHECK: %[[KEY_VIEW:.*]] = tosa.reshape %[[KEY_T]], %{{.*}} : (tensor<1x8x4xf32>, !tosa.shape<3>) -> tensor<1x8x4xf32>
  // CHECK: %[[QK:.*]] = tosa.matmul %{{.*}}, %[[KEY_VIEW]], %{{.*}}, %{{.*}} : (tensor<1x4x8xf32>, tensor<1x8x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4xf32>
  // CHECK: %[[SCORES:.*]] = tosa.reshape %[[QK]], %{{.*}} : (tensor<1x4x4xf32>, !tosa.shape<3>) -> tensor<1x4x4xf32>
  // CHECK: %[[SCALED:.*]] = tosa.mul %[[SCORES]], %{{.*}}, %{{.*}} : (tensor<1x4x4xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x4x4xf32>
  // CHECK: %[[CENTERED:.*]] = tosa.sub %[[SCALED]], %{{.*}} : (tensor<1x4x4xf32>, tensor<1x4x1xf32>) -> tensor<1x4x4xf32>
  // CHECK: %[[EXP:.*]] = tosa.exp %[[CENTERED]] : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  // CHECK: %[[DENOM:.*]] = tosa.reduce_sum %[[EXP]] {axis = 2 : i32} : (tensor<1x4x4xf32>) -> tensor<1x4x1xf32>
  // CHECK: %[[SOFTMAX:.*]] = tosa.mul %[[EXP]], %{{.*}}, %{{.*}} : (tensor<1x4x4xf32>, tensor<1x4x1xf32>, tensor<1xi8>) -> tensor<1x4x4xf32>
  // CHECK: %[[OUTPUT:.*]] = tosa.matmul %[[SOFTMAX]], %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x4x4xf32>, tensor<1x4x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x8xf32>
  // CHECK: return %{{.*}} : !torch.vtensor<[1,4,8],f32>
  func.func @scaled_dot_product_attention(
      %query: !torch.vtensor<[1,4,8],f32>,
      %key: !torch.vtensor<[1,4,8],f32>,
      %value: !torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[1,4,8],f32> {
    %none = torch.constant.none
    %zero = torch.constant.float 0.000000e+00
    %false = torch.constant.bool false
    %result = torch.aten.scaled_dot_product_attention %query, %key, %value, %none, %zero, %false, %none, %false :
      !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,4,8],f32>
    return %result : !torch.vtensor<[1,4,8],f32>
  }
}
