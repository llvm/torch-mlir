// RUN: torch-mlir-opt %s -convert-torch-to-tosa -split-input-file | FileCheck %s

// CHECK-LABEL:  func.func @convtranspose3d(
// CHECK-SAME:      %[[INPUT:.*]]: !torch.vtensor<[1,1,2,2,2],f32>) -> !torch.vtensor<[1,1,4,4,4],f32> {
// CHECK:          %[[BT:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[1,1,2,2,2],f32> -> tensor<1x1x2x2x2xf32>
// CHECK:          %[[WCONST:.*]] = "tosa.const"
// CHECK:          %[[PERM_INPUT:.*]] = tosa.transpose %[[BT]] {perms = array<i32: 0, 2, 3, 4, 1>}
// CHECK:          %[[PERM_WEIGHT:.*]] = tosa.transpose %[[WCONST]] {perms = array<i32: 1, 2, 3, 4, 0>}
// CHECK:          %[[REV1:.*]] = tosa.reverse %[[PERM_WEIGHT]] {axis = 1 : i32}
// CHECK:          %[[REV2:.*]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
// CHECK:          %[[REV3:.*]] = tosa.reverse %[[REV2]] {axis = 3 : i32}
// CHECK:          %[[CONV:.*]] = tosa.conv3d %{{.*}} %[[REV3]]
// CHECK:          %[[FINAL:.*]] = tosa.transpose %[[CONV]] {perms = array<i32: 0, 4, 1, 2, 3>}
// CHECK:          %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[FINAL]]
// CHECK:          return %[[RESULT]]
// CHECK:        }
func.func @convtranspose3d(%input: !torch.vtensor<[1,1,2,2,2],f32>) -> !torch.vtensor<[1,1,4,4,4],f32> {
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %weight = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x1x3x3x3xf32>) : !torch.vtensor<[1,1,3,3,3],f32>
  %bias = torch.vtensor.literal(dense<0.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
  %stride = torch.prim.ListConstruct %int2, %int2, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %out_padding = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %groups = torch.constant.int 1
  %result = torch.aten.convolution %input, %weight, %bias, %stride, %padding, %dilation, %true, %out_padding, %groups : !torch.vtensor<[1,1,2,2,2],f32>, !torch.vtensor<[1,1,3,3,3],f32>, !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,4,4,4],f32>
  return %result : !torch.vtensor<[1,1,4,4,4],f32>
}
