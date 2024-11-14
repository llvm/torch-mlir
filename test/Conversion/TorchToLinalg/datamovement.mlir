// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.permute(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[64,32,16,8,4],f32>) -> !torch.vtensor<[64,8,4,32,16],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64,32,16,8,4],f32> -> tensor<64x32x16x8x4xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x8x4x32x16xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.transpose ins(%[[VAL_1]] : tensor<64x32x16x8x4xf32>) outs(%[[VAL_2]] : tensor<64x8x4x32x16xf32>) permutation = [0, 3, 4, 1, 2]
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<64x8x4x32x16xf32> -> !torch.vtensor<[64,8,4,32,16],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[64,8,4,32,16],f32>
// CHECK:         }
func.func @torch.aten.permute(%arg0: !torch.vtensor<[64,32,16,8,4],f32>) -> !torch.vtensor<[64,8,4,32,16],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int0, %int3, %int4, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[64,32,16,8,4],f32>, !torch.list<int> -> !torch.vtensor<[64,8,4,32,16],f32>
  return %1 : !torch.vtensor<[64,8,4,32,16],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.permute$rank0(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_2:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_2]] : !torch.vtensor<[],f32>
// CHECK:         }
func.func @torch.aten.permute$rank0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}
