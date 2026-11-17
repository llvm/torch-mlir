// RUN: torch-mlir-opt %s -convert-torch-to-tmtensor -split-input-file -verify-diagnostics | FileCheck %s

// Negative dimensions must select the corresponding coordinate in the
// flattened scatter indices rather than indexing the coordinate vector directly.
// CHECK-LABEL: func.func @scatter_add_negative_dim
// CHECK: %[[INDEX:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xi64>
// CHECK: %[[SRC:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xf32>
// CHECK: %[[ROW:.*]] = arith.index_cast
// CHECK: linalg.yield %[[ROW]], %[[INDEX]], %[[SRC]]
// CHECK: tm_tensor.scatter
func.func @scatter_add_negative_dim(%self: !torch.vtensor<[2,4],f32>, %index: !torch.vtensor<[2,3],si64>, %src: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int -1
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2,4],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// -----

// scatter.src shares the lowering. The lowest valid dimension (-rank)
// must normalize to the first coordinate.
// CHECK-LABEL: func.func @scatter_src_negative_rank
// CHECK: %[[INDEX:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xi64>
// CHECK: %[[SRC:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xf32>
// CHECK: arith.index_cast
// CHECK: %[[COL:.*]] = arith.index_cast
// CHECK: linalg.yield %[[INDEX]], %[[COL]], %[[SRC]]
// CHECK: tm_tensor.scatter
func.func @scatter_src_negative_rank(%self: !torch.vtensor<[4,3],f32>, %index: !torch.vtensor<[2,3],si64>, %src: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[4,3],f32> {
  %dim = torch.constant.int -2
  %result = torch.aten.scatter.src %self, %dim, %index, %src : !torch.vtensor<[4,3],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[4,3],f32>
  return %result : !torch.vtensor<[4,3],f32>
}

// -----

// Reject dimensions outside [-rank, rank) before indexing the coordinate vector.
func.func @scatter_add_dim_too_large(%self: !torch.vtensor<[2,3],f32>, %index: !torch.vtensor<[2,3],si64>, %src: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %dim = torch.constant.int 2
  // expected-error @+1 {{failed to legalize operation 'torch.aten.scatter_add'}}
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2,3],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
  return %result : !torch.vtensor<[2,3],f32>
}

// -----

func.func @scatter_src_dim_too_negative(%self: !torch.vtensor<[2,3],f32>, %index: !torch.vtensor<[2,3],si64>, %src: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %dim = torch.constant.int -3
  // expected-error @+1 {{failed to legalize operation 'torch.aten.scatter.src'}}
  %result = torch.aten.scatter.src %self, %dim, %index, %src : !torch.vtensor<[2,3],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
  return %result : !torch.vtensor<[2,3],f32>
}
