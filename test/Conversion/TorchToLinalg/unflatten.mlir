// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.unflatten.int$static
// CHECK:           torch_c.to_builtin_tensor
// CHECK:           tensor.expand_shape
// CHECK:           torch_c.from_builtin_tensor
func.func @torch.aten.unflatten.int$static(%arg0: !torch.vtensor<[2,6,4],f32>) -> !torch.vtensor<[2,2,3,4],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.unflatten.int %arg0, %int1, %0 : !torch.vtensor<[2,6,4],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[2,2,3,4],f32>
  return %1 : !torch.vtensor<[2,2,3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unflatten.int$negative_dim
// CHECK:           torch_c.to_builtin_tensor
// CHECK:           tensor.expand_shape
// CHECK:           torch_c.from_builtin_tensor
func.func @torch.aten.unflatten.int$negative_dim(%arg0: !torch.vtensor<[2,6,4],f32>) -> !torch.vtensor<[2,2,3,4],f32> {
  %int-2 = torch.constant.int -2
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.unflatten.int %arg0, %int-2, %0 : !torch.vtensor<[2,6,4],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[2,2,3,4],f32>
  return %1 : !torch.vtensor<[2,2,3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unflatten.int$inferred_size
// CHECK:           torch_c.to_builtin_tensor
// CHECK:           tensor.expand_shape
// CHECK:           torch_c.from_builtin_tensor
func.func @torch.aten.unflatten.int$inferred_size(%arg0: !torch.vtensor<[3,12],f32>) -> !torch.vtensor<[3,2,6],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int2, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.unflatten.int %arg0, %int1, %0 : !torch.vtensor<[3,12],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[3,2,6],f32>
  return %1 : !torch.vtensor<[3,2,6],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unflatten.int$dynamic_input
// CHECK:           torch_c.to_builtin_tensor
// CHECK:           tensor.expand_shape
// CHECK:           torch_c.from_builtin_tensor
func.func @torch.aten.unflatten.int$dynamic_input(%arg0: !torch.vtensor<[?,6],f32>) -> !torch.vtensor<[?,2,3],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.unflatten.int %arg0, %int1, %0 : !torch.vtensor<[?,6],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[?,2,3],f32>
  return %1 : !torch.vtensor<[?,2,3],f32>
}
