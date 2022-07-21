// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.view$view_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[INT224:.*]] = torch.constant.int 224
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT]]-1, %[[INT]]224 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT]]-1
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT224]]
// CHECK:         %[[T4:.*]] = arith.trunci %[[T2]] : i64 to i32
// CHECK:         %[[T5:.*]] = arith.trunci %[[T3]] : i64 to i32
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[T4]], %[[T5]] : tensor<2xi32>
// CHECK:         %[[T7:.*]] = "chlo.dynamic_reshape"(%[[T0]], %[[T6]]) : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x224xf32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x224xf32> -> !torch.vtensor<[?,224],f32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,224],f32>
func.func @torch.aten.view$view_like(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
  %int-1 = torch.constant.int -1
  %int224 = torch.constant.int 224
  %0 = torch.prim.ListConstruct %int-1, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,224],f32>
  return %1 : !torch.vtensor<[?,224],f32>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.reshape$view_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?,?],f32> -> tensor<?x?x?x?x?xf32>
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[INT120:.*]] = torch.constant.int 120
// CHECK:         %[[INT4:.*]] = torch.constant.int 4
// CHECK:         %[[INT64:.*]] = torch.constant.int 64
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT]]-1, %[[INT]]120, %[[INT]]4, %[[INT]]64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT]]-1
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT120]]
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %[[INT4]]
// CHECK:         %[[T5:.*]] = torch_c.to_i64 %[[INT64]]
// CHECK:         %[[T6:.*]] = arith.trunci %[[T2]] : i64 to i32
// CHECK:         %[[T7:.*]] = arith.trunci %[[T3]] : i64 to i32
// CHECK:         %[[T8:.*]] = arith.trunci %[[T4]] : i64 to i32
// CHECK:         %[[T9:.*]] = arith.trunci %[[T5]] : i64 to i32
// CHECK:         %[[T10:.*]] = tensor.from_elements %[[T6]], %[[T7]], %[[T8]], %[[T9]] : tensor<4xi32>
// CHECK:         %[[T11:.*]] = "chlo.dynamic_reshape"(%[[T0]], %[[T10]]) : (tensor<?x?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x120x4x64xf32>
// CHECK:         %[[T12:.*]] = torch_c.from_builtin_tensor %[[T11]] : tensor<?x120x4x64xf32> -> !torch.vtensor<[?,120,4,64],f32>
// CHECK:         return %[[T12]] : !torch.vtensor<[?,120,4,64],f32>
func.func @torch.aten.reshape$view_like(%arg0: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
  %int-1 = torch.constant.int -1
  %int120 = torch.constant.int 120
  %int4 = torch.constant.int 4
  %int64 = torch.constant.int 64
  %0 = torch.prim.ListConstruct %int-1, %int120, %int4, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reshape %arg0, %0 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,120,4,64],f32>
  return %1 : !torch.vtensor<[?,120,4,64],f32>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.view.minus1$view_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,3,?,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,3,?,?],f32> -> tensor<2x3x?x?xf32>
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T1:.*]] = torch.aten.size.int %[[ARG0]], %[[INT0]] : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
// CHECK:         %[[T2:.*]] = torch.aten.size.int %[[ARG0]], %[[INT1]] : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
// CHECK:         %[[T3:.*]] = torch.prim.ListConstruct %[[T1]], %[[T2]], %[[INT]]-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %[[T1]]
// CHECK:         %[[T5:.*]] = torch_c.to_i64 %[[T2]]
// CHECK:         %[[T6:.*]] = torch_c.to_i64 %[[INT]]-1
// CHECK:         %[[T7:.*]] = arith.trunci %[[T4]] : i64 to i32
// CHECK:         %[[T8:.*]] = arith.trunci %[[T5]] : i64 to i32
// CHECK:         %[[T9:.*]] = arith.trunci %[[T6]] : i64 to i32
// CHECK:         %[[T10:.*]] = tensor.from_elements %[[T7]], %[[T8]], %[[T9]] : tensor<3xi32>
// CHECK:         %[[T11:.*]] = "chlo.dynamic_reshape"(%[[T0]], %[[T10]]) : (tensor<2x3x?x?xf32>, tensor<3xi32>) -> tensor<2x3x?xf32>
// CHECK:         %[[T12:.*]] = torch_c.from_builtin_tensor %[[T11]] : tensor<2x3x?xf32> -> !torch.vtensor<[2,3,?],f32>
// CHECK:         return %[[T12]] : !torch.vtensor<[2,3,?],f32>
func.func @torch.aten.view.minus1$view_like(%arg0: !torch.vtensor<[2,3,?,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
  %int-1 = torch.constant.int -1
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
  %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
  %2 = torch.prim.ListConstruct %0, %1, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[2,3,?,?],f32>, !torch.list<int> -> !torch.vtensor<[2,3,?],f32>
  return %3 : !torch.vtensor<[2,3,?],f32>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.view.to_rank1$view_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = "mhlo.reshape"(%[[T0]]) : (tensor<f32>) -> tensor<1xf32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[1],f32>
func.func @torch.aten.view.to_rank1$view_like(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1],f32>
  return %1 : !torch.vtensor<[1],f32>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.view.to_rank0$view_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1],f32> -> tensor<1xf32>
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:         %[[T2:.*]] = "mhlo.reshape"(%[[T0]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[],f32>
func.func @torch.aten.view.to_rank0$view_like(%arg0: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}
