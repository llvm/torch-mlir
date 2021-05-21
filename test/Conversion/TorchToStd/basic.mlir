// RUN: npcomp-opt <%s -convert-torch-to-std | FileCheck %s

// CHECK-LABEL:   func @aten.dim(
// CHECK-SAME:                   %[[ARG0:.*]]: !torch.vtensor<*,f32>) -> i64 {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch.to_builtin_tensor %[[ARG0]] : !torch.vtensor<*,f32> -> tensor<*xf32>
// CHECK:           %[[RANK_INDEX:.*]] = rank %[[BUILTIN_TENSOR]] : tensor<*xf32>
// CHECK:           %[[RANK_I64:.*]] = index_cast %[[RANK_INDEX]] : index to i64
// CHECK:           return %[[RANK_I64]] : i64
func @aten.dim(%arg0: !torch.vtensor<*,f32>) -> i64 {
  %0 = torch.aten.dim %arg0 : !torch.vtensor<*,f32> -> i64
  return %0 : i64
}

// CHECK-LABEL:   func @torch.aten.ne.int(
// CHECK-SAME:                      %[[ARG0:.*]]: i64,
// CHECK-SAME:                      %[[ARG1:.*]]: i64) -> !basicpy.BoolType {
// CHECK:           %[[I1:.*]] = cmpi ne, %[[ARG0]], %[[ARG1]] : i64
// CHECK:           %[[BASICPY_BOOL:.*]] = basicpy.bool_cast %[[I1]] : i1 -> !basicpy.BoolType
// CHECK:           return %[[BASICPY_BOOL]] : !basicpy.BoolType
func @torch.aten.ne.int(%arg0: i64, %arg1: i64) -> !basicpy.BoolType {
  %0 = torch.aten.ne.int %arg0, %arg1 : i64, i64 -> !basicpy.BoolType
  return %0 : !basicpy.BoolType
}

// CHECK-LABEL:   func @torch.aten.gt.int(
// CHECK-SAME:                      %[[ARG0:.*]]: i64,
// CHECK-SAME:                      %[[ARG1:.*]]: i64) -> !basicpy.BoolType {
// CHECK:           %[[I1:.*]] = cmpi sgt, %[[ARG0]], %[[ARG1]] : i64
// CHECK:           %[[BASICPY_BOOL:.*]] = basicpy.bool_cast %[[I1]] : i1 -> !basicpy.BoolType
// CHECK:           return %[[BASICPY_BOOL]] : !basicpy.BoolType
func @torch.aten.gt.int(%arg0: i64, %arg1: i64) -> !basicpy.BoolType {
  %0 = torch.aten.gt.int %arg0, %arg1 : i64, i64 -> !basicpy.BoolType
  return %0 : !basicpy.BoolType
}

// CHECK-LABEL:   func @torch.tensor$value() -> !torch.vtensor<[],f32> {
// CHECK:           %[[CST:.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VTENSOR:.*]] = torch.from_builtin_tensor %[[CST]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VTENSOR]] : !torch.vtensor<[],f32>
func @torch.tensor$value() -> !torch.vtensor<[],f32> {
  %0 = torch.tensor(dense<0.0> : tensor<f32>) : !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func @torch.tensor$nonval() -> !torch.tensor<[],f32> {
// CHECK:           %[[CST:.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VTENSOR:.*]] = torch.from_builtin_tensor %[[CST]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[NONVAL:.*]] = torch.copy.tensor %[[VTENSOR]] : !torch.vtensor<[],f32> -> !torch.tensor<[],f32>
// CHECK:           return %[[NONVAL]] : !torch.tensor<[],f32>
func @torch.tensor$nonval() -> !torch.tensor<[],f32> {
  %0 = torch.tensor(dense<0.0> : tensor<f32>) : !torch.tensor<[],f32>
  return %0 : !torch.tensor<[],f32>
}
