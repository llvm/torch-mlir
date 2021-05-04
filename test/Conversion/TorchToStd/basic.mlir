// RUN: npcomp-opt <%s -convert-torch-to-std | FileCheck %s

// CHECK-LABEL:   func @aten.dim(
// CHECK-SAME:                   %[[ARG0:.*]]: tensor<*x!numpy.any_dtype>) -> i64 {
// CHECK:           %[[RANK_INDEX:.*]] = rank %[[ARG0]] : tensor<*x!numpy.any_dtype>
// CHECK:           %[[RANK_I64:.*]] = index_cast %[[RANK_INDEX]] : index to i64
// CHECK:           return %[[RANK_I64]] : i64
func @aten.dim(%arg0: tensor<*x!numpy.any_dtype>) -> i64 {
  %0 = torch.aten.dim %arg0 : tensor<*x!numpy.any_dtype> -> i64
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
