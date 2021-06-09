// RUN: npcomp-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func @torch.aten.__is__
// CHECK:           %[[FALSE:.*]] = basicpy.bool_constant false
// CHECK:           return %[[FALSE]] : !basicpy.BoolType
func @torch.aten.__is__(%arg0: !torch.list<i64>, %arg1: !basicpy.NoneType) -> !basicpy.BoolType{
  %0 = torch.aten.__is__ %arg0, %arg1 : !torch.list<i64>, !basicpy.NoneType -> !basicpy.BoolType
  return %0 : !basicpy.BoolType
}

// CHECK-LABEL:   func @torch.aten.size$canonicalize_to_list(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.list<i64> {
// CHECK:           %[[C2:.*]] = constant 2 : i64
// CHECK:           %[[C3:.*]] = constant 3 : i64
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C2]], %[[C3]] : (i64, i64) -> !torch.list<i64>
// CHECK:           return %[[LIST]] : !torch.list<i64>
func @torch.aten.size$canonicalize_to_list(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.list<i64> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[2,3],f32> -> !torch.list<i64>
  return %0 : !torch.list<i64>
}

// One size unknown, so cannot canonicalize.
// TODO: For unknown sizes, insert the equivalent of a "dim" op.
// Then this will only require static rank.
// CHECK-LABEL:   func @torch.aten.size$unknown_size(
// CHECK-SAME:                                       %[[ARG:.*]]: !torch.vtensor<[?,3],f32>) -> !torch.list<i64> {
// CHECK:           %[[SIZE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor<[?,3],f32> -> !torch.list<i64>
func @torch.aten.size$unknown_size(%arg0: !torch.vtensor<[?,3],f32>) -> !torch.list<i64> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[?,3],f32> -> !torch.list<i64>
  return %0 : !torch.list<i64>
}

// CHECK-LABEL:   func @torch.aten.len.t$of_size(
// CHECK-SAME:                                   %[[ARG:.*]]: !torch.vtensor<*,f32>) -> i64 {
// CHECK:           %[[DIM:.*]] = torch.aten.dim %[[ARG]] : !torch.vtensor<*,f32> -> i64
// CHECK:           return %[[DIM]] : i64
func @torch.aten.len.t$of_size(%arg0: !torch.vtensor<*,f32>) -> i64 {
  %0 = torch.aten.size %arg0 : !torch.vtensor<*,f32> -> !torch.list<i64>
  %1 = torch.aten.len.t %0 : !torch.list<i64> -> i64
  return %1 : i64
}

// CHECK-LABEL:   func @torch.aten.dim$with_shape(
// CHECK-SAME:                                    %[[ARG:.*]]: !torch.vtensor<[?,?,?],f32>) -> i64 {
// CHECK:           %[[DIM:.*]] = constant 3 : i64
// CHECK:           return %[[DIM]] : i64
func @torch.aten.dim$with_shape(%arg0: !torch.vtensor<[?,?,?],f32>) -> i64 {
  %0 = torch.aten.dim %arg0 : !torch.vtensor<[?,?,?],f32> -> i64
  return %0 : i64
}

// CHECK-LABEL:   func @torch.aten.len.t$of_build_list(
// CHECK-SAME:                                         %[[ARG:.*]]: i64) -> i64 {
// CHECK:           %[[LEN:.*]] = constant 4 : i64
// CHECK:           return %[[LEN]] : i64
func @torch.aten.len.t$of_build_list(%arg0: i64) -> i64 {
  %0 = torch.prim.ListConstruct %arg0, %arg0, %arg0, %arg0 : (i64, i64, i64, i64) -> !torch.list<i64>
  %1 = torch.aten.len.t %0 : !torch.list<i64> -> i64
  return %1 : i64
}

// CHECK-LABEL:   func @torch.copy.tensor$value_copy_is_noop(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           return %[[ARG]] : !torch.vtensor
func @torch.copy.tensor$value_copy_is_noop(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.tensor %arg0 : !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func @torch.copy.tensor$unnecessary_intermediate_nonval_tensor(
// CHECK-SAME:                                                                    %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           return %[[ARG]] : !torch.vtensor
func @torch.copy.tensor$unnecessary_intermediate_nonval_tensor(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.tensor %arg0 : !torch.vtensor -> !torch.tensor
  %1 = torch.copy.tensor %0 : !torch.tensor -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func @torch.aten.__getitem__.t(
// CHECK:           %[[C5:.*]] = constant 5 : i64
// CHECK:           return %[[C5]] : i64
func @torch.aten.__getitem__.t() -> i64 {
    %c4_i64 = constant 4 : i64
    %c5_i64 = constant 5 : i64
    %c1_i64 = constant 1 : i64
    %0 = torch.prim.ListConstruct %c4_i64, %c5_i64 : (i64, i64) -> !torch.list<i64>
    %1 = torch.aten.__getitem__.t %0, %c1_i64 : !torch.list<i64>, i64 -> i64
    return %1 : i64
}

// Not canonicalized because of passed in index
// CHECK-LABEL:   func @torch.aten.__getitem__.t_no_change_test0(
// CHECK:           %[[C4:.*]] = constant 4 : i64
// CHECK:           %[[C5:.*]] = constant 5 : i64
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C4]], %[[C5]] : (i64, i64) -> !torch.list<i64>
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %[[LIST]], %arg0 : !torch.list<i64>, i64 -> i64
// CHECK:           return %[[ITEM]] : i64
func @torch.aten.__getitem__.t_no_change_test0(%arg0: i64) -> i64 {
  %c4_i64 = constant 4 : i64
  %c5_i64 = constant 5 : i64
  %0 = torch.prim.ListConstruct %c4_i64, %c5_i64 : (i64, i64) -> !torch.list<i64>
  %1 = torch.aten.__getitem__.t %0, %arg0 : !torch.list<i64>, i64 -> i64
  return %1 : i64
}

// Not canonicalized because of passed in list
// CHECK-LABEL:   func @torch.aten.__getitem__.t_no_change_test1(
// CHECK:           %[[C5:.*]] = constant 5 : i64
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %arg0, %[[C5]] : !torch.list<i64>, i64 -> i64
// CHECK:           return %[[ITEM]] : i64
func @torch.aten.__getitem__.t_no_change_test1(%arg0: !torch.list<i64>) -> i64 {
  %c5_i64 = constant 5 : i64
  %0 = torch.aten.__getitem__.t %arg0, %c5_i64 : !torch.list<i64>, i64 -> i64
  return %0 : i64
}
