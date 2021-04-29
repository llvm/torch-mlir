// RUN: npcomp-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func @aten.__is__
// CHECK:           %[[FALSE:.*]] = basicpy.bool_constant false
// CHECK:           return %[[FALSE]] : !basicpy.BoolType
func @aten.__is__(%arg0: !basicpy.ListType, %arg1: !basicpy.NoneType) -> !basicpy.BoolType{
  %0 = "aten.__is__"(%arg0, %arg1) : (!basicpy.ListType, !basicpy.NoneType) -> !basicpy.BoolType
  return %0 : !basicpy.BoolType
}
// CHECK-LABEL:   func @aten.size(
// CHECK:           %[[CM1:.*]] = constant -1 : i64
// CHECK:           %[[C3:.*]] = constant 3 : i64
// CHECK:           %[[RET:.*]] = basicpy.build_list %[[CM1]], %[[C3]] : (i64, i64) -> !basicpy.ListType
// CHECK:           return %[[RET]] : !basicpy.ListType
func @aten.size(%arg0: tensor<?x3xf32>) -> !basicpy.ListType {
  %0 = "aten.size"(%arg0) : (tensor<?x3xf32>) -> !basicpy.ListType
  return %0 : !basicpy.ListType
}

// CHECK-LABEL:   func @aten.len.t(
// CHECK:           %[[LENGTH:.*]] = constant 2 : i64
// CHECK:           return %[[LENGTH]] : i64
func @aten.len.t(%arg0: i64) -> i64 {
  %0 = basicpy.build_list %arg0, %arg0 : (i64, i64) -> !basicpy.ListType
  %1 = "aten.len.t"(%0) : (!basicpy.ListType) -> i64
  return %1 : i64
}
