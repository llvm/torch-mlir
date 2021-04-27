// RUN: npcomp-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func @aten.__is__
// CHECK:           %[[FALSE:.*]] = basicpy.bool_constant false
// CHECK:           return %[[FALSE]] : !basicpy.BoolType
func @aten.__is__(%arg0: !basicpy.ListType, %arg1: !basicpy.NoneType) -> !basicpy.BoolType{
  %0 = "aten.__is__"(%arg0, %arg1) : (!basicpy.ListType, !basicpy.NoneType) -> !basicpy.BoolType
  return %0 : !basicpy.BoolType
}
