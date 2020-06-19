// RUN: npcomp-opt -split-input-file %s | npcomp-opt -canonicalize | FileCheck --dump-input=fail %s

// CHECK-LABEL: func @unknown_cast_elide
func @unknown_cast_elide(%arg0 : i32) -> i32 {
  // CHECK-NOT: basicpy.unknown_cast
  %0 = basicpy.unknown_cast %arg0 : i32 -> i32
  return %0 : i32
}

// CHECK-LABEL: func @unknown_cast_preserve
func @unknown_cast_preserve(%arg0 : i32) -> !basicpy.UnknownType {
  // CHECK: basicpy.unknown_cast
  %0 = basicpy.unknown_cast %arg0 : i32 -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}
