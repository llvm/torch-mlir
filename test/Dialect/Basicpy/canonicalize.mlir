// RUN: npcomp-opt -split-input-file -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @unknown_cast_elide
func @unknown_cast_elide(%arg0 : i32) -> i32 {
  // CHECK-NOT: basicpy.unknown_cast
  %0 = basicpy.unknown_cast %arg0 : i32 -> i32
  return %0 : i32
}

// -----
// CHECK-LABEL: func @unknown_cast_preserve
func @unknown_cast_preserve(%arg0 : i32) -> !basicpy.UnknownType {
  // CHECK: basicpy.unknown_cast
  %0 = basicpy.unknown_cast %arg0 : i32 -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

// -----
// CHECK-LABEL: @numeric_constant_si32
func @numeric_constant_si32() -> si32 {
  // CHECK: %num-1_si32 = basicpy.numeric_constant -1 : si32
  %0 = basicpy.numeric_constant -1 : si32
  return %0 : si32
}

// -----
// CHECK-LABEL: @numeric_constant_ui32
func @numeric_constant_ui32() -> ui32 {
  // CHECK: %num1_ui32 = basicpy.numeric_constant 1 : ui32
  %0 = basicpy.numeric_constant 1 : ui32
  return %0 : ui32
}

// -----
// CHECK-LABEL: @numeric_constant_f32
func @numeric_constant_f32() -> f32 {
  // CHECK: %num = basicpy.numeric_constant 2.000000e+00 : f32
  %0 = basicpy.numeric_constant 2.0 : f32
  return %0 : f32
}

// -----
// CHECK-LABEL: @numeric_constant_complex_f32
func @numeric_constant_complex_f32() -> complex<f32> {
  // CHECK: %num = basicpy.numeric_constant [2.000000e+00 : f32, 3.000000e+00 : f32] : complex<f32>
  %0 = basicpy.numeric_constant [2.0 : f32, 3.0 : f32] : complex<f32>
  return %0 : complex<f32>
}

// -----
// CHECK-LABEL: @bool_constant
func @bool_constant() -> !basicpy.BoolType {
  // CHECK: %bool_true = basicpy.bool_constant true
  %0 = basicpy.bool_constant true
  return %0 : !basicpy.BoolType
}

// -----
// CHECK-LABEL: @bytes_constant
func @bytes_constant() -> !basicpy.BytesType {
  // CHECK: %bytes = basicpy.bytes_constant "foobar"
  %0 = basicpy.bytes_constant "foobar"
  return %0 : !basicpy.BytesType
}

// -----
// CHECK-LABEL: @str_constant
func @str_constant() -> !basicpy.StrType {
  // CHECK: %str = basicpy.str_constant "foobar"
  %0 = basicpy.str_constant "foobar"
  return %0 : !basicpy.StrType
}

// -----
// CHECK-LABEL: @bool_cast
func @bool_cast() -> i1 {
  // CHECK: %[[CTRUE:.*]] = constant true
  %0 = basicpy.bool_constant true
  %1 = basicpy.bool_cast %0 : !basicpy.BoolType -> i1
  // CHECK: return %[[CTRUE]] : i1
  return %1 : i1
}

// -----
// CHECK-LABEL: func @singleton_coalesce
func @singleton_coalesce() -> (!basicpy.NoneType, !basicpy.NoneType) {
  %0 = basicpy.singleton : !basicpy.NoneType
  %1 = basicpy.singleton : !basicpy.NoneType
  // CHECK-NEXT: %[[RET:.*]] = basicpy.singleton
  // CHECK-NEXT: return %[[RET]], %[[RET]]
  return %0, %1 : !basicpy.NoneType, !basicpy.NoneType
}
