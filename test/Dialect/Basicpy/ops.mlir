// RUN: npcomp-opt -split-input-file %s | npcomp-opt | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @build_dict_generic
func @build_dict_generic() -> !basicpy.DictType {
  // CHECK: basicpy.build_dict : () -> !basicpy.DictType
  %0 = basicpy.build_dict : () -> !basicpy.DictType
  return %0 : !basicpy.DictType
}

// -----
// CHECK-LABEL: @build_list_generic
func @build_list_generic(%arg0 : si32, %arg1 : si32) -> !basicpy.ListType {
  // CHECK: basicpy.build_list %arg0, %arg1 : (si32, si32) -> !basicpy.ListType
  %0 = basicpy.build_list %arg0, %arg1 : (si32, si32) -> !basicpy.ListType
  return %0 : !basicpy.ListType
}

// -----
// CHECK-LABEL: @build_tuple_generic
func @build_tuple_generic(%arg0 : si32, %arg1 : si32) -> !basicpy.TupleType {
  // CHECK: basicpy.build_tuple %arg0, %arg1 : (si32, si32) -> !basicpy.TupleType
  %0 = basicpy.build_tuple %arg0, %arg1 : (si32, si32) -> !basicpy.TupleType
  return %0 : !basicpy.TupleType
}

// -----
// CHECK-LABEL: @numeric_constant
func @numeric_constant() {
  // CHECK: %num-1_si32 = basicpy.numeric_constant -1 : si32
  %0 = basicpy.numeric_constant -1 : si32
  // CHECK: %num1_ui32 = basicpy.numeric_constant 1 : ui32
  %1 = basicpy.numeric_constant 1 : ui32
  // CHECK: %num = basicpy.numeric_constant 2.000000e+00 : f32
  %2 = basicpy.numeric_constant 2.0 : f32
  // CHECK: %num_0 = basicpy.numeric_constant [2.000000e+00 : f32, 3.000000e+00 : f32] : complex<f32>
  %3 = basicpy.numeric_constant [2.0 : f32, 3.0 : f32] : complex<f32>
  // CHECK: %bool_true = basicpy.bool_constant true
  %4 = basicpy.bool_constant true
  // CHECK: %bool_false = basicpy.bool_constant false
  %5 = basicpy.bool_constant false
  return
}
