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


