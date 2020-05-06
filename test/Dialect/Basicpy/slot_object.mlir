// RUN: npcomp-opt -split-input-file %s | npcomp-opt | FileCheck --dump-input=fail %s

// CHECK-LABEL: @slot_object_make
func @slot_object_make() -> (!basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType, !basicpy.NoneType>) {
  // CHECK: %[[N:.+]] = basicpy.singleton
  %0 = basicpy.singleton : !basicpy.NoneType
  // CHECK: basicpy.slot_object_make(%[[N]], %[[N]], %[[N]]) -> !basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType, !basicpy.NoneType>
  %1 = "basicpy.slot_object_make"(%0, %0, %0) {className = "slice" } : 
      (!basicpy.NoneType, !basicpy.NoneType, !basicpy.NoneType) -> 
      (!basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType, !basicpy.NoneType>)
  return %1 : !basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType, !basicpy.NoneType>
}

// -----
func @slot_object_get() -> (!basicpy.NoneType) {
  %0 = basicpy.singleton : !basicpy.NoneType
  // CHECK: %[[OBJ:.+]] = basicpy.slot_object_make
  %1 = basicpy.slot_object_make(%0, %0) -> (!basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType>)
  // CHECK:  basicpy.slot_object_get %[[OBJ]][1] : !basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType>
  %2 = basicpy.slot_object_get %1[1] : !basicpy.SlotObject<slice, !basicpy.NoneType, !basicpy.NoneType>
  return %2 : !basicpy.NoneType
}

// TODO: Verify illegal forms
