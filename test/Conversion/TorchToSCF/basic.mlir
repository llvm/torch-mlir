// RUN: npcomp-opt <%s -convert-torch-to-scf | FileCheck %s

// CHECK-LABEL:   func @torch.prim.if(
// CHECK-SAME:                        %[[VAL_0:.*]]: !torch.bool) -> !torch.int {
// CHECK:           %[[VAL_1:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.to_i1 %[[VAL_0]]
// CHECK:           %[[VAL_4:.*]] = scf.if %[[VAL_3]] -> (i64) {
// CHECK:             %[[VAL_5:.*]] = torch.to_i64 %[[VAL_1]]
// CHECK:             scf.yield %[[VAL_5]] : i64
// CHECK:           } else {
// CHECK:             %[[VAL_6:.*]] = torch.to_i64 %[[VAL_2]]
// CHECK:             scf.yield %[[VAL_6]] : i64
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = torch.from_i64 %[[VAL_8:.*]]
// CHECK:           return %[[VAL_7]] : !torch.int
func @torch.prim.if(%arg0: !torch.bool) -> !torch.int {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.prim.If %arg0 -> (!torch.int) {
    torch.prim.If.yield %int2 : !torch.int
  } else {
    torch.prim.If.yield %int1 : !torch.int
  }
  return %0 : !torch.int
}

// CHECK-LABEL:   func @aten.prim.if$nested(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.bool,
// CHECK-SAME:                              %[[VAL_1:.*]]: !torch.bool) -> !torch.int {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_5:.*]] = torch.to_i1 %[[VAL_0]]
// CHECK:           %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (i64) {
// CHECK:             %[[VAL_7:.*]] = torch.to_i1 %[[VAL_1]]
// CHECK:             %[[VAL_8:.*]] = scf.if %[[VAL_7]] -> (i64) {
// CHECK:               %[[VAL_9:.*]] = torch.to_i64 %[[VAL_2]]
// CHECK:               scf.yield %[[VAL_9]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_10:.*]] = torch.to_i64 %[[VAL_3]]
// CHECK:               scf.yield %[[VAL_10]] : i64
// CHECK:             }
// CHECK:             scf.yield %[[VAL_11:.*]] : i64
// CHECK:           } else {
// CHECK:             %[[VAL_12:.*]] = torch.to_i64 %[[VAL_4]]
// CHECK:             scf.yield %[[VAL_12]] : i64
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = torch.from_i64 %[[VAL_14:.*]]
// CHECK:           return %[[VAL_13]] : !torch.int
func @aten.prim.if$nested(%arg0: !torch.bool, %arg1: !torch.bool) -> !torch.int {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %0 = torch.prim.If %arg0 -> (!torch.int) {
    %1 = torch.prim.If %arg1 -> (!torch.int) {
      torch.prim.If.yield %int2 : !torch.int
    } else {
      torch.prim.If.yield %int3 : !torch.int
    }
    torch.prim.If.yield %1 : !torch.int
  } else {
    torch.prim.If.yield %int4 : !torch.int
  }
  return %0 : !torch.int
}
