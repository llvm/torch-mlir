// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.sym_constrain_range(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.vtensor<[],si64>) -> !torch.int {
// CHECK:           %[[VAL_1:.*]] = torch.constant.int 7
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = torch.aten.item %[[VAL_0]] : !torch.vtensor<[],si64> -> !torch.int
// CHECK:           %[[VAL_5:.*]] = torch_c.to_i64 %[[VAL_4]]
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_7:.*]] = arith.constant 9223372036854775807 : i64
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_6]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sle, %[[VAL_5]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_10:.*]] = arith.andi %[[VAL_8]], %[[VAL_9]] : i1
// CHECK:           cf.assert %[[VAL_10]], "Size constraint failed. Expected range: [0, 9223372036854775807]"
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_12:.*]] = arith.constant 7 : i64
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sle, %[[VAL_11]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sle, %[[VAL_5]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_15:.*]] = arith.andi %[[VAL_13]], %[[VAL_14]] : i1
// CHECK:           cf.assert %[[VAL_15]], "Size constraint failed. Expected range: [0, 7]"
// CHECK:           return %[[VAL_4]] : !torch.int

func.func @torch.aten.sym_constrain_range(%arg0: !torch.vtensor<[],si64>) -> !torch.int {
    %int7 = torch.constant.int 7
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %0 = torch.aten.item %arg0 : !torch.vtensor<[],si64> -> !torch.int
    torch.aten.sym_constrain_range %0, %int0, %none : !torch.int, !torch.int, !torch.none
    torch.aten.sym_constrain_range %0, %int0, %int7 : !torch.int, !torch.int, !torch.int
    return %0 : !torch.int
}
