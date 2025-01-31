// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.sym_constrain_range(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !torch.int) -> !torch.int {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_i64 %[[VAL_0]]
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 7
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.none
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 9223372036854775807 : i64
// CHECK:           %[[VAL_7:.*]] = arith.cmpi sle, %[[VAL_5]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_9:.*]] = arith.andi %[[VAL_7]], %[[VAL_8]] : i1
// CHECK:           cf.assert %[[VAL_9]], "Size constraint failed. Expected range: [0, 9223372036854775807]"
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_11:.*]] = arith.constant 7 : i64
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sle, %[[VAL_10]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_11]] : i64
// CHECK:           %[[VAL_14:.*]] = arith.andi %[[VAL_12]], %[[VAL_13]] : i1
// CHECK:           cf.assert %[[VAL_14]], "Size constraint failed. Expected range: [0, 7]"
// CHECK:           return %[[VAL_0]] : !torch.int
// CHECK:         }
func.func @torch.aten.sym_constrain_range(%arg0: !torch.int) -> !torch.int {
  %int7 = torch.constant.int 7
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  torch.aten.sym_constrain_range %arg0, %int0, %none : !torch.int, !torch.int, !torch.none
  torch.aten.sym_constrain_range %arg0, %int0, %int7 : !torch.int, !torch.int, !torch.int
  return %arg0 : !torch.int
}
