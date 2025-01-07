// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// -----

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
// CHECK:           cf.assert %[[VAL_10]], "Invalid value range for size between [0, 9223372036854775807]"
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_12:.*]] = arith.constant 7 : i64
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sle, %[[VAL_11]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sle, %[[VAL_5]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_15:.*]] = arith.andi %[[VAL_13]], %[[VAL_14]] : i1
// CHECK:           cf.assert %[[VAL_15]], "Invalid value range for size between [0, 7]"
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

// -----

// CHECK-LABEL:   func.func @torch.aten._assert_scalar(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[],si64>) -> !torch.int {
// CHECK:           %[[VAL_1:.*]] = torch.constant.str "Runtime assertion failed for expression u0 <= 7 on node 'le_1'"
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 7
// CHECK:           %[[VAL_3:.*]] = torch.constant.str "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'"
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.aten.item %[[VAL_0]] : !torch.vtensor<[],si64> -> !torch.int
// CHECK:           %[[VAL_7:.*]] = torch_c.to_i64 %[[VAL_6]]
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_9:.*]] = arith.constant 9223372036854775807 : i64
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sle, %[[VAL_8]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sle, %[[VAL_7]], %[[VAL_9]] : i64
// CHECK:           %[[VAL_12:.*]] = arith.andi %[[VAL_10]], %[[VAL_11]] : i1
// CHECK:           cf.assert %[[VAL_12]], "Invalid value range for size between [0, 9223372036854775807]"
// CHECK:           %[[VAL_13:.*]] = torch.aten.ge.int %[[VAL_6]], %[[VAL_4]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[VAL_14:.*]] = torch.aten.Int.bool %[[VAL_13]] : !torch.bool -> !torch.int
// CHECK:           %[[VAL_15:.*]] = torch_c.to_i64 %[[VAL_14]]
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:           cf.assert %[[VAL_17]], "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'"
// CHECK:           %[[VAL_18:.*]] = torch.aten.le.int %[[VAL_6]], %[[VAL_2]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[VAL_19:.*]] = torch.aten.Int.bool %[[VAL_18]] : !torch.bool -> !torch.int
// CHECK:           %[[VAL_20:.*]] = torch_c.to_i64 %[[VAL_19]]
// CHECK:           %[[VAL_21:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_20]], %[[VAL_21]] : i64
// CHECK:           cf.assert %[[VAL_22]], "Runtime assertion failed for expression u0 <= 7 on node 'le_1'"
// CHECK:           return %[[VAL_6]] : !torch.int
func.func @torch.aten._assert_scalar(%arg0: !torch.vtensor<[],si64>) -> !torch.int {
  %str = torch.constant.str "Runtime assertion failed for expression u0 <= 7 on node 'le_1'"
  %int7 = torch.constant.int 7
  %str_0 = torch.constant.str "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'"
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.item %arg0 : !torch.vtensor<[],si64> -> !torch.int
  torch.aten.sym_constrain_range %0, %int0, %none : !torch.int, !torch.int, !torch.none
  %1 = torch.aten.ge.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
  %2 = torch.aten.Int.bool %1 : !torch.bool -> !torch.int
  torch.aten._assert_scalar %2, %str_0 : !torch.int, !torch.str
  %3 = torch.aten.le.int %0, %int7 : !torch.int, !torch.int -> !torch.bool
  %4 = torch.aten.Int.bool %3 : !torch.bool -> !torch.int
  torch.aten._assert_scalar %4, %str : !torch.int, !torch.str
  return %0 : !torch.int
}
