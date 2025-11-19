// RUN: torch-mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.__range_length$fold() -> (!torch.int, !torch.int, !torch.int, !torch.int) {
// CHECK-DAG:       %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:       %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:       %[[INT3:.*]] = torch.constant.int 3
// CHECK-DAG:       %[[INTM1:.*]] = torch.constant.int -1
// CHECK:           %[[NEG_STEP:.*]] = torch.aten.__range_length %[[INT1]], %[[INT3]], %[[INTM1]] : !torch.int, !torch.int, !torch.int -> !torch.int
// CHECK:           return %[[INT2]], %[[INT2]], %[[INT1]], %[[NEG_STEP]] : !torch.int, !torch.int, !torch.int, !torch.int
func.func @torch.aten.__range_length$fold() -> (!torch.int, !torch.int, !torch.int, !torch.int) {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %int-1 = torch.constant.int -1
  %0 = torch.aten.__range_length %int0, %int4, %int2  : !torch.int, !torch.int, !torch.int -> !torch.int
  %1 = torch.aten.__range_length %int1, %int4, %int2  : !torch.int, !torch.int, !torch.int -> !torch.int
  %2 = torch.aten.__range_length %int1, %int3, %int2  : !torch.int, !torch.int, !torch.int -> !torch.int
  %3 = torch.aten.__range_length %int1, %int3, %int-1  : !torch.int, !torch.int, !torch.int -> !torch.int
  return %0, %1, %2, %3 : !torch.int, !torch.int, !torch.int, !torch.int
}

// CHECK-LABEL:   func.func @torch.runtime.assert
// CHECK-NEXT:      return
func.func @torch.runtime.assert() {
  %true = torch.constant.bool true
  torch.runtime.assert %true, "msg"
  return
}

// CHECK-LABEL:   func.func @torch.aten.assert_tensor_metadata
// CHECK-NEXT:      return
func.func @torch.aten.assert_tensor_metadata() {
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %1 = tensor.empty() : tensor<1x1x128x128xi64>
  %2 = torch_c.from_builtin_tensor %1 : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
  torch.aten._assert_tensor_metadata %2, %none, %none, %int4, %none, %none : !torch.vtensor<[1,1,128,128],si64>, !torch.none, !torch.none, !torch.int, !torch.none, !torch.none
  return
}

// CHECK-LABEL:   func.func @torch.aten.ones_item
// CHECK:           %[[CONST:.*]] = torch.constant.int 1
// CHECK:           return %[[CONST]] : !torch.int
func.func @torch.aten.ones_item() -> !torch.int {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %none = torch.constant.none
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.ones %0, %int3, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si32>
    %2 = torch.aten.item %1 : !torch.vtensor<[1],si32> -> !torch.int
    return %2 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.zeros_item
// CHECK:           %[[CONST:.*]] = torch.constant.int 0
// CHECK:           return %[[CONST]] : !torch.int
func.func @torch.aten.zeros_item() -> !torch.int {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %none = torch.constant.none
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.zeros %0, %int3, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si32>
    %2 = torch.aten.item %1 : !torch.vtensor<[1],si32> -> !torch.int
    return %2 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.full_item
// CHECK:           %[[CONST:.*]] = torch.constant.int 1337
// CHECK:           return %[[CONST]] : !torch.int
func.func @torch.aten.full_item() -> !torch.int {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 1337
    %int5 = torch.constant.int 5
    %none = torch.constant.none
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.full %0, %int3, %int5, %none, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si32>
    %2 = torch.aten.item %1 : !torch.vtensor<[1],si32> -> !torch.int
    return %2 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.is_floating_point$fold_true
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.is_floating_point$fold_true(%arg0: !torch.vtensor<[], f32>) -> !torch.bool {
  %0 = torch.aten.is_floating_point %arg0 : !torch.vtensor<[], f32> -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.is_floating_point$fold_false
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.is_floating_point$fold_false(%arg0: !torch.vtensor<[], si64>) -> !torch.bool {
  %0 = torch.aten.is_floating_point %arg0 : !torch.vtensor<[], si64> -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__is__
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.__is__(%arg0: !torch.list<int>, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__is__ %arg0, %arg1 : !torch.list<int>, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__is__$derefine_is_none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.__is__$derefine_is_none(%arg0: !torch.list<int>, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.derefine %arg0 : !torch.list<int> to !torch.optional<list<int>>
  %1 = torch.aten.__is__ %0, %arg1 : !torch.optional<list<int>>, !torch.none -> !torch.bool
  return %1 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__is__$none_is_none
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.__is__$none_is_none(%arg0: !torch.none, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__is__ %arg0, %arg1 : !torch.none, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__is__$is_none$derefine(
// CHECK-SAME:                                             %{{.*}}: !torch.vtensor) -> !torch.bool {
// CHECK:           %[[RESULT:.*]] = torch.constant.bool false
// CHECK:           return %[[RESULT]] : !torch.bool
func.func @torch.aten.__is__$is_none$derefine(%arg0: !torch.vtensor) -> !torch.bool {
  %none = torch.constant.none
  %0 = torch.derefine %arg0 : !torch.vtensor to !torch.optional<vtensor>
  %1 = torch.aten.__is__ %0, %none : !torch.optional<vtensor>, !torch.none -> !torch.bool
  return %1 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__isnot__
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.__isnot__(%arg0: !torch.list<int>, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__isnot__ %arg0, %arg1 : !torch.list<int>, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__isnot__$none_isnot_none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.__isnot__$none_isnot_none(%arg0: !torch.none, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__isnot__ %arg0, %arg1 : !torch.none, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.bool$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.eq.bool$same_value() -> !torch.bool {
  %a = torch.constant.bool false
  %b = torch.constant.bool false
  %0 = torch.aten.eq.bool %a, %b: !torch.bool, !torch.bool -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.bool$different_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.eq.bool$different_value() -> !torch.bool {
  %a = torch.constant.bool true
  %b = torch.constant.bool false
  %0 = torch.aten.eq.bool %a, %b: !torch.bool, !torch.bool -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.bool$same_operand(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.bool) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.eq.bool$same_operand(%arg0: !torch.bool) -> !torch.bool {
  %0 = torch.aten.eq.bool %arg0, %arg0: !torch.bool, !torch.bool -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.bool$different_operand(
// CHECK-SAME:                                               %[[ARG0:.*]]: !torch.bool) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.eq.bool %[[ARG0]], %[[FALSE]] : !torch.bool, !torch.bool -> !torch.bool
// CHECK:           return %[[RET]] : !torch.bool
func.func @torch.aten.eq.bool$different_operand(%a: !torch.bool) -> !torch.bool {
  %b = torch.constant.bool false
  %0 = torch.aten.eq.bool %a, %b: !torch.bool, !torch.bool -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.bool() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.ne.bool() -> !torch.bool {
  %a = torch.constant.bool true
  %b = torch.constant.bool false
  %0 = torch.aten.ne.bool %a, %b: !torch.bool, !torch.bool -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.bool$same_operand(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.bool) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.ne.bool$same_operand(%arg0: !torch.bool) -> !torch.bool {
  %0 = torch.aten.ne.bool %arg0, %arg0: !torch.bool, !torch.bool -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.bool$different_operand(
// CHECK-SAME:                                               %[[ARG0:.*]]: !torch.bool) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.ne.bool %[[ARG0]], %[[FALSE]] : !torch.bool, !torch.bool -> !torch.bool
// CHECK:           return %[[RET]] : !torch.bool
func.func @torch.aten.ne.bool$different_operand(%a: !torch.bool) -> !torch.bool {
  %b = torch.constant.bool false
  %0 = torch.aten.ne.bool %a, %b: !torch.bool, !torch.bool -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.size$canonicalize_to_list(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.list<int> {
// CHECK:           %[[C2:.*]] = torch.constant.int 2
// CHECK:           %[[C3:.*]] = torch.constant.int 3
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C2]], %[[C3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[LIST]] : !torch.list<int>
func.func @torch.aten.size$canonicalize_to_list(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.list<int> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[2,3],f32> -> !torch.list<int>
  return %0 : !torch.list<int>
}

// One size unknown, so cannot canonicalize.
// TODO: For unknown sizes, insert the equivalent of a "dim" op.
// Then this will only require static rank.
// CHECK-LABEL:   func.func @torch.aten.size$unknown_size(
// CHECK-SAME:                                       %[[ARG:.*]]: !torch.vtensor<[?,3],f32>) -> !torch.list<int> {
// CHECK:           %[[SIZE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor<[?,3],f32> -> !torch.list<int>
func.func @torch.aten.size$unknown_size(%arg0: !torch.vtensor<[?,3],f32>) -> !torch.list<int> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[?,3],f32> -> !torch.list<int>
  return %0 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.ne.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK-NEXT:       %[[FALSE:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[FALSE]] : !torch.bool
func.func @torch.aten.ne.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.aten.ne.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.int$same_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.ne.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.ne.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.int$different_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.ne.int$different_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.ne.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int$different_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.eq.int$different_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.eq.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK-NEXT:       %[[F:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[F]] : !torch.bool
func.func @torch.aten.eq.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.aten.eq.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.eq.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.eq.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int$of_size.int(
// CHECK-SAME:                                        %[[ARG:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.eq.int$of_size.int(%arg0: !torch.tensor) -> !torch.bool {
  %int-1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.tensor, !torch.int -> !torch.int
  %1 = torch.aten.eq.int %0, %int-1 : !torch.int, !torch.int -> !torch.bool
  return %1 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int$of_size.int_lhs_constant(
// CHECK-SAME:                                                     %[[ARG:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.eq.int$of_size.int_lhs_constant(%arg0: !torch.tensor) -> !torch.bool {
  %int-1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.tensor, !torch.int -> !torch.int
  %1 = torch.aten.eq.int %int-1, %0  : !torch.int, !torch.int -> !torch.bool
  return %1 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int$no_change_minus1(
// CHECK-SAME:                                             %[[ARG:.*]]: !torch.int) -> !torch.bool {
// CHECK:           %[[CM1:.*]] = torch.constant.int -1
// CHECK:           %[[RESULT:.*]] = torch.aten.eq.int %[[CM1]], %[[ARG]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           return %[[RESULT]] : !torch.bool
func.func @torch.aten.eq.int$no_change_minus1(%arg0: !torch.int) -> !torch.bool {
  %int-1 = torch.constant.int -1
  %1 = torch.aten.eq.int %int-1, %arg0  : !torch.int, !torch.int -> !torch.bool
  return %1 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.lt.int$evaluate_to_true() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.lt.int$evaluate_to_true() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.lt.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.lt.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.lt.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %2 = torch.aten.lt.int %arg0, %arg0: !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.lt.int$same_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.lt.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.lt.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.le.int$evaluate_to_true() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.le.int$evaluate_to_true() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.le.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.le.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.le.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %2 = torch.aten.le.int %arg0, %arg0: !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.le.int$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.le.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.le.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.gt.int$evaluate_to_true() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[T]] : !torch.bool
func.func @torch.aten.gt.int$evaluate_to_true() -> !torch.bool {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.gt.int %int4, %int2 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.gt.int$evaluate_to_false() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[T]] : !torch.bool
func.func @torch.aten.gt.int$evaluate_to_false() -> !torch.bool {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.gt.int %int2, %int4 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ge.int$evaluate_to_false() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.ge.int$evaluate_to_false() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.ge.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ge.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.ge.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %2 = torch.aten.ge.int %arg0, %arg0: !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ge.int$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.ge.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.ge.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.lt.float$evaluate_to_true() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.lt.float$evaluate_to_true() -> !torch.bool {
  %float4 = torch.constant.float 4.0
  %float5 = torch.constant.float 5.0
  %2 = torch.aten.lt.float %float4, %float5 : !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.lt.float$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.float) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.lt.float$same_operand(%arg0: !torch.float) -> !torch.bool {
  %2 = torch.aten.lt.float %arg0, %arg0: !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.lt.float$same_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.lt.float$same_value() -> !torch.bool {
  %float4 = torch.constant.float 4.0
  %float4_0 = torch.constant.float 4.0
  %2 = torch.aten.lt.float %float4, %float4_0 : !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.gt.float$evaluate_to_true() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[T]] : !torch.bool
func.func @torch.aten.gt.float$evaluate_to_true() -> !torch.bool {
  %float2 = torch.constant.float 2.0
  %float4 = torch.constant.float 4.0
  %0 = torch.aten.gt.float %float4, %float2 : !torch.float, !torch.float -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.gt.float$evaluate_to_false() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[T]] : !torch.bool
func.func @torch.aten.gt.float$evaluate_to_false() -> !torch.bool {
  %float2 = torch.constant.float 2.0
  %float4 = torch.constant.float 4.0
  %0 = torch.aten.gt.float %float2, %float4 : !torch.float, !torch.float -> !torch.bool
  return %0 : !torch.bool
}


// CHECK-LABEL:   func.func @comparison_with_torch.aten.size.int(
// CHECK-SAME:                                              %[[ARG0:.*]]: !torch.vtensor<[?,2],unk>) -> (!torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool) {
// CHECK:           %[[SIZE:.*]] = torch.aten.size.int %[[ARG0]], %int0 : !torch.vtensor<[?,2],unk>, !torch.int -> !torch.int
// CHECK:           %[[GE_0_LHS:.*]] = torch.aten.ge.int %int0, %[[SIZE]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[LT_0_LHS:.*]] = torch.aten.lt.int %int0, %[[SIZE]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[EQ_0_LHS:.*]] = torch.aten.eq.int %int0, %[[SIZE]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[NE_0_LHS:.*]] = torch.aten.ne.int %int0, %[[SIZE]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[GT_0_RHS:.*]] = torch.aten.gt.int %[[SIZE]], %int0 : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[LE_0_RHS:.*]] = torch.aten.le.int %[[SIZE]], %int0 : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[EQ_0_RHS:.*]] = torch.aten.eq.int %[[SIZE]], %int0 : !torch.int, !torch.int -> !torch.bool
// CHECK:           %[[NE_0_RHS:.*]] = torch.aten.ne.int %[[SIZE]], %int0 : !torch.int, !torch.int -> !torch.bool
// CHECK:           return %true, %true, %false, %false, %[[GE_0_LHS]], %[[LT_0_LHS]], %[[EQ_0_LHS]], %[[NE_0_LHS]], %[[GT_0_RHS]], %[[LE_0_RHS]], %[[EQ_0_RHS]], %[[NE_0_RHS]] : !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool
func.func @comparison_with_torch.aten.size.int(%arg0: !torch.vtensor<[?,2],unk>) -> (!torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool) {
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,2],unk>, !torch.int -> !torch.int
  // Cases we can fold.
  %1 = torch.aten.le.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
  %2 = torch.aten.ge.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
  %3 = torch.aten.lt.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
  %4 = torch.aten.gt.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
  // Cases we cannot fold.
  %5 = torch.aten.ge.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
  %6 = torch.aten.lt.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
  %7 = torch.aten.eq.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
  %8 = torch.aten.ne.int %int0, %0 : !torch.int, !torch.int -> !torch.bool
  %9 = torch.aten.gt.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
  %10 = torch.aten.le.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
  %11 = torch.aten.eq.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
  %12 = torch.aten.ne.int %0, %int0 : !torch.int, !torch.int -> !torch.bool
  return %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12 : !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool, !torch.bool
}


// CHECK-LABEL:   func.func @torch.aten.eq.float$different_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.eq.float$different_value() -> !torch.bool {
  %float4 = torch.constant.float 4.0
  %float5 = torch.constant.float 5.0
  %2 = torch.aten.eq.float %float4, %float5 : !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.float$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.eq.float$same_value() -> !torch.bool {
  %float4 = torch.constant.float 4.0
  %float4_0 = torch.constant.float 4.0
  %2 = torch.aten.eq.float %float4, %float4_0 : !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.str$different_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.eq.str$different_value() -> !torch.bool {
  %str4 = torch.constant.str "4"
  %str5 = torch.constant.str "5"
  %2 = torch.aten.eq.str %str4, %str5 : !torch.str, !torch.str -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.str$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.str) -> !torch.bool {
// CHECK-NEXT:       %[[TRUE:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[TRUE]] : !torch.bool
func.func @torch.aten.eq.str$same_operand(%arg0: !torch.str) -> !torch.bool {
  %0 = torch.aten.eq.str %arg0, %arg0 : !torch.str, !torch.str -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.str$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.eq.str$same_value() -> !torch.bool {
  %str4 = torch.constant.str "4"
  %str4_0 = torch.constant.str "4"
  %2 = torch.aten.eq.str %str4, %str4_0 : !torch.str, !torch.str -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.str$different_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.ne.str$different_value() -> !torch.bool {
  %str4 = torch.constant.str "4"
  %str5 = torch.constant.str "5"
  %2 = torch.aten.ne.str %str4, %str5 : !torch.str, !torch.str -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.str$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.str) -> !torch.bool {
// CHECK-NEXT:       %[[FALSE:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[FALSE]] : !torch.bool
func.func @torch.aten.ne.str$same_operand(%arg0: !torch.str) -> !torch.bool {
  %0 = torch.aten.ne.str %arg0, %arg0 : !torch.str, !torch.str -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ne.str$same_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.ne.str$same_value() -> !torch.bool {
  %str4 = torch.constant.str "4"
  %str4_0 = torch.constant.str "4"
  %2 = torch.aten.ne.str %str4, %str4_0 : !torch.str, !torch.str -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.len.str() -> !torch.int {
// CHECK:           %[[INT7:.*]] = torch.constant.int 7
// CHECK:           return %[[INT7]] : !torch.int
func.func @torch.aten.len.str() -> !torch.int {
  %str = torch.constant.str "example"
  %2 = torch.aten.len.str %str : !torch.str -> !torch.int
  return %2 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.len.str$empty() -> !torch.int {
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           return %[[INT0]] : !torch.int
func.func @torch.aten.len.str$empty() -> !torch.int {
  %str = torch.constant.str ""
  %2 = torch.aten.len.str %str : !torch.str -> !torch.int
  return %2 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.__contains__.str_list$false() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.__contains__.str_list$false() -> !torch.bool {
  %str = torch.constant.str "c"
  %str_0 = torch.constant.str "b"
  %str_1 = torch.constant.str "a"
  %1 = torch.prim.ListConstruct %str_1, %str_0 : (!torch.str, !torch.str) -> !torch.list<str>
  %2 = torch.aten.__contains__.str_list %1, %str : !torch.list<str>, !torch.str -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__contains__.str_list$true() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.__contains__.str_list$true() -> !torch.bool {
  %str = torch.constant.str "aa"
  %str_0 = torch.constant.str "aa"
  %str_1 = torch.constant.str "ccc"
  %1 = torch.prim.ListConstruct %str_1, %str_0 : (!torch.str, !torch.str) -> !torch.list<str>
  %2 = torch.aten.__contains__.str_list %1, %str : !torch.list<str>, !torch.str -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__not__
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.__not__() -> !torch.bool {
  %false = torch.constant.bool false
  %ret = torch.aten.__not__ %false : !torch.bool -> !torch.bool
  return %ret: !torch.bool
}

// CHECK-LABEL:   func.func @torch.prim.max.int$identity(
// CHECK-SAME:                                      %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK:           return %[[ARG]] : !torch.int
func.func @torch.prim.max.int$identity(%arg0: !torch.int) -> !torch.int {
  %0 = torch.prim.max.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.max.int$constant() -> !torch.int {
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           return %[[INT3]] : !torch.int
func.func @torch.prim.max.int$constant() -> !torch.int {
  %int-1 = torch.constant.int -1
  %int3 = torch.constant.int 3
  %0 = torch.prim.max.int %int-1, %int3 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.min.int$identity(
// CHECK-SAME:                                      %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK:           return %[[ARG]] : !torch.int
func.func @torch.prim.min.int$identity(%arg0: !torch.int) -> !torch.int {
  %0 = torch.prim.min.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.min.int$constant() -> !torch.int {
// CHECK:           %[[INT1:.*]] = torch.constant.int -1
// CHECK:           return %[[INT1]] : !torch.int
func.func @torch.prim.min.int$constant() -> !torch.int {
  %int-1 = torch.constant.int -1
  %int3 = torch.constant.int 3
  %0 = torch.prim.min.int %int-1, %int3 : !torch.int, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.min.self_int$basic() -> !torch.int {
// CHECK:           %[[M1:.*]] = torch.constant.int -1
// CHECK:           return %[[M1]] : !torch.int
func.func @torch.prim.min.self_int$basic() -> !torch.int {
  %int-1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int-1, %int0, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.min.self_int %0 : !torch.list<int> -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.min.self_int$nofold$dynamic(
// CHECK:           torch.prim.min.self_int
func.func @torch.prim.min.self_int$nofold$dynamic(%arg0: !torch.int) -> !torch.int {
  %int-1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int-1, %int0, %arg0: (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.min.self_int %0 : !torch.list<int> -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.len.t$of_size(
// CHECK-SAME:                                   %[[ARG:.*]]: !torch.vtensor<*,f32>) -> !torch.int {
// CHECK:           %[[DIM:.*]] = torch.aten.dim %[[ARG]] : !torch.vtensor<*,f32> -> !torch.int
// CHECK:           return %[[DIM]] : !torch.int
func.func @torch.aten.len.t$of_size(%arg0: !torch.vtensor<*,f32>) -> !torch.int {
  %0 = torch.aten.size %arg0 : !torch.vtensor<*,f32> -> !torch.list<int>
  %1 = torch.aten.len.t %0 : !torch.list<int> -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.dim$with_shape(
// CHECK-SAME:                                    %[[ARG:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.int {
// CHECK:           %[[DIM:.*]] = torch.constant.int 3
// CHECK:           return %[[DIM]] : !torch.int
func.func @torch.aten.dim$with_shape(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.int {
  %0 = torch.aten.dim %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.len.t$of_build_list(
// CHECK-SAME:                                         %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK:           %[[LEN:.*]] = torch.constant.int 4
// CHECK:           return %[[LEN]] : !torch.int
func.func @torch.aten.len.t$of_build_list(%arg0: !torch.int) -> !torch.int {
  %0 = torch.prim.ListConstruct %arg0, %arg0, %arg0, %arg0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.len.t %0 : !torch.list<int> -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL: func.func @torch.aten.len.t$no_fold_list_mutated()
func.func @torch.aten.len.t$no_fold_list_mutated() -> !torch.int {
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.append.t %0, %int4 : !torch.list<int>, !torch.int -> !torch.list<int>
  // CHECK: torch.aten.len.t
  %2 = torch.aten.len.t %0 : !torch.list<int> -> !torch.int
  return %2 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.mul.left_t(
// CHECK:           %[[C4:.*]] = torch.constant.int 4
// CHECK:           %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C4]], %[[C5]], %[[C4]], %[[C5]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[LIST]] : !torch.list<int>
func.func @torch.aten.mul.left_t() -> !torch.list<int> {
    %int4 = torch.constant.int 4
    %int5 = torch.constant.int 5
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.mul.left_t %0, %int2 : !torch.list<int>, !torch.int -> !torch.list<int>
    return %1 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.__getitem__.t(
// CHECK:           %[[C5:.*]] = torch.constant.int 5
// CHECK:           return %[[C5]] : !torch.int
func.func @torch.aten.__getitem__.t() -> !torch.int {
    %int4 = torch.constant.int 4
    %int5 = torch.constant.int 5
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.__getitem__.t %0, %int1 : !torch.list<int>, !torch.int -> !torch.int
    return %1 : !torch.int
}

// Not canonicalized because of passed in index
// CHECK-LABEL:   func.func @torch.aten.__getitem__.t$no_change_test0(
// CHECK:           %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[C4:.*]] = torch.constant.int 4
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C4]], %[[C5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %[[LIST]], %arg0 : !torch.list<int>, !torch.int -> !torch.int
// CHECK:           return %[[ITEM]] : !torch.int
func.func @torch.aten.__getitem__.t$no_change_test0(%arg0: !torch.int) -> !torch.int {
  %int5 = torch.constant.int 5
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.__getitem__.t %0, %arg0 : !torch.list<int>, !torch.int -> !torch.int
  return %1 : !torch.int
}

// Not canonicalized because of passed in list
// CHECK-LABEL:   func.func @torch.aten.__getitem__.t$no_change_test1(
// CHECK:           %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %arg0, %[[C5]] : !torch.list<int>, !torch.int -> !torch.int
// CHECK:           return %[[ITEM]] : !torch.int
func.func @torch.aten.__getitem__.t$no_change_test1(%arg0: !torch.list<int>) -> !torch.int {
  %int5 = torch.constant.int 5
  %0 = torch.aten.__getitem__.t %arg0, %int5 : !torch.list<int>, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.__getitem__.t$getitem_of_size(
// CHECK-SAME:                                                   %[[TENSOR:.*]]: !torch.tensor,
// CHECK-SAME:                                                   %[[INDEX:.*]]: !torch.int) -> !torch.int {
// CHECK:           %[[RESULT:.*]] = torch.aten.size.int %[[TENSOR]], %[[INDEX]] : !torch.tensor, !torch.int -> !torch.int
// CHECK:           return %[[RESULT]] : !torch.int
func.func @torch.aten.__getitem__.t$getitem_of_size(%arg0: !torch.tensor, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.size %arg0 : !torch.tensor -> !torch.list<int>
  %1 = torch.aten.__getitem__.t %0, %arg1 : !torch.list<int>, !torch.int -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.__getitem__.t$negative_index() -> !torch.int {
// CHECK:           %[[INT8:.*]] = torch.constant.int 8
// CHECK:           return %[[INT8]] : !torch.int
func.func @torch.aten.__getitem__.t$negative_index() -> !torch.int {
  %int7 = torch.constant.int 7
  %int8 = torch.constant.int 8
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int7, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.__getitem__.t %0, %int-1 : !torch.list<int>, !torch.int -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.__getitem__.t$invalid_index() -> !torch.int {
func.func @torch.aten.__getitem__.t$invalid_index() -> !torch.int {
  %int7 = torch.constant.int 7
  %int8 = torch.constant.int 8
  %int-1 = torch.constant.int -100
  %0 = torch.prim.ListConstruct %int7, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.__getitem__.t
  %1 = torch.aten.__getitem__.t %0, %int-1 : !torch.list<int>, !torch.int -> !torch.int
  return %1 : !torch.int
}

// Not canonicalized because of mutated lhs list
// CHECK-LABEL: func.func @torch.aten.add.t$no_canonicalize_lhs_mutated()
func.func @torch.aten.add.t$no_canonicalize_lhs_mutated() -> !torch.list<int> {
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %2 = torch.aten.append.t %0, %int4 : !torch.list<int>, !torch.int -> !torch.list<int>
  // CHECK: torch.aten.add.t
  %3 = torch.aten.add.t %0, %1 : !torch.list<int>, !torch.list<int> -> !torch.list<int>
  return %3 : !torch.list<int>
}

// Not canonicalized because of mutated rhs list
// CHECK-LABEL: func.func @torch.aten.add.t$no_canonicalize_rhs_mutated()
func.func @torch.aten.add.t$no_canonicalize_rhs_mutated() -> !torch.list<int> {
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %2 = torch.aten.append.t %1, %int4 : !torch.list<int>, !torch.int -> !torch.list<int>
  // CHECK: torch.aten.add.t
  %3 = torch.aten.add.t %0, %1 : !torch.list<int>, !torch.list<int> -> !torch.list<int>
  return %3 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.add.t$concat(
// CHECK-SAME:           %[[ARG0:.*]]: !torch.int,
// CHECK-SAME:           %[[ARG1:.*]]: !torch.int) -> !torch.list<int> {
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[ARG0]], %[[ARG1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[LIST]] : !torch.list<int>
func.func @torch.aten.add.t$concat(%arg0: !torch.int, %arg1: !torch.int) -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %arg1 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.add.t %0, %1 : !torch.list<int>, !torch.list<int> -> !torch.list<int>
  return %2 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.add.t$concat_empty(
// CHECK-SAME:           %[[ARG0:.*]]: !torch.int) -> !torch.list<int> {
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[ARG0]] : (!torch.int) -> !torch.list<int>
// CHECK:           return %[[LIST]] : !torch.list<int>
func.func @torch.aten.add.t$concat_empty(%arg0: !torch.int) -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct : () -> !torch.list<int>
  %2 = torch.aten.add.t %0, %1 : !torch.list<int>, !torch.list<int> -> !torch.list<int>
  return %2 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.slice.t$basic() -> !torch.list<int> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[RET]] : !torch.list<int>
func.func @torch.aten.slice.t$basic() -> !torch.list<int> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.slice.t %0, %int0, %int-1, %int1 : !torch.list<int>, !torch.int, !torch.int, !torch.int -> !torch.list<int>
  return %2 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.slice.t$none_start() -> !torch.list<int> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[RET]] : !torch.list<int>
func.func @torch.aten.slice.t$none_start() -> !torch.list<int> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.slice.t %0, %none, %int-1, %int1 : !torch.list<int>, !torch.none, !torch.int, !torch.int -> !torch.list<int>
  return %2 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.slice.t$none_end() -> !torch.list<int> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[RET:.*]] = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[RET]] : !torch.list<int>
func.func @torch.aten.slice.t$none_end() -> !torch.list<int> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.slice.t %0, %int0, %none, %int1 : !torch.list<int>, !torch.int, !torch.none, !torch.int -> !torch.list<int>
  return %2 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.slice.t$start_exceed_range() -> !torch.list<int> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[RET]] : !torch.list<int>
func.func @torch.aten.slice.t$start_exceed_range() -> !torch.list<int> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %int-1000 = torch.constant.int -1000
  %0 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.slice.t %0, %int-1000, %int-1, %int1 : !torch.list<int>, !torch.int, !torch.int, !torch.int -> !torch.list<int>
  return %2 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.slice.t$end_exceed_range() -> !torch.list<int> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[RET:.*]] = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[RET]] : !torch.list<int>
func.func @torch.aten.slice.t$end_exceed_range() -> !torch.list<int> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %int1000 = torch.constant.int 1000
  %0 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.slice.t %0, %int0, %int1000, %int1 : !torch.list<int>, !torch.int, !torch.int, !torch.int -> !torch.list<int>
  return %2 : !torch.list<int>
}

// Not canonicalized because of mutated l list
// CHECK-LABEL: func.func @torch.aten.slice.t$no_canonicalize_l_mutated()
func.func @torch.aten.slice.t$no_canonicalize_l_mutated() -> !torch.list<int> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.slice.t
  %2 = torch.aten.slice.t %0, %int0, %int-1, %int1 : !torch.list<int>, !torch.int, !torch.int, !torch.int -> !torch.list<int>
  %3 = torch.aten.append.t %0, %int-1 : !torch.list<int>, !torch.int -> !torch.list<int>
  return %2 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.eq.int_list$fold$literals_of_different_sizes
// CHECK:           %[[RET:.*]] = torch.constant.bool false
// CHECK:           return %[[RET]] : !torch.bool
func.func @torch.aten.eq.int_list$fold$literals_of_different_sizes(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.prim.ListConstruct : () -> !torch.list<int>
  %1 = torch.prim.ListConstruct %arg0 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.eq.int_list %0, %1 : !torch.list<int>, !torch.list<int> -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int_list$fold$same_literal
// CHECK:           %[[RET:.*]] = torch.constant.bool true
// CHECK:           return %[[RET]] : !torch.bool
func.func @torch.aten.eq.int_list$fold$same_literal(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.prim.ListConstruct %arg0 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %arg0 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.eq.int_list %0, %1 : !torch.list<int>, !torch.list<int> -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.eq.int_list$no_fold$different_literals(
func.func @torch.aten.eq.int_list$no_fold$different_literals(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.prim.ListConstruct %arg0 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %arg1 : (!torch.int) -> !torch.list<int>
  // CHECK: torch.aten.eq.int_list
  %2 = torch.aten.eq.int_list %0, %1 : !torch.list<int>, !torch.list<int> -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.Float.Scalar$constant_fold_int_to_float() -> !torch.float {
// CHECK:           %[[VAL_0:.*]] = torch.constant.float 3.000000e+00
// CHECK:           return %[[VAL_0]] : !torch.float
func.func @torch.aten.Float.Scalar$constant_fold_int_to_float() -> !torch.float {
  %0 = torch.constant.int 3
  %1 = torch.aten.Float.Scalar %0 : !torch.int -> !torch.float
  return %1 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.Float.Scalar$identity(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.float) -> !torch.float {
// CHECK:           return %[[VAL_0]] : !torch.float
func.func @torch.aten.Float.Scalar$identity(%arg0: !torch.float) -> !torch.float {
  %0 = torch.aten.Float.Scalar %arg0 : !torch.float -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.constant.none$constantlike() -> (!torch.none, !torch.none) {
// CHECK:           %[[C:.*]] = torch.constant.none
// CHECK:           return %[[C]], %[[C]] : !torch.none, !torch.none
func.func @torch.constant.none$constantlike() -> (!torch.none, !torch.none) {
  %0 = torch.constant.none
  %1 = torch.constant.none
  return %0, %1 : !torch.none, !torch.none
}

// CHECK-LABEL:   func.func @torch.constant.str$constantlike() -> (!torch.str, !torch.str, !torch.str) {
// CHECK:           %[[S:.*]] = torch.constant.str "s"
// CHECK:           %[[T:.*]] = torch.constant.str "t"
// CHECK:           return %[[S]], %[[S]], %[[T]] : !torch.str, !torch.str, !torch.str
func.func @torch.constant.str$constantlike() -> (!torch.str, !torch.str, !torch.str) {
  %0 = torch.constant.str "s"
  %1 = torch.constant.str "s"
  %2 = torch.constant.str "t"
  return %0, %1, %2 : !torch.str, !torch.str, !torch.str
}

// CHECK-LABEL:   func.func @torch.constant.bool$constantlike() -> (!torch.bool, !torch.bool, !torch.bool) {
// CHECK:           %[[T:.*]] = torch.constant.bool true
// CHECK:           %[[F:.*]] = torch.constant.bool false
// CHECK:           return %[[T]], %[[T]], %[[F]] : !torch.bool, !torch.bool, !torch.bool
func.func @torch.constant.bool$constantlike() -> (!torch.bool, !torch.bool, !torch.bool) {
  %0 = torch.constant.bool true
  %1 = torch.constant.bool true
  %2 = torch.constant.bool false
  return %0, %1, %2 : !torch.bool, !torch.bool, !torch.bool
}

// CHECK-LABEL:   func.func @torch.prim.If$erase_dead_branch(
// CHECK-SAME:                                          %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK-NEXT:       %[[RET:.*]] = torch.aten.add.int %[[ARG]], %[[ARG]] : !torch.int, !torch.int -> !torch.int
// CHECK-NEXT:       return %[[RET]] : !torch.int
func.func @torch.prim.If$erase_dead_branch(%arg0: !torch.int) -> !torch.int {
  %true = torch.constant.bool true
  %0 = torch.prim.If %true -> (!torch.int) {
    %1 = torch.aten.add.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.int
    torch.prim.If.yield %1 : !torch.int
  } else {
    %1 = torch.aten.mul.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.int
    torch.prim.If.yield %1 : !torch.int
  }
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.If$no_fold$side_effect(
// CHECK-SAME:                                            %[[ARG0:.*]]: !torch.bool) {
// CHECK:           %[[STR:.*]] = torch.constant.str "str"
// CHECK:           torch.prim.If %[[ARG0]] -> () {
// CHECK:             torch.prim.RaiseException %[[STR]], %[[STR]] : !torch.str, !torch.str
// CHECK:             torch.prim.If.yield
// CHECK:           } else {
// CHECK:             torch.prim.If.yield
// CHECK:           }
// CHECK:           return
func.func @torch.prim.If$no_fold$side_effect(%arg0: !torch.bool) {
  %str = torch.constant.str "str"
  torch.prim.If %arg0 -> () {
    torch.prim.RaiseException %str, %str : !torch.str, !torch.str
    torch.prim.If.yield
  } else {
    torch.prim.If.yield
  }
  return
}

// CHECK-LABEL:   func.func @torch.prim.If$fold_same_result(
// CHECK-SAME:                                         %[[PRED:.*]]: !torch.bool,
// CHECK-SAME:                                         %[[ARG1:.*]]: !torch.int) -> (!torch.int, !torch.int) {
// CHECK-NEXT:      return %[[ARG1]], %[[ARG1]] : !torch.int, !torch.int
func.func @torch.prim.If$fold_same_result(%arg0: !torch.bool, %arg1: !torch.int) -> (!torch.int, !torch.int) {
  %0, %1 = torch.prim.If %arg0 -> (!torch.int, !torch.int) {
    torch.prim.If.yield %arg1, %arg1 : !torch.int, !torch.int
  } else {
    torch.prim.If.yield %arg1, %arg1 : !torch.int, !torch.int
  }
  return %0, %1: !torch.int, !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.If$fold_same_result$subset_of_results(
// CHECK-SAME:                                                           %[[PRED:.*]]: !torch.bool,
// CHECK-SAME:                                                           %[[ARG1:.*]]: !torch.int,
// CHECK-SAME:                                                           %[[ARG2:.*]]: !torch.int) -> (!torch.int, !torch.int) {
// CHECK:           %[[IF_RESULT:.*]] = torch.prim.If %[[PRED]] -> (!torch.int) {
// CHECK:             torch.prim.If.yield %[[ARG1]] : !torch.int
// CHECK:           } else {
// CHECK:             torch.prim.If.yield %[[ARG2]] : !torch.int
// CHECK:           }
// CHECK:           return %[[ARG1]], %[[IF_RESULT:.*]] : !torch.int, !torch.int
func.func @torch.prim.If$fold_same_result$subset_of_results(%arg0: !torch.bool, %arg1: !torch.int, %arg2: !torch.int) -> (!torch.int, !torch.int) {
  %0, %1 = torch.prim.If %arg0 -> (!torch.int, !torch.int) {
    torch.prim.If.yield %arg1, %arg1: !torch.int, !torch.int
  } else {
    torch.prim.If.yield %arg1, %arg2: !torch.int, !torch.int
  }
  return %0, %1: !torch.int, !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.TupleUnpack(
// CHECK-SAME:                                         %[[ARG0:.*]]: !torch.tensor,
// CHECK-SAME:                                         %[[ARG1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[ARG0]] : !torch.tensor
func.func @torch.prim.TupleUnpack(%arg0: !torch.tensor, %arg1: !torch.tensor) -> !torch.tensor{
  %123 = torch.prim.TupleConstruct %arg0, %arg1: !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
  %124:2 = torch.prim.TupleUnpack %123 : !torch.tuple<tensor, tensor> -> !torch.tensor, !torch.tensor
  return %124#0 : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.prim.TupleUnpack.Derefined(
// CHECK-SAME:                                         %[[ARG:.*]]: !torch.tensor) -> !torch.optional<tensor> {
// CHECK:           %[[DEREFINED:.+]] = torch.derefine %[[ARG]] : !torch.tensor to !torch.optional<tensor>
// CHECK:           return %[[DEREFINED]] : !torch.optional<tensor>
func.func @torch.prim.TupleUnpack.Derefined(%arg: !torch.tensor) -> !torch.optional<tensor> {
  %tuple = torch.prim.TupleConstruct %arg : !torch.tensor -> !torch.tuple<tensor>
  %optional_tensor = torch.prim.TupleUnpack %tuple : !torch.tuple<tensor> -> !torch.optional<tensor>
  return %optional_tensor : !torch.optional<tensor>
}

// CHECK-LABEL:   func.func @torch.aten.__contains__.str(
// CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
// CHECK-SAME:        %[[K1:.*]]: !torch.str,
// CHECK-SAME:        %[[V1:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
// CHECK-SAME:        keys(%[[K0]], %[[K1]] : !torch.str, !torch.str)
// CHECK-SAME:        values(%[[V0]], %[[V1]] : !torch.tensor, !torch.tensor)
// CHECK-SAME:        -> !torch.dict<str, tensor>
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.__contains__.str(%k0 : !torch.str, %v0: !torch.tensor, %k1: !torch.str, %v1: !torch.tensor) -> !torch.bool{
  %dict = torch.prim.DictConstruct keys(%k0, %k1: !torch.str, !torch.str) values(%v0,  %v1: !torch.tensor, !torch.tensor) -> !torch.dict<str, tensor>
  %pred = torch.aten.__contains__.str %dict, %k0 : !torch.dict<str, tensor>, !torch.str -> !torch.bool
  return %pred : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__contains__.str$with_dict_modified(
// CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
// CHECK-SAME:        %[[K1:.*]]: !torch.str, %[[V1:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
// CHECK-SAME:        keys(%[[K0]], %[[K1]] : !torch.str, !torch.str)
// CHECK-SAME:        values(%[[V0]], %[[V1]] : !torch.tensor, !torch.tensor)
// CHECK-SAME:        -> !torch.dict<str, tensor>
// CHECK:           torch.aten._set_item.str %[[DICT]], %[[K0]], %[[V1]] :
// CHECK-SAME:        !torch.dict<str, tensor>, !torch.str, !torch.tensor
// CHECK:           %[[RET:.*]] = torch.aten.__contains__.str %[[DICT]], %[[K0]] :
// CHECK-SAME:        !torch.dict<str, tensor>, !torch.str -> !torch.bool
// CHECK:           return %[[RET]] : !torch.bool

func.func @torch.aten.__contains__.str$with_dict_modified(%k0 : !torch.str, %v0: !torch.tensor, %k1: !torch.str, %v1: !torch.tensor) -> !torch.bool{
  %dict = torch.prim.DictConstruct keys(%k0, %k1: !torch.str, !torch.str) values(%v0,  %v1: !torch.tensor, !torch.tensor) -> !torch.dict<str, tensor>
  torch.aten._set_item.str %dict, %k0, %v1 : !torch.dict<str, tensor>, !torch.str, !torch.tensor
  %pred = torch.aten.__contains__.str %dict, %k0 : !torch.dict<str, tensor>, !torch.str -> !torch.bool
  return %pred : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.__getitem__.Dict_str(
// CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
// CHECK-SAME:        %[[K1:.*]]: !torch.str, %[[V1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
// CHECK-SAME:        keys(%[[K0]], %[[K1]] : !torch.str, !torch.str)
// CHECK-SAME:        values(%[[V0]], %[[V1]] : !torch.tensor, !torch.tensor)
// CHECK-SAME:        -> !torch.dict<str, tensor>
// CHECK:           return %[[V0]] : !torch.tensor
func.func @torch.aten.__getitem__.Dict_str(%k0 : !torch.str, %v0: !torch.tensor, %k1: !torch.str, %v1: !torch.tensor) -> !torch.tensor {
  %dict = torch.prim.DictConstruct keys(%k0, %k1: !torch.str, !torch.str) values(%v0,  %v1: !torch.tensor, !torch.tensor) -> !torch.dict<str, tensor>
  %v = torch.aten.__getitem__.Dict_str %dict, %k0 : !torch.dict<str, tensor>, !torch.str -> !torch.tensor
  return %v : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.aten.add.int() -> !torch.int {
// CHECK:           %[[CST9:.*]] = torch.constant.int 9
// CHECK:           return %[[CST9]] : !torch.int
func.func @torch.aten.add.int() -> !torch.int {
    %cst4 = torch.constant.int 4
    %cst5 = torch.constant.int 5
    %ret = torch.aten.add.int %cst4, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.add.float_int() -> !torch.float {
// CHECK:           %[[CST9:.*]] = torch.constant.float 9.000000e+00
// CHECK:           return %[[CST9]] : !torch.float
func.func @torch.aten.add.float_int() -> !torch.float {
    %cst4 = torch.constant.float 4.0
    %cst5 = torch.constant.int 5
    %ret = torch.aten.add.float_int %cst4, %cst5: !torch.float, !torch.int -> !torch.float
    return %ret : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.sub.int() -> !torch.int {
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           return %[[CST1]] : !torch.int
func.func @torch.aten.sub.int() -> !torch.int {
    %cst6 = torch.constant.int 6
    %cst5 = torch.constant.int 5
    %ret = torch.aten.sub.int %cst6, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.mul.int() -> !torch.int {
// CHECK:           %[[CST30:.*]] = torch.constant.int 30
// CHECK:           return %[[CST30]] : !torch.int
func.func @torch.aten.mul.int() -> !torch.int {
    %cst6 = torch.constant.int 6
    %cst5 = torch.constant.int 5
    %ret = torch.aten.mul.int %cst6, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.mul.int$canonicalize(
// CHECK-SAME:           %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK:           %[[CST30:.*]] = torch.constant.int 30
// CHECK:           %[[RET:.*]] = torch.aten.mul.int %[[ARG]], %[[CST30]] : !torch.int, !torch.int -> !torch.int
// CHECK:           return %[[RET]] : !torch.int
func.func @torch.aten.mul.int$canonicalize(%arg0: !torch.int) -> !torch.int {
    %cst6 = torch.constant.int 6
    %cst5 = torch.constant.int 5
    %1 = torch.aten.mul.int %arg0, %cst5: !torch.int, !torch.int -> !torch.int
    %ret = torch.aten.mul.int %1, %cst6: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.mul.int_float() -> !torch.float {
// CHECK:           %[[CST6:.*]] = torch.constant.float 6.000000e+00
// CHECK:           return %[[CST6]] : !torch.float
func.func @torch.aten.mul.int_float() -> !torch.float {
    %cst2 = torch.constant.int 2
    %cst3 = torch.constant.float 3.0
    %ret = torch.aten.mul.int_float %cst2, %cst3: !torch.int, !torch.float -> !torch.float
    return %ret : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.mul.float() -> !torch.float {
// CHECK:           %[[CST30:.*]] = torch.constant.float 3.000000e+01
// CHECK:           return %[[CST30]] : !torch.float
func.func @torch.aten.mul.float() -> !torch.float {
    %cst6 = torch.constant.float 6.0
    %cst5 = torch.constant.float 5.0
    %ret = torch.aten.mul.float %cst6, %cst5: !torch.float, !torch.float -> !torch.float
    return %ret : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.mul.float_int() -> !torch.float {
// CHECK:           %[[CST6:.*]] = torch.constant.float 6.000000e+00
// CHECK:           return %[[CST6]] : !torch.float
func.func @torch.aten.mul.float_int() -> !torch.float {
    %cst2 = torch.constant.float 2.0
    %cst3 = torch.constant.int 3
    %ret = torch.aten.mul.float_int %cst2, %cst3: !torch.float, !torch.int -> !torch.float
    return %ret : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.neg.float() -> !torch.float {
// CHECK:           %[[CST_6:.*]] = torch.constant.float -6.000000e+00
// CHECK:           return %[[CST_6]] : !torch.float
func.func @torch.aten.neg.float() -> !torch.float {
    %cst6 = torch.constant.float 6.0
    %ret = torch.aten.neg.float %cst6: !torch.float -> !torch.float
    return %ret : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.mul.int$with_zero() -> !torch.int {
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           return %[[CST0]] : !torch.int
func.func @torch.aten.mul.int$with_zero() -> !torch.int {
    %cst6 = torch.constant.int 6
    %cst0 = torch.constant.int 0
    %ret = torch.aten.mul.int %cst6, %cst0: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.floordiv.int() -> !torch.int {
// CHECK:           %[[CST3:.*]] = torch.constant.int 3
// CHECK:           return %[[CST3]] : !torch.int
func.func @torch.aten.floordiv.int() -> !torch.int {
    %cst18 = torch.constant.int 18
    %cst5 = torch.constant.int 5
    %ret = torch.aten.floordiv.int %cst18, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.floordiv.int$canonicalize(
// CHECK-SAME:           %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK:           return %[[ARG]] : !torch.int
func.func @torch.aten.floordiv.int$canonicalize(%arg0: !torch.int) -> !torch.int {
    %cst6 = torch.constant.int 6
    %1 = torch.aten.mul.int %arg0, %cst6: !torch.int, !torch.int -> !torch.int
    %ret = torch.aten.floordiv.int %1, %cst6: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.remainder.int() -> !torch.int {
// CHECK:           %[[CST3:.*]] = torch.constant.int 3
// CHECK:           return %[[CST3]] : !torch.int
func.func @torch.aten.remainder.int() -> !torch.int {
    %cst18 = torch.constant.int 18
    %cst5 = torch.constant.int 5
    %ret = torch.aten.remainder.int %cst18, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL: func.func @torch.aten.pow.Tensor_Tensor$canonicalize
// CHECK:         %[[SCALAR_EXP:.*]] = torch.constant.float 3.5
// CHECK:         %[[POW:.*]] = torch.aten.pow.Tensor_Scalar %arg0, %[[SCALAR_EXP]]
// CHECK:         return %[[POW]]
func.func @torch.aten.pow.Tensor_Tensor$canonicalize(%arg0 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %exponent = torch.vtensor.literal(dense<3.500000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
  %pow = torch.aten.pow.Tensor_Tensor %arg0, %exponent : !torch.vtensor<[?,?],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[?,?],f32>
  return %pow : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL: func.func @torch.aten.pow.Tensor_Scalar$canonicalize
// CHECK:         %[[INT_EXP:.*]] = torch.constant.int 3
// CHECK:         %[[POW:.*]] = torch.aten.pow.Tensor_Scalar %arg0, %[[INT_EXP]]
// CHECK:         return %[[POW]]
func.func @torch.aten.pow.Tensor_Scalar$canonicalize(%arg0 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %float_exponent = torch.constant.float 3.0
  %pow = torch.aten.pow.Tensor_Scalar %arg0, %float_exponent : !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
  return %pow : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:   func.func @torch.aten.pow.int_float() -> !torch.float {
// CHECK:           %[[FLOAT_8:.*]] = torch.constant.float 8.000000e+00
// CHECK:           return %[[FLOAT_8]] : !torch.float
func.func @torch.aten.pow.int_float() -> !torch.float {
    %cst2 = torch.constant.int 2
    %float3.0 = torch.constant.float 3.0
    %ret = torch.aten.pow.int_float %cst2, %float3.0: !torch.int, !torch.float -> !torch.float
    return %ret : !torch.float
}

// CHECK-LABEL:   func.func @torch.prim.dtype$bfloat16(
// CHECK-SAME:             %[[T:.*]]: !torch.tensor<*,bf16>) -> !torch.int {
// CHECK:           %[[CST:.*]] = torch.constant.int 15
// CHECK:           return %[[CST]] : !torch.int
func.func @torch.prim.dtype$bfloat16(%t : !torch.tensor<*,bf16>) -> !torch.int {
    %ret = torch.prim.dtype %t: !torch.tensor<*,bf16> -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.dtype$float(
// CHECK-SAME:             %[[T:.*]]: !torch.tensor<*,f32>) -> !torch.int {
// CHECK:           %[[CST:.*]] = torch.constant.int 6
// CHECK:           return %[[CST]] : !torch.int
func.func @torch.prim.dtype$float(%t : !torch.tensor<*,f32>) -> !torch.int {
    %ret = torch.prim.dtype %t: !torch.tensor<*,f32> -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.dtype$bool(
// CHECK-SAME:              %[[T:.*]]: !torch.tensor<*,i1>) -> !torch.int {
// CHECK:           %[[CST:.*]] = torch.constant.int 11
// CHECK:           return %[[CST]] : !torch.int
func.func @torch.prim.dtype$bool(%t : !torch.tensor<*,i1>) -> !torch.int {
    %ret = torch.prim.dtype %t: !torch.tensor<*,i1> -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.dtype$int64(
// CHECK-SAME:            %[[T:.*]]: !torch.tensor<*,si64>) -> !torch.int {
// CHECK:           %[[CST:.*]] = torch.constant.int 4
// CHECK:           return %[[CST]] : !torch.int
func.func @torch.prim.dtype$int64(%t : !torch.tensor<*,si64>) -> !torch.int {
    %ret = torch.prim.dtype %t: !torch.tensor<*,si64> -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.size.int$neg_dim(
// CHECK-SAME:            %[[T:.*]]: !torch.tensor<[2,3],f32>) -> !torch.int {
// CHECK:           %[[RET:.*]] = torch.constant.int 2
// CHECK:           return %[[RET]] : !torch.int
func.func @torch.aten.size.int$neg_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.int {
  %int-2 = torch.constant.int -2
  %ret = torch.aten.size.int %t, %int-2 : !torch.tensor<[2,3],f32>, !torch.int -> !torch.int
  return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.size.int$pos_dim(
// CHECK-SAME:            %[[T:.*]]: !torch.tensor<[2,3],f32>) -> !torch.int {
// CHECK:           %[[RET:.*]] = torch.constant.int 3
// CHECK:           return %[[RET]] : !torch.int
func.func @torch.aten.size.int$pos_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.int {
  %int1 = torch.constant.int 1
  %ret = torch.aten.size.int %t, %int1 : !torch.tensor<[2,3],f32>, !torch.int -> !torch.int
  return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.size.int$invalid_dim(
// CHECK-SAME:            %[[T:.*]]: !torch.tensor<[2,3],f32>) -> !torch.int {
// CHECK:           %[[CST3:.*]] = torch.constant.int 3
// CHECK:           %[[RET:.*]] = torch.aten.size.int %[[T]], %[[CST3]] : !torch.tensor<[2,3],f32>, !torch.int -> !torch.int
// CHECK:           return %[[RET]] : !torch.int
func.func @torch.aten.size.int$invalid_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.int {
  %int3 = torch.constant.int 3
  %ret = torch.aten.size.int %t, %int3 : !torch.tensor<[2,3],f32>, !torch.int -> !torch.int
  return %ret : !torch.int
}

// CHECK-LABEL:   func.func @torch.prim.unchecked_cast$derefine_identity(
// CHECK-SAME:                                                      %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK:           return %[[ARG]] : !torch.int
func.func @torch.prim.unchecked_cast$derefine_identity(%arg0: !torch.int) -> !torch.int {
  %0 = torch.derefine %arg0 : !torch.int to !torch.optional<int>
  %1 = torch.prim.unchecked_cast %0 : !torch.optional<int> -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.derefine$of_unchecked_cast(
// CHECK-SAME:                                           %[[ARG:.*]]: !torch.optional<int>) -> !torch.optional<int> {
// CHECK:           return %[[ARG]] : !torch.optional<int>
func.func @torch.derefine$of_unchecked_cast(%arg0: !torch.optional<int>) -> !torch.optional<int> {
  %0 = torch.prim.unchecked_cast %arg0 : !torch.optional<int> -> !torch.int
  %1 = torch.derefine %0 : !torch.int to !torch.optional<int>
  return %1 : !torch.optional<int>
}

// CHECK-LABEL:   func.func @torch.derefine$use_allows_type_refinement(
// CHECK-SAME:                                              %{{.*}}: !torch.int) -> (!torch.vtensor, !torch.optional<int>) {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[DEREFINED:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<int>
//                  For the use that allows type refinement, we replace it with the refined value.
// CHECK:           %[[ARANGE:.*]] = torch.aten.arange.start %{{.*}}, %{{.*}}, %[[NONE]], %{{.*}}, %{{.*}}, %{{.*}} : !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor
//                  For the use that does not allow type refinement, don't replace.
// CHECK:           return %[[ARANGE]], %[[DEREFINED]] : !torch.vtensor, !torch.optional<int>
func.func @torch.derefine$use_allows_type_refinement(%arg0: !torch.int) -> (!torch.vtensor, !torch.optional<int>) {
  %none = torch.constant.none
  %optional = torch.derefine %none : !torch.none to !torch.optional<int>
  %ret = torch.aten.arange.start %arg0, %arg0, %optional, %none, %none, %none: !torch.int, !torch.int, !torch.optional<int>, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %ret, %optional : !torch.vtensor, !torch.optional<int>
}


// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$downcast_first(
// CHECK-SAME:            %[[T:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[T]] : !torch.tensor
func.func @torch.tensor_static_info_cast$downcast_first(%t: !torch.tensor) -> !torch.tensor {
  %downcast = torch.tensor_static_info_cast %t : !torch.tensor to !torch.tensor<[?,?],f64>
  %upcast = torch.tensor_static_info_cast %downcast : !torch.tensor<[?,?],f64> to !torch.tensor
  return %upcast: !torch.tensor
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$upcast_first(
// CHECK-SAME:            %[[T:.*]]: !torch.tensor<[?,?],f64>) -> !torch.tensor<[?,?],f64> {
// CHECK:           return %[[T]] : !torch.tensor<[?,?],f64>
func.func @torch.tensor_static_info_cast$upcast_first(%t: !torch.tensor<[?,?],f64>) -> !torch.tensor<[?,?],f64> {
  %upcast = torch.tensor_static_info_cast %t : !torch.tensor<[?,?],f64> to !torch.tensor
  %downcast = torch.tensor_static_info_cast %upcast : !torch.tensor to !torch.tensor<[?,?],f64>
  return %downcast: !torch.tensor<[?,?],f64>
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$refine(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor {
// CHECK-NEXT:       %[[RESULT:.*]] = torch.aten.relu %[[ARG]] : !torch.vtensor<[],f32> -> !torch.vtensor
// CHECK-NEXT:       return %[[RESULT]] : !torch.vtensor
func.func @torch.tensor_static_info_cast$refine(%arg0: !torch.vtensor<[], f32>) -> !torch.vtensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[],f32> to !torch.vtensor
  %1 = torch.aten.relu %0 : !torch.vtensor -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$refine$dtype(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor {
// CHECK-NEXT:       %[[RESULT:.*]] = torch.aten.relu %[[ARG]] : !torch.vtensor<[],f32> -> !torch.vtensor
// CHECK-NEXT:       return %[[RESULT]] : !torch.vtensor
func.func @torch.tensor_static_info_cast$refine$dtype(%arg0: !torch.vtensor<[], f32>) -> !torch.vtensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[],f32> to !torch.vtensor<[],unk>
  %1 = torch.aten.relu %0 : !torch.vtensor<[],unk> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$refine$shape(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor {
// CHECK-NEXT:       %[[RESULT:.*]] = torch.aten.relu %[[ARG]] : !torch.vtensor<[],f32> -> !torch.vtensor
// CHECK-NEXT:       return %[[RESULT]] : !torch.vtensor
func.func @torch.tensor_static_info_cast$refine$shape(%arg0: !torch.vtensor<[], f32>) -> !torch.vtensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[],f32> to !torch.vtensor<*,f32>
  %1 = torch.aten.relu %0 : !torch.vtensor<*,f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$no_refine(
// CHECK-SAME:                                                  %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor to !torch.vtensor<[],f32>
// CHECK:           %[[RESULT:.*]] = torch.aten.relu %[[CAST]] : !torch.vtensor<[],f32> -> !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func.func @torch.tensor_static_info_cast$no_refine(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor to !torch.vtensor<[],f32>
  %1 = torch.aten.relu %0 : !torch.vtensor<[],f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$no_refine$dtype(
// CHECK-SAME:                                                  %[[ARG:.*]]: !torch.vtensor<[],unk>) -> !torch.vtensor {
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[],unk> to !torch.vtensor<[],f32>
// CHECK:           %[[RESULT:.*]] = torch.aten.relu %[[CAST]] : !torch.vtensor<[],f32> -> !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func.func @torch.tensor_static_info_cast$no_refine$dtype(%arg0: !torch.vtensor<[],unk>) -> !torch.vtensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[],unk> to !torch.vtensor<[],f32>
  %1 = torch.aten.relu %0 : !torch.vtensor<[],f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$no_refine$shape(
// CHECK-SAME:                                                  %[[ARG:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor {
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<*,f32> to !torch.vtensor<[],f32>
// CHECK:           %[[RESULT:.*]] = torch.aten.relu %[[CAST]] : !torch.vtensor<[],f32> -> !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func.func @torch.tensor_static_info_cast$no_refine$shape(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<*,f32> to !torch.vtensor<[],f32>
  %1 = torch.aten.relu %0 : !torch.vtensor<[],f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$refine_allowed_ops(
// CHECK-SAME:                                                           %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.tuple<vtensor, vtensor> {
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[],f32> to !torch.vtensor
// CHECK:           %[[RELU:.*]] = torch.aten.relu %[[ARG]] : !torch.vtensor<[],f32> -> !torch.vtensor
// CHECK:           %[[RESULT:.*]] = torch.prim.TupleConstruct %[[CAST]], %[[RELU]] : !torch.vtensor, !torch.vtensor -> !torch.tuple<vtensor, vtensor>
// CHECK:           return %[[RESULT]] : !torch.tuple<vtensor, vtensor>
func.func @torch.tensor_static_info_cast$refine_allowed_ops(%arg0: !torch.vtensor<[], f32>) -> !torch.tuple<vtensor, vtensor> {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[],f32> to !torch.vtensor
  %1 = torch.aten.relu %0 : !torch.vtensor -> !torch.vtensor
  // prim.TupleConstruct does not allow type refinements
  %2 = torch.prim.TupleConstruct %0, %1 : !torch.vtensor, !torch.vtensor -> !torch.tuple<vtensor, vtensor>
  return %2 : !torch.tuple<vtensor, vtensor>
}

// CHECK-LABEL:   func.func @torch.prim.TupleIndex(
// CHECK-SAME:            %[[T0:.*]]: !torch.tensor, %[[T1:.*]]: !torch.tensor, %[[T2:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[T1]] : !torch.tensor
func.func @torch.prim.TupleIndex(%t0: !torch.tensor, %t1: !torch.tensor, %t2: !torch.tensor) -> !torch.tensor {
    %0 = torch.prim.TupleConstruct %t0, %t1, %t2 : !torch.tensor, !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor, tensor>
    %int1 = torch.constant.int 1
    %1 = torch.prim.TupleIndex %0, %int1 : !torch.tuple<tensor, tensor, tensor>, !torch.int -> !torch.tensor
    return %1 : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.prim.TupleIndex$out_of_bound(
// CHECK-SAME:            %[[T0:.*]]: !torch.tensor, %[[T1:.*]]: !torch.tensor, %[[T2:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           %[[INDEX3:.*]] = torch.constant.int 3
// CHECK:           %[[TUPLE:.*]] = torch.prim.TupleConstruct %[[T0]], %[[T1]], %[[T2]] :
// CHECK-SAME:            !torch.tensor, !torch.tensor, !torch.tensor ->
// CHECK-SAME:            !torch.tuple<tensor, tensor, tensor>
// CHECK:           %[[RET:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[INDEX3]] :
// CHECK-SAME:            !torch.tuple<tensor, tensor, tensor>, !torch.int -> !torch.tensor
// CHECK:           return %[[RET]] : !torch.tensor
func.func @torch.prim.TupleIndex$out_of_bound(%t0: !torch.tensor, %t1: !torch.tensor, %t2: !torch.tensor) -> !torch.tensor {
    %0 = torch.prim.TupleConstruct %t0, %t1, %t2 : !torch.tensor, !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor, tensor>
    %int3 = torch.constant.int 3
    %1 = torch.prim.TupleIndex %0, %int3 : !torch.tuple<tensor, tensor, tensor>, !torch.int -> !torch.tensor
    return %1 : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.prim.TupleIndex$adjust_type$tensor(
// CHECK-SAME:                 %[[ARG:.*]]: !torch.tensor<[7],f32>) -> !torch.tensor {
// CHECK:           %[[RETURN:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.tensor<[7],f32> to !torch.tensor
// CHECK:           return %[[RETURN]] : !torch.tensor
func.func @torch.prim.TupleIndex$adjust_type$tensor(%arg0: !torch.tensor<[7],f32>) -> !torch.tensor {
  %int0 = torch.constant.int 0
  %0 = torch.prim.TupleConstruct %arg0 : !torch.tensor<[7],f32> -> !torch.tuple<tensor<[7],f32>>
  %1 = torch.prim.TupleIndex %0, %int0 : !torch.tuple<tensor<[7],f32>>, !torch.int -> !torch.tensor
  return %1 : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.prim.unchecked_cast$derefine
// CHECK-next:      return %arg0 : !torch.list<int>
func.func @torch.prim.unchecked_cast$derefine(%arg0: !torch.list<int>) -> !torch.list<int> {
  %0 = torch.derefine %arg0 : !torch.list<int> to !torch.optional<list<int>>
  %1 = torch.prim.unchecked_cast %0 : !torch.optional<list<int>> -> !torch.list<int>
  return %1 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.Int.Tensor(
// CHECK-SAME:            %[[NUM:.*]]: !torch.int) -> !torch.int {
// CHECK:           %[[T:.*]] = torch.prim.NumToTensor.Scalar %[[NUM]] : !torch.int -> !torch.vtensor<[],si64>
// CHECK:           return %[[NUM]] : !torch.int
func.func @torch.aten.Int.Tensor(%arg0: !torch.int) -> !torch.int {
  %tensor = torch.prim.NumToTensor.Scalar %arg0: !torch.int -> !torch.vtensor<[],si64>
  %scalar = torch.aten.Int.Tensor %tensor : !torch.vtensor<[],si64> -> !torch.int
  return %scalar : !torch.int
}

// CHECK-LABEL:  @torch.aten.Int.Tensor$canonicalize_0d_const() -> !torch.int {
// CHECK:           %[[NUM:.*]] = torch.constant.int 1
// CHECK:           return %[[NUM]] : !torch.int
func.func @torch.aten.Int.Tensor$canonicalize_0d_const() -> !torch.int {
  %cst = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
  %scalar = torch.aten.Int.Tensor %cst : !torch.vtensor<[],si64> -> !torch.int
  return %scalar : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.Int.float() -> !torch.int {
// CHECK:             %[[NUM:.*]] = torch.constant.int 1
// CHECK:             return %[[NUM]] : !torch.int
func.func @torch.aten.Int.float() -> !torch.int {
    %float1 = torch.constant.float 1.0
    %int1 = torch.aten.Int.float %float1 : !torch.float -> !torch.int
    return %int1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.Float.Tensor(
// CHECK-SAME:            %[[NUM:.*]]: !torch.float) -> !torch.float {
// CHECK:           %[[T:.*]] = torch.prim.NumToTensor.Scalar %[[NUM]] : !torch.float -> !torch.vtensor<[],f64>
// CHECK:           return %[[NUM]] : !torch.float
func.func @torch.aten.Float.Tensor(%arg0: !torch.float) -> !torch.float {
  %tensor = torch.prim.NumToTensor.Scalar %arg0: !torch.float -> !torch.vtensor<[],f64>
  %scalar = torch.aten.Float.Tensor %tensor : !torch.vtensor<[],f64> -> !torch.float
  return %scalar : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.squeeze$zero_rank(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.vtensor<[],f32>
func.func @torch.aten.squeeze$zero_rank(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.squeeze %arg0 : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim$zero_rank(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.vtensor<[],f32>
func.func @torch.aten.squeeze.dim$zero_rank(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.squeeze.dim %arg0, %int0 : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim$cst() -> !torch.vtensor<[2],si64> {
// CHECK-NEXT:       %[[CST:.+]] = torch.vtensor.literal(dense<[127, 128]> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
// CHECK-NEXT:       return %[[CST]]
func.func @torch.aten.squeeze.dim$cst() -> !torch.vtensor<[2],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense<[[127], [128]]> : tensor<2x1xsi64>) : !torch.vtensor<[2,1],si64>
  %1 = torch.aten.squeeze.dim %0, %int1 : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2],si64>
  return %1 : !torch.vtensor<[2],si64>
}

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim$cst_i1() -> !torch.vtensor<[3],i1> {
// CHECK-NEXT:       %[[CST:.+]] = torch.vtensor.literal(dense<[true, false, true]> : tensor<3xi1>) : !torch.vtensor<[3],i1>
// CHECK-NEXT:       return %[[CST]]
func.func @torch.aten.squeeze.dim$cst_i1() -> !torch.vtensor<[3],i1> {
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense<[[true], [false], [true]]> : tensor<3x1xi1>) : !torch.vtensor<[3,1],i1>
  %1 = torch.aten.squeeze.dim %0, %int1 : !torch.vtensor<[3,1],i1>, !torch.int -> !torch.vtensor<[3],i1>
  return %1 : !torch.vtensor<[3],i1>
}

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim$cst_splat_i1() -> !torch.vtensor<[3],i1> {
// CHECK-NEXT:       %[[CST:.+]] = torch.vtensor.literal(dense<true> : tensor<3xi1>) : !torch.vtensor<[3],i1>
// CHECK-NEXT:       return %[[CST]]
func.func @torch.aten.squeeze.dim$cst_splat_i1() -> !torch.vtensor<[3],i1> {
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense<true> : tensor<3x1xi1>) : !torch.vtensor<[3,1],i1>
  %1 = torch.aten.squeeze.dim %0, %int1 : !torch.vtensor<[3,1],i1>, !torch.int -> !torch.vtensor<[3],i1>
  return %1 : !torch.vtensor<[3],i1>
}

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim$same_shape(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[2,1],si64>) -> !torch.vtensor<[2,1],si64> {
// CHECK-NEXT:        return %[[ARG]]
func.func @torch.aten.squeeze.dim$same_shape(%arg0: !torch.vtensor<[2,1],si64>) -> !torch.vtensor<[2,1],si64> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.squeeze.dim %arg0, %int0 : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2,1],si64>
  return %0 : !torch.vtensor<[2,1],si64>
}

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim$not_fold
// CHECK:             torch.aten.squeeze.dim
func.func @torch.aten.squeeze.dim$not_fold(%arg0: !torch.vtensor<[2,1],si64>) -> !torch.vtensor<[2],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.squeeze.dim %arg0, %int1 : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// CHECK-LABEL:   func.func @torch.aten.tensor$one_elem(
// CHECK-NEXT: torch.vtensor.literal(dense<42> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
func.func @torch.aten.tensor$one_elem() -> (!torch.vtensor<[1],si64>) {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int42 = torch.constant.int 42
  %66 = torch.prim.ListConstruct %int42 : (!torch.int) -> !torch.list<int>
  %67 = torch.aten.tensor %66, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1],si64>
  return %67 : !torch.vtensor<[1],si64>
}

// CHECK-LABEL:   func.func @torch.aten.tensor$no_fold(
// CHECK:   torch.aten.tensor %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.tensor
func.func @torch.aten.tensor$no_fold(%arg0: !torch.tensor) -> (!torch.tensor) {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %1 = torch.aten.size %arg0 : !torch.tensor -> !torch.list<int>
  %2 = torch.aten.tensor %1, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.tensor
  return %2 : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.aten.tensor.float(
// CHECK-NEXT: torch.vtensor.literal(dense<1.000000e+01> : tensor<f32>) : !torch.vtensor<[],f32>
func.func @torch.aten.tensor.float() -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %float1.000000e01 = torch.constant.float 1.000000e+01
  %67 = torch.aten.tensor.float %float1.000000e01, %none, %none, %false : !torch.float, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
  return %67 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @torch.aten.tensor.int(
// CHECK-NEXT: torch.vtensor.literal(dense<45> : tensor<si32>) : !torch.vtensor<[],si32>
func.func @torch.aten.tensor.int() -> !torch.vtensor<[],si32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int45 = torch.constant.int 45
  %67 = torch.aten.tensor.int %int45, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],si32>
  return %67 : !torch.vtensor<[],si32>
}

// CHECK-LABEL:   func.func @torch.aten.to.dtype$same_dtype(
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<*,f32>) -> !torch.tensor<*,f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.tensor<*,f32>
func.func @torch.aten.to.dtype$same_dtype(%arg0: !torch.tensor<*,f32>) -> !torch.tensor<*,f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int6 = torch.constant.int 6
  %0 = torch.aten.to.dtype %arg0, %int6, %false, %false, %none : !torch.tensor<*,f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor<*,f32>
  return %0 : !torch.tensor<*,f32>
}

// CHECK-LABEL:   func.func @torch.aten.to.dtype$no_fold$unk_dtype(
// CHECK-SAME:                                        %[[ARG:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           %[[RESULT:.*]] = torch.aten.to.dtype %[[ARG]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.tensor, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
// CHECK:           return %[[RESULT]] : !torch.tensor
func.func @torch.aten.to.dtype$no_fold$unk_dtype(%arg0: !torch.tensor) -> !torch.tensor {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int6 = torch.constant.int 6
  %0 = torch.aten.to.dtype %arg0, %int6, %false, %false, %none : !torch.tensor, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   @torch.aten.to.dtype$fold_splat(
func.func @torch.aten.to.dtype$fold_splat() -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[4,4],si32>, !torch.vtensor<[10],si32>, !torch.vtensor<[5,5],f64>, !torch.vtensor<[3,3],f16>, !torch.vtensor<[2,2],bf16>, !torch.vtensor<[4],si64>, !torch.vtensor<[3],si16>, !torch.vtensor<[2],i1>, !torch.vtensor<[2],i1>) {
    // CHECK-NOT: torch.aten.to.dtype
	%false = torch.constant.bool false
	%none  = torch.constant.none

	// int32 splat → float32
	%int_splat = torch.vtensor.literal(dense<42> : tensor<2x3xsi32>) : !torch.vtensor<[2,3],si32>
	%int6 = torch.constant.int 6 // torch.float32
	// CHECK: %[[R1:.*]] = torch.vtensor.literal(dense<4.200000e+01> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
	%result1 = torch.aten.to.dtype %int_splat, %int6, %false, %false, %none
						: !torch.vtensor<[2,3],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[2,3],f32>

	// float32 splat → int32 (rmTowardZero)
	%float_splat = torch.vtensor.literal(dense<3.14159> : tensor<4x4xf32>) : !torch.vtensor<[4,4],f32>
	%int3 = torch.constant.int 3 // torch.int32
	// CHECK: %[[R2:.*]] = torch.vtensor.literal(dense<3> : tensor<4x4xsi32>) : !torch.vtensor<[4,4],si32>
	%result2 = torch.aten.to.dtype %float_splat, %int3, %false, %false, %none
						: !torch.vtensor<[4,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[4,4],si32>

	// int64 splat (max int32 + 1) → int32 (trunc)
	%int64_splat = torch.vtensor.literal(dense<2147483648> : tensor<10xsi64>) : !torch.vtensor<[10],si64>
	// CHECK: %[[R3:.*]] = torch.vtensor.literal(dense<-2147483648> : tensor<10xsi32>) : !torch.vtensor<[10],si32>
	%result3 = torch.aten.to.dtype %int64_splat, %int3, %false, %false, %none
						: !torch.vtensor<[10],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[10],si32>

	// float32 splat → float64
	%float32_splat = torch.vtensor.literal(dense<2.71828> : tensor<5x5xf32>) : !torch.vtensor<[5,5],f32>
	%int7 = torch.constant.int 7 // torch.float64
	// CHECK: %[[R4:.*]] = torch.vtensor.literal(dense<2.7182800769805908> : tensor<5x5xf64>) : !torch.vtensor<[5,5],f64>
	%result4 = torch.aten.to.dtype %float32_splat, %int7, %false, %false, %none
						: !torch.vtensor<[5,5],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[5,5],f64>

	// float64 splat → float16
	%float64_splat = torch.vtensor.literal(dense<1.2> : tensor<3x3xf64>) : !torch.vtensor<[3,3],f64>
	%int5 = torch.constant.int 5 // torch.float16
	// CHECK: %[[R5:.*]] = torch.vtensor.literal(dense<1.200200e+00> : tensor<3x3xf16>) : !torch.vtensor<[3,3],f16>
	%result5 = torch.aten.to.dtype %float64_splat, %int5, %false, %false, %none
						: !torch.vtensor<[3,3],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[3,3],f16>

	// float32 splat → bfloat16
	%float32_bf16 = torch.vtensor.literal(dense<-0.51> : tensor<2x2xf32>) : !torch.vtensor<[2,2],f32>
	%int15 = torch.constant.int 15 // torch.bfloat16
	// CHECK: %[[R6:.*]] = torch.vtensor.literal(dense<-5.117190e-01> : tensor<2x2xbf16>) : !torch.vtensor<[2,2],bf16>
	%result6 = torch.aten.to.dtype %float32_bf16, %int15, %false, %false, %none
						: !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[2,2],bf16>

	// int32 splat → int64 (sign-extend)
	%int32_ext = torch.vtensor.literal(dense<-1000> : tensor<4xsi32>) : !torch.vtensor<[4],si32>
	%int4 = torch.constant.int 4 // torch.int64
	// CHECK: %[[R7:.*]] = torch.vtensor.literal(dense<-1000> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
	%result7 = torch.aten.to.dtype %int32_ext, %int4, %false, %false, %none
						: !torch.vtensor<[4],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[4],si64>

	// int32 splat → int16 (trunc)
	%int32_trunc = torch.vtensor.literal(dense<32768> : tensor<3xsi32>) : !torch.vtensor<[3],si32>
	%int2 = torch.constant.int 2 // torch.int16
	// CHECK: %[[R8:.*]] = torch.vtensor.literal(dense<-32768> : tensor<3xsi16>) : !torch.vtensor<[3],si16>
	%result8 = torch.aten.to.dtype %int32_trunc, %int2, %false, %false, %none
						: !torch.vtensor<[3],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none
						-> !torch.vtensor<[3],si16>

    // int32 splat → bool (i1), non-zero
    %int40_splat = torch.vtensor.literal(dense<40> : tensor<2xsi32>) : !torch.vtensor<[2],si32>
    %int11 = torch.constant.int 11 // torch.bool
    // CHECK: %[[R9:.*]] = torch.vtensor.literal(dense<true> : tensor<2xi1>) : !torch.vtensor<[2],i1>
    %result9 = torch.aten.to.dtype %int40_splat, %int11, %false, %false, %none
                        : !torch.vtensor<[2],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none
                        -> !torch.vtensor<[2],i1>

    // float32 splat → bool (i1), zero
    %float_zero = torch.vtensor.literal(dense<0.0> : tensor<2xf32>) : !torch.vtensor<[2],f32>
    // CHECK: %[[R11:.*]] = torch.vtensor.literal(dense<false> : tensor<2xi1>) : !torch.vtensor<[2],i1>
    %result10 = torch.aten.to.dtype %float_zero, %int11, %false, %false, %none
                        : !torch.vtensor<[2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none
                        -> !torch.vtensor<[2],i1>

	return %result1, %result2, %result3, %result4, %result5, %result6, %result7, %result8, %result9, %result10
		: !torch.vtensor<[2,3],f32>, !torch.vtensor<[4,4],si32>, !torch.vtensor<[10],si32>, !torch.vtensor<[5,5],f64>, !torch.vtensor<[3,3],f16>, !torch.vtensor<[2,2],bf16>, !torch.vtensor<[4],si64>, !torch.vtensor<[3],si16>, !torch.vtensor<[2],i1>, !torch.vtensor<[2],i1>
}

// CHECK-LABEL: func.func @torch.aten.to.other$basic(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.tensor, %[[ARG_1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[CPU:.*]] = torch.constant.device "cpu"
// CHECK:         %[[VAR_0:.*]] = torch.prim.dtype %[[ARG_1]] : !torch.tensor -> !torch.int
// CHECK:         %[[VAR_1:.*]] = torch.aten.to.device %[[ARG_0]], %[[CPU]], %[[VAR_0]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.tensor, !torch.Device, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
// CHECK:         return %[[VAR_1]] : !torch.tensor
func.func @torch.aten.to.other$basic(%arg0 : !torch.tensor, %arg1 : !torch.tensor) -> !torch.tensor {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.to.other %arg0, %arg1, %false, %false, %none : !torch.tensor, !torch.tensor, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.aten.view$1D(
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<[?],f32>) -> !torch.tensor<[?],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.tensor<[?],f32>
func.func @torch.aten.view$1D(%arg0: !torch.tensor<[?],f32>) -> !torch.tensor<[?],f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.tensor<[?],f32>, !torch.list<int> -> !torch.tensor<[?],f32>
  return %1 : !torch.tensor<[?],f32>
}

// CHECK-LABEL: func.func @torch.aten.view$fold_splat(
// CHECK:         %[[SPLAT:.*]] = torch.vtensor.literal(dense<2> : tensor<2x4x1xsi64>) : !torch.vtensor<[2,4,1],si64>
// CHECK:         return %[[SPLAT]] : !torch.vtensor<[2,4,1],si64>
func.func @torch.aten.view$fold_splat() -> !torch.vtensor<[2,4,1],si64> {
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense<2> : tensor<8xsi64>) : !torch.vtensor<[8],si64>
  %1 = torch.prim.ListConstruct %int2, %int4, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.view %0, %1 : !torch.vtensor<[8],si64>, !torch.list<int> -> !torch.vtensor<[2,4,1],si64>
  return %2 : !torch.vtensor<[2,4,1],si64>
}

// CHECK-LABEL: func.func @torch.aten.view$fold_literal(
// CHECK:         %[[LITERAL:.*]] = torch.vtensor.literal(dense<[
// CHECK-SAME:                                                   [
// CHECK-SAME:                                                    [0, 1], [2, 3], [4, 5], [6, 7]]]> : tensor<1x4x2xsi64>) : !torch.vtensor<[1,4,2],si64>
// CHECK:         return %[[LITERAL]] : !torch.vtensor<[1,4,2],si64>
func.func @torch.aten.view$fold_literal() -> !torch.vtensor<[1,4,2],si64> {
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense<[0,1,2,3,4,5,6,7]> : tensor<8xsi64>) : !torch.vtensor<[8],si64>
  %1 = torch.prim.ListConstruct %int1, %int4, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.view %0, %1 : !torch.vtensor<[8],si64>, !torch.list<int> -> !torch.vtensor<[1,4,2],si64>
  return %2 : !torch.vtensor<[1,4,2],si64>
}

// CHECK-LABEL:  func.func @torch.aten.transpose.int$fold_literal(
// CHECK:    %[[LIT:.*]] = torch.vtensor.literal(dense<[
// CHECK-SAME:                                          [0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xsi64>) : !torch.vtensor<[2,4],si64>
// CHECK:    return %[[LIT]] : !torch.vtensor<[2,4],si64>
func.func @torch.aten.transpose.int$fold_literal() -> !torch.vtensor<[2,4],si64> {
  %int-1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %0 = torch.vtensor.literal(dense<[[0,1],[2,3],[4,5],[6,7]]> : tensor<4x2xsi64>) : !torch.vtensor<[4,2],si64>
  %1 = torch.aten.transpose.int %0, %int-1, %int0 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4], si64>
  return %1 : !torch.vtensor<[2,4],si64>
}

// CHECK-LABEL:  func.func @torch.aten.transpose.int$fold_noop(
// CHECK:    return %arg0 : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.transpose.int$fold_noop(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int-1 = torch.constant.int -1
  %int3 = torch.constant.int 3
  %0 = torch.aten.transpose.int %arg0, %int-1, %int3 : !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.slice.Tensor$flip_slice_fold(
// CHECK:    %[[LIT:.*]] = torch.vtensor.literal(dense<[
// CHECK-SAME:                                          [6, 7], [4, 5], [2, 3], [0, 1]]> : tensor<4x2xsi64>) : !torch.vtensor<[4,2],si64>
// CHECK:    return %[[LIT]] : !torch.vtensor<[4,2],si64>
func.func @torch.aten.slice.Tensor$flip_slice_fold() -> !torch.vtensor<[4,2],si64> {
    %int-9223372036854775807 = torch.constant.int -9223372036854775807
    %int-1 = torch.constant.int -1
    %int0 = torch.constant.int 0
    %0 = torch.vtensor.literal(dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xsi64>) : !torch.vtensor<[4,2],si64>
    %1 = torch.aten.slice.Tensor %0, %int0, %int-1, %int-9223372036854775807, %int-1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,2],si64>
    return %1 : !torch.vtensor<[4,2],si64>
}

// CHECK-LABEL:  func.func @torch.aten.slice.Tensor$negative_two_stride_fold(
// CHECK:    %[[LIT:.*]] = torch.vtensor.literal(dense<[
// CHECK-SAME:                                          [6, 7], [2, 3]]> : tensor<2x2xsi64>) : !torch.vtensor<[2,2],si64>
// CHECK:    return %[[LIT]] : !torch.vtensor<[2,2],si64>
func.func @torch.aten.slice.Tensor$negative_two_stride_fold() -> !torch.vtensor<[2,2],si64> {
    %int-5 = torch.constant.int -5
    %int-1 = torch.constant.int -1
    %int-2 = torch.constant.int -2
    %int0 = torch.constant.int 0
    %0 = torch.vtensor.literal(dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xsi64>) : !torch.vtensor<[4,2],si64>
    %1 = torch.aten.slice.Tensor %0, %int0, %int-1, %int-5, %int-2 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,2],si64>
    return %1 : !torch.vtensor<[2,2],si64>
}

// CHECK-LABEL:   func.func @torch.aten.div.float$fold_zero_dividend(
// CHECK:           %[[CST0:.*]] = torch.constant.float 0.000000e+00
// CHECK:           return %[[CST0]] : !torch.float
func.func @torch.aten.div.float$fold_zero_dividend() -> !torch.float {
  %float0 = torch.constant.float 0.0
  %float5 = torch.constant.float 5.0
  %0 = torch.aten.div.float %float0, %float5 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.div.float$fold_one_divisor(
// CHECK:           %[[CST4:.*]] = torch.constant.float 4.000000e+00
// CHECK:           return %[[CST4]] : !torch.float
func.func @torch.aten.div.float$fold_one_divisor() -> !torch.float {
  %float4 = torch.constant.float 4.0
  %float1 = torch.constant.float 1.0
  %0 = torch.aten.div.float %float4, %float1 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.div.float$fold_cst_operands(
// CHECK:           %[[CST2:.*]] = torch.constant.float 2.000000e+00
// CHECK:           return %[[CST2]] : !torch.float
func.func @torch.aten.div.float$fold_cst_operands() -> !torch.float {
  %float4 = torch.constant.float 4.0
  %float2 = torch.constant.float 2.0
  %0 = torch.aten.div.float %float4, %float2 : !torch.float, !torch.float -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.div.int$fold_cst_operands(
// CHECK:           %[[CST:.*]] = torch.constant.float 5.000000e-01
// CHECK:           return %[[CST]] : !torch.float
func.func @torch.aten.div.int$fold_cst_operands() -> !torch.float {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.div.int %int2, %int4 : !torch.int, !torch.int -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.to.dtype_layout$same_dtype(
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.tensor<[?,?],f32>
func.func @torch.aten.to.dtype_layout$same_dtype(%arg0: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int6 = torch.constant.int 6
  %0 = torch.aten.to.dtype_layout %arg0, %int6, %none, %none, %none, %false, %false, %none : !torch.tensor<[?,?],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.bool, !torch.none -> !torch.tensor<[?,?],f32>
  return %0 : !torch.tensor<[?,?],f32>
}

// CHECK-LABEL:   func.func @torch.aten.to.dtype_layout$to_device(
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f32> {
// CHECK-NEXT:      %[[INT6:.*]] = torch.constant.int 6
// CHECK-NEXT:      %[[FALSE:.*]] = torch.constant.bool false
// CHECK-NEXT:      %[[NONE:.*]] = torch.constant.none
// CHECK-NEXT:      %[[CPU:.*]] = torch.constant.device "cpu"
// CHECK-NEXT:      %[[RESULT:.*]] = torch.aten.to.device %[[ARG]], %[[CPU]], %[[INT6]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.tensor<[?,?],f32>, !torch.Device, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor<[?,?],f32>
// CHECK-NEXT:      return %[[RESULT]] : !torch.tensor<[?,?],f32>
func.func @torch.aten.to.dtype_layout$to_device(%arg0: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f32> {
  %none = torch.constant.none
  %device = torch.constant.device "cpu"
  %false = torch.constant.bool false
  %int6 = torch.constant.int 6
  %0 = torch.aten.to.dtype_layout %arg0, %int6, %none, %device, %none, %false, %false, %none : !torch.tensor<[?,?],f32>, !torch.int, !torch.none, !torch.Device, !torch.none, !torch.bool, !torch.bool, !torch.none -> !torch.tensor<[?,?],f32>
  return %0 : !torch.tensor<[?,?],f32>
}

// CHECK-LABEL:   func.func @torch.aten.to.dtype_layout$to_dtype(
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f16> {
// CHECK-NEXT:      %[[NONE:.*]] = torch.constant.none
// CHECK-NEXT:      %[[FALSE:.*]] = torch.constant.bool false
// CHECK-NEXT:      %[[INT5:.*]] = torch.constant.int 5
// CHECK-NEXT:      %[[RESULT:.*]] = torch.aten.to.dtype %[[ARG]], %[[INT5]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.tensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor<[?,?],f16>
// CHECK-NEXT:      return %[[RESULT]] : !torch.tensor<[?,?],f16>
func.func @torch.aten.to.dtype_layout$to_dtype(%arg0: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f16> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int5 = torch.constant.int 5
  %0 = torch.aten.to.dtype_layout %arg0, %int5, %none, %none, %none, %false, %false, %none : !torch.tensor<[?,?],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.bool, !torch.none -> !torch.tensor<[?,?],f16>
  return %0 : !torch.tensor<[?,?],f16>
}

// CHECK-LABEL:   func.func @torch.aten.ge.float$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.float) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.ge.float$same_operand(%arg0: !torch.float) -> !torch.bool {
  %2 = torch.aten.ge.float %arg0, %arg0: !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ge.float$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func.func @torch.aten.ge.float$same_value() -> !torch.bool {
  %float4 = torch.constant.float 4.0
  %float4_0 = torch.constant.float 4.0
  %2 = torch.aten.ge.float %float4, %float4_0: !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ge.float$different_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func.func @torch.aten.ge.float$different_value() -> !torch.bool {
  %float4 = torch.constant.float 4.0
  %float4_0 = torch.constant.float 5.0
  %2 = torch.aten.ge.float %float4, %float4_0: !torch.float, !torch.float -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.ceil.float$fold_cst() -> !torch.int {
// CHECK:           %[[CST2:.*]] = torch.constant.int 2
// CHECK:           return %[[CST2]] : !torch.int
func.func @torch.aten.ceil.float$fold_cst() -> !torch.int {
  %float = torch.constant.float 1.5
  %1 = torch.aten.ceil.float %float : !torch.float -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.ceil.float$no_fold(
// CHECK-SAME:            %[[ARG:.*]]: !torch.float) -> !torch.int {
// CHECK:           %[[RESULT:.*]] = torch.aten.ceil.float %[[ARG]] : !torch.float -> !torch.int
// CHECK:           return %[[RESULT]] : !torch.int
func.func @torch.aten.ceil.float$no_fold(%arg0 : !torch.float) -> !torch.int {
  %1 = torch.aten.ceil.float %arg0 : !torch.float -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.sqrt.int$fold_cst() -> !torch.float {
// CHECK:           %[[CST:.*]] = torch.constant.float 2.2360679774997898
// CHECK:           return %[[CST]] : !torch.float
func.func @torch.aten.sqrt.int$fold_cst() -> !torch.float {
  %int = torch.constant.int 5
  %0 = torch.aten.sqrt.int %int : !torch.int -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.sqrt.int$no_fold(
// CHECK-SAME:            %[[ARG:.*]]: !torch.int) -> !torch.float {
// CHECK:           %[[RESULT:.*]] = torch.aten.sqrt.int %[[ARG]] : !torch.int -> !torch.float
// CHECK:           return %[[RESULT]] : !torch.float
func.func @torch.aten.sqrt.int$no_fold(%arg0 : !torch.int) -> !torch.float {
  %0 = torch.aten.sqrt.int %arg0 : !torch.int -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.Bool.float$fold_cst() -> !torch.bool {
// CHECK:           %[[CST2:.*]] = torch.constant.bool true
// CHECK:           return %[[CST2]] : !torch.bool
func.func @torch.aten.Bool.float$fold_cst() -> !torch.bool {
  %float = torch.constant.float 1.5
  %1 = torch.aten.Bool.float %float : !torch.float -> !torch.bool
  return %1 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.Bool.int$fold_cst() -> !torch.bool {
// CHECK:           %[[CST2:.*]] = torch.constant.bool true
// CHECK:           return %[[CST2]] : !torch.bool
func.func @torch.aten.Bool.int$fold_cst() -> !torch.bool {
  %int = torch.constant.int 2
  %1 = torch.aten.Bool.int %int : !torch.int -> !torch.bool
  return %1 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.add.Tensor$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:      %[[CST:.*]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:      return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.add.Tensor$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.NumToTensor.Scalar %int0 : !torch.int -> !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.add.Tensor %0, %1, %int3 : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   @torch.aten.add.Tensor$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:      %[[CST:.*]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:      return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.add.Tensor$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.add.Tensor %0, %1, %int3 : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   @torch.aten.size$copy(
// CHECK-SAME:        %[[ARG:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.list<int> {
// CHECK:           %[[TWO:.*]] = torch.constant.int 2
// CHECK:           %[[THREE:.*]] = torch.constant.int 3
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[TWO]], %[[THREE]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[LIST]] : !torch.list<int>
// CHECK:         }
func.func @torch.aten.size$copy(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.list<int> {
    %cast = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3],f32> to !torch.vtensor
    %non_value_tensor = torch.copy.to_tensor %cast : !torch.tensor
    %value_tensor = torch.copy.to_vtensor %non_value_tensor : !torch.vtensor
    %size = torch.aten.size %value_tensor : !torch.vtensor -> !torch.list<int>
    return %size : !torch.list<int>
}

// CHECK-LABEL:   @torch.aten.size.int$copy(
// CHECK-SAME:        %[[ARG:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.int {
// CHECK:           %[[TWO:.*]] = torch.constant.int 2
// CHECK:           return %[[TWO]] : !torch.int
// CHECK:         }
func.func @torch.aten.size.int$copy(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.int {
  %cast = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3],f32> to !torch.vtensor
  %non_value_tensor = torch.copy.to_tensor %cast : !torch.tensor
  %value_tensor = torch.copy.to_vtensor %non_value_tensor : !torch.vtensor
  %zero = torch.constant.int 0
  %size = torch.aten.size.int %value_tensor, %zero : !torch.vtensor, !torch.int -> !torch.int
  return %size : !torch.int
}

// CHECK-LABEL:   func.func @prim.ListUnpack$fold_list(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>,
// CHECK-SAME:        %[[ARG1:.*]]: !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[2,3],f32>,  !torch.vtensor<[2,3],f32>) {
// CHECK:           return %[[ARG0]], %[[ARG1]] : !torch.vtensor<[2,3],f32>,  !torch.vtensor<[2,3],f32>
func.func @prim.ListUnpack$fold_list(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>) {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>) -> !torch.list<vtensor>
  %1:2 = torch.prim.ListUnpack %0 : !torch.list<vtensor> -> !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>
  return %1#0, %1#1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL:   func.func @prim.ListUnpack$fold_list_cast(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>,
// CHECK-SAME:        %[[ARG1:.*]]: !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[?,?],f32>,  !torch.vtensor<[?,?],f32>) {
// CHECK:           %[[CAST0:.+]] = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3],f32> to !torch.vtensor<[?,?],f32>
// CHECK:           %[[CAST1:.+]] = torch.tensor_static_info_cast %arg1 : !torch.vtensor<[2,3],f32> to !torch.vtensor<[?,?],f32>
// CHECK:           return %[[CAST0]], %[[CAST1]] : !torch.vtensor<[?,?],f32>,  !torch.vtensor<[?,?],f32>
func.func @prim.ListUnpack$fold_list_cast(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>) -> !torch.list<vtensor>
  %1:2 = torch.prim.ListUnpack %0 : !torch.list<vtensor> -> !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>
  return %1#0, %1#1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.*]] = torch.vtensor.literal(dense<3> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.div.Tensor_mode$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %int6 = torch.constant.int 6
    %str = torch.constant.str "floor"
    %0 = torch.vtensor.literal(dense<2> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int6 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.div.Tensor_mode %1, %0, %str : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.str -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<3> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.div.Tensor_mode$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int6 = torch.constant.int 6
    %int2 = torch.constant.int 2
    %str = torch.constant.str "floor"
    %0 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int6 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.div.Tensor_mode %1, %0, %str : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.str -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.add.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.add.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.NumToTensor.Scalar %int0 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.add.Scalar %0, %int2, %int3 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.add.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.add.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %2 = torch.aten.add.Scalar %0, %int2, %int3 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.sub.Tensor$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:      %[[CST:.+]] = torch.vtensor.literal(dense<-6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:      return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.sub.Tensor$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.NumToTensor.Scalar %int0 : !torch.int -> !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.sub.Tensor %0, %1, %int3 : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   @torch.aten.sub.Tensor$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:       %[[CST:.+]] = torch.vtensor.literal(dense<-6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:       return %[[CST]]
func.func @torch.aten.sub.Tensor$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.sub.Tensor %0, %1, %int3 : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.sub.Tensor$mixed_dtype() -> !torch.vtensor<[],f64> {
// CHECK:       %[[CST:.*]] = torch.vtensor.literal(dense<2.750000e+01> : tensor<f64>) : !torch.vtensor<[],f64>
// CEHCK:       return %[[CST]]
func.func @torch.aten.sub.Tensor$mixed_dtype() -> !torch.vtensor<[],f64> {
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense<28> : tensor<si64>) : !torch.vtensor<[],si64>
  %1 = torch.vtensor.literal(dense<5.000000e-01> : tensor<f64>) : !torch.vtensor<[],f64>
  %2 = torch.aten.sub.Tensor %0, %1, %int1 : !torch.vtensor<[],si64>, !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor<[],f64>
  return %2 : !torch.vtensor<[],f64>
}

// CHECK-LABEL:   func.func @torch.aten.sub.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<-6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.sub.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.NumToTensor.Scalar %int0 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.sub.Scalar %0, %int2, %int3 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.sub.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<-6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.sub.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %2 = torch.aten.sub.Scalar %0, %int2, %int3 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.sub.float$fold() -> !torch.float {
// CHECK:             %[[FLOAT_1:.*]] = torch.constant.float -1.000000e+00
// CHECK:             return %[[FLOAT_1]] : !torch.float
func.func @torch.aten.sub.float$fold() -> !torch.float {
    %float1 = torch.constant.float 1.0
    %float2 = torch.constant.float 2.0
    %0 = torch.aten.sub.float %float1, %float2 : !torch.float, !torch.float -> !torch.float
    return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.mul.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.mul.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %int3 = torch.constant.int 3
    %0 = torch.vtensor.literal(dense<2> : tensor<si64>) : !torch.vtensor<[],si64>
    %2 = torch.aten.mul.Scalar %0, %int3 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.mul.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.mul.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.mul.Scalar %0, %int3 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.mul.Tensor$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.mul.Tensor$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %0 = torch.vtensor.literal(dense<2> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.vtensor.literal(dense<3> : tensor<si64>) : !torch.vtensor<[],si64>
    %2 = torch.aten.mul.Tensor %0, %1 : !torch.vtensor<[],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.mul.Tensor$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<6> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.mul.Tensor$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int3 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.mul.Tensor %0, %1 : !torch.vtensor<[],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$canonicalize_numtotensor_0d_trunc() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<3> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.div.Tensor_mode$canonicalize_numtotensor_0d_trunc() -> !torch.vtensor<[],si64> {
    %int6 = torch.constant.int 6
    %int2 = torch.constant.int 2
    %str = torch.constant.str "trunc"
    %0 = torch.prim.NumToTensor.Scalar %int2 : !torch.int -> !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int6 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.div.Tensor_mode %1, %0, %str : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.str -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.div.Tensor_mode$canonicalize_literal_0d_trunc() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<3> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.div.Tensor_mode$canonicalize_literal_0d_trunc() -> !torch.vtensor<[],si64> {
    %int6 = torch.constant.int 6
    %str = torch.constant.str "trunc"
    %0 = torch.vtensor.literal(dense<2> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.prim.NumToTensor.Scalar %int6 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.div.Tensor_mode %1, %0, %str : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.str -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:   func.func @torch.aten.sort.int$reverse_false() -> !torch.list<int> {
// CHECK:             %[[INT0:.*]] = torch.constant.int 0
// CHECK:             %[[INT1:.*]] = torch.constant.int 1
// CHECK:             %[[INT2:.*]] = torch.constant.int 2
// CHECK:             %[[INT3:.*]] = torch.constant.int 3
// CHECK:             %[[RESULT:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT1]], %[[INT2]], %[[INT3]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             return %[[RESULT]] : !torch.list<int>
func.func @torch.aten.sort.int$reverse_false() -> !torch.list<int> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1, %int0, %int3, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  torch.aten.sort.int %0, %false : !torch.list<int>, !torch.bool
  return %0 : !torch.list<int>
}

// CHECK-LABEL:   func.func @torch.aten.sort.int$reverse_true() -> !torch.list<int> {
// CHECK:             %[[INT3:.*]] = torch.constant.int 3
// CHECK:             %[[INT2:.*]] = torch.constant.int 2
// CHECK:             %[[INT1:.*]] = torch.constant.int 1
// CHECK:             %[[INT0:.*]] = torch.constant.int 0
// CHECK:             %[[RESULT:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT2]], %[[INT1]], %[[INT0]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             return %[[RESULT]] : !torch.list<int>
func.func @torch.aten.sort.int$reverse_true() -> !torch.list<int> {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1, %int0, %int3, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  torch.aten.sort.int %0, %true : !torch.list<int>, !torch.bool
  return %0 : !torch.list<int>
}

// CHECK-LABEL: @torch.aten.sort$unary_element
// CHECK      : %[[INDICES:.*]] = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
// CHECK-NOT  : torch.aten.sort %arg
// CHECK      : return %arg0, %[[INDICES]] : !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
func.func @torch.aten.sort$unary_element(%arg0 : !torch.vtensor<[1],si64>, %arg1 : !torch.int, %arg2 : !torch.bool) -> (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) {
  %0, %1 = torch.aten.sort %arg0, %arg1, %arg2 : !torch.vtensor<[1],si64>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
  return %0, %1 : !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
}


// CHECK-LABEL: @torch.aten.sort$unary_dim
// CHECK      : %[[INDICES:.*]] = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
// CHECK-NOT  : torch.aten.sort %arg
// CHECK      : return %arg0, %[[INDICES]] : !torch.vtensor<[3, 1,4],si64>, !torch.vtensor<[1],si64>
func.func @torch.aten.sort$unary_dim(%arg0 : !torch.vtensor<[3, 1, 4],si64>, %arg1 : !torch.bool) -> (!torch.vtensor<[3, 1, 4],si64>, !torch.vtensor<[1],si64>) {
  %dim = torch.constant.int 1
  %0, %1 = torch.aten.sort %arg0, %dim, %arg1 : !torch.vtensor<[3, 1, 4],si64>, !torch.int, !torch.bool -> !torch.vtensor<[3, 1, 4],si64>, !torch.vtensor<[1],si64>
  return %0, %1 : !torch.vtensor<[3, 1,4],si64>, !torch.vtensor<[1],si64>
}

// CHECK-LABEL: @torch.aten.sort$nofold
// CHECK      : torch.aten.sort %arg
func.func @torch.aten.sort$nofold (%arg0 : !torch.vtensor<[3, 1, 4],si64>, %arg1 : !torch.bool) -> (!torch.vtensor<[3, 1, 4],si64>, !torch.vtensor<[3],si64>) {
  %dim = torch.constant.int 0
  %0, %1 = torch.aten.sort %arg0, %dim, %arg1 : !torch.vtensor<[3, 1, 4],si64>, !torch.int, !torch.bool -> !torch.vtensor<[3, 1, 4],si64>, !torch.vtensor<[3],si64>
  return %0, %1 : !torch.vtensor<[3, 1, 4],si64>, !torch.vtensor<[3],si64>
}

// -----

//  CHECK-LABEL:    @torch.aten.cat$fold_single_operand
//   CHECK-SAME:      %[[ARG0:.+]]: !torch.tensor
//        CHECK:        return %[[ARG0]] : !torch.tensor
func.func @torch.aten.cat$fold_single_operand(%arg0: !torch.tensor) -> !torch.tensor {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %arg0 : (!torch.tensor) -> !torch.list<tensor>
  %1 = torch.aten.cat %0, %int1 : !torch.list<tensor>, !torch.int -> !torch.tensor
  return %1: !torch.tensor
}

// -----

//  CHECK-LABEL:    @torch.aten.cat$fold_zero_dim_operand
//        CHECK:    %[[FOLD:.+]] = torch.vtensor.literal(dense<[1, 3, 2, 2]> : tensor<4xsi32>)
//        CHECK:    return %[[FOLD]] : !torch.vtensor
func.func @torch.aten.cat$fold_zero_dim_operand() -> !torch.vtensor<[4],si32> {
  %0 = torch.vtensor.literal(dense<[1, 3]> : tensor<2xsi32>) : !torch.vtensor<[2],si32>
  %1 = torch.vtensor.literal(dense<2> : tensor<2xsi32>) : !torch.vtensor<[2],si32>
  %int0 = torch.constant.int 0
  %list = torch.prim.ListConstruct %0, %1 : (!torch.vtensor<[2],si32>, !torch.vtensor<[2],si32>) -> !torch.list<vtensor>
  %cat = torch.aten.cat %list, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4],si32>
  return %cat: !torch.vtensor<[4],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$fold(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[3,4,2],f32>) -> !torch.vtensor<[3,4,2],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.vtensor<[3,4,2],f32>
func.func @torch.aten.broadcast_to$fold(%arg0: !torch.vtensor<[3,4,2],f32>) -> !torch.vtensor<[3,4,2],f32> {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %list = torch.prim.ListConstruct %int3, %int4, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.broadcast_to %arg0, %list : !torch.vtensor<[3,4,2],f32>, !torch.list<int> -> !torch.vtensor<[3,4,2],f32>
  return %0 : !torch.vtensor<[3,4,2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$fold_splat
// CHECK:                 %[[CST:.+]] = torch.vtensor.literal(dense<3.000000e+00> : tensor<3x4x2xf32>) : !torch.vtensor<[3,4,2],f32>
// CHECK:                 return %[[CST]]
func.func @torch.aten.broadcast_to$fold_splat() -> !torch.vtensor<[3,4,2],f32> {
  %tensor = torch.vtensor.literal(dense<3.0> : tensor<1x4x1xf32>) : !torch.vtensor<[1,4,1],f32>
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %list = torch.prim.ListConstruct %int3, %int4, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.broadcast_to %tensor, %list : !torch.vtensor<[1,4,1],f32>, !torch.list<int> -> !torch.vtensor<[3,4,2],f32>
  return %0 : !torch.vtensor<[3,4,2],f32>
}

// -----

//  CHECK-LABEL:    @torch.aten.slice.tensor$not_fold_slice
//   CHECK-SAME:      %[[ARG0:.+]]: !torch.vtensor<[4],f32>
//        CHECK:        torch.aten.slice.Tensor
func.func @torch.aten.slice.tensor$not_fold_slice(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3],f32> {
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int0, %int-1, %int1 : !torch.vtensor<[4], f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3], f32>
  return %0 : !torch.vtensor<[3],f32>
}

//  CHECK-LABEL:    @torch.aten.slice.tensor$fold_full_slice
//   CHECK-SAME:      %[[ARG0:.+]]: !torch.vtensor<[?],f32>
//        CHECK:        return %[[ARG0]] : !torch.vtensor<[?],f32>
func.func @torch.aten.slice.tensor$fold_full_slice(%arg0: !torch.vtensor<[?],f32>, %dim: !torch.int) -> !torch.vtensor<[?],f32> {
  %int1 = torch.constant.int 1
  %int9223372036854775807  = torch.constant.int 9223372036854775807
  %int0 = torch.constant.int 0
  %0 = torch.aten.slice.Tensor %arg0, %dim, %int0, %int9223372036854775807, %int1 : !torch.vtensor<[?], f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?], f32>
  return %0 : !torch.vtensor<[?],f32>
}

//  CHECK-LABEL:    @torch.aten.slice.tensor$fold_oob_start
//        CHECK:        %[[LIT:.*]] = torch.vtensor.literal(dense<[0, 1, 2]> : tensor<3xsi64>) : !torch.vtensor<[3],si64>
//        CHECK:        return %[[LIT]] : !torch.vtensor<[3],si64>
func.func @torch.aten.slice.tensor$fold_oob_start() -> !torch.vtensor<[3],si64> {
  %0 = torch.vtensor.literal(dense<[0,1,2,3]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %int1 = torch.constant.int 1
  %int-1  = torch.constant.int -1
  %int-10 = torch.constant.int -10
  %int0 = torch.constant.int 0
  %1 = torch.aten.slice.Tensor %0, %int0, %int-10, %int-1, %int1 : !torch.vtensor<[4], si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3], si64>
  return %1 : !torch.vtensor<[3],si64>
}

//  CHECK-LABEL:    @torch.aten.slice.tensor$nofold_invalid_shape
//        CHECK:    %[[SLICE:.*]] = torch.aten.slice.Tensor
//        CHECK:    return %[[SLICE]]
func.func @torch.aten.slice.tensor$nofold_invalid_shape() -> !torch.vtensor<[4],si64> {
  %0 = torch.vtensor.literal(dense<[0,1,2,3]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %int1 = torch.constant.int 1
  %int-1  = torch.constant.int -1
  %int-10 = torch.constant.int -10
  %int0 = torch.constant.int 0
  %1 = torch.aten.slice.Tensor %0, %int0, %int-10, %int-1, %int1 : !torch.vtensor<[4], si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4], si64>
  return %1 : !torch.vtensor<[4],si64>
}

//  CHECK-LABEL:    @torch.aten.slice.tensor$no_fold_step
//        CHECK: torch.aten.slice.Tensor
func.func @torch.aten.slice.tensor$no_fold_step(%arg0: !torch.vtensor<[?],f32>, %dim: !torch.int) -> !torch.vtensor<[?],f32> {
  %int2 = torch.constant.int 2
  %int9223372036854775807  = torch.constant.int 9223372036854775807
  %int0 = torch.constant.int 0
  %0 = torch.aten.slice.Tensor %arg0, %dim, %int0, %int9223372036854775807, %int2 : !torch.vtensor<[?], f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?], f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.slice.tensor$fold_dim_1() -> (!torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>) {
// CHECK-NOT:       torch.aten.slice.Tensor
// CHECK:           %[[RET_0:.*]] = torch.vtensor.literal(dense<50> : tensor<1x1xsi64>) : !torch.vtensor<[1,1],si64>
// CHECK-NOT:       torch.aten.slice.Tensor
// CHECK:           %[[RET_1:.*]] = torch.vtensor.literal(dense<70> : tensor<1x1xsi64>) : !torch.vtensor<[1,1],si64>
// CHECK-NOT:       torch.aten.slice.Tensor
// CHECK:           return %[[RET_0]], %[[RET_1]]
func.func @torch.aten.slice.tensor$fold_dim_1() -> (!torch.vtensor<[1, 1],si64>, !torch.vtensor<[1, 1],si64>) {
  %tensor = torch.vtensor.literal(dense<[[10,20,30,40,50,60,70,80,90,100]]> : tensor<1x10xsi64>) : !torch.vtensor<[1, 10],si64>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %int6 = torch.constant.int 6
  %int7 = torch.constant.int 7
  %dim = torch.constant.int 1
  %0 = torch.aten.slice.Tensor %tensor, %dim, %int4, %int5, %int1 : !torch.vtensor<[1, 10], si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1, 1], si64>
  %1 = torch.aten.slice.Tensor %tensor, %dim, %int6, %int7, %int1 : !torch.vtensor<[1, 10], si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1, 1], si64>
  return %0, %1 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>
}


// -----
// CHECK-LABEL:   func.func @torch.aten.slice.tensor$fold_small() -> !torch.vtensor<[2],si32> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<[3, 5]> : tensor<2xsi32>) : !torch.vtensor<[2],si32>
// CHECK:             return %[[CST]]
func.func @torch.aten.slice.tensor$fold_small() -> (!torch.vtensor<[2],si32>) {
  %tensor = torch.vtensor.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xsi32>) : !torch.vtensor<[10],si32>
  %dim = torch.constant.int 0
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int7 = torch.constant.int 7
  %0 = torch.aten.slice.Tensor %tensor, %dim, %int3, %int7, %int2 : !torch.vtensor<[10], si32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2], si32>
  return %0 : !torch.vtensor<[2],si32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.slice.tensor$fold_dim_0() -> (!torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>) {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<1.600000e+01> : tensor<1x1xf32>) : !torch.vtensor<[1,1],f32>
// CHECK:             %[[CST0:.+]] = torch.vtensor.literal(dense<6.400000e+01> : tensor<1x1xf32>) : !torch.vtensor<[1,1],f32>
// CHECK:             return %[[CST]], %[[CST0]]
func.func @torch.aten.slice.tensor$fold_dim_0() -> (!torch.vtensor<[1, 1],f32>, !torch.vtensor<[1, 1],f32>) {
  %tensor = torch.vtensor.literal(dense<[[2.0],[4.0],[8.0],[16.0],[32.0],[64.0],[128.0],[256.0],[512.0],[1024.0]]> : tensor<10x1xf32>) : !torch.vtensor<[10, 1],f32>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %intn7 = torch.constant.int -7
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %int6 = torch.constant.int 6
  %dim = torch.constant.int 0
  %0 = torch.aten.slice.Tensor %tensor, %dim, %intn7, %int4, %int1 : !torch.vtensor<[10, 1], f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1, 1], f32>
  %1 = torch.aten.slice.Tensor %tensor, %dim, %int5, %int6, %int1 : !torch.vtensor<[10, 1], f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1, 1], f32>
  return %0, %1 : !torch.vtensor<[1, 1],f32>, !torch.vtensor<[1, 1], f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.slice.tensor$fold_dim_0_non_contiguous() -> !torch.vtensor<[4,1],si64> {
// CHECK{LITERAL}:             %0 = torch.vtensor.literal(dense<[[28], [14], [7], [4]]> : tensor<4x1xsi64>) : !torch.vtensor<[4,1],si64>
// CHECK:             return %0
func.func @torch.aten.slice.tensor$fold_dim_0_non_contiguous() -> (!torch.vtensor<[4,1],si64>) {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.vtensor.literal(dense<[[28, 28], [14, 14], [7, 7], [4, 4]]> : tensor<4x2xsi64>) : !torch.vtensor<[4,2],si64>
  %1 = torch.aten.slice.Tensor %0, %int1, %int1, %int2, %int1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],si64>
  return %1 : !torch.vtensor<[4,1],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.rsub.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<-1> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.rsub.Scalar$canonicalize_literal_0d() -> !torch.vtensor<[],si64> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %2 = torch.aten.rsub.Scalar %0, %int2, %int3 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.rsub.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
// CHECK:             %[[CST:.+]] = torch.vtensor.literal(dense<-1> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:             return %[[CST]] : !torch.vtensor<[],si64>
func.func @torch.aten.rsub.Scalar$canonicalize_numtotensor_0d() -> !torch.vtensor<[],si64> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.vtensor<[],si64>
    %2 = torch.aten.rsub.Scalar %0, %int2, %int3 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
    return %2 : !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ScalarImplicit$canonicalize_numtotensor_0d() -> !torch.number {
// CHECK:             %int1 = torch.constant.int 1
// CHECK:             %[[VAL_1:.*]] = torch.derefine %int1 : !torch.int to !torch.number
// CHECK:             return %[[VAL_1]] : !torch.number
func.func @torch.aten.ScalarImplicit$canonicalize_numtotensor_0d() -> !torch.number {
    %int1 = torch.constant.int 1
    %0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.vtensor<[],si64>
    %1 = torch.aten.ScalarImplicit %0 : !torch.vtensor<[],si64> -> !torch.number
    return %1 : !torch.number
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ScalarImplicit$canonicalize_literal_0d() -> !torch.number {
// CHECK:             %int1 = torch.constant.int 1
// CHECK:             %[[VAL_0:.*]] = torch.derefine %int1 : !torch.int to !torch.number
// CHECK:             return %[[VAL_0]] : !torch.number
func.func @torch.aten.ScalarImplicit$canonicalize_literal_0d() -> !torch.number {
    %0 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.aten.ScalarImplicit %0 : !torch.vtensor<[],si64> -> !torch.number
    return %1 : !torch.number
}

// -----

// CHECK-LABEL:   func.func @torch.aten.ScalarImplicit$canonicalize_literal_0d_float() -> !torch.number {
// CHECK:             %float1.000000e00 = torch.constant.float 1.000000e+00
// CHECK:             %[[VAL_0:.*]] = torch.derefine %float1.000000e00 : !torch.float to !torch.number
// CHECK:             return %[[VAL_0]] : !torch.number
func.func @torch.aten.ScalarImplicit$canonicalize_literal_0d_float() -> !torch.number {
    %0 = torch.vtensor.literal(dense<1.0> : tensor<f64>) : !torch.vtensor<[],f64>
    %1 = torch.aten.ScalarImplicit %0 : !torch.vtensor<[],f64> -> !torch.number
    return %1 : !torch.number
}

// -----

// CHECK-LABEL:   func.func @torch.aten.FloatImplicit$canonicalize_numtotensor_0d() -> !torch.float {
// CHECK:             %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:             return %[[FLOAT1]] : !torch.float
func.func @torch.aten.FloatImplicit$canonicalize_numtotensor_0d() -> !torch.float {
    %float1 = torch.constant.float 1.0
    %0 = torch.prim.NumToTensor.Scalar %float1 : !torch.float -> !torch.vtensor<[],f64>
    %1 = torch.aten.FloatImplicit %0 : !torch.vtensor<[],f64> -> !torch.float
    return %1 : !torch.float
}

// -----

// CHECK-LABEL:   func.func @torch.aten.FloatImplicit$canonicalize_literal_0d() -> !torch.float {
// CHECK:             %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:             return %[[FLOAT1]] : !torch.float
func.func @torch.aten.FloatImplicit$canonicalize_literal_0d() -> !torch.float {
    %0 = torch.vtensor.literal(dense<1.0> : tensor<f64>) : !torch.vtensor<[],f64>
    %1 = torch.aten.FloatImplicit %0 : !torch.vtensor<[],f64> -> !torch.float
    return %1 : !torch.float
}

// -----

// CHECK-LABEL:   func.func @torch.aten.IntImplicit$canonicalize_numtotensor_0d() -> !torch.int {
// CHECK:             %[[INT1:.*]] = torch.constant.int 1
// CHECK:             return %[[INT1]] : !torch.int
func.func @torch.aten.IntImplicit$canonicalize_numtotensor_0d() -> !torch.int {
    %int1 = torch.constant.int 1
    %0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.vtensor<[],si64>
    %1 = torch.aten.IntImplicit %0 : !torch.vtensor<[],si64> -> !torch.int
    return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.IntImplicit$canonicalize_literal_0d() -> !torch.int {
// CHECK:             %[[INT1:.*]] = torch.constant.int 1
// CHECK:             return %[[INT1]] : !torch.int
func.func @torch.aten.IntImplicit$canonicalize_literal_0d() -> !torch.int {
    %0 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.aten.IntImplicit %0 : !torch.vtensor<[],si64> -> !torch.int
    return %1 : !torch.int
}

// -----

// CHECK-LABEL:   func.func @torch.prims.view_of$fold(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[3,4,2],f32>) -> !torch.vtensor<[3,4,2],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.vtensor<[3,4,2],f32>
func.func @torch.prims.view_of$fold(%arg0: !torch.vtensor<[3,4,2],f32>) -> !torch.vtensor<[3,4,2],f32> {
  %0 = torch.prims.view_of %arg0 : !torch.vtensor<[3,4,2],f32> -> !torch.vtensor<[3,4,2],f32>
  return %0 : !torch.vtensor<[3,4,2],f32>
}

// CHECK-LABEL:  func.func @torch.aten.cuda$canonicalize
// CHECK-SAME:           %[[ARG:.*]]: !torch.tensor
// CHECK-NEXT:     return %[[ARG]] : !torch.tensor
func.func @torch.aten.cuda$canonicalize(%arg0: !torch.tensor) -> !torch.tensor {
  %0 = torch.aten.cuda %arg0 : !torch.tensor -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:  func.func @torch.aten.device.with_index$canonicalize
// CHECK-NEXT:     %[[VAL:.*]] = torch.constant.device "cuda:0"
// CHECK-NEXT:     return %[[VAL]] : !torch.Device
func.func @torch.aten.device.with_index$canonicalize() -> !torch.Device {
  %str = torch.constant.str "cuda"
  %int0 = torch.constant.int 0
  %0 = torch.aten.device.with_index %str, %int0 : !torch.str, !torch.int -> !torch.Device
  return %0 : !torch.Device
}

// CHECK-LABEL:   func.func @torch.aten.add$fold() -> !torch.float {
// CHECK:             %[[FLOAT_1:.*]] = torch.constant.float 3.000000e+00
// CHECK:             return %[[FLOAT_1]] : !torch.float
func.func @torch.aten.add$fold() -> !torch.float {
    %float1 = torch.constant.float 1.0
    %float2 = torch.constant.float 2.0
    %0 = torch.aten.add %float1, %float2 : !torch.float, !torch.float -> !torch.float
    return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch.aten.any.bool$fold() -> !torch.bool {
// CHECK:           %[[CST_TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[CST_TRUE]] : !torch.bool
func.func @torch.aten.any.bool$fold() -> !torch.bool {
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %input = torch.prim.ListConstruct %false, %true, %false : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %0 = torch.aten.any.bool %input : !torch.list<bool> -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.all.bool$fold() -> !torch.bool {
// CHECK:           %[[CST_TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[CST_TRUE]] : !torch.bool
func.func @torch.aten.all.bool$fold() -> !torch.bool {
  %true = torch.constant.bool true
  %input = torch.prim.ListConstruct %true, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %0 = torch.aten.all.bool %input : !torch.list<bool> -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch.aten.floor$canonicalize
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[?,?],si64>
// CHECK-NEXT:      return %[[ARG]] : !torch.vtensor<[?,?],si64>
func.func @torch.aten.floor$canonicalize(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
  %0 = torch.aten.floor %arg0 : !torch.vtensor<[?,?],si64> -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}

// CHECK-LABEL:   func.func @torch.aten.trunc$canonicalize
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[?,?],si64>
// CHECK-NEXT:      return %[[ARG]] : !torch.vtensor<[?,?],si64>
func.func @torch.aten.trunc$canonicalize(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?],si64> {
  %0 = torch.aten.trunc %arg0 : !torch.vtensor<[?,?],si64> -> !torch.vtensor<[?,?],si64>
  return %0 : !torch.vtensor<[?,?],si64>
}

// CHECK-LABEL:   func.func @torch.aten.numel$canonicalize
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[3,4],f32>
// CHECK-NEXT:      %int12 = torch.constant.int 12
// CHECK-NEXT:      return %int12 : !torch.int
func.func @torch.aten.numel$canonicalize(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.int {
  %0 = torch.aten.numel %arg0 : !torch.vtensor<[3,4],f32> -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch.aten.masked_fill.Tensor$canonicalize
// CHECK-NEXT:      torch.constant.float -1.000000e+09
// CHECK-NEXT:      torch.aten.masked_fill.Scalar
// CHECK-NEXT:      return
func.func @torch.aten.masked_fill.Tensor$canonicalize(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.vtensor.literal(dense<-1.000000e+09> : tensor<f32>) : !torch.vtensor<[],f32>
  %1 = torch.aten.masked_fill.Tensor %arg0, %arg1, %0 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>, !torch.vtensor<[],f32> -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:   func.func @torch.aten.detach$canonicalize
// CHECK-NEXT:      torch.aten.detach
func.func @torch.aten.detach$canonicalize(%arg0: !torch.tensor<[1],f32>) -> !torch.tensor {
  %1 = torch.aten.detach %arg0 : !torch.tensor<[1],f32> -> !torch.tensor
  return %1 : !torch.tensor
}

// CHECK-LABEL:   func.func @torch.aten.index_select$noop(
// CHECK-SAME:      %[[ARG:.*]]: !torch.vtensor<[1,2,3],si64>
// CHECK-NEXT:      return %[[ARG]] : !torch.vtensor<[1,2,3],si64>
func.func @torch.aten.index_select$noop(%arg0 : !torch.vtensor<[1,2,3],si64>, %arg1 : !torch.int, %arg2 : !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,2,3],si64> {
  %0 = torch.aten.index_select %arg0, %arg1, %arg2 : !torch.vtensor<[1,2,3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1,2,3],si64>
  return %0 : !torch.vtensor<[1,2,3],si64>
}

// CHECK-LABEL:   func.func @torch.aten.index_select$const_si_si(
// CHECK-NEXT:      %[[RES:.*]] = torch.vtensor.literal(dense<60> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
// CHECK-NEXT:      return %[[RES]] : !torch.vtensor<[1],si64>
func.func @torch.aten.index_select$const_si_si() -> !torch.vtensor<[1],si64> {
  %tensor = torch.vtensor.literal(dense<[10,20,30,40,50,60,70,80,90,100]> : tensor<10xsi64>) : !torch.vtensor<[10],si64>
  %dim = torch.constant.int 0
  %index = torch.vtensor.literal(dense<5> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  %0 = torch.aten.index_select %tensor, %dim, %index : !torch.vtensor<[10],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  return %0 : !torch.vtensor<[1],si64>
}

// CHECK-LABEL:   func.func @torch.aten.index_select$const_si_ui(
// CHECK-NEXT:      %[[RES:.*]] = torch.vtensor.literal(dense<60> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
// CHECK-NEXT:      return %[[RES]] : !torch.vtensor<[1],si64>
func.func @torch.aten.index_select$const_si_ui() -> !torch.vtensor<[1],si64> {
  %tensor = torch.vtensor.literal(dense<[10,20,30,40,50,60,70,80,90,100]> : tensor<10xsi64>) : !torch.vtensor<[10],si64>
  %dim = torch.constant.int 0
  %index = torch.vtensor.literal(dense<5> : tensor<1xui64>) : !torch.vtensor<[1],ui64>
  %0 = torch.aten.index_select %tensor, %dim, %index : !torch.vtensor<[10],si64>, !torch.int, !torch.vtensor<[1],ui64> -> !torch.vtensor<[1],si64>
  return %0 : !torch.vtensor<[1],si64>
}

// CHECK-LABEL:   func.func @torch.aten.index_select$const_f32_ui(
// CHECK-NEXT:      %[[RES:.*]] = torch.vtensor.literal(dense<6.6{{.*}}> : tensor<1xf32>) : !torch.vtensor<[1],f32>
// CHECK-NEXT:      return %[[RES]] : !torch.vtensor<[1],f32>
func.func @torch.aten.index_select$const_f32_ui() -> !torch.vtensor<[1],f32> {
  %tensor = torch.vtensor.literal(dense<[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.0]> : tensor<10xf32>) : !torch.vtensor<[10],f32>
  %dim = torch.constant.int 0
  %index = torch.vtensor.literal(dense<5> : tensor<1xui64>) : !torch.vtensor<[1],ui64>
  %0 = torch.aten.index_select %tensor, %dim, %index : !torch.vtensor<[10],f32>, !torch.int, !torch.vtensor<[1],ui64> -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// CHECK-LABEL:   func.func @torch.aten.index_select$const_f32_si_neg(
// CHECK-NEXT:      %[[RES:.*]] = torch.vtensor.literal(dense<7.{{.*}}> : tensor<1xf32>) : !torch.vtensor<[1],f32>
// CHECK-NEXT:      return %[[RES]] : !torch.vtensor<[1],f32>
func.func @torch.aten.index_select$const_f32_si_neg() -> !torch.vtensor<[1],f32> {
  %tensor = torch.vtensor.literal(dense<[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.0]> : tensor<10xf32>) : !torch.vtensor<[10],f32>
  %dim = torch.constant.int -1
  %index = torch.vtensor.literal(dense<-4> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  %0 = torch.aten.index_select %tensor, %dim, %index : !torch.vtensor<[10],f32>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL: @fold_aten_where_true_attr
func.func @fold_aten_where_true_attr() -> !torch.vtensor<[4],si64> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  // CHECK: return %[[RET]]
  %bool = torch.vtensor.literal(dense<1> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %rhs = torch.vtensor.literal(dense<11> : tensor<si64>) : !torch.vtensor<[],si64>
  %where = torch.aten.where.self %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_prim_numtotensor_scalar
func.func @fold_prim_numtotensor_scalar() -> !torch.vtensor<[1],si64> {
  %int42 = torch.constant.int 42
  // CHECK: %[[TENSOR:.+]] = torch.vtensor.literal(dense<42> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // CHECK: return %[[TENSOR]]
  %0 = torch.prim.NumToTensor.Scalar %int42 : !torch.int -> !torch.vtensor<[1],si64>
  return %0 : !torch.vtensor<[1],si64>
}

// -----

// CHECK-LABEL: @fold_aten_where_false_attr
func.func @fold_aten_where_false_attr() -> !torch.vtensor<[4],si64> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<11> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  // CHECK: return %[[RET]]
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %rhs = torch.vtensor.literal(dense<11> : tensor<si64>) : !torch.vtensor<[],si64>
  %where = torch.aten.where.self %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_where_true_value
func.func @fold_aten_where_true_value(%arg0 : !torch.vtensor<[4],si64>, %arg1 : !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
  // CHECK: return %arg0
  %bool = torch.vtensor.literal(dense<1> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %where = torch.aten.where.self %bool, %arg0, %arg1 : !torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_where_false_value
func.func @fold_aten_where_false_value(%arg0 : !torch.vtensor<[4],si64>, %arg1 : !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
  // CHECK: return %arg1
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %where = torch.aten.where.self %bool, %arg0, %arg1 : !torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}


// -----

// CHECK-LABEL: @fold_aten_where_true_value_nofold
func.func @fold_aten_where_true_value_nofold(%arg0 : !torch.vtensor<[],si64>, %arg1 : !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
  // CHECK: torch.aten.where.self
  %bool = torch.vtensor.literal(dense<1> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %where = torch.aten.where.self %bool, %arg0, %arg1 : !torch.vtensor<[4],i1>, !torch.vtensor<[],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_where_true_scalar_int
func.func @fold_aten_where_true_scalar_int() -> !torch.vtensor<[4],si64> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  // CHECK: return %[[RET]]
  %bool = torch.vtensor.literal(dense<1> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.constant.int 7
  %rhs = torch.constant.int 11
  %where = torch.aten.where.Scalar %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.int, !torch.int -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_where_false_scalar_int
func.func @fold_aten_where_false_scalar_int() -> !torch.vtensor<[4],ui8> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<11> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  // CHECK: return %[[RET]]
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.constant.int 7
  %rhs = torch.constant.int 11
  %where = torch.aten.where.Scalar %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.int, !torch.int -> !torch.vtensor<[4],ui8>
  return %where : !torch.vtensor<[4],ui8>
}

// -----

// CHECK-LABEL: @fold_aten_where_false_scalar_fp
func.func @fold_aten_where_false_scalar_fp() -> !torch.vtensor<[4],f32> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<1.100000e+01> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  // CHECK: return %[[RET]]
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.constant.float 7.0
  %rhs = torch.constant.float 11.0
  %where = torch.aten.where.Scalar %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.float, !torch.float -> !torch.vtensor<[4],f32>
  return %where : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @fold_aten_where_true_sother_int
func.func @fold_aten_where_true_sother_int() -> !torch.vtensor<[4],si64> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  // CHECK: %[[RET]]
  %bool = torch.vtensor.literal(dense<1> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %rhs = torch.constant.int 11
  %where = torch.aten.where.ScalarOther %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}


// -----

// CHECK-LABEL: @fold_aten_where_false_sother_int
func.func @fold_aten_where_false_sother_int() -> !torch.vtensor<[4],ui8> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<11> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  // CHECK: return %[[RET]]
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.vtensor.literal(dense<7> : tensor<ui8>) : !torch.vtensor<[],ui8>
  %rhs = torch.constant.int 11
  %where = torch.aten.where.ScalarOther %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[4],ui8>
  return %where : !torch.vtensor<[4],ui8>
}

// -----

// CHECK-LABEL: @fold_aten_where_false_sother_fp
func.func @fold_aten_where_false_sother_fp() -> !torch.vtensor<[4],f32> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<1.100000e+01> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  // CHECK: %[[RET]]
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.vtensor.literal(dense<7.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %rhs = torch.constant.float 11.0
  %where = torch.aten.where.ScalarOther %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
  return %where : !torch.vtensor<[4],f32>
}


// -----

// CHECK-LABEL: @fold_aten_where_true_sself_int
func.func @fold_aten_where_true_sself_int() -> !torch.vtensor<[4],si64> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  // CHECK: %[[RET]]
  %bool = torch.vtensor.literal(dense<1> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.constant.int 7
  %rhs = torch.vtensor.literal(dense<11> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %where = torch.aten.where.ScalarSelf %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.int, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
  return %where : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_where_false_sself_int
func.func @fold_aten_where_false_sself_int() -> !torch.vtensor<[4],ui8> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<11> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  // CHECK: return %[[RET]]
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.constant.int 7
  %rhs = torch.vtensor.literal(dense<11> : tensor<ui8>) : !torch.vtensor<[],ui8>
  %where = torch.aten.where.ScalarSelf %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.int, !torch.vtensor<[],ui8> -> !torch.vtensor<[4],ui8>
  return %where : !torch.vtensor<[4],ui8>
}

// -----

// CHECK-LABEL: @fold_aten_where_false_sself_fp
func.func @fold_aten_where_false_sself_fp() -> !torch.vtensor<[4],f32> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<1.100000e+01> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  // CHECK: %[[RET]]
  %bool = torch.vtensor.literal(dense<0> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  %lhs = torch.constant.float 7.0
  %rhs = torch.vtensor.literal(dense<11.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %where = torch.aten.where.ScalarSelf %bool, %lhs, %rhs : !torch.vtensor<[4],i1>, !torch.float, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %where : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @aten_select_int_fold_splat
func.func @aten_select_int_fold_splat(%arg0 : !torch.int, %arg1 : !torch.int) -> !torch.vtensor<[1],si64> {
  %splat = torch.vtensor.literal(dense<4> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %select = torch.aten.select.int %splat, %arg0, %arg1 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<4> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // CHECK: return %[[RET]]
  return %select : !torch.vtensor<[1],si64>
}

// -----

// CHECK-LABEL: @aten_select_int_fold_1D
func.func @aten_select_int_fold_1D() -> !torch.vtensor<[1],si64> {
  %index = torch.constant.int 1
  %dim = torch.constant.int 0
  %splat = torch.vtensor.literal(dense<[5,6,7,8]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %select = torch.aten.select.int %splat, %dim, %index : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<6> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // CHECK: return %[[RET]]
  return %select : !torch.vtensor<[1],si64>
}

// -----

// CHECK-LABEL: @aten_select_int_fold_3D
func.func @aten_select_int_fold_3D() -> !torch.vtensor<[1, 1, 1],si64> {
  %index = torch.constant.int 2
  %dim = torch.constant.int 2
  %splat = torch.vtensor.literal(dense<[[[5,6,7,8]]]> : tensor<1x1x4xsi64>) : !torch.vtensor<[1,1,4],si64>
  %select = torch.aten.select.int %splat, %dim, %index : !torch.vtensor<[1,1,4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,1,1],si64>
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<7> : tensor<1x1x1xsi64>) : !torch.vtensor<[1,1,1],si64>
  // CHECK: return %[[RET]]
  return %select : !torch.vtensor<[1,1,1],si64>
}

// -----


// CHECK-LABEL: @aten_eq_tensor_args
func.func @aten_eq_tensor_args(%arg0 : !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %0 = torch.aten.eq.Tensor %arg0, %arg0 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_splats_int_false
func.func @aten_eq_tensor_splats_int_false() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<false> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<4> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %rhs = torch.vtensor.literal(dense<5> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_splats_int_true
func.func @aten_eq_tensor_splats_int_true() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<5> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %rhs = torch.vtensor.literal(dense<5> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_splats_fp_false
func.func @aten_eq_tensor_splats_fp_false() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<false> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<4.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %rhs = torch.vtensor.literal(dense<5.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_splats_fp_true
func.func @aten_eq_tensor_splats_fp_true() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<5.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %rhs = torch.vtensor.literal(dense<5.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_splat_dense_fp
func.func @aten_eq_tensor_splat_dense_fp() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<[false, true, false, true]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<5.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %rhs = torch.vtensor.literal(dense<[4.0, 5.0, 6.0, 5.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_dense_fp
func.func @aten_eq_tensor_dense_fp() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<[true, false, true, false]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<[4.0, 5.5, 6.0, 6.4]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %rhs = torch.vtensor.literal(dense<[4.0, 5.0, 6.0, 5.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_splat_dense_int
func.func @aten_eq_tensor_splat_dense_int() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<[false, true, false, true]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<5> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %rhs = torch.vtensor.literal(dense<[4, 5, 6, 5]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_eq_tensor_dense_int
func.func @aten_eq_tensor_dense_int() -> !torch.vtensor<[4],i1> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<[true, true, true, false]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[RET]]
  %lhs = torch.vtensor.literal(dense<[4, 5, 6, 6]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %rhs = torch.vtensor.literal(dense<[4, 5, 6, 5]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %0 = torch.aten.eq.Tensor %lhs, %rhs : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_shape_to_tensor
func.func @aten_shape_to_tensor(%arg0 : !torch.vtensor<[4,5,6],f32>) -> !torch.vtensor<[3],si32> {
  // CHECK: %[[CST:.+]] = torch.vtensor.literal(dense<[4, 5, 6]> : tensor<3xsi32>) : !torch.vtensor<[3],si32>
  %0 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[4,5,6],f32> -> !torch.vtensor<[3],si32>
  // CHECK: return %[[CST]]
  return %0 : !torch.vtensor<[3],si32>
}

// -----

// CHECK-LABEL: @aten_cat_zero
func.func @aten_cat_zero(%arg0 : !torch.vtensor<[4,5,6],f32>, %arg1 : !torch.vtensor<[4,0,6],f32>) -> !torch.vtensor<[4,5,6],f32> {
  // CHECK: return %arg0 : !torch.vtensor<[4,5,6],f32>
  %list = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[4,5,6],f32>, !torch.vtensor<[4,0,6],f32>) -> !torch.list<vtensor>
  %dim = torch.constant.int -2
  %0 = torch.aten.cat %list, %dim : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,5,6],f32>
  return %0 : !torch.vtensor<[4,5,6],f32>
}

// -----

// CHECK-LABEL: @aten_tensor_scalar_lt
func.func @aten_tensor_scalar_lt() -> (!torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>) {
  // CHECK: %[[CST:.+]] = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[CST]], %[[CST]] : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
  %intTensor = torch.vtensor.literal(dense<1> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %fpTensor = torch.vtensor.literal(dense<1.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %intScalar = torch.constant.int 2
  %fpScalar = torch.constant.float 2.0
  %intBool = torch.aten.lt.Scalar %intTensor, %intScalar : !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4],i1>
  %fpBool = torch.aten.lt.Scalar %fpTensor, %fpScalar : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],i1>
  return %intBool, %fpBool : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
}


// -----

// CHECK-LABEL: @aten_tensor_tensor_lt
func.func @aten_tensor_tensor_lt() -> (!torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>) {
  // CHECK: %[[UNSIGN:.+]] = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: %[[SIGNED:.+]] = torch.vtensor.literal(dense<[true, false, false, false]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[UNSIGN]], %[[SIGNED]], %[[SIGNED]]
  %intTensor = torch.vtensor.literal(dense<[127, -128, -127, -126]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %uintTensor = torch.vtensor.literal(dense<[127, 128, 129, 130]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  %fpTensor = torch.vtensor.literal(dense<[127.0, 128.0, 129.0, 130.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %intScalar = torch.constant.int 128
  %fpScalar = torch.constant.float 128.0
  %intBool = torch.aten.lt.Scalar %intTensor, %intScalar : !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4],i1>
  %uintBool = torch.aten.lt.Scalar %uintTensor, %intScalar : !torch.vtensor<[4],ui8>, !torch.int -> !torch.vtensor<[4],i1>
  %fpBool = torch.aten.lt.Scalar %fpTensor, %fpScalar : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],i1>
  return %intBool, %uintBool, %fpBool : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_tensor_tensor_le
func.func @aten_tensor_tensor_le() -> (!torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>) {
  // CHECK: %[[UNSIGN:.+]] = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: %[[SIGNED:.+]] = torch.vtensor.literal(dense<[true, true, false, false]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[UNSIGN]], %[[SIGNED]], %[[SIGNED]]
  %intTensor = torch.vtensor.literal(dense<[127, -128, -127, -126]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %uintTensor = torch.vtensor.literal(dense<[127, 128, 129, 130]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  %fpTensor = torch.vtensor.literal(dense<[127.0, 128.0, 129.0, 130.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %intScalar = torch.constant.int 128
  %fpScalar = torch.constant.float 128.0
  %intBool = torch.aten.le.Scalar %intTensor, %intScalar : !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4],i1>
  %uintBool = torch.aten.le.Scalar %uintTensor, %intScalar : !torch.vtensor<[4],ui8>, !torch.int -> !torch.vtensor<[4],i1>
  %fpBool = torch.aten.le.Scalar %fpTensor, %fpScalar : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],i1>
  return %intBool, %uintBool, %fpBool : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
}


// -----

// CHECK-LABEL: @aten_tensor_tensor_ge
func.func @aten_tensor_tensor_ge() -> (!torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>) {
  // CHECK: %[[UNSIGN:.+]] = torch.vtensor.literal(dense<false> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: %[[SIGNED:.+]] = torch.vtensor.literal(dense<[false, true, true, true]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[UNSIGN]], %[[SIGNED]], %[[SIGNED]]
  %intTensor = torch.vtensor.literal(dense<[127, -128, -127, -126]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %uintTensor = torch.vtensor.literal(dense<[127, 128, 129, 130]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  %fpTensor = torch.vtensor.literal(dense<[127.0, 128.0, 129.0, 130.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %intScalar = torch.constant.int 128
  %fpScalar = torch.constant.float 128.0
  %intBool = torch.aten.ge.Scalar %intTensor, %intScalar : !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4],i1>
  %uintBool = torch.aten.ge.Scalar %uintTensor, %intScalar : !torch.vtensor<[4],ui8>, !torch.int -> !torch.vtensor<[4],i1>
  %fpBool = torch.aten.ge.Scalar %fpTensor, %fpScalar : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],i1>
  return %intBool, %uintBool, %fpBool : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_tensor_tensor_gt
func.func @aten_tensor_tensor_gt() -> (!torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>) {
  // CHECK: %[[UNSIGN:.+]] = torch.vtensor.literal(dense<false> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: %[[SIGNED:.+]] = torch.vtensor.literal(dense<[false, false, true, true]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[UNSIGN]], %[[SIGNED]], %[[SIGNED]]
  %intTensor = torch.vtensor.literal(dense<[127, -128, -127, -126]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %uintTensor = torch.vtensor.literal(dense<[127, 128, 129, 130]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  %fpTensor = torch.vtensor.literal(dense<[127.0, 128.0, 129.0, 130.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %intScalar = torch.constant.int 128
  %fpScalar = torch.constant.float 128.0
  %intBool = torch.aten.gt.Scalar %intTensor, %intScalar : !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4],i1>
  %uintBool = torch.aten.gt.Scalar %uintTensor, %intScalar : !torch.vtensor<[4],ui8>, !torch.int -> !torch.vtensor<[4],i1>
  %fpBool = torch.aten.gt.Scalar %fpTensor, %fpScalar : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],i1>
  return %intBool, %uintBool, %fpBool : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_tensor_tensor_eq
func.func @aten_tensor_tensor_eq() -> (!torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>) {
  // CHECK: %[[UNSIGN:.+]] = torch.vtensor.literal(dense<false> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: %[[SIGNED:.+]] = torch.vtensor.literal(dense<[false, true, false, false]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[UNSIGN]], %[[SIGNED]], %[[SIGNED]]
  %intTensor = torch.vtensor.literal(dense<[127, -128, -127, -126]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %uintTensor = torch.vtensor.literal(dense<[127, 128, 129, 130]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  %fpTensor = torch.vtensor.literal(dense<[127.0, 128.0, 129.0, 130.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %intScalar = torch.constant.int 128
  %fpScalar = torch.constant.float 128.0
  %intBool = torch.aten.eq.Scalar %intTensor, %intScalar : !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4],i1>
  %uintBool = torch.aten.eq.Scalar %uintTensor, %intScalar : !torch.vtensor<[4],ui8>, !torch.int -> !torch.vtensor<[4],i1>
  %fpBool = torch.aten.eq.Scalar %fpTensor, %fpScalar : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],i1>
  return %intBool, %uintBool, %fpBool : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_tensor_tensor_ne
func.func @aten_tensor_tensor_ne() -> (!torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>) {
  // CHECK: %[[UNSIGN:.+]] = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: %[[SIGNED:.+]] = torch.vtensor.literal(dense<[true, false, true, true]> : tensor<4xi1>) : !torch.vtensor<[4],i1>
  // CHECK: return %[[UNSIGN]], %[[SIGNED]], %[[SIGNED]]
  %intTensor = torch.vtensor.literal(dense<[127, -128, -127, -126]> : tensor<4xsi8>) : !torch.vtensor<[4],si8>
  %uintTensor = torch.vtensor.literal(dense<[127, 128, 129, 130]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  %fpTensor = torch.vtensor.literal(dense<[127.0, 128.0, 129.0, 130.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %intScalar = torch.constant.int 128
  %fpScalar = torch.constant.float 128.0
  %intBool = torch.aten.ne.Scalar %intTensor, %intScalar : !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4],i1>
  %uintBool = torch.aten.ne.Scalar %uintTensor, %intScalar : !torch.vtensor<[4],ui8>, !torch.int -> !torch.vtensor<[4],i1>
  %fpBool = torch.aten.ne.Scalar %fpTensor, %fpScalar : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],i1>
  return %intBool, %uintBool, %fpBool : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: @aten_log$fold_splat_i1
func.func @aten_log$fold_splat_i1() -> !torch.vtensor<[4], f32> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  // CHECK: return %[[RET]] : !torch.vtensor<[4],f32>
  %cst = torch.vtensor.literal(dense<true> : tensor<4xi1>) : !torch.vtensor<[4], i1>
  %result = torch.aten.log %cst : !torch.vtensor<[4], i1> -> !torch.vtensor<[4], f32>
  return %result : !torch.vtensor<[4], f32>
}

// -----

// CHECK-LABEL: @aten_log$fold_splat_si32
func.func @aten_log$fold_splat_si32() -> !torch.vtensor<[4], f32> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<1.09861231> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  // CHECK: return %[[RET]] : !torch.vtensor<[4],f32>
  %cst = torch.vtensor.literal(dense<3> : tensor<4xsi32>) : !torch.vtensor<[4], si32>
  %result = torch.aten.log %cst : !torch.vtensor<[4], si32> -> !torch.vtensor<[4], f32>
  return %result : !torch.vtensor<[4], f32>
}

// -----

// CHECK-LABEL: @aten_log$fold_splat_f32
func.func @aten_log$fold_splat_f32() -> !torch.vtensor<[4], f32> {
  // CHECK: %[[RET:.+]] = torch.vtensor.literal(dense<1.09861231> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  // CHECK: return %[[RET]] : !torch.vtensor<[4],f32>
  %cst = torch.vtensor.literal(dense<3.0> : tensor<4xf32>) : !torch.vtensor<[4], f32>
  %result = torch.aten.log %cst : !torch.vtensor<[4], f32> -> !torch.vtensor<[4], f32>
  return %result : !torch.vtensor<[4], f32>
}

// -----

// CHECK-LABEL:   func.func @torch.prims.convert_element_type$fold(
// CHECK:            %[[ARG:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[64],f32> {
// CHECK:      return %[[ARG]] : !torch.vtensor<[64],f32>
func.func @torch.prims.convert_element_type$fold(%arg0: !torch.vtensor<[64],f32>) -> !torch.vtensor<[64],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.prims.convert_element_type %arg0, %int6 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],f32>
  return %0 : !torch.vtensor<[64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.prims.convert_element_type$no_fold(
// CHECK:            %[[ARG:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[64],si32> {
// CHECK:      %[[RET:.*]] = torch.prims.convert_element_type %[[ARG]], %{{.*}} : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],si32>
// CHECK:      return %[[RET]] : !torch.vtensor<[64],si32>
func.func @torch.prims.convert_element_type$no_fold(%arg0: !torch.vtensor<[64],f32>) -> !torch.vtensor<[64],si32> {
  %int6 = torch.constant.int 6
  %0 = torch.prims.convert_element_type %arg0, %int6 : !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[64],si32>
  return %0 : !torch.vtensor<[64],si32>
}

// -----

// CHECK-LABEL:   @torch.aten.max_pool2d_with_indices$canonicalize(
// CHECK:            %[[ARG:.*]]: !torch.vtensor<[10,64,112,112],f32>) -> !torch.vtensor<[10,64,56,56],f32> {
// CHECK:      %[[RET:.*]] = torch.aten.max_pool2d %[[ARG]]
// CHECK:      return %[[RET]] : !torch.vtensor<[10,64,56,56],f32>
func.func @torch.aten.max_pool2d_with_indices$canonicalize(%arg0: !torch.vtensor<[10,64,112,112],f32>) -> !torch.vtensor<[10,64,56,56],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %29 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %30 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %31 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %29, %30, %31, %31, %false : !torch.vtensor<[10,64,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[10,64,56,56],f32>, !torch.vtensor<[10,64,56,56],si64>
  return %result0 : !torch.vtensor<[10,64,56,56],f32>
}

// -----

// CHECK-LABEL:   @torch.aten.max_pool3d_with_indices$canonicalize(
// CHECK:            %[[ARG:.*]]: !torch.vtensor<[10,64,112,112,112],f32>) -> !torch.vtensor<[10,64,56,56,56],f32> {
// CHECK:      %[[RET:.*]] = torch.aten.max_pool3d %[[ARG]]
// CHECK:      return %[[RET]] : !torch.vtensor<[10,64,56,56,56],f32>
func.func @torch.aten.max_pool3d_with_indices$canonicalize(%arg0: !torch.vtensor<[10,64,112,112,112],f32>) -> !torch.vtensor<[10,64,56,56,56],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %29 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %30 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %31 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool3d_with_indices %arg0, %29, %30, %31, %31, %false : !torch.vtensor<[10,64,112,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[10,64,56,56,56],f32>, !torch.vtensor<[10,64,56,56,56],si64>
  return %result0 : !torch.vtensor<[10,64,56,56,56],f32>
}

// -----

// CHECK-LABEL:   @torch.aten.clone$no_fold(
func.func @torch.aten.clone$no_fold(%arg0: !torch.vtensor<[1,2,50,4],f32>) -> (!torch.tensor) {
  // CHECK: %{{.*}} = torch.aten.clone %{{.*}}, %{{.*}} : !torch.vtensor<[1,2,50,4],f32>, !torch.none -> !torch.vtensor
  %none = torch.constant.none
  %0 = torch.aten.clone %arg0, %none : !torch.vtensor<[1,2,50,4],f32>, !torch.none -> !torch.vtensor
  %1 = torch.copy.to_tensor %0 : !torch.tensor
  return %1 : !torch.tensor
}

// -----

// CHECK-LABEL:   @torch.symbolic_int$canonicalize(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:        %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:               %[[S0:.*]] = torch.symbolic_int "s0" {min_val = 3, max_val = 6} : !torch.int
// CHECK-NOT:           %[[S1:.*]] = torch.symbolic_int "s0 + 1" {min_val = 4, max_val = 7} : !torch.int
// CHECK:               torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
// CHECK:               torch.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0 + 1)> : !torch.vtensor<[?],f32>
// CHECK:               %[[V1:.*]] = torch.aten.slice.Tensor %[[ARG1]], {{.*}}, {{.*}}, {{.*}}, {{.*}} : !torch.vtensor<[?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:               torch.bind_symbolic_shape %[[V1]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
// CHECK:               %[[V2:.*]] = torch.aten.add.Tensor %[[ARG0]], {{.*}}, {{.*}} : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:               torch.bind_symbolic_shape %[[V2]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
// CHECK:               return %[[V2]] : !torch.vtensor<[?],f32>
func.func @torch.symbolic_int$canonicalize(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.symbolic_int "s0" {min_val = 3, max_val = 6} : !torch.int
  %1 = torch.symbolic_int "s0 + 1" {min_val = 4, max_val = 7} : !torch.int
  torch.bind_symbolic_shape %arg0, [%0], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
  torch.bind_symbolic_shape %arg1, [%0], affine_map<()[s0] -> (s0 + 1)> : !torch.vtensor<[?],f32>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %int1_0 = torch.constant.int 1
  %2 = torch.aten.slice.Tensor %arg1, %int0, %int1, %int9223372036854775807, %int1_0 : !torch.vtensor<[?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
  torch.bind_symbolic_shape %2, [%0], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
  %int1_1 = torch.constant.int 1
  %3 = torch.aten.add.Tensor %arg0, %2, %int1_1 : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],f32>
  torch.bind_symbolic_shape %3, [%0], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
  return %3 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:   func.func @ttorch.aten.ones$float_fold() -> !torch.vtensor<[2,3,4],f32> {
// CHECK:           %[[VAL_0:.*]] = torch.vtensor.literal(dense<1.000000e+00> : tensor<2x3x4xf32>) : !torch.vtensor<[2,3,4],f32>
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[2,3,4],f32>
// CHECK:         }
func.func @ttorch.aten.ones$float_fold() -> !torch.vtensor<[2,3,4],f32> {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.ones %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3,4],f32>
  return %1 : !torch.vtensor<[2,3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @ttorch.aten.ones$int_fold() -> !torch.vtensor<[2,3,4],si64> {
// CHECK:           %[[VAL_0:.*]] = torch.vtensor.literal(dense<1> : tensor<2x3x4xsi64>) : !torch.vtensor<[2,3,4],si64>
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[2,3,4],si64>
// CHECK:         }
func.func @ttorch.aten.ones$int_fold() -> !torch.vtensor<[2,3,4],si64> {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.ones %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3,4],si64>
  return %1 : !torch.vtensor<[2,3,4],si64>
}

// -----

// CHECK-LABEL:   func.func @test_aten_zeros$float_fold() -> !torch.vtensor<[2,3,4],f32> {
// CHECK:           %[[VAL_0:.*]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<2x3x4xf32>) : !torch.vtensor<[2,3,4],f32>
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[2,3,4],f32>
// CHECK:         }
func.func @test_aten_zeros$float_fold() -> !torch.vtensor<[2,3,4],f32> {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.zeros %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3,4],f32>
  return %1 : !torch.vtensor<[2,3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @test_aten_zeros$int_fold() -> !torch.vtensor<[2,3,4],si64> {
// CHECK:           %[[VAL_0:.*]] = torch.vtensor.literal(dense<0> : tensor<2x3x4xsi64>) : !torch.vtensor<[2,3,4],si64>
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[2,3,4],si64>
// CHECK:         }
func.func @test_aten_zeros$int_fold() -> !torch.vtensor<[2,3,4],si64> {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.zeros %0, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3,4],si64>
  return %1 : !torch.vtensor<[2,3,4],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.full$float_fold() -> !torch.vtensor<[2,1,4],f32> {
// CHECK:           %[[VAL_0:.*]] = torch.vtensor.literal(dense<0xFF800000> : tensor<2x1x4xf32>) : !torch.vtensor<[2,1,4],f32>
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[2,1,4],f32>
// CHECK:         }
func.func @torch.aten.full$float_fold() -> !torch.vtensor<[2,1,4],f32> {
  %float-Inf = torch.constant.float 0xFFF0000000000000
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int1, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %float-Inf, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,1,4],f32>
  return %1 : !torch.vtensor<[2,1,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.full$int_fold() -> !torch.vtensor<[2,1,4],si64> {
// CHECK:           %[[VAL_0:.*]] = torch.vtensor.literal(dense<0> : tensor<2x1x4xsi64>) : !torch.vtensor<[2,1,4],si64>
// CHECK:           return %[[VAL_0]] : !torch.vtensor<[2,1,4],si64>
// CHECK:         }
func.func @torch.aten.full$int_fold() -> !torch.vtensor<[2,1,4],si64> {
  %int-Inf = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int1, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %int-Inf, %none, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,1,4],si64>
  return %1 : !torch.vtensor<[2,1,4],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.avg_pool2d.single_int_tuple(
// CHECK-SAME:      %[[ARG0:.*]]: !torch.vtensor<[2,4,20,20],f32>) -> !torch.vtensor<[2,4,9,9],f32> {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[C_6:.*]] = torch.constant.int 6
// CHECK:           %[[C_1:.*]] = torch.constant.int 1
// CHECK:           %[[C_2:.*]] = torch.constant.int 2
// CHECK:           %[[KERNEL:.*]] = torch.prim.ListConstruct %[[C_6]], %[[C_6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C_1]], %[[C_1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[PAD:.*]] = torch.prim.ListConstruct %[[C_2]], %[[C_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[POOL:.*]] = torch.aten.avg_pool2d %[[ARG0]], %[[KERNEL]], %[[PAD]], %[[STRIDE]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[2,4,20,20],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,4,9,9],f32>
// CHECK:           return %[[POOL]]
func.func @torch.aten.avg_pool2d.single_int_tuple(%arg0: !torch.vtensor<[2,4,20,20],f32>) -> !torch.vtensor<[2,4,9,9],f32> {
    %int6 = torch.constant.int 6
    %0 = torch.prim.ListConstruct %int6 : (!torch.int) -> !torch.list<int>
    %int2 = torch.constant.int 2
    %1 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %none = torch.constant.none
    %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %false_0, %none : !torch.vtensor<[2,4,20,20],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,4,9,9],f32>
    return %3 : !torch.vtensor<[2,4,9,9],f32>
  }
