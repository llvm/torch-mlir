// RUN: torch-mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.__range_length$fold() -> (!torch.int, !torch.int, !torch.int, !torch.int) {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[INTM1:.*]] = torch.constant.int -1
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
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
// CHECK-NEXT:       %[[F:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[F]] : !torch.bool
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
// CHECK:           %[[C4:.*]] = torch.constant.int 4
// CHECK:           %[[C5:.*]] = torch.constant.int 5
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
// CHECK:           %[[T:.*]] = torch.constant.str "t"
// CHECK:           %[[S:.*]] = torch.constant.str "s"
// CHECK:           return %[[S]], %[[S]], %[[T]] : !torch.str, !torch.str, !torch.str
func.func @torch.constant.str$constantlike() -> (!torch.str, !torch.str, !torch.str) {
  %0 = torch.constant.str "s"
  %1 = torch.constant.str "s"
  %2 = torch.constant.str "t"
  return %0, %1, %2 : !torch.str, !torch.str, !torch.str
}

// CHECK-LABEL:   func.func @torch.constant.bool$constantlike() -> (!torch.bool, !torch.bool, !torch.bool) {
// CHECK:           %[[F:.*]] = torch.constant.bool false
// CHECK:           %[[T:.*]] = torch.constant.bool true
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

// CHECK-LABEL:   func.func @torch.aten.remainder.int() -> !torch.int {
// CHECK:           %[[CST3:.*]] = torch.constant.int 3
// CHECK:           return %[[CST3]] : !torch.int
func.func @torch.aten.remainder.int() -> !torch.int {
    %cst18 = torch.constant.int 18
    %cst5 = torch.constant.int 5
    %ret = torch.aten.remainder.int %cst18, %cst5: !torch.int, !torch.int -> !torch.int
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
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.tensor<[],f32>
func.func @torch.aten.squeeze$zero_rank(%arg0: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
  %0 = torch.aten.squeeze %arg0 : !torch.tensor<[],f32> -> !torch.tensor<[],f32>
  return %0 : !torch.tensor<[],f32>
}

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim$zero_rank(
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.tensor<[],f32>
func.func @torch.aten.squeeze.dim$zero_rank(%arg0: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.squeeze.dim %arg0, %int0 : !torch.tensor<[],f32>, !torch.int -> !torch.tensor<[],f32>
  return %0 : !torch.tensor<[],f32>
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

// CHECK-LABEL:   func.func @torch.aten.view$1D(
// CHECK-SAME:            %[[ARG:.*]]: !torch.tensor<[?],f32>) -> !torch.tensor<[?],f32> {
// CHECK-NEXT:      return %[[ARG]] : !torch.tensor<[?],f32>
func.func @torch.aten.view$1D(%arg0: !torch.tensor<[?],f32>) -> !torch.tensor<[?],f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.tensor<[?],f32>, !torch.list<int> -> !torch.tensor<[?],f32>
  return %1 : !torch.tensor<[?],f32>
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
