// RUN: npcomp-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func @torch.aten.__is__
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.__is__(%arg0: !torch.list<!torch.int>, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__is__ %arg0, %arg1 : !torch.list<!torch.int>, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.__is__$none_is_none
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.__is__$none_is_none(%arg0: !torch.none, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__is__ %arg0, %arg1 : !torch.none, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.__isnot__
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.__isnot__(%arg0: !torch.list<!torch.int>, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__isnot__ %arg0, %arg1 : !torch.list<!torch.int>, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.__isnot__$none_isnot_none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.__isnot__$none_isnot_none(%arg0: !torch.none, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__isnot__ %arg0, %arg1 : !torch.none, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.size$canonicalize_to_list(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.list<!torch.int> {
// CHECK:           %[[C2:.*]] = torch.constant.int 2
// CHECK:           %[[C3:.*]] = torch.constant.int 3
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C2]], %[[C3]] : (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           return %[[LIST]] : !torch.list<!torch.int>
func @torch.aten.size$canonicalize_to_list(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.list<!torch.int> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[2,3],f32> -> !torch.list<!torch.int>
  return %0 : !torch.list<!torch.int>
}

// One size unknown, so cannot canonicalize.
// TODO: For unknown sizes, insert the equivalent of a "dim" op.
// Then this will only require static rank.
// CHECK-LABEL:   func @torch.aten.size$unknown_size(
// CHECK-SAME:                                       %[[ARG:.*]]: !torch.vtensor<[?,3],f32>) -> !torch.list<!torch.int> {
// CHECK:           %[[SIZE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor<[?,3],f32> -> !torch.list<!torch.int>
func @torch.aten.size$unknown_size(%arg0: !torch.vtensor<[?,3],f32>) -> !torch.list<!torch.int> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[?,3],f32> -> !torch.list<!torch.int>
  return %0 : !torch.list<!torch.int>
}

// CHECK-LABEL:   func @torch.aten.ne.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK-NEXT:       %[[FALSE:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[FALSE]] : !torch.bool
func @torch.aten.ne.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.aten.ne.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.ne.int$same_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.ne.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.ne.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.ne.int$different_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.ne.int$different_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.ne.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.eq.int$different_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.eq.int$different_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.eq.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.eq.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK-NEXT:       %[[F:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[F]] : !torch.bool
func @torch.aten.eq.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.aten.eq.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.eq.int$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.eq.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.eq.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.lt.int$evaluate_to_true() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.lt.int$evaluate_to_true() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.lt.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.lt.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.lt.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %2 = torch.aten.lt.int %arg0, %arg0: !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.lt.int$same_value() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.lt.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.lt.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.le.int$evaluate_to_true() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.le.int$evaluate_to_true() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.le.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.le.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.le.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %2 = torch.aten.le.int %arg0, %arg0: !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.le.int$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.le.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.le.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.gt.int$evaluate_to_true() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[T]] : !torch.bool
func @torch.aten.gt.int$evaluate_to_true() -> !torch.bool {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.gt.int %int4, %int2 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.gt.int$evaluate_to_false() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[T]] : !torch.bool
func @torch.aten.gt.int$evaluate_to_false() -> !torch.bool {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.gt.int %int2, %int4 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.ge.int$evaluate_to_false() -> !torch.bool {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.ge.int$evaluate_to_false() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %2 = torch.aten.ge.int %int4, %int5 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.ge.int$same_operand(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.ge.int$same_operand(%arg0: !torch.int) -> !torch.bool {
  %2 = torch.aten.ge.int %arg0, %arg0: !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.ge.int$same_value() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.ge.int$same_value() -> !torch.bool {
  %int4 = torch.constant.int 4
  %int4_0 = torch.constant.int 4
  %2 = torch.aten.ge.int %int4, %int4_0 : !torch.int, !torch.int -> !torch.bool
  return %2 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.__not__
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.__not__() -> !torch.bool {
  %false = torch.constant.bool false
  %ret = torch.aten.__not__ %false : !torch.bool -> !torch.bool
  return %ret: !torch.bool
}

// CHECK-LABEL:   func @torch.aten.len.t$of_size(
// CHECK-SAME:                                   %[[ARG:.*]]: !torch.vtensor<*,f32>) -> !torch.int {
// CHECK:           %[[DIM:.*]] = torch.aten.dim %[[ARG]] : !torch.vtensor<*,f32> -> !torch.int
// CHECK:           return %[[DIM]] : !torch.int
func @torch.aten.len.t$of_size(%arg0: !torch.vtensor<*,f32>) -> !torch.int {
  %0 = torch.aten.size %arg0 : !torch.vtensor<*,f32> -> !torch.list<!torch.int>
  %1 = torch.aten.len.t %0 : !torch.list<!torch.int> -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func @torch.aten.dim$with_shape(
// CHECK-SAME:                                    %[[ARG:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.int {
// CHECK:           %[[DIM:.*]] = torch.constant.int 3
// CHECK:           return %[[DIM]] : !torch.int
func @torch.aten.dim$with_shape(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.int {
  %0 = torch.aten.dim %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func @torch.aten.len.t$of_build_list(
// CHECK-SAME:                                         %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK:           %[[LEN:.*]] = torch.constant.int 4
// CHECK:           return %[[LEN]] : !torch.int
func @torch.aten.len.t$of_build_list(%arg0: !torch.int) -> !torch.int {
  %0 = torch.prim.ListConstruct %arg0, %arg0, %arg0, %arg0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
  %1 = torch.aten.len.t %0 : !torch.list<!torch.int> -> !torch.int
  return %1 : !torch.int
}

// CHECK-LABEL:   func @torch.aten.__getitem__.t(
// CHECK:           %[[C5:.*]] = torch.constant.int 5
// CHECK:           return %[[C5]] : !torch.int
func @torch.aten.__getitem__.t() -> !torch.int {
    %int4 = torch.constant.int 4
    %int5 = torch.constant.int 5
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %1 = torch.aten.__getitem__.t %0, %int1 : !torch.list<!torch.int>, !torch.int -> !torch.int
    return %1 : !torch.int
}

// Not canonicalized because of passed in index
// CHECK-LABEL:   func @torch.aten.__getitem__.t$no_change_test0(
// CHECK:           %[[C4:.*]] = torch.constant.int 4
// CHECK:           %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C4]], %[[C5]] : (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %[[LIST]], %arg0 : !torch.list<!torch.int>, !torch.int -> !torch.int
// CHECK:           return %[[ITEM]] : !torch.int
func @torch.aten.__getitem__.t$no_change_test0(%arg0: !torch.int) -> !torch.int {
  %int5 = torch.constant.int 5
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %1 = torch.aten.__getitem__.t %0, %arg0 : !torch.list<!torch.int>, !torch.int -> !torch.int
  return %1 : !torch.int
}

// Not canonicalized because of passed in list
// CHECK-LABEL:   func @torch.aten.__getitem__.t$no_change_test1(
// CHECK:           %[[C5:.*]] = torch.constant.int 5
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %arg0, %[[C5]] : !torch.list<!torch.int>, !torch.int -> !torch.int
// CHECK:           return %[[ITEM]] : !torch.int
func @torch.aten.__getitem__.t$no_change_test1(%arg0: !torch.list<!torch.int>) -> !torch.int {
  %int5 = torch.constant.int 5
  %0 = torch.aten.__getitem__.t %arg0, %int5 : !torch.list<!torch.int>, !torch.int -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func @torch.constant.none$constantlike() -> (!torch.none, !torch.none) {
// CHECK:           %[[C:.*]] = torch.constant.none
// CHECK:           return %[[C]], %[[C]] : !torch.none, !torch.none
func @torch.constant.none$constantlike() -> (!torch.none, !torch.none) {
  %0 = torch.constant.none
  %1 = torch.constant.none
  return %0, %1 : !torch.none, !torch.none
}

// CHECK-LABEL:   func @torch.constant.str$constantlike() -> (!torch.str, !torch.str, !torch.str) {
// CHECK:           %[[T:.*]] = torch.constant.str "t"
// CHECK:           %[[S:.*]] = torch.constant.str "s"
// CHECK:           return %[[S]], %[[S]], %[[T]] : !torch.str, !torch.str, !torch.str
func @torch.constant.str$constantlike() -> (!torch.str, !torch.str, !torch.str) {
  %0 = torch.constant.str "s"
  %1 = torch.constant.str "s"
  %2 = torch.constant.str "t"
  return %0, %1, %2 : !torch.str, !torch.str, !torch.str
}

// CHECK-LABEL:   func @torch.constant.bool$constantlike() -> (!torch.bool, !torch.bool, !torch.bool) {
// CHECK:           %[[F:.*]] = torch.constant.bool false
// CHECK:           %[[T:.*]] = torch.constant.bool true
// CHECK:           return %[[T]], %[[T]], %[[F]] : !torch.bool, !torch.bool, !torch.bool
func @torch.constant.bool$constantlike() -> (!torch.bool, !torch.bool, !torch.bool) {
  %0 = torch.constant.bool true
  %1 = torch.constant.bool true
  %2 = torch.constant.bool false
  return %0, %1, %2 : !torch.bool, !torch.bool, !torch.bool
}

// CHECK-LABEL:   func @torch.prim.If$erase_dead_branch(
// CHECK-SAME:                                          %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK-NEXT:       %[[RET:.*]] = torch.aten.add.int %[[ARG]], %[[ARG]] : !torch.int, !torch.int -> !torch.int
// CHECK-NEXT:       return %[[RET]] : !torch.int
func @torch.prim.If$erase_dead_branch(%arg0: !torch.int) -> !torch.int {
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

// CHECK-LABEL:   builtin.func @torch.prim.TupleUnpack(
// CHECK-SAME:                                         %[[ARG0:.*]]: !torch.tensor,
// CHECK-SAME:                                         %[[ARG1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[ARG0]] : !torch.tensor
func @torch.prim.TupleUnpack(%arg0: !torch.tensor, %arg1: !torch.tensor) -> !torch.tensor{
  %123 = torch.prim.TupleConstruct %arg0, %arg1: !torch.tensor, !torch.tensor -> !torch.tuple<!torch.tensor, !torch.tensor>
  %124:2 = torch.prim.TupleUnpack %123 : !torch.tuple<!torch.tensor, !torch.tensor> -> !torch.tensor, !torch.tensor
  return %124#0 : !torch.tensor
}


// CHECK-LABEL:   builtin.func @torch.aten.__contains__.str(
// CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
// CHECK-SAME:        %[[K1:.*]]: !torch.str,
// CHECK-SAME:        %[[V1:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
// CHECK-SAME:        keys(%[[K0]], %[[K1]] : !torch.str, !torch.str)
// CHECK-SAME:        values(%[[V0]], %[[V1]] : !torch.tensor, !torch.tensor)
// CHECK-SAME:        -> !torch.dict<!torch.str, !torch.tensor>
// CHECK:           return %[[TRUE]] : !torch.bool
func @torch.aten.__contains__.str(%k0 : !torch.str, %v0: !torch.tensor, %k1: !torch.str, %v1: !torch.tensor) -> !torch.bool{
  %dict = torch.prim.DictConstruct keys(%k0, %k1: !torch.str, !torch.str) values(%v0,  %v1: !torch.tensor, !torch.tensor) -> !torch.dict<!torch.str, !torch.tensor>
  %pred = torch.aten.__contains__.str %dict, %k0 : !torch.dict<!torch.str, !torch.tensor>, !torch.str -> !torch.bool
  return %pred : !torch.bool
}

// CHECK-LABEL:   builtin.func @torch.aten.__contains__.str$with_dict_modified(
// CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
// CHECK-SAME:        %[[K1:.*]]: !torch.str, %[[V1:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
// CHECK-SAME:        keys(%[[K0]], %[[K1]] : !torch.str, !torch.str)
// CHECK-SAME:        values(%[[V0]], %[[V1]] : !torch.tensor, !torch.tensor)
// CHECK-SAME:        -> !torch.dict<!torch.str, !torch.tensor>
// CHECK:           torch.aten._set_item.str %[[DICT]], %[[K0]], %[[V1]] :
// CHECK-SAME:        !torch.dict<!torch.str, !torch.tensor>, !torch.str, !torch.tensor
// CHECK:           %[[RET:.*]] = torch.aten.__contains__.str %[[DICT]], %[[K0]] :
// CHECK-SAME:        !torch.dict<!torch.str, !torch.tensor>, !torch.str -> !torch.bool
// CHECK:           return %[[RET]] : !torch.bool

func @torch.aten.__contains__.str$with_dict_modified(%k0 : !torch.str, %v0: !torch.tensor, %k1: !torch.str, %v1: !torch.tensor) -> !torch.bool{
  %dict = torch.prim.DictConstruct keys(%k0, %k1: !torch.str, !torch.str) values(%v0,  %v1: !torch.tensor, !torch.tensor) -> !torch.dict<!torch.str, !torch.tensor>
  torch.aten._set_item.str %dict, %k0, %v1 : !torch.dict<!torch.str, !torch.tensor>, !torch.str, !torch.tensor
  %pred = torch.aten.__contains__.str %dict, %k0 : !torch.dict<!torch.str, !torch.tensor>, !torch.str -> !torch.bool
  return %pred : !torch.bool
}

// CHECK-LABEL:   builtin.func @torch.aten.__getitem__.Dict_str(
// CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
// CHECK-SAME:        %[[K1:.*]]: !torch.str, %[[V1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
// CHECK-SAME:        keys(%[[K0]], %[[K1]] : !torch.str, !torch.str)
// CHECK-SAME:        values(%[[V0]], %[[V1]] : !torch.tensor, !torch.tensor)
// CHECK-SAME:        -> !torch.dict<!torch.str, !torch.tensor>
// CHECK:           return %[[V0]] : !torch.tensor
func @torch.aten.__getitem__.Dict_str(%k0 : !torch.str, %v0: !torch.tensor, %k1: !torch.str, %v1: !torch.tensor) -> !torch.tensor {
  %dict = torch.prim.DictConstruct keys(%k0, %k1: !torch.str, !torch.str) values(%v0,  %v1: !torch.tensor, !torch.tensor) -> !torch.dict<!torch.str, !torch.tensor>
  %v = torch.aten.__getitem__.Dict_str %dict, %k0 : !torch.dict<!torch.str, !torch.tensor>, !torch.str -> !torch.tensor
  return %v : !torch.tensor
}

// CHECK-LABEL:   builtin.func @torch.aten.add.int() -> !torch.int {
// CHECK:           %[[CST9:.*]] = torch.constant.int 9
// CHECK:           return %[[CST9]] : !torch.int
// CHECK:         }
func @torch.aten.add.int() -> !torch.int {
    %cst4 = torch.constant.int 4
    %cst5 = torch.constant.int 5
    %ret = torch.aten.add.int %cst4, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   builtin.func @torch.aten.sub.int() -> !torch.int {
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           return %[[CST1]] : !torch.int
// CHECK:         }
func @torch.aten.sub.int() -> !torch.int {
    %cst6 = torch.constant.int 6
    %cst5 = torch.constant.int 5
    %ret = torch.aten.sub.int %cst6, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   builtin.func @torch.aten.mul.int() -> !torch.int {
// CHECK:           %[[CST30:.*]] = torch.constant.int 30
// CHECK:           return %[[CST30]] : !torch.int
// CHECK:         }
func @torch.aten.mul.int() -> !torch.int {
    %cst6 = torch.constant.int 6
    %cst5 = torch.constant.int 5
    %ret = torch.aten.mul.int %cst6, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   builtin.func @torch.aten.mul.int$with_zero() -> !torch.int {
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           return %[[CST0]] : !torch.int
// CHECK:         }
func @torch.aten.mul.int$with_zero() -> !torch.int {
    %cst6 = torch.constant.int 6
    %cst0 = torch.constant.int 0
    %ret = torch.aten.mul.int %cst6, %cst0: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   builtin.func @torch.aten.floordiv.int() -> !torch.int {
// CHECK:           %[[CST3:.*]] = torch.constant.int 3
// CHECK:           return %[[CST3]] : !torch.int
// CHECK:         }
func @torch.aten.floordiv.int() -> !torch.int {
    %cst18 = torch.constant.int 18
    %cst5 = torch.constant.int 5
    %ret = torch.aten.floordiv.int %cst18, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}

// CHECK-LABEL:   builtin.func @torch.aten.remainder.int() -> !torch.int {
// CHECK:           %[[CST3:.*]] = torch.constant.int 3
// CHECK:           return %[[CST3]] : !torch.int
// CHECK:         }
func @torch.aten.remainder.int() -> !torch.int {
    %cst18 = torch.constant.int 18
    %cst5 = torch.constant.int 5
    %ret = torch.aten.remainder.int %cst18, %cst5: !torch.int, !torch.int -> !torch.int
    return %ret : !torch.int
}
