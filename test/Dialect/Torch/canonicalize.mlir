// RUN: npcomp-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func @torch.aten.__is__
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.__is__(%arg0: !torch.list<!torch.int>, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__is__ %arg0, %arg1 : !torch.list<!torch.int>, !torch.none -> !torch.bool
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

// CHECK-LABEL:   func @torch.aten.gt.int$evaluate() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[T]] : !torch.bool
func @torch.aten.gt.int$evaluate() -> !torch.bool {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.gt.int %int4, %int2 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.ne.int$same_value(
// CHECK-SAME:                                       %{{.*}}: !torch.int) -> !torch.bool {
// CHECK-NEXT:       %[[F:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[F]] : !torch.bool
func @torch.aten.ne.int$same_value(%arg0: !torch.int) -> !torch.bool {
  %0 = torch.aten.ne.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
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
