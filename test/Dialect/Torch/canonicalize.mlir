// RUN: npcomp-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func @torch.aten.__is__
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           return %[[FALSE]] : !torch.bool
func @torch.aten.__is__(%arg0: !torch.list<i64>, %arg1: !torch.none) -> !torch.bool {
  %0 = torch.aten.__is__ %arg0, %arg1 : !torch.list<i64>, !torch.none -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.size$canonicalize_to_list(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.list<i64> {
// CHECK:           %[[C2:.*]] = torch.constant.int 2 : i64
// CHECK:           %[[C3:.*]] = torch.constant.int 3 : i64
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C2]], %[[C3]] : (i64, i64) -> !torch.list<i64>
// CHECK:           return %[[LIST]] : !torch.list<i64>
func @torch.aten.size$canonicalize_to_list(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.list<i64> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[2,3],f32> -> !torch.list<i64>
  return %0 : !torch.list<i64>
}

// One size unknown, so cannot canonicalize.
// TODO: For unknown sizes, insert the equivalent of a "dim" op.
// Then this will only require static rank.
// CHECK-LABEL:   func @torch.aten.size$unknown_size(
// CHECK-SAME:                                       %[[ARG:.*]]: !torch.vtensor<[?,3],f32>) -> !torch.list<i64> {
// CHECK:           %[[SIZE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor<[?,3],f32> -> !torch.list<i64>
func @torch.aten.size$unknown_size(%arg0: !torch.vtensor<[?,3],f32>) -> !torch.list<i64> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[?,3],f32> -> !torch.list<i64>
  return %0 : !torch.list<i64>
}

// CHECK-LABEL:   func @torch.aten.gt.int$evaluate() -> !torch.bool {
// CHECK-NEXT:       %[[T:.*]] = torch.constant.bool true
// CHECK-NEXT:       return %[[T]] : !torch.bool
func @torch.aten.gt.int$evaluate() -> !torch.bool {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.gt.int %int4, %int2 : i64, i64 -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.ne.int$same_value(
// CHECK-SAME:                                       %{{.*}}: i64) -> !torch.bool {
// CHECK-NEXT:       %[[F:.*]] = torch.constant.bool false
// CHECK-NEXT:       return %[[F]] : !torch.bool
func @torch.aten.ne.int$same_value(%arg0: i64) -> !torch.bool {
  %0 = torch.aten.ne.int %arg0, %arg0 : i64, i64 -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.len.t$of_size(
// CHECK-SAME:                                   %[[ARG:.*]]: !torch.vtensor<*,f32>) -> i64 {
// CHECK:           %[[DIM:.*]] = torch.aten.dim %[[ARG]] : !torch.vtensor<*,f32> -> i64
// CHECK:           return %[[DIM]] : i64
func @torch.aten.len.t$of_size(%arg0: !torch.vtensor<*,f32>) -> i64 {
  %0 = torch.aten.size %arg0 : !torch.vtensor<*,f32> -> !torch.list<i64>
  %1 = torch.aten.len.t %0 : !torch.list<i64> -> i64
  return %1 : i64
}

// CHECK-LABEL:   func @torch.aten.dim$with_shape(
// CHECK-SAME:                                    %[[ARG:.*]]: !torch.vtensor<[?,?,?],f32>) -> i64 {
// CHECK:           %[[DIM:.*]] = torch.constant.int 3 : i64
// CHECK:           return %[[DIM]] : i64
func @torch.aten.dim$with_shape(%arg0: !torch.vtensor<[?,?,?],f32>) -> i64 {
  %0 = torch.aten.dim %arg0 : !torch.vtensor<[?,?,?],f32> -> i64
  return %0 : i64
}

// CHECK-LABEL:   func @torch.aten.len.t$of_build_list(
// CHECK-SAME:                                         %[[ARG:.*]]: i64) -> i64 {
// CHECK:           %[[LEN:.*]] = torch.constant.int 4 : i64
// CHECK:           return %[[LEN]] : i64
func @torch.aten.len.t$of_build_list(%arg0: i64) -> i64 {
  %0 = torch.prim.ListConstruct %arg0, %arg0, %arg0, %arg0 : (i64, i64, i64, i64) -> !torch.list<i64>
  %1 = torch.aten.len.t %0 : !torch.list<i64> -> i64
  return %1 : i64
}

// CHECK-LABEL:   func @torch.copy.tensor$value_copy_is_noop(
// CHECK-SAME:                                               %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           return %[[ARG]] : !torch.vtensor
func @torch.copy.tensor$value_copy_is_noop(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.tensor %arg0 : !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func @torch.copy.tensor$unnecessary_intermediate_nonval_tensor(
// CHECK-SAME:                                                                    %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           return %[[ARG]] : !torch.vtensor
func @torch.copy.tensor$unnecessary_intermediate_nonval_tensor(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.tensor %arg0 : !torch.vtensor -> !torch.tensor
  %1 = torch.copy.tensor %0 : !torch.tensor -> !torch.vtensor
  return %1 : !torch.vtensor
}

// CHECK-LABEL:   func @torch.aten.__getitem__.t(
// CHECK:           %[[C5:.*]] = torch.constant.int 5 : i64
// CHECK:           return %[[C5]] : i64
func @torch.aten.__getitem__.t() -> i64 {
    %c4_i64 = torch.constant.int 4 : i64
    %c5_i64 = torch.constant.int 5 : i64
    %c1_i64 = torch.constant.int 1 : i64
    %0 = torch.prim.ListConstruct %c4_i64, %c5_i64 : (i64, i64) -> !torch.list<i64>
    %1 = torch.aten.__getitem__.t %0, %c1_i64 : !torch.list<i64>, i64 -> i64
    return %1 : i64
}

// Not canonicalized because of passed in index
// CHECK-LABEL:   func @torch.aten.__getitem__.t$no_change_test0(
// CHECK:           %[[C4:.*]] = torch.constant.int 4 : i64
// CHECK:           %[[C5:.*]] = torch.constant.int 5 : i64
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[C4]], %[[C5]] : (i64, i64) -> !torch.list<i64>
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %[[LIST]], %arg0 : !torch.list<i64>, i64 -> i64
// CHECK:           return %[[ITEM]] : i64
func @torch.aten.__getitem__.t$no_change_test0(%arg0: i64) -> i64 {
  %c5_i64 = torch.constant.int 5 : i64
  %c4_i64 = torch.constant.int 4 : i64
  %0 = torch.prim.ListConstruct %c4_i64, %c5_i64 : (i64, i64) -> !torch.list<i64>
  %1 = torch.aten.__getitem__.t %0, %arg0 : !torch.list<i64>, i64 -> i64
  return %1 : i64
}

// Not canonicalized because of passed in list
// CHECK-LABEL:   func @torch.aten.__getitem__.t$no_change_test1(
// CHECK:           %[[C5:.*]] = torch.constant.int 5 : i64
// CHECK:           %[[ITEM:.*]] = torch.aten.__getitem__.t %arg0, %[[C5]] : !torch.list<i64>, i64 -> i64
// CHECK:           return %[[ITEM]] : i64
func @torch.aten.__getitem__.t$no_change_test1(%arg0: !torch.list<i64>) -> i64 {
  %c5_i64 = torch.constant.int 5 : i64
  %0 = torch.aten.__getitem__.t %arg0, %c5_i64 : !torch.list<i64>, i64 -> i64
  return %0 : i64
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
// CHECK-SAME:                                          %[[ARG:.*]]: i64) -> i64 {
// CHECK-NEXT:       %[[RET:.*]] = torch.aten.add.int %[[ARG]], %[[ARG]] : i64, i64 -> i64
// CHECK-NEXT:       return %[[RET]] : i64
func @torch.prim.If$erase_dead_branch(%arg0: i64) -> i64 {
  %true = torch.constant.bool true
  %0 = torch.prim.If %true -> (i64) {
    %1 = torch.aten.add.int %arg0, %arg0 : i64, i64 -> i64
    torch.prim.If.yield %1 : i64
  } else {
    %1 = torch.aten.mul.int %arg0, %arg0 : i64, i64 -> i64
    torch.prim.If.yield %1 : i64
  }
  return %0 : i64
}
