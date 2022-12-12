// RUN: torch-mlir-opt -torch-drop-abstract-interp-calculations -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @basic$shape_calculate(
// CHECK-SAME:                 %[[ARG:.*]]: !torch.vtensor<[2,?],unk>) -> !torch.vtensor {
// CHECK:           %[[TANH:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor<[2,?],unk> -> !torch.vtensor<[2,?],unk>
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[TANH]] : !torch.vtensor<[2,?],unk> to !torch.vtensor
// CHECK:           return %[[ERASED]] : !torch.vtensor
func.func @basic$shape_calculate(%arg0: !torch.vtensor<[2,?],unk>) -> !torch.vtensor {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.shape.calculate  {
    %2 = torch.aten.tanh %arg0 : !torch.vtensor<[2,?],unk> -> !torch.vtensor<[2,?],unk>
    torch.shape.calculate.yield %2 : !torch.vtensor<[2,?],unk>
  } shapes  {
    %2 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[2,?],unk>, !torch.int -> !torch.int
    %3 = torch.prim.ListConstruct %int2, %2 : (!torch.int, !torch.int) -> !torch.list<int>
    torch.shape.calculate.yield.shapes %3 : !torch.list<int>
  } : !torch.vtensor<[2,?],unk>
  %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<[2,?],unk> to !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func @basic$dtype_calculate(
// CHECK-SAME:                     %[[ARG:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor {
// CHECK:           %[[INT_6:.*]] = torch.constant.int 6
// CHECK:           %[[TANH:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[TANH]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @basic$dtype_calculate(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %int6 = torch.constant.int 6
  %0 = torch.dtype.calculate {
    %2 = torch.aten.tanh %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
    torch.dtype.calculate.yield %2 : !torch.vtensor<*,f32>
  } dtypes {
    torch.dtype.calculate.yield.dtypes %int6 : !torch.int
  } : !torch.vtensor<*,f32>
  %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<*,f32> to !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func @shape_calc_in_loop(
// CHECK-SAME:                 %[[ARG:.*]]: !torch.vtensor<[2,?],unk>) -> !torch.vtensor<[2,?],unk> {
func.func @shape_calc_in_loop(%arg: !torch.vtensor<[2,?],unk>) -> !torch.vtensor<[2,?],unk> {
  %one = torch.constant.int 1
  // CHECK: %[[ONE:.*]] = torch.constant.int 1

  %two = torch.constant.int 2
  // CHECK: %[[TWO:.*]] = torch.constant.int 2

  %true = torch.constant.bool true
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true

  torch.prim.Loop %one, %true, init() {
  // CHECK: torch.prim.Loop %[[ONE]], %[[TRUE]], init() {

    ^bb0(%in: !torch.int):
      %shape_calc = torch.shape.calculate {
        %tanh = torch.aten.tanh %arg : !torch.vtensor<[2,?],unk> -> !torch.vtensor<[2,?],unk>
        torch.shape.calculate.yield %tanh : !torch.vtensor<[2,?],unk>
      } shapes {
        %size = torch.aten.size.int %arg, %one : !torch.vtensor<[2,?],unk>, !torch.int -> !torch.int
        %list = torch.prim.ListConstruct %two, %size : (!torch.int, !torch.int) -> !torch.list<int>
        torch.shape.calculate.yield.shapes %list : !torch.list<int>
      } : !torch.vtensor<[2,?],unk>
    // CHECK: torch.aten.tanh %[[ARG]] : !torch.vtensor<[2,?],unk> -> !torch.vtensor<[2,?],unk>

    torch.prim.Loop.condition %true, iter()
    // CHECK: torch.prim.Loop.condition %[[TRUE]], iter()
  } : (!torch.int, !torch.bool) -> ()

  return %arg : !torch.vtensor<[2,?],unk>
  // CHECK: return %[[ARG]] : !torch.vtensor<[2,?],unk>
}
