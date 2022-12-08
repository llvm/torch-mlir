// RUN: torch-mlir-opt -torch-decompose-complex-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @matmul_no_decompose
// CHECK:           torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_no_decompose(%arg0: !torch.vtensor<[?,?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}


// -----

// CHECK-LABEL:   func.func @matmul_decompose_2d
// CHECK:           torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
func.func @matmul_decompose_2d(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @matmul_decompose_3d(
// CHECK:           torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_decompose_3d(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:   func @torch.aten.adaptive_avg_pool2d$non_unit_output_size(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[OUTPUT_SIZE:.*]] = torch.prim.ListConstruct %[[CST7]], %[[CST7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[CST2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[SELF]], %[[CST2]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[CST3:.*]] = torch.constant.int 3
// CHECK:           %[[DIM3:.*]] = torch.aten.size.int %[[SELF]], %[[CST3]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[COND1:.*]] = torch.aten.eq.int %[[DIM2]], %[[CST7]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[COND1]], "unimplemented: only support cases where input and output size are equal for non-unit output size"
// CHECK:           %[[T1:.*]] = torch.aten.sub.int %[[CST7]], %[[CST1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[T2:.*]] = torch.aten.sub.int %[[DIM2]], %[[T1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[COND2:.*]] = torch.aten.eq.int %[[DIM3]], %[[CST7]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[COND2]], "unimplemented: only support cases where input and output size are equal for non-unit output size"
// CHECK:           %[[T3:.*]] = torch.aten.sub.int %[[CST7]], %[[CST1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[T4:.*]] = torch.aten.sub.int %[[DIM3]], %[[T3]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[T5:.*]] = torch.prim.ListConstruct %[[T2]], %[[T4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T6:.*]] = torch.prim.ListConstruct %[[CST1]], %[[CST1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T7:.*]]  = torch.prim.ListConstruct %[[CST0]], %[[CST0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[OUT:.*]] = torch.aten.avg_pool2d %[[SELF]], %[[T5]], %[[T6]], %[[T7]], %[[FALSE]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.adaptive_avg_pool2d$non_unit_output_size(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int7 = torch.constant.int 7
  %output_size = torch.prim.ListConstruct %int7, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.adaptive_avg_pool2d %arg0, %output_size : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.adaptive_avg_pool2d$unit_output_size(
// CHECK-SAME:                                                               %[[SELF:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[CST_1:.*]] = torch.constant.int 1
// CHECK:           %[[OUTPUT_SIZE:.*]] = torch.prim.ListConstruct %[[CST_1]], %[[CST_1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[CST_2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM_2:.*]] = torch.aten.size.int %[[SELF]], %[[CST_2]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[CST_3:.*]] = torch.constant.int 3
// CHECK:           %[[DIM_3:.*]] = torch.aten.size.int %[[SELF]], %[[CST_3]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[CST_1_1:.*]] = torch.constant.int 1
// CHECK:           %[[CST_0:.*]] = torch.constant.int 0
// CHECK:           %[[CEIL_MODE:.*]] = torch.constant.bool false
// CHECK:           %[[COUNT_INCLUDE_PAD:.*]] = torch.constant.bool true
// CHECK:           %[[DIVISOR_OVERRIDE:.*]] = torch.constant.none
// CHECK:           %[[KERNEL_SIZE:.*]] = torch.prim.ListConstruct %[[DIM_2]], %[[DIM_3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[STRIDE:.*]] = torch.prim.ListConstruct %[[CST_1_1]], %[[CST_1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[PADDING:.*]] = torch.prim.ListConstruct %[[CST_0]], %[[CST_0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[OUT:.*]] = torch.aten.avg_pool2d %[[SELF]], %[[KERNEL_SIZE]], %[[STRIDE]], %[[PADDING]], %[[CEIL_MODE]], %[[COUNT_INCLUDE_PAD]], %[[DIVISOR_OVERRIDE]] : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.adaptive_avg_pool2d$unit_output_size(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %output_size = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.adaptive_avg_pool2d %arg0, %output_size : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}
