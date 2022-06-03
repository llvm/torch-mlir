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
// CHECK-LABEL:   func.func @torch.aten.softmax.int(
// CHECK-SAME:                                 %[[T:.*]]: !torch.tensor<[2,3],f32>,
// CHECK-SAME:                                 %[[DIM:.*]]: !torch.int) -> !torch.tensor<[2,3],f32> {
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[KEEP_DIM0:.*]] = torch.constant.bool true
// CHECK:           %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[T]], %[[DIM]], %[[KEEP_DIM0]] :
// CHECK-SAME:                 !torch.tensor<[2,3],f32>, !torch.int, !torch.bool -> !torch.tensor<[?,?],f32>, !torch.tensor<[?,?],si64>
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB:.*]] = torch.aten.sub.Tensor %[[T]], %[[VAL]], %[[FLOAT1]] : !torch.tensor<[2,3],f32>,
// CHECK-SAME:          !torch.tensor<[?,?],f32>, !torch.float -> !torch.tensor<[2,3],f32>
// CHECK:           %[[EXP:.*]] = torch.aten.exp %[[SUB]] : !torch.tensor<[2,3],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME:          !torch.tensor<[2,3],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.tensor<[?,?],f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<[2,3],f32>, !torch.tensor<[?,?],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<[2,3],f32> to !torch.tensor<[2,3],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[2,3],f32>
func.func @torch.aten.softmax.int(%t: !torch.tensor<[2,3],f32>, %dim: !torch.int) -> !torch.tensor<[2,3],f32> {
  %dtype = torch.constant.none
  %ret = torch.aten.softmax.int %t, %dim, %dtype: !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}


// -----
// CHECK-LABEL:   func.func @torch.aten.softmax.int$cst_dim(
// CHECK-SAME:                                         %[[T:.*]]: !torch.tensor<[2,3],f32>) -> !torch.tensor<[2,3],f32> {
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[DIM:.*]] = torch.constant.int 1
// CHECK:           %[[TRU:.*]] = torch.constant.bool true
// CHECK:           %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[T]], %[[DIM]], %[[TRU]] : !torch.tensor<[2,3],f32>, !torch.int, !torch.bool ->
// CHECK-SAME:              !torch.tensor<[2,1],f32>, !torch.tensor<[2,1],si64>
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB:.*]] = torch.aten.sub.Tensor %[[T]], %[[VAL]], %[[FLOAT1]] : !torch.tensor<[2,3],f32>,
// CHECK-SAME:          !torch.tensor<[2,1],f32>, !torch.float -> !torch.tensor<[2,3],f32>
// CHECK:           %[[EXP:.*]] = torch.aten.exp %[[SUB]] : !torch.tensor<[2,3],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME           !torch.tensor<[2,3],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.tensor<[2,1],f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<[2,3],f32>, !torch.tensor<[2,1],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<[2,3],f32> to !torch.tensor<[2,3],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[2,3],f32>
func.func @torch.aten.softmax.int$cst_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.tensor<[2,3],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.softmax.int$dyn_shape(
// CHECK-SAME:                                           %[[T:.*]]: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f32> {
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[DIM:.*]] = torch.constant.int 1
// CHECK:           %[[TRU:.*]] = torch.constant.bool true
// CHECK:           %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[T]], %[[DIM]], %[[TRU]] : !torch.tensor<[?,?],f32>, !torch.int, !torch.bool ->
// CHECK-SAME:          !torch.tensor<[?,1],f32>, !torch.tensor<[?,1],si64>
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB:.*]] = torch.aten.sub.Tensor %[[T]], %[[VAL]], %[[FLOAT1]] : !torch.tensor<[?,?],f32>,
// CHECK-SAME:          !torch.tensor<[?,1],f32>, !torch.float -> !torch.tensor<[?,?],f32>
// CHECK:           %[[EXP:.*]] = torch.aten.exp %[[SUB]] : !torch.tensor<[?,?],f32> -> !torch.tensor<[?,?],f32>
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME:          !torch.tensor<[?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.tensor<[?,1],f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<[?,?],f32>, !torch.tensor<[?,1],f32> -> !torch.tensor<[?,?],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<[?,?],f32> to !torch.tensor<[?,?],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[?,?],f32>
func.func @torch.aten.softmax.int$dyn_shape(%t: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[?,?],f32>, !torch.int, !torch.none -> !torch.tensor<[?,?],f32>
  return %ret : !torch.tensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.softmax.int$unknown_shape(
// CHECK-SAME:                                               %[[T:.*]]: !torch.tensor<*,f32>) -> !torch.tensor<*,f32> {
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[DIM:.*]] = torch.constant.int 1
// CHECK:           %[[TRU:.*]] = torch.constant.bool true
// CHECK:           %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[T]], %[[DIM]], %[[TRU]] : !torch.tensor<*,f32>, !torch.int, !torch.bool
// CHECK-SAME:          -> !torch.tensor<*,f32>, !torch.tensor<*,si64>
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB:.*]] = torch.aten.sub.Tensor %[[T]], %[[VAL]], %[[FLOAT1]] : !torch.tensor<*,f32>, !torch.tensor<*,f32>,
// CHECK-SAME:          !torch.float -> !torch.tensor<*,f32>
// CHECK:           %[[EXP:.*]] = torch.aten.exp %[[SUB]] : !torch.tensor<*,f32> -> !torch.tensor<*,f32>
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME:          !torch.tensor<*,f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.tensor<*,f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<*,f32>, !torch.tensor<*,f32> -> !torch.tensor<*,f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<*,f32> to !torch.tensor<*,f32>
// CHECK:           return %[[RET]] : !torch.tensor<*,f32>
func.func @torch.aten.softmax.int$unknown_shape(%t: !torch.tensor<*,f32>) -> !torch.tensor<*,f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<*,f32>, !torch.int, !torch.none -> !torch.tensor<*,f32>
  return %ret : !torch.tensor<*,f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.size(
// CHECK-SAME:                         %[[T:.*]]: !torch.vtensor<[?,3],f32>) -> !torch.list<int> {
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[T]], %[[CST0]] : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.int
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[T]], %[[CST1]] : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.int
// CHECK:           %[[SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           return %[[SIZE]] : !torch.list<int>
func.func @torch.aten.size(%arg0: !torch.vtensor<[?,3],f32>) -> !torch.list<int> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[?,3],f32> -> !torch.list<int>
  return %0 : !torch.list<int>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.arange() -> !torch.vtensor<[?],si64> {
// CHECK:           %[[CST5:.*]] = torch.constant.int 5
// CHECK:           %[[CSTN:.*]] = torch.constant.none
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT:.*]] = torch.aten.arange.start_step %[[CST0]], %[[CST5]], %[[CST1]], %[[CSTN]], %[[CSTN]], %[[CSTN]], %[[CSTN]] :
// CHECK-SAME:          !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],si64>
func.func @torch.aten.arange() -> !torch.vtensor<[?],si64> {
  %int5 = torch.constant.int 5
  %none = torch.constant.none
  %0 = torch.aten.arange %int5, %none, %none, %none, %none : !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.arange.start() -> !torch.vtensor<[?],si64> {
// CHECK:           %[[CST10:.*]] = torch.constant.int 10
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CSTN:.*]] = torch.constant.none
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT:.*]] = torch.aten.arange.start_step %[[CST0]], %[[CST10]], %[[CST1]], %[[CSTN]], %[[CSTN]], %[[CSTN]], %[[CSTN]] :
// CHECK-SAME:          !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],si64>
func.func @torch.aten.arange.start() -> !torch.vtensor<[?],si64> {
  %int10 = torch.constant.int 10
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.arange.start %int0, %int10, %none, %none, %none, %none : !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.argmax(
// CHECK-SAME:      %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[1,?],si64> {
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[INP]], %[[CST0]], %[[TRUE]] :
// CHECK-SAME:        !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,?],f32>, !torch.vtensor<[1,?],si64>
// CHECK:           return %[[IND]] : !torch.vtensor<[1,?],si64>
func.func @torch.aten.argmax(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[1,?],si64> {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %0 = torch.aten.argmax %arg0, %int0, %true : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,?],si64>
  return %0 : !torch.vtensor<[1,?],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.argmax$reduceall(
// CHECK-SAME:      %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],si64> {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[INP]], %[[CST0]], %[[CST1]] :
// CHECK-SAME:         !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:           %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[FLATTEN]], %[[CST0]], %[[FALSE]] :
// CHECK-SAME:         !torch.vtensor<[?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
// CHECK:           return %[[IND]] : !torch.vtensor<[],si64>
func.func @torch.aten.argmax$reduceall(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],si64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[?,?],f32>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.square(
// CHECK-SAME:                            %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[SQUARE:.*]] = torch.aten.mul.Tensor %[[INPUT]], %[[INPUT]] :
// CHECK-SAME:         !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[SQUARE]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.square(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.square %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.var$unbiased(
// CHECK-SAME:                                  %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[UNBIASED:.*]] = torch.constant.bool true
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum %[[INPUT]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.numel %[[INPUT]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[INPUT]], %[[MEAN]], %[[ALPHA]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum %[[SUB_MEAN_SQUARE]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_NUM_ELEMENTS:.*]] = torch.aten.numel %[[SUB_MEAN_SQUARE]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[NUM_ELEMENTS_SUB1:.*]] = torch.aten.sub.int %[[SUB_MEAN_SQUARE_NUM_ELEMENTS]], %[[CST1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[UNBIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_SUB1]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           return %[[UNBIASED_VAR]] : !torch.vtensor<[],f32>
func.func @torch.aten.var$unbiased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %true = torch.constant.bool true
  %0 = torch.aten.var %arg0, %true: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.var$biased(
// CHECK-SAME:                         %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[UNBIASED:.*]] = torch.constant.bool false
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum %[[INPUT]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.numel %[[INPUT]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[INPUT]], %[[MEAN]], %[[ALPHA]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum %[[SUB_MEAN_SQUARE]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_NUM_ELEMENTS:.*]] = torch.aten.numel %[[SUB_MEAN_SQUARE]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[BIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[SUB_MEAN_SQUARE_NUM_ELEMENTS]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           return %[[BIASED_VAR]] : !torch.vtensor<[],f32>
func.func @torch.aten.var$biased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %false = torch.constant.bool false
  %0 = torch.aten.var %arg0, %false: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.std$unbiased(
// CHECK-SAME:                                  %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[UNBIASED:.*]] = torch.constant.bool true
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum %[[INPUT]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.numel %[[INPUT]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[INPUT]], %[[MEAN]], %[[ALPHA]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum %[[SUB_MEAN_SQUARE]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_NUM_ELEMENTS:.*]] = torch.aten.numel %[[SUB_MEAN_SQUARE]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[NUM_ELEMENTS_SUB1:.*]] = torch.aten.sub.int %[[SUB_MEAN_SQUARE_NUM_ELEMENTS]], %[[CST1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[UNBIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_SUB1]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[UNBIASED_STD:.*]] = torch.aten.sqrt %[[UNBIASED_VAR]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[UNBIASED_STD]] : !torch.vtensor<[],f32>
func.func @torch.aten.std$unbiased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %true = torch.constant.bool true
  %0 = torch.aten.std %arg0, %true: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.std$biased(
// CHECK-SAME:                         %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[UNBIASED:.*]] = torch.constant.bool false
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum %[[INPUT]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.numel %[[INPUT]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[INPUT]], %[[MEAN]], %[[ALPHA]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum %[[SUB_MEAN_SQUARE]], %[[DTYPE]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[SUB_MEAN_SQUARE_NUM_ELEMENTS:.*]] = torch.aten.numel %[[SUB_MEAN_SQUARE]] : !torch.vtensor<[?,?,?],f32> -> !torch.int
// CHECK:           %[[BIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[SUB_MEAN_SQUARE_NUM_ELEMENTS]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[BIASED_STD:.*]] = torch.aten.sqrt %[[BIASED_VAR]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[BIASED_STD]] : !torch.vtensor<[],f32>
func.func @torch.aten.std$biased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %false = torch.constant.bool false
  %0 = torch.aten.std %arg0, %false: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten._unsafe_view$static
// CHECK-SAME:      (%[[ARG0:.*]]: !torch.vtensor<[1,512,32],f32>)
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct
// CHECK-NOT:       torch.aten._unsafe_view
// CHECK-NEXT:      %[[RES:.*]] = torch.aten.view %[[ARG0]], %[[LIST]]
// CHECK-NEXT:      return
func.func @torch.aten._unsafe_view$static(%arg0: !torch.vtensor<[1,512,32],f32>) -> !torch.vtensor<[1,2,256,32],f32> {
  %c1 = torch.constant.int 1
  %c2 = torch.constant.int 2
  %c256 = torch.constant.int 256
  %c32 = torch.constant.int 32
  %0 = torch.prim.ListConstruct %c1, %c2, %c256, %c32 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten._unsafe_view %arg0, %0 : !torch.vtensor<[1,512,32],f32>, !torch.list<int> -> !torch.vtensor<[1,2,256,32],f32>
  return %1 : !torch.vtensor<[1,2,256,32],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten._reshape_alias$static
// CHECK-SAME:      (%[[ARG0:.*]]: !torch.vtensor<[1],f32>)
// CHECK:           %[[LIST1:.*]] = torch.prim.ListConstruct
// CHECK:           %[[LIST2:.*]] = torch.prim.ListConstruct
// CHECK-NOT:       torch.aten._reshape_alias
// CHECK-NEXT:      %[[RES:.*]] = torch.aten.view %[[ARG0]], %[[LIST1]]
// CHECK-NEXT:      return
func.func @torch.aten._reshape_alias$static(%arg0: !torch.vtensor<[1],f32>) -> !torch.vtensor<[12,32],f32> {
  %int1 = torch.constant.int 1
  %int32 = torch.constant.int 32
  %int12 = torch.constant.int 12
  %0 = torch.prim.ListConstruct %int12, %int32 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int32, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten._reshape_alias %arg0, %0, %1 :  !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int> ->  !torch.vtensor<[12,32],f32>
  return %2 : !torch.vtensor<[12,32],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten._unsafe_view$dynamic
// CHECK-SAME:      (%[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>)
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct
// CHECK-NOT:       torch.aten._unsafe_view
// CHECK-NEXT:      %[[RES:.*]] = torch.aten.view %[[ARG0]], %[[LIST]]
// CHECK-NEXT:      return
func.func @torch.aten._unsafe_view$dynamic(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[512,32],f32> {
  %c256 = torch.constant.int 512
  %c32 = torch.constant.int 32
  %0 = torch.prim.ListConstruct %c256, %c32 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten._unsafe_view %arg0, %0 : !torch.vtensor<[?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[512,32],f32>
  return %1 : !torch.vtensor<[512,32],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten._log_softmax(
// CHECK-SAME:               %[[INP:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[INP]], %[[INT0]], %[[TRUE]] :
// CHECK-SAME:         !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,?,?],f32>, !torch.vtensor<[1,?,?],si64>
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB:.*]] = torch.aten.sub.Tensor %[[INP]], %[[VAL]], %[[FLOAT1]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1,?,?],f32>, !torch.float -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[EXP:.*]] = torch.aten.exp %[[SUB]] : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[PRIM:.*]] = torch.prim.ListConstruct %[[INT0]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[TRU:.*]] = torch.constant.bool true
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[SUM_DIM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[PRIM]], %[[TRU]], %[[NONE]] :
// CHECK-SAME:      !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,?],f32>
// CHECK:           %[[LOG:.*]] = torch.aten.log %[[SUM_DIM]] : !torch.vtensor<[1,?,?],f32> -> !torch.vtensor<[1,?,?],f32>
// CHECK:           %[[FLOAT_1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB1:.*]] = torch.aten.sub.Tensor %[[SUB]], %[[LOG]], %[[FLOAT_1]] : !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:      !torch.vtensor<[1,?,?],f32>, !torch.float -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[SUB1]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten._log_softmax(%arg0: !torch.vtensor<[?,?,?],f32> loc(unknown)) -> !torch.vtensor<[?,?,?],f32> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten._log_softmax %arg0, %int0, %false : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.bernoulli
// CHECK-SAME:               (%[[INP:.*]]: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[CON2FLOAT:.*]] = torch.aten.to.dtype %[[INP]], %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] :
// CHECK-SAME:          !torch.vtensor<[?,?,?],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[FLOAT0:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT0]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT1]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT2]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[TENSOR_SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]], %[[DIM2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[TENSOR_SIZE]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[UNF:.*]] = torch.valsem.aten.uniform %[[EMPTY]], %[[FLOAT0]], %[[FLOAT1]], %[[NONE_2]] : !torch.vtensor<[?,?,?],f64>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[?,?,?],f64>

// CHECK:           %[[CMP:.*]] = torch.aten.lt.Tensor %[[UNF]], %[[INP]] : !torch.vtensor<[?,?,?],f64>, !torch.vtensor<[?,?,?],f64> -> !torch.vtensor<[?,?,?],i1>
// CHECK:           %[[INT7_2:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_3:.*]] = torch.constant.none
// CHECK:           %[[TODTYPE:.*]] = torch.aten.to.dtype %[[CMP]], %[[INT7_2]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_3]] :
// CHECK-SAME:        !torch.vtensor<[?,?,?],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[TODTYPE]] : !torch.vtensor<[?,?,?],f64> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @torch.aten.bernoulli(%arg0: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
    %none = torch.constant.none
    %0 = torch.aten.bernoulli %arg0, %none : !torch.vtensor<[?,?,?],f64>, !torch.none -> !torch.vtensor<[?,?,?],f64>
    %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<[?,?,?],f64> to !torch.vtensor
    return %1 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @torch.valsem.aten.bernoulli.float
// CHECK-SAME:               (%[[INP:.*]]: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[PROB:.*]] = torch.constant.float 4.000000e-01
// CHECK:           %[[PROB_TENSOR:.*]] = torch.prim.NumToTensor.Scalar %[[PROB]] : !torch.float -> !torch.vtensor<[],f64>
// CHECK:           %[[INT7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[CON2FLOAT:.*]] = torch.aten.to.dtype %[[INP]], %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] :
// CHECK-SAME:          !torch.vtensor<[?,?,?],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[FLOAT0:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT0]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT1]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT2]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[TENSOR_SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]], %[[DIM2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[TENSOR_SIZE]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[UNF:.*]] = torch.valsem.aten.uniform %[[EMPTY]], %[[FLOAT0]], %[[FLOAT1]], %[[NONE_2]] : !torch.vtensor<[?,?,?],f64>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[CMP:.*]] = torch.aten.lt.Tensor %[[UNF]], %[[PROB_TENSOR]] : !torch.vtensor<[?,?,?],f64>, !torch.vtensor<[],f64> -> !torch.vtensor<[?,?,?],i1>
// CHECK:           %[[INT7_2:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_3:.*]] = torch.constant.none
// CHECK:           %[[TODTYPE:.*]] = torch.aten.to.dtype %[[CMP]], %[[INT7_2]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_3]] :
// CHECK-SAME:        !torch.vtensor<[?,?,?],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[TODTYPE]] : !torch.vtensor<[?,?,?],f64> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @torch.valsem.aten.bernoulli.float(%arg0: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
    %none = torch.constant.none
    %prob = torch.constant.float 4.000000e-01
    %0 = torch.valsem.aten.bernoulli.float %arg0, %prob, %none : !torch.vtensor<[?,?,?],f64>, !torch.float, !torch.none -> !torch.vtensor<[?,?,?],f64>
    %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<[?,?,?],f64> to !torch.vtensor
    return %1 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @torch.valsem.aten.bernoulli.Tensor(
// CHECK-SAME:                       %[[INP:.*]]: !torch.vtensor<[?,?,?],f64>,
// CHECK-SAME:                       %[[PROB:.*]]: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[CON2FLOAT:.*]] = torch.aten.to.dtype %[[INP]], %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] :
// CHECK-SAME:          !torch.vtensor<[?,?,?],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[FLOAT0:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT0]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT1]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT2]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[TENSOR_SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]], %[[DIM2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[TENSOR_SIZE]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[UNF:.*]] = torch.valsem.aten.uniform %[[EMPTY]], %[[FLOAT0]], %[[FLOAT1]], %[[NONE_2]] : !torch.vtensor<[?,?,?],f64>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[CMP:.*]] = torch.aten.lt.Tensor %[[UNF]], %[[PROB]] : !torch.vtensor<[?,?,?],f64>, !torch.vtensor<[?,?,?],f64> -> !torch.vtensor<[?,?,?],i1>
// CHECK:           %[[INT7_2:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_3:.*]] = torch.constant.none
// CHECK:           %[[TODTYPE:.*]] = torch.aten.to.dtype %[[CMP]], %[[INT7_2]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_3]] :
// CHECK-SAME:        !torch.vtensor<[?,?,?],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[TODTYPE]] : !torch.vtensor<[?,?,?],f64> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @torch.valsem.aten.bernoulli.Tensor(%arg0: !torch.vtensor<[?,?,?],f64>, %arg1: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
    %none = torch.constant.none
    %0 = torch.valsem.aten.bernoulli.Tensor %arg0, %arg1, %none : !torch.vtensor<[?,?,?],f64>, !torch.vtensor<[?,?,?],f64>, !torch.none -> !torch.vtensor<[?,?,?],f64>
    %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<[?,?,?],f64> to !torch.vtensor
    return %1 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.rand_like(
// CHECK-SAME:                               %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
// CHECK:           %[[INT6:.*]] = torch.constant.int 6
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[FLOAT0:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[INPUT]], %[[INT0]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[INPUT]], %[[INT1]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[INPUT]], %[[INT2]] : !torch.vtensor<[?,?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[TENSOR_SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]], %[[DIM2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[TENSOR_SIZE]], %[[INT6]], %[[NONE_0]], %[[NONE_0]], %[[NONE_0]], %[[NONE_0]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[UNIFORM:.*]] = torch.valsem.aten.uniform %[[EMPTY]], %[[FLOAT0]], %[[FLOAT1]], %[[NONE_1]] : !torch.vtensor<[?,?,?],f32>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[UNIFORM]] : !torch.vtensor<[?,?,?],f32> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @torch.aten.rand_like(%arg0: !torch.vtensor<[?,?,?],f64>) -> !torch.vtensor {
    %int6 = torch.constant.int 6
    %none = torch.constant.none
    %0 = torch.aten.rand_like %arg0, %int6, %none, %none, %none, %none : !torch.vtensor<[?,?,?],f64>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],f32>
    %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<[?,?,?],f32> to !torch.vtensor
    return %1 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.select.int(
// CHECK-SAME:                          %[[T:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?],si64> {
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[END:.*]] = torch.aten.add.int %[[CST0]], %[[CST1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[SLICE:.*]] = torch.aten.slice.Tensor %[[T]], %[[CST0]], %[[CST0]], %[[END]], %[[CST1]] :
// CHECK-SAME:        !torch.vtensor<[?,?],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
// CHECK:           %[[SELECT:.*]] = torch.aten.squeeze.dim %[[SLICE]], %[[CST0]] :
// CHECK-SAME:        !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[?],si64>
// CHECK:           return %[[SELECT]] : !torch.vtensor<[?],si64>
func.func @torch.aten.select.int(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?],si64> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.select.int %arg0, %int0, %int0 : !torch.vtensor<[?,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.hardsigmoid(
// CHECK-SAME:               %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[CST2:.*]] = torch.constant.int 3
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[ADD:.*]] = torch.aten.add.Scalar %[[INPUT]], %[[CST2]], %[[CST1]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[DIV:.*]] = torch.aten.div.Scalar %[[ADD]], %[[CST6]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[SIZES]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:        !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[CST1_TENSOR:.*]] = torch.valsem.aten.fill.Scalar %[[EMPTY]], %[[CST1]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[MIN:.*]] = torch.aten.minimum %[[CST1_TENSOR]], %[[DIV]] : !torch.vtensor<[],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[EMPTY_1:.*]] = torch.aten.empty.memory_format %[[SIZES]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] :
// CHECK-SAME:        !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[CST0_TENSOR:.*]] = torch.valsem.aten.fill.Scalar %[[EMPTY_1]], %[[CST0]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[RET:.*]] = torch.aten.maximum %[[CST0_TENSOR]], %[[MIN]] : !torch.vtensor<[],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RET]] : !torch.vtensor<[?,?],f32>
// CHECK:         }
func.func @torch.aten.hardsigmoid(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.hardsigmoid %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.hardswish(
// CHECK-SAME:                    %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           %[[INT6:.*]] = torch.constant.int 6
// CHECK:           %[[ADD:.*]] = torch.aten.add.Scalar %[[INP]], %[[INT3]], %[[INT1]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[RELU:.*]] = torch.aten.relu %[[ADD]] : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[INT6_:.*]] = torch.constant.int 6
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[MEM:.*]] = torch.aten.empty.memory_format %[[LIST]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:        !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[FILL:.*]] = torch.valsem.aten.fill.Scalar %[[MEM]], %[[INT6_]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
// CHECK:           %[[MIN:.*]] = torch.aten.minimum %[[RELU]], %[[FILL]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[DIV:.*]] = torch.aten.div.Scalar %[[MIN]], %[[INT6]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[MUL:.*]] = torch.aten.mul.Tensor %[[DIV]], %[[INP]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[MUL]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.hardswish(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.hardswish %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.hardtanh(
// CHECK-SAME:                              %[[INPUT:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                              %[[MIN_VAL:.*]]: !torch.float,
// CHECK-SAME:                              %[[MAX_VAL:.*]]: !torch.float) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[SIZES]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:        !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[MIN_TENSOR:.*]] = torch.valsem.aten.fill.Scalar %[[EMPTY]], %[[MIN_VAL]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
// CHECK:           %[[MIN:.*]] = torch.aten.maximum %[[INPUT]], %[[MIN_TENSOR]] : !torch.vtensor<[?],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[?],f32>
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[VAL_10:.*]] = torch.aten.empty.memory_format %[[SIZES]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:        !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[MAX_TENSOR:.*]] = torch.valsem.aten.fill.Scalar %[[VAL_10]], %[[MAX_VAL]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
// CHECK:           %[[RET:.*]] = torch.aten.minimum %[[MAX_TENSOR]], %[[MIN]] : !torch.vtensor<[],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?],f32>
// CHECK:           return %[[RET]] : !torch.vtensor<[?],f32>
func.func @torch.aten.hardtanh(%arg0: !torch.vtensor<[?],f32>, %min: !torch.float, %max: !torch.float) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.hardtanh %arg0, %min, %max : !torch.vtensor<[?],f32>, !torch.float, !torch.float -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.new_zeros
// CHECK-SAME:                    %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:             %[[NONE:.*]] = torch.constant.none
// CHECK:             %[[INT2:.*]] = torch.constant.int 2
// CHECK:             %[[INT3:.*]] = torch.constant.int 3
// CHECK:             %[[SIZE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[INT6:.*]] = torch.constant.int 6
// CHECK:             %[[RES:.*]] = torch.aten.zeros %[[SIZE]], %[[INT6]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
// CHECK:             return %[[RES]] : !torch.vtensor<[2,3],f32>
// CHECK:           }
func.func @torch.aten.new_zeros(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[2,3],f32> {
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.new_zeros %arg0, %0, %none, %none, %none, %none : !torch.vtensor<[?,?],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
  return %1 : !torch.vtensor<[2,3],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.new_ones
// CHECK-SAME:                    %[[INP:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[3,4],si64> {
// CHECK:             %[[NONE:.*]] = torch.constant.none
// CHECK:             %[[INT3:.*]] = torch.constant.int 3
// CHECK:             %[[INT4:.*]] = torch.constant.int 4
// CHECK:             %[[SIZE:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[INT4_0:.*]] = torch.constant.int 4
// CHECK:             %[[RES:.*]] = torch.aten.ones %[[SIZE]], %[[INT4_0]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4],si64>
// CHECK:             return %[[RES]] : !torch.vtensor<[3,4],si64>
// CHECK:           }
func.func @torch.aten.new_ones(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[3,4],si64> {
  %none = torch.constant.none
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.new_ones %arg0, %0, %none, %none, %none, %none : !torch.vtensor<[?,?],si64>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4],si64>
  return %1 : !torch.vtensor<[3,4],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.silu(
// CHECK-SAME:                  %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
// CHECK:           %[[SIGMOID:.*]] = torch.aten.sigmoid %[[INP]] : !torch.vtensor<[?,?],f32> -> !torch.vtensor
// CHECK:           %[[MUL:.*]] = torch.aten.mul.Tensor %[[SIGMOID]], %[[INP]] : !torch.vtensor, !torch.vtensor<[?,?],f32> -> !torch.vtensor
// CHECK:           return %[[MUL]] : !torch.vtensor
func.func @torch.aten.silu(%arg0: !torch.vtensor<[?,?],f32> loc(unknown)) -> !torch.vtensor {
    %0 = torch.aten.silu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor
    return %0 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.full
// CHECK-SAME:                  () -> !torch.vtensor<[2,3],f32> {
// CHECK:           %[[FLOAT5:.*]] = torch.constant.float 5.000000e+00
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[SIZE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[MEM_FORMAT:.*]] = torch.constant.none
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[SIZE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[MEM_FORMAT]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
// CHECK:           %[[RES:.*]] = torch.valsem.aten.fill.Scalar %[[EMPTY]], %[[FLOAT5]] : !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[2,3],f32>
func.func @torch.aten.full() -> !torch.vtensor<[2,3],f32> {
  %float5.000000e00 = torch.constant.float 5.000000e+00
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %float5.000000e00, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
  return %1 : !torch.vtensor<[2,3],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.full_like(
// CHECK-SAME:                  %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[INT5:.*]] = torch.constant.int 5
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[INP]], %[[INT0]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[INP]], %[[INT1]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[SIZE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[RES:.*]] = torch.valsem.aten.fill.Scalar %[[EMPTY]], %[[INT5]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.full_like(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int5 = torch.constant.int 5
  %none = torch.constant.none
  %0 = torch.aten.full_like %arg0, %int5, %none, %none, %none, %none, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.index_put(
// CHECK-SAME:                              %[[INP:.*]]: !torch.vtensor<[?],f32>, %[[INDEX:.*]]: !torch.vtensor<[?],si64>,
// CHECK-SAME:                              %[[VALUES:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                              %[[ACCUM:.*]]: !torch.bool) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[INDICES:.*]] = torch.prim.ListConstruct %[[INDEX]] : (!torch.vtensor<[?],si64>) -> !torch.list<vtensor>
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RES:.*]] = torch.valsem.aten.index_put_impl %[[INP]], %[[INDICES]], %[[VALUES]], %[[ACCUM]], %[[FALSE]] : !torch.vtensor<[?],f32>, !torch.list<vtensor>, !torch.vtensor<[?],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[?],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[?],f32>
func.func @torch.aten.index_put(%input: !torch.vtensor<[?],f32>, %index: !torch.vtensor<[?],si64>, %values: !torch.vtensor<[?],f32>, %accumulate : !torch.bool) -> !torch.vtensor<[?],f32> {
  %indices = torch.prim.ListConstruct %index : (!torch.vtensor<[?],si64>) -> !torch.list<vtensor>
  %0 = torch.aten.index_put %input, %indices, %values, %accumulate : !torch.vtensor<[?],f32>, !torch.list<vtensor>, !torch.vtensor<[?],f32>, !torch.bool -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.expand_as(
// CHECK-SAME:                  %[[INP:.*]]: !torch.vtensor<[?,1,1],f32>, %[[OTHER:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[OTHER]], %[[INT0]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[OTHER]], %[[INT1]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[OTHER]], %[[INT2]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]], %[[DIM2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[RES:.*]] = torch.aten.broadcast_to %[[INP]], %[[SIZE]] : !torch.vtensor<[?,1,1],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.expand_as(%arg0: !torch.vtensor<[?,1,1],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.expand_as %arg0, %arg1 : !torch.vtensor<[?,1,1],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten._to_copy(
// CHECK-SAME:                              %[[INP:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[INP]], %[[INT0]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[INP]], %[[INT1]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[INP]], %[[INT2]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]], %[[DIM2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[SIZE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[RES:.*]] = torch.valsem.aten.copy %[[EMPTY]], %[[INP]], %[[FALSE]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten._to_copy(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten._to_copy %arg0, %none, %none, %none, %none, %false, %none : !torch.vtensor<[?,?,?],f32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.dropout$eval(
// CHECK-SAME:                  %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[PROB:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[TRAIN:.*]] = torch.constant.bool false
// CHECK:           return %[[INP:.*]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.dropout$eval(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %false = torch.constant.bool false
  %0 = torch.aten.dropout %arg0, %float1.000000e-01, %false : !torch.vtensor<[?,?],f32>, !torch.float, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.dropout$train(
// CHECK-SAME:                  %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[PROB:.*]] = torch.constant.float 3.000000e-01
// CHECK:           %[[TRAIN:.*]] = torch.constant.bool true
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[CST1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[ONEMINUSP:.*]] = torch.aten.sub.float %[[CST1]], %[[PROB]] : !torch.float, !torch.float -> !torch.float
// CHECK:           %[[PROB_TENSOR:.*]] = torch.prim.NumToTensor.Scalar %[[ONEMINUSP]] : !torch.float -> !torch.vtensor<[],f64>
// CHECK:           %[[INT7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[CON2FLOAT:.*]] = torch.aten.to.dtype %[[INP]], %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] :
// CHECK-SAME:          !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[FLOAT0:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[CON2FLOAT]], %[[INT1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[TENSOR_SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[EMPTY:.*]] = torch.aten.empty.memory_format %[[TENSOR_SIZE]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[UNF:.*]] = torch.valsem.aten.uniform %[[EMPTY]], %[[FLOAT0]], %[[FLOAT1]], %[[NONE_2]] : !torch.vtensor<[?,?],f64>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[CMP:.*]] = torch.aten.lt.Tensor %[[UNF]], %[[PROB_TENSOR]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[],f64> -> !torch.vtensor<[?,?],i1>
// CHECK:           %[[INT6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_3:.*]] = torch.constant.none
// CHECK:           %[[BOOL_MASK:.*]] = torch.aten.to.dtype %[[CMP]], %[[INT6]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_3]] :
// CHECK-SAME:        !torch.vtensor<[?,?],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[MASK_INP:.*]] = torch.aten.mul.Tensor %[[BOOL_MASK]], %[[INP]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[OUT:.*]] = torch.aten.div.Scalar %[[MASK_INP]], %[[ONEMINUSP]] : !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.dropout$train(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %float3.000000e-01 = torch.constant.float 3.000000e-01
  %true = torch.constant.bool true
  %0 = torch.aten.dropout %arg0, %float3.000000e-01, %true : !torch.vtensor<[?,?],f32>, !torch.float, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.zero.functional(
// CHECK-SAME:                  %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ZERO:.*]] = torch.constant.int 0
// CHECK:           %[[OUT:.*]] = torch.valsem.aten.fill.Scalar %[[INP]], %[[ZERO]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.zero.functional(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.zero.functional %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.new_empty
// CHECK-SAME:                    %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:             %[[NONE:.*]] = torch.constant.none
// CHECK:             %[[INT2:.*]] = torch.constant.int 2
// CHECK:             %[[INT3:.*]] = torch.constant.int 3
// CHECK:             %[[SIZE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[NONE_0:.*]] = torch.constant.none
// CHECK:             %[[INT6:.*]] = torch.constant.int 6
// CHECK:             %[[RES:.*]] = torch.aten.empty.memory_format %[[SIZE]], %[[INT6]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE_0]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
// CHECK:             return %[[RES]] : !torch.vtensor<[2,3],f32>
func.func @torch.aten.new_empty(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[2,3],f32> {
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.new_empty %arg0, %0, %none, %none, %none, %none : !torch.vtensor<[?,?],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
  return %1 : !torch.vtensor<[2,3],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.where.Scalar(
// CHECK-SAME:                  %[[COND:.*]]: !torch.vtensor<[?,?,?],i1>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[CST8:.*]] = torch.constant.float 8.000000e+00
// CHECK:           %[[CST4:.*]] = torch.constant.float 4.000000e+00
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[ALLOC:.*]] = torch.aten.empty.memory_format %[[LIST]], %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[FILL_SELF:.*]] = torch.valsem.aten.fill.Scalar %[[ALLOC]], %[[CST4]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
// CHECK:           %[[LIST2:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE2:.*]] = torch.constant.none
// CHECK:           %[[ALLOC2:.*]] = torch.aten.empty.memory_format %[[LIST2]], %none_0, %none_0, %none_0, %none_0, %none_0 : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[FILL_OTHER:.*]] = torch.valsem.aten.fill.Scalar %[[ALLOC2]], %[[CST8]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
// CHECK:           %[[OUT:.*]] = torch.aten.where.self %[[COND]], %[[FILL_SELF]], %[[FILL_OTHER]] : !torch.vtensor<[?,?,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.where.Scalar(%arg0: !torch.vtensor<[?,?,?],i1>) -> !torch.vtensor<[?,?,?],f32> {
  %cst8 = torch.constant.float 8.000000e+00
  %cst4 = torch.constant.float 4.000000e+00
  %0 = torch.aten.where.Scalar %arg0, %cst4, %cst8 : !torch.vtensor<[?,?,?],i1>, !torch.float, !torch.float -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.where.ScalarSelf(
// CHECK-SAME:                  %[[COND:.*]]: !torch.vtensor<[?,?,?],i1>, %[[OTHER:.*]]: !torch.vtensor<[?,?],f64>) -> !torch.vtensor<[?,?,?],f64> {
// CHECK:           %[[CST:.*]] = torch.constant.float 4.000000e+00
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[ALLOC:.*]] = torch.aten.empty.memory_format %[[LIST]], %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[FILL:.*]] = torch.valsem.aten.fill.Scalar %[[ALLOC]], %[[CST]] : !torch.vtensor<[],f64>, !torch.float -> !torch.vtensor<[],f64>
// CHECK:           %[[OUT:.*]] = torch.aten.where.self %[[COND]], %[[FILL]], %[[OTHER]] : !torch.vtensor<[?,?,?],i1>, !torch.vtensor<[],f64>, !torch.vtensor<[?,?],f64> -> !torch.vtensor<[?,?,?],f64>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?],f64>
func.func @torch.aten.where.ScalarSelf(%arg0: !torch.vtensor<[?,?,?],i1>, %arg1: !torch.vtensor<[?,?],f64>) -> !torch.vtensor<[?,?,?],f64> {
  %cst = torch.constant.float 4.000000e+00
  %0 = torch.aten.where.ScalarSelf %arg0, %cst, %arg1 : !torch.vtensor<[?,?,?],i1>, !torch.float, !torch.vtensor<[?,?],f64> -> !torch.vtensor<[?,?,?],f64>
  return %0 : !torch.vtensor<[?,?,?],f64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.where.ScalarOther(
// CHECK-SAME:                  %[[COND:.*]]: !torch.vtensor<[?,?,?],i1>, %[[SELF:.*]]: !torch.vtensor<[?,?],f64>) -> !torch.vtensor<[?,?,?],f64> {
// CHECK:           %[[CST:.*]] = torch.constant.float 4.000000e+00
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[ALLOC:.*]] = torch.aten.empty.memory_format %[[LIST]], %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[FILL:.*]] = torch.valsem.aten.fill.Scalar %[[ALLOC]], %[[CST]] : !torch.vtensor<[],f64>, !torch.float -> !torch.vtensor<[],f64>
// CHECK:           %[[OUT:.*]] = torch.aten.where.self %[[COND]], %[[SELF]], %[[FILL]] : !torch.vtensor<[?,?,?],i1>, !torch.vtensor<[?,?],f64>, !torch.vtensor<[],f64> -> !torch.vtensor<[?,?,?],f64>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?],f64>
func.func @torch.aten.where.ScalarOther(%arg0: !torch.vtensor<[?,?,?],i1>, %arg1: !torch.vtensor<[?,?],f64>) -> !torch.vtensor<[?,?,?],f64> {
  %cst = torch.constant.float 4.000000e+00
  %0 = torch.aten.where.ScalarOther %arg0, %arg1, %cst : !torch.vtensor<[?,?,?],i1>, !torch.vtensor<[?,?],f64>, !torch.float -> !torch.vtensor<[?,?,?],f64>
  return %0 : !torch.vtensor<[?,?,?],f64>
}

// -----
// CHECK-LABEL: func.func @torch.aten.pad
// CHECK-SAME:  (%[[SELF:.*]]: !torch.vtensor<[?,?,?],f64>, %[[VALUE:.*]]: !torch.float) -> !torch.vtensor<[?,?,?],f64> {
// CHECK-NOT:       torch.aten.pad 
// CHECK:           %[[STRING:.*]] = torch.constant.str "constant"
// CHECK-NEXT:      %[[LIST:.*]] = torch.prim.ListConstruct
// CHECK-NEXT:      %[[PAD_ND:.*]] = torch.aten.constant_pad_nd %[[SELF]], %[[LIST]], %[[VALUE]]
// CHECK-NEXT:      return %[[PAD_ND]]
func.func @torch.aten.pad(%arg0: !torch.vtensor<[?,?,?],f64>, %arg1: !torch.float) -> !torch.vtensor<[?,?,?],f64> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1 
  %int2 = torch.constant.int 2 
  %int3 = torch.constant.int 3
  %str = torch.constant.str "constant"
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.pad %arg0, %0, %str, %arg1 : !torch.vtensor<[?,?,?],f64>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[?,?,?],f64>
  return %1 : !torch.vtensor<[?,?,?],f64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.to.dtype_layout(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f64> {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[OUT:.*]] = torch.aten.to.dtype %[[SELF]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f64>
func.func @torch.aten.to.dtype_layout(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int7 = torch.constant.int 7
  %0 = torch.aten.to.dtype_layout %arg0, %int7, %int0, %none, %none, %false, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// -----
// CHECK-LABEL:   func @torch.aten.adaptive_avg_pool2d(
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
func.func @torch.aten.adaptive_avg_pool2d(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int7 = torch.constant.int 7
  %output_size = torch.prim.ListConstruct %int7, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.adaptive_avg_pool2d %arg0, %output_size : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.clamp_min(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[MIN:.*]] = torch.constant.int -2
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[OUT:.*]] = torch.aten.clamp %[[SELF]], %[[MIN]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.none -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.clamp_min(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %min = torch.constant.int -2
  %0 = torch.aten.clamp_min %arg0, %min : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.clamp_max(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[MAX:.*]] = torch.constant.int 7
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[OUT:.*]] = torch.aten.clamp %[[SELF]], %[[NONE]], %[[MAX]]  : !torch.vtensor<[?,?],f32>, !torch.none, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.clamp_max(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %max = torch.constant.int 7
  %0 = torch.aten.clamp_max %arg0, %max : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
