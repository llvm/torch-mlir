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
// CHECK:           %[[CST:.*]]-9223372036854775808 = torch.constant.int -9223372036854775808
// CHECK:           %[[T0:.*]] = torch.prim.ListConstruct %[[CST]]-9223372036854775808 : (!torch.int) -> !torch.list<int>
// CHECK:           %[[FLATTEN:.*]] = torch.aten.view %[[INP]], %[[T0]] :
// CHECK-SAME:         !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?],f32>
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
// CHECK-SAME:                         %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[CST_TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST0]], %[[CST1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool false
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[UPCAST_INPUT:.*]] = torch.aten.to.dtype %[[INPUT]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[CST_TRUE_0:.*]] = torch.constant.bool true
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[UPCAST_INPUT]], %[[DIMS]], %[[CST_TRUE_0]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST1_0:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL:.*]] = torch.aten.mul.int %[[CST1_0]], %[[DIM0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.mul.int %[[MUL]], %[[DIM1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[1,1],f64>, !torch.int -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST7_1:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[MEAN_CAST:.*]] = torch.aten.to.dtype %[[MEAN]], %[[CST7_1]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_0]] : !torch.vtensor<[1,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[UPCAST_INPUT]], %[[MEAN_CAST]], %[[ALPHA]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[1,1],f64>, !torch.float -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64> -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_MEAN_SQUARE]], %[[DIMS]], %[[KEEPDIM]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[CST1_1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0_0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL_0:.*]] = torch.aten.mul.int %[[CST1_1]], %[[DIM0_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1_0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS_0:.*]] = torch.aten.mul.int %[[MUL_0]], %[[DIM1_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[CST1_2:.*]] = torch.constant.int 1
// CHECK:           %[[NUM_ELEMENTS_0_SUB_1:.*]] = torch.aten.sub.int %[[NUM_ELEMENTS_0]], %[[CST1_2]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[UNBIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_0_SUB_1]] : !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor<[],f64>
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_1:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[DOWNCAST_RESULT:.*]] = torch.aten.to.dtype %[[UNBIASED_VAR]], %[[CST6]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           return %[[DOWNCAST_RESULT]] : !torch.vtensor<[],f32>
func.func @torch.aten.var$unbiased(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
  %true = torch.constant.bool true
  %0 = torch.aten.var %arg0, %true: !torch.vtensor<[?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.var$biased(
// CHECK-SAME:                         %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[CST_FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST0]], %[[CST1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool false
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[UPCAST_INPUT:.*]] = torch.aten.to.dtype %[[INPUT]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[CST_TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[UPCAST_INPUT]], %[[DIMS]], %[[CST_TRUE]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST1_0:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL:.*]] = torch.aten.mul.int %[[CST1_0]], %[[DIM0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.mul.int %[[MUL]], %[[DIM1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[1,1],f64>, !torch.int -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST7_1:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[MEAN_CAST:.*]] = torch.aten.to.dtype %[[MEAN]], %[[CST7_1]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_0]] : !torch.vtensor<[1,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[UPCAST_INPUT]], %[[MEAN_CAST]], %[[ALPHA]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[1,1],f64>, !torch.float -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64> -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_MEAN_SQUARE]], %[[DIMS]], %[[KEEPDIM]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[CST1_1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0_0:.*]] = torch.aten.size.int %[[SUB_MEAN_SQUARE]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL_0:.*]] = torch.aten.mul.int %[[CST1_1]], %[[DIM0_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1_0:.*]] = torch.aten.size.int %[[SUB_MEAN_SQUARE]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS_0:.*]] = torch.aten.mul.int %[[MUL_0]], %[[DIM1_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[BIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_0]] : !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor<[],f64>
// CHECK:           %[[CST7_2:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_1:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[BIASED_VAR_CAST:.*]] = torch.aten.to.dtype %[[BIASED_VAR]], %[[CST7_2]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[DOWNCAST_RESULT:.*]] = torch.aten.to.dtype %[[BIASED_VAR_CAST]], %[[CST6]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_2]] : !torch.vtensor<[],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           return %[[DOWNCAST_RESULT]] : !torch.vtensor<[],f32>
func.func @torch.aten.var$biased(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
  %false = torch.constant.bool false
  %0 = torch.aten.var %arg0, %false: !torch.vtensor<[?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.std$unbiased(
// CHECK-SAME:                         %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[CST_TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST0]], %[[CST1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool false
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[UPCAST_INPUT:.*]] = torch.aten.to.dtype %[[INPUT]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[CST_TRUE_0:.*]] = torch.constant.bool true
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[UPCAST_INPUT]], %[[DIMS]], %[[CST_TRUE_0]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST1_0:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL:.*]] = torch.aten.mul.int %[[CST1_0]], %[[DIM0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.mul.int %[[MUL]], %[[DIM1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[1,1],f64>, !torch.int -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST7_1:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[MEAN_CAST:.*]] = torch.aten.to.dtype %[[MEAN]], %[[CST7_1]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_0]] : !torch.vtensor<[1,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[UPCAST_INPUT]], %[[MEAN_CAST]], %[[ALPHA]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[1,1],f64>, !torch.float -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64> -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_MEAN_SQUARE]], %[[DIMS]], %[[KEEPDIM]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[CST1_1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0_0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL_0:.*]] = torch.aten.mul.int %[[CST1_1]], %[[DIM0_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1_0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS_0:.*]] = torch.aten.mul.int %[[MUL_0]], %[[DIM1_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[CST1_2:.*]] = torch.constant.int 1
// CHECK:           %[[NUM_ELEMENTS_0_SUB_1:.*]] = torch.aten.sub.int %[[NUM_ELEMENTS_0]], %[[CST1_2]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[UNBIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_0_SUB_1]] : !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor<[],f64>
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_1:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[DOWNCAST_VAR:.*]] = torch.aten.to.dtype %[[UNBIASED_VAR]], %[[CST6]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[UNBIASED_STD:.*]] = torch.aten.sqrt %[[DOWNCAST_VAR]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[UNBIASED_STD]] : !torch.vtensor<[],f32>
func.func @torch.aten.std$unbiased(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
  %true = torch.constant.bool true
  %0 = torch.aten.std %arg0, %true: !torch.vtensor<[?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.std$biased(
// CHECK-SAME:                         %[[INPUT:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[CST_FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST0]], %[[CST1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool false
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[UPCAST_INPUT:.*]] = torch.aten.to.dtype %[[INPUT]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[CST_TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[UPCAST_INPUT]], %[[DIMS]], %[[CST_TRUE]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST1_0:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL:.*]] = torch.aten.mul.int %[[CST1_0]], %[[DIM0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.mul.int %[[MUL]], %[[DIM1]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[1,1],f64>, !torch.int -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[CST7_1:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[MEAN_CAST:.*]] = torch.aten.to.dtype %[[MEAN]], %[[CST7_1]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_0]] : !torch.vtensor<[1,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f64>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[UPCAST_INPUT]], %[[MEAN_CAST]], %[[ALPHA]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[1,1],f64>, !torch.float -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64> -> !torch.vtensor<[?,?],f64>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_MEAN_SQUARE]], %[[DIMS]], %[[KEEPDIM]], %[[DTYPE]] : !torch.vtensor<[?,?],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[CST1_1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM0_0:.*]] = torch.aten.size.int %[[SUB_MEAN_SQUARE]], %[[CST0]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[MUL_0:.*]] = torch.aten.mul.int %[[CST1_1]], %[[DIM0_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[DIM1_0:.*]] = torch.aten.size.int %[[SUB_MEAN_SQUARE]], %[[CST1]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS_0:.*]] = torch.aten.mul.int %[[MUL_0]], %[[DIM1_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[BIASED_VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_0]] : !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor<[],f64>
// CHECK:           %[[CST7_2:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_1:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[BIASED_VAR_CAST:.*]] = torch.aten.to.dtype %[[BIASED_VAR]], %[[CST7_2]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[DOWNCAST_VAR:.*]] = torch.aten.to.dtype %[[BIASED_VAR_CAST]], %[[CST6]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_2]] : !torch.vtensor<[],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
// CHECK:           %[[BIASED_STD:.*]] = torch.aten.sqrt %[[DOWNCAST_VAR]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[BIASED_STD]] : !torch.vtensor<[],f32>
func.func @torch.aten.std$biased(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32> {
  %false = torch.constant.bool false
  %0 = torch.aten.std %arg0, %false: !torch.vtensor<[?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
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
// CHECK-LABEL:   func.func @torch.aten.index_put(
// CHECK-SAME:                              %[[INP:.*]]: !torch.vtensor<[?],f32>, %[[INDEX:.*]]: !torch.vtensor<[?],si64>,
// CHECK-SAME:                              %[[VALUES:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                              %[[ACCUM:.*]]: !torch.bool) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[INDICES:.*]] = torch.prim.ListConstruct %[[INDEX]] : (!torch.vtensor<[?],si64>) -> !torch.list<vtensor>
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RES:.*]] = torch.aten._index_put_impl %[[INP]], %[[INDICES]], %[[VALUES]], %[[ACCUM]], %[[FALSE]] : !torch.vtensor<[?],f32>, !torch.list<vtensor>, !torch.vtensor<[?],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[?],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[?],f32>
func.func @torch.aten.index_put(%input: !torch.vtensor<[?],f32>, %index: !torch.vtensor<[?],si64>, %values: !torch.vtensor<[?],f32>, %accumulate : !torch.bool) -> !torch.vtensor<[?],f32> {
  %indices = torch.prim.ListConstruct %index : (!torch.vtensor<[?],si64>) -> !torch.list<vtensor>
  %0 = torch.aten.index_put %input, %indices, %values, %accumulate : !torch.vtensor<[?],f32>, !torch.list<vtensor>, !torch.vtensor<[?],f32>, !torch.bool -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
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
// CHECK-LABEL:   func.func @torch.aten.zero(
// CHECK-SAME:                  %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ZERO:.*]] = torch.constant.int 0
// CHECK:           %[[OUT:.*]] = torch.aten.fill.Scalar %[[INP]], %[[ZERO]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.zero(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.zero %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
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

// -----
// CHECK-LABEL:   func @torch.aten.baddbmm(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[?,?,?],f32>, %[[BATCH1:.*]]: !torch.vtensor<[?,?,?],f32>,
// CHECK-SAME:                  %[[BATCH2:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[BMM:.*]] = torch.aten.bmm %[[BATCH1]], %[[BATCH2]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[MUL:.*]] = torch.aten.mul.Scalar %[[BMM]], %[[CST1]] : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?,?],f32>
// CHECK:           %[[OUT:.*]] = torch.aten.add.Tensor %[[MUL]], %[[SELF]], %[[CST1]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>, !torch.int  -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.baddbmm(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>, %arg2: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.baddbmm %arg0, %arg1, %arg2, %int1, %int1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>, !torch.int , !torch.int -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.floor_divide(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                  %[[OTHER:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[CSTTRUNC:.*]] = torch.constant.str "trunc"
// CHECK:           %[[OUT:.*]] = torch.aten.div.Tensor_mode %[[SELF]], %[[OTHER]], %[[CSTTRUNC]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.str -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.floor_divide(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.floor_divide %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.numpy_T$rank_two(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[5,4],f32>) -> !torch.vtensor<[4,5],f32> {
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST1]], %[[CST0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[OUT:.*]] = torch.aten.permute %[[SELF]], %[[DIMS]] : !torch.vtensor<[5,4],f32>, !torch.list<int> -> !torch.vtensor<[4,5],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[4,5],f32>
func.func @torch.aten.numpy_T$rank_two(%arg0: !torch.vtensor<[5,4],f32>) -> !torch.vtensor<[4,5],f32> {
  %0 = torch.aten.numpy_T %arg0 : !torch.vtensor<[5,4],f32> -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.numpy_T$rank_three(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[5,4,3],f32>) -> !torch.vtensor<[3,4,5],f32> {
// CHECK:           %[[CST2:.*]] = torch.constant.int 2
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST2]], %[[CST1]], %[[CST0]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[OUT:.*]] = torch.aten.permute %[[SELF]], %[[DIMS]] : !torch.vtensor<[5,4,3],f32>, !torch.list<int> -> !torch.vtensor<[3,4,5],f32>
// CHECK:           return %[[OUT]] : !torch.vtensor<[3,4,5],f32>
func.func @torch.aten.numpy_T$rank_three(%arg0: !torch.vtensor<[5,4,3],f32>) -> !torch.vtensor<[3,4,5],f32> {
  %0 = torch.aten.numpy_T %arg0 : !torch.vtensor<[5,4,3],f32> -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.repeat(
// CHECK-SAME:      %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.int, %[[ARG2:.*]]: !torch.int, %[[ARG3:.*]]: !torch.int) -> !torch.vtensor<[?,?,?],f32> {                                                                                    
// CHECK:     %[[T0:.*]] = torch.prim.ListConstruct %[[ARG1]], %[[ARG2]], %[[ARG3]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>                                                                                                                                       
// CHECK:     %[[INT1:.*]] = torch.constant.int 1
// CHECK:     %[[INT0:.*]] = torch.constant.int 0
// CHECK:     %[[T1:.*]] = torch.aten.size.int %[[ARG0]], %[[INT0]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
// CHECK:     %[[T2:.*]] = torch.aten.mul.int %[[T1]], %[[ARG2]] : !torch.int, !torch.int -> !torch.int
// CHECK:     %[[INT1_0:.*]] = torch.constant.int 1
// CHECK:     %[[T3:.*]] = torch.aten.size.int %[[ARG0]], %[[INT1_0]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
// CHECK:     %[[T4:.*]] = torch.aten.mul.int %[[T3]], %[[ARG3]] : !torch.int, !torch.int -> !torch.int
// CHECK:     %[[T5:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]], %[[T1]], %[[INT1]], %[[T3]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>                                                                                                  
// CHECK:     %[[T6:.*]] = torch.prim.ListConstruct %[[ARG1]], %[[ARG2]], %[[T1]], %[[ARG3]], %[[T3]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>                                                                                                  
// CHECK:     %[[T7:.*]] = torch.prim.ListConstruct %[[ARG1]], %[[T2]], %[[T4]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:     %[[T8:.*]] = torch.aten.view %[[ARG0]], %[[T5]] : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[1,1,?,1,?],f32>                                                                                                                                            
// CHECK:     %[[T9:.*]] = torch.aten.broadcast_to %[[T8]], %[[T6]] : !torch.vtensor<[1,1,?,1,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?,?],f32>                                                                                                                                
// CHECK:     %[[T10:.*]] = torch.aten.view %[[T9]], %[[T7]] : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?],f32>                                                                                                                                           
// CHECK:     return %[[T10]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.repeat(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
  %1 = torch.prim.ListConstruct %arg1, %arg2, %arg3: (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.repeat %arg0, %1 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?],f32>
  return %2 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL: func @torch.aten.select_scatter
// CHECK-SAME:  (%[[SELF:.*]]: !torch.vtensor<[?,?],f32>, %[[SRC:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK-NEXT:    %[[INT0:.*]] = torch.constant.int 0
// CHECK-NEXT:    %[[INT1:.*]] = torch.constant.int 1
// CHECK-NEXT:    %[[INT1_0:.*]] = torch.constant.int 1
// CHECK-NEXT:    %[[T0:.*]] = torch.aten.add.int %[[INT0]], %[[INT1_0]] : !torch.int, !torch.int -> !torch.int
// CHECK-NEXT:    %[[T1:.*]] = torch.aten.unsqueeze %[[SRC]], %[[INT1]] : !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?,1],f32>
// CHECK-NEXT:    %[[INT1_1:.*]] = torch.constant.int 1
// CHECK-NEXT:    %[[INT0_2:.*]] = torch.constant.int 0
// CHECK-NEXT:    %[[NONE:.*]] = torch.constant.none
// CHECK-NEXT:    %[[T2:.*]] = torch.aten.size.int %[[SELF]], %[[INT1]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
// CHECK-NEXT:    %[[INT0_3:.*]] = torch.constant.int 0
// CHECK-NEXT:    %[[INT1_4:.*]] = torch.constant.int 1
// CHECK-NEXT:    %[[T3:.*]] = torch.aten.arange.start_step %[[INT0_3]], %[[T2]], %[[INT1_4]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
// CHECK-NEXT:    %[[T4:.*]] = torch.prim.ListConstruct %[[INT1_1]], %[[T2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:    %[[T5:.*]] = torch.aten.view %[[T3]], %[[T4]] : !torch.vtensor<[?],si64>, !torch.list<int> -> !torch.vtensor<[1,?],si64>
// CHECK-NEXT:    %[[T6:.*]] = torch.aten.sub.Scalar %[[T5]], %[[INT0]], %[[INT1_1]] : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
// CHECK-NEXT:    %[[T7:.*]] = torch.aten.remainder.Scalar %[[T6]], %[[INT1_0]] : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
// CHECK-NEXT:    %[[T8:.*]] = torch.aten.eq.Scalar %[[T7]], %[[INT0_2]] : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],i1>
// CHECK-NEXT:    %[[T9:.*]] = torch.aten.ge.Scalar %[[T6]], %[[INT0_2]] : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],i1>
// CHECK-NEXT:    %[[T10:.*]] = torch.aten.lt.Scalar %[[T5]], %[[T0]] : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],i1>
// CHECK-NEXT:    %[[T11:.*]] = torch.aten.bitwise_and.Tensor %[[T8]], %[[T9]] : !torch.vtensor<[1,?],i1>, !torch.vtensor<[1,?],i1> -> !torch.vtensor<[1,?],i1>
// CHECK-NEXT:    %[[T12:.*]] = torch.aten.bitwise_and.Tensor %[[T11]], %[[T10]] : !torch.vtensor<[1,?],i1>, !torch.vtensor<[1,?],i1> -> !torch.vtensor<[1,?],i1>
// CHECK-NEXT:    %[[T13:.*]] = torch.aten.where.self %[[T12]], %[[T1]], %[[SELF]] : !torch.vtensor<[1,?],i1>, !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK-NEXT:    return %[[T13]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.select_scatter(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0 
  %int1 = torch.constant.int 1
  %0 = torch.aten.select_scatter %arg0, %arg1, %int1, %int0 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.var.dim(
// CHECK-SAME:                                  %[[INPUT:.*]]: !torch.vtensor<[3,4,7],f32>) -> !torch.vtensor<[3,4,1],f32> {
// CHECK:           %[[CST2:.*]] = torch.constant.int 2
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[UNBIASED:.*]] = torch.constant.bool false
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool true
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[UPCAST_INPUT:.*]] = torch.aten.to.dtype %[[INPUT]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4,7],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,7],f64>
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[KEEPDIM_0:.*]] = torch.constant.bool true
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[UPCAST_INPUT]], %[[DIMS]], %[[KEEPDIM_0]], %[[NONE_0]] : !torch.vtensor<[3,4,7],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST2]] : !torch.vtensor<[3,4,7],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.mul.int %[[CST1]], %[[DIM2]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[3,4,1],f64>, !torch.int -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST7_1:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[MEAN_CAST:.*]] = torch.aten.to.dtype %[[MEAN]], %[[CST7_1]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_1]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[UPCAST_INPUT]], %[[MEAN_CAST]], %[[ALPHA]] : !torch.vtensor<[3,4,7],f64>, !torch.vtensor<[3,4,1],f64>, !torch.float -> !torch.vtensor<[3,4,7],f64>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[3,4,7],f64>, !torch.vtensor<[3,4,7],f64> -> !torch.vtensor<[3,4,7],f64>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_MEAN_SQUARE]], %[[DIMS]], %[[KEEPDIM]], %[[NONE_0]] : !torch.vtensor<[3,4,7],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST1_0:.*]] = torch.constant.int 1
// CHECK:           %[[DIM2_0:.*]] = torch.aten.size.int %[[SUB_MEAN_SQUARE]], %[[CST2]] : !torch.vtensor<[3,4,7],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS_0:.*]] = torch.aten.mul.int %[[CST1_0]], %[[DIM2_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_0]] : !torch.vtensor<[3,4,1],f64>, !torch.int -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST7_2:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_1:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[VAR_CAST:.*]] = torch.aten.to.dtype %[[VAR]], %[[CST7_2]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_2]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_3:.*]] = torch.constant.none
// CHECK:           %[[DOWNCAST_RESULT:.*]] = torch.aten.to.dtype %[[VAR_CAST]], %[[CST6]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_3]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f32>
// CHECK:           return %[[DOWNCAST_RESULT]] : !torch.vtensor<[3,4,1],f32>
func.func @torch.aten.var.dim(%arg0: !torch.vtensor<[3,4,7],f32>) -> !torch.vtensor<[3,4,1],f32> {
  %int2 = torch.constant.int 2
  %dims = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %unbiased = torch.constant.bool false
  %keepdim = torch.constant.bool true
  %0 = torch.aten.var.dim %arg0, %dims, %unbiased, %keepdim: !torch.vtensor<[3,4,7],f32>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[3,4,1],f32>
  return %0 : !torch.vtensor<[3,4,1],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.softplus(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.tensor<[2,3],f32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !torch.int) -> !torch.tensor<[2,3],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.aten.mul.Scalar %[[VAL_0]], %[[VAL_1]] : !torch.tensor<[2,3],f32>, !torch.int -> !torch.tensor<[2,3],f32>
// CHECK:           %[[VAL_4:.*]] = torch.aten.exp %[[VAL_3]] : !torch.tensor<[2,3],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[VAL_5:.*]] = torch.aten.log1p %[[VAL_4]] : !torch.tensor<[2,3],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[VAL_6:.*]] = torch.aten.div.Scalar %[[VAL_5]], %[[VAL_1]] : !torch.tensor<[2,3],f32>, !torch.int -> !torch.tensor<[2,3],f32>
// CHECK:           %[[VAL_7:.*]] = torch.aten.gt.Scalar %[[VAL_3]], %[[VAL_2]] : !torch.tensor<[2,3],f32>, !torch.int -> !torch.tensor<[2,3],i1>
// CHECK:           %[[VAL_8:.*]] = torch.aten.where.self %[[VAL_7]], %[[VAL_0]], %[[VAL_6]] : !torch.tensor<[2,3],i1>, !torch.tensor<[2,3],f32>, !torch.tensor<[2,3],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           return %[[VAL_8]] : !torch.tensor<[2,3],f32>
// CHECK:         }
func.func @torch.aten.softplus(%t: !torch.tensor<[2,3],f32>, %dim: !torch.int) -> !torch.tensor<[2,3],f32> {
  %int0 = torch.constant.int 0
  %ret = torch.aten.softplus %t, %dim, %int0: !torch.tensor<[2,3],f32>, !torch.int, !torch.int -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.var.correction(
// CHECK-SAME:                                  %[[INPUT:.*]]: !torch.vtensor<[3,4,7],f32>) -> !torch.vtensor<[3,4,1],f32> {
// CHECK:           %[[CST2:.*]] = torch.constant.int 2
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool true
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[UPCAST_INPUT:.*]] = torch.aten.to.dtype %[[INPUT]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4,7],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,7],f64>
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[KEEPDIM_0:.*]] = torch.constant.bool true
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[UPCAST_INPUT]], %[[DIMS]], %[[KEEPDIM_0]], %[[NONE_0]] : !torch.vtensor<[3,4,7],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST2]] : !torch.vtensor<[3,4,7],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.mul.int %[[CST1]], %[[DIM2]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[3,4,1],f64>, !torch.int -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST7_1:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[MEAN_CAST:.*]] = torch.aten.to.dtype %[[MEAN]], %[[CST7_1]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_1]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[UPCAST_INPUT]], %[[MEAN_CAST]], %[[ALPHA]] : !torch.vtensor<[3,4,7],f64>, !torch.vtensor<[3,4,1],f64>, !torch.float -> !torch.vtensor<[3,4,7],f64>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[3,4,7],f64>, !torch.vtensor<[3,4,7],f64> -> !torch.vtensor<[3,4,7],f64>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_MEAN_SQUARE]], %[[DIMS]], %[[KEEPDIM]], %[[NONE_0]] : !torch.vtensor<[3,4,7],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST1_0:.*]] = torch.constant.int 1
// CHECK:           %[[DIM2_0:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST2]] : !torch.vtensor<[3,4,7],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS_0:.*]] = torch.aten.mul.int %[[CST1_0]], %[[DIM2_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[CST2_0:.*]] = torch.constant.int 2
// CHECK:           %[[NUM_ELEMENTS_PLUS_ONE:.*]] = torch.aten.add.int %[[NUM_ELEMENTS_0]], %[[CST1_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[PRED:.*]] = torch.aten.ge.int %[[NUM_ELEMENTS_PLUS_ONE]], %[[CST2_0]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[PRED]], "correction value should be less than or equal to productDimSize + 1"
// CHECK:           %[[NUM_ELEMENTS_MINUS_CORRECTION:.*]] = torch.aten.sub.int %[[NUM_ELEMENTS_0]], %[[CST2_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_MINUS_CORRECTION]] : !torch.vtensor<[3,4,1],f64>, !torch.int -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_1:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[DOWNCAST_RESULT:.*]] = torch.aten.to.dtype %[[VAR]], %[[CST6]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_2]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f32>
// CHECK:           return %[[DOWNCAST_RESULT]] : !torch.vtensor<[3,4,1],f32>
func.func @torch.aten.var.correction(%arg0: !torch.vtensor<[3,4,7],f32>) -> !torch.vtensor<[3,4,1],f32> {
  %int2 = torch.constant.int 2
  %dims = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %keepdim = torch.constant.bool true
  %0 = torch.aten.var.correction %arg0, %dims, %int2, %keepdim: !torch.vtensor<[3,4,7],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[3,4,1],f32>
  return %0 : !torch.vtensor<[3,4,1],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.std.dim(
// CHECK-SAME:                                  %[[INPUT:.*]]: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,1],f32> {
// CHECK:           %[[CST2:.*]] = torch.constant.int 2
// CHECK:           %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST2]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[UNBIASED:.*]] = torch.constant.bool false
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool true
// CHECK:           %[[CST7:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[UPCAST_INPUT:.*]] = torch.aten.to.dtype %[[INPUT]], %[[CST7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,5],f64>
// CHECK:           %[[NONE_0:.*]] = torch.constant.none
// CHECK:           %[[KEEPDIM_0:.*]] = torch.constant.bool true
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[UPCAST_INPUT]], %[[DIMS]], %[[KEEPDIM_0]], %[[NONE_0]] : !torch.vtensor<[3,4,5],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[UPCAST_INPUT]], %[[CST2]] : !torch.vtensor<[3,4,5],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS:.*]] = torch.aten.mul.int %[[CST1]], %[[DIM2]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[MEAN:.*]] = torch.aten.div.Scalar %[[SUM]], %[[NUM_ELEMENTS]] : !torch.vtensor<[3,4,1],f64>, !torch.int -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST7_1:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_1:.*]] = torch.constant.none
// CHECK:           %[[MEAN_CAST:.*]] = torch.aten.to.dtype %[[MEAN]], %[[CST7_1]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_1]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[SUB_MEAN:.*]] = torch.aten.sub.Tensor %[[UPCAST_INPUT]], %[[MEAN_CAST]], %[[ALPHA]] : !torch.vtensor<[3,4,5],f64>, !torch.vtensor<[3,4,1],f64>, !torch.float -> !torch.vtensor<[3,4,5],f64>
// CHECK:           %[[SUB_MEAN_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB_MEAN]], %[[SUB_MEAN]] : !torch.vtensor<[3,4,5],f64>, !torch.vtensor<[3,4,5],f64> -> !torch.vtensor<[3,4,5],f64>
// CHECK:           %[[SUB_MEAN_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_MEAN_SQUARE]], %[[DIMS]], %[[KEEPDIM]], %[[NONE_0]] : !torch.vtensor<[3,4,5],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST1_0:.*]] = torch.constant.int 1
// CHECK:           %[[DIM2_0:.*]] = torch.aten.size.int %[[SUB_MEAN_SQUARE]], %[[CST2]] : !torch.vtensor<[3,4,5],f64>, !torch.int -> !torch.int
// CHECK:           %[[NUM_ELEMENTS_0:.*]] = torch.aten.mul.int %[[CST1_0]], %[[DIM2_0]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[VAR:.*]] = torch.aten.div.Scalar %[[SUB_MEAN_SQUARE_SUM]], %[[NUM_ELEMENTS_0]] : !torch.vtensor<[3,4,1],f64>, !torch.int -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST7_2:.*]] = torch.constant.int 7
// CHECK:           %[[FALSE_1:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_2:.*]] = torch.constant.none
// CHECK:           %[[VAR_CAST:.*]] = torch.aten.to.dtype %[[VAR]], %[[CST7_2]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_2]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f64>
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[FALSE_2:.*]] = torch.constant.bool false
// CHECK:           %[[NONE_3:.*]] = torch.constant.none
// CHECK:           %[[DOWNCAST_RESULT:.*]] = torch.aten.to.dtype %[[VAR_CAST]], %[[CST6]], %[[FALSE_2]], %[[FALSE_2]], %[[NONE_3]] : !torch.vtensor<[3,4,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4,1],f32>
// CHECK:           %[[STD:.*]] = torch.aten.sqrt %[[DOWNCAST_RESULT]] : !torch.vtensor<[3,4,1],f32> -> !torch.vtensor<[3,4,1],f32>
// CHECK:           return %[[STD]] : !torch.vtensor<[3,4,1],f32>
func.func @torch.aten.std.dim(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,1],f32> {
  %int2 = torch.constant.int 2
  %dims = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %unbiased = torch.constant.bool false
  %keepdim = torch.constant.bool true
  %0 = torch.aten.std.dim %arg0, %dims, %unbiased, %keepdim: !torch.vtensor<[3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[3,4,1],f32>
  return %0 : !torch.vtensor<[3,4,1],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.flatten.using_ints(
// CHECK-SAME:                                            %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[INT3:.*]] = torch.constant.int 3
// CHECK:         %[[INT:.*]]-9223372036854775808 = torch.constant.int -9223372036854775808
// CHECK:         %[[T0:.*]] = torch.prim.ListConstruct %[[INT]]-9223372036854775808 : (!torch.int) -> !torch.list<int>
// CHECK:         %[[T1:.*]] = torch.aten.view %[[ARG0]], %[[T0]] : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?],f32>
// CHECK:         return %[[T1]] : !torch.vtensor<[?],f32>
func.func @torch.aten.flatten.using_ints(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %1 = torch.aten.flatten.using_ints %arg0, %int0, %int3: !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
  return %1 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.roll(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.int, %[[ARG2:.*]]: !torch.int) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch.prim.ListConstruct %[[ARG1]], %[[ARG2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[INT:.*]]-2 = torch.constant.int -2
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT]]-2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[INT1_0:.*]] = torch.constant.int 1
// CHECK:         %[[T2:.*]] = torch.aten.neg.int %[[ARG1]] : !torch.int -> !torch.int
// CHECK:         %[[T3:.*]] = torch.aten.slice.Tensor %[[ARG0]], %[[INT1]], %[[T2]], %[[NONE]], %[[INT1]]_0 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[T4:.*]] = torch.aten.slice.Tensor %[[ARG0]], %[[INT1]], %[[INT0]], %[[T2]], %[[INT1]]_0 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[T5:.*]] = torch.prim.ListConstruct %[[T3]], %[[T4]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor<[?,?],f32>>
// CHECK:         %[[T6:.*]] = torch.aten.cat %[[T5]], %[[INT1]] : !torch.list<vtensor<[?,?],f32>>, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[T7:.*]] = torch.aten.neg.int %[[ARG2]] : !torch.int -> !torch.int
// CHECK:         %[[T8:.*]] = torch.aten.slice.Tensor %[[T6]], %[[INT]]-2, %[[T7]], %[[NONE]], %[[INT]]1_0 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[T9:.*]] = torch.aten.slice.Tensor %[[T6]], %[[INT]]-2, %[[INT]]0, %[[T7]], %[[INT]]1_0 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[T10:.*]] = torch.prim.ListConstruct %[[T8]], %[[T9]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor<[?,?],f32>>
// CHECK:         %[[T11:.*]] = torch.aten.cat %[[T10]], %[[INT]]-2 : !torch.list<vtensor<[?,?],f32>>, !torch.int -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T11]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.roll(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.prim.ListConstruct %arg1, %arg2: (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %int-2 = torch.constant.int -2
  %1 = torch.prim.ListConstruct %int1, %int-2: (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.roll %arg0, %0, %1 : !torch.vtensor<[?,?],f32>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
  return %2 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.mse_loss$no_reduction(
// CHECK-SAME:                                   %[[SELF:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                   %[[TARGET:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[REDUCTION:.*]] = torch.constant.int 0
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:         %[[SUB:.*]] = torch.aten.sub.Tensor %[[SELF]], %[[TARGET]], %[[ALPHA]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[SUB_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB]], %[[SUB]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[SUB_SQUARE]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.mse_loss$no_reduction(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.mse_loss %arg0, %arg1, %int0 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.mse_loss$mean_reduction(
// CHECK-SAME:                                   %[[SELF:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                   %[[TARGET:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[REDUCTION:.*]] = torch.constant.int 1
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:         %[[SUB:.*]] = torch.aten.sub.Tensor %[[SELF]], %[[TARGET]], %[[ALPHA]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[SUB_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB]], %[[SUB]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[SUB_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_SQUARE]], %[[NONE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f64>
// CHECK:         %[[NUMEL:.*]] = torch.aten.numel %[[SUB_SQUARE]] : !torch.vtensor<[?,?],f32> -> !torch.int
// CHECK:         %[[SUB_SQUARE_MEAN:.*]] = torch.aten.div.Scalar %[[SUB_SQUARE_SUM]], %[[NUMEL]] : !torch.vtensor<[?,?],f64>, !torch.int -> !torch.vtensor<[?,?],f64>
// CHECK:         %[[CST6:.*]] = torch.constant.int 6
// CHECK:         %[[FALSE_0:.*]] = torch.constant.bool false
// CHECK:         %[[NONE_1:.*]] = torch.constant.none
// CHECK:         %[[SUB_SQUARE_MEAN_CAST:.*]] = torch.aten.to.dtype %[[SUB_SQUARE_MEAN]], %[[CST6]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_1]] : !torch.vtensor<[?,?],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[SUB_SQUARE_MEAN_CAST]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.mse_loss$mean_reduction(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.mse_loss %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.mse_loss$sum_reduction(
// CHECK-SAME:                                   %[[SELF:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                                   %[[TARGET:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[REDUCTION:.*]] = torch.constant.int 2
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:         %[[SUB:.*]] = torch.aten.sub.Tensor %[[SELF]], %[[TARGET]], %[[ALPHA]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[SUB_SQUARE:.*]] = torch.aten.mul.Tensor %[[SUB]], %[[SUB]] : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[SUB_SQUARE_SUM:.*]] = torch.aten.sum.dim_IntList %[[SUB_SQUARE]], %[[NONE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[?,?],f32>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[SUB_SQUARE_SUM]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.mse_loss$sum_reduction(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.mse_loss %arg0, %arg1, %int2 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
