// RUN: torch-mlir-opt -torch-decompose-complex-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL:  func @matmul_no_decompose
// CHECK:   torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func @matmul_no_decompose(%arg0: !torch.vtensor<[?,?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}


// -----

// CHECK-LABEL:  func @matmul_decompose_2d
// CHECK:    torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
func @matmul_decompose_2d(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:  func @matmul_decompose_3d(
// CHECK:    torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func @matmul_decompose_3d(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// ----
// CHECK-LABEL:   func @torch.aten.softmax.int(
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
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME:          !torch.tensor<[2,3],f32>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.tensor<[?,?],f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<[2,3],f32>, !torch.tensor<[?,?],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<[2,3],f32> to !torch.tensor<[2,3],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[2,3],f32>
func @torch.aten.softmax.int(%t: !torch.tensor<[2,3],f32>, %dim: !torch.int) -> !torch.tensor<[2,3],f32> {
  %dtype = torch.constant.none
  %ret = torch.aten.softmax.int %t, %dim, %dtype: !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}


// ----
// CHECK-LABEL:   func @torch.aten.softmax.int$cst_dim(
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
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME           !torch.tensor<[2,3],f32>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.tensor<[2,1],f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<[2,3],f32>, !torch.tensor<[2,1],f32> -> !torch.tensor<[2,3],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<[2,3],f32> to !torch.tensor<[2,3],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[2,3],f32>
func @torch.aten.softmax.int$cst_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.tensor<[2,3],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}

// ----
// CHECK-LABEL:   func @torch.aten.softmax.int$dyn_shape(
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
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME:          !torch.tensor<[?,?],f32>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.tensor<[?,1],f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<[?,?],f32>, !torch.tensor<[?,1],f32> -> !torch.tensor<[?,?],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<[?,?],f32> to !torch.tensor<[?,?],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[?,?],f32>
func @torch.aten.softmax.int$dyn_shape(%t: !torch.tensor<[?,?],f32>) -> !torch.tensor<[?,?],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[?,?],f32>, !torch.int, !torch.none -> !torch.tensor<[?,?],f32>
  return %ret : !torch.tensor<[?,?],f32>
}

// ----
// CHECK-LABEL:   func @torch.aten.softmax.int$unknown_shape(
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
// CHECK:           %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[KEEP_DIM:.*]] = torch.constant.bool true
// CHECK:           %[[SUM_DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SUM:.*]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIM_LIST]], %[[KEEP_DIM]], %[[SUM_DTYPE]] :
// CHECK-SAME:          !torch.tensor<*,f32>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.tensor<*,f32>
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.div.Tensor %[[EXP]], %[[SUM]] : !torch.tensor<*,f32>, !torch.tensor<*,f32> -> !torch.tensor<*,f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<*,f32> to !torch.tensor<*,f32>
// CHECK:           return %[[RET]] : !torch.tensor<*,f32>
func @torch.aten.softmax.int$unknown_shape(%t: !torch.tensor<*,f32>) -> !torch.tensor<*,f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<*,f32>, !torch.int, !torch.none -> !torch.tensor<*,f32>
  return %ret : !torch.tensor<*,f32>
}

// ----
// CHECK-LABEL:   func @torch.aten.size(
// CHECK-SAME:                         %[[T:.*]]: !torch.vtensor<[?,3],f32>) -> !torch.list<!torch.int> {
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[DIM0:.*]] = torch.aten.size.int %[[T]], %[[CST0]] : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.int
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[DIM1:.*]] = torch.aten.size.int %[[T]], %[[CST1]] : !torch.vtensor<[?,3],f32>, !torch.int -> !torch.int
// CHECK:           %[[SIZE:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           return %[[SIZE]] : !torch.list<!torch.int>
func @torch.aten.size(%arg0: !torch.vtensor<[?,3],f32>) -> !torch.list<!torch.int> {
  %0 = torch.aten.size %arg0 : !torch.vtensor<[?,3],f32> -> !torch.list<!torch.int>
  return %0 : !torch.list<!torch.int>
}

// ----
// CHECK-LABEL:   func @torch.aten.arange() -> !torch.vtensor<[?],si64> {
// CHECK:           %[[CST5:.*]] = torch.constant.int 5
// CHECK:           %[[CSTN:.*]] = torch.constant.none
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT:.*]] = torch.aten.arange.start_step %[[CST0]], %[[CST5]], %[[CST1]], %[[CSTN]], %[[CSTN]], %[[CSTN]], %[[CSTN]] :
// CHECK-SAME:          !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],si64>
func @torch.aten.arange() -> !torch.vtensor<[?],si64> {
  %int5 = torch.constant.int 5
  %none = torch.constant.none
  %0 = torch.aten.arange %int5, %none, %none, %none, %none : !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// ----
// CHECK-LABEL:   func @torch.aten.arange.start() -> !torch.vtensor<[?],si64> {
// CHECK:           %[[CST10:.*]] = torch.constant.int 10
// CHECK:           %[[CST0:.*]] = torch.constant.int 0
// CHECK:           %[[CSTN:.*]] = torch.constant.none
// CHECK:           %[[CST1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT:.*]] = torch.aten.arange.start_step %[[CST0]], %[[CST10]], %[[CST1]], %[[CSTN]], %[[CSTN]], %[[CSTN]], %[[CSTN]] :
// CHECK-SAME:          !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?],si64>
func @torch.aten.arange.start() -> !torch.vtensor<[?],si64> {
  %int10 = torch.constant.int 10
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.arange.start %int0, %int10, %none, %none, %none, %none : !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// ----
// CHECK-LABEL: func @torch.aten.argmax(
// CHECK-SAME:  %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[1,?],si64> {
// CHECK:       %[[CST0:.*]] = torch.constant.int 0
// CHECK:       %[[TRUE:.*]] = torch.constant.bool true
// CHECK:       %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[INP]], %[[CST0]], %[[TRUE]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,?],f32>, !torch.vtensor<[1,?],si64>
// CHECK:       return %[[IND]] : !torch.vtensor<[1,?],si64>
func @torch.aten.argmax(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[1,?],si64> {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %0 = torch.aten.argmax %arg0, %int0, %true : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,?],si64>
  return %0 : !torch.vtensor<[1,?],si64>
}

// ----
// CHECK-LABEL: func @torch.aten.argmax$reduceall(
// CHECK-SAME:   %[[INP:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],si64> {
// CHECK:        %[[NONE:.*]] = torch.constant.none
// CHECK:        %[[FALSE:.*]] = torch.constant.bool false
// CHECK:        %[[CST0:.*]] = torch.constant.int 0
// CHECK:        %[[CST1:.*]] = torch.constant.int 1
// CHECK:        %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[INP]], %[[CST0]], %[[CST1]] : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:        %[[VAL:.*]], %[[IND:.*]] = torch.aten.max.dim %[[FLATTEN]], %[[CST0]], %[[FALSE]] : !torch.vtensor<[?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
// CHECK:        return %[[IND]] : !torch.vtensor<[],si64>
func @torch.aten.argmax$reduceall(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],si64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[?,?],f32>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----
// CHECK-LABEL:   func @torch.aten.square(
// CHECK-SAME:                            %[[INPUT:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[SQUARE:.*]] = torch.aten.mul.Tensor %[[INPUT]], %[[INPUT]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[SQUARE]] : !torch.vtensor<[?,?,?],f32>
func @torch.aten.square(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.square %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.var$unbiased(
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
func @torch.aten.var$unbiased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %true = torch.constant.bool true
  %0 = torch.aten.var %arg0, %true: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.var$biased(
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
func @torch.aten.var$biased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %false = torch.constant.bool false
  %0 = torch.aten.var %arg0, %false: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.std$unbiased(
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
func @torch.aten.std$unbiased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %true = torch.constant.bool true
  %0 = torch.aten.std %arg0, %true: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func @torch.aten.std$biased(
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
func @torch.aten.std$biased(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %false = torch.constant.bool false
  %0 = torch.aten.std %arg0, %false: !torch.vtensor<[?,?,?],f32>, !torch.bool -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
