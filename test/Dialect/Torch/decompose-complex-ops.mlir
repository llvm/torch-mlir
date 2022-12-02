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
// CHECK:           %[[CST:.*]]-1 = torch.constant.int -1
// CHECK:           %[[T0:.*]] = torch.prim.ListConstruct %[[CST]]-1 : (!torch.int) -> !torch.list<int>
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
// CHECK-LABEL:   func.func @torch.aten.repeat(
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
