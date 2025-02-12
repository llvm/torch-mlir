// RUN: torch-mlir-opt -torch-adjust-calling-conventions -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[NONVAL_TENSOR:.*]] = torch.copy.to_tensor %[[ERASED]] : !torch.tensor
// CHECK:           return %[[NONVAL_TENSOR]] : !torch.tensor
func.func @basic(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func.func @no_type_bound(
// CHECK-SAME:                        %[[ARG:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[ARG]] : !torch.tensor
func.func @no_type_bound(%arg0: !torch.tensor) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func.func @call(
// CHECK-SAME:               %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ARG_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[ARG_NONVAL:.*]] = torch.copy.to_tensor %[[ARG_ERASED]] : !torch.tensor
// CHECK:           %[[INFO_ADDED:.*]] = torch.tensor_static_info_cast %[[ARG_NONVAL]] : !torch.tensor to !torch.tensor<[2,3,?],f32>
// CHECK:           %[[CALL_ARG:.*]] = torch.copy.to_vtensor %[[INFO_ADDED]] : !torch.vtensor<[2,3,?],f32>
// CHECK:           %[[CALL_RES:.*]] = call @call(%[[CALL_ARG]]) : (!torch.vtensor<[2,3,?],f32>) -> !torch.tensor
// CHECK:           return %[[ARG_NONVAL]] : !torch.tensor
func.func @call(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  %0 = call @call(%arg0) : (!torch.tensor) -> !torch.tensor
  return %arg0 : !torch.tensor
}

// COM:   func.func @none_return() {
// COM:           %[[NONE:.*]] = torch.constant.none
// COM:           return
// func.func @none_return() -> !torch.none {
//   %1 = torch.constant.none
//   return %1 : !torch.none
// }

// COM:   func.func @none_call_return() {
// COM:           call @none_return() : () -> ()
// COM:           %[[NONE:.*]] = torch.constant.none
// COM:           "test.use"(%[[NONE]]) : (!torch.none) -> ()
// COM:           return
// func.func @none_call_return() {
//   %0 = call @none_return() : () -> !torch.none
//   "test.use"(%0) : (!torch.none) -> ()
//   return
// }

// COM:   func.func @tuple_return(
// COM:                       %[[ARG0:.*]]: !torch.vtensor<[?],f32>,
// COM:                       %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> (!torch.tensor, !torch.tensor) {
// COM:       %[[ARG0_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[?],f32> to !torch.vtensor
// COM:       %[[ARG0_NONVAL:.*]] = torch.copy.to_tensor %[[ARG0_ERASED]] : !torch.tensor
// COM:       %[[ARG1_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG1]] : !torch.vtensor<[?],f32> to !torch.vtensor
// COM:       %[[ARG1_NONVAL:.*]] = torch.copy.to_tensor %[[ARG1_ERASED]] : !torch.tensor
// COM:           %[[TUPLE:.*]] = torch.prim.TupleConstruct %[[ARG0_NONVAL]], %[[ARG1_NONVAL]] :
// COM:          !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
// COM:           %[[CST0:.*]] = torch.constant.int 0
// COM:           %[[RET0:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST0]] :
// COM:          !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// COM:           %[[CST1:.*]] = torch.constant.int 1
// COM:           %[[RET1:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST1]] :
// COM:          !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// COM:           return %[[RET0]], %[[RET1]] : !torch.tensor, !torch.tensor
// func.func @tuple_return(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>},
//                    %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>}) -> !torch.tuple<tensor, tensor> {
//   %1 = torch.prim.TupleConstruct %arg0, %arg1 : !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
//   return %1 : !torch.tuple<tensor, tensor>
// }

// COM:   func.func @call_tuple_return(
// COM:                            %[[ARG0:.*]]: !torch.vtensor<[?],f32>,
// COM:                            %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> (!torch.tensor, !torch.tensor) {
// COM:       %[[ARG0_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[?],f32> to !torch.vtensor
// COM:       %[[ARG0_NONVAL:.*]] = torch.copy.to_tensor %[[ARG0_ERASED]] : !torch.tensor
// COM:       %[[ARG1_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG1]] : !torch.vtensor<[?],f32> to !torch.vtensor
// COM:       %[[ARG1_NONVAL:.*]] = torch.copy.to_tensor %[[ARG1_ERASED]] : !torch.tensor
// COM:           %[[ARG0_NONVAL_SHAPED:.*]] = torch.tensor_static_info_cast %[[ARG0_NONVAL]] : !torch.tensor to !torch.tensor<[?],f32>
// COM:           %[[ARG0_VAL_SHAPED:.*]] = torch.copy.to_vtensor %[[ARG0_NONVAL_SHAPED]] : !torch.vtensor<[?],f32>
// COM:           %[[ARG1_NONVAL_SHAPED:.*]] = torch.tensor_static_info_cast %[[ARG1_NONVAL]] : !torch.tensor to !torch.tensor<[?],f32>
// COM:           %[[ARG1_VAL_SHAPED:.*]] = torch.copy.to_vtensor %[[ARG1_NONVAL_SHAPED]] : !torch.vtensor<[?],f32>
// COM:           %[[RETS:.*]]:2 = call @tuple_return(%[[ARG0_VAL_SHAPED]], %[[ARG1_VAL_SHAPED]]) :
// COM:          (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) -> (!torch.tensor, !torch.tensor)
// COM:           %[[TUPLE:.*]] = torch.prim.TupleConstruct %[[RETS]]#0, %[[RETS]]#1 :
// COM:          !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
// COM:           %[[CST0:.*]] = torch.constant.int 0
// COM:           %[[RET0:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST0]] :
// COM:          !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// COM:           %[[CST1:.*]] = torch.constant.int 1
// COM:           %[[RET1:.*]] = torch.prim.TupleIndex %[[TUPLE]], %[[CST1]] :
// COM:          !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
// COM:           return %[[RET0]], %[[RET1]] : !torch.tensor, !torch.tensor
// func.func @call_tuple_return(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>},
//                         %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[?],f32>}) -> !torch.tuple<tensor, tensor> {
//   %0 = call @tuple_return(%arg0, %arg1) : (!torch.tensor, !torch.tensor) -> !torch.tuple<tensor, tensor>
//   return %0 : !torch.tuple<tensor, tensor>
// }
