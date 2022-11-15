// RUN: torch-mlir-opt %s -refback-munge-calling-conventions -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @f(
// CHECK-SAME:            %[[ARG0:.*]]: memref<*xf32>) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[RESULT:.*]] = memref.cast %[[VAL]] : memref<?xf32> to memref<*xf32>
// CHECK:           call @refbackend_consume_func_return_mrf32(%[[RESULT]]) : (memref<*xf32>) -> ()
// CHECK:           return
func.func @f(%arg0: memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// -----

// CHECK-LABEL:   func.func @i(
// CHECK-SAME:            %[[ARG0:.*]]: memref<*xi64>) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL:.*]] = memref.cast %[[ARG0]] : memref<*xi64> to memref<?xi64>
// CHECK:           %[[RESULT:.*]] = memref.cast %[[VAL]] : memref<?xi64> to memref<*xi64>
// CHECK:           call @refbackend_consume_func_return_mri64(%[[RESULT]]) : (memref<*xi64>) -> ()
// CHECK:           return
func.func @i(%arg0: memref<?xi64>) -> memref<?xi64> {
  return %arg0 : memref<?xi64>
}

// -----

// CHECK-LABEL:   func.func @elemental_type(
// CHECK-SAME:             %[[ARG0:.*]]: memref<*xi64>) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL:.*]] = memref.cast %[[ARG0]] : memref<*xi64> to memref<i64>
// CHECK:           %[[RESULT:.*]] = memref.load %[[VAL]][] : memref<i64>
// CHECK:           call @refbackend_consume_func_return_i64(%[[RESULT]]) : (i64) -> ()
// CHECK:           return
func.func @elemental_type(%arg0: memref<i64>) -> i64 {
  %0 = memref.load %arg0[] : memref<i64>
  return %0 : i64
}

// -----

// CHECK-LABEL:   func.func @multiple_return_values(
// CHECK-SAME:                                 %[[ARG0:.*]]: memref<*xf32>, %[[ARG1:.*]]: memref<*xf32>,
// CHECK-SAME:                                 %[[ARG2:.*]]: memref<*xf32>) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[VAL1:.*]] = memref.cast %[[ARG1]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[VAL2:.*]] = memref.cast %[[ARG2]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[RET0:.*]] = memref.cast %[[VAL0]] : memref<?xf32> to memref<*xf32>
// CHECK:           %[[RET1:.*]] = memref.cast %[[VAL1]] : memref<?xf32> to memref<*xf32>
// CHECK:           %[[RET2:.*]] = memref.cast %[[VAL2]] : memref<?xf32> to memref<*xf32>
// CHECK:           call @refbackend_consume_func_return_mrf32_mrf32_mrf32(%[[RET0]], %[[RET1]], %[[RET2]])
// CHECK-SAME:          : (memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
// CHECK:           return

func.func @multiple_return_values(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>, memref<?xf32>) {
  return %arg0 ,%arg1, %arg2 : memref<?xf32>, memref<?xf32>, memref<?xf32>
}

// -----

// CHECK-LABEL:   func.func @two_return_values(
// CHECK-SAME:                                 %[[ARG0:.*]]: memref<*xf32>, %[[ARG1:.*]]: memref<*xi64>)
// CHECK-SAME:                                 attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[VAL1:.*]] = memref.cast %[[ARG1]] : memref<*xi64> to memref<?xi64>
// CHECK:           %[[RET0:.*]] = memref.cast %[[VAL0]] : memref<?xf32> to memref<*xf32>
// CHECK:           %[[RET1:.*]] = memref.cast %[[VAL1]] : memref<?xi64> to memref<*xi64>
// CHECK:           call @refbackend_consume_func_return_mrf32_mri64(%[[RET0]], %[[RET1]])
// CHECK-SAME:          : (memref<*xf32>, memref<*xi64>) -> ()
// CHECK:           return

func.func @two_return_values(%arg0: memref<?xf32>, %arg1: memref<?xi64>) -> (memref<?xf32>, memref<?xi64>) {
  return %arg0 ,%arg1 : memref<?xf32>, memref<?xi64>
}

// -----

// expected-error-re @+1 {{argument must be a memref of {{.*}} but got 'tensor<?xf32>'}}
func.func @f(%arg0: tensor<?xf32>) {
  return
}
