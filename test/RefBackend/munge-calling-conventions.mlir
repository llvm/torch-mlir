// RUN: torch-mlir-opt %s -refback-munge-calling-conventions -split-input-file | FileCheck %s

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG0:.*]]: memref<*xf32>) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[RESULT:.*]] = memref.cast %[[VAL]] : memref<?xf32> to memref<*xf32>
// CHECK:           call @refbackend_consume_float32_func_return(%[[RESULT]]) : (memref<*xf32>) -> ()
// CHECK:           return
func @f(%arg0: memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// -----

// CHECK-LABEL:   func @i(
// CHECK-SAME:            %[[ARG0:.*]]: memref<*xi64>) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL:.*]] = memref.cast %[[ARG0]] : memref<*xi64> to memref<?xi64>
// CHECK:           %[[RESULT:.*]] = memref.cast %[[VAL]] : memref<?xi64> to memref<*xi64>
// CHECK:           call @refbackend_consume_int64_func_return(%[[RESULT]]) : (memref<*xi64>) -> ()
// CHECK:           return
func @i(%arg0: memref<?xi64>) -> memref<?xi64> {
  return %arg0 : memref<?xi64>
}
