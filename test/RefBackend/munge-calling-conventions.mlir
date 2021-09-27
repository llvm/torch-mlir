// RUN: torch-mlir-opt %s -refback-munge-calling-conventions | FileCheck %s

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG0:.*]]: memref<*xf32>) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[RESULT:.*]] = memref.cast %[[VAL]] : memref<?xf32> to memref<*xf32>
// CHECK:           call @refbackend_consume_func_return(%[[RESULT]]) : (memref<*xf32>) -> ()
// CHECK:           return
func @f(%arg0: memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}
