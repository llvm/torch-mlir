// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-backend-to-tosa-linalg-backend-pipeline)' -split-input-file -verify-diagnostics %s | FileCheck %s

//-----

// CHECK-LABEL:   func.func @tm_scan(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x512xi32>) -> (tensor<1x512xi64>, tensor<1xi64>) {
// CHECK-DAG:     %[[ARG1:.*]] = arith.constant 512 : index
// CHECK-DAG:     %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[RESULT:.*]] = arith.constant 0 : index
// CHECK:         %[[VAL_1:.*]] = tosa.cast %[[ARG0]] : (tensor<1x512xi32>) -> tensor<1x512xi64>
// CHECK:         %[[VAL_2:.*]] = memref.alloc() : memref<1x512xi64>
// CHECK:         %[[VAL_3:.*]] = memref.alloc() : memref<1xi64>
// CHECK:         scf.for %[[VAL_4:.*]] = %[[RESULT]] to %[[ARG1]] step %[[VAL_0]] {
// CHECK:             %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_4]], %[[RESULT]] : index
// CHECK:             scf.if %[[VAL_5]] {
// CHECK:               %[[VAL_6:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[RESULT]], %[[VAL_4]]] : tensor<1x512xi64>
// CHECK:               memref.store %[[VAL_6]], %[[VAL_2]]{{\[}}%[[RESULT]], %[[VAL_4]]] : memref<1x512xi64>
// CHECK:             } else {
// CHECK:               %[[VAL_7:.*]] = arith.subi %[[VAL_4]], %[[VAL_0]] : index
// CHECK:               %[[VAL_8:.*]] = memref.load %[[VAL_2]]{{\[}}%[[RESULT]], %[[VAL_7]]] : memref<1x512xi64>
// CHECK:               %[[VAL_9:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[RESULT]], %[[VAL_4]]] : tensor<1x512xi64>
// CHECK:               %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i64
// CHECK:               memref.store %[[VAL_10]], %[[VAL_2]]{{\[}}%[[RESULT]], %[[VAL_4]]] : memref<1x512xi64>
// CHECK:               memref.store %[[VAL_10]], %[[VAL_3]]{{\[}}%[[RESULT]]] : memref<1xi64>
// CHECK:             }
// CHECK:          }
// CHECK:          %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_3]] : memref<1xi64>
// CHECK:          %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_2]] : memref<1x512xi64>
// CHECK:          return %[[VAL_12]], %[[VAL_11]] : tensor<1x512xi64>, tensor<1xi64>
// CHECK:         }

func.func @tm_scan(%arg0: tensor<1x512xi32>) -> (tensor<1x512xi64>, tensor<1xi64>) {
    %c0_i64 = arith.constant 0 : i64
    %0 = tosa.cast %arg0 : (tensor<1x512xi32>) -> tensor<1x512xi64>
    %1 = tensor.empty() : tensor<1x512xi64>
    %2 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<1x512xi64>) -> tensor<1x512xi64>
    %3 = tensor.empty() : tensor<1xi64>
    %4 = linalg.fill ins(%c0_i64 : i64) outs(%3 : tensor<1xi64>) -> tensor<1xi64>
    %5:2 = tm_tensor.scan dimension(1) inclusive(true) ins(%0 : tensor<1x512xi64>) outs(%2, %4 : tensor<1x512xi64>, tensor<1xi64>) {
    ^bb0(%arg2: i64, %arg3: i64):
      %6 = arith.addi %arg2, %arg3 : i64
      tm_tensor.yield %6 : i64
    } -> tensor<1x512xi64>, tensor<1xi64>
    return %5#0, %5#1 : tensor<1x512xi64>, tensor<1xi64>
}
