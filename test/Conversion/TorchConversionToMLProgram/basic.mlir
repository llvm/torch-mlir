// RUN: torch-mlir-opt %s -convert-torch-conversion-to-mlprogram -split-input-file | FileCheck %s

// CHECK-LABEL:   ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
// CHECK-LABEL:   func.func @f() -> i64 {
// CHECK:           %[[GLOBAL:.*]] = ml_program.global_load @global_seed : tensor<i64>
// CHECK:           %[[SEED:.*]] = tensor.extract %[[GLOBAL]][] : tensor<i64>
// CHECK:           %[[MULTIPLIER:.*]] = arith.constant 6364136223846793005 : i64
// CHECK:           %[[INC:.*]] = arith.constant 1442695040888963407 : i64
// CHECK:           %[[MUL:.*]] = arith.muli %[[SEED]], %[[MULTIPLIER]] : i64
// CHECK:           %[[NEXT_SEED:.*]] = arith.addi %[[MUL]], %[[INC]] : i64
// CHECK:           %[[INSERTED:.*]] = tensor.insert %[[NEXT_SEED]] into %[[GLOBAL]][] : tensor<i64>
// CHECK:           ml_program.global_store @global_seed = %[[INSERTED]] : tensor<i64>
// CHECK:           return %[[NEXT_SEED]] : i64
module {
  func.func @f() -> i64 {
    %seed = torch_c.get_next_seed : () -> i64
    return %seed : i64
  }
}

// -----

module {
  func.func @no_seed_needed(%arg0: tensor<2x3xf32>) -> !torch.vtensor<[2,3],f32> {
    %0 = torch_c.from_builtin_tensor %arg0 : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
    return %0 : !torch.vtensor<[2,3],f32>
  }
}

// CHECK-NOT: ml_program.global
// CHECK-LABEL: @no_seed_needed
// CHECK-NEXT: torch_c.from_builtin_tensor
