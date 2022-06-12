// RUN: torch-mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:  func.func @torch_c.from_i64_to_i64() -> !torch.vtensor<[4],si64> {
// CHECK:            %[[CST:.*]] = arith.constant dense<[5, 1, 4, 2]> : tensor<4xi64>
// CHECK:            %[[T0:.*]] = torch_c.from_builtin_tensor %cst : tensor<4xi64> -> !torch.vtensor<[4],si64>
// CHECK:            return %[[T0:.*]] !torch.vtensor<[4],si64>
func.func @torch_c.from_i64_to_i64() -> !torch.vtensor<[4],si64> {
  %c5_i64 = arith.constant 5 : i64
  %0 = torch_c.from_i64 %c5_i64
  %c1_i64 = arith.constant 1 : i64
  %1 = torch_c.from_i64 %c1_i64
  %c4_i64 = arith.constant 4 : i64
  %2 = torch_c.from_i64 %c4_i64
  %c2_i64 = arith.constant 2 : i64
  %3 = torch_c.from_i64 %c2_i64
  %4 = torch_c.to_i64 %0
  %5 = arith.index_cast %4 : i64 to index
  %6 = arith.index_cast %5 : index to i64
  %7 = torch_c.to_i64 %1
  %8 = arith.index_cast %7 : i64 to index
  %9 = arith.index_cast %8 : index to i64
  %10 = torch_c.to_i64 %2
  %11 = arith.index_cast %10 : i64 to index
  %12 = arith.index_cast %11 : index to i64
  %13 = torch_c.to_i64 %3
  %14 = arith.index_cast %13 : i64 to index
  %15 = arith.index_cast %14 : index to i64
  %16 = tensor.from_elements %6, %9, %12, %15 : tensor<4xi64>
  %17 = torch_c.from_builtin_tensor %16 : tensor<4xi64> -> !torch.vtensor<[4],si64>
  return %17 : !torch.vtensor<[4],si64>
}
