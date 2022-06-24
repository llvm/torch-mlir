// RUN: torch-mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @torch_c.from_i64() -> !torch.int {
// CHECK:     %[[INT5:.*]] = torch.constant.int 5
// CHECK:     return %[[INT5]] : !torch.int
func.func @torch_c.from_i64() -> !torch.int {
  %c5_i64 = arith.constant 5 : i64
  %0 = torch_c.from_i64 %c5_i64
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch_c.to_i64() -> i64 {
// CHECK:     %[[C5_I64:.*]] = arith.constant 5 : i64
// CHECK:     return %[[C5_I64]] : i64
func.func @torch_c.to_i64() -> i64 {
  %int5 = torch.constant.int 5
  %0 = torch_c.to_i64 %int5
  return %0 : i64
}

// CHECK-LABEL:   func.func @torch_c.from_i64$to_i64() -> i64 {
// CHECK:     %[[C5_I64:.*]] = arith.constant 5 : i64
// CHECK:     return %[[C5_I64]] : i64
func.func @torch_c.from_i64$to_i64() -> i64 {
  %c5_i64 = arith.constant 5 : i64
  %0 = torch_c.from_i64 %c5_i64
  %1 = torch_c.to_i64 %0
  return %1 : i64
}

// CHECK-LABEL:   func.func @torch_c.to_i64$from_i64() -> !torch.int {
// CHECK:     %[[INT5:.*]] = torch.constant.int 5
// CHECK:     return %[[INT5]] : !torch.int
func.func @torch_c.to_i64$from_i64() -> !torch.int {
  %int5 = torch.constant.int 5
  %0 = torch_c.to_i64 %int5
  %1 = torch_c.from_i64 %0
  return %1 : !torch.int
}
