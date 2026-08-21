 // RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Bincount: dynamic input
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @torch.aten.bincount
// CHECK-SAME: ([[INPUT:.*]]: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],si64>

// CHECK: [[INPUT_TENSOR:.*]] = torch_c.to_builtin_tensor [[INPUT]]

// Find the maximum value in the input.
// CHECK: [[INIT_MAX:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK: [[MAX:.*]] = stablehlo.reduce({{.*}}) applies stablehlo.maximum
// CHECK: tensor<i64>

// max(input) + 1, clamped to at least 0.
// CHECK: [[ADD:.*]] = arith.addi
// CHECK: [[MAX_SIZE:.*]] = arith.maxsi
// CHECK: [[SIZE:.*]] = arith.index_cast

// Construct the dynamic output shape.
// CHECK: [[DIM:.*]] = tensor.dim
// CHECK: [[SHAPE:.*]] = tensor.from_elements
// CHECK: stablehlo.dynamic_broadcast_in_dim

// Create [0, 1, ..., max(input)].
// CHECK: [[IOTA_SHAPE:.*]] = tensor.from_elements
// CHECK: [[IOTA:.*]] = stablehlo.dynamic_iota

// Broadcast input and iota to compare every input value against
// every possible bin.
// CHECK: stablehlo.dynamic_broadcast_in_dim
// CHECK: stablehlo.compare EQ
// CHECK: stablehlo.convert

// Count the matches for each bin.
// CHECK: [[COUNTS:.*]] = stablehlo.reduce({{.*}}) applies stablehlo.add
// CHECK: tensor<?xi64>

// CHECK: torch_c.from_builtin_tensor
// CHECK-NOT: torch.aten.bincount
func.func @torch.aten.bincount(%arg0: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],si64> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.bincount %arg0, %none, %int0 : !torch.vtensor<[?],si64>, !torch.none, !torch.int -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// -----

//===----------------------------------------------------------------------===//
// Bincount: static input size
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @torch.aten.bincount_static_size
// CHECK-SAME: ([[INPUT:.*]]: !torch.vtensor<[200],si64>) -> !torch.vtensor<[?],si64>

// CHECK: [[INPUT_TENSOR:.*]] = torch_c.to_builtin_tensor [[INPUT]]

// CHECK: [[INIT_MAX:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK: [[MAX:.*]] = stablehlo.reduce({{.*}}) applies stablehlo.maximum
// CHECK: tensor<i64>

// max(input) + 1, clamped to at least 0.
// CHECK: [[ADD:.*]] = arith.addi
// CHECK: [[MAX_SIZE:.*]] = arith.maxsi
// CHECK: [[SIZE:.*]] = arith.index_cast

// Construct the dynamic output shape.
// CHECK: [[DIM:.*]] = tensor.dim
// CHECK: [[SHAPE:.*]] = tensor.from_elements
// CHECK: stablehlo.dynamic_broadcast_in_dim

// Create [0, 1, ..., max(input)].
// CHECK: [[IOTA_SHAPE:.*]] = tensor.from_elements
// CHECK: [[IOTA:.*]] = stablehlo.dynamic_iota

// Broadcast input and iota to compare every input value against
// every possible bin.
// CHECK: stablehlo.dynamic_broadcast_in_dim
// CHECK: stablehlo.compare EQ
// CHECK: stablehlo.convert

// Count the matches for each bin.
// CHECK: [[COUNTS:.*]] = stablehlo.reduce({{.*}}) applies stablehlo.add
// CHECK: tensor<?xi64>

// CHECK: torch_c.from_builtin_tensor
// CHECK-NOT: torch.aten.bincount
func.func @torch.aten.bincount_static_size(%arg0: !torch.vtensor<[200],si64>) -> !torch.vtensor<[?],si64> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.bincount %arg0, %none, %int0 : !torch.vtensor<[200],si64>, !torch.none, !torch.int -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}

// -----

//===----------------------------------------------------------------------===//
// Bincount: minlength = 600
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @torch.aten.bincount_minlength
// CHECK-SAME: ([[INPUT:.*]]: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],si64>

// CHECK: [[INPUT_TENSOR:.*]] = torch_c.to_builtin_tensor [[INPUT]]

// minlength = 600.
// CHECK: [[MINLENGTH:.*]] = arith.constant 600 : i64

// Find the maximum value in the input.
// CHECK: [[INIT_MAX:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK: [[MAX:.*]] = stablehlo.reduce({{.*}}) applies stablehlo.maximum
// CHECK: tensor<i64>

// max(input) + 1.
// CHECK: [[ADD:.*]] = arith.addi

// Use max(max(input) + 1, minlength).
// CHECK: [[MAX_SIZE:.*]] = arith.maxsi
// CHECK: [[SIZE:.*]] = arith.index_cast

// Construct the dynamic output shape.
// CHECK: [[DIM:.*]] = tensor.dim
// CHECK: [[SHAPE:.*]] = tensor.from_elements
// CHECK: stablehlo.dynamic_broadcast_in_dim

// Create [0, 1, ..., output_size - 1].
// CHECK: [[IOTA_SHAPE:.*]] = tensor.from_elements
// CHECK: [[IOTA:.*]] = stablehlo.dynamic_iota

// Broadcast input and iota to compare every input value against
// every possible bin.
// CHECK: stablehlo.dynamic_broadcast_in_dim
// CHECK: stablehlo.compare EQ
// CHECK: stablehlo.convert

// Count the matches for each bin.
// CHECK: [[COUNTS:.*]] = stablehlo.reduce({{.*}}) applies stablehlo.add
// CHECK: tensor<?xi64>

// CHECK: torch_c.from_builtin_tensor
// CHECK-NOT: torch.aten.bincount
func.func @torch.aten.bincount_minlength(%arg0: !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],si64> {
  %minlength = torch.constant.int 600
  %none = torch.constant.none
  %0 = torch.aten.bincount %arg0, %none, %minlength : !torch.vtensor<[?],si64>, !torch.none, !torch.int -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}
