// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.mm$basic(
// CHECK-SAME:                        %[[LHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                        %[[RHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
// CHECK:           %[[LHS:.*]] = torch_c.to_builtin_tensor %[[LHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RHS:.*]] = torch_c.to_builtin_tensor %[[RHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[RHS_DIM_0:.*]] = tensor.dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[RHS_DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[EQ:.*]] = arith.cmpi eq, %[[LHS_DIM_1]], %[[RHS_DIM_0]] : index
// CHECK:           assert %[[EQ]], "mismatching contracting dimension for torch.aten.mm"
// CHECK:           %[[INIT_TENSOR:.*]] = linalg.init_tensor [%[[LHS_DIM_0]], %[[RHS_DIM_1]]] : tensor<?x?xf32>
// CHECK:           %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ZEROFILL:.*]] = linalg.fill ins(%[[CF0]] : f32) outs(%[[INIT_TENSOR]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZEROFILL]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CASTED:.*]] = tensor.cast %[[MATMUL]] : tensor<?x?xf32> to tensor<?x2xf32>
// CHECK:           %[[RESULT_VTENSOR:.*]] = torch_c.from_builtin_tensor %[[CASTED]] : tensor<?x2xf32> -> !torch.vtensor<[?,2],f32>
// CHECK:           return %[[RESULT_VTENSOR]] : !torch.vtensor<[?,2],f32>
func.func @torch.aten.mm$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,2],f32>
  return %0 : !torch.vtensor<[?,2],f32>
}

// -----

// If the operands are missing dtype, we cannot lower it.
func.func @torch.aten.mm$no_convert$missing_dtype(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// Correctly handle the case that operands are statically the wrong rank
// (rank 1 vs rank 2 expected for matmul.)
func.func @torch.aten.mm$no_convert$wrong_rank(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// If the result is missing dtype, we cannot lower it.
func.func @torch.aten.mm$no_convert$result_missing_dtype(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],si64>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],si64> -> tensor<i64>
// CHECK:               %[[EXT:.*]] = tensor.extract %[[I]][] : tensor<i64>
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$zero_rank(%arg0: !torch.vtensor<[],si64>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[],si64> -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$non_zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM0:.*]] = tensor.dim %[[I]], %[[C0]] : tensor<?x?xi64>
// CHECK:               %[[C1:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM1:.*]] = tensor.dim %[[I]], %[[C1]] : tensor<?x?xi64>
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:               %[[DIM0_INDEX:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:               %[[PRED0:.*]] = arith.cmpi eq, %[[DIM0_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED0]], "mismatching contracting dimension"
// CHECK:               %[[DIM1_INDEX:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:               %[[PRED1:.*]] = arith.cmpi eq, %[[DIM1_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED1]], "mismatching contracting dimension"
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:               %[[EXT:.*]] = tensor.extract %[[I]][%[[ZERO]], %[[ZERO]]] : tensor<?x?xi64>
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$non_zero_rank(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[?,?],si64> -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Float.Tensor$zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],f64>) -> !torch.float {
// CHECK:               %[[F:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],f64> -> tensor<f64>
// CHECK:               %[[EXT:.*]] = tensor.extract %[[F]][] : tensor<f64>
// CHECK:               %[[RET:.*]] = torch_c.from_f64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.float
func.func @torch.aten.Float.Tensor$zero_rank(%arg0: !torch.vtensor<[],f64>) -> !torch.float {
  %0 = torch.aten.Float.Tensor %arg0 : !torch.vtensor<[],f64> -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Float.Tensor$non_zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[?,?],f64>) -> !torch.float {
// CHECK:               %[[F:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f64> -> tensor<?x?xf64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM0:.*]] = tensor.dim %[[F]], %[[C0]] : tensor<?x?xf64>
// CHECK:               %[[C1:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM1:.*]] = tensor.dim %[[F]], %[[C1]] : tensor<?x?xf64>
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:               %[[DIM0_INDEX:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:               %[[PRED0:.*]] = arith.cmpi eq, %[[DIM0_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED0]], "mismatching contracting dimension"
// CHECK:               %[[DIM1_INDEX:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:               %[[PRED1:.*]] = arith.cmpi eq, %[[DIM1_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED1]], "mismatching contracting dimension"
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:               %[[EXT:.*]] = tensor.extract %[[F]][%[[ZERO]], %[[ZERO]]] : tensor<?x?xf64>
// CHECK:               %[[RET:.*]] = torch_c.from_f64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.float
func.func @torch.aten.Float.Tensor$non_zero_rank(%arg0: !torch.vtensor<[?,?],f64>) -> !torch.float {
  %0 = torch.aten.Float.Tensor %arg0 : !torch.vtensor<[?,?],f64> -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Bool.Tensor$zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],i1>) -> !torch.bool {
// CHECK:               %[[B:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],i1> -> tensor<i1>
// CHECK:               %[[EXT:.*]] = tensor.extract %[[B]][] : tensor<i1>
// CHECK:               %[[RES:.*]] = torch_c.from_i1 %[[EXT]]
// CHECK:               return %[[RES]] : !torch.bool
func.func @torch.aten.Bool.Tensor$zero_rank(%arg0: !torch.vtensor<[],i1>) -> !torch.bool {
  %0 = torch.aten.Bool.Tensor %arg0 : !torch.vtensor<[],i1> -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Bool.Tensor$non_zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[?,?],i1>) -> !torch.bool {
// CHECK:               %[[B:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM0:.*]] = tensor.dim %[[B]], %[[C0]] : tensor<?x?xi1>
// CHECK:               %[[C1:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM1:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?xi1>
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:               %[[DIM0_INDEX:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:               %[[PRED0:.*]] = arith.cmpi eq, %[[DIM0_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED0]], "mismatching contracting dimension"
// CHECK:               %[[DIM1_INDEX:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:               %[[PRED1:.*]] = arith.cmpi eq, %[[DIM1_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED1]], "mismatching contracting dimension"
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:               %[[EXT:.*]] = tensor.extract %[[I]][%[[ZERO]], %[[ZERO]]] : tensor<?x?xi1>
// CHECK:               %[[RET:.*]] = torch_c.from_i1 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.bool
func.func @torch.aten.Bool.Tensor$non_zero_rank(%arg0: !torch.vtensor<[?,?],i1>) -> !torch.bool {
  %0 = torch.aten.Bool.Tensor %arg0 : !torch.vtensor<[?,?],i1> -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK:    func.func @torch.prim.NumToTensor.Scalar$basic(%[[IN:.*]]: !torch.int) -> !torch.vtensor<[],si64> {
// CHECK:      %[[INI64:.*]] = torch_c.to_i64 %[[IN]]
// CHECK:      %[[NEWVEC:.*]] = linalg.init_tensor [] : tensor<i64>
// CHECK:      %[[FILLVEC:.*]] = linalg.fill ins(%[[INI64]] : i64) outs(%[[NEWVEC]] : tensor<i64>) -> tensor<i64>
// CHECK:      %[[OUTVEC:.*]] = torch_c.from_builtin_tensor %[[FILLVEC]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:      return %[[OUTVEC]] : !torch.vtensor<[],si64>
func.func @torch.prim.NumToTensor.Scalar$basic(%arg0: !torch.int) -> !torch.vtensor<[],si64> {
  %0 = torch.prim.NumToTensor.Scalar %arg0 : !torch.int -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$basic(
// CHECK-SAME:                                              %[[VALUE_T:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[4],f32> {
// CHECK:           %[[T:.*]] = torch_c.to_builtin_tensor %[[VALUE_T]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:           %[[T_CAST:.*]] = tensor.cast %[[T]] : tensor<?xf32> to tensor<4xf32>
// CHECK:           %[[VALUE_T_CAST:.*]] = torch_c.from_builtin_tensor %[[T_CAST]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
// CHECK:           return %[[VALUE_T_CAST]] : !torch.vtensor<[4],f32>
func.func @torch.tensor_static_info_cast$basic(%t: !torch.vtensor<[?],f32>) -> !torch.vtensor<[4],f32> {
  %t_cast = torch.tensor_static_info_cast %t : !torch.vtensor<[?],f32> to !torch.vtensor<[4],f32>
  return %t_cast : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL:     func.func @torch.aten.neg
// CHECK: linalg.generic {{.*}} {
// CHECK-NEXT:    ^bb0(%[[LHS:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[NEG:.*]] = arith.negf %[[LHS]] : f32
// CHECK-NEXT:      linalg.yield %[[NEG]] : f32
// CHECK-NEXT:    } -> tensor<?x?xf32>
func.func @torch.aten.neg(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:     func.func @torch.aten.neg.bf16
// CHECK: linalg.generic {{.*}} {
// CHECK-NEXT:    ^bb0(%[[LHS:.*]]: bf16, %{{.*}}: bf16):
// CHECK-NEXT:      %[[NEG:.*]] = arith.negf %[[LHS]] : bf16
// CHECK-NEXT:      linalg.yield %[[NEG]] : bf16
// CHECK-NEXT:    } -> tensor<?x?xbf16>
func.func @torch.aten.neg.bf16(%arg0: !torch.vtensor<[?,?],bf16>) -> !torch.vtensor<[?,?],bf16> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],bf16> -> !torch.vtensor<[?,?],bf16>
  return %0 : !torch.vtensor<[?,?],bf16>
}
