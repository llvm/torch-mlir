// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// RUN: torch-mlir-opt <%s -convert-torch-to-linalg=allow-non-finites=false -split-input-file -verify-diagnostics | FileCheck %s -check-prefix=UNSUPPORTED-NON-FINITES


// CHECK-LABEL:   func.func @torch.aten.mm$basic(
// CHECK-SAME:                        %[[LHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                        %[[RHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
// CHECK-DAG:       %[[LHS:.*]] = torch_c.to_builtin_tensor %[[LHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[RHS:.*]] = torch_c.to_builtin_tensor %[[RHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[RHS_DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[RHS_DIM_0:.*]] = tensor.dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[EQ:.*]] = arith.cmpi eq, %[[LHS_DIM_1]], %[[RHS_DIM_0]] : index
// CHECK:           assert %[[EQ]], "mismatching contracting dimension for torch.aten.mm"
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty(%[[LHS_DIM_0]], %[[RHS_DIM_1]]) : tensor<?x?xf32>
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

// CHECK-LABEL: func.func @torch.aten.matmul.2d
func.func @torch.aten.matmul.2d(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
  // CHECK-DAG:  %[[LHS:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
  // CHECK-DAG:  %[[RHS:.+]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
  // CHECK-DAG:  %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:  %[[EMPTY:.+]] = tensor.empty() : tensor<8x8xf32>
  // CHECK-DAG:  %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK:  %[[MATMUL:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<8x16xf32>, tensor<16x8xf32>) outs(%[[FILL]] : tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul.4d
// CHECK-DAG:  %[[LHS:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,2,32,400],f32> -> tensor<1x2x32x400xf32>
// CHECK-DAG:  %[[RHS:.+]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,2,400,32],f32> -> tensor<1x2x400x32xf32>
// CHECK-DAG:  %[[LHS_CAST:.*]] = tensor.cast %[[LHS]] : tensor<1x2x32x400xf32> to tensor<1x2x32x400xf32>
// CHECK-DAG:  %[[RHS_CAST:.*]] = tensor.cast %[[RHS]] : tensor<1x2x400x32xf32> to tensor<1x2x400x32xf32>
// CHECK-DAG:  %[[COLLAPSED_LHS:.+]] = tensor.collapse_shape %[[LHS_CAST]] {{\[\[}}0, 1], [2], [3]] : tensor<1x2x32x400xf32> into tensor<2x32x400xf32>
// CHECK-DAG:  %[[COLLAPSED_RHS:.+]] = tensor.collapse_shape %[[RHS_CAST]] {{\[\[}}0, 1], [2], [3]] : tensor<1x2x400x32xf32> into tensor<2x400x32xf32>
// CHECK:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[COLLAPSED_RHS]], %[[COLLAPSED_LHS]] : tensor<2x400x32xf32>, tensor<2x32x400xf32>) outs(%{{.*}} : tensor<2x400x400xf32>) -> tensor<2x400x400xf32>
// CHECK:  %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[\[}}0, 1], [2], [3]] output_shape [1, 2, 400, 400] : tensor<2x400x400xf32> into tensor<1x2x400x400xf32>
func.func @torch.aten.matmul.4d(%arg0: !torch.vtensor<[1,2,32,400],f32>, %arg1: !torch.vtensor<[1,2,400,32],f32>) -> !torch.vtensor<[1,2,400,400],f32> {
  %0 = torch.aten.matmul %arg1, %arg0 : !torch.vtensor<[1,2,400,32],f32>, !torch.vtensor<[1,2,32,400],f32> -> !torch.vtensor<[1,2,400,400],f32>
  return %0 : !torch.vtensor<[1,2,400,400],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.mm$basic_strict(
// CHECK-NOT: assert
func.func @torch.aten.mm$basic_strict(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32>
  attributes {torch.assume_strict_symbolic_shapes}
{
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,2],f32>
  return %0 : !torch.vtensor<[?,2],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.mm$basic_unsigned(
// CHECK: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
func.func @torch.aten.mm$basic_unsigned(%arg0: !torch.vtensor<[?,?],ui32>, %arg1: !torch.vtensor<[?,?],ui32>) -> !torch.vtensor<[?,2],ui32>
  attributes {torch.assume_strict_symbolic_shapes}
{
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],ui32>, !torch.vtensor<[?,?],ui32> -> !torch.vtensor<[?,2],ui32>
  return %0 : !torch.vtensor<[?,2],ui32>
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

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$zero_rank$byte_dtype
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],ui8>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],ui8> -> tensor<i8>
// CHECK:               %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[EXTRACT:.*]] = tensor.extract %[[I]][] : tensor<i8>
// CHECK:               %[[RES:.*]] = arith.extui %[[EXTRACT]] : i8 to i64
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$zero_rank$byte_dtype(%arg0: !torch.vtensor<[],ui8>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[],ui8> -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$zero_rank$char_dtype
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],si8>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],si8> -> tensor<i8>
// CHECK:               %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[EXTRACT:.*]] = tensor.extract %[[I]][] : tensor<i8>
// CHECK:               %[[RES:.*]] = arith.extsi %[[EXTRACT]] : i8 to i64
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$zero_rank$char_dtype(%arg0: !torch.vtensor<[],si8>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[],si8> -> !torch.int
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
// CHECK:      %[[NEWVEC:.*]] = tensor.empty() : tensor<i64>
// CHECK:      %[[FILLVEC:.*]] = linalg.fill ins(%[[INI64]] : i64) outs(%[[NEWVEC]] : tensor<i64>) -> tensor<i64>
// CHECK:      %[[OUTVEC:.*]] = torch_c.from_builtin_tensor %[[FILLVEC]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:      return %[[OUTVEC]] : !torch.vtensor<[],si64>
func.func @torch.prim.NumToTensor.Scalar$basic(%arg0: !torch.int) -> !torch.vtensor<[],si64> {
  %0 = torch.prim.NumToTensor.Scalar %arg0 : !torch.int -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$bf16$f_to_f
// CHECK:       %[[F64:.*]] = torch_c.to_f64 %{{.*}}
// CHECK:       %[[TRUNC:.*]] = arith.truncf %[[F64]] : f64 to bf16
// CHECK:       %[[EMPTY:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xbf16>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[TRUNC]] : bf16) outs(%[[EMPTY]] : tensor<?x?xbf16>) -> tensor<?x?xbf16>
func.func @torch.aten.full$bf16$f_to_f(%dim0: !torch.int, %dim1: !torch.int, %fill: !torch.float) -> !torch.vtensor<[?,?],bf16> {
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %dim0, %dim1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %fill, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],bf16>
  return %1 : !torch.vtensor<[?,?],bf16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$bf16$si_to_f
// CHECK:       %[[SITOFP:.*]] = arith.sitofp %{{.*}} : i64 to bf16
// CHECK:       %[[EMPTY:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xbf16>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[SITOFP]] : bf16) outs(%[[EMPTY]] : tensor<?x?xbf16>) -> tensor<?x?xbf16>
func.func @torch.aten.full$bf16$si_to_f(%dim0: !torch.int, %dim1: !torch.int, %fill: !torch.int) -> !torch.vtensor<[?,?],bf16> {
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %dim0, %dim1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %fill, %none, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],bf16>
  return %1 : !torch.vtensor<[?,?],bf16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$si32$f_to_si
// CHECK:       %[[F64:.*]] = torch_c.to_f64 %{{.*}}
// CHECK:       %[[FPTOSI:.*]] = arith.fptosi %[[F64]] : f64 to i32
// CHECK:       %[[EMPTY:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xi32>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[FPTOSI]] : i32) outs(%[[EMPTY]] : tensor<?x?xi32>) -> tensor<?x?xi32>
func.func @torch.aten.full$si32$f_to_si(%dim0: !torch.int, %dim1: !torch.int, %fill: !torch.float) -> !torch.vtensor<[?,?],si32> {
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %dim0, %dim1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %fill, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],si32>
  return %1 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$f32$truncf
// CHECK:       %[[F64:.*]] = torch_c.to_f64 %{{.*}}
// CHECK:       %[[TRUNC:.*]] = arith.truncf %[[F64]] : f64 to f32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<2048x7168xf32>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[TRUNC]] : f32) outs(%[[EMPTY]] : tensor<2048x7168xf32>) -> tensor<2048x7168xf32>
func.func @torch.aten.full$f32$truncf() -> !torch.vtensor<[2048,7168],f32> {
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %none = torch.constant.none
  %int2048 = torch.constant.int 2048
  %int7168 = torch.constant.int 7168
  %0 = torch.prim.ListConstruct %int2048, %int7168 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %float0.000000e00, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2048,7168],f32>
  return %1 : !torch.vtensor<[2048,7168],f32>
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

// -----

// CHECK-LABEL:     func.func @torch.aten.neg.f16
// CHECK: linalg.generic {{.*}} {
// CHECK-NEXT:    ^bb0(%[[LHS:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:      %[[NEG:.*]] = arith.negf %[[LHS]] : f16
// CHECK-NEXT:      linalg.yield %[[NEG]] : f16
// CHECK-NEXT:    } -> tensor<?x?xf16>
func.func @torch.aten.neg.f16(%arg0: !torch.vtensor<[?,?],f16>) -> !torch.vtensor<[?,?],f16> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],f16> -> !torch.vtensor<[?,?],f16>
  return %0 : !torch.vtensor<[?,?],f16>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.cat$convert(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T0:.*]] = torch.prim.ListConstruct %[[ARG0]], %[[ARG1]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si32>) -> !torch.list<vtensor>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:         %[[T3:.*]] = linalg.generic {{.*}} ins(%[[T2]] : tensor<?x?xi32>) outs(%{{.*}}: tensor<?x?xf32>)
// CHECK:         %[[T4:.*]] = tensor.concat dim(0) %[[T1]], %[[T3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = torch_c.from_builtin_tensor %[[T4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.cat$convert(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cat(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_0:.*]] = torch.prim.ListConstruct %[[ARG_0]], %[[ARG_1]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.concat dim(0) %[[VAL_1]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.cat(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.transpose$basic(
// CHECK-SAME:                                     %[[ARG_0:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[IN_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
// CHECK:           %[[TRANSP:.*]] = linalg.transpose ins(%[[IN_0]] : tensor<4x3xf32>) outs(%1 : tensor<3x4xf32>) permutation = [1, 0]
// CHECK:           %[[OUT_0:.*]] = torch_c.from_builtin_tensor %{{.*}}  : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[OUT_0]] : !torch.vtensor<[3,4],f32>
func.func @torch.aten.transpose$basic(%arg0: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[4,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-LABEL:   func.func @test_rotary_embedding(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[1,3,2,6],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[1,2],si64>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[1,3,2,6],f32>
func.func @test_rotary_embedding(%arg0: !torch.vtensor<[1,3,2,6],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[1,3,2,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_3]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,2],si64> -> tensor<1x2xi64>
  // CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,3,2,6],f32> -> tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_8:.*]] = torch.constant.none
  // CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.constant.float 1.000000e+00
  // CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_15:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_16:.*]] = linalg.fill ins(%[[VAL_15]] : f32) outs(%[[VAL_14]] : tensor<1x3x2x6xf32>) -> tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_17:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:           %[[VAL_18:.*]] = arith.constant -1.000000e+00 : f32
  // CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_21:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_22:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_23:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_7]], %[[VAL_6]] : tensor<1x3x2x6xf32>, tensor<1x2xi64>) outs(%[[VAL_16]] : tensor<1x3x2x6xf32>) {
  // CHECK:           ^bb0(%[[VAL_24:.*]]: f32, %[[VAL_25:.*]]: i64, %[[VAL_26:.*]]: f32):
  // CHECK:             %[[VAL_27:.*]] = linalg.index 0 : index
  // CHECK:             %[[VAL_28:.*]] = linalg.index 1 : index
  // CHECK:             %[[VAL_29:.*]] = linalg.index 2 : index
  // CHECK:             %[[VAL_30:.*]] = linalg.index 3 : index
  // CHECK:             %[[VAL_31:.*]] = arith.remsi %[[VAL_30]], %[[VAL_22]] : index
  // CHECK:             %[[VAL_32:.*]] = arith.cmpi sge, %[[VAL_30]], %[[VAL_22]] : index
  // CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_30]], %[[VAL_22]] : index
  // CHECK:             %[[VAL_34:.*]] = arith.remsi %[[VAL_33]], %[[VAL_21]] : index
  // CHECK:             %[[VAL_35:.*]] = arith.index_cast %[[VAL_25]] : i64 to index
  // CHECK:             %[[VAL_36:.*]] = tensor.extract %[[VAL_5]]{{\[}}%[[VAL_35]], %[[VAL_31]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_37:.*]] = tensor.extract %[[VAL_4]]{{\[}}%[[VAL_35]], %[[VAL_31]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_38:.*]] = tensor.extract %[[VAL_7]]{{\[}}%[[VAL_27]], %[[VAL_28]], %[[VAL_29]], %[[VAL_34]]] : tensor<1x3x2x6xf32>
  // CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_32]], %[[VAL_17]], %[[VAL_18]] : f32
  // CHECK:             %[[VAL_40:.*]] = arith.mulf %[[VAL_24]], %[[VAL_36]] : f32
  // CHECK:             %[[VAL_41:.*]] = arith.mulf %[[VAL_38]], %[[VAL_37]] : f32
  // CHECK:             %[[VAL_42:.*]] = arith.mulf %[[VAL_41]], %[[VAL_39]] : f32
  // CHECK:             %[[VAL_43:.*]] = arith.addf %[[VAL_40]], %[[VAL_42]] : f32
  // CHECK:             linalg.yield %[[VAL_43]] : f32
  // CHECK:           } -> tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_44:.*]] = tensor.cast %[[VAL_23]] : tensor<1x3x2x6xf32> to tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_45:.*]] = torch_c.from_builtin_tensor %[[VAL_44]] : tensor<1x3x2x6xf32> -> !torch.vtensor<[1,3,2,6],f32>
  // CHECK:           return %[[VAL_45]] : !torch.vtensor<[1,3,2,6],f32>
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %4 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float1.000000e00 : !torch.vtensor<[1,3,2,6],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,3,2,6],f32>
  return %4 : !torch.vtensor<[1,3,2,6],f32>
}

// -----

// Test rotary embedding with rank 3 input (batch, seq, hidden)
// CHECK-LABEL:   func.func @test_rotary_embedding_rank3(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[1,2,6],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[1,2],si64>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[1,2,6],f32>
func.func @test_rotary_embedding_rank3(%arg0: !torch.vtensor<[1,2,6],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[1,2,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_3]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,2],si64> -> tensor<1x2xi64>
  // CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,2,6],f32> -> tensor<1x2x6xf32>
  // CHECK:           %[[VAL_8:.*]] = torch.constant.none
  // CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.constant.float 1.000000e+00
  // CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : index to i64
  // CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : index to i64
  // CHECK:           %[[VAL_18:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_19:.*]] = arith.index_cast %[[VAL_18]] : index to i64
  // CHECK:           %[[VAL_20:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_21:.*]] = arith.index_cast %[[VAL_20]] : index to i64
  // CHECK:           %[[VAL_22:.*]] = tensor.from_elements %[[VAL_15]], %[[VAL_17]], %[[VAL_19]], %[[VAL_21]] : tensor<4xi64>
  // CHECK:           %[[VAL_23:.*]] = tensor.reshape %[[VAL_7]](%[[VAL_22]]) : (tensor<1x2x6xf32>, tensor<4xi64>) -> tensor<1x1x2x6xf32>
  // CHECK:           %[[VAL_24:.*]] = tensor.empty() : tensor<1x1x2x6xf32>
  // CHECK:           %[[VAL_25:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_26:.*]] = linalg.fill ins(%[[VAL_25]] : f32) outs(%[[VAL_24]] : tensor<1x1x2x6xf32>) -> tensor<1x1x2x6xf32>
  // CHECK:           %[[VAL_27:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:           %[[VAL_28:.*]] = arith.constant -1.000000e+00 : f32
  // CHECK:           %[[VAL_29:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_30:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_31:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_32:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_33:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_23]], %[[VAL_6]] : tensor<1x1x2x6xf32>, tensor<1x2xi64>) outs(%[[VAL_26]] : tensor<1x1x2x6xf32>) {
  // CHECK:           ^bb0(%[[VAL_34:.*]]: f32, %[[VAL_35:.*]]: i64, %[[VAL_36:.*]]: f32):
  // CHECK:             %[[VAL_37:.*]] = linalg.index 0 : index
  // CHECK:             %[[VAL_38:.*]] = linalg.index 1 : index
  // CHECK:             %[[VAL_39:.*]] = linalg.index 2 : index
  // CHECK:             %[[VAL_40:.*]] = linalg.index 3 : index
  // CHECK:             %[[VAL_41:.*]] = arith.remsi %[[VAL_40]], %[[VAL_32]] : index
  // CHECK:             %[[VAL_42:.*]] = arith.cmpi sge, %[[VAL_40]], %[[VAL_32]] : index
  // CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_40]], %[[VAL_32]] : index
  // CHECK:             %[[VAL_44:.*]] = arith.remsi %[[VAL_43]], %[[VAL_31]] : index
  // CHECK:             %[[VAL_45:.*]] = arith.index_cast %[[VAL_35]] : i64 to index
  // CHECK:             %[[VAL_46:.*]] = tensor.extract %[[VAL_5]]{{\[}}%[[VAL_45]], %[[VAL_41]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_47:.*]] = tensor.extract %[[VAL_4]]{{\[}}%[[VAL_45]], %[[VAL_41]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_48:.*]] = tensor.extract %[[VAL_23]]{{\[}}%[[VAL_37]], %[[VAL_38]], %[[VAL_39]], %[[VAL_44]]] : tensor<1x1x2x6xf32>
  // CHECK:             %[[VAL_49:.*]] = arith.select %[[VAL_42]], %[[VAL_27]], %[[VAL_28]] : f32
  // CHECK:             %[[VAL_50:.*]] = arith.mulf %[[VAL_34]], %[[VAL_46]] : f32
  // CHECK:             %[[VAL_51:.*]] = arith.mulf %[[VAL_48]], %[[VAL_47]] : f32
  // CHECK:             %[[VAL_52:.*]] = arith.mulf %[[VAL_51]], %[[VAL_49]] : f32
  // CHECK:             %[[VAL_53:.*]] = arith.addf %[[VAL_50]], %[[VAL_52]] : f32
  // CHECK:             linalg.yield %[[VAL_53]] : f32
  // CHECK:           } -> tensor<1x1x2x6xf32>
  // CHECK:           %[[VAL_54:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_55:.*]] = arith.index_cast %[[VAL_54]] : index to i64
  // CHECK:           %[[VAL_56:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_57:.*]] = arith.index_cast %[[VAL_56]] : index to i64
  // CHECK:           %[[VAL_58:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_59:.*]] = arith.index_cast %[[VAL_58]] : index to i64
  // CHECK:           %[[VAL_60:.*]] = tensor.from_elements %[[VAL_55]], %[[VAL_57]], %[[VAL_59]] : tensor<3xi64>
  // CHECK:           %[[VAL_61:.*]] = tensor.reshape %[[VAL_33]](%[[VAL_60]]) : (tensor<1x1x2x6xf32>, tensor<3xi64>) -> tensor<1x2x6xf32>
  // CHECK:           %[[VAL_62:.*]] = tensor.cast %[[VAL_61]] : tensor<1x2x6xf32> to tensor<1x2x6xf32>
  // CHECK:           %[[VAL_63:.*]] = torch_c.from_builtin_tensor %[[VAL_62]] : tensor<1x2x6xf32> -> !torch.vtensor<[1,2,6],f32>
  // CHECK:           return %[[VAL_63]] : !torch.vtensor<[1,2,6],f32>
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %4 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float1.000000e00 : !torch.vtensor<[1,2,6],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,2,6],f32>
  return %4 : !torch.vtensor<[1,2,6],f32>
}

// -----

// Test rotary embedding with dynamic batch and sequence dimensions (rank 4)
// Input shape: [batch, num_heads, seq, head_size] = [?, 1, ?, 6]
// batch (dim 0) and seq (dim 2) are dynamic; num_heads (dim 1) and head_size (dim 3) must be static
// CHECK-LABEL:   func.func @test_rotary_embedding_dynamic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,1,?,6],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,1,?,6],f32>
func.func @test_rotary_embedding_dynamic(%arg0: !torch.vtensor<[?,1,?,6],f32>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,1,?,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_3]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
  // CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,1,?,6],f32> -> tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_8:.*]] = torch.constant.none
  // CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.constant.float 1.000000e+00
  // CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_7]], %[[VAL_14]] : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_16:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_7]], %[[VAL_16]] : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_18:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_19:.*]] = tensor.dim %[[VAL_7]], %[[VAL_18]] : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_20:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_21:.*]] = tensor.dim %[[VAL_7]], %[[VAL_20]] : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_22:.*]] = tensor.empty(%[[VAL_19]], %[[VAL_21]]) : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_23:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_24:.*]] = linalg.fill ins(%[[VAL_23]] : f32) outs(%[[VAL_22]] : tensor<?x1x?x6xf32>) -> tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_25:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:           %[[VAL_26:.*]] = arith.constant -1.000000e+00 : f32
  // CHECK:           %[[VAL_27:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_28:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_29:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_30:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_31:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_7]], %[[VAL_6]] : tensor<?x1x?x6xf32>, tensor<?x?xi64>) outs(%[[VAL_24]] : tensor<?x1x?x6xf32>) {
  // CHECK:           ^bb0(%[[VAL_32:.*]]: f32, %[[VAL_33:.*]]: i64, %[[VAL_34:.*]]: f32):
  // CHECK:             %[[VAL_35:.*]] = linalg.index 0 : index
  // CHECK:             %[[VAL_36:.*]] = linalg.index 1 : index
  // CHECK:             %[[VAL_37:.*]] = linalg.index 2 : index
  // CHECK:             %[[VAL_38:.*]] = linalg.index 3 : index
  // CHECK:             %[[VAL_39:.*]] = arith.remsi %[[VAL_38]], %[[VAL_30]] : index
  // CHECK:             %[[VAL_40:.*]] = arith.cmpi sge, %[[VAL_38]], %[[VAL_30]] : index
  // CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_38]], %[[VAL_30]] : index
  // CHECK:             %[[VAL_42:.*]] = arith.remsi %[[VAL_41]], %[[VAL_29]] : index
  // CHECK:             %[[VAL_43:.*]] = arith.index_cast %[[VAL_33]] : i64 to index
  // CHECK:             %[[VAL_44:.*]] = tensor.extract %[[VAL_5]]{{\[}}%[[VAL_43]], %[[VAL_39]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_45:.*]] = tensor.extract %[[VAL_4]]{{\[}}%[[VAL_43]], %[[VAL_39]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_46:.*]] = tensor.extract %[[VAL_7]]{{\[}}%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_42]]] : tensor<?x1x?x6xf32>
  // CHECK:             %[[VAL_47:.*]] = arith.select %[[VAL_40]], %[[VAL_25]], %[[VAL_26]] : f32
  // CHECK:             %[[VAL_48:.*]] = arith.mulf %[[VAL_32]], %[[VAL_44]] : f32
  // CHECK:             %[[VAL_49:.*]] = arith.mulf %[[VAL_46]], %[[VAL_45]] : f32
  // CHECK:             %[[VAL_50:.*]] = arith.mulf %[[VAL_49]], %[[VAL_47]] : f32
  // CHECK:             %[[VAL_51:.*]] = arith.addf %[[VAL_48]], %[[VAL_50]] : f32
  // CHECK:             linalg.yield %[[VAL_51]] : f32
  // CHECK:           } -> tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_52:.*]] = tensor.cast %[[VAL_31]] : tensor<?x1x?x6xf32> to tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_53:.*]] = torch_c.from_builtin_tensor %[[VAL_52]] : tensor<?x1x?x6xf32> -> !torch.vtensor<[?,1,?,6],f32>
  // CHECK:           return %[[VAL_53]] : !torch.vtensor<[?,1,?,6],f32>
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %4 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float1.000000e00 : !torch.vtensor<[?,1,?,6],f32>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[?,1,?,6],f32>
  return %4 : !torch.vtensor<[?,1,?,6],f32>
}

// -----

// Test rotary embedding with rank 3 input and dynamic batch/sequence dimensions
// CHECK-LABEL:   func.func @test_rotary_embedding_rank3_dynamic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,6],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,?,6],f32>
func.func @test_rotary_embedding_rank3_dynamic(%arg0: !torch.vtensor<[?,?,6],f32>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,?,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_3]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
  // CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,6],f32> -> tensor<?x?x6xf32>
  // CHECK:           %[[VAL_8:.*]] = torch.constant.none
  // CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.constant.float 1.000000e+00
  // CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_7]], %[[VAL_14]] : tensor<?x?x6xf32>
  // CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_7]], %[[VAL_16]] : tensor<?x?x6xf32>
  // CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_15]] : index to i64
  // CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_20:.*]] = arith.index_cast %[[VAL_19]] : index to i64
  // CHECK:           %[[VAL_21:.*]] = arith.index_cast %[[VAL_17]] : index to i64
  // CHECK:           %[[VAL_22:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_23:.*]] = arith.index_cast %[[VAL_22]] : index to i64
  // CHECK:           %[[VAL_24:.*]] = tensor.from_elements %[[VAL_18]], %[[VAL_20]], %[[VAL_21]], %[[VAL_23]] : tensor<4xi64>
  // CHECK:           %[[VAL_25:.*]] = tensor.reshape %[[VAL_7]](%[[VAL_24]]) : (tensor<?x?x6xf32>, tensor<4xi64>) -> tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_26:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_27:.*]] = tensor.dim %[[VAL_25]], %[[VAL_26]] : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_28:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_29:.*]] = tensor.dim %[[VAL_25]], %[[VAL_28]] : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_30:.*]] = tensor.empty(%[[VAL_27]], %[[VAL_29]]) : tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_31:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_32:.*]] = linalg.fill ins(%[[VAL_31]] : f32) outs(%[[VAL_30]] : tensor<?x1x?x6xf32>) -> tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_33:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:           %[[VAL_34:.*]] = arith.constant -1.000000e+00 : f32
  // CHECK:           %[[VAL_35:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_36:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_37:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_38:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_39:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_25]], %[[VAL_6]] : tensor<?x1x?x6xf32>, tensor<?x?xi64>) outs(%[[VAL_32]] : tensor<?x1x?x6xf32>) {
  // CHECK:           ^bb0(%[[VAL_40:.*]]: f32, %[[VAL_41:.*]]: i64, %[[VAL_42:.*]]: f32):
  // CHECK:             %[[VAL_43:.*]] = linalg.index 0 : index
  // CHECK:             %[[VAL_44:.*]] = linalg.index 1 : index
  // CHECK:             %[[VAL_45:.*]] = linalg.index 2 : index
  // CHECK:             %[[VAL_46:.*]] = linalg.index 3 : index
  // CHECK:             %[[VAL_47:.*]] = arith.remsi %[[VAL_46]], %[[VAL_38]] : index
  // CHECK:             %[[VAL_48:.*]] = arith.cmpi sge, %[[VAL_46]], %[[VAL_38]] : index
  // CHECK:             %[[VAL_49:.*]] = arith.addi %[[VAL_46]], %[[VAL_38]] : index
  // CHECK:             %[[VAL_50:.*]] = arith.remsi %[[VAL_49]], %[[VAL_37]] : index
  // CHECK:             %[[VAL_51:.*]] = arith.index_cast %[[VAL_41]] : i64 to index
  // CHECK:             %[[VAL_52:.*]] = tensor.extract %[[VAL_5]]{{\[}}%[[VAL_51]], %[[VAL_47]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_53:.*]] = tensor.extract %[[VAL_4]]{{\[}}%[[VAL_51]], %[[VAL_47]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_54:.*]] = tensor.extract %[[VAL_25]]{{\[}}%[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_50]]] : tensor<?x1x?x6xf32>
  // CHECK:             %[[VAL_55:.*]] = arith.select %[[VAL_48]], %[[VAL_33]], %[[VAL_34]] : f32
  // CHECK:             %[[VAL_56:.*]] = arith.mulf %[[VAL_40]], %[[VAL_52]] : f32
  // CHECK:             %[[VAL_57:.*]] = arith.mulf %[[VAL_54]], %[[VAL_53]] : f32
  // CHECK:             %[[VAL_58:.*]] = arith.mulf %[[VAL_57]], %[[VAL_55]] : f32
  // CHECK:             %[[VAL_59:.*]] = arith.addf %[[VAL_56]], %[[VAL_58]] : f32
  // CHECK:             linalg.yield %[[VAL_59]] : f32
  // CHECK:           } -> tensor<?x1x?x6xf32>
  // CHECK:           %[[VAL_60:.*]] = arith.index_cast %[[VAL_15]] : index to i64
  // CHECK:           %[[VAL_61:.*]] = arith.index_cast %[[VAL_17]] : index to i64
  // CHECK:           %[[VAL_62:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_63:.*]] = arith.index_cast %[[VAL_62]] : index to i64
  // CHECK:           %[[VAL_64:.*]] = tensor.from_elements %[[VAL_60]], %[[VAL_61]], %[[VAL_63]] : tensor<3xi64>
  // CHECK:           %[[VAL_65:.*]] = tensor.reshape %[[VAL_39]](%[[VAL_64]]) : (tensor<?x1x?x6xf32>, tensor<3xi64>) -> tensor<?x?x6xf32>
  // CHECK:           %[[VAL_66:.*]] = tensor.cast %[[VAL_65]] : tensor<?x?x6xf32> to tensor<?x?x6xf32>
  // CHECK:           %[[VAL_67:.*]] = torch_c.from_builtin_tensor %[[VAL_66]] : tensor<?x?x6xf32> -> !torch.vtensor<[?,?,6],f32>
  // CHECK:           return %[[VAL_67]] : !torch.vtensor<[?,?,6],f32>
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %4 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float1.000000e00 : !torch.vtensor<[?,?,6],f32>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[?,?,6],f32>
  return %4 : !torch.vtensor<[?,?,6],f32>
}

// -----

// Test rotary embedding with scale != 1.0
// CHECK-LABEL:   func.func @test_rotary_embedding_with_scale(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[1,3,2,6],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[1,2],si64>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[1,3,2,6],f32>
func.func @test_rotary_embedding_with_scale(%arg0: !torch.vtensor<[1,3,2,6],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[1,3,2,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_3]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK:           %[[VAL_6:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[1,2],si64> -> tensor<1x2xi64>
  // CHECK:           %[[VAL_7:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,3,2,6],f32> -> tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_8:.*]] = torch.constant.none
  // CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.constant.float 2.000000e+00
  // CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_15:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_16:.*]] = linalg.fill ins(%[[VAL_15]] : f32) outs(%[[VAL_14]] : tensor<1x3x2x6xf32>) -> tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_17:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:           %[[VAL_18:.*]] = arith.constant -1.000000e+00 : f32
  // CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_21:.*]] = arith.constant 6 : index
  // CHECK:           %[[VAL_22:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_23:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_7]], %[[VAL_6]] : tensor<1x3x2x6xf32>, tensor<1x2xi64>) outs(%[[VAL_16]] : tensor<1x3x2x6xf32>) {
  // CHECK:           ^bb0(%[[VAL_24:.*]]: f32, %[[VAL_25:.*]]: i64, %[[VAL_26:.*]]: f32):
  // CHECK:             %[[VAL_27:.*]] = linalg.index 0 : index
  // CHECK:             %[[VAL_28:.*]] = linalg.index 1 : index
  // CHECK:             %[[VAL_29:.*]] = linalg.index 2 : index
  // CHECK:             %[[VAL_30:.*]] = linalg.index 3 : index
  // CHECK:             %[[VAL_31:.*]] = arith.remsi %[[VAL_30]], %[[VAL_22]] : index
  // CHECK:             %[[VAL_32:.*]] = arith.cmpi sge, %[[VAL_30]], %[[VAL_22]] : index
  // CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_30]], %[[VAL_22]] : index
  // CHECK:             %[[VAL_34:.*]] = arith.remsi %[[VAL_33]], %[[VAL_21]] : index
  // CHECK:             %[[VAL_35:.*]] = arith.index_cast %[[VAL_25]] : i64 to index
  // CHECK:             %[[VAL_36:.*]] = tensor.extract %[[VAL_5]]{{\[}}%[[VAL_35]], %[[VAL_31]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_37:.*]] = tensor.extract %[[VAL_4]]{{\[}}%[[VAL_35]], %[[VAL_31]]] : tensor<4x3xf32>
  // CHECK:             %[[VAL_38:.*]] = tensor.extract %[[VAL_7]]{{\[}}%[[VAL_27]], %[[VAL_28]], %[[VAL_29]], %[[VAL_34]]] : tensor<1x3x2x6xf32>
  // CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_32]], %[[VAL_17]], %[[VAL_18]] : f32
  // CHECK:             %[[VAL_40:.*]] = arith.mulf %[[VAL_24]], %[[VAL_36]] : f32
  // CHECK:             %[[VAL_41:.*]] = arith.mulf %[[VAL_38]], %[[VAL_37]] : f32
  // CHECK:             %[[VAL_42:.*]] = arith.mulf %[[VAL_41]], %[[VAL_39]] : f32
  // CHECK:             %[[VAL_43:.*]] = arith.addf %[[VAL_40]], %[[VAL_42]] : f32
  // CHECK:             linalg.yield %[[VAL_43]] : f32
  // CHECK:           } -> tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_44:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK:           %[[VAL_45:.*]] = tensor.empty() : tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_46:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_23]] : tensor<1x3x2x6xf32>) outs(%[[VAL_45]] : tensor<1x3x2x6xf32>) {
  // CHECK:           ^bb0(%[[VAL_47:.*]]: f32, %[[VAL_48:.*]]: f32):
  // CHECK:             %[[VAL_49:.*]] = arith.mulf %[[VAL_47]], %[[VAL_44]] : f32
  // CHECK:             linalg.yield %[[VAL_49]] : f32
  // CHECK:           } -> tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_50:.*]] = tensor.cast %[[VAL_46]] : tensor<1x3x2x6xf32> to tensor<1x3x2x6xf32>
  // CHECK:           %[[VAL_51:.*]] = torch_c.from_builtin_tensor %[[VAL_50]] : tensor<1x3x2x6xf32> -> !torch.vtensor<[1,3,2,6],f32>
  // CHECK:           return %[[VAL_51]] : !torch.vtensor<[1,3,2,6],f32>
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %4 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float2.000000e00 : !torch.vtensor<[1,3,2,6],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,3,2,6],f32>
  return %4 : !torch.vtensor<[1,3,2,6],f32>
}

// -----

// CHECK-LABEL: func.func @torch.ops.aten.replication_pad3d$basic(
// CHECK-SAME: %[[ARG_0:.*]]: !torch.vtensor<[4,3,5],f32>) -> !torch.vtensor<[7,7,6],f32>
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,3,5],f32> -> tensor<4x3x5xf32>
// CHECK-DAG: %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG: %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[IDX5:.*]] = arith.constant 5 : index
// CHECK: %[[IDX1:.*]] = arith.constant 1 : index
// CHECK: %[[SUB2:.*]] = arith.subi %[[IDX5]], %[[IDX1]] : index
// CHECK: %[[SLICE1:.*]] = tensor.extract_slice %[[T0]][0, 0, %[[SUB2]]] [4, 3, 1] [1, 1, 1] : tensor<4x3x5xf32> to tensor<4x3x1xf32>
// CHECK: %[[CONCAT1:.*]] = tensor.concat dim(2) %[[T0]], %[[SLICE1]] : (tensor<4x3x5xf32>, tensor<4x3x1xf32>) -> tensor<4x3x6xf32>
// CHECK: %[[SLICE2:.*]] = tensor.extract_slice %[[CONCAT1]][0, 0, 0] [4, 1, 6] [1, 1, 1] : tensor<4x3x6xf32> to tensor<4x1x6xf32>
// CHECK: %[[CONCAT2:.*]] = tensor.concat dim(1) %[[SLICE2]], %[[SLICE2]], %[[SLICE2]] : (tensor<4x1x6xf32>, tensor<4x1x6xf32>, tensor<4x1x6xf32>) -> tensor<4x3x6xf32>
// CHECK: %[[SUB3:.*]] = arith.subi {{.*}}, {{.*}} : index
// CHECK: %[[SLICE3:.*]] = tensor.extract_slice %[[CONCAT1]][0, %[[SUB3]], 0] [4, 1, 6] [1, 1, 1] : tensor<4x3x6xf32> to tensor<4x1x6xf32>
// CHECK: %[[CONCAT3:.*]] = tensor.concat dim(1) %[[CONCAT2]], %[[CONCAT1]], %[[SLICE3]] : (tensor<4x3x6xf32>, tensor<4x3x6xf32>, tensor<4x1x6xf32>) -> tensor<4x7x6xf32>
// CHECK: %[[SUB4:.*]] = arith.subi {{.*}}, {{.*}} : index
// CHECK: %[[SLICE4:.*]] = tensor.extract_slice %[[CONCAT3]][%[[SUB4]], 0, 0] [1, 7, 6] [1, 1, 1] : tensor<4x7x6xf32> to tensor<1x7x6xf32>
// CHECK: %[[CONCAT4:.*]] = tensor.concat dim(0) %[[SLICE4]], %[[SLICE4]], %[[SLICE4]] : (tensor<1x7x6xf32>, tensor<1x7x6xf32>, tensor<1x7x6xf32>) -> tensor<3x7x6xf32>
// CHECK: %[[CONCAT5:.*]] = tensor.concat dim(0) %[[CONCAT3]], %[[CONCAT4]] : (tensor<4x7x6xf32>, tensor<3x7x6xf32>) -> tensor<7x7x6xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[CONCAT5]] : tensor<7x7x6xf32> to tensor<7x7x6xf32>
// CHECK: %[[OUT:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<7x7x6xf32> -> !torch.vtensor<[7,7,6],f32>
// CHECK: return %[[OUT]] : !torch.vtensor<[7,7,6],f32>
func.func @torch.ops.aten.replication_pad3d$basic(%arg0: !torch.vtensor<[4,3,5],f32>) -> !torch.vtensor<[7,7,6],f32> {
  %c0 = torch.constant.int 0
  %c1 = torch.constant.int 1
  %c3 = torch.constant.int 3
  %padding = torch.prim.ListConstruct %c0, %c1, %c3, %c1, %c0, %c3 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.replication_pad3d %arg0, %padding : !torch.vtensor<[4,3,5],f32>, !torch.list<int> -> !torch.vtensor<[7,7,6],f32>
  return %0 : !torch.vtensor<[7,7,6],f32>
}

// -----

// CHECK-LABEL: func.func @torch.ops.aten.replication_pad2d$basic
// CHECK: [[TENSOR:%.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,3,5],f32> -> tensor<4x3x5xf32>
// CHECK: [[INT1:%.*]] = torch.constant.int 1
// CHECK: [[INT3:%.*]] = torch.constant.int 3
// CHECK: [[PAD_LIST:%.*]] = torch.prim.ListConstruct [[INT1]], [[INT3]], [[INT1]], [[INT3]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: [[SLICE_LEFT:%.*]] = tensor.extract_slice [[TENSOR]][0, 0, 0] [4, 3, 1] [1, 1, 1] : tensor<4x3x5xf32> to tensor<4x3x1xf32>
// CHECK: [[IDX_RIGHT:%.*]] = arith.subi %c5_5, %c1_6 : index
// CHECK: [[SLICE_RIGHT:%.*]] = tensor.extract_slice [[TENSOR]][0, 0, [[IDX_RIGHT]]] [4, 3, 1] [1, 1, 1] : tensor<4x3x5xf32> to tensor<4x3x1xf32>
// CHECK: [[RIGHT_PAD:%.*]] = tensor.concat dim(2) [[SLICE_RIGHT]], [[SLICE_RIGHT]], [[SLICE_RIGHT]] : (tensor<4x3x1xf32>, tensor<4x3x1xf32>, tensor<4x3x1xf32>) -> tensor<4x3x3xf32>
// CHECK: [[PADDED_W:%.*]] = tensor.concat dim(2) [[SLICE_LEFT]], [[TENSOR]], [[RIGHT_PAD]] : (tensor<4x3x1xf32>, tensor<4x3x5xf32>, tensor<4x3x3xf32>) -> tensor<4x3x9xf32>
// CHECK: [[SLICE_TOP:%.*]] = tensor.extract_slice [[PADDED_W]][0, 0, 0] [4, 1, 9] [1, 1, 1] : tensor<4x3x9xf32> to tensor<4x1x9xf32>
// CHECK: [[IDX_BOTTOM:%.*]] = arith.subi %c3_18, %c1_21 : index
// CHECK: [[SLICE_BOTTOM:%.*]] = tensor.extract_slice [[PADDED_W]][0, [[IDX_BOTTOM]], 0] [4, 1, 9] [1, 1, 1] : tensor<4x3x9xf32> to tensor<4x1x9xf32>
// CHECK: [[BOTTOM_PAD:%.*]] = tensor.concat dim(1) [[SLICE_BOTTOM]], [[SLICE_BOTTOM]], [[SLICE_BOTTOM]] : (tensor<4x1x9xf32>, tensor<4x1x9xf32>, tensor<4x1x9xf32>) -> tensor<4x3x9xf32>
// CHECK: [[PADDED_HW:%.*]] = tensor.concat dim(1) [[SLICE_TOP]], [[PADDED_W]], [[BOTTOM_PAD]] : (tensor<4x1x9xf32>, tensor<4x3x9xf32>, tensor<4x3x9xf32>) -> tensor<4x7x9xf32>
// CHECK: [[CAST:%.*]] = tensor.cast [[PADDED_HW]] : tensor<4x7x9xf32> to tensor<4x7x9xf32>
// CHECK: [[TORCH_TENSOR:%.*]] = torch_c.from_builtin_tensor [[CAST]] : tensor<4x7x9xf32> -> !torch.vtensor<[4,7,9],f32>
// CHECK: return [[TORCH_TENSOR]] : !torch.vtensor<[4,7,9],f32>
func.func @torch.ops.aten.replication_pad2d$basic(%arg0: !torch.vtensor<[4,3,5],f32>) -> !torch.vtensor<[4,7,9],f32> {
  %c1 = torch.constant.int 1
  %c3 = torch.constant.int 3
  %padding = torch.prim.ListConstruct %c1, %c3, %c1, %c3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.replication_pad2d %arg0, %padding : !torch.vtensor<[4,3,5],f32>, !torch.list<int> -> !torch.vtensor<[4,7,9],f32>
  return %0 : !torch.vtensor<[4,7,9],f32>
}

// -----


// CHECK-LABEL: func.func @torch.ops.aten.replication_pad1d$basic
// CHECK: [[TENSOR:%.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[3,5],f32> -> tensor<3x5xf32>
// CHECK: [[INT1:%.*]] = torch.constant.int 1
// CHECK: [[INT2:%.*]] = torch.constant.int 2
// CHECK: [[PAD_LIST:%.*]] = torch.prim.ListConstruct [[INT1]], [[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[C3:%.*]] = arith.constant 3 : index
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[C5:%.*]] = arith.constant 5 : index
// CHECK: [[SLICE_LEFT:%.*]] = tensor.extract_slice [[TENSOR]][0, 0] [3, 1] [1, 1] : tensor<3x5xf32> to tensor<3x1xf32>
// CHECK: [[C0_0:%.*]] = arith.constant 0 : index
// CHECK: [[C3_1:%.*]] = arith.constant 3 : index
// CHECK: [[C1_2:%.*]] = arith.constant 1 : index
// CHECK: [[C5_3:%.*]] = arith.constant 5 : index
// CHECK: [[C1_4:%.*]] = arith.constant 1 : index
// CHECK: [[IDX_RIGHT:%.*]] = arith.subi [[C5_3]], [[C1_4]] : index
// CHECK: [[SLICE_RIGHT:%.*]] = tensor.extract_slice [[TENSOR]][0, [[IDX_RIGHT]]] [3, 1] [1, 1] : tensor<3x5xf32> to tensor<3x1xf32>
// CHECK: [[RIGHT_PAD:%.*]] = tensor.concat dim(1) [[SLICE_RIGHT]], [[SLICE_RIGHT]] : (tensor<3x1xf32>, tensor<3x1xf32>) -> tensor<3x2xf32>
// CHECK: [[PADDED:%.*]] = tensor.concat dim(1) [[SLICE_LEFT]], [[TENSOR]], [[RIGHT_PAD]] : (tensor<3x1xf32>, tensor<3x5xf32>, tensor<3x2xf32>) -> tensor<3x8xf32>
// CHECK: [[CAST:%.*]] = tensor.cast [[PADDED]] : tensor<3x8xf32> to tensor<3x8xf32>
// CHECK: [[TORCH_TENSOR:%.*]] = torch_c.from_builtin_tensor [[CAST]] : tensor<3x8xf32> -> !torch.vtensor<[3,8],f32>
// CHECK: return [[TORCH_TENSOR]] : !torch.vtensor<[3,8],f32>
func.func @torch.ops.aten.replication_pad1d$basic(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3,8],f32> {
  %c1 = torch.constant.int 1
  %c2 = torch.constant.int 2
  %padding = torch.prim.ListConstruct %c1, %c2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.replication_pad1d %arg0, %padding : !torch.vtensor<[3,5],f32>, !torch.list<int> -> !torch.vtensor<[3,8],f32>
  return %0 : !torch.vtensor<[3,8],f32>
}

// -----
// COM: this test only locks down that allow-non-finites config produces inf/realmax correctly
// CHECK-LABEL: func.func @torch.aten.min.dim$basic
// CHECK: arith.constant 0x7F800000 : f32
// CHECK-NOT: arith.constant 3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES: arith.constant 3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES-NOT: arith.constant 0x7F800000 : f32
func.func @torch.aten.min.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.min.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}

// -----
// COM: this test only locks down that allow-non-finites config produces inf/realmax correctly
// CHECK-LABEL: func.func @torch.aten.max.dim$basic
// CHECK: arith.constant 0xFF800000 : f32
// CHECK-NOT: arith.constant -3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES: arith.constant -3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES-NOT: arith.constant 0xFF800000 : f32
func.func @torch.aten.max.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.max.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}
