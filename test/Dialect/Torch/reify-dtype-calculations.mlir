// RUN: torch-mlir-opt -torch-reify-dtype-calculations -split-input-file %s | FileCheck %s

// CHECK: module {
// CHECK: func.func private @__torch_mlir_dtype_fn.aten.tanh(

// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[RESULT:.*]] = torch.dtype.calculate  {
// CHECK:             %[[TANH:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor -> !torch.vtensor
// CHECK:             torch.dtype.calculate.yield %[[TANH]] : !torch.vtensor
// CHECK:           } dtypes  {
// CHECK:             %[[SIZE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor -> !torch.list<int>
// CHECK:             %[[RANK:.*]] = torch.aten.len.t %[[SIZE]] : !torch.list<int> -> !torch.int
// CHECK:             %[[DTYPE:.*]] = torch.prim.dtype %[[ARG]] : !torch.vtensor -> !torch.int
// CHECK:             %[[RESULT_DTYPE:.*]] = func.call @__torch_mlir_dtype_fn.aten.tanh(%[[RANK]], %[[DTYPE]]) : (!torch.int, !torch.int) -> !torch.int
// CHECK:             torch.dtype.calculate.yield.dtypes %[[RESULT_DTYPE]] : !torch.int
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func.func @basic(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func private @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.library_generator.promote_dtypes(
// CHECK:          {{.*}} = torch.promote_dtypes {{.*}} : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int

// CHECK-LABEL:   func.func private @__torch_mlir_dtype_fn.aten.floor_divide(
// CHECK:           {{.*}} = call @__torch__.torch_mlir.dialects.torch.importer.jit_ir.build_tools.library_generator.promote_dtypes({{.*}}

// CHECK-LABEL:   func.func @op_with_dtype_promotion(
// CHECK:             {{.*}} = func.call @__torch_mlir_dtype_fn.aten.floor_divide({{.*}}
func.func @op_with_dtype_promotion(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.floor_divide %arg0, %arg1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func private @__torch_mlir_dtype_fn.aten.floor_divide(

// CHECK-LABEL:   func.func @turn_tensors_into_rank_and_dtype_args(
// CHECK-SAME:                                                     %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                                     %[[ARG1:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:             %[[SIZE0:.*]] = torch.aten.size %[[ARG0]] : !torch.vtensor -> !torch.list<int>
// CHECK:             %[[RANK0:.*]] = torch.aten.len.t %[[SIZE0]] : !torch.list<int> -> !torch.int
// CHECK:             %[[DTYPE0:.*]] = torch.prim.dtype %[[ARG0]] : !torch.vtensor -> !torch.int
// CHECK:             %[[SIZE1:.*]] = torch.aten.size %[[ARG1]] : !torch.vtensor -> !torch.list<int>
// CHECK:             %[[RANK1:.*]] = torch.aten.len.t %[[SIZE1]] : !torch.list<int> -> !torch.int
// CHECK:             %[[DTYPE1:.*]] = torch.prim.dtype %[[ARG1]] : !torch.vtensor -> !torch.int
// CHECK:             {{.*}} = func.call @__torch_mlir_dtype_fn.aten.floor_divide(%[[RANK0]], %[[DTYPE0]], %[[RANK1]], %[[DTYPE1]]) : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.int
func.func @turn_tensors_into_rank_and_dtype_args(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.floor_divide %arg0, %arg1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}
