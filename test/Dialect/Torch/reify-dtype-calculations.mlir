// RUN: torch-mlir-opt -torch-reify-dtype-calculations -split-input-file %s | FileCheck %s

// CHECK: module {
// CHECK: func.func private @__torch_mlir_dtype_fn.aten.expm1(

// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[RESULT:.*]] = torch.dtype.calculate  {
// CHECK:             %[[EXPM1:.*]] = torch.aten.expm1 %[[ARG]] : !torch.vtensor -> !torch.vtensor
// CHECK:             torch.dtype.calculate.yield %[[EXPM1]] : !torch.vtensor
// CHECK:           } dtypes  {
// CHECK:             %[[SIZE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor -> !torch.list<int>
// CHECK:             %[[RANK:.*]] = torch.aten.len.t %[[SIZE]] : !torch.list<int> -> !torch.int
// CHECK:             %[[DTYPE:.*]] = torch.prim.dtype %[[ARG]] : !torch.vtensor -> !torch.int
// CHECK:             %[[RANK_DTYPE:.*]] = torch.prim.TupleConstruct %[[RANK]], %[[DTYPE]] : !torch.int, !torch.int -> !torch.tuple<int, int>
// CHECK:             %[[RESULT_DTYPE:.*]] = func.call @__torch_mlir_dtype_fn.aten.expm1(%[[RANK_DTYPE]]) : (!torch.tuple<int, int>) -> !torch.int
// CHECK:             torch.dtype.calculate.yield.dtypes %[[RESULT_DTYPE]] : !torch.int
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func.func @basic(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.expm1 %arg0 : !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func private @__torch__.torch_mlir.jit_ir_importer.build_tools.library_generator.promote_dtypes(
// CHECK:          {{.*}} = torch.promote_dtypes {{.*}} : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int

// CHECK-LABEL:   func.func private @__torch_mlir_dtype_fn.aten.floor_divide(
// CHECK:           {{.*}} = call @__torch__.torch_mlir.jit_ir_importer.build_tools.library_generator.promote_dtypes({{.*}}

// CHECK-LABEL:   func.func @op_with_dtype_promotion(
// CHECK:             {{.*}} = func.call @__torch_mlir_dtype_fn.aten.floor_divide({{.*}}
func.func @op_with_dtype_promotion(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.floor_divide %arg0, %arg1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func private @__torch_mlir_dtype_fn.aten._convolution.deprecated(

// CHECK-LABEL:   func.func @op_with_optional_tensor_arg$none(
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:             %[[OPTIONAL_TUPLE:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<tuple<int, int>>
// CHECK:             {{.*}} = func.call @__torch_mlir_dtype_fn.aten._convolution.deprecated({{.*}}, %[[OPTIONAL_TUPLE]], {{.*}}) : ({{.*}}, !torch.optional<tuple<int, int>>, {{.*}}) -> !torch.int
func.func @op_with_optional_tensor_arg$none(%input: !torch.vtensor, %weight: !torch.vtensor, %stride: !torch.list<int>, %padding: !torch.list<int>, %dilation: !torch.list<int>, %transposed: !torch.bool, %output_padding: !torch.list<int>, %groups: !torch.int) -> !torch.vtensor {
  %bias_none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten._convolution.deprecated %input, %weight, %bias_none, %stride, %padding, %dilation, %transposed, %output_padding, %groups, %false, %false, %false : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor
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
// CHECK:             %[[RANK_DTYPE0:.*]] = torch.prim.TupleConstruct %[[RANK0]], %[[DTYPE0]] : !torch.int, !torch.int -> !torch.tuple<int, int>
// CHECK:             %[[SIZE1:.*]] = torch.aten.size %[[ARG1]] : !torch.vtensor -> !torch.list<int>
// CHECK:             %[[RANK1:.*]] = torch.aten.len.t %[[SIZE1]] : !torch.list<int> -> !torch.int
// CHECK:             %[[DTYPE1:.*]] = torch.prim.dtype %[[ARG1]] : !torch.vtensor -> !torch.int
// CHECK:             %[[RANK_DTYPE1:.*]] = torch.prim.TupleConstruct %[[RANK1]], %[[DTYPE1]] : !torch.int, !torch.int -> !torch.tuple<int, int>
// CHECK:             {{.*}} = func.call @__torch_mlir_dtype_fn.aten.floor_divide(%[[RANK_DTYPE0]], %[[RANK_DTYPE1]]) : (!torch.tuple<int, int>, !torch.tuple<int, int>) -> !torch.int
func.func @turn_tensors_into_rank_and_dtype_args(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.floor_divide %arg0, %arg1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func private @__torch_mlir_dtype_fn.aten.arange(

// CHECK-LABEL:   func.func @derefine_int_to_number() -> !torch.vtensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:             %[[NUMBER:.*]] = torch.derefine %[[INT1]] : !torch.int to !torch.number
// CHECK:             {{.*}} = func.call @__torch_mlir_dtype_fn.aten.arange(%[[NUMBER]], {{.*}}) : (!torch.number, {{.*}}) -> !torch.int
func.func @derefine_int_to_number() -> !torch.vtensor {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.aten.arange %int1, %none, %none, %none, %none : !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func private @__torch_mlir_dtype_fn.onnx.rotary_embedding(

// CHECK-LABEL:   func.func @custom_onnx_rotary_embedding(
// CHECK:           func.call @__torch_mlir_dtype_fn.onnx.rotary_embedding
func.func @custom_onnx_rotary_embedding(%arg0: !torch.vtensor<[1,3,2,6],f32>, %arg1: !torch.vtensor, %arg2: !torch.vtensor, %arg3: !torch.vtensor) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %4 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float1.000000e00 : !torch.vtensor<[1,3,2,6],f32>, !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor
  return %4 : !torch.vtensor
}
