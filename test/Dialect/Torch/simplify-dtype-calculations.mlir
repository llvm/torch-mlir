// RUN: torch-mlir-opt -torch-simplify-dtype-calculations -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                     %[[ARG:.*]]: !torch.vtensor<*,f32>) -> !torch.vtensor {
// CHECK:           %[[DTYPE_INT:.*]] = torch.constant.int 6
// CHECK:           %[[RESULT:.*]] = torch.dtype.calculate {
// CHECK:             %[[TANH:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor<*,f32> -> !torch.vtensor<*,f32>
// CHECK:             torch.dtype.calculate.yield %[[TANH]] : !torch.vtensor<*,f32>
// CHECK:           } dtypes {
// CHECK:             torch.dtype.calculate.yield.dtypes %[[DTYPE_INT]] : !torch.int
// CHECK:           } : !torch.vtensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RESULT]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @basic(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %0 = torch.dtype.calculate {
    %1 = torch.aten.tanh %arg0 : !torch.vtensor<*,f32> -> !torch.vtensor
    torch.dtype.calculate.yield %1 : !torch.vtensor
  } dtypes {
    %2 = torch.prim.dtype %arg0 : !torch.vtensor<*,f32> -> !torch.int
    torch.dtype.calculate.yield.dtypes %2 : !torch.int
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$tensor_tensor_same_category_different_width(
// CHECK:             {{.*}} = torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[1],f64>
func.func @promote_dtypes$tensor_tensor_same_category_different_width(%arg0: !torch.vtensor<[1],f32>, %arg1: !torch.vtensor<[1],f64>, %arg2: !torch.float) {
  %int1 = torch.constant.int 1
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[1],f32>, !torch.vtensor<[1],f64>, !torch.float -> !torch.vtensor<[1],unk>
    torch.dtype.calculate.yield %1 : !torch.vtensor<[1],unk>
  } dtypes {
    %ranks = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<optional<int>>
    %f32_dtype = torch.prim.dtype %arg0 : !torch.vtensor<[1],f32> -> !torch.int
    %f64_dtype = torch.prim.dtype %arg1 : !torch.vtensor<[1],f64> -> !torch.int
    %dtypes = torch.prim.ListConstruct %f32_dtype, %f64_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %3 : !torch.int
  } : !torch.vtensor<[1],unk>
  return
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$tensor_tensor_different_category(
// CHECK:             {{.*}} = torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[1],f64>
func.func @promote_dtypes$tensor_tensor_different_category(%arg0: !torch.vtensor<[1],si32>, %arg1: !torch.vtensor<[1],f64>, %arg2: !torch.float) {
  %int1 = torch.constant.int 1
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[1],si32>, !torch.vtensor<[1],f64>, !torch.float -> !torch.vtensor<[1],unk>
    torch.dtype.calculate.yield %1 : !torch.vtensor<[1],unk>
  } dtypes {
    %si32_dtype = torch.prim.dtype %arg0 : !torch.vtensor<[1],si32> -> !torch.int
    %f64_dtype = torch.prim.dtype %arg1 : !torch.vtensor<[1],f64> -> !torch.int
    %ranks = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<optional<int>>
    %dtypes = torch.prim.ListConstruct %si32_dtype, %f64_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %3 : !torch.int
  } : !torch.vtensor<[1],unk>
  return
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$tensor_tensor_same_category_zero_rank_wider(
// CHECK:             {{.*}} = torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[1],f32>
func.func @promote_dtypes$tensor_tensor_same_category_zero_rank_wider(%arg0: !torch.vtensor<[1],f32>, %arg1: !torch.vtensor<[],f64>, %arg2: !torch.int) {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[1],f32>, !torch.vtensor<[],f64>, !torch.int -> !torch.vtensor<[1],unk>
    torch.dtype.calculate.yield %1 : !torch.vtensor<[1],unk>
  } dtypes {
    %f32_dtype = torch.prim.dtype %arg0 : !torch.vtensor<[1],f32> -> !torch.int
    %f64_dtype = torch.prim.dtype %arg1 : !torch.vtensor<[],f64> -> !torch.int
    %ranks = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<optional<int>>
    %dtypes = torch.prim.ListConstruct %f32_dtype, %f64_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %3 : !torch.int
  } : !torch.vtensor<[1],unk>
  return
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$tensor_tensor_zero_rank_higher_category(
// CHECK:             {{.*}} = torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[1],f32>
func.func @promote_dtypes$tensor_tensor_zero_rank_higher_category(%arg0: !torch.vtensor<[1],si64>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.int) {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[1],si64>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[1],unk>
    torch.dtype.calculate.yield %1 : !torch.vtensor<[1],unk>
  } dtypes {
    %si64_dtype = torch.prim.dtype %arg0 : !torch.vtensor<[1],si64> -> !torch.int
    %f32_dtype = torch.prim.dtype %arg1 : !torch.vtensor<[],f32> -> !torch.int
    %ranks = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<optional<int>>
    %dtypes = torch.prim.ListConstruct %si64_dtype, %f32_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %3 : !torch.int
  } : !torch.vtensor<[1],unk>
  return
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$tensor_tensor_alpha_wider_no_contribution(
// CHECK:             {{.*}} = torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[1],f32>
func.func @promote_dtypes$tensor_tensor_alpha_wider_no_contribution(%arg0: !torch.vtensor<[1],f32>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.float) {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.float -> !torch.vtensor<[1],unk>
    torch.dtype.calculate.yield %1 : !torch.vtensor<[1],unk>
  } dtypes {
    %f32_dtype = torch.prim.dtype %arg0 : !torch.vtensor<[1],f32> -> !torch.int
    %alpha_as_tensor = torch.prim.NumToTensor.Scalar %arg2 : !torch.float -> !torch.tensor<[],f64>
    %f64_dtype = torch.prim.dtype %alpha_as_tensor : !torch.tensor<[],f64> -> !torch.int
    %ranks = torch.prim.ListConstruct %int1, %int1, %none : (!torch.int, !torch.int, !torch.none) -> !torch.list<optional<int>>
    %dtypes = torch.prim.ListConstruct %f32_dtype, %f32_dtype, %f64_dtype : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %3 : !torch.int
  } : !torch.vtensor<[1],unk>
  return
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$tensor_scalar_scalar_higher_category(
// CHECK:             {{.*}} = torch.aten.add.Scalar {{.*}} -> !torch.vtensor<[1],f32>
func.func @promote_dtypes$tensor_scalar_scalar_higher_category(%arg0: !torch.vtensor<[1],si64>, %arg1: !torch.float, %arg2: !torch.int) {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[1],si64>, !torch.float, !torch.int -> !torch.vtensor<[1],unk>
    torch.dtype.calculate.yield %1 : !torch.vtensor<[1],unk>
  } dtypes {
    %si64_dtype = torch.prim.dtype %arg0 : !torch.vtensor<[1],si64> -> !torch.int
    %ranks = torch.prim.ListConstruct %int1, %none : (!torch.int, !torch.none) -> !torch.list<optional<int>>
    %arg1_as_tensor = torch.prim.NumToTensor.Scalar %arg1 : !torch.float -> !torch.tensor<[],f64>
    %f64_dtype = torch.prim.dtype %arg1_as_tensor : !torch.tensor<[],f64> -> !torch.int
    %dtypes = torch.prim.ListConstruct %si64_dtype, %f64_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %5 : !torch.int
  } : !torch.vtensor<[1],unk>
  return
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$tensor_scalar_scalar_same_category_wider(
// CHECK:             {{.*}} = torch.aten.add.Scalar {{.*}} -> !torch.vtensor<[1],si32>
func.func @promote_dtypes$tensor_scalar_scalar_same_category_wider(%arg0: !torch.vtensor<[1],si32>, %arg1: !torch.int, %arg2: !torch.int) {
  %none = torch.constant.none
  %int3 = torch.constant.int 3
  %int1 = torch.constant.int 1
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[1],si32>, !torch.int, !torch.int -> !torch.vtensor<[1],unk>
    torch.dtype.calculate.yield %1 : !torch.vtensor<[1],unk>
  } dtypes {
    %si32_dtype = torch.prim.dtype %arg0 : !torch.vtensor<[1],si32> -> !torch.int
    %ranks = torch.prim.ListConstruct %int1, %none : (!torch.int, !torch.none) -> !torch.list<optional<int>>
    %arg1_as_tensor = torch.prim.NumToTensor.Scalar %arg1 : !torch.int -> !torch.tensor<[],si64>
    %si64_dtype = torch.prim.dtype %arg1_as_tensor : !torch.tensor<[],si64> -> !torch.int
    %dtypes = torch.prim.ListConstruct %si32_dtype, %si64_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %5 : !torch.int
  } : !torch.vtensor<[1],unk>
  return
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$scalar_scalar_different_category(
// CHECK:             {{.*}} = torch.aten.add {{.*}} -> !torch.float
func.func @promote_dtypes$scalar_scalar_different_category(%arg0: !torch.float, %arg1: !torch.int) -> !torch.number {
  %none = torch.constant.none
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add %arg0, %arg1 : !torch.float, !torch.int -> !torch.number
    torch.dtype.calculate.yield %1 : !torch.number
  } dtypes {
    %ranks = torch.prim.ListConstruct %none, %none : (!torch.none, !torch.none) -> !torch.list<optional<int>>
    %arg0_as_tensor = torch.prim.NumToTensor.Scalar %arg0 : !torch.float -> !torch.tensor<[],f64>
    %f64_dtype = torch.prim.dtype %arg0_as_tensor : !torch.tensor<[],f64> -> !torch.int
    %arg1_as_tensor = torch.prim.NumToTensor.Scalar %arg1 : !torch.int -> !torch.tensor<[],si64>
    %si64_dtype = torch.prim.dtype %arg1_as_tensor : !torch.tensor<[],si64> -> !torch.int
    %dtypes = torch.prim.ListConstruct %f64_dtype, %si64_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %7 : !torch.int
  } : !torch.number
  return %0 : !torch.number
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$scalar_scalar_same_category(
// CHECK:             {{.*}} = torch.aten.add {{.*}} -> !torch.int
func.func @promote_dtypes$scalar_scalar_same_category(%arg0: !torch.int, %arg1: !torch.int) -> !torch.number {
  %none = torch.constant.none
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add %arg0, %arg1 : !torch.int, !torch.int -> !torch.number
    torch.dtype.calculate.yield %1 : !torch.number
  } dtypes {
    %ranks = torch.prim.ListConstruct %none, %none : (!torch.none, !torch.none) -> !torch.list<optional<int>>
    %arg0_as_tensor = torch.prim.NumToTensor.Scalar %arg0 : !torch.int -> !torch.tensor<[],si64>
    %si64_dtype = torch.prim.dtype %arg0_as_tensor : !torch.tensor<[],si64> -> !torch.int
    %dtypes = torch.prim.ListConstruct %si64_dtype, %si64_dtype : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
    torch.dtype.calculate.yield.dtypes %7 : !torch.int
  } : !torch.number
  return %0 : !torch.number
}

// -----

// CHECK-LABEL:   func.func @promote_dtypes$list_tensors
// CHECK:             {{.*}} = torch.aten.cat {{.*}} : !torch.list<vtensor>, !torch.int -> !torch.vtensor<*,f32>
func.func @promote_dtypes$list_tensors(%arg0: !torch.vtensor<[1,8,320,384],f32>) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %int-3 = torch.constant.int -3
  %0 = torch.prim.ListConstruct %arg0 : (!torch.vtensor<[1,8,320,384],f32>) -> !torch.list<vtensor>
  %1 = torch.dtype.calculate {
    %2 = torch.aten.cat %0, %int-3 : !torch.list<vtensor>, !torch.int -> !torch.vtensor
    torch.dtype.calculate.yield %2 : !torch.vtensor
  } dtypes {
    %2 = torch.prim.ListConstruct  : () -> !torch.list<tuple<int, int>>
    torch.prim.Loop %int1, %true, init() {
    ^bb0(%arg1: !torch.int):
      %5 = torch.aten.__getitem__.t %0, %arg1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor
      %6 = torch.aten.dim %5 : !torch.vtensor -> !torch.int
      %7 = torch.prim.dtype %5 : !torch.vtensor -> !torch.int
      %8 = torch.prim.TupleConstruct %6, %7 : !torch.int, !torch.int -> !torch.tuple<int, int>
      %9 = torch.aten.append.t %2, %8 : !torch.list<tuple<int, int>>, !torch.tuple<int, int> -> !torch.list<tuple<int, int>>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %3 = torch.aten.__getitem__.t %2, %int0 : !torch.list<tuple<int, int>>, !torch.int -> !torch.tuple<int, int>
    %4:2 = torch.prim.TupleUnpack %3 : !torch.tuple<int, int> -> !torch.int, !torch.int
    torch.dtype.calculate.yield.dtypes %4#1 : !torch.int
  } : !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func @refine_dtype$invalid_dtype_for_scalar(
// CHECK:             {{.*}} = torch.aten.add {{.*}} -> !torch.number
func.func @refine_dtype$invalid_dtype_for_scalar(%arg0: !torch.int, %arg1: !torch.int) -> !torch.number {
  %none = torch.constant.none
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add %arg0, %arg1 : !torch.int, !torch.int -> !torch.number
    torch.dtype.calculate.yield %1 : !torch.number
  } dtypes {
    // dtype int for int32
    %int3 = torch.constant.int 3
    torch.dtype.calculate.yield.dtypes %int3 : !torch.int
  } : !torch.number
  return %0 : !torch.number
}

// -----

// CHECK-LABEL:   func.func @refine_dtype$no_simplification
// CHECK:             {{.*}} = torch.aten.add {{.*}} -> !torch.number
func.func @refine_dtype$no_simplification(%arg0: !torch.int, %arg1: !torch.int, %dtype: !torch.int) -> !torch.number {
  %none = torch.constant.none
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add %arg0, %arg1 : !torch.int, !torch.int -> !torch.number
    torch.dtype.calculate.yield %1 : !torch.number
  } dtypes {
    torch.dtype.calculate.yield.dtypes %dtype : !torch.int
  } : !torch.number
  return %0 : !torch.number
}

// -----

// If result type is already refined (even if wrong, as is the case here),
// don't make any changes to result type.
// TODO: This case should result in an error
// CHECK-LABEL:   func.func @refine_dtype$result_type_already_refined
// CHECK:             {{.*}} = torch.aten.add {{.*}} -> !torch.int
func.func @refine_dtype$result_type_already_refined(%arg0: !torch.float, %arg1: !torch.float) -> !torch.int {
  %none = torch.constant.none
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add %arg0, %arg1 : !torch.float, !torch.float -> !torch.int
    torch.dtype.calculate.yield %1 : !torch.int
  } dtypes {
    // dtype int for float64
    %int7 = torch.constant.int 7
    torch.dtype.calculate.yield.dtypes %int7 : !torch.int
  } : !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:   func.func @refine_dtype$derefine_result_type(
// CHECK:             {{.*}} = torch.aten.add {{.*}} -> !torch.int
// CHECK:           %[[ERASED:.*]] = torch.derefine {{.*}} : !torch.int to !torch.number
// CHECK:           return %[[ERASED]] : !torch.number
func.func @refine_dtype$derefine_result_type(%arg0: !torch.int, %arg1: !torch.int) -> !torch.number {
  %none = torch.constant.none
  %0 = torch.dtype.calculate {
    %1 = torch.aten.add %arg0, %arg1 : !torch.int, !torch.int -> !torch.number
    torch.dtype.calculate.yield %1 : !torch.number
  } dtypes {
    // dtype int for int64
    %int4 = torch.constant.int 4
    torch.dtype.calculate.yield.dtypes %int4 : !torch.int
  } : !torch.number
  return %0 : !torch.number
}

// CHECK-LABEL:   func.func @refine_dtype$complex_type(
// CHECK:             {{.*}} = torch.aten.fft_fft{{.*}}-> !torch.vtensor<*,complex<f32>>
func.func @refine_dtype$complex_type(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
    // dtype for ComplexFloat, a.k.a Complex64
    %int9 = torch.constant.int 9
    %none = torch.constant.none
    %int-1 = torch.constant.int -1
    %0 = torch.dtype.calculate {
      %2 = torch.aten.fft_fft %arg0, %none, %int-1, %none : !torch.vtensor<*,f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<*,complex<f32>>
      torch.dtype.calculate.yield %2 : !torch.vtensor<*,complex<f32>>
    } dtypes {
      torch.dtype.calculate.yield.dtypes %int9 : !torch.int
    } : !torch.vtensor<*,complex<f32>>
    %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<*,complex<f32>> to !torch.vtensor
    return %1 : !torch.vtensor
  }
