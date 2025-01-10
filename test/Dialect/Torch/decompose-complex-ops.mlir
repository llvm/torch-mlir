// RUN: torch-mlir-opt -torch-decompose-complex-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @matmul_no_decompose
// CHECK:           torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_no_decompose(%arg0: !torch.vtensor<[?,?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}


// -----

// CHECK-LABEL:   func.func @matmul_decompose_2d
// CHECK:           torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
func.func @matmul_decompose_2d(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @matmul_decompose_3d(
// CHECK:           torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_decompose_3d(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL: func.func @argmax_rank_1
// CHECK:         %[[I0:.*]] = torch.constant.int 0
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[VALUES:.*]], %[[INDICES:.*]] = torch.aten.max.dim %arg0, %[[I0]], %[[FALSE]] : !torch.vtensor<[20],si32>, !torch.int, !torch.bool -> !torch.vtensor<[],si32>, !torch.vtensor<[],si64>
// CHECK:         return %[[INDICES]] : !torch.vtensor<[],si64>
func.func @argmax_rank_1(%arg0: !torch.vtensor<[20],si32>) -> !torch.vtensor<[],si64> {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %7 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[20],si32>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    return %7 : !torch.vtensor<[],si64>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.type_as$basic(
// CHECK-SAME:                                %[[ARG_0:.*]]: !torch.tensor, %[[ARG_1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK-DAG:      %[[FALSE:.*]] = torch.constant.bool false
// CHECK-DAG:      %[[NONE:.*]] = torch.constant.none
// CHECK:          %[[DTYPE:.*]] = torch.prim.dtype %[[ARG_1]] : !torch.tensor -> !torch.int
// CHECK:          %[[VAR:.*]] = torch.aten.to.dtype %[[ARG_0]], %[[DTYPE]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.tensor, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
// CHECK:          return %[[VAR]] : !torch.tensor
func.func @torch.aten.type_as$basic(%arg0: !torch.tensor, %arg1: !torch.tensor) -> !torch.tensor {
  %0 = torch.aten.type_as %arg0, %arg1 : !torch.tensor, !torch.tensor -> !torch.tensor
  return %0 : !torch.tensor
}

// -----

// CHECK-LABEL:   func.func @torch.aten.type_as$fold(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.tensor<[?],f16>, %[[ARG_1:.*]]: !torch.tensor<[?,?],f16>) -> !torch.tensor<[?],f16> {
// CHECK:           return %[[ARG_0]] : !torch.tensor<[?],f16>
func.func @torch.aten.type_as$fold(%arg0: !torch.tensor<[?], f16>, %arg1: !torch.tensor<[?,?],f16>) -> !torch.tensor<[?],f16> {
  %0 = torch.aten.type_as %arg0, %arg1 : !torch.tensor<[?], f16>, !torch.tensor<[?,?],f16> -> !torch.tensor<[?], f16>
  return %0 : !torch.tensor<[?], f16>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.one_hot$fold(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[3],si64>, %arg1: !torch.int) -> !torch.vtensor<[3,?],si64> {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[INT4:.*]] = torch.constant.int 4
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[ARANGE:.*]] = torch.aten.arange.start_step %[[INT0]], %arg1, %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
// CHECK:           %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ARG_0]], %[[INT1]] : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3,1],si64>
// CHECK:           %[[EQ:.*]] = torch.aten.eq.Tensor %[[UNSQUEEZE]], %[[ARANGE]] : !torch.vtensor<[3,1],si64>, !torch.vtensor<[?],si64> -> !torch.vtensor<[3,?],i1>
// CHECK:           %[[RESULT:.*]] = torch.aten.to.dtype %[[EQ]], %[[INT4]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,?],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,?],si64>
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor<[3,?],si64>
func.func @torch.aten.one_hot$fold(%arg0: !torch.vtensor<[3],si64>, %arg1: !torch.int) -> !torch.vtensor<[3,?],si64> {
  %0 = torch.aten.one_hot %arg0, %arg1 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3,?],si64>
  return %0 : !torch.vtensor<[3,?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, %[[ARG_1:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                 %[[ARG_2:.*]]: !torch.vtensor<[1],si32>, %[[ARG_3:.*]]: !torch.vtensor<[1],si64>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[CONST1:.*]] = torch.constant.int 127
// CHECK:           %[[CONST2:.*]] = torch.constant.int -128
// CHECK:           %[[RESULT:.*]] = torch.aten.fake_quantize_per_tensor_affine.tensor_qparams %[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[CONST2]], %[[CONST1]] : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.vtensor<[1],si32>, %arg3: !torch.vtensor<[1],si64>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int127 = torch.constant.int 127
  %int-128 = torch.constant.int -128
  %0:2 = torch.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams %arg0, %arg1, %arg2, %arg3, %int-128, %int127 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],i1>
  return %0#0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fake_quantize_per_channel_affine_cachemask(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, %[[ARG_1:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                                 %[[ARG_2:.*]]: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[CONST0:.*]] = torch.constant.int 0
// CHECK:           %[[CONST1:.*]] = torch.constant.int 127
// CHECK:           %[[CONST2:.*]] = torch.constant.int -128
// CHECK:           %[[RESULT:.*]] = torch.aten.fake_quantize_per_channel_affine %[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[CONST0]], %[[CONST2]], %[[CONST1]] : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.fake_quantize_per_channel_affine_cachemask(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?],f32>, %arg2: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int0 = torch.constant.int 0
  %int127 = torch.constant.int 127
  %int-128 = torch.constant.int -128
  %0:2 = torch.aten.fake_quantize_per_channel_affine_cachemask %arg0, %arg1, %arg2, %int0, %int-128, %int127 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],i1>
  return %0#0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: test_einsum_inner_prod
func.func @test_einsum_inner_prod(%arg0: !torch.vtensor<[5],f64>, %arg1: !torch.vtensor<[5],f64>) -> !torch.vtensor<[],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK-DAG:  %[[INT5:.+]] = torch.constant.int 5
  // CHECK-DAG:  %[[INT1:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[INT0:.+]] = torch.constant.int 0
  // CHECK:  %[[LHS_LIST:.+]] = torch.prim.ListConstruct %[[INT0]]
  // CHECK:  %[[LHS_PERM:.+]] = torch.aten.permute %arg0, %[[LHS_LIST]]
  // CHECK:  %[[RHS_LIST:.+]] = torch.prim.ListConstruct %[[INT0]]
  // CHECK:  %[[RHS_PERM:.+]] = torch.aten.permute %arg1, %[[RHS_LIST]]
  // CHECK:  %[[LHS_SHP:.+]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]], %[[INT5]]
  // CHECK:  %[[LHS_VIEW:.+]] = torch.aten.view %[[LHS_PERM]], %[[LHS_SHP]]
  // CHECK:  %[[RHS_SHP:.+]] = torch.prim.ListConstruct %[[INT1]], %[[INT5]], %[[INT1]]
  // CHECK:  %[[RHS_VIEW:.+]] = torch.aten.view %[[RHS_PERM]], %[[RHS_SHP]]
  // CHECK:  %[[BMM:.+]] = torch.aten.bmm %[[LHS_VIEW]], %[[RHS_VIEW]]
  // CHECK:  %[[EMPTY:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK:  %[[OUT_VIEW:.+]] = torch.aten.view %[[BMM]], %[[EMPTY]]
  // CHECK:  %[[EMPTY:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK:  %[[OUT_PERM:.+]] = torch.aten.permute %[[OUT_VIEW]], %[[EMPTY]]
  // CHECK:  return %[[OUT_PERM]]
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[5],f64>, !torch.vtensor<[5],f64>) -> !torch.list<vtensor>
  %str = torch.constant.str "i,i"
  %none_0 = torch.constant.none
  %1 = torch.aten.einsum %str, %0, %none_0 : !torch.str, !torch.list<vtensor>, !torch.none -> !torch.vtensor<[],f64>
  return %1 : !torch.vtensor<[],f64>
}

// -----

// CHECK:   func.func @torch.aten.fmod_int(%[[ARG0:.+]]: !torch.vtensor<[?],si32>, %[[ARG1:.+]]: !torch.vtensor<[1],si32>) -> !torch.vtensor<[?],si32> {
// CHECK:     %[[FLOAT1:.+]] = torch.constant.float 1.000000e+00
// CHECK:     %[[V0:.+]] = torch.aten.div.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[?],si32>, !torch.vtensor<[1],si32> -> !torch.vtensor<[?],si32>
// CHECK:     %[[V1:.+]] = torch.aten.mul.Tensor %[[V0]], %[[ARG1]] : !torch.vtensor<[?],si32>, !torch.vtensor<[1],si32> -> !torch.vtensor<[?],si32>
// CHECK:     %[[V2:.+]] = torch.aten.sub.Tensor %[[ARG0]], %[[V1]], %[[FLOAT1]] : !torch.vtensor<[?],si32>, !torch.vtensor<[?],si32>, !torch.float -> !torch.vtensor<[?],si32>
// CHECK:     return %[[V2]] : !torch.vtensor<[?],si32>
func.func @torch.aten.fmod_int(%arg0: !torch.vtensor<[?],si32>, %arg1: !torch.vtensor<[1],si32>) -> !torch.vtensor<[?],si32> {
    %0 = torch.aten.fmod.Tensor %arg0, %arg1 : !torch.vtensor<[?],si32>, !torch.vtensor<[1],si32> -> !torch.vtensor<[?],si32>
    return %0 : !torch.vtensor<[?],si32>
}

// -----

// CHECK:   func.func @torch.aten.fmod_float(%[[ARG0:.+]]: !torch.vtensor<[?],f16>, %[[ARG1:.+]]: !torch.vtensor<[1],f16>) -> !torch.vtensor<[?],f16> {
// CHECK:     %[[FLOAT1:.+]] = torch.constant.float 1.000000e+00
// CHECK:     %[[V0:.+]] = torch.vtensor.literal(dense<-1> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:     %[[V1:.+]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:     %[[NONE:.+]] = torch.constant.none
// CHECK:     %[[FALSE:.+]] = torch.constant.bool false
// CHECK:     %[[INT5:.+]] = torch.constant.int 5
// CHECK:     %[[V2:.+]] = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
// CHECK:     %[[INT0:.+]] = torch.constant.int 0
// CHECK:     %[[V3:.+]] = torch.aten.div.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[?],f16>, !torch.vtensor<[1],f16> -> !torch.vtensor<[?],f16>
// CHECK:     %[[V4:.+]] = torch.aten.gt.Scalar %[[V3]], %[[INT0]] : !torch.vtensor<[?],f16>, !torch.int -> !torch.vtensor<[?],i1>
// CHECK:     %[[V5:.+]] = torch.aten.lt.Scalar %[[V3]], %[[INT0]] : !torch.vtensor<[?],f16>, !torch.int -> !torch.vtensor<[?],i1>
// CHECK:     %[[V6:.+]] = torch.aten.to.dtype %[[V2]], %[[INT5]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
// CHECK:     %[[V7:.+]] = torch.aten.to.dtype %[[V1]], %[[INT5]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
// CHECK:     %[[V8:.+]] = torch.aten.where.self %[[V4]], %[[V6]], %[[V7]] : !torch.vtensor<[?],i1>, !torch.vtensor<[],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[?],f16>
// CHECK:     %[[V9:.+]] = torch.aten.to.dtype %[[V0]], %[[INT5]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
// CHECK:     %[[V10:.+]] = torch.aten.where.self %[[V5]], %[[V9]], %[[V8]] : !torch.vtensor<[?],i1>, !torch.vtensor<[],f16>, !torch.vtensor<[?],f16> -> !torch.vtensor<[?],f16>
// CHECK:     %[[V11:.+]] = torch.aten.abs %[[V3]] : !torch.vtensor<[?],f16> -> !torch.vtensor<[?],f16>
// CHECK:     %[[V12:.+]] = torch.aten.floor %[[V11]] : !torch.vtensor<[?],f16> -> !torch.vtensor<[?],f16>
// CHECK:     %[[V13:.+]] = torch.aten.mul.Tensor %[[V10]], %[[V12]] : !torch.vtensor<[?],f16>, !torch.vtensor<[?],f16> -> !torch.vtensor<[?],f16>
// CHECK:     %[[V14:.+]] = torch.aten.mul.Tensor %[[V13]], %[[ARG1]] : !torch.vtensor<[?],f16>, !torch.vtensor<[1],f16> -> !torch.vtensor<[?],f16>
// CHECK:     %[[V15:.+]] = torch.aten.sub.Tensor %[[ARG0]], %[[V14]], %[[FLOAT1]] : !torch.vtensor<[?],f16>, !torch.vtensor<[?],f16>, !torch.float -> !torch.vtensor<[?],f16>
// CHECK:     return %[[V15]] : !torch.vtensor<[?],f16>
func.func @torch.aten.fmod_float(%arg0: !torch.vtensor<[?],f16>, %arg1: !torch.vtensor<[1],f16>) -> !torch.vtensor<[?],f16> {
    %0 = torch.aten.fmod.Tensor %arg0, %arg1 : !torch.vtensor<[?],f16>, !torch.vtensor<[1],f16> -> !torch.vtensor<[?],f16>
    return %0 : !torch.vtensor<[?],f16>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fft_rfft$2d_last_dim(
// CHECK-SAME:           %arg0: !torch.vtensor<[16,9],f32>) -> !torch.vtensor<[16,5],complex<f32>> {
// CHECK-DAG:         %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:         %[[INT5:.*]] = torch.constant.int 5
// CHECK-DAG:         %[[INT16:.*]] = torch.constant.int 16
// CHECK:             %[[VAR0:.*]] = torch.vtensor.literal(dense<{{.*}}> : tensor<9x10xf32>) : !torch.vtensor<[9,10],f32>
// CHECK:             %[[VAR1:.*]] = torch.aten.mm %arg0, %[[VAR0]] : !torch.vtensor<[16,9],f32>, !torch.vtensor<[9,10],f32> -> !torch.vtensor<[16,10],f32>
// CHECK:             %[[VAR2:.*]] = torch.prim.ListConstruct %[[INT16]], %[[INT5]], %[[INT2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR3:.*]] = torch.aten.view %[[VAR1]], %[[VAR2]] : !torch.vtensor<[16,10],f32>, !torch.list<int> -> !torch.vtensor<[16,5,2],f32>
// CHECK:             %[[VAR4:.*]] = torch.aten.view_as_complex %[[VAR3]] : !torch.vtensor<[16,5,2],f32> -> !torch.vtensor<[16,5],complex<f32>>
// CHECK:             return %[[VAR4]] : !torch.vtensor<[16,5],complex<f32>>
func.func @torch.aten.fft_rfft$2d_last_dim(%arg0: !torch.vtensor<[16,9],f32>) -> !torch.vtensor<[16,5],complex<f32>> {
  %int-1 = torch.constant.int -1
  %none = torch.constant.none
  %out = torch.aten.fft_rfft %arg0, %none, %int-1, %none : !torch.vtensor<[16,9],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[16,5],complex<f32>>
  return %out : !torch.vtensor<[16,5],complex<f32>>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fft_rfft$2d_first_dim(
// CHECK-SAME:           %arg0: !torch.vtensor<[36,23],f32>) -> !torch.vtensor<[19,23],complex<f32>> {
// CHECK-DAG:         %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:         %[[INT19:.*]] = torch.constant.int 19
// CHECK-DAG:         %[[INT23:.*]] = torch.constant.int 23
// CHECK-DAG:             %[[VAR0:.*]] = torch.vtensor.literal(dense<{{.*}}> : tensor<36x38xf32>) : !torch.vtensor<[36,38],f32>
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:             %[[VAR1:.*]] = torch.aten.transpose.int %arg0, %[[INT0]], %[[INT1]] : !torch.vtensor<[36,23],f32>, !torch.int, !torch.int -> !torch.vtensor<[23,36],f32>
// CHECK:             %[[VAR2:.*]] = torch.aten.mm %[[VAR1]], %[[VAR0]] : !torch.vtensor<[23,36],f32>, !torch.vtensor<[36,38],f32> -> !torch.vtensor<[23,38],f32>
// CHECK:             %[[VAR3:.*]] = torch.prim.ListConstruct %[[INT23]], %[[INT19]], %[[INT2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR4:.*]] = torch.aten.view %[[VAR2]], %[[VAR3]] : !torch.vtensor<[23,38],f32>, !torch.list<int> -> !torch.vtensor<[23,19,2],f32>
// CHECK:             %[[VAR5:.*]] = torch.aten.view_as_complex %[[VAR4]] : !torch.vtensor<[23,19,2],f32> -> !torch.vtensor<[23,19],complex<f32>>
// CHECK:             %[[VAR6:.*]] = torch.aten.transpose.int %[[VAR5]], %[[INT0]], %[[INT1]] : !torch.vtensor<[23,19],complex<f32>>, !torch.int, !torch.int -> !torch.vtensor<[19,23],complex<f32>>
// CHECK:             return %[[VAR6]] : !torch.vtensor<[19,23],complex<f32>>
func.func @torch.aten.fft_rfft$2d_first_dim(%arg0: !torch.vtensor<[36,23],f32>) -> !torch.vtensor<[19,23],complex<f32>> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %out = torch.aten.fft_rfft %arg0, %none, %int0, %none : !torch.vtensor<[36,23],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[19,23],complex<f32>>
  return %out : !torch.vtensor<[19,23],complex<f32>>
}

// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_1D(
// CHECK-SAME:           %arg0: !torch.vtensor<[40],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,37],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT4:.*]] = torch.constant.int 4
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK-DAG:         %[[INT40:.*]] = torch.constant.int 40
// CHECK-DAG:         %[[INT37:.*]] = torch.constant.int 37
// CHECK-DAG:         %[[INT3:.*]] = torch.constant.int 3
// CHECK:             %[[VAR0:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT37]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR1:.*]] = torch.aten.empty.memory_format %[[VAR0]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,37],complex<f32>>
// CHECK:             %[[VAR2:.*]] = torch.prim.Loop %[[INT37]], %[[TRUE]], init(%[[VAR1]]) {
// CHECK:             ^bb0(%arg2: !torch.int, %arg3: !torch.vtensor<[3,37],complex<f32>>):
// CHECK:               %[[VAR3:.*]] = torch.aten.add.int %arg2, %[[INT4]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR4:.*]] = torch.prim.min.int %[[VAR3]], %[[INT40]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR5:.*]] = torch.aten.sub.int %[[VAR4]], %arg2 : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR6:.*]] = torch.aten.sub.int %[[INT4]], %[[VAR5]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR7:.*]] = torch.aten.add.int %arg2, %[[VAR5]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR8:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %arg2, %[[VAR7]], %[[INT1]] : !torch.vtensor<[40],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:               %[[VAR9:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR10:.*]] = torch.aten.constant_pad_nd %[[VAR8]], %[[VAR9]], %float0.000000e00 : !torch.vtensor<[?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[?],f32>
// CHECK:               %[[VAR11:.*]] = torch.tensor_static_info_cast %[[VAR10]] : !torch.vtensor<[?],f32> to !torch.vtensor<[4],f32>
// CHECK:               %[[VAR12:.*]] = torch.aten.mul.Tensor %[[VAR11]], %arg1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
// CHECK:               %[[VAR13:.*]] = torch.aten.fft_fft %[[VAR12]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[4],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[3],complex<f32>>
// CHECK:               %[[VAR14:.*]] = torch.aten.unsqueeze %[[VAR13]], %[[INTM1]] : !torch.vtensor<[3],complex<f32>>, !torch.int -> !torch.vtensor<[3,1],complex<f32>>
// CHECK:               %[[VAR15:.*]] = torch.aten.slice_scatter %arg3, %[[VAR14]], %[[INTM1]], %arg2, %[[NONE]], %[[INT1]] : !torch.vtensor<[3,37],complex<f32>>, !torch.vtensor<[3,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[3,37],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR15]] : !torch.vtensor<[3,37],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[3,37],complex<f32>>) -> !torch.vtensor<[3,37],complex<f32>>
// CHECK:             return %[[VAR2]] : !torch.vtensor<[3,37],complex<f32>>
func.func @torch.aten.stft.center_1D(%arg0: !torch.vtensor<[40],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,37],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 4
  %hoplen = torch.constant.int 1
  %winlen = torch.constant.int 4
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %winlen, %arg1, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[40],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[4],f32>, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[3,37],complex<f32>>
  return %0 : !torch.vtensor<[3,37],complex<f32>>
}


// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_1D_unk_sig_len(
// CHECK-SAME:           %arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[10],f32>) -> !torch.vtensor<[6,?],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT6:.*]] = torch.constant.int 6
// CHECK-DAG:         %[[INT10:.*]] = torch.constant.int 10
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK:             %[[VAR0:.*]] = torch.aten.size.int %arg0, %[[INTM1]] : !torch.vtensor<[?],f32>, !torch.int -> !torch.int
// CHECK:             %[[VAR1:.*]] = torch.aten.sub.int %[[VAR0]], %[[INT10]] : !torch.int, !torch.int -> !torch.int
// CHECK:             %[[VAR2:.*]] = torch.aten.floordiv.int %[[VAR1]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
// CHECK:             %[[VAR3:.*]] = torch.aten.add.int %[[INT1]], %[[VAR2]] : !torch.int, !torch.int -> !torch.int
// CHECK:             %[[VAR4:.*]] = torch.prim.ListConstruct %[[INT6]], %[[VAR3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR5:.*]] = torch.aten.empty.memory_format %[[VAR4]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[6,?],complex<f32>>
// CHECK:             %[[VAR6:.*]] = torch.prim.Loop %[[VAR3]], %[[TRUE]], init(%[[VAR5]]) {
// CHECK:             ^bb0(%arg2: !torch.int, %arg3: !torch.vtensor<[6,?],complex<f32>>):
// CHECK:               %[[VAR7:.*]] = torch.aten.add.int %arg2, %[[INT10]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR8:.*]] = torch.prim.min.int %[[VAR7]], %[[VAR0]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR9:.*]] = torch.aten.sub.int %[[VAR8]], %arg2 : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR10:.*]] = torch.aten.sub.int %[[INT10]], %[[VAR9]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR11:.*]] = torch.aten.add.int %arg2, %[[VAR9]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR12:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %arg2, %[[VAR11]], %[[INT1]] : !torch.vtensor<[?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
// CHECK:               %[[VAR13:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR14:.*]] = torch.aten.constant_pad_nd %[[VAR12]], %[[VAR13]], %float0.000000e00 : !torch.vtensor<[?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[?],f32>
// CHECK:               %[[VAR15:.*]] = torch.tensor_static_info_cast %[[VAR14]] : !torch.vtensor<[?],f32> to !torch.vtensor<[10],f32>
// CHECK:               %[[VAR16:.*]] = torch.aten.mul.Tensor %[[VAR15]], %arg1 : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
// CHECK:               %[[VAR17:.*]] = torch.aten.fft_fft %[[VAR16]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[6],complex<f32>>
// CHECK:               %[[VAR18:.*]] = torch.aten.unsqueeze %[[VAR17]], %[[INTM1]] : !torch.vtensor<[6],complex<f32>>, !torch.int -> !torch.vtensor<[6,1],complex<f32>>
// CHECK:               %[[VAR19:.*]] = torch.aten.slice_scatter %arg3, %[[VAR18]], %[[INTM1]], %arg2, %[[NONE]], %[[INT1]] : !torch.vtensor<[6,?],complex<f32>>, !torch.vtensor<[6,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[6,?],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR19]] : !torch.vtensor<[6,?],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[6,?],complex<f32>>) -> !torch.vtensor<[6,?],complex<f32>>
// CHECK:             return %[[VAR6]] : !torch.vtensor<[6,?],complex<f32>>
func.func @torch.aten.stft.center_1D_unk_sig_len(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[10],f32>) -> !torch.vtensor<[6,?],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 10
  %hoplen = torch.constant.int 1
  %winlen = torch.constant.int 10
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %winlen, %arg1, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[?],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[10],f32>, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[6,?],complex<f32>>
  return %0 : !torch.vtensor<[6,?],complex<f32>>
}


// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_2D(
// CHECK-SAME:           %arg0: !torch.vtensor<[4,46],f32>, %arg1: !torch.vtensor<[7],f32>) -> !torch.vtensor<[4,4,40],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT4:.*]] = torch.constant.int 4
// CHECK-DAG:         %[[INT7:.*]] = torch.constant.int 7
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK-DAG:         %[[INT46:.*]] = torch.constant.int 46
// CHECK-DAG:         %[[INT40:.*]] = torch.constant.int 40
// CHECK:             %[[VAR0:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR1:.*]] = torch.aten.view %arg1, %[[VAR0]] : !torch.vtensor<[7],f32>, !torch.list<int> -> !torch.vtensor<[1,7],f32>
// CHECK:             %[[VAR2:.*]] = torch.prim.ListConstruct %[[INT4]], %[[INT4]], %[[INT40]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR3:.*]] = torch.aten.empty.memory_format %[[VAR2]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4,4,40],complex<f32>>
// CHECK:             %[[VAR4:.*]] = torch.prim.Loop %[[INT40]], %[[TRUE]], init(%[[VAR3]]) {
// CHECK:             ^bb0(%arg2: !torch.int, %arg3: !torch.vtensor<[4,4,40],complex<f32>>):
// CHECK:               %[[VAR5:.*]] = torch.aten.add.int %arg2, %[[INT7]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR6:.*]] = torch.prim.min.int %[[VAR5]], %[[INT46]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR7:.*]] = torch.aten.sub.int %[[VAR6]], %arg2 : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR8:.*]] = torch.aten.sub.int %[[INT7]], %[[VAR7]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR9:.*]] = torch.aten.add.int %arg2, %[[VAR7]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR10:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %arg2, %[[VAR9]], %[[INT1]] : !torch.vtensor<[4,46],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?],f32>
// CHECK:               %[[VAR11:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR12:.*]] = torch.aten.constant_pad_nd %[[VAR10]], %[[VAR11]], %float0.000000e00 : !torch.vtensor<[4,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[4,?],f32>
// CHECK:               %[[VAR13:.*]] = torch.tensor_static_info_cast %[[VAR12]] : !torch.vtensor<[4,?],f32> to !torch.vtensor<[4,7],f32>
// CHECK:               %[[VAR14:.*]] = torch.aten.mul.Tensor %[[VAR13]], %[[VAR1]] : !torch.vtensor<[4,7],f32>, !torch.vtensor<[1,7],f32> -> !torch.vtensor<[4,7],f32>
// CHECK:               %[[VAR15:.*]] = torch.aten.fft_fft %[[VAR14]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[4,7],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[4,4],complex<f32>>
// CHECK:               %[[VAR16:.*]] = torch.aten.unsqueeze %[[VAR15]], %[[INTM1]] : !torch.vtensor<[4,4],complex<f32>>, !torch.int -> !torch.vtensor<[4,4,1],complex<f32>>
// CHECK:               %[[VAR17:.*]] = torch.aten.slice_scatter %arg3, %[[VAR16]], %[[INTM1]], %arg2, %[[NONE]], %[[INT1]] : !torch.vtensor<[4,4,40],complex<f32>>, !torch.vtensor<[4,4,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[4,4,40],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR17]] : !torch.vtensor<[4,4,40],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[4,4,40],complex<f32>>) -> !torch.vtensor<[4,4,40],complex<f32>>
// CHECK:             return %[[VAR4]] : !torch.vtensor<[4,4,40],complex<f32>>
func.func @torch.aten.stft.center_2D(%arg0: !torch.vtensor<[4,46],f32>, %arg1: !torch.vtensor<[7],f32>) -> !torch.vtensor<[4,4,40],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 7
  %hoplen = torch.constant.int 1
  %winlen = torch.constant.int 7
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %winlen, %arg1, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[4,46],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[7],f32>, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[4,4,40],complex<f32>>
  return %0 : !torch.vtensor<[4,4,40],complex<f32>>
}


// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_2D_win_unk_size(
// CHECK-SAME:           %arg0: !torch.vtensor<[3,38],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[3,4,32],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT3:.*]] = torch.constant.int 3
// CHECK-DAG:         %[[INT4:.*]] = torch.constant.int 4
// CHECK-DAG:         %[[INT32:.*]] = torch.constant.int 32
// CHECK-DAG:         %[[INT38:.*]] = torch.constant.int 38
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[INT6:.*]] = torch.constant.int 6
// CHECK-DAG:         %[[INT7:.*]] = torch.constant.int 7
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:             %[[VAR0:.*]] = torch.aten.size.int %arg1, %[[INT0]] : !torch.vtensor<[?],f32>, !torch.int -> !torch.int
// CHECK:             %[[VAR1:.*]] = torch.aten.eq.int %[[VAR0]], %[[INT6]] : !torch.int, !torch.int -> !torch.bool
// CHECK:             torch.runtime.assert %[[VAR1]], "window size should be equal to win_length"
// CHECK:             %[[VAR2:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR3:.*]] = torch.aten.constant_pad_nd %arg1, %[[VAR2]], %float0.000000e00 : !torch.vtensor<[?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[7],f32>
// CHECK:             %[[VAR4:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR5:.*]] = torch.aten.view %[[VAR3]], %[[VAR4]] : !torch.vtensor<[7],f32>, !torch.list<int> -> !torch.vtensor<[1,7],f32>
// CHECK:             %[[VAR6:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT4]], %[[INT32]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR7:.*]] = torch.aten.empty.memory_format %[[VAR6]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,4,32],complex<f32>>
// CHECK:             %[[VAR8:.*]] = torch.prim.Loop %[[INT32]], %[[TRUE]], init(%[[VAR7]]) {
// CHECK:             ^bb0(%arg2: !torch.int, %arg3: !torch.vtensor<[3,4,32],complex<f32>>):
// CHECK:               %[[VAR9:.*]] = torch.aten.add.int %arg2, %[[INT7]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR10:.*]] = torch.prim.min.int %[[VAR9]], %[[INT38]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR11:.*]] = torch.aten.sub.int %[[VAR10]], %arg2 : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR12:.*]] = torch.aten.sub.int %[[INT7]], %[[VAR11]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR13:.*]] = torch.aten.add.int %arg2, %[[VAR11]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR14:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %arg2, %[[VAR13]], %[[INT1]] : !torch.vtensor<[3,38],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?],f32>
// CHECK:               %[[VAR15:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR12]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR16:.*]] = torch.aten.constant_pad_nd %[[VAR14]], %[[VAR15]], %float0.000000e00 : !torch.vtensor<[3,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
// CHECK:               %[[VAR17:.*]] = torch.tensor_static_info_cast %[[VAR16]] : !torch.vtensor<[3,?],f32> to !torch.vtensor<[3,7],f32>
// CHECK:               %[[VAR18:.*]] = torch.aten.mul.Tensor %[[VAR17]], %[[VAR5]] : !torch.vtensor<[3,7],f32>, !torch.vtensor<[1,7],f32> -> !torch.vtensor<[3,7],f32>
// CHECK:               %[[VAR19:.*]] = torch.aten.fft_fft %[[VAR18]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[3,7],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[3,4],complex<f32>>
// CHECK:               %[[VAR20:.*]] = torch.aten.unsqueeze %[[VAR19]], %[[INTM1]] : !torch.vtensor<[3,4],complex<f32>>, !torch.int -> !torch.vtensor<[3,4,1],complex<f32>>
// CHECK:               %[[VAR21:.*]] = torch.aten.slice_scatter %arg3, %[[VAR20]], %[[INTM1]], %arg2, %[[NONE]], %[[INT1]] : !torch.vtensor<[3,4,32],complex<f32>>, !torch.vtensor<[3,4,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[3,4,32],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR21]] : !torch.vtensor<[3,4,32],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[3,4,32],complex<f32>>) -> !torch.vtensor<[3,4,32],complex<f32>>
// CHECK:             return %[[VAR8]] : !torch.vtensor<[3,4,32],complex<f32>>
func.func @torch.aten.stft.center_2D_win_unk_size(%arg0: !torch.vtensor<[3,38],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[3,4,32],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 7
  %hoplen = torch.constant.int 1
  %winlen = torch.constant.int 6
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %winlen, %arg1, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[3,38],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[?],f32>, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[3,4,32],complex<f32>>
  return %0 : !torch.vtensor<[3,4,32],complex<f32>>
}


// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_2D_no_window(
// CHECK-SAME:           %arg0: !torch.vtensor<[2,32],f32>) -> !torch.vtensor<[2,5,25],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT8:.*]] = torch.constant.int 8
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK-DAG:         %[[INT32:.*]] = torch.constant.int 32
// CHECK-DAG:         %[[INT25:.*]] = torch.constant.int 25
// CHECK-DAG:         %[[INT5:.*]] = torch.constant.int 5
// CHECK-DAG:         %[[INT2:.*]] = torch.constant.int 2
// CHECK:             %[[VAR0:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT5]], %[[INT25]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR1:.*]] = torch.aten.empty.memory_format %[[VAR0]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,5,25],complex<f32>>
// CHECK:             %[[VAR2:.*]] = torch.prim.Loop %[[INT25]], %[[TRUE]], init(%[[VAR1]]) {
// CHECK:             ^bb0(%arg1: !torch.int, %arg2: !torch.vtensor<[2,5,25],complex<f32>>):
// CHECK:               %[[VAR3:.*]] = torch.aten.add.int %arg1, %[[INT8]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR4:.*]] = torch.prim.min.int %[[VAR3]], %[[INT32]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR5:.*]] = torch.aten.sub.int %[[VAR4]], %arg1 : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR6:.*]] = torch.aten.sub.int %[[INT8]], %[[VAR5]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR7:.*]] = torch.aten.add.int %arg1, %[[VAR5]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR8:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %arg1, %[[VAR7]], %[[INT1]] : !torch.vtensor<[2,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,?],f32>
// CHECK:               %[[VAR9:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR6]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR10:.*]] = torch.aten.constant_pad_nd %[[VAR8]], %[[VAR9]], %float0.000000e00 : !torch.vtensor<[2,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[2,?],f32>
// CHECK:               %[[VAR11:.*]] = torch.tensor_static_info_cast %[[VAR10]] : !torch.vtensor<[2,?],f32> to !torch.vtensor<[2,8],f32>
// CHECK:               %[[VAR12:.*]] = torch.aten.fft_fft %[[VAR11]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[2,8],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[2,5],complex<f32>>
// CHECK:               %[[VAR13:.*]] = torch.aten.unsqueeze %[[VAR12]], %[[INTM1]] : !torch.vtensor<[2,5],complex<f32>>, !torch.int -> !torch.vtensor<[2,5,1],complex<f32>>
// CHECK:               %[[VAR14:.*]] = torch.aten.slice_scatter %arg2, %[[VAR13]], %[[INTM1]], %arg1, %[[NONE]], %[[INT1]] : !torch.vtensor<[2,5,25],complex<f32>>, !torch.vtensor<[2,5,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[2,5,25],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR14]] : !torch.vtensor<[2,5,25],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[2,5,25],complex<f32>>) -> !torch.vtensor<[2,5,25],complex<f32>>
// CHECK:             return %[[VAR2]] : !torch.vtensor<[2,5,25],complex<f32>>
func.func @torch.aten.stft.center_2D_no_window(%arg0: !torch.vtensor<[2,32],f32>) -> !torch.vtensor<[2,5,25],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 8
  %hoplen = torch.constant.int 1
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %cstnone = torch.constant.none
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %cstnone, %cstnone, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[2,32],f32>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[2,5,25],complex<f32>>
  return %0 : !torch.vtensor<[2,5,25],complex<f32>>
}


// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_2D_hop_length_2(
// CHECK-SAME:           %arg0: !torch.vtensor<[2,61],f32>, %arg1: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,5,27],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT5:.*]] = torch.constant.int 5
// CHECK-DAG:         %[[INT8:.*]] = torch.constant.int 8
// CHECK-DAG:         %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK-DAG:         %[[INT61:.*]] = torch.constant.int 61
// CHECK-DAG:         %[[INT27:.*]] = torch.constant.int 27
// CHECK:             %[[VAR0:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR1:.*]] = torch.aten.view %arg1, %[[VAR0]] : !torch.vtensor<[8],f32>, !torch.list<int> -> !torch.vtensor<[1,8],f32>
// CHECK:             %[[VAR2:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT5]], %[[INT27]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR3:.*]] = torch.aten.empty.memory_format %[[VAR2]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,5,27],complex<f32>>
// CHECK:             %[[VAR4:.*]] = torch.prim.Loop %[[INT27]], %[[TRUE]], init(%[[VAR3]]) {
// CHECK:             ^bb0(%arg2: !torch.int, %arg3: !torch.vtensor<[2,5,27],complex<f32>>):
// CHECK:               %[[VAR5:.*]] = torch.aten.mul.int %arg2, %[[INT2]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR6:.*]] = torch.aten.add.int %[[VAR5]], %[[INT8]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR7:.*]] = torch.prim.min.int %[[VAR6]], %[[INT61]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR8:.*]] = torch.aten.sub.int %[[VAR7]], %[[VAR5]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR9:.*]] = torch.aten.sub.int %[[INT8]], %[[VAR8]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR10:.*]] = torch.aten.add.int %[[VAR5]], %[[VAR8]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR11:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %[[VAR5]], %[[VAR10]], %[[INT1]] : !torch.vtensor<[2,61],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,?],f32>
// CHECK:               %[[VAR12:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR9]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR13:.*]] = torch.aten.constant_pad_nd %[[VAR11]], %[[VAR12]], %float0.000000e00 : !torch.vtensor<[2,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[2,?],f32>
// CHECK:               %[[VAR14:.*]] = torch.tensor_static_info_cast %[[VAR13]] : !torch.vtensor<[2,?],f32> to !torch.vtensor<[2,8],f32>
// CHECK:               %[[VAR15:.*]] = torch.aten.mul.Tensor %[[VAR14]], %[[VAR1]] : !torch.vtensor<[2,8],f32>, !torch.vtensor<[1,8],f32> -> !torch.vtensor<[2,8],f32>
// CHECK:               %[[VAR16:.*]] = torch.aten.fft_fft %[[VAR15]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[2,8],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[2,5],complex<f32>>
// CHECK:               %[[VAR17:.*]] = torch.aten.unsqueeze %[[VAR16]], %[[INTM1]] : !torch.vtensor<[2,5],complex<f32>>, !torch.int -> !torch.vtensor<[2,5,1],complex<f32>>
// CHECK:               %[[VAR18:.*]] = torch.aten.slice_scatter %arg3, %[[VAR17]], %[[INTM1]], %arg2, %[[NONE]], %[[INT1]] : !torch.vtensor<[2,5,27],complex<f32>>, !torch.vtensor<[2,5,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[2,5,27],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR18]] : !torch.vtensor<[2,5,27],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[2,5,27],complex<f32>>) -> !torch.vtensor<[2,5,27],complex<f32>>
// CHECK:             return %[[VAR4]] : !torch.vtensor<[2,5,27],complex<f32>>
func.func @torch.aten.stft.center_2D_hop_length_2(%arg0: !torch.vtensor<[2,61],f32>, %arg1: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,5,27],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 8
  %hoplen = torch.constant.int 2
  %winlen = torch.constant.int 8
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %winlen, %arg1, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[2,61],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[8],f32>, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[2,5,27],complex<f32>>
  return %0 : !torch.vtensor<[2,5,27],complex<f32>>
}


// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_2D_window_pad_left(
// CHECK-SAME:           %arg0: !torch.vtensor<[2,68],f32>, %arg1: !torch.vtensor<[6],f32>) -> !torch.vtensor<[2,4,31],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT4:.*]] = torch.constant.int 4
// CHECK-DAG:         %[[INT7:.*]] = torch.constant.int 7
// CHECK-DAG:         %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK-DAG:         %[[INT68:.*]] = torch.constant.int 68
// CHECK-DAG:         %[[INT31:.*]] = torch.constant.int 31
// CHECK:             %[[VAR0:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR1:.*]] = torch.aten.constant_pad_nd %arg1, %[[VAR0]], %float0.000000e00 : !torch.vtensor<[6],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[7],f32>
// CHECK:             %[[VAR2:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR3:.*]] = torch.aten.view %[[VAR1]], %[[VAR2]] : !torch.vtensor<[7],f32>, !torch.list<int> -> !torch.vtensor<[1,7],f32>
// CHECK:             %[[VAR4:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT4]], %[[INT31]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR5:.*]] = torch.aten.empty.memory_format %[[VAR4]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,4,31],complex<f32>>
// CHECK:             %[[VAR6:.*]] = torch.prim.Loop %[[INT31]], %[[TRUE]], init(%[[VAR5]]) {
// CHECK:             ^bb0(%arg2: !torch.int, %arg3: !torch.vtensor<[2,4,31],complex<f32>>):
// CHECK:               %[[VAR7:.*]] = torch.aten.mul.int %arg2, %[[INT2]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR8:.*]] = torch.aten.add.int %[[VAR7]], %[[INT7]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR9:.*]] = torch.prim.min.int %[[VAR8]], %[[INT68]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR10:.*]] = torch.aten.sub.int %[[VAR9]], %[[VAR7]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR11:.*]] = torch.aten.sub.int %[[INT7]], %[[VAR10]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR12:.*]] = torch.aten.add.int %[[VAR7]], %[[VAR10]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR13:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %[[VAR7]], %[[VAR12]], %[[INT1]] : !torch.vtensor<[2,68],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,?],f32>
// CHECK:               %[[VAR14:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR11]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR15:.*]] = torch.aten.constant_pad_nd %[[VAR13]], %[[VAR14]], %float0.000000e00 : !torch.vtensor<[2,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[2,?],f32>
// CHECK:               %[[VAR16:.*]] = torch.tensor_static_info_cast %[[VAR15]] : !torch.vtensor<[2,?],f32> to !torch.vtensor<[2,7],f32>
// CHECK:               %[[VAR17:.*]] = torch.aten.mul.Tensor %[[VAR16]], %[[VAR3]] : !torch.vtensor<[2,7],f32>, !torch.vtensor<[1,7],f32> -> !torch.vtensor<[2,7],f32>
// CHECK:               %[[VAR18:.*]] = torch.aten.fft_fft %[[VAR17]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[2,7],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[2,4],complex<f32>>
// CHECK:               %[[VAR19:.*]] = torch.aten.unsqueeze %[[VAR18]], %[[INTM1]] : !torch.vtensor<[2,4],complex<f32>>, !torch.int -> !torch.vtensor<[2,4,1],complex<f32>>
// CHECK:               %[[VAR20:.*]] = torch.aten.slice_scatter %arg3, %[[VAR19]], %[[INTM1]], %arg2, %[[NONE]], %[[INT1]] : !torch.vtensor<[2,4,31],complex<f32>>, !torch.vtensor<[2,4,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[2,4,31],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR20]] : !torch.vtensor<[2,4,31],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[2,4,31],complex<f32>>) -> !torch.vtensor<[2,4,31],complex<f32>>
// CHECK:             return %[[VAR6]] : !torch.vtensor<[2,4,31],complex<f32>>
func.func @torch.aten.stft.center_2D_window_pad_left(%arg0: !torch.vtensor<[2,68],f32>, %arg1: !torch.vtensor<[6],f32>) -> !torch.vtensor<[2,4,31],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 7
  %hoplen = torch.constant.int 2
  %winlen = torch.constant.int 6
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %winlen, %arg1, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[2,68],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[6],f32>, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[2,4,31],complex<f32>>
  return %0 : !torch.vtensor<[2,4,31],complex<f32>>
}


// -----


// CHECK-LABEL:   func.func @torch.aten.stft.center_2D_hop_length_3_window_pad_both(
// CHECK-SAME:           %arg0: !torch.vtensor<[3,90],f32>, %arg1: !torch.vtensor<[8],f32>) -> !torch.vtensor<[3,6,27],complex<f32>> {
// CHECK-DAG:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:         %[[INT9:.*]] = torch.constant.int 9
// CHECK-DAG:         %[[INT6:.*]] = torch.constant.int 6
// CHECK-DAG:         %[[INT10:.*]] = torch.constant.int 10
// CHECK-DAG:         %[[INT3:.*]] = torch.constant.int 3
// CHECK-DAG:         %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:         %[[NONE:.*]] = torch.constant.none
// CHECK-DAG:         %float0.000000e00 = torch.constant.float 0.000000e+00
// CHECK-DAG:         %[[INT90:.*]] = torch.constant.int 90
// CHECK-DAG:         %[[INT27:.*]] = torch.constant.int 27
// CHECK:             %[[VAR0:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR1:.*]] = torch.aten.constant_pad_nd %arg1, %[[VAR0]], %float0.000000e00 : !torch.vtensor<[8],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[10],f32>
// CHECK:             %[[VAR2:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT10]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR3:.*]] = torch.aten.view %[[VAR1]], %[[VAR2]] : !torch.vtensor<[10],f32>, !torch.list<int> -> !torch.vtensor<[1,10],f32>
// CHECK:             %[[VAR4:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT6]], %[[INT27]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAR5:.*]] = torch.aten.empty.memory_format %[[VAR4]], %[[INT9]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,6,27],complex<f32>>
// CHECK:             %[[VAR6:.*]] = torch.prim.Loop %[[INT27]], %[[TRUE]], init(%[[VAR5]]) {
// CHECK:             ^bb0(%arg2: !torch.int, %arg3: !torch.vtensor<[3,6,27],complex<f32>>):
// CHECK:               %[[VAR7:.*]] = torch.aten.mul.int %arg2, %[[INT3]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR8:.*]] = torch.aten.add.int %[[VAR7]], %[[INT10]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR9:.*]] = torch.prim.min.int %[[VAR8]], %[[INT90]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR10:.*]] = torch.aten.sub.int %[[VAR9]], %[[VAR7]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR11:.*]] = torch.aten.sub.int %[[INT10]], %[[VAR10]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR12:.*]] = torch.aten.add.int %[[VAR7]], %[[VAR10]] : !torch.int, !torch.int -> !torch.int
// CHECK:               %[[VAR13:.*]] = torch.aten.slice.Tensor %arg0, %[[INTM1]], %[[VAR7]], %[[VAR12]], %[[INT1]] : !torch.vtensor<[3,90],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?],f32>
// CHECK:               %[[VAR14:.*]] = torch.prim.ListConstruct %[[INT0]], %[[VAR11]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:               %[[VAR15:.*]] = torch.aten.constant_pad_nd %[[VAR13]], %[[VAR14]], %float0.000000e00 : !torch.vtensor<[3,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
// CHECK:               %[[VAR16:.*]] = torch.tensor_static_info_cast %[[VAR15]] : !torch.vtensor<[3,?],f32> to !torch.vtensor<[3,10],f32>
// CHECK:               %[[VAR17:.*]] = torch.aten.mul.Tensor %[[VAR16]], %[[VAR3]] : !torch.vtensor<[3,10],f32>, !torch.vtensor<[1,10],f32> -> !torch.vtensor<[3,10],f32>
// CHECK:               %[[VAR18:.*]] = torch.aten.fft_fft %[[VAR17]], %[[NONE]], %[[INTM1]], %[[NONE]] : !torch.vtensor<[3,10],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[3,6],complex<f32>>
// CHECK:               %[[VAR19:.*]] = torch.aten.unsqueeze %[[VAR18]], %[[INTM1]] : !torch.vtensor<[3,6],complex<f32>>, !torch.int -> !torch.vtensor<[3,6,1],complex<f32>>
// CHECK:               %[[VAR20:.*]] = torch.aten.slice_scatter %arg3, %[[VAR19]], %[[INTM1]], %arg2, %[[NONE]], %[[INT1]] : !torch.vtensor<[3,6,27],complex<f32>>, !torch.vtensor<[3,6,1],complex<f32>>, !torch.int, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[3,6,27],complex<f32>>
// CHECK:               torch.prim.Loop.condition %[[TRUE]], iter(%[[VAR20]] : !torch.vtensor<[3,6,27],complex<f32>>)
// CHECK:             } : (!torch.int, !torch.bool, !torch.vtensor<[3,6,27],complex<f32>>) -> !torch.vtensor<[3,6,27],complex<f32>>
// CHECK:             return %[[VAR6]] : !torch.vtensor<[3,6,27],complex<f32>>
func.func @torch.aten.stft.center_2D_hop_length_3_window_pad_both(%arg0: !torch.vtensor<[3,90],f32>, %arg1: !torch.vtensor<[8],f32>) -> !torch.vtensor<[3,6,27],complex<f32>> {
  %padmode = torch.constant.str "reflect"
  %nfft = torch.constant.int 10
  %hoplen = torch.constant.int 3
  %winlen = torch.constant.int 8
  %cstfalse = torch.constant.bool false
  %csttrue = torch.constant.bool true
  %0 = torch.aten.stft.center %arg0, %nfft, %hoplen, %winlen, %arg1, %cstfalse, %padmode, %cstfalse, %cstfalse, %csttrue : !torch.vtensor<[3,90],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[8],f32>, !torch.bool, !torch.str, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[3,6,27],complex<f32>>
  return %0 : !torch.vtensor<[3,6,27],complex<f32>>
}
