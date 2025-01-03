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
