// RUN: torch-mlir-opt <%s -convert-torch-to-tosa --canonicalize -split-input-file | FileCheck %s
// COM: --canonicalize is used to clean up the IR after conversion to make resulting IR easier to read


// CHECK-LABEL:   func.func @AtenMmQint8(
// CHECK-SAME:      %[[LHS:.*]]: !torch.vtensor<[3,4],si8>,
// CHECK-SAME:      %[[RHS:.*]]: !torch.vtensor<[4,3],si8>) -> !torch.vtensor<[3,3],f32> {
// CHECK-DAG:           %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG:           %[[OUT_SCALE:.*]] = "tosa.const"() <{values = dense<3.784000e-04> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK-DAG:           %[[MUL_OUT_SHAPE:.*]] = tosa.const_shape  {values = dense<3> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:           %[[RHS_SHAPE:.*]] = tosa.const_shape  {values = dense<[1, 4, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[LHS_SHAPE:.*]] = tosa.const_shape  {values = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:           %[[RHS_ZP:.*]] = "tosa.const"() <{values = dense<18> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG:           %[[LHS_ZP:.*]] = "tosa.const"() <{values = dense<-25> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[RHS_TENSOR:.*]] = torch_c.to_builtin_tensor %[[RHS]] : !torch.vtensor<[4,3],si8> -> tensor<4x3xi8>
// CHECK:           %[[LHS_TENSOR:.*]] = torch_c.to_builtin_tensor %[[LHS]] : !torch.vtensor<[3,4],si8> -> tensor<3x4xi8>
// CHECK:           %[[LHS_RESHAPED:.*]] = tosa.reshape %[[LHS_TENSOR]], %[[LHS_SHAPE]] : (tensor<3x4xi8>, !tosa.shape<3>) -> tensor<1x3x4xi8>
// CHECK:           %[[RHS_RESHAPED:.*]] = tosa.reshape %[[RHS_TENSOR]], %[[RHS_SHAPE]] : (tensor<4x3xi8>, !tosa.shape<3>) -> tensor<1x4x3xi8>
// CHECK:           %[[MATMUL:.*]] = tosa.matmul %[[LHS_RESHAPED]], %[[RHS_RESHAPED]], %[[LHS_ZP]], %[[RHS_ZP]] : (tensor<1x3x4xi8>, tensor<1x4x3xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3xi32>
// CHECK:           %[[MATMUL_RESHAPE:.*]] = tosa.reshape %[[MATMUL]], %[[MUL_OUT_SHAPE]] : (tensor<1x3x3xi32>, !tosa.shape<2>) -> tensor<3x3xi32>
// CHECK:           %[[MATMUL_FP32:.*]] = tosa.cast %[[MATMUL_RESHAPE]] : (tensor<3x3xi32>) -> tensor<3x3xf32>
// CHECK:           %[[OUT_SCALED:.*]] = tosa.mul %[[MATMUL_FP32]], %[[OUT_SCALE]], %[[SHIFT]] : (tensor<3x3xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<3x3xf32>
// CHECK:           %[[RES:.*]] = torch_c.from_builtin_tensor %[[OUT_SCALED]] : tensor<3x3xf32> -> !torch.vtensor<[3,3],f32>
// CHECK:           return %[[RES]]
func.func @AtenMmQint8(%arg0: !torch.vtensor<[3,4],si8>, %arg1: !torch.vtensor<[4,3],si8>) -> !torch.vtensor<[3,3],f32>
{
  %float3.784000e-04 = torch.constant.float 3.784000e-04
  %int0 = torch.constant.int 0
  %int18 = torch.constant.int 18
  %float1.760000e-02 = torch.constant.float 1.760000e-02
  %float2.150000e-02 = torch.constant.float 2.150000e-02
  %int-25 = torch.constant.int -25
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %0 = torch.aten.clamp %arg0, %int-128, %int127 : !torch.vtensor<[3,4],si8>, !torch.int, !torch.int -> !torch.vtensor<[3,4],si8>
  %1 = torch.aten.clamp %arg1, %int-128, %int127 : !torch.vtensor<[4,3],si8>, !torch.int, !torch.int -> !torch.vtensor<[4,3],si8>
  %2 = torch.aten._make_per_tensor_quantized_tensor %0, %float2.150000e-02, %int-25 : !torch.vtensor<[3,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,4],!torch.qint8>
  %3 = torch.aten._make_per_tensor_quantized_tensor %1, %float1.760000e-02, %int18 : !torch.vtensor<[4,3],si8>, !torch.float, !torch.int -> !torch.vtensor<[4,3],!torch.qint8>
  %4 = torch.aten.mm %2, %3 : !torch.vtensor<[3,4],!torch.qint8>, !torch.vtensor<[4,3],!torch.qint8> -> !torch.vtensor<[3,3],!torch.qint32>
  %5 = torch.aten.int_repr %4 : !torch.vtensor<[3,3],!torch.qint32> -> !torch.vtensor<[3,3],si32>
  %6 = torch.aten._make_per_tensor_quantized_tensor %5, %float3.784000e-04, %int0 : !torch.vtensor<[3,3],si32>, !torch.float, !torch.int -> !torch.vtensor<[3,3],!torch.qint32>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[3,3],!torch.qint32> -> !torch.vtensor<[3,3],f32>
  return %7 : !torch.vtensor<[3,3],f32>
}

// -----
// CHECK-LABEL:   func.func @quantization_per_tensor(
// CHECK-SAME:      %[[IN:.*]]: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[2,4,4],!torch.qint8> {
// CHECK:           %[[ZP:.*]] = "tosa.const"() <{values = dense<3> : tensor<1x1x1xi8>}> : () -> tensor<1x1x1xi8>
// CHECK:           %[[C2:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           %[[CHALF:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           %[[C10:.*]] = "tosa.const"() <{values = dense<1.000000e+01> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           %[[MUL_SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[IN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[IN]] : !torch.vtensor<[2,4,4],f32> -> tensor<2x4x4xf32>
// CHECK:           %[[RESCALE:.*]] = tosa.mul %[[IN_TENSOR]], %[[C10]], %[[MUL_SHIFT]] : (tensor<2x4x4xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<2x4x4xf32>
// CHECK:           %[[FLOOR:.*]] = tosa.floor %[[RESCALE]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:           %[[FRAC:.*]] = tosa.sub %[[RESCALE]], %[[FLOOR]] : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:           %[[CEIL:.*]] = tosa.ceil %[[RESCALE]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:           %[[FLOOR_DIV_BY_2:.*]] = tosa.mul %[[FLOOR]], %[[CHALF]], %[[MUL_SHIFT]] : (tensor<2x4x4xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<2x4x4xf32>
// CHECK:           %[[FLOOR_DIV:.*]] = tosa.floor %[[FLOOR_DIV_BY_2]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:           %[[EVEN_COMP:.*]] = tosa.mul %[[FLOOR_DIV]], %[[C2]], %[[MUL_SHIFT]] : (tensor<2x4x4xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<2x4x4xf32>
// CHECK:           %[[FLOOR_INPUT_EVEN:.*]] = tosa.equal %[[FLOOR]], %[[EVEN_COMP]] : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xi1>
// CHECK:           %[[FRAC_EQ_HALF:.*]] = tosa.equal %[[FRAC]], %[[CHALF]] : (tensor<2x4x4xf32>, tensor<1x1x1xf32>) -> tensor<2x4x4xi1>
// CHECK:           %[[GRTR:.*]] = tosa.greater %[[CHALF]], %[[FRAC]] : (tensor<1x1x1xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xi1>
// CHECK:           %[[AND:.*]] = tosa.logical_and %[[FRAC_EQ_HALF]], %[[FLOOR_INPUT_EVEN]] : (tensor<2x4x4xi1>, tensor<2x4x4xi1>) -> tensor<2x4x4xi1>
// CHECK:           %[[OR:.*]] = tosa.logical_or %[[GRTR]], %[[AND]] : (tensor<2x4x4xi1>, tensor<2x4x4xi1>) -> tensor<2x4x4xi1>
// CHECK:           %[[SELECT:.*]] = tosa.select %[[OR]], %[[FLOOR]], %[[CEIL]] : (tensor<2x4x4xi1>, tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:           %[[CAST:.*]] = tosa.cast %[[SELECT]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xi8>
// CHECK:           %[[ADD:.*]] = tosa.add %[[CAST]], %[[ZP]] : (tensor<2x4x4xi8>, tensor<1x1x1xi8>) -> tensor<2x4x4xi8>
// CHECK:           %[[RES:.*]] = torch_c.from_builtin_tensor %[[ADD]] : tensor<2x4x4xi8> -> !torch.vtensor<[2,4,4],!torch.qint8>
// CHECK:           return %[[RES]]
func.func @quantization_per_tensor(%arg0: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[2,4,4],!torch.qint8> {
  %dtype = torch.constant.int 12
  %scale = torch.constant.float 0.1
  %zp = torch.constant.int 3
  %0 = torch.aten.quantize_per_tensor %arg0, %scale, %zp, %dtype : !torch.vtensor<[2,4,4],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[2,4,4],!torch.qint8>
  return %0 : !torch.vtensor<[2,4,4],!torch.qint8>
}


// -----
// CHECK-LABEL:   func.func @dequantize.self(
// CHECK-SAME:      %[[IN:.*]]: !torch.vtensor<[3,4,3,2],si8>,
// CHECK-SAME:      %[[SCALE:.*]]: !torch.vtensor<[3],f32>,
// CHECK-SAME:      %[[ZP:.*]]: !torch.vtensor<[3],si8>) -> !torch.vtensor<[3,4,3,2],f32> {
// CHECK:           %[[MUL_SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[QUANT_PARAM_SHAPE:.*]] = tosa.const_shape  {values = dense<[3, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[IN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[IN]] : !torch.vtensor<[3,4,3,2],si8> -> tensor<3x4x3x2xi8>
// CHECK:           %[[IN_I32:.*]] = tosa.cast %[[IN_TENSOR]] : (tensor<3x4x3x2xi8>) -> tensor<3x4x3x2xi32>
// CHECK:           %[[ZP_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[3],si8> -> tensor<3xi8>
// CHECK:           %[[ZP_I32:.*]] = tosa.cast %[[ZP_TENSOR]] : (tensor<3xi8>) -> tensor<3xi32>
// CHECK:           %[[ZP_RESHAPED:.*]] = tosa.reshape %[[ZP_I32]], %[[QUANT_PARAM_SHAPE]] : (tensor<3xi32>, !tosa.shape<4>) -> tensor<3x1x1x1xi32>
// CHECK:           %[[SUB:.*]] = tosa.sub %[[IN_I32]], %[[ZP_RESHAPED]] : (tensor<3x4x3x2xi32>, tensor<3x1x1x1xi32>) -> tensor<3x4x3x2xi32>
// CHECK:           %[[SUB_CAST:.*]] = tosa.cast %[[SUB]] : (tensor<3x4x3x2xi32>) -> tensor<3x4x3x2xf32>
// CHECK:           %[[SCALE_TENSOR:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[SCALE_RESHAPED:.*]] = tosa.reshape %[[SCALE_TENSOR]], %[[QUANT_PARAM_SHAPE]] : (tensor<3xf32>, !tosa.shape<4>) -> tensor<3x1x1x1xf32>
// CHECK:           %[[MUL:.*]] = tosa.mul %[[SUB_CAST]], %[[SCALE_RESHAPED]], %[[MUL_SHIFT]] : (tensor<3x4x3x2xf32>, tensor<3x1x1x1xf32>, tensor<1xi8>) -> tensor<3x4x3x2xf32>
// CHECK:           %[[RES:.*]] = torch_c.from_builtin_tensor %[[MUL]]
func.func @dequantize.self(%arg0: !torch.vtensor<[3,4,3,2],si8>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],si8>) -> !torch.vtensor<[3,4,3,2],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.aten._make_per_channel_quantized_tensor %arg0, %arg1, %arg2, %int0 : !torch.vtensor<[3,4,3,2],si8>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],si8>, !torch.int -> !torch.vtensor<[3,4,3,2],!torch.qint8>
    %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[3,4,3,2],!torch.qint8> -> !torch.vtensor<[3,4,3,2],f32>
    return %1 : !torch.vtensor<[3,4,3,2],f32>
}


// -----
// CHECK-LABEL:   func.func @quantized_conv(
// CHECK:           %[[WTS_ZP:.*]] = "tosa.const"() <{values = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[IN_ZP:.*]] = "tosa.const"() <{values = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[CONV:.*]] = tosa.conv2d
// CHECK-SAME:      %[[IN_ZP]], %[[WTS_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x7x8x4xi8>, tensor<3x3x2x4xi8>, tensor<?xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x5x7x3xi32>
// CHECK-NOT: torch.aten.quantize_per_tensor
// CHECK-NOT: torch.aten.dequantize.self
// CHECK-NOT: torch.aten._make_per_tensor_quantized_tensor
// CHECK-NOT: torch.aten.dequantize.tensor

func.func @quantized_conv(%arg0: !torch.vtensor<[?,4,7,8],si8>, %arg1: !torch.vtensor<[3,4,3,2],si8>, %arg2: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,3,5,7],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %float1.000000e-04 = torch.constant.float 1.000000e-04
  %int3 = torch.constant.int 3
  %int7 = torch.constant.int 7
  %float1.000000e-02 = torch.constant.float 1.000000e-02
  %int14 = torch.constant.int 14
  %0 = torch.aten.quantize_per_tensor %arg2, %float1.000000e-04, %int0, %int14 : !torch.vtensor<[?],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[?],!torch.qint32>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[?],!torch.qint32> -> !torch.vtensor<[?],f32>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten._make_per_tensor_quantized_tensor %arg0, %float1.000000e-02, %int7 : !torch.vtensor<[?,4,7,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[?,4,7,8],!torch.qint8>
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg1, %float1.000000e-02, %int3 : !torch.vtensor<[3,4,3,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,4,3,2],!torch.qint8>
  %7 = torch.aten.quantize_per_tensor %1, %float1.000000e-04, %int0, %int14 : !torch.vtensor<[?],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[?],!torch.qint32>
  %8 = torch.aten.int_repr %7 : !torch.vtensor<[?],!torch.qint32> -> !torch.vtensor<[?],si32>
  %9 = torch.aten.convolution %5, %6, %8, %2, %3, %2, %false, %4, %int1 : !torch.vtensor<[?,4,7,8],!torch.qint8>, !torch.vtensor<[3,4,3,2],!torch.qint8>, !torch.vtensor<[?],si32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,3,5,7],si32>
  %10 = torch.aten._make_per_tensor_quantized_tensor %9, %float1.000000e-04, %int0 : !torch.vtensor<[?,3,5,7],si32>, !torch.float, !torch.int -> !torch.vtensor<[?,3,5,7],!torch.qint32>
  %11 = torch.aten.dequantize.tensor %10 : !torch.vtensor<[?,3,5,7],!torch.qint32> -> !torch.vtensor<[?,3,5,7],f32>
  return %11 : !torch.vtensor<[?,3,5,7],f32>
}
