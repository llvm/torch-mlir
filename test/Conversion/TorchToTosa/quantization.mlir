// RUN: torch-mlir-opt <%s -convert-torch-to-tosa --canonicalize -split-input-file | FileCheck %s
// COM: --canonicalize is used to clean up the IR after conversion to make resulting IR easier to read


// CHECK-LABEL:   func.func @AtenMmQint8(
// CHECK-SAME:      %[[LHS:.*]]: !torch.vtensor<[3,4],si8>,
// CHECK-SAME:      %[[RHS:.*]]: !torch.vtensor<[4,3],si8>) -> !torch.vtensor<[3,3],f32> {
// CHECK:           %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[OUT_SCALE:.*]] = "tosa.const"() <{values = dense<3.784000e-04> : tensor<3x3xf32>}> : () -> tensor<3x3xf32>
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
// CHECK:           %[[OUT_SCALED:.*]] = tosa.mul %[[MATMUL_FP32]], %[[OUT_SCALE]], %[[SHIFT]] : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<1xi8>) -> tensor<3x3xf32>
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
