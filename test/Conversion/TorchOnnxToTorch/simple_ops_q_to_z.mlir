// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.

// CHECK-LABEL: @test_quantizelinear_si8
func.func @test_quantizelinear_si8(%arg0: !torch.vtensor<[6],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>

  // CHECK: %[[C12:.+]] = torch.constant.int 12
  // CHECK: %[[SCALE:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[ZP:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],si8> -> !torch.int
  // CHECK: %[[QUANT:.+]] = torch.aten.quantize_per_tensor %arg0, %[[SCALE]], %[[ZP]], %[[C12]]
  // CHECK: %[[REPR:.+]] = torch.aten.int_repr %[[QUANT]]
  // CHECK: return %[[REPR]]
  return %0 : !torch.vtensor<[6],si8>
}

// -----

// CHECK-LABEL: @test_quantizelinear_ui8
func.func @test_quantizelinear_ui8(%arg0: !torch.vtensor<[6],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],ui8>) -> !torch.vtensor<[6],ui8> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[6],ui8>
  // CHECK: %[[C13:.+]] = torch.constant.int 13
  // CHECK: %[[SCALE:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[ZP:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[QUANT:.+]] = torch.aten.quantize_per_tensor %arg0, %[[SCALE]], %[[ZP]], %[[C13]]
  // CHECK: %[[REPR:.+]] = torch.aten.int_repr %[[QUANT]]
  // CHECK: return %[[REPR]]
  return %0 : !torch.vtensor<[6],ui8>
}

// -----

// CHECK-LABEL: @test_quantizelinear_i32
func.func @test_quantizelinear_i32(%arg0: !torch.vtensor<[6],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si32>) -> !torch.vtensor<[6],si32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[6],si32>
  // CHECK: %[[C14:.+]] = torch.constant.int 14
  // CHECK: %[[SCALE:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[ZP:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK: %[[QUANT:.+]] = torch.aten.quantize_per_tensor %arg0, %[[SCALE]], %[[ZP]], %[[C14]]
  // CHECK: %[[REPR:.+]] = torch.aten.int_repr %[[QUANT]]
  // CHECK: return %[[REPR]]
  return %0 : !torch.vtensor<[6],si32>
}

// -----

// CHECK-LABEL: @test_qlinearconv_nobias
func.func @test_qlinearconv_nobias(%arg0: !torch.vtensor<[1,1,7,7],ui8>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[1,1,1,1],ui8>, %arg4: !torch.vtensor<[1],f32>, %arg5: !torch.vtensor<[1],ui8>, %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>) -> !torch.vtensor<[1,1,7,7],ui8> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (!torch.vtensor<[1,1,7,7],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>, !torch.vtensor<[1,1,1,1],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[1,1,7,7],ui8>
  // CHECK: %[[aZp:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[bZp:.+]] = torch.aten.item %arg5 : !torch.vtensor<[1],ui8> -> !torch.int
  // CHECK: %[[cZp:.+]] = torch.aten.item %arg7 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[aScale:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[bScale:.+]] = torch.aten.item %arg4 : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[cScale:.+]] = torch.aten.item %arg6 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[aScale]], %[[aZp]] : !torch.vtensor<[1,1,7,7],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.quint8>
  // CHECK: %[[B:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg3, %[[bScale]], %[[bZp]] : !torch.vtensor<[1,1,1,1],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1,1],!torch.quint8>
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.+]] = torch.constant.int 0
  // CHECK: %[[PAD:.+]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT0_1]]
  // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_2:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_3:.+]] = torch.constant.int 1
  // CHECK: %[[INT0_2:.+]] = torch.constant.int 0
  // CHECK: %[[KERNEL:.+]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]]
  // CHECK: %[[DILATION:.+]] = torch.prim.ListConstruct %[[INT1_2]], %[[INT1_3]]
  // CHECK: %[[STRIDE:.+]] = torch.prim.ListConstruct %[[INT0_2]], %[[INT0_2]]
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT1_5:.+]] = torch.constant.int 1
  // CHECK: %[[CONV:.+]] = torch.aten.convolution %[[A]], %[[B]], %[[NONE]], %[[DILATION]], %[[PAD]], %[[KERNEL]], %[[FALSE]], %[[STRIDE]], %[[INT1_5]] : !torch.vtensor<[1,1,7,7],!torch.quint8>, !torch.vtensor<[1,1,1,1],!torch.quint8>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.qint32>
  // CHECK: %[[convScale:.+]] = torch.aten.mul.float %[[aScale]], %[[bScale]] : !torch.float, !torch.float -> !torch.float
  // CHECK: %[[INT0_6:.+]] = torch.constant.int 0
  // CHECK: %[[C:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[CONV]], %[[convScale]], %[[INT0_6]] : !torch.vtensor<[1,1,7,7],!torch.qint32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.qint32>
  // CHECK: %[[DEQ:.+]] = torch.aten.dequantize.self %[[C]] : !torch.vtensor<[1,1,7,7],!torch.qint32> -> !torch.vtensor<[1,1,7,7],f32>
  // CHECK: %[[INT13:.+]] = torch.constant.int 13
  // CHECK: %[[QUANT:.+]] = torch.aten.quantize_per_tensor %[[DEQ]], %[[cScale]], %[[cZp]], %[[INT13]] : !torch.vtensor<[1,1,7,7],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.quint8>
  // CHECK: %[[INT:.+]] = torch.aten.int_repr %[[QUANT]] : !torch.vtensor<[1,1,7,7],!torch.quint8> -> !torch.vtensor<[1,1,7,7],ui8>
  // CHECK: return %[[INT]] : !torch.vtensor<[1,1,7,7],ui8>
  return %0 : !torch.vtensor<[1,1,7,7],ui8>
}

// -----

// CHECK-LABEL: @test_qlinearconv_bias
func.func @test_qlinearconv_bias(%arg0: !torch.vtensor<[1,1,7,7],ui8>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[1,1,1,1],ui8>, %arg4: !torch.vtensor<[1],f32>, %arg5: !torch.vtensor<[1],ui8>, %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>, %arg8 : !torch.vtensor<[7],si32>) -> !torch.vtensor<[1,1,7,7],ui8> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (!torch.vtensor<[1,1,7,7],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>, !torch.vtensor<[1,1,1,1],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>, !torch.vtensor<[7],si32>) -> !torch.vtensor<[1,1,7,7],ui8>
  // CHECK: %[[aZp:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[bZp:.+]] = torch.aten.item %arg5 : !torch.vtensor<[1],ui8> -> !torch.int
  // CHECK: %[[cZp:.+]] = torch.aten.item %arg7 : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK: %[[aScale:.+]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[bScale:.+]] = torch.aten.item %arg4 : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[cScale:.+]] = torch.aten.item %arg6 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[A:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[aScale]], %[[aZp]] : !torch.vtensor<[1,1,7,7],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.quint8>
  // CHECK: %[[B:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg3, %[[bScale]], %[[bZp]] : !torch.vtensor<[1,1,1,1],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1,1],!torch.quint8>
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.+]] = torch.constant.int 0
  // CHECK: %[[PAD:.+]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT0_1]]
  // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_2:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_3:.+]] = torch.constant.int 1
  // CHECK: %[[INT0_2:.+]] = torch.constant.int 0
  // CHECK: %[[KERNEL:.+]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]]
  // CHECK: %[[DILATION:.+]] = torch.prim.ListConstruct %[[INT1_2]], %[[INT1_3]]
  // CHECK: %[[STRIDE:.+]] = torch.prim.ListConstruct %[[INT0_2]], %[[INT0_2]]
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[INT1_5:.+]] = torch.constant.int 1
  // CHECK: %[[CONV:.+]] = torch.aten.convolution %[[A]], %[[B]], %arg8, %[[DILATION]], %[[PAD]], %[[KERNEL]], %[[FALSE]], %[[STRIDE]], %[[INT1_5]] : !torch.vtensor<[1,1,7,7],!torch.quint8>, !torch.vtensor<[1,1,1,1],!torch.quint8>, !torch.vtensor<[7],si32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.qint32>
  // CHECK: %[[convScale:.+]] = torch.aten.mul.float %[[aScale]], %[[bScale]] : !torch.float, !torch.float -> !torch.float
  // CHECK: %[[INT0_6:.+]] = torch.constant.int 0
  // CHECK: %[[C:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[CONV]], %[[convScale]], %[[INT0_6]] : !torch.vtensor<[1,1,7,7],!torch.qint32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.qint32>
  // CHECK: %[[DEQ:.+]] = torch.aten.dequantize.self %[[C]] : !torch.vtensor<[1,1,7,7],!torch.qint32> -> !torch.vtensor<[1,1,7,7],f32>
  // CHECK: %[[INT13:.+]] = torch.constant.int 13
  // CHECK: %[[QUANT:.+]] = torch.aten.quantize_per_tensor %[[DEQ]], %[[cScale]], %[[cZp]], %[[INT13]] : !torch.vtensor<[1,1,7,7],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[1,1,7,7],!torch.quint8>
  // CHECK: %[[INT:.+]] = torch.aten.int_repr %[[QUANT]] : !torch.vtensor<[1,1,7,7],!torch.quint8> -> !torch.vtensor<[1,1,7,7],ui8>
  // CHECK: return %[[INT]] : !torch.vtensor<[1,1,7,7],ui8>
  return %0 : !torch.vtensor<[1,1,7,7],ui8>
}

// -----

// CHECK-LABEL: @test_qlinearmatmul_2D
func.func @test_qlinearmatmul_2D(%arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.vtensor<[1],ui8>, %arg3: !torch.vtensor<[4,3],ui8>, %arg4: !torch.vtensor<[1],f32>, %arg5: !torch.vtensor<[1],ui8>, %arg6: !torch.vtensor<[1],f32>, %arg7: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[2,3],ui8> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>, !torch.vtensor<[4,3],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[2,3],ui8>
  // CHECK-DAG: %[[EMPTY:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK-DAG: %[[RESH0:.+]] = torch.aten.reshape %arg2, %[[EMPTY]] : !torch.vtensor<[1],ui8>, !torch.list<int> -> !torch.vtensor<[],ui8>
  // CHECK-DAG: %[[RESH1:.+]] = torch.aten.reshape %arg5, %[[EMPTY]] : !torch.vtensor<[1],ui8>, !torch.list<int> -> !torch.vtensor<[],ui8>
  // CHECK-DAG: %[[RESH2:.+]] = torch.aten.reshape %arg7, %[[EMPTY]] : !torch.vtensor<[1],ui8>, !torch.list<int> -> !torch.vtensor<[],ui8>
  // CHECK-DAG: %[[RESH3:.+]] = torch.aten.reshape %arg1, %[[EMPTY]] : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RESH4:.+]] = torch.aten.reshape %arg4, %[[EMPTY]] : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RESH5:.+]] = torch.aten.reshape %arg6, %[[EMPTY]] : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[AZP:.+]] = torch.aten.item %[[RESH0]] : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK-DAG: %[[BZP:.+]] = torch.aten.item %[[RESH1]] : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK-DAG: %[[CZP:.+]] = torch.aten.item %[[RESH2]] : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK-DAG: %[[ASCALE:.+]] = torch.aten.item %[[RESH3]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK-DAG: %[[BSCALE:.+]] = torch.aten.item %[[RESH4]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK-DAG: %[[CCSCALE:.+]] = torch.aten.item %[[RESH5]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK-DAG: %[[LHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[ASCALE]], %[[AZP]] : !torch.vtensor<[2,4],ui8>, !torch.float, !torch.int -> !torch.vtensor<[2,4],!torch.quint8>
  // CHECK-DAG: %[[RHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg3, %[[BSCALE]], %[[BZP]] : !torch.vtensor<[4,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[4,3],!torch.quint8>
  // CHECK: %[[MM:.+]] = torch.aten.mm %[[LHS]], %[[RHS]] : !torch.vtensor<[2,4],!torch.quint8>, !torch.vtensor<[4,3],!torch.quint8> -> !torch.vtensor<[2,3],si32>
  // CHECK: %[[CSCALE:.+]] = torch.aten.mul.float %[[ASCALE]], %[[BSCALE]] : !torch.float, !torch.float -> !torch.float
  // CHECK: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK: %[[QC:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[MM]], %[[CSCALE]], %[[ZERO]] : !torch.vtensor<[2,3],si32>, !torch.float, !torch.int -> !torch.vtensor<[2,3],!torch.qint32>
  // CHECK: %[[FC:.+]] = torch.aten.dequantize.self %[[QC]] : !torch.vtensor<[2,3],!torch.qint32> -> !torch.vtensor<[2,3],f32>
  // CHECK: %[[DTY:.+]] = torch.constant.int 13
  // CHECK: %[[QO:.+]] = torch.aten.quantize_per_tensor %[[FC]], %[[CCSCALE]], %[[CZP]], %[[DTY]] : !torch.vtensor<[2,3],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[2,3],!torch.quint8>
  // CHECK: %[[OUT:.+]] = torch.aten.int_repr %[[QO]] : !torch.vtensor<[2,3],!torch.quint8> -> !torch.vtensor<[2,3],ui8>
  // CHECK: return %[[OUT]]
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// CHECK-LABEL: @test_qlinearmatmul_3D
func.func @test_qlinearmatmul_3D(%arg0: !torch.vtensor<[2,2,4],ui8>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.vtensor<[1],ui8>, %arg3: !torch.vtensor<[2,4,3],ui8>, %arg4: !torch.vtensor<[1],f32>, %arg5: !torch.vtensor<[1],ui8>, %arg6: !torch.vtensor<[1],f32>, %arg7: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[2,2,3],ui8> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (!torch.vtensor<[2,2,4],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>, !torch.vtensor<[2,4,3],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[2,2,3],ui8>
  // CHECK-DAG: %[[EMPTY:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK-DAG: %[[RESH0:.+]] = torch.aten.reshape %arg2, %[[EMPTY]] : !torch.vtensor<[1],ui8>, !torch.list<int> -> !torch.vtensor<[],ui8>
  // CHECK-DAG: %[[RESH1:.+]] = torch.aten.reshape %arg5, %[[EMPTY]] : !torch.vtensor<[1],ui8>, !torch.list<int> -> !torch.vtensor<[],ui8>
  // CHECK-DAG: %[[RESH2:.+]] = torch.aten.reshape %arg7, %[[EMPTY]] : !torch.vtensor<[1],ui8>, !torch.list<int> -> !torch.vtensor<[],ui8>
  // CHECK-DAG: %[[RESH3:.+]] = torch.aten.reshape %arg1, %[[EMPTY]] : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RESH4:.+]] = torch.aten.reshape %arg4, %[[EMPTY]] : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[RESH5:.+]] = torch.aten.reshape %arg6, %[[EMPTY]] : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  // CHECK-DAG: %[[AZP:.+]] = torch.aten.item %[[RESH0]] : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK-DAG: %[[BZP:.+]] = torch.aten.item %[[RESH1]] : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK-DAG: %[[CZP:.+]] = torch.aten.item %[[RESH2]] : !torch.vtensor<[],ui8> -> !torch.int
  // CHECK-DAG: %[[ASCALE:.+]] = torch.aten.item %[[RESH3]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK-DAG: %[[BSCALE:.+]] = torch.aten.item %[[RESH4]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK-DAG: %[[CCSCALE:.+]] = torch.aten.item %[[RESH5]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK-DAG: %[[LHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[ASCALE]], %[[AZP]] : !torch.vtensor<[2,2,4],ui8>, !torch.float, !torch.int -> !torch.vtensor<[2,2,4],!torch.quint8>
  // CHECK-DAG: %[[RHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg3, %[[BSCALE]], %[[BZP]] : !torch.vtensor<[2,4,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[2,4,3],!torch.quint8>
  // CHECK: %[[MM:.+]] = torch.aten.bmm %[[LHS]], %[[RHS]] : !torch.vtensor<[2,2,4],!torch.quint8>, !torch.vtensor<[2,4,3],!torch.quint8> -> !torch.vtensor<[2,2,3],si32>
  // CHECK: %[[CSCALE:.+]] = torch.aten.mul.float %[[ASCALE]], %[[BSCALE]] : !torch.float, !torch.float -> !torch.float
  // CHECK: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK: %[[QC:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[MM]], %[[CSCALE]], %[[ZERO]] : !torch.vtensor<[2,2,3],si32>, !torch.float, !torch.int -> !torch.vtensor<[2,2,3],!torch.qint32>
  // CHECK: %[[FC:.+]] = torch.aten.dequantize.self %[[QC]] : !torch.vtensor<[2,2,3],!torch.qint32> -> !torch.vtensor<[2,2,3],f32>
  // CHECK: %[[DTY:.+]] = torch.constant.int 13
  // CHECK: %[[QO:.+]] = torch.aten.quantize_per_tensor %[[FC]], %[[CCSCALE]], %[[CZP]], %[[DTY]] : !torch.vtensor<[2,2,3],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[2,2,3],!torch.quint8>
  // CHECK: %[[OUT:.+]] = torch.aten.int_repr %[[QO]] : !torch.vtensor<[2,2,3],!torch.quint8> -> !torch.vtensor<[2,2,3],ui8>
  // CHECK: return %[[OUT]]
  return %0 : !torch.vtensor<[2,2,3],ui8>
}

// -----

// CHECK-LABEL: func.func @test_reciprocal
func.func @test_reciprocal(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.reciprocal %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Reciprocal"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_relu
func.func @test_relu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.relu %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Relu"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_round
func.func @test_round(%arg0: !torch.vtensor<[15],f32>) -> !torch.vtensor<[15],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  //CHECK: torch.aten.round %arg0 : !torch.vtensor<[15],f32> -> !torch.vtensor<[15],f32>
  %0 = torch.operator "onnx.Round"(%arg0) : (!torch.vtensor<[15],f32>) -> !torch.vtensor<[15],f32>
  return %0 : !torch.vtensor<[15],f32>
}

// -----

// CHECK-LABEL: func.func @test_scatter
func.func @test_scatter(%arg0: !torch.vtensor<[3,3],f32>, %arg1: !torch.vtensor<[2,3],si64>, %arg2: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[RESULT:.*]] = torch.aten.scatter.src %arg0, %[[INT0]], %arg1, %arg2 : !torch.vtensor<[3,3],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,3],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[3,3],f32>
  %0 = torch.operator "onnx.Scatter"(%arg0, %arg1, %arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,3],f32>, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,3],f32>
  return %0 : !torch.vtensor<[3,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_scatter_elements_with_axis
func.func @test_scatter_elements_with_axis(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.scatter.src %arg0, %int1, %arg1, %arg2 : !torch.vtensor<[1,5],f32>, !torch.int, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32> -> !torch.vtensor<[1,5],f32>
  %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32>
  return %0 : !torch.vtensor<[1,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_scatter_elements_with_duplicate_indices
func.func @test_scatter_elements_with_duplicate_indices(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[STR:.*]] = torch.constant.str "add"
  // CHECK: torch.aten.scatter.reduce %arg0, %int1, %arg1, %arg2, %str : !torch.vtensor<[1,5],f32>, !torch.int, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>, !torch.str -> !torch.vtensor<[1,5],f32>
  %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64, torch.onnx.reduction = "add"} : (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32>
  return %0 : !torch.vtensor<[1,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_scatter_elements_without_axis
func.func @test_scatter_elements_without_axis(%arg0: !torch.vtensor<[3,3],f32>, %arg1: !torch.vtensor<[2,3],si64>, %arg2: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[3,3],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,3],f32>
  %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3,3],f32>, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,3],f32>
  return %0 : !torch.vtensor<[3,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_scatter_elements_with_reduction_mul
func.func @test_scatter_elements_with_reduction_mul(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[STR:.*]] = torch.constant.str "multiply"
  // CHECK: torch.aten.scatter.reduce %arg0, %int1, %arg1, %arg2, %str : !torch.vtensor<[1,5],f32>, !torch.int, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>, !torch.str -> !torch.vtensor<[1,5],f32>
  %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64, torch.onnx.reduction = "mul"} : (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32>
  return %0 : !torch.vtensor<[1,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_sigmoid_example
func.func @test_sigmoid_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.sigmoid %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sigmoid"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_sin_example
func.func @test_sin_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.sin %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sin"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_tanh_example
func.func @test_tanh_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.tanh %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Tanh"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_sqrt_example
func.func @test_sqrt_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.sqrt %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sqrt"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_sub_bcast
func.func @test_sub_bcast(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Sub"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_sub_example
func.func @test_sub_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sub"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_sub
func.func @test_sub(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Sub"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_sub_uint8
func.func @test_sub_uint8(%arg0: !torch.vtensor<[3,4,5],ui8>, %arg1: !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>, !torch.int -> !torch.vtensor<[3,4,5],ui8>
  %0 = torch.operator "onnx.Sub"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],ui8>, !torch.vtensor<[3,4,5],ui8>) -> !torch.vtensor<[3,4,5],ui8>
  return %0 : !torch.vtensor<[3,4,5],ui8>
}

// -----

// CHECK-LABEL: func.func @test_sum_example
func.func @test_sum_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[SUM:.*]] = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  // CHECK: %[[SUM_1:.*]] = torch.aten.add.Tensor %[[SUM]], %arg2, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  // CHECK: %[[SUM_2:.*]] = torch.aten.add.Tensor %[[SUM_1]], %arg3, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sum"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_sum_one_input
func.func @test_sum_one_input(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.Sum"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_sum_two_inputs
func.func @test_sum_two_inputs(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sum"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_where_example
func.func @test_where_example(%arg0: !torch.vtensor<[2,2],i1>, %arg1: !torch.vtensor<[2,2],f32>, %arg2: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[2,2],i1>, !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
  %0 = torch.operator "onnx.Where"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,2],i1>, !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32>
  return %0 : !torch.vtensor<[2,2],f32>
}

// CHECK-LABEL: func.func @test_where_long_example
func.func @test_where_long_example(%arg0: !torch.vtensor<[2,2],i1>, %arg1: !torch.vtensor<[2,2],si64>, %arg2: !torch.vtensor<[2,2],si64>) -> !torch.vtensor<[2,2],si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[2,2],i1>, !torch.vtensor<[2,2],si64>, !torch.vtensor<[2,2],si64> -> !torch.vtensor<[2,2],si64>
  %0 = torch.operator "onnx.Where"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,2],i1>, !torch.vtensor<[2,2],si64>, !torch.vtensor<[2,2],si64>) -> !torch.vtensor<[2,2],si64>
  return %0 : !torch.vtensor<[2,2],si64>
}

// -----

// CHECK-LABEL: func.func @test_xor2d
func.func @test_xor2d(%arg0: !torch.vtensor<[3,4],i1>, %arg1: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[3,4],i1>, !torch.vtensor<[3,4],i1> -> !torch.vtensor<[3,4],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[3,4],i1>, !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1>
  return %0 : !torch.vtensor<[3,4],i1>
}

// -----

// CHECK-LABEL: func.func @test_xor3d
func.func @test_xor3d(%arg0: !torch.vtensor<[3,4,5],i1>, %arg1: !torch.vtensor<[3,4,5],i1>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],i1> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],i1>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_xor4d
func.func @test_xor4d(%arg0: !torch.vtensor<[3,4,5,6],i1>, %arg1: !torch.vtensor<[3,4,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],i1>, !torch.vtensor<[3,4,5,6],i1> -> !torch.vtensor<[3,4,5,6],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],i1>, !torch.vtensor<[3,4,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1>
  return %0 : !torch.vtensor<[3,4,5,6],i1>
}

// -----

// CHECK-LABEL: func.func @test_xor_bcast3v1d
func.func @test_xor_bcast3v1d(%arg0: !torch.vtensor<[3,4,5],i1>, %arg1: !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_xor_bcast4v4d
func.func @test_xor_bcast4v4d(%arg0: !torch.vtensor<[1,4,1,6],i1>, %arg1: !torch.vtensor<[3,1,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[1,4,1,6],i1>, !torch.vtensor<[3,1,5,6],i1> -> !torch.vtensor<[3,4,5,6],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[1,4,1,6],i1>, !torch.vtensor<[3,1,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1>
  return %0 : !torch.vtensor<[3,4,5,6],i1>
}

// -----

// CHECK-LABEL: func.func @test_squeeze_no_axes
func.func @test_squeeze_no_axes(%arg0: !torch.vtensor<[1,3,1,4,1,5,1,1],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.squeeze %arg0 : !torch.vtensor<[1,3,1,4,1,5,1,1],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Squeeze"(%arg0) : (!torch.vtensor<[1,3,1,4,1,5,1,1],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_squeeze_five_axes
func.func @test_squeeze_five_axes(%arg0: !torch.vtensor<[1,3,1,4,1,5,1,1],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[3,1,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[INT6:.*]] = torch.constant.int 6
  // CHECK: %[[INT7:.*]] = torch.constant.int 7
  // CHECK: %[[SQUEEZE_DIMS:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT4]], %[[INT6]], %[[INT7]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.prims.squeeze %arg0, %[[SQUEEZE_DIMS]] : !torch.vtensor<[1,3,1,4,1,5,1,1],f32>, !torch.list<int> -> !torch.vtensor<[3,1,4,5],f32>
  %0 = torch.operator "onnx.Squeeze"(%arg0, %arg1) : (!torch.vtensor<[1,3,1,4,1,5,1,1],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[3,1,4,5],f32>
  return %0 : !torch.vtensor<[3,1,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_squeeze
func.func @test_squeeze(%arg0: !torch.vtensor<[1,3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[SQUEEZE_DIMS:.*]] = torch.prim.ListConstruct %[[INT0]] : (!torch.int) -> !torch.list<int>
  // CHECK: torch.prims.squeeze %arg0, %[[SQUEEZE_DIMS]] : !torch.vtensor<[1,3,4,5],f32>, !torch.list<int> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Squeeze"(%arg0, %arg1) : (!torch.vtensor<[1,3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_squeeze_two_axes
func.func @test_squeeze_two_axes(%arg0: !torch.vtensor<[3,1,4,5,1],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[SQUEEZE_DIMS:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.prims.squeeze %arg0, %[[SQUEEZE_DIMS]] : !torch.vtensor<[3,1,4,5,1],f32>, !torch.list<int> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Squeeze"(%arg0, %arg1) : (!torch.vtensor<[3,1,4,5,1],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_unsqueeze_axis_0
func.func @test_unsqueeze_axis_0(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.unsqueeze %arg0, %[[INT0:.*]] : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[1,3,4,5],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,3,4,5],f32>
  return %0 : !torch.vtensor<[1,3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_unsqueeze_axis_1
func.func @test_unsqueeze_axis_1(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.unsqueeze %arg0, %[[INT1]] : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,1,4,5],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,4,5],f32>
  return %0 : !torch.vtensor<[3,1,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_unsqueeze_axis_2
func.func @test_unsqueeze_axis_2(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,1,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: torch.aten.unsqueeze %arg0, %[[INT2]] : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,4,1,5],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,1,5],f32>
  return %0 : !torch.vtensor<[3,4,1,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_unsqueeze_three_axes
func.func @test_unsqueeze_three_axes(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,1,5,1,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %arg0, %[[INT2]] : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,4,1,5],f32>
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE]], %[[INT4]] : !torch.vtensor<[3,4,1,5],f32>, !torch.int -> !torch.vtensor<[3,4,1,5,1],f32>
  // CHECK: %[[INT5:.*]] = torch.constant.int 5
  // CHECK: torch.aten.unsqueeze %[[UNSQUEEZE_1]], %[[INT5]] : !torch.vtensor<[3,4,1,5,1],f32>, !torch.int -> !torch.vtensor<[3,4,1,5,1,1],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,1,5,1,1],f32>
  return %0 : !torch.vtensor<[3,4,1,5,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_softmax_axis_0
func.func @test_softmax_axis_0(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int0, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_softmax_axis_1
func.func @test_softmax_axis_1(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int1, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_softmax_axis_2
func.func @test_softmax_axis_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int2, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_softmax_default_axis
func.func @test_softmax_default_axis(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int2, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_softmax_large_number
func.func @test_softmax_large_number(%arg0: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int1, %none : !torch.vtensor<[2,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) : (!torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_softmax_negative_axis
func.func @test_softmax_negative_axis(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int2, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_softsign
func.func @test_softsign(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[ABS:.+]] = torch.aten.abs %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[RES:.+]] = torch.aten.add.Scalar %[[ABS]], %[[INT1]], %[[INT1]] : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4,5],f32>
  // CHECK: %[[SCALE_T:.*]] = torch.aten.div.Tensor %arg0, %[[RES]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  // CHECK: return %[[SCALE_T]] : !torch.vtensor<[3,4,5],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.Softsign"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_selu
func.func @test_selu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 6 : si64} {
  // CHECK-DAG: %[[F1:.+]] = torch.constant.float 1
  // CHECK-DAG: %[[F2:.+]] = torch.constant.float 2
  // CHECK-DAG: %[[F3:.+]] = torch.constant.float 3
  // CHECK: %[[ELU:.+]] = torch.aten.elu %arg0, %[[F2]], %[[F3]], %[[F1]]
  %0 = torch.operator "onnx.Selu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32, torch.onnx.gamma = 3.000000e+00 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_max_empty_set_fp
func.func @test_reduce_max_empty_set_fp(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INF:.+]] = torch.constant.float 0x7FF0000000000000
  // CHECK-DAG: %[[INT2:.+]] = torch.constant.int 2
  // CHECK-DAG: %[[INT1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[INT4:.+]] = torch.constant.int 4
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT1]], %[[INT4]]
  // CHECK-DAG: %[[FULL:.+]] = torch.aten.full %[[LIST]], %[[INF]], %[[NONE]], %[[NONE]], %[[NONE]]
  // CHECK: return %[[FULL]]
  %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32>
  return %0 : !torch.vtensor<[2,1,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_max_empty_set_int
func.func @test_reduce_max_empty_set_int(%arg0: !torch.vtensor<[2,0,4],si32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],si32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INF:.+]] = torch.constant.int 2147483647
  // CHECK-DAG: %[[INT2:.+]] = torch.constant.int 2
  // CHECK-DAG: %[[INT1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[INT4:.+]] = torch.constant.int 4
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT1]], %[[INT4]]
  // CHECK-DAG: %[[FULL:.+]] = torch.aten.full %[[LIST]], %[[INF]], %[[NONE]], %[[NONE]], %[[NONE]]
  // CHECK: return %[[FULL]]
  %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],si32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],si32>
  return %0 : !torch.vtensor<[2,1,4],si32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_max_bool_inputs
func.func @test_reduce_max_bool_inputs(%arg0: !torch.vtensor<[4,2],i1>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[IDX:.+]] = torch.constant.int 0
  // CHECK: %[[SZ:.+]] = torch.constant.int 0
  // CHECK: %[[SEL:.+]] = torch.aten.select.int %arg1, %[[IDX]], %[[SZ]]
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SEL]]
  // CHECK: %[[DIM:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[C0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[ITEM]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[DIM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.+]] = torch.aten.add.int %[[ITEM]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LST:.+]] = torch.prim.ListConstruct %[[ADD]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[AMAX:.+]] = torch.aten.amax %arg0, %[[LST]], %[[TRUE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,1],i1>
  // CHECK: return %[[AMAX]] : !torch.vtensor<[4,1],i1>
  %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[4,2],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1>
  return %0 : !torch.vtensor<[4,1],i1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_max_bool_inputs_nokeepdims
func.func @test_reduce_max_bool_inputs_nokeepdims(%arg0: !torch.vtensor<[4,2],i1>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[IDX:.+]] = torch.constant.int 0
  // CHECK: %[[SZ:.+]] = torch.constant.int 0
  // CHECK: %[[SEL:.+]] = torch.aten.select.int %arg1, %[[IDX]], %[[SZ]]
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SEL]]
  // CHECK: %[[DIM:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[C0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[ITEM]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[DIM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.+]] = torch.aten.add.int %[[ITEM]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LST:.+]] = torch.prim.ListConstruct %[[ADD]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[AMAX:.+]] = torch.aten.amax %arg0, %[[LST]], %[[FALSE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],i1>
  // CHECK: return %[[AMAX]] : !torch.vtensor<[4],i1>
  %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[4,2],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_max_all_dims_default
func.func @test_reduce_max_all_dims_default(%arg0: !torch.vtensor<[4,2],i1>) -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[I0:.+]] = torch.constant.int 0
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[RANK:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[C0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[I0]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[A0:.+]] = torch.aten.add.int %[[I0]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[I1]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[A1:.+]] = torch.aten.add.int %[[I1]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[A0]], %[[A1]]
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[MAX:.+]] = torch.aten.amax %arg0, %[[LIST]], %[[FALSE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[MAX]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.ReduceMax"(%arg0) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[4,2],i1>) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

func.func @test_reduce_max_attr(%arg0: !torch.vtensor<[4,2],i1>) -> !torch.vtensor<[4],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[DIM:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[INT1]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[DIM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.+]] = torch.aten.add.int %[[INT1]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[ADD]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[AMAX:.+]] = torch.aten.amax %arg0, %[[LIST]], %[[FALSE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],i1>
  // CHECK: return %[[AMAX]]
  %0 = torch.operator "onnx.ReduceMax"(%arg0) {torch.onnx.keepdims = 0 : si64, torch.onnx.axes=[1 : si64]} : (!torch.vtensor<[4,2],i1>) -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_l1_default_axes_keepdims_example
func.func @test_reduce_l1_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[ABS:.+]] = torch.aten.abs %arg0 : !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[ABS]], %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceL1"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_l1_keep_dims_example
func.func @test_reduce_l1_keep_dims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[ABS:.+]] = torch.aten.abs %arg0 : !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[ABS]], %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[3,2,1],f32>
  %0 = torch.operator "onnx.ReduceL1"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32>
  return %0 : !torch.vtensor<[3,2,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_l1_do_not_keepdims_example
func.func @test_reduce_l1_do_not_keepdims_example(%arg0:!torch.vtensor<[3,2,2],f32>, %arg1:!torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[ABS:.+]] = torch.aten.abs %arg0 : !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[ABS]], %[[DIMS]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceL1"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_l2_default_axes_keepdims_example
func.func @test_reduce_l2_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[TRUE_0:.+]] = torch.constant.bool true
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[TRUE_0]], %[[NONE_0]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %[[SUM]], %[[INT6_0]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[1,1,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: %[[SQRT:.+]] = torch.aten.sqrt %[[CAST]] : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
  // CHECK: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[SQRT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[1,1,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceL2"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_l2_do_not_keepdims_example_expanded
func.func @test_reduce_l2_do_not_keepdims_example_expanded(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0_0]], %[[INT0_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE_0:.+]] = torch.constant.bool false
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[FALSE_0]], %[[NONE_0]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  // CHECK: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[FALSE_1:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %[[SUM]], %[[INT6_0]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[3,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  // CHECK: %[[SQRT:.+]] = torch.aten.sqrt %[[CAST]] : !torch.vtensor<[3,2],f32> -> !torch.vtensor<[3,2],f32>
  // CHECK: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[SQRT]], %[[INT6_1]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[3,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceL2"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_l2_keep_dims_example
func.func @test_reduce_l2_keep_dims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0_0]], %[[INT0_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[TRUE]], %[[NONE_0]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %[[SUM]], %[[INT6_0]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[3,2,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: %[[SQRT:.+]] = torch.aten.sqrt %[[CAST]] : !torch.vtensor<[3,2,1],f32> -> !torch.vtensor<[3,2,1],f32>
  // CHECK: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[SQRT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[3,2,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[3,2,1],f32>

  %0 = torch.operator "onnx.ReduceL2"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32>
  return %0 : !torch.vtensor<[3,2,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_l2_keep_dims_int_input_example
func.func @test_reduce_l2_keep_dims_int_input_example(%arg0: !torch.vtensor<[3,2,2],si64>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],si64>, !torch.vtensor<[3,2,2],si64> -> !torch.vtensor<[3,2,2],si64>
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0_0]], %[[INT0_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[TRUE]], %[[NONE_0]] : !torch.vtensor<[3,2,2],si64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: %[[INT6_0:.+]] = torch.constant.int 6
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %[[SUM]], %[[INT6_0]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[3,2,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: %[[SQRT:.+]] = torch.aten.sqrt %[[CAST]] : !torch.vtensor<[3,2,1],f32> -> !torch.vtensor<[3,2,1],f32>
  // CHECK: %[[INT6_1:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[SQRT]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[3,2,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[3,2,1],f32>

  %0 = torch.operator "onnx.ReduceL2"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32>
  return %0 : !torch.vtensor<[3,2,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_log_sum_default_axes_keepdims_example
func.func @test_reduce_log_sum_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: %[[LOG:.+]] = torch.aten.log %[[SUM]] : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
  // CHECK: return %[[LOG]] : !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceLogSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_log_sum_keep_dims_example
func.func @test_reduce_log_sum_keep_dims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: %[[LOG:.+]] = torch.aten.log %[[SUM]] : !torch.vtensor<[3,2,1],f32> -> !torch.vtensor<[3,2,1],f32>
  // CHECK: return %[[LOG]] : !torch.vtensor<[3,2,1],f32>
  %0 = torch.operator "onnx.ReduceLogSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32>
  return %0 : !torch.vtensor<[3,2,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_log_sum_do_not_keepdims_example
func.func @test_reduce_log_sum_do_not_keepdims_example(%arg0:!torch.vtensor<[3,2,2],f32>, %arg1:!torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  // CHECK: %[[LOG:.+]] = torch.aten.log %[[SUM]] : !torch.vtensor<[3,2],f32> -> !torch.vtensor<[3,2],f32>
  // CHECK: return %[[LOG]] : !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceLogSum"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_log_sum_exp_default_axes_keepdims_example
func.func @test_reduce_log_sum_exp_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT7:.+]] = torch.constant.int 7
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %arg0, %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] : !torch.vtensor<[3,2,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[EXP:.+]] = torch.aten.exp %[[CAST]] : !torch.vtensor<[3,2,2],f64> -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIMS]], %[[TRUE]], %[[NONE_1]] : !torch.vtensor<[3,2,2],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f64>
  // CHECK: %[[LOG:.+]] = torch.aten.log %[[SUM]] : !torch.vtensor<[1,1,1],f64> -> !torch.vtensor<[1,1,1],f64>
  // CHECK: %[[INT6:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[LOG]], %[[INT6]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] : !torch.vtensor<[1,1,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceLogSumExp"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_log_sum_exp_do_not_keepdims_example_expanded
func.func @test_reduce_log_sum_exp_do_not_keepdims_example_expanded(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT7:.+]] = torch.constant.int 7
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[FALSE_0:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %arg0, %[[INT7]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_0]] : !torch.vtensor<[3,2,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[EXP:.+]] = torch.aten.exp %[[CAST]] : !torch.vtensor<[3,2,2],f64> -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE_1:.+]] = torch.constant.bool false
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIMS]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[3,2,2],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f64>
  // CHECK: %[[LOG:.+]] = torch.aten.log %[[SUM]] : !torch.vtensor<[3,2],f64> -> !torch.vtensor<[3,2],f64>
  // CHECK: %[[INT6:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[LOG]], %[[INT6]], %[[FALSE_0]], %[[FALSE_0]], %[[NONE_0]] : !torch.vtensor<[3,2],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceLogSumExp"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_log_sum_exp_keep_dims_example
func.func @test_reduce_log_sum_exp_keep_dims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT7:.+]] = torch.constant.int 7
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %arg0, %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] : !torch.vtensor<[3,2,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[EXP:.+]] = torch.aten.exp %[[CAST]] : !torch.vtensor<[3,2,2],f64> -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIMS]], %[[TRUE]], %[[NONE_1]] : !torch.vtensor<[3,2,2],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f64>
  // CHECK: %[[LOG:.+]] = torch.aten.log %[[SUM]] : !torch.vtensor<[3,2,1],f64> -> !torch.vtensor<[3,2,1],f64>
  // CHECK: %[[INT6:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[LOG]], %[[INT6]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] : !torch.vtensor<[3,2,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[3,2,1],f32>
  %0 = torch.operator "onnx.ReduceLogSumExp"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32>
  return %0 : !torch.vtensor<[3,2,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_log_sum_exp_keep_dims_int_input_example
func.func @test_reduce_log_sum_exp_keep_dims_int_input_example(%arg0: !torch.vtensor<[3,2,2],si64>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT7:.+]] = torch.constant.int 7
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %arg0, %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] : !torch.vtensor<[3,2,2],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[EXP:.+]] = torch.aten.exp %[[CAST]] : !torch.vtensor<[3,2,2],f64> -> !torch.vtensor<[3,2,2],f64>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE_1:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[EXP]], %[[DIMS]], %[[TRUE]], %[[NONE_1]] : !torch.vtensor<[3,2,2],f64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f64>
  // CHECK: %[[LOG:.+]] = torch.aten.log %[[SUM]] : !torch.vtensor<[3,2,1],f64> -> !torch.vtensor<[3,2,1],f64>
  // CHECK: %[[INT6:.+]] = torch.constant.int 6
  // CHECK: %[[CASTLIKE:.+]] = torch.aten.to.dtype %[[LOG]], %[[INT6]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] : !torch.vtensor<[3,2,1],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],f32>
  // CHECK: return %[[CASTLIKE]] : !torch.vtensor<[3,2,1],f32>
  %0 = torch.operator "onnx.ReduceLogSumExp"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2,1],f32>
  return %0 : !torch.vtensor<[3,2,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_default_axes_keepdims_example
func.func @test_reduce_sum_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_do_not_keepdims_example
func.func @test_reduce_sum_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_empty_axes_input_noop_example
func.func @test_reduce_sum_empty_axes_input_noop_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[3,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64, torch.onnx.noop_with_empty_axes = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[3,2,2],f32>
  return %0 : !torch.vtensor<[3,2,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_empty_set_non_reduced_axis_zero
func.func @test_reduce_sum_empty_set_non_reduced_axis_zero(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[INT2]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[2,0,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,0,1],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32>
  return %0 : !torch.vtensor<[2,0,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_keepdims_example
func.func @test_reduce_sum_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VAL_1:.*]] = torch.vtensor.literal(dense<2> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %[[VAL_1]], %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %arg1 = torch.vtensor.literal(dense<2> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_negative_axes_keepdims_example
func.func @test_reduce_sum_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[DIM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.sum.dim_IntList %arg0, %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_square_default_axes_keepdims_example
func.func @test_reduce_sum_square_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceSumSquare"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_square_do_not_keepdims_example
func.func @test_reduce_sum_square_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceSumSquare"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_square_empty_set_non_reduced_axis_zero
func.func @test_reduce_sum_square_empty_set_non_reduced_axis_zero(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32> attributes {torch.onnx_meta.ir_version = 8: si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[2,0,4],f32>, !torch.vtensor<[2,0,4],f32> -> !torch.vtensor<[2,0,4],f32>
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[INT2]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[2,0,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,0,1],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[2,0,1],f32>
  %0 = torch.operator "onnx.ReduceSumSquare"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32>
  return %0 : !torch.vtensor<[2,0,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_square_keepdims_example
func.func @test_reduce_sum_square_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32> -> !torch.vtensor<[3,2,2],f32>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceSumSquare"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_square_keepdims_int_example
func.func @test_reduce_sum_square_keepdims_int_example(%arg0: !torch.vtensor<[3,2,2],si64>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MULT:.+]] = torch.aten.mul.Tensor %arg0, %arg0 : !torch.vtensor<[3,2,2],si64>, !torch.vtensor<[3,2,2],si64> -> !torch.vtensor<[3,2,2],si64>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[ITEM]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[SUM:.+]] = torch.aten.sum.dim_IntList %[[MULT]], %[[DIMS]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],si64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  // CHECK: return %[[SUM]] : !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceSumSquare"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: @test_reduce_mean_negative_axes_keepdims_example
func.func @test_reduce_mean_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK:  %[[TENSOR:.+]] = torch.vtensor.literal(dense<-2> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // CHECK:  %[[DIM:.+]] = torch.constant.int 0
  // CHECK:  %[[A0:.+]] = torch.constant.int 0
  // CHECK:  %[[SEL0:.+]] = torch.aten.select.int %[[TENSOR]], %[[DIM]], %[[A0]]
  // CHECK:  %[[ITEM0:.+]] = torch.aten.item %[[SEL0]]
  // CHECK:  %[[LIST:.+]] = torch.prim.ListConstruct %[[ITEM0]]
  // CHECK:  %[[TRUE:.+]] = torch.constant.bool true
  // CHECK:  %[[NONE:.+]] = torch.constant.none
  // CHECK:  %[[SUM:.+]] = torch.aten.mean.dim %arg0, %[[LIST]], %[[TRUE]], %[[NONE]]
  // CHECK:  return %[[SUM]]
  %cst = torch.vtensor.literal(dense<-2> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %cst) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: @test_reduce_mean_one_axes_dropdims_example
func.func @test_reduce_mean_one_axes_dropdims_example(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK:  %[[TENSOR:.+]] = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // CHECK:  %[[DIM:.+]] = torch.constant.int 0
  // CHECK:  %[[A0:.+]] = torch.constant.int 0
  // CHECK:  %[[SEL0:.+]] = torch.aten.select.int %[[TENSOR]], %[[DIM]], %[[A0]]
  // CHECK:  %[[ITEM0:.+]] = torch.aten.item %[[SEL0]]
  // CHECK:  %[[LIST:.+]] = torch.prim.ListConstruct %[[ITEM0]]
  // CHECK:  %[[FALSE:.+]] = torch.constant.bool false
  // CHECK:  %[[NONE:.+]] = torch.constant.none
  // CHECK:  %[[SUM:.+]] = torch.aten.mean.dim %arg0, %[[LIST]], %[[FALSE]], %[[NONE]]
  // CHECK:  return %[[SUM]]
  %cst = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %cst) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}
// -----

// CHECK-LABEL: @test_reduce_mean_one_axesattr_dropdims_example
func.func @test_reduce_mean_one_axesattr_dropdims_example(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[INT1]]
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[MEAN:.+]] = torch.aten.mean.dim %arg0, %[[LIST]], %[[FALSE]], %[[NONE]]
  // CHECK: return %[[MEAN]]
  %0 = torch.operator "onnx.ReduceMean"(%arg0) {torch.onnx.keepdims = 0 : si64, torch.onnx.axes = [1 : si64]} : (!torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_mean_all_dims_keepdims_example
func.func @test_reduce_mean_all_dims_keepdims_example(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[INT0]], %[[INT1]], %[[INT2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[KEEPDIM:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.mean.dim %arg0, %[[DIMS]], %[[KEEPDIM]], %[[NONE]] : !torch.vtensor<[3,2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_mean_do_not_keepdims_example
func.func @test_reduce_mean_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 2
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[INT2]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[KEEPDIM:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.mean.dim %arg0, %[[DIMS]], %[[KEEPDIM]], %[[NONE]] : !torch.vtensor<[3,2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_mean_keepdims_example
func.func @test_reduce_mean_keepdims_example(%arg0: !torch.vtensor<[3,2,4,5,1,6,7],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,1,4,1,1,6,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT6:.+]] = torch.constant.int 6
  // CHECK: %[[DIMS:.+]] = torch.prim.ListConstruct %[[INT1]], %[[INT3]], %[[INT6]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[KEEPDIM:.+]] = torch.constant.bool true
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: torch.aten.mean.dim %arg0, %[[DIMS]], %[[KEEPDIM]], %[[NONE]] : !torch.vtensor<[3,2,4,5,1,6,7],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,4,1,1,6,1],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,4,5,1,6,7],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,1,4,1,1,6,1],f32>
  return %0 : !torch.vtensor<[3,1,4,1,1,6,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_empty_set_fp
func.func @test_reduce_min_empty_set_fp(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INF:.+]] = torch.constant.float 0x7FF0000000000000
  // CHECK-DAG: %[[INT2:.+]] = torch.constant.int 2
  // CHECK-DAG: %[[INT1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[INT4:.+]] = torch.constant.int 4
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT1]], %[[INT4]]
  // CHECK-DAG: %[[FULL:.+]] = torch.aten.full %[[LIST]], %[[INF]], %[[NONE]], %[[NONE]], %[[NONE]]
  // CHECK: return %[[FULL]]
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32>
  return %0 : !torch.vtensor<[2,1,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_empty_set_int
func.func @test_reduce_min_empty_set_int(%arg0: !torch.vtensor<[2,0,4],si32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],si32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INF:.+]] = torch.constant.int 2147483647
  // CHECK-DAG: %[[INT2:.+]] = torch.constant.int 2
  // CHECK-DAG: %[[INT1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[INT4:.+]] = torch.constant.int 4
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT1]], %[[INT4]]
  // CHECK-DAG: %[[FULL:.+]] = torch.aten.full %[[LIST]], %[[INF]], %[[NONE]], %[[NONE]], %[[NONE]]
  // CHECK: return %[[FULL]]
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],si32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],si32>
  return %0 : !torch.vtensor<[2,1,4],si32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_bool_inputs
func.func @test_reduce_min_bool_inputs(%arg0: !torch.vtensor<[4,2],i1>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[IDX:.+]] = torch.constant.int 0
  // CHECK: %[[SZ:.+]] = torch.constant.int 0
  // CHECK: %[[SEL:.+]] = torch.aten.select.int %arg1, %[[IDX]], %[[SZ]]
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SEL]]
  // CHECK: %[[DIM:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[C0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[ITEM]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[DIM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.+]] = torch.aten.add.int %[[ITEM]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LST:.+]] = torch.prim.ListConstruct %6 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[AMIN:.+]] = torch.aten.amin %arg0, %[[LST]], %[[TRUE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,1],i1>
  // CHECK: return %[[AMIN]] : !torch.vtensor<[4,1],i1>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[4,2],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1>
  return %0 : !torch.vtensor<[4,1],i1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_bool_inputs_nokeepdims
func.func @test_reduce_min_bool_inputs_nokeepdims(%arg0: !torch.vtensor<[4,2],i1>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[IDX:.+]] = torch.constant.int 0
  // CHECK: %[[SZ:.+]] = torch.constant.int 0
  // CHECK: %[[SEL:.+]] = torch.aten.select.int %arg1, %[[IDX]], %[[SZ]]
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[SEL]]
  // CHECK: %[[DIM:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[C0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[ITEM]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[DIM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.+]] = torch.aten.add.int %[[ITEM]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LST:.+]] = torch.prim.ListConstruct %6 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[AMIN:.+]] = torch.aten.amin %arg0, %[[LST]], %[[FALSE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],i1>
  // CHECK: return %[[AMIN]] : !torch.vtensor<[4],i1>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[4,2],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_all_dims_default
func.func @test_reduce_min_all_dims_default(%arg0: !torch.vtensor<[4,2],i1>) -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[I0:.+]] = torch.constant.int 0
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[RANK:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[C0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[I0]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[A0:.+]] = torch.aten.add.int %[[I0]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[I1]], %[[C0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[A1:.+]] = torch.aten.add.int %[[I1]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[A0]], %[[A1]]
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[MIN:.+]] = torch.aten.amin %arg0, %[[LIST]], %[[FALSE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[MIN]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.ReduceMin"(%arg0) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[4,2],i1>) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

func.func @test_reduce_min_attr(%arg0: !torch.vtensor<[4,2],i1>) -> !torch.vtensor<[4],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[DIM:.+]] = torch.aten.dim %arg0 : !torch.vtensor<[4,2],i1> -> !torch.int
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[LT:.+]] = torch.aten.lt.int %[[INT1]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.+]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.+]] = torch.aten.mul.int %[[BOOL]], %[[DIM]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.+]] = torch.aten.add.int %[[INT1]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[ADD]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[AMIN:.+]] = torch.aten.amin %arg0, %[[LIST]], %[[FALSE]] : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],i1>
  // CHECK: return %[[AMIN]]
  %0 = torch.operator "onnx.ReduceMin"(%arg0) {torch.onnx.keepdims = 0 : si64, torch.onnx.axes=[1 : si64]} : (!torch.vtensor<[4,2],i1>) -> !torch.vtensor<[4],i1>
  return %0 : !torch.vtensor<[4],i1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_prod_default_axes_keepdims_random
func.func @test_reduce_prod_default_axes_keepdims_random(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[RANK:.*]] = torch.aten.dim %arg0 : !torch.vtensor<[3,2,2],f32> -> !torch.int
  // CHECK: %[[LT:.*]] = torch.aten.lt.int %[[INT0_0]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL:.*]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL:.*]] = torch.aten.mul.int %[[BOOL]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[INT0_0]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LT_0:.*]] = torch.aten.lt.int %[[INT1]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL_0:.*]] = torch.aten.Int.bool %[[LT_0]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL_0:.*]] = torch.aten.mul.int %[[BOOL_0]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD_0:.*]] = torch.aten.add.int %[[INT1]], %[[MUL_0]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[LT_1:.*]] = torch.aten.lt.int %[[INT2]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[BOOL_1:.*]] = torch.aten.Int.bool %[[LT_1]] : !torch.bool -> !torch.int
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.int %[[BOOL_1]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.int %[[INT2]], %[[MUL_1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[PROD_0:.*]] = torch.aten.prod.dim_int %arg0, %[[ADD]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[PROD_1:.*]] = torch.aten.prod.dim_int %[[PROD_0]], %[[ADD_0]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[PROD_2:.*]] = torch.aten.prod.dim_int %[[PROD_1]], %[[ADD_1]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  // CHECK: return %[[PROD_2]] : !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceProd"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_prod_keepdims_random
func.func @test_reduce_prod_keepdims_random(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT0_0:.*]] = torch.constant.int 0
// CHECK: %[[SELECT:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
// CHECK: %[[ITEM:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
// CHECK: %[[DIM:.*]] = torch.aten.dim %arg0 : !torch.vtensor<[3,2,2],f32> -> !torch.int
// CHECK: %[[LT:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
// CHECK: %[[BOOL:.*]] = torch.aten.Int.bool %[[LT]] : !torch.bool -> !torch.int
// CHECK: %[[MUL:.*]] = torch.aten.mul.int %[[BOOL]], %[[DIM]] : !torch.int, !torch.int -> !torch.int
// CHECK: %[[ADD:.*]] = torch.aten.add.int %[[ITEM]], %[[MUL]] : !torch.int, !torch.int -> !torch.int
// CHECK: %[[BOOL:.*]] = torch.constant.bool true
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[PROD:.*]] = torch.aten.prod.dim_int %arg0, %[[ADD]], %[[BOOL]], %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
// CHECK: return %[[PROD]] : !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceProd"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_sinh
func.func @test_sinh_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 9 : si64} {
  // CHECK: torch.aten.sinh %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sinh"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL:   func.func @test_split_variable_parts_2d_opset18(
// CHECK-SAME:                                                    %[[VAL_INPUT:.*]]: !torch.vtensor<[2,6],f32>,
// CHECK-SAME:                                                    %[[VAL_SPLIT:.*]]: !torch.vtensor<[2],si64>
// CHECK:           %[[VAL_SPLIT_LIST:.*]] = torch.prim.tolist(%[[VAL_SPLIT]]) : !torch.vtensor<[2],si64> -> !torch.list<int>
// CHECK:           %[[VAL_AXIS:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_RESULT_LIST:.*]] = torch.aten.split_with_sizes %[[VAL_INPUT]], %[[VAL_SPLIT_LIST]], %[[VAL_AXIS]] : !torch.vtensor<[2,6],f32>, !torch.list<int>, !torch.int -> !torch.list<vtensor<[2,?],f32>>
// CHECK:           %[[VAL_VARIADIC_RETURN_VALUE:.*]]:2 = torch.prim.ListUnpack %[[VAL_RESULT_LIST]] : !torch.list<vtensor<[2,?],f32>> -> !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,4],f32>
// CHECK:           return %[[VAL_VARIADIC_RETURN_VALUE]]#0, %[[VAL_VARIADIC_RETURN_VALUE]]#1 : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,4],f32>
func.func @test_split_variable_parts_2d_opset18(%arg0: !torch.vtensor<[2,6],f32>, %arg1: !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,4],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.Split"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,6],f32>, !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_split_2d_uneven_split_opset18(
// CHECK-SAME: %[[INPUT_TENSOR:.*]]: !torch.vtensor<[2,8],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,2],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK-DAG: %[[DIM:.+]] = torch.constant.int 1
// CHECK-DAG: %[[SPLITS:.+]] = torch.constant.int 3
// CHECK-DAG: %[[ONE:.+]] = torch.constant.int 1
// CHECK-DAG: %[[ZERO:.+]] = torch.constant.int 0
// CHECK-DAG: %[[SZ1:.+]] = torch.aten.size.int %arg0, %[[DIM]]
// CHECK-DAG: %[[ADD:.+]] = torch.aten.add.int %[[SZ1]], %[[SPLITS]]
// CHECK-DAG: %[[SUB:.+]] = torch.aten.sub.int %[[ADD]], %[[ONE]]
// CHECK-DAG: %[[SLICESZ:.+]] = torch.aten.floordiv.int %[[SUB]], %[[SPLITS]]
// CHECK-DAG: %[[START1:.+]] = torch.aten.add.int %[[ZERO]], %[[SLICESZ]] : !torch.int, !torch.int -> !torch.int
// CHECK-DAG: %[[SLICE0:.+]] = torch.aten.slice.Tensor %arg0, %[[DIM]], %[[ZERO]], %[[START1]], %[[ONE]] : !torch.vtensor<[2,8],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK-DAG: %[[START2:.+]] = torch.aten.add.int %[[START1]], %[[SLICESZ]] : !torch.int, !torch.int -> !torch.int
// CHECK-DAG: %[[SLICE1:.+]] = torch.aten.slice.Tensor %arg0, %[[DIM]], %[[START1]], %[[START2]], %[[ONE]] : !torch.vtensor<[2,8],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK-DAG: %[[SLICE2:.+]] = torch.aten.slice.Tensor %arg0, %[[DIM]], %[[START2]], %[[SZ1]], %[[ONE]] : !torch.vtensor<[2,8],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,2],f32>
// CHECK: return %[[SLICE0]], %[[SLICE1]], %[[SLICE2]]
func.func @test_split_2d_uneven_split_opset18(%arg0: !torch.vtensor<[2,8],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,2],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:3 = torch.operator "onnx.Split"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.num_outputs = 3 : si64} : (!torch.vtensor<[2,8],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,2],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_tan
func.func @test_tan(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TAN:.+]] = torch.aten.tan %arg0
  %0 = torch.operator "onnx.Tan"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_transpose_default
func.func @test_transpose_default(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,3,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[I2:.+]] = torch.constant.int 2
  // CHECK: %[[TRANSPOSE:.+]] = torch.aten.transpose.int %arg0, %[[I0]], %[[I2]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,3,2],f32>
  %0 = torch.operator "onnx.Transpose"(%arg0) : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,3,2],f32>

  // CHECK: return %[[TRANSPOSE]]
  return %0 : !torch.vtensor<[4,3,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_transpose_all_permutations_4
func.func @test_transpose_all_permutations_4(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,2,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG:  %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG:  %[[I2:.+]] = torch.constant.int 2
  // CHECK:  %[[TRANSPOSE0:.+]] = torch.aten.transpose.int %arg0, %[[I0]], %[[I2]] : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,3,2],f32>
  // CHECK-DAG:  %[[I1:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[I2:.+]] = torch.constant.int 2
  // CHECK:  %[[TRANSPOSE1:.+]] = torch.aten.transpose.int %[[TRANSPOSE0]], %[[I1]], %[[I2]] : !torch.vtensor<[4,3,2],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,2,3],f32>
  %0 = torch.operator "onnx.Transpose"(%arg0) {torch.onnx.perm = [2 : si64, 0 : si64, 1 : si64]} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,2,3],f32>

  // CHECK:  return %[[TRANSPOSE1]]
  return %0 : !torch.vtensor<[4,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_transpose_dynamic
func.func @test_transpose_dynamic(%arg0: !torch.vtensor<[?,32,5,128],f32>) -> !torch.vtensor<[?,5,32,128],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG:  %[[I1:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[I2:.+]] = torch.constant.int 2
  // CHECK:  %[[TRANSPOSE:.+]] = torch.aten.transpose.int %arg0, %[[I1]], %[[I2]] : !torch.vtensor<[?,32,5,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,5,32,128],f32>
  %0 = torch.operator "onnx.Transpose"(%arg0) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64, 3 : si64]} : (!torch.vtensor<[?,32,5,128],f32>) -> !torch.vtensor<[?,5,32,128],f32>
  return %0 : !torch.vtensor<[?,5,32,128],f32>
}


// -----

// CHECK-LABEL: func.func @test_slice
func.func @test_slice(%arg0: !torch.vtensor<[20,10,5],f32>, %arg1: !torch.vtensor<[2],si64>, %arg2: !torch.vtensor<[2],si64>, %arg3: !torch.vtensor<[2],si64>, %arg4: !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,10,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  //CHECK: %[[INDEX_TO_GRAB:.*]] = torch.constant.int 0

  //CHECK: %[[CONST_0:.*]] = torch.constant.int 0
  //CHECK: %[[ZERO_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_0:.*]] : !torch.int -> !torch.vtensor<[1],si64>
  //CHECK: %[[STARTS_INDEX_VEC_0:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[STARTS_ELEMENT_0:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: %[[ENDS_INDEX_VEC_0:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[ENDS_ELEMENT_0:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: %[[AXES_INDEX_VEC_0:.*]] = torch.aten.index_select %arg3, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[AXES_ELEMENT_0:.*]] = torch.aten.item %[[AXES_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: %[[STEPS_INDEX_VEC_0:.*]] = torch.aten.index_select %arg4, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[STEPS_ELEMENT_0:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: %[[SLICE_0:.*]] = torch.aten.slice.Tensor %arg0, %[[AXES_ELEMENT_0:.*]], %[[STARTS_ELEMENT_0:.*]], %[[ENDS_ELEMENT_0:.*]], %[[STEPS_ELEMENT_0:.*]] : !torch.vtensor<[20,10,5],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,10,5],f32>

  //CHECK: %[[CONST_1:.*]] = torch.constant.int 1
  //CHECK: %[[ONE_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_1:.*]] : !torch.int -> !torch.vtensor<[1],si64>
  //CHECK: %[[STARTS_INDEX_VEC_1:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[STARTS_ELEMENT_1:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: %[[ENDS_INDEX_VEC_1:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[ENDS_ELEMENT_1:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: %[[AXES_INDEX_VEC_1:.*]] = torch.aten.index_select %arg3, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[AXES_ELEMENT_2:.*]] = torch.aten.item %[[AXES_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: %[[STEPS_INDEX_VEC_1:.*]] = torch.aten.index_select %arg4, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  //CHECK: %[[STEPS_ELEMENT_1:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
  //CHECK: torch.aten.slice.Tensor %[[SLICE_0:.*]], %[[AXES_ELEMENT_1:.*]], %[[STARTS_ELEMENT_1:.*]], %[[ENDS_ELEMENT_1:.*]], %[[STEPS_ELEMENT_1:.*]] : !torch.vtensor<[?,10,5],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,10,5],f32>
  %0 = torch.operator "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[20,10,5],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,10,5],f32>
  return %0 : !torch.vtensor<[3,10,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_slice_default_axes_and_slices
func.func @test_slice_default_axes_and_slices(%arg0: !torch.vtensor<[20,10,5],f32>, %arg1: !torch.vtensor<[3],si64>, %arg2: !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    //CHECK: %[[NONE_1:.*]] = torch.constant.none
    //CHECK: %[[AXES_DEFAULT_SIZE:.*]] = torch.constant.int 3
    //CHECK: %[[DEFAULT_SIZE_INPUT:.*]] = torch.prim.ListConstruct %[[DEFAULT_SIZE_AMOUNT:.*]] : (!torch.int) -> !torch.list<int>
    //CHECK: %[[DEFAULT_SIZES:.*]] = torch.aten.ones %[[DEFAULT_SIZE_INPUT:.*]], %[[NONE_2:.*]], %[[NONE_2:.*]], %[[NONE_2:.*]], %[[NONE_2:.*]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3],si64>
    //CHECK: %[[INDEX_TO_GRAB:.*]] = torch.constant.int 0

    //CHECK: %[[CONST_0:.*]] = torch.constant.int 0
    //CHECK: %[[ZERO_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_0:.*]] : !torch.int -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_INDEX_VEC_0:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_ELEMENT_0:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[ENDS_INDEX_VEC_0:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[ENDS_ELEMENT_0:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[STEPS_INDEX_VEC_0:.*]] = torch.aten.index_select %[[DEFAULT_SIZES:.*]], %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STEPS_ELEMENT_0:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[SLICE_0:.*]] = torch.aten.slice.Tensor %arg0, %[[CONST_0:.*]], %[[STARTS_ELEMENT_0:.*]], %[[ENDS_ELEMENT_0:.*]], %[[STEPS_ELEMENT_0:.*]] : !torch.vtensor<[20,10,5],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[20,10,?],f32>

    //CHECK: %[[CONST_1:.*]] = torch.constant.int 1
    //CHECK: %[[ONE_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_1:.*]] : !torch.int -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_INDEX_VEC_1:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_ELEMENT_1:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[ENDS_INDEX_VEC_1:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[ENDS_ELEMENT_1:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[STEPS_INDEX_VEC_1:.*]] = torch.aten.index_select %[[DEFAULT_SIZES:.*]], %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STEPS_ELEMENT_1:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[TWO_INDEX_VEC:.*]] = torch.aten.slice.Tensor %[[SLICE_0:.*]], %[[CONST_1:.*]], %[[STARTS_ELEMENT_1:.*]], %[[ENDS_ELEMENT_1:.*]], %[[STEPS_ELEMENT_1:.*]] : !torch.vtensor<[20,10,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[20,10,?],f32>

    //CHECK: %[[CONST_2:.*]] = torch.constant.int 2
    //CHECK: %[[TWO_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_2:.*]] : !torch.int -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_INDEX_VEC_2:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[TWO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_ELEMENT_2:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_2:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[ENDS_INDEX_VEC_2:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[TWO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[ENDS_ELEMENT_2:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_2:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[STEPS_INDEX_VEC_2:.*]] = torch.aten.index_select %[[DEFAULT_SIZES:.*]], %[[INDEX_TO_GRAB:.*]], %[[TWO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STEPS_ELEMENT_2:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_2:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: torch.aten.slice.Tensor %[[TWO_INDEX_VEC:.*]], %[[CONST_2:.*]], %[[STARTS_ELEMENT_2:.*]], %[[ENDS_ELEMENT_2:.*]], %[[STEPS_ELEMENT_2:.*]] : !torch.vtensor<[20,10,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[20,10,1],f32>
  %0 = torch.operator "onnx.Slice"(%arg0, %arg1, %arg2) : (!torch.vtensor<[20,10,5],f32>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,1],f32>
  return %0 : !torch.vtensor<[20,10,1],f32>
}

// -----

// CHECK-LABEL: @test_slice_default_axes_and_steps
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[20,10,5],f32>,
// CHECK-SAME:    %[[ARG1:.*]]: !torch.vtensor<[1],si64>,
// CHECK-SAME:    %[[ARG2:.*]]: !torch.vtensor<[1],si64>

// CHECK: %[[ZERO0:.*]] = torch.constant.int 0
// CHECK-NEXT: %[[ZERO1:.*]] = torch.constant.int 0
// CHECK-NEXT: %[[SCALAR:.*]] = torch.prim.NumToTensor.Scalar %[[ZERO1]] : !torch.int -> !torch.vtensor<[1],si64>
// CHECK-NEXT: %[[SELECT0:.*]] = torch.aten.index_select %[[ARG1]], %[[ZERO0]], %[[SCALAR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
// CHECK-NEXT: %[[ITEM0:.*]] = torch.aten.item %[[SELECT0]] : !torch.vtensor<[1],si64> -> !torch.int
// CHECK-NEXT: %[[SELECT1:.*]] = torch.aten.index_select %[[ARG2]], %[[ZERO0]], %[[SCALAR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
// CHECK-NEXT: %[[ITEM1:.*]] = torch.aten.item %[[SELECT1]] : !torch.vtensor<[1],si64> -> !torch.int
// CHECK-NEXT: %[[SELECT3:.*]] = torch.aten.index_select %{{.*}}, %[[ZERO0]], %[[SCALAR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
// CHECK-NEXT: %[[ITEM3:.*]] = torch.aten.item %[[SELECT3]] : !torch.vtensor<[1],si64> -> !torch.int
// CHECK: torch.aten.slice.Tensor %[[ARG0]], %[[ZERO1]], %[[ITEM0]], %[[ITEM1]], %[[ITEM3]] : !torch.vtensor<[20,10,5],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[20,10,1],f32>

func.func @test_slice_default_axes_and_steps(%arg0: !torch.vtensor<[20,10,5],f32>, %arg1: !torch.vtensor<[1],si64>, %arg2: !torch.vtensor<[1],si64>) -> !torch.vtensor<[20,10,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  %0 = torch.operator "onnx.Slice"(%arg0, %arg1, %arg2) : (!torch.vtensor<[20,10,5],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[20,10,1],f32>
  return %0 : !torch.vtensor<[20,10,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_slice_default_steps
func.func @test_slice_default_steps(%arg0: !torch.vtensor<[20,10,5],f32>, %arg1: !torch.vtensor<[3],si64>, %arg2: !torch.vtensor<[3],si64>, %arg3: !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    //CHECK: %[[NONE:.*]] = torch.constant.none
    //CHECK: %[[DEFAULT_SIZE_AMOUNT:.*]] = torch.constant.int 3
    //CHECK: %[[DEFAULT_SIZE_INPUT:.*]] = torch.prim.ListConstruct %[[DEFAULT_SIZE_AMOUNT:.*]] : (!torch.int) -> !torch.list<int>
    //CHECK: %[[DEFAULT_SIZES:.*]] = torch.aten.ones %[[DEFAULT_SIZE_INPUT:.*]], %[[NONE:.*]], %[[NONE:.*]], %[[NONE:.*]], %[[NONE:.*]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3],si64>
    //CHECK: %[[INDEX_TO_GRAB:.*]] = torch.constant.int 0

    //CHECK: %[[CONST_0:.*]] = torch.constant.int 0
    //CHECK: %[[ZERO_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_0:.*]] : !torch.int -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_INDEX_VEC_0:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_ELEMENT_0:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[ENDS_INDEX_VEC_0:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[ENDS_ELEMENT_0:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[AXES_INDEX_VEC_0:.*]] = torch.aten.index_select %arg3, %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[AXES_ELEMENT_0:.*]] = torch.aten.item %[[AXES_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[STEPS_INDEX_VEC_0:.*]] = torch.aten.index_select %[[DEFAULT_SIZES:.*]], %[[INDEX_TO_GRAB:.*]], %[[ZERO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STEPS_ELEMENT_0:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_0:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[SLICE_0:.*]] = torch.aten.slice.Tensor %arg0, %[[AXES_ELEMENT_0:.*]], %[[STARTS_ELEMENT_0:.*]], %[[ENDS_ELEMENT_0:.*]], %[[STEPS_ELEMENT_0:.*]] : !torch.vtensor<[20,10,5],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[20,10,?],f32>

    //CHECK: %[[CONST_1:.*]] = torch.constant.int 1
    //CHECK: %[[ONE_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_1:.*]] : !torch.int -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_INDEX_VEC_1:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_ELEMENT_1:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[ENDS_INDEX_VEC_1:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[ENDS_ELEMENT_1:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[AXES_INDEX_VEC_1:.*]] = torch.aten.index_select %arg3, %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[AXES_ELEMENT_1:.*]] = torch.aten.item %[[AXES_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[STEPS_INDEX_VEC_1:.*]] = torch.aten.index_select %[[DEFAULT_SIZES:.*]], %[[INDEX_TO_GRAB:.*]], %[[ONE_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STEPS_ELEMENT_1:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_1:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[TWO_INDEX_VEC:.*]] = torch.aten.slice.Tensor %[[SLICE_0:.*]], %[[AXES_ELEMENT_1:.*]], %[[STARTS_ELEMENT_1:.*]], %[[ENDS_ELEMENT_1:.*]], %[[STEPS_ELEMENT_1:.*]] : !torch.vtensor<[20,10,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[20,10,?],f32>

    //CHECK: %[[CONST_1:.*]] = torch.constant.int 2
    //CHECK: %[[TWO_INDEX_VEC:.*]] = torch.prim.NumToTensor.Scalar %[[CONST_1:.*]] : !torch.int -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_INDEX_VEC_2:.*]] = torch.aten.index_select %arg1, %[[INDEX_TO_GRAB:.*]], %[[TWO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STARTS_ELEMENT_2:.*]] = torch.aten.item %[[STARTS_INDEX_VEC_2:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[ENDS_INDEX_VEC_2:.*]] = torch.aten.index_select %arg2, %[[INDEX_TO_GRAB:.*]], %[[TWO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[ENDS_ELEMENT_2:.*]] = torch.aten.item %[[ENDS_INDEX_VEC_2:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[AXES_INDEX_VEC_2:.*]] = torch.aten.index_select %arg3, %[[INDEX_TO_GRAB:.*]], %[[TWO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[AXES_ELEMENT_2:.*]] = torch.aten.item %[[AXES_INDEX_VEC_2:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: %[[STEPS_INDEX_VEC_2:.*]] = torch.aten.index_select %[[DEFAULT_SIZES:.*]], %[[INDEX_TO_GRAB:.*]], %[[TWO_INDEX_VEC:.*]] : !torch.vtensor<[3],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    //CHECK: %[[STEPS_ELEMENT_2:.*]] = torch.aten.item %[[STEPS_INDEX_VEC_2:.*]] : !torch.vtensor<[1],si64> -> !torch.int
    //CHECK: torch.aten.slice.Tensor %[[TWO_INDEX_VEC:.*]], %[[AXES_ELEMENT_2:.*]], %[[STARTS_ELEMENT_2:.*]], %[[ENDS_ELEMENT_2:.*]], %[[STEPS_ELEMENT_2:.*]] : !torch.vtensor<[20,10,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[20,10,1],f32>
  %0 = torch.operator "onnx.Slice"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[20,10,5],f32>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,1],f32>
  return %0 : !torch.vtensor<[20,10,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_reshape_negative_dim
func.func @test_reshape_negative_dim(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,6,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT6:.+]] = torch.constant.int 6
  // CHECK: %[[INT2_0:.+]] = torch.constant.int 2
  // CHECK: %[[RESULT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT6]], %[[INT2_0]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.reshape %arg0, %[[RESULT_SHAPE]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,6,2],f32>
  %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,6,2],f32>
  return %0 : !torch.vtensor<[2,6,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reshape_negative_extended_dims
func.func @test_reshape_negative_extended_dims(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,2,3,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT4:.+]] = torch.constant.int 4
  // CHECK: %[[RESULT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT1]], %[[INT2]], %[[INT3]], %[[INT4]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.reshape %arg0, %[[RESULT_SHAPE]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,3,4],f32>
  %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,2,3,4],f32>
  return %0 : !torch.vtensor<[1,2,3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_reshape_one_dim
func.func @test_reshape_one_dim(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[24],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT24:.+]] = torch.constant.int 24
  // CHECK: %[[RESULT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT24]] : (!torch.int) -> !torch.list<int>
  // CHECK: torch.aten.reshape %arg0, %[[RESULT_SHAPE]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[24],f32>
  %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[24],f32>
  return %0 : !torch.vtensor<[24],f32>
}

// -----

// CHECK-LABEL: func.func @test_reshape_reduced_dims
func.func @test_reshape_reduced_dims(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,12],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT12:.+]] = torch.constant.int 12
  // CHECK: %[[RESULT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT12]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.reshape %arg0, %[[RESULT_SHAPE]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,12],f32>
  %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,12],f32>
  return %0 : !torch.vtensor<[2,12],f32>
}

// -----

// CHECK-LABEL: func.func @test_reshape_reordered_all_dims
func.func @test_reshape_reordered_all_dims(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[4,2,3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT4:.+]] = torch.constant.int 4
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[RESULT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT4]], %[[INT2]], %[[INT3]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.reshape %arg0, %[[RESULT_SHAPE]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[4,2,3],f32>
  %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[4,2,3],f32>
  return %0 : !torch.vtensor<[4,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_reshape_zero_and_negative_dim
func.func @test_reshape_zero_and_negative_dim(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[2,3,1,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[INT4:.+]] = torch.constant.int 4
  // CHECK: %[[RESULT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT3]], %[[INT1]], %[[INT4]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.reshape %arg0, %[[RESULT_SHAPE]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,3,1,4],f32>
  %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[2,3,1,4],f32>
  return %0 : !torch.vtensor<[2,3,1,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_range_float64_type
 func.func @test_range_float64_type(%arg0: !torch.vtensor<[],f64>, %arg1: !torch.vtensor<[],f64>, %arg2: !torch.vtensor<[],f64>) -> !torch.vtensor<[2],f64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[NONE:.*]] torch.constant.none
    // CHECK: torch.aten.item %arg0 : !torch.vtensor<[],f64> -> !torch.float
    // CHECK: torch.aten.item %arg1 : !torch.vtensor<[],f64> -> !torch.float
    // CHECK: torch.aten.item %arg2 : !torch.vtensor<[],f64> -> !torch.float
    // CHECK: torch.aten.arange.start_step %0, %1, %2, %none, %none, %none, %none : !torch.float, !torch.float, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],f64>
    %0 = torch.operator "onnx.Range"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],f64>, !torch.vtensor<[],f64>, !torch.vtensor<[],f64>) -> !torch.vtensor<[2],f64>
    return %0 : !torch.vtensor<[2],f64>
  }

// -----

// CHECK-LABEL: func.func @test_range_float32_type
 func.func @test_range_float32_type(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[2],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[NONE:.*]] torch.constant.none
    // CHECK: torch.aten.item %arg0 : !torch.vtensor<[],f32> -> !torch.float
    // CHECK: torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
    // CHECK: torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
    // CHECK: torch.aten.arange.start_step %0, %1, %2, %none, %none, %none, %none : !torch.float, !torch.float, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],f32>
    %0 = torch.operator "onnx.Range"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[2],f32>
    return %0 : !torch.vtensor<[2],f32>
  }

// -----

// CHECK-LABEL: func.func @test_range_int64_type
  func.func @test_range_int64_type(%arg0: !torch.vtensor<[],si64>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[],si64>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[NONE:.*]] torch.constant.none
    // CHECK: torch.aten.item %arg0 : !torch.vtensor<[],si64> -> !torch.int
    // CHECK: torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
    // CHECK: torch.aten.item %arg2 : !torch.vtensor<[],si64> -> !torch.int
    // CHECK: torch.aten.arange.start_step %0, %1, %2, %none, %none, %none, %none : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si64>
    %0 = torch.operator "onnx.Range"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[2],si64>
    return %0 : !torch.vtensor<[2],si64>
  }

// -----

// CHECK-LABEL: func.func @test_range_int32_type
  func.func @test_range_int32_type(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>) -> !torch.vtensor<[2],si32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[NONE:.*]] torch.constant.none
    // CHECK: torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
    // CHECK: torch.aten.item %arg1 : !torch.vtensor<[],si32> -> !torch.int
    // CHECK: torch.aten.item %arg2 : !torch.vtensor<[],si32> -> !torch.int
    // CHECK: torch.aten.arange.start_step %0, %1, %2, %none, %none, %none, %none : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si32>
    %0 = torch.operator "onnx.Range"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[2],si32>
    return %0 : !torch.vtensor<[2],si32>
  }

// -----

 // CHECK-LABEL: func.func @test_range_int16_type
  func.func @test_range_int16_type(%arg0: !torch.vtensor<[],si16>, %arg1: !torch.vtensor<[],si16>, %arg2: !torch.vtensor<[],si16>) -> !torch.vtensor<[2],si16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[NONE:.*]] torch.constant.none
    // CHECK: torch.aten.item %arg0 : !torch.vtensor<[],si16> -> !torch.int
    // CHECK: torch.aten.item %arg1 : !torch.vtensor<[],si16> -> !torch.int
    // CHECK: torch.aten.item %arg2 : !torch.vtensor<[],si16> -> !torch.int
    // CHECK: torch.aten.arange.start_step %0, %1, %2, %none, %none, %none, %none : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si16>
    %0 = torch.operator "onnx.Range"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],si16>, !torch.vtensor<[],si16>, !torch.vtensor<[],si16>) -> !torch.vtensor<[2],si16>
    return %0 : !torch.vtensor<[2],si16>
  }

// -----

// CHECK-LABEL : func.func @test_top_k
func.func @test_top_k(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
  // CHECK: %[[AXIS:.*]] = torch.constant.int 1
  // CHECK: %[[LARGEST:.*]] = torch.constant.bool true
  // CHECK: %[[SORTED:.*]] = torch.constant.bool true
  // CHECK: %[[K:.*]] = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.topk %arg0, %[[K]], %[[AXIS]], %[[LARGEST]], %[[SORTED]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
  %0:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
  return %0#0, %0#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
}

// -----

// CHECK-LABEL: func.func @test_top_k_smallest
func.func @test_top_k_smallest(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[AXIS:.*]] = torch.constant.int 1
  // CHECK: %[[LARGEST:.*]] = torch.constant.bool false
  // CHECK: %[[SORTED:.*]] = torch.constant.bool true
  // CHECK: %[[K:.*]] = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.topk %arg0, %[[K]], %[[AXIS]], %[[LARGEST]], %[[SORTED]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
  %0:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = 1 : si64, torch.onnx.largest = 0 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
  return %0#0, %0#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
}

// -----

// CHECK-LABEL: func.func @test_top_k_negative_axis
func.func @test_top_k_negative_axis(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
  // CHECK: %[[AXIS:.*]] = torch.constant.int 1
  // CHECK: %[[LARGEST:.*]] = torch.constant.bool true
  // CHECK: %[[SORTED:.*]] = torch.constant.bool true
  // CHECK: %[[K:.*]] = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.topk %arg0, %[[K]], %[[AXIS]], %[[LARGEST]], %[[SORTED]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
  %0:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
  return %0#0, %0#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
}

// -----

// CHECK-LABEL: func.func @test_tile
func.func @test_tile(%arg0: !torch.vtensor<[2, 3, 4],f32>, %arg1: !torch.vtensor<[3], si64>) -> !torch.vtensor<[2,12,4],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: %[[EXTRACT_0:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ELE_0:.*]] = torch.aten.item %[[EXTRACT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[EXTRACT_1:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT1]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ELE_1:.*]] = torch.aten.item %[[EXTRACT_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[EXTRACT_2:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT2]] : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ELE_2:.*]] = torch.aten.item %[[EXTRACT_2]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[DIM_LIST:.*]] = torch.prim.ListConstruct %[[ELE_0]], %[[ELE_1]], %[[ELE_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %7 = torch.aten.tile %arg0, %[[DIM_LIST]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,12,4],f32>
  %0 = torch.operator "onnx.Tile"(%arg0, %arg1) : (!torch.vtensor<[2, 3, 4],f32>, !torch.vtensor<[3], si64>) -> !torch.vtensor<[2, 12, 4],f32>
  return %0 : !torch.vtensor<[2, 12, 4],f32>
}

// -----

// CHECK-LABEL: func.func @test_sign
func.func @test_sign(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.sign %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Sign"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_size
func.func @test_size(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],si32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 9 : si64} {
  // CHECK-DAG  %[[INT0:.+]] = torch.constant.int 0
  // CHECK-DAG  %[[INT1:.+]] = torch.constant.int 1
  // CHECK-DAG  %[[INT2:.+]] = torch.constant.int 2
  // CHECK-DAG  %[[D0:.+]] = torch.aten.size.int %arg0, %[[INT0]]
  // CHECK-DAG  %[[D1:.+]] = torch.aten.size.int %arg0, %[[INT1]]
  // CHECK-DAG  %[[D2:.+]] = torch.aten.size.int %arg0, %[[INT2]]
  // CHECK-DAG  %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG  %[[MUL0:.+]] = torch.aten.mul.int %[[D0]], %[[D1]]
  // CHECK-DAG  %[[MUL1:.+]] = torch.aten.mul.int %[[MUL0]], %[[D3]]
  // CHECK-DAG  %[[TENSOR:.+]] = torch.aten.tensor.int %[[MUL1]], %[[NONE]], %[[NONE]], %[[FALSE]]
  // CHECK      return %[[TENSOR]]
  %0 = torch.operator "onnx.Size"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],si32>
  return %0 : !torch.vtensor<[],si32>
}

// -----

// CHECK-LABEL: func.func @test_softplus
func.func @test_softplus(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[EXP:.*]] = torch.aten.exp %arg0 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  // CHECK: torch.aten.log1p %[[EXP]] : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Softplus"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @test_tril
func.func @test_tril(%arg0: !torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.constant.int 0
  // CHECK: torch.aten.tril %arg0, %[[DIAGONAL]] : !torch.vtensor<[4,5],si64>, !torch.int -> !torch.vtensor<[4,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0) {torch.onnx.upper = 0 : si64} : (!torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64>
  return %0 : !torch.vtensor<[4,5],si64>
}

// -----

// CHECK-LABEL: func.func @test_tril_neg
func.func @test_tril_neg(%arg0: !torch.vtensor<[4,5],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[4,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: torch.aten.tril %arg0, %[[DIAGONAL]] : !torch.vtensor<[4,5],si64>, !torch.int -> !torch.vtensor<[4,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0, %arg1) {torch.onnx.upper = 0 : si64} : (!torch.vtensor<[4,5],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[4,5],si64>
  return %0 : !torch.vtensor<[4,5],si64>
}

// -----

// CHECK-LABEL: func.func @test_tril_one_row_neg
func.func @test_tril_one_row_neg(%arg0: !torch.vtensor<[3,1,5],si64>) -> !torch.vtensor<[3,1,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.constant.int 0
  // CHECK: torch.aten.tril %arg0, %[[DIAGONAL]] : !torch.vtensor<[3,1,5],si64>, !torch.int -> !torch.vtensor<[3,1,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0) {torch.onnx.upper = 0 : si64} : (!torch.vtensor<[3,1,5],si64>) -> !torch.vtensor<[3,1,5],si64>
  return %0 : !torch.vtensor<[3,1,5],si64>
}

// -----

// CHECK-LABEL: func.func @test_tril_square
func.func @test_tril_square(%arg0: !torch.vtensor<[2,3,3],si64>) -> !torch.vtensor<[2,3,3],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.constant.int 0
  // CHECK: torch.aten.tril %arg0, %[[DIAGONAL]] : !torch.vtensor<[2,3,3],si64>, !torch.int -> !torch.vtensor<[2,3,3],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0) {torch.onnx.upper = 0 : si64} : (!torch.vtensor<[2,3,3],si64>) -> !torch.vtensor<[2,3,3],si64>
  return %0 : !torch.vtensor<[2,3,3],si64>
}

// -----

// CHECK-LABEL: func.func @test_tril_zero
func.func @test_tril_zero(%arg0: !torch.vtensor<[3,0,5],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[3,0,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: torch.aten.tril %arg0, %[[DIAGONAL]] : !torch.vtensor<[3,0,5],si64>, !torch.int -> !torch.vtensor<[3,0,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0, %arg1) {torch.onnx.upper = 0 : si64} : (!torch.vtensor<[3,0,5],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[3,0,5],si64>
  return %0 : !torch.vtensor<[3,0,5],si64>
}

// -----

// CHECK-LABEL: func.func @test_triu
func.func @test_triu(%arg0: !torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.constant.int 0
  // CHECK: torch.aten.triu %arg0, %[[DIAGONAL]] : !torch.vtensor<[4,5],si64>, !torch.int -> !torch.vtensor<[4,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0) : (!torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64>
  return %0 : !torch.vtensor<[4,5],si64>
}

// -----

// CHECK-LABEL: func.func @test_triu_one_row
func.func @test_triu_one_row(%arg0: !torch.vtensor<[3,1,5],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[3,1,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: torch.aten.triu %arg0, %[[DIAGONAL]] : !torch.vtensor<[3,1,5],si64>, !torch.int -> !torch.vtensor<[3,1,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0, %arg1) : (!torch.vtensor<[3,1,5],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[3,1,5],si64>
  return %0 : !torch.vtensor<[3,1,5],si64>
}

// -----

// CHECK-LABEL: func.func @test_triu_square
func.func @test_triu_square(%arg0: !torch.vtensor<[2,3,3],si64>) -> !torch.vtensor<[2,3,3],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.constant.int 0
  // CHECK: torch.aten.triu %arg0, %[[DIAGONAL]] : !torch.vtensor<[2,3,3],si64>, !torch.int -> !torch.vtensor<[2,3,3],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0) : (!torch.vtensor<[2,3,3],si64>) -> !torch.vtensor<[2,3,3],si64>
  return %0 : !torch.vtensor<[2,3,3],si64>
}

// -----

// CHECK-LABEL: func.func @test_triu_zero
func.func @test_triu_zero(%arg0: !torch.vtensor<[0,5],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[0,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[DIAGONAL:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: torch.aten.triu %arg0, %[[DIAGONAL]] : !torch.vtensor<[0,5],si64>, !torch.int -> !torch.vtensor<[0,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0, %arg1) : (!torch.vtensor<[0,5],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[0,5],si64>
  return %0 : !torch.vtensor<[0,5],si64>
}

// -----

// CHECK-LABEL: func.func @test_random_normal
func.func @test_random_normal() -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG:  %[[I6:.+]] = torch.constant.int 6
  // CHECK-DAG:  %[[I10:.+]] = torch.constant.int 10
  // CHECK: %[[SHAPE:.+]] = torch.prim.ListConstruct %[[I10]] : (!torch.int) -> !torch.list<int>
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[EMPTY_TENSOR:.+]] = torch.aten.empty.memory_format %[[SHAPE]], %[[I6]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG:  %[[F0:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG:  %[[F1:.+]] = torch.constant.float 1.000000e+00
  // CHECK: torch.aten.normal_functional %[[EMPTY_TENSOR]], %[[F0]], %[[F1]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[10],f32>
  %0 = torch.operator "onnx.RandomNormal"() {torch.onnx.dtype = 1 : si64, torch.onnx.mean = 0.000000e+00 : f32, torch.onnx.scale = 1.000000e+00 : f32, torch.onnx.shape = [10 : si64]} : () -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_random_normal_like
func.func @test_random_normal_like(%arg0: !torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG:  %[[I6:.+]] = torch.constant.int 6
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %arg0, %[[I6]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG:  %[[F0:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG:  %[[F1:.+]] = torch.constant.float 1.000000e+00
  // CHECK: torch.aten.normal_functional %[[CAST]], %[[F0]], %[[F1]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[10],f32>
  %0 = torch.operator "onnx.RandomNormalLike"(%arg0) {torch.onnx.dtype = 1 : si64, torch.onnx.mean = 0.000000e+00 : f32, torch.onnx.scale = 1.000000e+00 : f32} : (!torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_random_uniform
func.func @test_random_uniform() -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG:  %[[I6:.+]] = torch.constant.int 6
  // CHECK-DAG:  %[[I10:.+]] = torch.constant.int 10
  // CHECK: %[[SHAPE:.+]] = torch.prim.ListConstruct %[[I10]] : (!torch.int) -> !torch.list<int>
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[EMPTY_TENSOR:.+]] = torch.aten.empty.memory_format %[[SHAPE]], %[[I6]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG:  %[[F1:.+]] = torch.constant.float 1.000000e+00
  // CHECK-DAG:  %[[F0:.+]] = torch.constant.float 0.000000e+00
  // CHECK: torch.aten.uniform %[[EMPTY_TENSOR]], %[[F0]], %[[F1]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[10],f32>
  %0 = torch.operator "onnx.RandomUniform"() {torch.onnx.dtype = 1 : si64, torch.onnx.high = 1.000000e+00 : f32, torch.onnx.low = 0.000000e+00 : f32, torch.onnx.shape = [10 : si64]} : () -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_random_uniform_like
func.func @test_random_uniform_like(%arg0: !torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG:  %[[I6:.+]] = torch.constant.int 6
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[CAST:.+]] = torch.aten.to.dtype %arg0, %[[I6]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[10],f32>
  // CHECK-DAG:  %[[F0:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG:  %[[F1:.+]] = torch.constant.float 1.000000e+00
  // CHECK: torch.aten.uniform %[[CAST]], %[[F0]], %[[F1]], %[[NONE]] : !torch.vtensor<[10],f32>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[10],f32>
  %0 = torch.operator "onnx.RandomUniformLike"(%arg0) {torch.onnx.dtype = 1 : si64, torch.onnx.high = 1.000000e+00 : f32, torch.onnx.low = 0.000000e+00 : f32} : (!torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f32>
  return %0 : !torch.vtensor<[10],f32>
}

// -----

// CHECK-LABEL: func.func @test_sequence_construct_3
module {
  func.func @test_sequence_construct_3(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,3,4],f32>, %arg2: !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK: %[[SEQ:.+]] = torch.prim.ListConstruct %arg0, %arg1, %arg2 : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
// CHECK: return %[[SEQ]] : !torch.list<vtensor<[2,3,4],f32>>
    %0 = torch.operator "onnx.SequenceConstruct"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
    return %0 : !torch.list<vtensor<[2,3,4],f32>>
  }
}

// -----

// CHECK-LABEL: func.func @test_sequence_construct_1
module {
  func.func @test_sequence_construct_1(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK: %[[SEQ:.+]] = torch.prim.ListConstruct %arg0 : (!torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
// CHECK: return %[[SEQ]] : !torch.list<vtensor<[2,3,4],f32>>
    %0 = torch.operator "onnx.SequenceConstruct"(%arg0) : (!torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
    return %0 : !torch.list<vtensor<[2,3,4],f32>>
  }
}

// -----

// CHECK-LABEL: func.func @test_sequence_length
module {
  func.func @test_sequence_length(%arg0: !torch.list<vtensor<[?,?,?],f32>>) ->  !torch.vtensor<[],si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK: %[[FALSE:.+]] = torch.constant.bool false
// CHECK: %[[NONE:.+]] = torch.constant.none
// CHECK: %[[LEN:.+]] = torch.aten.len.t %arg0 : !torch.list<vtensor<[?,?,?],f32>> -> !torch.int
// CHECK: %[[LEN_AS_TEN:.+]] = torch.aten.tensor.int %[[LEN]],  %[[NONE]],  %[[NONE]], %[[FALSE]] : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
// CHECK: return %[[LEN_AS_TEN]] : !torch.vtensor<[],si64>
    %0 = torch.operator "onnx.SequenceLength"(%arg0) : (!torch.list<vtensor<[?,?,?],f32>>) -> !torch.vtensor<[],si64>
    return %0 : !torch.vtensor<[],si64>
  }
}

// -----

// CHECK-LABEL: func.func @test_sce_mean_3d
func.func @test_sce_mean_3d(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[IGNORE_INDEX:.+]] = torch.constant.int -100
  // CHECK: %[[REDUCTION:.+]] = torch.constant.int 1
  // CHECK: %[[LABEL_SMOOTHING:.+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[LOSS:.+]] = torch.aten.cross_entropy_loss %arg0, %arg1, %[[NONE]], %[[REDUCTION]], %[[IGNORE_INDEX:.+]], %[[LABEL_SMOOTHING]] : !torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>, !torch.none, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[],f32>
  // CHECK: return %[[LOSS]] : !torch.vtensor<[],f32>
  %0 = torch.operator "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1) {torch.onnx.reduction = "mean"} : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @test_sce_mean_3d_log_prob
func.func @test_sce_mean_3d_log_prob(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[3,5,2],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[IGNORE_INDEX:.+]] = torch.constant.int -100
  // CHECK: %[[REDUCTION:.+]] = torch.constant.int 1
  // CHECK: %[[LABEL_SMOOTHING:.+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[LOSS:.+]] = torch.aten.cross_entropy_loss %arg0, %arg1, %[[NONE]], %[[REDUCTION]], %[[IGNORE_INDEX:.+]], %[[LABEL_SMOOTHING]] : !torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>, !torch.none, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[],f32>
  // CHECK: %[[DIM:.+]] = torch.constant.int 1
  // CHECK: %[[NONE_0:.+]] = torch.constant.none
  // CHECK: %[[PROB:.+]] = torch.aten.log_softmax.int %arg0, %[[DIM]], %[[NONE_0]] : !torch.vtensor<[3,5,2],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,5,2],f32>
  // CHECK: return %[[LOSS]], %[[PROB]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,5,2],f32>
  %0:2 = torch.operator "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1) {torch.onnx.reduction = "mean"} : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[3,5,2],f32>)
  return %0#0, %0#1 : !torch.vtensor<[],f32>, !torch.vtensor<[3,5,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_resize_sizes_nearest
  func.func @test_resize_sizes_nearest(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    // CHECK: torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[1,1,2,4],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
    %0 = torch.operator "onnx.Resize"(%arg0, %none, %none, %arg1) {torch.onnx.coordinate_transformation_mode = "asymmetric", torch.onnx.cubic_coeff_a = -7.500000e-01 : f32, torch.onnx.mode = "nearest", torch.onnx.nearest_mode = "floor"} : (!torch.vtensor<[1,1,2,4],f32>, !torch.none, !torch.none, !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],f32>
    return %0 : !torch.vtensor<[?,?,?,?],f32>
  }

// -----

// CHECK-LABEL: func.func @test_resize_sizes_nearest
func.func @test_resize_sizes_nearest(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  // CHECK: %[[STR:.+]] = torch.constant.str "nearest_half_pixel,round_prefer_floor"
  // CHECK: torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %[[STR]], %false, %none_0, %false : !torch.vtensor<[1,1,2,4],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  %0 = torch.operator "onnx.Resize"(%arg0, %none, %none, %arg1) {
    torch.onnx.coordinate_transformation_mode = "half_pixel",
    torch.onnx.mode = "nearest"} : (!torch.vtensor<[1,1,2,4],f32>, !torch.none, !torch.none, !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: func.func @test_resize_sizes_linear
  func.func @test_resize_sizes_linear(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],
f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    // CHECK: torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[1,1,2,4],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
    %0 = torch.operator "onnx.Resize"(%arg0, %none, %none, %arg1) {torch.onnx.mode = "linear"} : (!torch.vtensor<[1,1,2,4],f32>, !torch.none, !torch.none, !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],f32>
    return %0 : !torch.vtensor<[?,?,?,?],f32>
  }

// -----

// CHECK-LABEL: @test_roialign_avg
  func.func @test_roialign_avg(%arg0: !torch.vtensor<[6,2,100,100],f32>, %arg1: !torch.vtensor<[30,4],f32>, %arg2: !torch.vtensor<[30],si64>) -> !torch.vtensor<[30,2,5,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[Dim:.*]] = torch.constant.int 1
    // CHECK: %[[Unsqueeze:.*]] = torch.aten.unsqueeze %arg2, %[[Dim]]
    // CHECK: %[[cst6:.*]] = torch.constant.int 6
    // CHECK: %[[Cast:.*]] = torch.aten.to.dtype %[[Unsqueeze]], %[[cst6]]
    // CHECK: %[[List:.*]] = torch.prim.ListConstruct %[[Cast]], %arg1
    // CHECK: %[[Cat:.*]] = torch.aten.cat %[[List]], %[[Dim]]
    // CHECK: %[[Align:.*]] = torch.torchvision.roi_align %arg0, %[[Cat]]
    %0 = torch.operator "onnx.RoiAlign"(%arg0, %arg1, %arg2) {torch.onnx.coordinate_transformation_mode = "output_half_pixel", torch.onnx.mode = "avg", torch.onnx.output_height = 5 : si64, torch.onnx.output_width = 5 : si64, torch.onnx.sampling_ratio = 0 : si64, torch.onnx.spatial_scale = 1.000000e+00 : f32} : (!torch.vtensor<[6,2,100,100],f32>, !torch.vtensor<[30,4],f32>, !torch.vtensor<[30],si64>) -> !torch.vtensor<[30,2,5,5],f32>
    return %0 : !torch.vtensor<[30,2,5,5],f32>
  }

// -----

// CHECK-LABEL: @test_roialign_max
  func.func @test_roialign_max(%arg0: !torch.vtensor<[6,2,100,100],f32>, %arg1: !torch.vtensor<[30,4],f32>, %arg2: !torch.vtensor<[30],si64>) -> !torch.vtensor<[30,2,5,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[Dim:.*]] = torch.constant.int 1
    // CHECK: %[[Unsqueeze:.*]] = torch.aten.unsqueeze %arg2, %[[Dim]]
    // CHECK: %[[cst6:.*]] = torch.constant.int 6
    // CHECK: %[[Cast:.*]] = torch.aten.to.dtype %[[Unsqueeze]], %[[cst6]]
    // CHECK: %[[List:.*]] = torch.prim.ListConstruct %[[Cast]], %arg1
    // CHECK: %[[Cat:.*]] = torch.aten.cat %[[List]], %[[Dim]]
    // CHECK: %[[Pool:.*]], %[[Indices:.*]] = torch.torchvision.roi_pool %arg0, %[[Cat]]
    // CHECK: return %[[Pool]]
    %0 = torch.operator "onnx.RoiAlign"(%arg0, %arg1, %arg2) {torch.onnx.coordinate_transformation_mode = "half_pixel", torch.onnx.mode = "max", torch.onnx.output_height = 5 : si64, torch.onnx.output_width = 5 : si64, torch.onnx.sampling_ratio = 0 : si64, torch.onnx.spatial_scale = 1.000000e+00 : f32} : (!torch.vtensor<[6,2,100,100],f32>, !torch.vtensor<[30,4],f32>, !torch.vtensor<[30],si64>) -> !torch.vtensor<[30,2,5,5],f32>
    return %0 : !torch.vtensor<[30,2,5,5],f32>
  }

// -----

// CHECK-LABEL: @test_spacetodepth_example
func.func @test_spacetodepth_example(%arg0: !torch.vtensor<[1,1,4,6],f32>) -> !torch.vtensor<[1,4,2,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[B:.*]] = torch.aten.size.int %arg0, %[[C0]] : !torch.vtensor<[1,1,4,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C:.*]] = torch.aten.size.int %arg0, %[[C1]] : !torch.vtensor<[1,1,4,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[H:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[1,1,4,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[W:.*]] = torch.aten.size.int %arg0, %[[C3]] : !torch.vtensor<[1,1,4,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C4:.*]] = torch.constant.int 4
  // CHECK: %[[H_DIV_BS:.*]] = torch.aten.div.int %[[H]], %[[C2_0]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[W_DIV_BS:.*]] = torch.aten.div.int %[[W]], %[[C2_0]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[H_DIV_BS_INT:.*]] = torch.aten.Int.float %[[H_DIV_BS]] : !torch.float -> !torch.int
  // CHECK: %[[W_DIV_BS_INT:.*]] = torch.aten.Int.float %[[W_DIV_BS]] : !torch.float -> !torch.int
  // CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[B]], %[[C]], %[[H_DIV_BS_INT]], %[[C2_0]], %[[W_DIV_BS_INT]], %[[C2_0]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESHAPE:.*]] = torch.aten.reshape %arg0, %[[RESHAPE_LIST]] : !torch.vtensor<[1,1,4,6],f32>, !torch.list<int> -> !torch.vtensor<[1,1,2,2,3,2],f32>
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C2_1:.*]] = torch.constant.int 2
  // CHECK: %[[C4_0:.*]] = torch.constant.int 4
  // CHECK: %[[PERMUTE_DIMS:.*]] = torch.prim.ListConstruct %[[C0_0]], %[[C3_0]], %[[C5]], %[[C1_0]], %[[C2_1]], %[[C4_0]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PERMUTE:.*]] = torch.aten.permute %[[RESHAPE]], %[[PERMUTE_DIMS]] : !torch.vtensor<[1,1,2,2,3,2],f32>, !torch.list<int> -> !torch.vtensor<[1,2,2,1,2,3],f32>
  // CHECK: %[[MUL:.*]] = torch.aten.mul.int %[[C]], %[[C4]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[RESHAPE_LIST_0:.*]] = torch.prim.ListConstruct %[[B]], %[[MUL]], %[[H_DIV_BS_INT]], %[[W_DIV_BS_INT]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.reshape %[[PERMUTE]], %[[RESHAPE_LIST_0]] : !torch.vtensor<[1,2,2,1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[1,4,2,3],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[1,4,2,3],f32
  %0 = torch.operator "onnx.SpaceToDepth"(%arg0) {torch.onnx.blocksize = 2 : si64} : (!torch.vtensor<[1,1,4,6],f32>) -> !torch.vtensor<[1,4,2,3],f32>
  return %0 : !torch.vtensor<[1,4,2,3],f32>
}

// -----

// CHECK-LABEL: @test_spacetodepth
func.func @test_spacetodepth(%arg0: !torch.vtensor<[2,2,6,6],f32>) -> !torch.vtensor<[2,8,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[B:.*]] = torch.aten.size.int %arg0, %[[C0]] : !torch.vtensor<[2,2,6,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C:.*]] = torch.aten.size.int %arg0, %[[C1]] : !torch.vtensor<[2,2,6,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[H:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[2,2,6,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[W:.*]] = torch.aten.size.int %arg0, %[[C3]] : !torch.vtensor<[2,2,6,6],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C4:.*]] = torch.constant.int 4
  // CHECK: %[[H_DIV_BS:.*]] = torch.aten.div.int %[[H]], %[[C2_0]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[W_DIV_BS:.*]] = torch.aten.div.int %[[W]], %[[C2_0]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[H_DIV_BS_INT:.*]] = torch.aten.Int.float %[[H_DIV_BS]] : !torch.float -> !torch.int
  // CHECK: %[[W_DIV_BS_INT:.*]] = torch.aten.Int.float %[[W_DIV_BS]] : !torch.float -> !torch.int
  // CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[B]], %[[C]], %[[H_DIV_BS_INT]], %[[C2_0]], %[[W_DIV_BS_INT]], %[[C2_0]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESHAPE:.*]] = torch.aten.reshape %arg0, %[[RESHAPE_LIST]] : !torch.vtensor<[2,2,6,6],f32>, !torch.list<int> -> !torch.vtensor<[2,2,3,2,3,2],f32>
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C2_1:.*]] = torch.constant.int 2
  // CHECK: %[[C4_0:.*]] = torch.constant.int 4
  // CHECK: %[[PERMUTE_DIMS:.*]] = torch.prim.ListConstruct %[[C0_0]], %[[C3_0]], %[[C5]], %[[C1_0]], %[[C2_1]], %[[C4_0]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PERMUTE:.*]] = torch.aten.permute %[[RESHAPE]], %[[PERMUTE_DIMS]] : !torch.vtensor<[2,2,3,2,3,2],f32>, !torch.list<int> -> !torch.vtensor<[2,2,2,2,3,3],f32>
  // CHECK: %[[MUL:.*]] = torch.aten.mul.int %[[C]], %[[C4]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[RESHAPE_LIST_0:.*]] = torch.prim.ListConstruct %[[B]], %[[MUL]], %[[H_DIV_BS_INT]], %[[W_DIV_BS_INT]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.reshape %[[PERMUTE]], %[[RESHAPE_LIST_0]] : !torch.vtensor<[2,2,2,2,3,3],f32>, !torch.list<int> -> !torch.vtensor<[2,8,3,3],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[2,8,3,3],f32
  %0 = torch.operator "onnx.SpaceToDepth"(%arg0) {torch.onnx.blocksize = 2 : si64} : (!torch.vtensor<[2,2,6,6],f32>) -> !torch.vtensor<[2,8,3,3],f32>
  return %0 : !torch.vtensor<[2,8,3,3],f32>
}

// -----

// CHECK-LABEL: @test_spacetodepth
func.func @test_spacetodepth_dynamic_dims(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[B:.*]] = torch.aten.size.int %arg0, %[[C0]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C:.*]] = torch.aten.size.int %arg0, %[[C1]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[H:.*]] = torch.aten.size.int %arg0, %[[C2]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[W:.*]] = torch.aten.size.int %arg0, %[[C3]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
  // CHECK: %[[C2_0:.*]] = torch.constant.int 2
  // CHECK: %[[C4:.*]] = torch.constant.int 4
  // CHECK: %[[H_DIV_BS:.*]] = torch.aten.div.int %[[H]], %[[C2_0]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[W_DIV_BS:.*]] = torch.aten.div.int %[[W]], %[[C2_0]] : !torch.int, !torch.int -> !torch.float
  // CHECK: %[[H_DIV_BS_INT:.*]] = torch.aten.Int.float %[[H_DIV_BS]] : !torch.float -> !torch.int
  // CHECK: %[[W_DIV_BS_INT:.*]] = torch.aten.Int.float %[[W_DIV_BS]] : !torch.float -> !torch.int
  // CHECK: %[[RESHAPE_LIST:.*]] = torch.prim.ListConstruct %[[B]], %[[C]], %[[H_DIV_BS_INT]], %[[C2_0]], %[[W_DIV_BS_INT]], %[[C2_0]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESHAPE:.*]] = torch.aten.reshape %arg0, %[[RESHAPE_LIST]] : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,2,?,2],f32>
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C2_1:.*]] = torch.constant.int 2
  // CHECK: %[[C4_0:.*]] = torch.constant.int 4
  // CHECK: %[[PERMUTE_DIMS:.*]] = torch.prim.ListConstruct %[[C0_0]], %[[C3_0]], %[[C5]], %[[C1_0]], %[[C2_1]], %[[C4_0]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PERMUTE:.*]] = torch.aten.permute %[[RESHAPE]], %[[PERMUTE_DIMS]] : !torch.vtensor<[?,?,?,2,?,2],f32>, !torch.list<int> -> !torch.vtensor<[?,2,2,?,?,?],f32>
  // CHECK: %[[MUL:.*]] = torch.aten.mul.int %[[C]], %[[C4]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[RESHAPE_LIST_0:.*]] = torch.prim.ListConstruct %[[B]], %[[MUL]], %[[H_DIV_BS_INT]], %[[W_DIV_BS_INT]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.reshape %[[PERMUTE]], %[[RESHAPE_LIST_0]] : !torch.vtensor<[?,2,2,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[?,?,?,?],f32
  %0 = torch.operator "onnx.SpaceToDepth"(%arg0) {torch.onnx.blocksize = 2 : si64} : (!torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: func.func @Shrink
func.func @Shrink(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %float1.500000e00 = torch.constant.float 1.500000e+00
  // CHECK: %float1.500000e00_0 = torch.constant.float 1.500000e+00
  // CHECK: %float0.000000e00 = torch.constant.float 0.000000e+00
  // CHECK: %float1.000000e00 = torch.constant.float 1.000000e+00
  // CHECK: %float-1.500000e00 = torch.constant.float -1.500000e+00
  // CHECK: %0 = torch.aten.lt.Scalar %arg0, %float-1.500000e00 : !torch.vtensor<[5],f32>, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %1 = torch.aten.add.Scalar %arg0, %float1.500000e00_0, %float1.000000e00 : !torch.vtensor<[5],f32>, !torch.float, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %2 = torch.aten.sub.Scalar %arg0, %float1.500000e00_0, %float1.000000e00 : !torch.vtensor<[5],f32>, !torch.float, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %3 = torch.aten.gt.Scalar %arg0, %float1.500000e00 : !torch.vtensor<[5],f32>, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %4 = torch.aten.where.ScalarOther %3, %2, %float0.000000e00 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32>, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %5 = torch.aten.where.self %0, %1, %4 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[5],f32>
  // CHECK: return %5 : !torch.vtensor<[5],f32>
  %0 = torch.operator "onnx.Shrink"(%arg0) {torch.onnx.bias = 1.500000e+00 : f32, torch.onnx.lambd = 1.500000e+00 : f32} : (!torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32>
  return %0 : !torch.vtensor<[5],f32>
}

// -----

// CHECK-LABEL: func.func @test_shrink_hard
func.func @test_shrink_hard(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %float1.500000e00 = torch.constant.float 1.500000e+00
  // CHECK: %float0.000000e00 = torch.constant.float 0.000000e+00
  // CHECK: %float0.000000e00_0 = torch.constant.float 0.000000e+00
  // CHECK: %float1.000000e00 = torch.constant.float 1.000000e+00
  // CHECK: %float-1.500000e00 = torch.constant.float -1.500000e+00
  // CHECK: %0 = torch.aten.lt.Scalar %arg0, %float-1.500000e00 : !torch.vtensor<[5],f32>, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %1 = torch.aten.add.Scalar %arg0, %float0.000000e00, %float1.000000e00 : !torch.vtensor<[5],f32>, !torch.float, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %2 = torch.aten.sub.Scalar %arg0, %float0.000000e00, %float1.000000e00 : !torch.vtensor<[5],f32>, !torch.float, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %3 = torch.aten.gt.Scalar %arg0, %float1.500000e00 : !torch.vtensor<[5],f32>, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %4 = torch.aten.where.ScalarOther %3, %2, %float0.000000e00_0 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32>, !torch.float -> !torch.vtensor<[5],f32>
  // CHECK: %5 = torch.aten.where.self %0, %1, %4 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[5],f32>
  // CHECK: return %5 : !torch.vtensor<[5],f32>
  %0 = torch.operator "onnx.Shrink"(%arg0) {torch.onnx.lambd = 1.500000e+00 : f32} : (!torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32>
  return %0 : !torch.vtensor<[5],f32>
}

// -----

// CHECK-LABEL: func.func @test_sequence_at
func.func @test_sequence_at(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,3,4],f32>, %arg2: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VTENSOR:.*]] = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[VTENSOR_1:.*]] = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[SEQUENCE:.*]] = torch.prim.ListConstruct %arg0, %arg1, %arg2 : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[LENGTH:.*]] = torch.aten.len.t %[[SEQUENCE]] : !torch.list<vtensor<[2,3,4],f32>> -> !torch.int
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[VTENSOR]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[CMP:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[COND:.*]] = torch.aten.Int.bool %[[CMP]] : !torch.bool -> !torch.int
  // CHECK: %[[OFFSET:.*]] = torch.aten.mul.int %[[COND]], %[[LENGTH]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[POSITION:.*]] = torch.aten.add.int %[[ITEM]], %[[OFFSET]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[NONE]], %[[POSITION]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.none, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[POSITION]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[ADD]], %[[LENGTH]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.int, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[CONCAT_LIST:.*]] = torch.aten.add.t %[[SLICE]], %[[SLICE_1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.list<vtensor<[2,3,4],f32>> -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[VTENSOR_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[RESULT:.*]] = torch.aten.__getitem__.t %[[CONCAT_LIST]], %[[ITEM_0]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.int -> !torch.vtensor<[2,3,4],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[2,3,4],f32>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %2 = torch.operator "onnx.SequenceConstruct"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  %3 = torch.operator "onnx.SequenceErase"(%2, %0) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[],si64>) -> !torch.list<vtensor<[2,3,4],f32>>
  %4 = torch.operator "onnx.SequenceAt"(%3, %1) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[],si64>) -> !torch.vtensor<[2,3,4],f32>
  return %4 : !torch.vtensor<[2,3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_sequence_insert
func.func @test_sequence_insert(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,3,4],f32>, %arg2: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VTENSOR:.*]] = torch.vtensor.literal(dense<-3> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[VTENSOR_1:.*]] = torch.vtensor.literal(dense<-1> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[VTENSOR_2:.*]] = torch.vtensor.literal(dense<-1> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[SEQUENCE:.*]] = torch.prim.ListConstruct %arg0, %arg1, %arg2 : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[LENGTH:.*]] = torch.aten.len.t %[[SEQUENCE]] : !torch.list<vtensor<[2,3,4],f32>> -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[VTENSOR]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[CMP:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[COND:.*]] = torch.aten.Int.bool %[[CMP]] : !torch.bool -> !torch.int
  // CHECK: %[[OFFSET:.*]] = torch.aten.mul.int %[[COND]], %[[LENGTH]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[POSITION:.*]] = torch.aten.add.int %[[ITEM]], %[[OFFSET]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[NONE]], %[[POSITION]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.none, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[POSITION]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[ADD]], %[[LENGTH]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.int, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[CONCAT_LIST:.*]] = torch.aten.add.t %[[SLICE]], %[[SLICE_1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.list<vtensor<[2,3,4],f32>> -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[VTENSOR_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: torch.aten.insert.t %[[CONCAT_LIST]], %[[ITEM_0]], %arg0 : !torch.list<vtensor<[2,3,4],f32>>, !torch.int, !torch.vtensor<[2,3,4],f32>
  // CHECK: %[[ITEM_1:.*]] = torch.aten.item %[[VTENSOR_2]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[RESULT:.*]] = torch.aten.__getitem__.t %[[CONCAT_LIST]], %[[ITEM_1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.int -> !torch.vtensor<[2,3,4],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[2,3,4],f32>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-3> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-1> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-1> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %3 = torch.operator "onnx.SequenceConstruct"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  %4 = torch.operator "onnx.SequenceErase"(%3, %0) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[],si64>) -> !torch.list<vtensor<[2,3,4],f32>>
  %5 = torch.operator "onnx.SequenceInsert"(%4, %arg0, %1) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[],si64>) -> !torch.list<vtensor<[2,3,4],f32>>
  %6 = torch.operator "onnx.SequenceAt"(%5, %2) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[],si64>) -> !torch.vtensor<[2,3,4],f32>
  return %6 : !torch.vtensor<[2,3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_sequence_erase_at_beginning
func.func @test_sequence_erase_at_beginning(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,3,4],f32>, %arg2: !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VTENSOR:.*]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[SEQUENCE:.*]] = torch.prim.ListConstruct %arg0, %arg1, %arg2 : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[LENGTH:.*]] = torch.aten.len.t %[[SEQUENCE]] : !torch.list<vtensor<[2,3,4],f32>> -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[VTENSOR]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[CMP:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[COND:.*]] = torch.aten.Int.bool %[[CMP]] : !torch.bool -> !torch.int
  // CHECK: %[[OFFSET:.*]] = torch.aten.mul.int %[[COND]], %[[LENGTH]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[POSITION:.*]] = torch.aten.add.int %[[ITEM]], %[[OFFSET]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[NONE]], %[[POSITION]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.none, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[POSITION]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[ADD]], %[[LENGTH]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.int, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[RESULT:.*]] = torch.aten.add.t %[[SLICE]], %[[SLICE_1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.list<vtensor<[2,3,4],f32>> -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: return %[[RESULT]] : !torch.list<vtensor<[2,3,4],f32>>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %3 = torch.operator "onnx.SequenceConstruct"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  %4 = torch.operator "onnx.SequenceErase"(%3, %0) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[],si64>) -> !torch.list<vtensor<[2,3,4],f32>>
  return %4 : !torch.list<vtensor<[2,3,4],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_erase_at_end
func.func @test_sequence_erase_at_end(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,3,4],f32>, %arg2: !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VTENSOR:.*]] = torch.vtensor.literal(dense<2> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[SEQUENCE:.*]] = torch.prim.ListConstruct %arg0, %arg1, %arg2 : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[LENGTH:.*]] = torch.aten.len.t %[[SEQUENCE]] : !torch.list<vtensor<[2,3,4],f32>> -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[VTENSOR]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[CMP:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[COND:.*]] = torch.aten.Int.bool %[[CMP]] : !torch.bool -> !torch.int
  // CHECK: %[[OFFSET:.*]] = torch.aten.mul.int %[[COND]], %[[LENGTH]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[POSITION:.*]] = torch.aten.add.int %[[ITEM]], %[[OFFSET]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[NONE]], %[[POSITION]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.none, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[POSITION]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[ADD]], %[[LENGTH]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.int, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[RESULT:.*]] = torch.aten.add.t %[[SLICE]], %[[SLICE_1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.list<vtensor<[2,3,4],f32>> -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: return %[[RESULT]] : !torch.list<vtensor<[2,3,4],f32>>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %3 = torch.operator "onnx.SequenceConstruct"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  %4 = torch.operator "onnx.SequenceErase"(%3, %0) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[],si64>) -> !torch.list<vtensor<[2,3,4],f32>>
  return %4 : !torch.list<vtensor<[2,3,4],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_erase_negative_idx
func.func @test_sequence_erase_negative_idx(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,3,4],f32>, %arg2: !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VTENSOR:.*]] = torch.vtensor.literal(dense<-2> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[SEQUENCE:.*]] = torch.prim.ListConstruct %arg0, %arg1, %arg2 : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[LENGTH:.*]] = torch.aten.len.t %[[SEQUENCE]] : !torch.list<vtensor<[2,3,4],f32>> -> !torch.int
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[VTENSOR]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[CMP:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[COND:.*]] = torch.aten.Int.bool %[[CMP]] : !torch.bool -> !torch.int
  // CHECK: %[[OFFSET:.*]] = torch.aten.mul.int %[[COND]], %[[LENGTH]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[POSITION:.*]] = torch.aten.add.int %[[ITEM]], %[[OFFSET]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[NONE]], %[[POSITION]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.none, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[POSITION]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[ADD]], %[[LENGTH]], %[[INT1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.int, !torch.int, !torch.int -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: %[[RESULT:.*]] = torch.aten.add.t %[[SLICE]], %[[SLICE_1]] : !torch.list<vtensor<[2,3,4],f32>>, !torch.list<vtensor<[2,3,4],f32>> -> !torch.list<vtensor<[2,3,4],f32>>
  // CHECK: return %[[RESULT]] : !torch.list<vtensor<[2,3,4],f32>>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-2> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %3 = torch.operator "onnx.SequenceConstruct"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) -> !torch.list<vtensor<[2,3,4],f32>>
  %4 = torch.operator "onnx.SequenceErase"(%3, %0) : (!torch.list<vtensor<[2,3,4],f32>>, !torch.vtensor<[],si64>) -> !torch.list<vtensor<[2,3,4],f32>>
  return %4 : !torch.list<vtensor<[2,3,4],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_erase_empty
func.func @test_sequence_erase_empty() -> !torch.list<vtensor<[],f32>> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VTENSOR:.*]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[INT6:.*]] = torch.constant.int 6
  // CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[EMPTY_TENSOR:.*]] = torch.aten.empty.memory_format %[[SHAPE_LIST]], %[[INT6]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[SEQUENCE:.*]] = torch.prim.ListConstruct %[[EMPTY_TENSOR]] : (!torch.vtensor<[],f32>) -> !torch.list<vtensor<[],f32>>
  // CHECK: %[[LENGTH:.*]] = torch.aten.len.t %[[SEQUENCE]] : !torch.list<vtensor<[],f32>> -> !torch.int
  // CHECK: %[[NONE_0:.*]] = torch.constant.none
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[VTENSOR]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[CMP:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
  // CHECK: %[[COND:.*]] = torch.aten.Int.bool %[[CMP]] : !torch.bool -> !torch.int
  // CHECK: %[[OFFSET:.*]] = torch.aten.mul.int %[[COND]], %[[LENGTH]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[POSITION:.*]] = torch.aten.add.int %[[ITEM]], %[[OFFSET]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[NONE_0]], %[[POSITION]], %[[INT1]] : !torch.list<vtensor<[],f32>>, !torch.none, !torch.int, !torch.int -> !torch.list<vtensor<[],f32>>
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[POSITION]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.t %[[SEQUENCE]], %[[ADD]], %[[LENGTH]], %[[INT1]] : !torch.list<vtensor<[],f32>>, !torch.int, !torch.int, !torch.int -> !torch.list<vtensor<[],f32>>
  // CHECK: %[[RESULT:.*]] = torch.aten.add.t %[[SLICE]], %[[SLICE_1]] : !torch.list<vtensor<[],f32>>, !torch.list<vtensor<[],f32>> -> !torch.list<vtensor<[],f32>>
  // CHECK: return %[[RESULT]] : !torch.list<vtensor<[],f32>>
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si64>} : () -> !torch.vtensor<[],si64>
  %1 = torch.operator "onnx.SequenceEmpty"() : () -> !torch.list<vtensor<[],f32>>
  %4 = torch.operator "onnx.SequenceErase"(%1, %0) : (!torch.list<vtensor<[],f32>>, !torch.vtensor<[],si64>) -> !torch.list<vtensor<[],f32>>
  return %4 : !torch.list<vtensor<[],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_empty
func.func @test_sequence_empty() -> !torch.list<vtensor<[],f32>> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT6:.*]] = torch.constant.int 6
  // CHECK: %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[EMPTY_TENSOR:.*]] = torch.aten.empty.memory_format %[[SHAPE_LIST]], %[[INT6]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[RESULT:.*]] = torch.prim.ListConstruct %[[EMPTY_TENSOR]] : (!torch.vtensor<[],f32>) -> !torch.list<vtensor<[],f32>>
  // CHECK: return %[[RESULT]] : !torch.list<vtensor<[],f32>>
  %0 = torch.operator "onnx.SequenceEmpty"() : () -> !torch.list<vtensor<[],f32>>
  return %0 : !torch.list<vtensor<[],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_map_add
func.func @test_sequence_map_add(%arg0: !torch.list<vtensor<[2,3],f32>>, %arg1: !torch.vtensor<[2,3],f32>) -> !torch.list<vtensor<[2,3],f32>> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[C2]], %[[C3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[ALLOC:.*]] = torch.aten.empty.memory_format %[[SHAPE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
  // CHECK: %[[RESULT:.*]] = torch.prim.ListConstruct %[[ALLOC]] : (!torch.vtensor<[2,3],f32>) -> !torch.list<vtensor<[2,3],f32>>
  // CHECK: %[[LEN:.*]] = torch.aten.len.t %arg0 : !torch.list<vtensor<[2,3],f32>> -> !torch.int
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[LOOP:.*]] = torch.prim.Loop %[[LEN]], %[[TRUE]], init(%[[RESULT]]) {
  // CHECK: ^bb0(%[[ITER_NUM:.*]]: !torch.int, %[[SEQ:.*]]: !torch.list<vtensor<[2,3],f32>>):
  // CHECK:   %[[SAMPLE:.*]] = torch.aten.__getitem__.t %arg0, %[[ITER_NUM]] : !torch.list<vtensor<[2,3],f32>>, !torch.int -> !torch.vtensor<[2,3],f32>
  // CHECK:   %[[C1:.*]] = torch.constant.int 1
  // CHECK:   %[[ADD:.*]] = torch.aten.add.Tensor %[[SAMPLE]], %arg1, %[[C1]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[2,3],f32>
  // CHECK:   %[[APPEND:.*]] = torch.aten.append.t %[[SEQ]], %[[ADD]] : !torch.list<vtensor<[2,3],f32>>, !torch.vtensor<[2,3],f32> -> !torch.list<vtensor<[2,3],f32>>
  // CHECK:   torch.prim.Loop.condition %[[TRUE]], iter(%[[APPEND]] : !torch.list<vtensor<[2,3],f32>>)
  // CHECK: } : (!torch.int, !torch.bool, !torch.list<vtensor<[2,3],f32>>) -> !torch.list<vtensor<[2,3],f32>>
  // CHECK: return %[[LOOP]] : !torch.list<vtensor<[2,3],f32>>
  %0 = torch.operator "onnx.SequenceMap"(%arg0, %arg1) : (!torch.list<vtensor<[2,3],f32>>, !torch.vtensor<[2,3],f32>) -> !torch.list<vtensor<[2,3],f32>> {
  ^bb0(%arg2: !torch.vtensor<[2,3],f32>, %arg3: !torch.vtensor<[2,3],f32>):
    %1 = torch.operator "onnx.Add"(%arg2, %arg3) : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32>
    torch.operator_terminator %1 : !torch.vtensor<[2,3],f32>
  }
  return %0 : !torch.list<vtensor<[2,3],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_map_add_sequence_variadic
func.func @test_sequence_map_add_sequence_variadic(%arg0: !torch.list<vtensor<[?],f32>>, %arg1: !torch.list<vtensor<[?],f32>>, %arg2: !torch.vtensor<[?],f32>) -> !torch.list<vtensor<[?],f32>> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NEG1:.*]] = torch.constant.int -1
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[NEG1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[ALLOC:.*]] = torch.aten.empty.memory_format %[[SHAPE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],f32>
  // CHECK: %[[RESULT:.*]] = torch.prim.ListConstruct %[[ALLOC]] : (!torch.vtensor<[?],f32>) -> !torch.list<vtensor<[?],f32>>
  // CHECK: %[[LEN:.*]] = torch.aten.len.t %arg0 : !torch.list<vtensor<[?],f32>> -> !torch.int
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[LOOP:.*]] = torch.prim.Loop %[[LEN]], %[[TRUE]], init(%[[RESULT]]) {
  // CHECK: ^bb0(%[[ITER_NUM:.*]]: !torch.int, %[[SEQ:.*]]: !torch.list<vtensor<[?],f32>>):
  // CHECK:   %[[SAMPLE:.*]] = torch.aten.__getitem__.t %arg0, %[[ITER_NUM]] : !torch.list<vtensor<[?],f32>>, !torch.int -> !torch.vtensor<[?],f32>
  // CHECK:   %[[ADDITION_INPUT:.*]] = torch.aten.__getitem__.t %arg1, %[[ITER_NUM]] : !torch.list<vtensor<[?],f32>>, !torch.int -> !torch.vtensor<[?],f32>
  // CHECK:   %[[C1:.*]] = torch.constant.int 1
  // CHECK:   %[[ADD:.*]] = torch.aten.add.Tensor %[[SAMPLE]], %[[ADDITION_INPUT]], %[[C1]] : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],f32>
  // CHECK:   %[[C1_0:.*]] = torch.constant.int 1
  // CHECK:   %[[ADD_0:.*]] = torch.aten.add.Tensor %[[ADD]], %arg2, %[[C1_0]] : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],f32>
  // CHECK:   %[[APPEND:.*]] = torch.aten.append.t %[[SEQ]], %[[ADD_0]] : !torch.list<vtensor<[?],f32>>, !torch.vtensor<[?],f32> -> !torch.list<vtensor<[?],f32>>
  // CHECK:   torch.prim.Loop.condition %[[TRUE]], iter(%[[APPEND]] : !torch.list<vtensor<[?],f32>>)
  // CHECK: } : (!torch.int, !torch.bool, !torch.list<vtensor<[?],f32>>) -> !torch.list<vtensor<[?],f32>>
  // CHECK: return %[[LOOP]] : !torch.list<vtensor<[?],f32>>
  %0 = torch.operator "onnx.SequenceMap"(%arg0, %arg1, %arg2) : (!torch.list<vtensor<[?],f32>>, !torch.list<vtensor<[?],f32>>, !torch.vtensor<[?],f32>) -> !torch.list<vtensor<[?],f32>> {
  ^bb0(%arg3: !torch.vtensor<[?],f32>, %arg4: !torch.vtensor<[?],f32>, %arg5: !torch.vtensor<[?],f32>):
    %1 = torch.operator "onnx.Add"(%arg3, %arg4) : (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32>
    %2 = torch.operator "onnx.Add"(%1, %arg5) : (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32>
    torch.operator_terminator %2 : !torch.vtensor<[?],f32>
  }
  return %0 : !torch.list<vtensor<[?],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_map_identity
func.func @test_sequence_map_identity(%arg0: !torch.list<vtensor<[?,?,?],f32>>) -> !torch.list<vtensor<[?,?,?],f32>> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NEG1:.*]] = torch.constant.int -1
  // CHECK: %[[NEG1_0:.*]] = torch.constant.int -1
  // CHECK: %[[NEG1_1:.*]] = torch.constant.int -1
  // CHECK: %[[SHAPE:.*]] = torch.prim.ListConstruct %[[NEG1]], %[[NEG1_0]], %[[NEG1_1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[ALLOC:.*]] = torch.aten.empty.memory_format %[[SHAPE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[RESULT:.*]] = torch.prim.ListConstruct %[[ALLOC]] : (!torch.vtensor<[?,?,?],f32>) -> !torch.list<vtensor<[?,?,?],f32>>
  // CHECK: %[[LEN:.*]] = torch.aten.len.t %arg0 : !torch.list<vtensor<[?,?,?],f32>> -> !torch.int
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[LOOP:.*]] = torch.prim.Loop %[[LEN]], %[[TRUE]], init(%[[RESULT]]) {
  // CHECK: ^bb0(%[[ITER_NUM:.*]]: !torch.int, %[[SEQ:.*]]: !torch.list<vtensor<[?,?,?],f32>>):
  // CHECK:   %[[SAMPLE:.*]] = torch.aten.__getitem__.t %arg0, %[[ITER_NUM]] : !torch.list<vtensor<[?,?,?],f32>>, !torch.int -> !torch.vtensor<[?,?,?],f32>
  // CHECK:   %[[NONE_0:.*]] = torch.constant.none
  // CHECK:   %[[CLONE:.*]] = torch.aten.clone %[[SAMPLE]], %[[NONE_0]] : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[?,?,?],f32>
  // CHECK:   %[[APPEND:.*]] = torch.aten.append.t %[[SEQ]], %[[CLONE]] : !torch.list<vtensor<[?,?,?],f32>>, !torch.vtensor<[?,?,?],f32> -> !torch.list<vtensor<[?,?,?],f32>>
  // CHECK:   torch.prim.Loop.condition %[[TRUE]], iter(%[[APPEND]] : !torch.list<vtensor<[?,?,?],f32>>)
  // CHECK: } : (!torch.int, !torch.bool, !torch.list<vtensor<[?,?,?],f32>>) -> !torch.list<vtensor<[?,?,?],f32>>
  // CHECK: return %[[LOOP]] : !torch.list<vtensor<[?,?,?],f32>>
  %0 = torch.operator "onnx.SequenceMap"(%arg0) : (!torch.list<vtensor<[?,?,?],f32>>) -> !torch.list<vtensor<[?,?,?],f32>> {
  ^bb0(%arg1: !torch.vtensor<[?,?,?],f32>):
    %1 = torch.operator "onnx.Identity"(%arg1) : (!torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32>
    torch.operator_terminator %1 : !torch.vtensor<[?,?,?],f32>
  }
  return %0 : !torch.list<vtensor<[?,?,?],f32>>
}

// -----

// CHECK-LABEL: func.func @test_sequence_map_extract_shapes
func.func @test_sequence_map_extract_shapes(%arg0: !torch.list<vtensor<[?,?,?],f32>>) -> !torch.list<vtensor<[3],si64>> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[SHAPE]] = torch.prim.ListConstruct %[[C3]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[ALLOC:.*]] = torch.aten.empty.memory_format %[[SHAPE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3],si64>
  // CHECK: %[[RESULT:.*]] = torch.prim.ListConstruct %[[ALLOC]] : (!torch.vtensor<[3],si64>) -> !torch.list<vtensor<[3],si64>>
  // CHECK: %[[LEN:.*]] = torch.aten.len.t %arg0 : !torch.list<vtensor<[?,?,?],f32>> -> !torch.int
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[LOOP:.*]] = torch.prim.Loop %[[LEN]], %[[TRUE]], init(%[[RESULT]]) {
  // CHECK: ^bb0(%[[ITER_NUM:.*]]: !torch.int, %[[SEQ:.*]]: !torch.list<vtensor<[3],si64>>):
  // CHECK:   %[[SAMPLE:.*]] = torch.aten.__getitem__.t %arg0, %[[ITER_NUM]] : !torch.list<vtensor<[?,?,?],f32>>, !torch.int -> !torch.vtensor<[?,?,?],f32>
  // CHECK:   %[[SHAPE_0:.*]] = torch.aten._shape_as_tensor %[[SAMPLE]] : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[3],si64>
  // CHECK:   %[[APPEND:.*]] = torch.aten.append.t %[[SEQ]], %[[SHAPE_0]] : !torch.list<vtensor<[3],si64>>, !torch.vtensor<[3],si64> -> !torch.list<vtensor<[3],si64>>
  // CHECK:   torch.prim.Loop.condition %[[TRUE]], iter(%[[APPEND]] : !torch.list<vtensor<[3],si64>>)
  // CHECK: } : (!torch.int, !torch.bool, !torch.list<vtensor<[3],si64>>) -> !torch.list<vtensor<[3],si64>>
  // CHECK: return %[[LOOP]] : !torch.list<vtensor<[3],si64>>
  %0 = torch.operator "onnx.SequenceMap"(%arg0) : (!torch.list<vtensor<[?,?,?],f32>>) -> !torch.list<vtensor<[3],si64>> {
  ^bb0(%arg1: !torch.vtensor<[?,?,?],f32>):
    %1 = torch.operator "onnx.Shape"(%arg1) : (!torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[3],si64>
    torch.operator_terminator %1 : !torch.vtensor<[3],si64>
  }
  return %0 : !torch.list<vtensor<[3],si64>>
}

// -----

// CHECK-LABEL: func.func @test_upsample_nearest
func.func @test_upsample_nearest(%arg0: !torch.vtensor<[1,1,2,2],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,4,6],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT2]] : !torch.vtensor<[4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  // CHECK: %[[SCALE:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[SELECT_0:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT3]] : !torch.vtensor<[4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  // CHECK: %[[SCALE_0:.*]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[SCALE_LIST:.*]] = torch.prim.ListConstruct %[[SCALE]], %[[SCALE_0]] : (!torch.float, !torch.float) -> !torch.list<float>
  // CHECK: %[[MODE:.*]] = torch.constant.str "nearest"
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[UPSAMPLE:.*]] = torch.aten.__interpolate.size_list_scale_list %arg0, %[[NONE]], %[[SCALE_LIST:.*]], %[[MODE]], %[[NONE]], %[[NONE]], %[[FALSE]] : !torch.vtensor<[1,1,2,2],f32>, !torch.none, !torch.list<float>, !torch.str, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1,1,4,6],f32>
  // CHECK: return %[[UPSAMPLE]] : !torch.vtensor<[1,1,4,6],f32>
  %0 = torch.operator "onnx.Upsample"(%arg0, %arg1) {torch.onnx.mode = "nearest"} : (!torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,4,6],f32>
  return %0 : !torch.vtensor<[1,1,4,6],f32>
}

// -----

// CHECK-LABEL: func.func @test_upsample_bilinear
func.func @test_upsample_bilinear(%arg0: !torch.vtensor<[1,1,2,2],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,4,6],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT2]] : !torch.vtensor<[4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  // CHECK: %[[SCALE:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[SELECT_0:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT3]] : !torch.vtensor<[4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  // CHECK: %[[SCALE_0:.*]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[SCALE_LIST:.*]] = torch.prim.ListConstruct %[[SCALE]], %[[SCALE_0]] : (!torch.float, !torch.float) -> !torch.list<float>
  // CHECK: %[[MODE:.*]] = torch.constant.str "bilinear"
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[UPSAMPLE:.*]] = torch.aten.__interpolate.size_list_scale_list %arg0, %[[NONE]], %[[SCALE_LIST:.*]], %[[MODE]], %[[NONE]], %[[NONE]], %[[FALSE]] : !torch.vtensor<[1,1,2,2],f32>, !torch.none, !torch.list<float>, !torch.str, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1,1,4,6],f32>
  // CHECK: return %[[UPSAMPLE]] : !torch.vtensor<[1,1,4,6],f32>
  %0 = torch.operator "onnx.Upsample"(%arg0, %arg1) {torch.onnx.mode = "linear"} : (!torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,4,6],f32>
  return %0 : !torch.vtensor<[1,1,4,6],f32>
}

// -----

// CHECK-LABEL: func.func @test_stft
func.func @test_stft(%arg0: !torch.vtensor<[1,128,1],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[],si64>) -> !torch.vtensor<[1,15,9,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:  %[[FRAMELEN:.*]] = torch.aten.item %arg2 : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:  %[[FRAMESTEP:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:  %[[NONEVAL:.*]] = torch.constant.none
  // CHECK:  %[[ONESSHAPE:.*]] = torch.prim.ListConstruct %[[FRAMELEN]] : (!torch.int) -> !torch.list<int>
  // CHECK:  %[[ONESLIST:.*]] = torch.aten.ones %[[ONESSHAPE]], %[[NONEVAL]], %[[NONEVAL]], %[[NONEVAL]], %[[NONEVAL]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],f32>
  // CHECK:  %[[INT2_0:.*]] = torch.constant.int 2
  // CHECK:  %[[SQUEEZE:.*]] = torch.aten.squeeze.dim %arg0, %[[INT2_0]] : !torch.vtensor<[1,128,1],f32>, !torch.int -> !torch.vtensor<[1,128],f32>
  // CHECK:  %[[FALSEVAL:.*]] = torch.constant.bool false
  // CHECK:  %[[TRUEVAL:.*]] = torch.constant.bool true
  // CHECK:  %[[STFT:.*]] = torch.aten.stft %[[SQUEEZE]], %[[FRAMELEN]], %[[FRAMESTEP]], %[[FRAMELEN]], %[[ONESLIST]], %[[FALSEVAL]], %[[TRUEVAL]], %[[TRUEVAL]] : !torch.vtensor<[1,128],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[?],f32>, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[1,9,15],complex<f32>>
  // CHECK:  %[[INT0:.*]] = torch.constant.int 0
  // CHECK:  %[[INT2_1:.*]] = torch.constant.int 2
  // CHECK:  %[[INT1:.*]] = torch.constant.int 1
  // CHECK:  %[[PERMUTELIST:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT2_1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:  %[[PERMUTE:.*]] = torch.aten.permute %[[STFT]], %[[PERMUTELIST]] : !torch.vtensor<[1,9,15],complex<f32>>, !torch.list<int> -> !torch.vtensor<[1,15,9],complex<f32>>
  // CHECK:  %[[VIEWASREAL:.*]] = torch.aten.view_as_real %[[PERMUTE]] : !torch.vtensor<[1,15,9],complex<f32>> -> !torch.vtensor<[1,15,9,2],f32>
  // CHECK:  return %[[VIEWASREAL]] : !torch.vtensor<[1,15,9,2],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.STFT"(%arg0, %arg1, %none, %arg2) : (!torch.vtensor<[1,128,1],f32>, !torch.vtensor<[],si64>, !torch.none, !torch.vtensor<[],si64>) -> !torch.vtensor<[1,15,9,2],f32>
  return %0 : !torch.vtensor<[1,15,9,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_stft_with_window
func.func @test_stft_with_window(%arg0: !torch.vtensor<[1,128,1],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,15,9,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:  %[[FRAMELEN:.*]] = torch.constant.int 16
  // CHECK:  %[[FRAMESTEP:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:  %[[INT2_0:.*]] = torch.constant.int 2
  // CHECK:  %[[SQUEEZE:.*]] = torch.aten.squeeze.dim %arg0, %[[INT2_0]] : !torch.vtensor<[1,128,1],f32>, !torch.int -> !torch.vtensor<[1,128],f32>
  // CHECK:  %[[WINDOWLEN:.*]] = torch.constant.int 16
  // CHECK:  %[[FALSEVAL:.*]] = torch.constant.bool false
  // CHECK:  %[[TRUEVAL:.*]] = torch.constant.bool true
  // CHECK:  %[[STFT:.*]] = torch.aten.stft %[[SQUEEZE]], %[[FRAMELEN]], %[[FRAMESTEP]], %[[WINDOWLEN]], %arg2, %[[FALSEVAL]], %[[TRUEVAL]], %[[TRUEVAL]] : !torch.vtensor<[1,128],f32>, !torch.int, !torch.int, !torch.int, !torch.vtensor<[16],f32>, !torch.bool, !torch.bool, !torch.bool -> !torch.vtensor<[1,9,15],complex<f32>>
  // CHECK:  %[[INT0:.*]] = torch.constant.int 0
  // CHECK:  %[[INT2_1:.*]] = torch.constant.int 2
  // CHECK:  %[[INT1:.*]] = torch.constant.int 1
  // CHECK:  %[[PERMUTEDIMS:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT2_1]], %[[INT1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:  %[[PERMUTE:.*]] = torch.aten.permute %[[STFT]], %[[PERMUTEDIMS]] : !torch.vtensor<[1,9,15],complex<f32>>, !torch.list<int> -> !torch.vtensor<[1,15,9],complex<f32>>
  // CHECK:  %[[VIEWASREAL:.*]] = torch.aten.view_as_real %[[PERMUTE]] : !torch.vtensor<[1,15,9],complex<f32>> -> !torch.vtensor<[1,15,9,2],f32>
  // CHECK:  return %[[VIEWASREAL]] : !torch.vtensor<[1,15,9,2],f32>
  %0 = torch.operator "onnx.STFT"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,128,1],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,15,9,2],f32>
  return %0 : !torch.vtensor<[1,15,9,2],f32>
}


// CHECK-LABEL: @test_reversesequence_batch
func.func @test_reversesequence_batch(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,4],f32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[C0_1]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %arg0, %[[C0_0]], %[[C0_1]], %[[ADD]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[INDEX:.*]] = torch.prim.NumToTensor.Scalar %[[C0_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_0:.*]] = torch.aten.slice.Tensor %[[SLICE]], %[[C1_0]], %[[C0]], %[[ITEM]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[DIM:.*]] = torch.prim.ListConstruct %[[C1_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %[[SLICE_0]], %[[DIM]] : !torch.vtensor<[1,?],f32>, !torch.list<int> -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[EMBED:.*]] = torch.aten.slice_scatter %[[SLICE]], %[[FLIP]], %[[C1_0]], %[[C0]], %[[ITEM]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[EMBED_0:.*]] = torch.aten.slice_scatter %arg0, %[[EMBED]], %[[C0_0]], %[[C0_1]], %[[ADD]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[ADD_0:.*]] = torch.aten.add.int %[[C1_1]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[EMBED_0]], %[[C0_0]], %[[C1_1]], %[[ADD_0]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[INDEX_0:.*]] = torch.prim.NumToTensor.Scalar %[[C1_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_0:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_0]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[SLICE_1]], %[[C1_0]], %[[C0]], %[[ITEM_0]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[DIM_0:.*]] = torch.prim.ListConstruct %[[C1_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP_0:.*]] = torch.aten.flip %[[SLICE_2]], %[[DIM_0]] : !torch.vtensor<[1,?],f32>, !torch.list<int> -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[EMBED_1:.*]] = torch.aten.slice_scatter %[[SLICE_1]], %[[FLIP_0]], %[[C1_0]], %[[C0]], %[[ITEM_0]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[EMBED_2:.*]] = torch.aten.slice_scatter %[[EMBED_0]], %[[EMBED_1]], %[[C0_0]], %[[C1_1]], %[[ADD_0]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.int %[[C2]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_3:.*]] = torch.aten.slice.Tensor %[[EMBED_2]], %[[C0_0]], %[[C2]], %[[ADD_1]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[INDEX_1:.*]] = torch.prim.NumToTensor.Scalar %[[C2]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_1:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_1:.*]] = torch.aten.item %[[SELECT_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_4:.*]] = torch.aten.slice.Tensor %[[SLICE_3]], %[[C1_0]], %[[C0]], %[[ITEM_1]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[DIM_1:.*]] = torch.prim.ListConstruct %[[C1_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP_1:.*]] = torch.aten.flip %[[SLICE_4]], %[[DIM_1]] : !torch.vtensor<[1,?],f32>, !torch.list<int> -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[EMBED_3:.*]] = torch.aten.slice_scatter %[[SLICE_3]], %[[FLIP_1]], %[[C1_0]], %[[C0]], %[[ITEM_1]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[EMBED_4:.*]] = torch.aten.slice_scatter %[[EMBED_2]], %[[EMBED_3]], %[[C0_0]], %[[C2]], %[[ADD_1]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.int %[[C3]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_5:.*]] = torch.aten.slice.Tensor %[[EMBED_4]], %[[C0_0]], %[[C3]], %[[ADD_2]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[INDEX_2:.*]] = torch.prim.NumToTensor.Scalar %[[C3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_2:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_2]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_2:.*]] = torch.aten.item %[[SELECT_2]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_6:.*]] = torch.aten.slice.Tensor %[[SLICE_5]], %[[C1_0]], %[[C0]], %[[ITEM_2]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[DIM_2:.*]] = torch.prim.ListConstruct %[[C1_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP_2:.*]] = torch.aten.flip %[[SLICE_6]], %[[DIM_2]] : !torch.vtensor<[1,?],f32>, !torch.list<int> -> !torch.vtensor<[1,?],f32>
  // CHECK: %[[EMBED_5:.*]] = torch.aten.slice_scatter %[[SLICE_5]], %[[FLIP_2]], %[[C1_0]], %[[C0]], %[[ITEM_2]], %[[C1]] : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: torch.aten.slice_scatter %[[EMBED_4]], %[[EMBED_5]], %[[C0_0]], %[[C3]], %[[ADD_2]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  %0 = torch.operator "onnx.ReverseSequence"(%arg0, %arg1) {torch.onnx.batch_axis = 0 : si64, torch.onnx.time_axis = 1 : si64} : (!torch.vtensor<[4,4],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL: @test_reversesequence_time
func.func @test_reversesequence_time(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,4],f32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C1_0:.*]] = torch.constant.int 1
  // CHECK: %[[C0_0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_1:.*]] = torch.constant.int 0
  // CHECK: %[[ADD:.*]] = torch.aten.add.int %[[C0_1]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %arg0, %[[C1_0]], %[[C0_1]], %[[ADD]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[INDEX:.*]] = torch.prim.NumToTensor.Scalar %[[C0_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_0:.*]] = torch.aten.slice.Tensor %[[SLICE]], %[[C0_0]], %[[C0]], %[[ITEM]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[DIM:.*]] = torch.prim.ListConstruct %[[C0_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP:.*]] = torch.aten.flip %[[SLICE_0]], %[[DIM]] : !torch.vtensor<[?,1],f32>, !torch.list<int> -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[EMBED:.*]] = torch.aten.slice_scatter %[[SLICE]], %[[FLIP]], %[[C0_0]], %[[C0]], %[[ITEM]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.vtensor<[?,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[EMBED_0:.*]] = torch.aten.slice_scatter %arg0, %[[EMBED]], %[[C1_0]], %[[C0_1]], %[[ADD]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[C1_1:.*]] = torch.constant.int 1
  // CHECK: %[[ADD_0:.*]] = torch.aten.add.int %[[C1_1]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[EMBED_0]], %[[C1_0]], %[[C1_1]], %[[ADD_0]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[INDEX_0:.*]] = torch.prim.NumToTensor.Scalar %[[C1_1]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_0:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_0]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[SLICE_1]], %[[C0_0]], %[[C0]], %[[ITEM_0]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[DIM_0:.*]] = torch.prim.ListConstruct %[[C0_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP_0:.*]] = torch.aten.flip %[[SLICE_2]], %[[DIM_0]] : !torch.vtensor<[?,1],f32>, !torch.list<int> -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[EMBED_1:.*]] = torch.aten.slice_scatter %[[SLICE_1]], %[[FLIP_0]], %[[C0_0]], %[[C0]], %[[ITEM_0]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.vtensor<[?,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[EMBED_2:.*]] = torch.aten.slice_scatter %[[EMBED_0]], %[[EMBED_1]], %[[C1_0]], %[[C1_1]], %[[ADD_0]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[C2:.*]] = torch.constant.int 2
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.int %[[C2]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_3:.*]] = torch.aten.slice.Tensor %[[EMBED_2]], %[[C1_0]], %[[C2]], %[[ADD_1]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[INDEX_1:.*]] = torch.prim.NumToTensor.Scalar %[[C2]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_1:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_1:.*]] = torch.aten.item %[[SELECT_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_4:.*]] = torch.aten.slice.Tensor %[[SLICE_3]], %[[C0_0]], %[[C0]], %[[ITEM_1]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[DIM_1:.*]] = torch.prim.ListConstruct %[[C0_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP_1:.*]] = torch.aten.flip %[[SLICE_4]], %[[DIM_1]] : !torch.vtensor<[?,1],f32>, !torch.list<int> -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[EMBED_3:.*]] = torch.aten.slice_scatter %[[SLICE_3]], %[[FLIP_1]], %[[C0_0]], %[[C0]], %[[ITEM_1]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.vtensor<[?,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[EMBED_4:.*]] = torch.aten.slice_scatter %[[EMBED_2]], %[[EMBED_3]], %[[C1_0]], %[[C2]], %[[ADD_1]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.int %[[C3]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SLICE_5:.*]] = torch.aten.slice.Tensor %[[EMBED_4]], %[[C1_0]], %[[C3]], %[[ADD_2]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[INDEX_2:.*]] = torch.prim.NumToTensor.Scalar %[[C3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[SELECT_2:.*]] = torch.aten.index_select %arg1, %[[C0]], %[[INDEX_2]] : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[ITEM_2:.*]] = torch.aten.item %[[SELECT_2]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[SLICE_6:.*]] = torch.aten.slice.Tensor %[[SLICE_5]], %[[C0_0]], %[[C0]], %[[ITEM_2]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[DIM_2:.*]] = torch.prim.ListConstruct %[[C0_0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FLIP_2:.*]] = torch.aten.flip %[[SLICE_6]], %[[DIM_2]] : !torch.vtensor<[?,1],f32>, !torch.list<int> -> !torch.vtensor<[?,1],f32>
  // CHECK: %[[EMBED_5:.*]] = torch.aten.slice_scatter %[[SLICE_5]], %[[FLIP_2]], %[[C0_0]], %[[C0]], %[[ITEM_2]], %[[C1]] : !torch.vtensor<[4,1],f32>, !torch.vtensor<[?,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: torch.aten.slice_scatter %[[EMBED_4]], %[[EMBED_5]], %[[C1_0]], %[[C3]], %[[ADD_2]], %[[C1]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  %0 = torch.operator "onnx.ReverseSequence"(%arg0, %arg1) {torch.onnx.batch_axis = 1 : si64, torch.onnx.time_axis = 0 : si64} : (!torch.vtensor<[4,4],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// -----

// CHECK-LABEL:   func.func @test_scatternd(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.vtensor<[4,4,4],f32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: !torch.vtensor<[2,1],si64>,
// CHECK-SAME:                              %[[VAL_2:.*]]: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_scatternd(%arg0: !torch.vtensor<[4,4,4],f32>, %arg1: !torch.vtensor<[2,1],si64>, %arg2: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_3:.*]] = torch.constant.none
  // CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_5:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_4]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_7:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_6]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
  // CHECK:           %[[VAL_9:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_8]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.aten.size.int %[[VAL_1]], %[[VAL_12]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_14:.*]] = torch.aten.mul.int %[[VAL_11]], %[[VAL_13]] : !torch.int, !torch.int -> !torch.int
  // CHECK:           %[[VAL_15:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_16:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_17:.*]] = torch.aten.slice.Tensor %[[VAL_1]], %[[VAL_15]], %[[VAL_10]], %[[VAL_16]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_18:.*]] = torch.aten.lt.Scalar %[[VAL_17]], %[[VAL_10]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2,1],i1>
  // CHECK:           %[[VAL_19:.*]] = torch.aten.add.Scalar %[[VAL_17]], %[[VAL_5]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_20:.*]] = torch.aten.where.self %[[VAL_18]], %[[VAL_19]], %[[VAL_17]] : !torch.vtensor<[2,1],i1>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,1],si64> -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_21:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_11]], %[[VAL_11]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_22:.*]] = torch.aten.view %[[VAL_20]], %[[VAL_21]] : !torch.vtensor<[2,1],si64>, !torch.list<int> -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_23:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_24:.*]] = torch.aten.flatten.using_ints %[[VAL_22]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,1,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_25:.*]] = torch.aten.flatten.using_ints %[[VAL_2]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4,4],f32>
  // CHECK:           %[[VAL_26:.*]] = torch.prim.ListConstruct %[[VAL_14]], %[[VAL_7]], %[[VAL_9]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_27:.*]] = torch.constant.bool false
  // CHECK:           %[[VAL_28:.*]] = torch.aten.expand %[[VAL_24]], %[[VAL_26]], %[[VAL_27]] : !torch.vtensor<[2,1,1],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,4,4],si64>
  // CHECK:           %[[VAL_29:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_30:.*]] = torch.aten.flatten.using_ints %[[VAL_0]], %[[VAL_10]], %[[VAL_29]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           %[[VAL_31:.*]] = torch.aten.scatter.src %[[VAL_30]], %[[VAL_10]], %[[VAL_28]], %[[VAL_25]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.vtensor<[2,4,4],si64>, !torch.vtensor<[2,4,4],f32> -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           return %[[VAL_31]] : !torch.vtensor<[4,4,4],f32>
  // CHECK:         }
  %none = torch.constant.none
  %0 = torch.operator "onnx.ScatterND"(%arg0, %arg1, %arg2) : (!torch.vtensor<[4,4,4],f32>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32>
  return %0 : !torch.vtensor<[4,4,4],f32>
}

// CHECK-LABEL:   func.func @test_scatternd_add(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[4,4,4],f32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !torch.vtensor<[2,1],si64>,
// CHECK-SAME:                                  %[[VAL_2:.*]]: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_scatternd_add(%arg0: !torch.vtensor<[4,4,4],f32>, %arg1: !torch.vtensor<[2,1],si64>, %arg2: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_3:.*]] = torch.constant.none
  // CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_5:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_4]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_7:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_6]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
  // CHECK:           %[[VAL_9:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_8]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.aten.size.int %[[VAL_1]], %[[VAL_12]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_14:.*]] = torch.aten.mul.int %[[VAL_11]], %[[VAL_13]] : !torch.int, !torch.int -> !torch.int
  // CHECK:           %[[VAL_15:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_16:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_17:.*]] = torch.aten.slice.Tensor %[[VAL_1]], %[[VAL_15]], %[[VAL_10]], %[[VAL_16]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_18:.*]] = torch.aten.lt.Scalar %[[VAL_17]], %[[VAL_10]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2,1],i1>
  // CHECK:           %[[VAL_19:.*]] = torch.aten.add.Scalar %[[VAL_17]], %[[VAL_5]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_20:.*]] = torch.aten.where.self %[[VAL_18]], %[[VAL_19]], %[[VAL_17]] : !torch.vtensor<[2,1],i1>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,1],si64> -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_21:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_11]], %[[VAL_11]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_22:.*]] = torch.aten.view %[[VAL_20]], %[[VAL_21]] : !torch.vtensor<[2,1],si64>, !torch.list<int> -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_23:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_24:.*]] = torch.aten.flatten.using_ints %[[VAL_22]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,1,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_25:.*]] = torch.aten.flatten.using_ints %[[VAL_2]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4,4],f32>
  // CHECK:           %[[VAL_26:.*]] = torch.prim.ListConstruct %[[VAL_14]], %[[VAL_7]], %[[VAL_9]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_27:.*]] = torch.constant.bool false
  // CHECK:           %[[VAL_28:.*]] = torch.aten.expand %[[VAL_24]], %[[VAL_26]], %[[VAL_27]] : !torch.vtensor<[2,1,1],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,4,4],si64>
  // CHECK:           %[[VAL_29:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_30:.*]] = torch.aten.flatten.using_ints %[[VAL_0]], %[[VAL_10]], %[[VAL_29]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           %[[VAL_31:.*]] = torch.constant.str "sum"
  // CHECK:           %[[VAL_32:.*]] = torch.constant.bool true
  // CHECK:           %[[VAL_33:.*]] = torch.aten.scatter_reduce.two %[[VAL_30]], %[[VAL_10]], %[[VAL_28]], %[[VAL_25]], %[[VAL_31]], %[[VAL_32]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.vtensor<[2,4,4],si64>, !torch.vtensor<[2,4,4],f32>, !torch.str, !torch.bool -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           return %[[VAL_33]] : !torch.vtensor<[4,4,4],f32>
  // CHECK:         }
  %none = torch.constant.none
  %0 = torch.operator "onnx.ScatterND"(%arg0, %arg1, %arg2) {torch.onnx.reduction = "add"} : (!torch.vtensor<[4,4,4],f32>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32>
    return %0 : !torch.vtensor<[4,4,4],f32>
}

// CHECK-LABEL:   func.func @test_scatternd_mul(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[4,4,4],f32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !torch.vtensor<[2,1],si64>,
// CHECK-SAME:                                  %[[VAL_2:.*]]: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_scatternd_mul(%arg0: !torch.vtensor<[4,4,4],f32>, %arg1: !torch.vtensor<[2,1],si64>, %arg2: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_3:.*]] = torch.constant.none
  // CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_5:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_4]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_7:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_6]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
  // CHECK:           %[[VAL_9:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_8]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.aten.size.int %[[VAL_1]], %[[VAL_12]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_14:.*]] = torch.aten.mul.int %[[VAL_11]], %[[VAL_13]] : !torch.int, !torch.int -> !torch.int
  // CHECK:           %[[VAL_15:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_16:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_17:.*]] = torch.aten.slice.Tensor %[[VAL_1]], %[[VAL_15]], %[[VAL_10]], %[[VAL_16]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_18:.*]] = torch.aten.lt.Scalar %[[VAL_17]], %[[VAL_10]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2,1],i1>
  // CHECK:           %[[VAL_19:.*]] = torch.aten.add.Scalar %[[VAL_17]], %[[VAL_5]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_20:.*]] = torch.aten.where.self %[[VAL_18]], %[[VAL_19]], %[[VAL_17]] : !torch.vtensor<[2,1],i1>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,1],si64> -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_21:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_11]], %[[VAL_11]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_22:.*]] = torch.aten.view %[[VAL_20]], %[[VAL_21]] : !torch.vtensor<[2,1],si64>, !torch.list<int> -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_23:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_24:.*]] = torch.aten.flatten.using_ints %[[VAL_22]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,1,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_25:.*]] = torch.aten.flatten.using_ints %[[VAL_2]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4,4],f32>
  // CHECK:           %[[VAL_26:.*]] = torch.prim.ListConstruct %[[VAL_14]], %[[VAL_7]], %[[VAL_9]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_27:.*]] = torch.constant.bool false
  // CHECK:           %[[VAL_28:.*]] = torch.aten.expand %[[VAL_24]], %[[VAL_26]], %[[VAL_27]] : !torch.vtensor<[2,1,1],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,4,4],si64>
  // CHECK:           %[[VAL_29:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_30:.*]] = torch.aten.flatten.using_ints %[[VAL_0]], %[[VAL_10]], %[[VAL_29]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           %[[VAL_31:.*]] = torch.constant.str "prod"
  // CHECK:           %[[VAL_32:.*]] = torch.constant.bool true
  // CHECK:           %[[VAL_33:.*]] = torch.aten.scatter_reduce.two %[[VAL_30]], %[[VAL_10]], %[[VAL_28]], %[[VAL_25]], %[[VAL_31]], %[[VAL_32]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.vtensor<[2,4,4],si64>, !torch.vtensor<[2,4,4],f32>, !torch.str, !torch.bool -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           return %[[VAL_33]] : !torch.vtensor<[4,4,4],f32>
  // CHECK:         }
  %none = torch.constant.none
  %0 = torch.operator "onnx.ScatterND"(%arg0, %arg1, %arg2) {torch.onnx.reduction = "mul"} : (!torch.vtensor<[4,4,4],f32>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32>
    return %0 : !torch.vtensor<[4,4,4],f32>
}

// CHECK-LABEL:   func.func @test_scatternd_max(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[4,4,4],f32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !torch.vtensor<[2,1],si64>,
// CHECK-SAME:                                  %[[VAL_2:.*]]: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_scatternd_max(%arg0: !torch.vtensor<[4,4,4],f32>, %arg1: !torch.vtensor<[2,1],si64>, %arg2: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_3:.*]] = torch.constant.none
  // CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_5:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_4]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_7:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_6]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
  // CHECK:           %[[VAL_9:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_8]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.aten.size.int %[[VAL_1]], %[[VAL_12]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_14:.*]] = torch.aten.mul.int %[[VAL_11]], %[[VAL_13]] : !torch.int, !torch.int -> !torch.int
  // CHECK:           %[[VAL_15:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_16:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_17:.*]] = torch.aten.slice.Tensor %[[VAL_1]], %[[VAL_15]], %[[VAL_10]], %[[VAL_16]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_18:.*]] = torch.aten.lt.Scalar %[[VAL_17]], %[[VAL_10]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2,1],i1>
  // CHECK:           %[[VAL_19:.*]] = torch.aten.add.Scalar %[[VAL_17]], %[[VAL_5]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_20:.*]] = torch.aten.where.self %[[VAL_18]], %[[VAL_19]], %[[VAL_17]] : !torch.vtensor<[2,1],i1>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,1],si64> -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_21:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_11]], %[[VAL_11]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_22:.*]] = torch.aten.view %[[VAL_20]], %[[VAL_21]] : !torch.vtensor<[2,1],si64>, !torch.list<int> -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_23:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_24:.*]] = torch.aten.flatten.using_ints %[[VAL_22]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,1,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_25:.*]] = torch.aten.flatten.using_ints %[[VAL_2]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4,4],f32>
  // CHECK:           %[[VAL_26:.*]] = torch.prim.ListConstruct %[[VAL_14]], %[[VAL_7]], %[[VAL_9]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_27:.*]] = torch.constant.bool false
  // CHECK:           %[[VAL_28:.*]] = torch.aten.expand %[[VAL_24]], %[[VAL_26]], %[[VAL_27]] : !torch.vtensor<[2,1,1],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,4,4],si64>
  // CHECK:           %[[VAL_29:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_30:.*]] = torch.aten.flatten.using_ints %[[VAL_0]], %[[VAL_10]], %[[VAL_29]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           %[[VAL_31:.*]] = torch.constant.str "amax"
  // CHECK:           %[[VAL_32:.*]] = torch.constant.bool true
  // CHECK:           %[[VAL_33:.*]] = torch.aten.scatter_reduce.two %[[VAL_30]], %[[VAL_10]], %[[VAL_28]], %[[VAL_25]], %[[VAL_31]], %[[VAL_32]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.vtensor<[2,4,4],si64>, !torch.vtensor<[2,4,4],f32>, !torch.str, !torch.bool -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           return %[[VAL_33]] : !torch.vtensor<[4,4,4],f32>
  // CHECK:         }
  %none = torch.constant.none
  %0 = torch.operator "onnx.ScatterND"(%arg0, %arg1, %arg2) {torch.onnx.reduction = "max"} : (!torch.vtensor<[4,4,4],f32>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32>
    return %0 : !torch.vtensor<[4,4,4],f32>
}

// CHECK-LABEL:   func.func @test_scatternd_min(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[4,4,4],f32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !torch.vtensor<[2,1],si64>,
// CHECK-SAME:                                  %[[VAL_2:.*]]: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_scatternd_min(%arg0: !torch.vtensor<[4,4,4],f32>, %arg1: !torch.vtensor<[2,1],si64>, %arg2: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_3:.*]] = torch.constant.none
  // CHECK:           %[[VAL_4:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_5:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_4]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_7:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_6]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_8:.*]] = torch.constant.int 2
  // CHECK:           %[[VAL_9:.*]] = torch.aten.size.int %[[VAL_0]], %[[VAL_8]] : !torch.vtensor<[4,4,4],f32>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_10:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_13:.*]] = torch.aten.size.int %[[VAL_1]], %[[VAL_12]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.int
  // CHECK:           %[[VAL_14:.*]] = torch.aten.mul.int %[[VAL_11]], %[[VAL_13]] : !torch.int, !torch.int -> !torch.int
  // CHECK:           %[[VAL_15:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_16:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_17:.*]] = torch.aten.slice.Tensor %[[VAL_1]], %[[VAL_15]], %[[VAL_10]], %[[VAL_16]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_18:.*]] = torch.aten.lt.Scalar %[[VAL_17]], %[[VAL_10]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2,1],i1>
  // CHECK:           %[[VAL_19:.*]] = torch.aten.add.Scalar %[[VAL_17]], %[[VAL_5]], %[[VAL_11]] : !torch.vtensor<[2,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_20:.*]] = torch.aten.where.self %[[VAL_18]], %[[VAL_19]], %[[VAL_17]] : !torch.vtensor<[2,1],i1>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,1],si64> -> !torch.vtensor<[2,1],si64>
  // CHECK:           %[[VAL_21:.*]] = torch.prim.ListConstruct %[[VAL_13]], %[[VAL_11]], %[[VAL_11]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_22:.*]] = torch.aten.view %[[VAL_20]], %[[VAL_21]] : !torch.vtensor<[2,1],si64>, !torch.list<int> -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_23:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_24:.*]] = torch.aten.flatten.using_ints %[[VAL_22]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,1,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,1,1],si64>
  // CHECK:           %[[VAL_25:.*]] = torch.aten.flatten.using_ints %[[VAL_2]], %[[VAL_10]], %[[VAL_23]] : !torch.vtensor<[2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4,4],f32>
  // CHECK:           %[[VAL_26:.*]] = torch.prim.ListConstruct %[[VAL_14]], %[[VAL_7]], %[[VAL_9]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:           %[[VAL_27:.*]] = torch.constant.bool false
  // CHECK:           %[[VAL_28:.*]] = torch.aten.expand %[[VAL_24]], %[[VAL_26]], %[[VAL_27]] : !torch.vtensor<[2,1,1],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,4,4],si64>
  // CHECK:           %[[VAL_29:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_30:.*]] = torch.aten.flatten.using_ints %[[VAL_0]], %[[VAL_10]], %[[VAL_29]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           %[[VAL_31:.*]] = torch.constant.str "amin"
  // CHECK:           %[[VAL_32:.*]] = torch.constant.bool true
  // CHECK:           %[[VAL_33:.*]] = torch.aten.scatter_reduce.two %[[VAL_30]], %[[VAL_10]], %[[VAL_28]], %[[VAL_25]], %[[VAL_31]], %[[VAL_32]] : !torch.vtensor<[4,4,4],f32>, !torch.int, !torch.vtensor<[2,4,4],si64>, !torch.vtensor<[2,4,4],f32>, !torch.str, !torch.bool -> !torch.vtensor<[4,4,4],f32>
  // CHECK:           return %[[VAL_33]] : !torch.vtensor<[4,4,4],f32>
  // CHECK:         }
  %none = torch.constant.none
  %0 = torch.operator "onnx.ScatterND"(%arg0, %arg1, %arg2) {torch.onnx.reduction = "min"} : (!torch.vtensor<[4,4,4],f32>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32>
    return %0 : !torch.vtensor<[4,4,4],f32>
}

//--

// CHECK-LABEL: func.func @test_split_to_sequence_1
func.func @test_split_to_sequence_1(%arg0: !torch.vtensor<[3,6],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.list<vtensor<[3,6],f32>> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VAL_0:.*]]: !torch.vtensor<[3,6],f32>
  // CHECK: %[[VAL_1:.*]]: !torch.vtensor<[1],si64>) -> !torch.list<vtensor<[3,6],f32>>
  // CHECK: %[[VAL_2:.*]] = torch.constant.none
  // CHECK: %[[VAL_3:.*]] = torch.constant.int 1
  // CHECK: %[[VAL_4:.*]] = torch.aten.item %[[VAL_1]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: %[[VAL_5:.*]] = torch.aten.split.Tensor %[[VAL_0]], %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[3,6],f32>, !torch.int, !torch.int -> !torch.list<vtensor<[3,6],f32>>
  // CHECK: return %[[VAL_5]] : !torch.list<vtensor<[3,6],f32>>
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %0 = torch.aten.item %arg1 : !torch.vtensor<[1],si64> -> !torch.int
  %1 = torch.aten.split.Tensor %arg0, %0, %int1 : !torch.vtensor<[3,6],f32>, !torch.int, !torch.int -> !torch.list<vtensor<[3,6],f32>>
  return %1 : !torch.list<vtensor<[3,6],f32>>
}

//----

// CHECK-LABEL: func.func @test_split_to_sequence_2
func.func @test_split_to_sequence_2(%arg0: !torch.vtensor<[2,6],f32>, %arg1: !torch.vtensor<[],si64>) -> !torch.list<vtensor<[1,6],f32>> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[VAL_0:.*]]: !torch.vtensor<[2,6],f32>
  // CHECK: %[[VAL_1:.*]]: !torch.vtensor<[],si64>) -> !torch.list<vtensor<[1,6],f32>>
  // CHECK: %[[VAL_2:.*]] = torch.constant.none
  // CHECK: %[[VAL_3:.*]] = torch.constant.int 0
  // CHECK: %[[VAL_4:.*]] = torch.aten.item %[[VAL_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[VAL_5:.*]] = torch.aten.split.Tensor %[[VAL_0]], %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[2,6],f32>, !torch.int, !torch.int -> !torch.list<vtensor<[1,6],f32>>
  // CHECK: return %[[VAL_5]] : !torch.list<vtensor<[1,6],f32>>
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %0 = torch.aten.item %arg1 : !torch.vtensor<[],si64> -> !torch.int
  %1 = torch.aten.split.Tensor %arg0, %0, %int0 : !torch.vtensor<[2,6],f32>, !torch.int, !torch.int -> !torch.list<vtensor<[1,6],f32>>
  return %1 : !torch.list<vtensor<[1,6],f32>>
}

// ---

// CHECK-LABEL:   func.func @test_split_to_sequence_with_list(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[4,6],f32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[2],si64>) -> !torch.list<vtensor<[2,6],f32>> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]] : (!torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.aten.split.sizes %[[VAL_0]], %[[VAL_5]], %[[VAL_3]] : !torch.vtensor<[4,6],f32>, !torch.list<int>, !torch.int -> !torch.list<vtensor<[2,6],f32>>
// CHECK:           return %[[VAL_6]] : !torch.list<vtensor<[2,6],f32>>
  func.func @test_split_to_sequence_with_list(%arg0: !torch.vtensor<[4,6],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.list<vtensor<[2,6],f32>> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.SplitToSequence"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[4,6],f32>, !torch.vtensor<[2],si64>) -> !torch.list<vtensor<[2,6],f32>>
    return %0 : !torch.list<vtensor<[2,6],f32>>
  }

