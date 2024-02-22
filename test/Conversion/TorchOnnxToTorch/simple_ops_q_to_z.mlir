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
  // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_2:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_3:.+]] = torch.constant.int 1
  // CHECK: %[[INT0_2:.+]] = torch.constant.int 0
  // CHECK: %[[PAD:.+]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT0_1]]
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
  // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_2:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_3:.+]] = torch.constant.int 1
  // CHECK: %[[INT0_2:.+]] = torch.constant.int 0
  // CHECK: %[[PAD:.+]] = torch.prim.ListConstruct %[[INT0_0]], %[[INT0_1]]
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

// CHECK-LABEL: func.func @test_scatter_elements_with_axis
func.func @test_scatter_elements_with_axis(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.scatter.src %arg0, %int1, %arg1, %arg2 : !torch.vtensor<[1,5],f32>, !torch.int, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32> -> !torch.vtensor<[1,5],f32>
  %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32>
  return %0 : !torch.vtensor<[1,5],f32>
}

// CHECK-LABEL: func.func @test_scatter_elements_with_duplicate_indices
func.func @test_scatter_elements_with_duplicate_indices(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[STR:.*]] = torch.constant.str "add"
  // CHECK: torch.aten.scatter.reduce %arg0, %int1, %arg1, %arg2, %str : !torch.vtensor<[1,5],f32>, !torch.int, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>, !torch.str -> !torch.vtensor<[1,5],f32>
  %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64, torch.onnx.reduction = "add"} : (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32>
  return %0 : !torch.vtensor<[1,5],f32>
}

// CHECK-LABEL: func.func @test_scatter_elements_without_axis
func.func @test_scatter_elements_without_axis(%arg0: !torch.vtensor<[3,3],f32>, %arg1: !torch.vtensor<[2,3],si64>, %arg2: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.scatter.src %arg0, %int0, %arg1, %arg2 : !torch.vtensor<[3,3],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,3],f32>
  %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3,3],f32>, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,3],f32>
  return %0 : !torch.vtensor<[3,3],f32>
}

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

// CHECK-LABEL: func.func @test_sub_example
func.func @test_sub_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sub"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// CHECK-LABEL: func.func @test_sub
func.func @test_sub(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Sub"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

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
  // CHECK: torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  // CHECK: torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor
  // CHECK: torch.aten.add.Tensor %1, %arg3, %int1 : !torch.vtensor, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Sum"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// CHECK-LABEL: func.func @test_sum_one_input
func.func @test_sum_one_input(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.Sum"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

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

// CHECK-LABEL: func.func @test_xor3d
func.func @test_xor3d(%arg0: !torch.vtensor<[3,4,5],i1>, %arg1: !torch.vtensor<[3,4,5],i1>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],i1> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],i1>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// CHECK-LABEL: func.func @test_xor4d
func.func @test_xor4d(%arg0: !torch.vtensor<[3,4,5,6],i1>, %arg1: !torch.vtensor<[3,4,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[3,4,5,6],i1>, !torch.vtensor<[3,4,5,6],i1> -> !torch.vtensor<[3,4,5,6],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],i1>, !torch.vtensor<[3,4,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1>
  return %0 : !torch.vtensor<[3,4,5,6],i1>
}

// CHECK-LABEL: func.func @test_xor_bcast3v1d
func.func @test_xor_bcast3v1d(%arg0: !torch.vtensor<[3,4,5],i1>, %arg1: !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[5],i1>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// CHECK-LABEL: func.func @test_xor_bcast4v4d
func.func @test_xor_bcast4v4d(%arg0: !torch.vtensor<[1,4,1,6],i1>, %arg1: !torch.vtensor<[3,1,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.logical_xor %arg0, %arg1 : !torch.vtensor<[1,4,1,6],i1>, !torch.vtensor<[3,1,5,6],i1> -> !torch.vtensor<[3,4,5,6],i1>
  %0 = torch.operator "onnx.Xor"(%arg0, %arg1) : (!torch.vtensor<[1,4,1,6],i1>, !torch.vtensor<[3,1,5,6],i1>) -> !torch.vtensor<[3,4,5,6],i1>
  return %0 : !torch.vtensor<[3,4,5,6],i1>
}

// -----

// CHECK-LABEL: func.func @test_squeeze
func.func @test_squeeze(%arg0: !torch.vtensor<[1,3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: torch.prims.squeeze %arg0, %6 : !torch.vtensor<[1,3,4,5],f32>, !torch.list<int> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Squeeze"(%arg0, %arg1) : (!torch.vtensor<[1,3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: func.func @test_squeeze_two_axes
func.func @test_squeeze_two_axes(%arg0: !torch.vtensor<[3,1,4,5,1],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT5:.*]] = torch.constant.int 5
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int5 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %arg1, %int0, %int1 : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %6 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %8 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %9, %int5 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %7, %10 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5, %11 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.prims.squeeze %arg0, %12 : !torch.vtensor<[3,1,4,5,1],f32>, !torch.list<int> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Squeeze"(%arg0, %arg1) : (!torch.vtensor<[3,1,4,5,1],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_unsqueeze_axis_0
func.func @test_unsqueeze_axis_0(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: torch.constant.bool false
  // CHECK: torch.constant.none
  // CHECK: torch.aten.tensor %6, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.sort %7, %int0, %false : !torch.vtensor<[1],si64>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %values, %int0, %int0_1 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %8 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %arg0, %9 : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[1,3,4,5],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,3,4,5],f32>
  return %0 : !torch.vtensor<[1,3,4,5],f32>
}

// CHECK-LABEL: func.func @test_unsqueeze_axis_1
func.func @test_unsqueeze_axis_1(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.tensor %6, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.sort %7, %int0, %false : !torch.vtensor<[1],si64>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %values, %int0, %int0_1 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %8 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %arg0, %9 : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,1,4,5],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,4,5],f32>
  return %0 : !torch.vtensor<[3,1,4,5],f32>
}

// CHECK-LABEL: func.func @test_unsqueeze_axis_2
func.func @test_unsqueeze_axis_2(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,1,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.tensor %6, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.sort %7, %int0, %false : !torch.vtensor<[1],si64>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %values, %int0, %int0_1 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %8 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %arg0, %9 : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor<[3,4,1,5],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,1,5],f32>
  return %0 : !torch.vtensor<[3,4,1,5],f32>
}

// CHECK-LABEL: func.func @test_unsqueeze_negative_axes
func.func @test_unsqueeze_negative_axes(%arg0: !torch.vtensor<[1,3,1,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,3,1,1,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.tensor %6, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.sort %7, %int0, %false : !torch.vtensor<[1],si64>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %values, %int0, %int0_1 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %8 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %arg0, %9 : !torch.vtensor<[1,3,1,5],f32>, !torch.int -> !torch.vtensor<[1,3,1,1,5],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[1,3,1,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,3,1,1,5],f32>
  return %0 : !torch.vtensor<[1,3,1,1,5],f32>
}

// CHECK-LABEL: func.func @test_unsqueeze_three_axes
func.func @test_unsqueeze_three_axes(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,1,5,1,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %arg1, %int0, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %6 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %8 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %9, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %7, %10 : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %12 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %14 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %15, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %13, %16 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5, %11, %17 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.tensor %18, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[3],si64>
  // CHECK: torch.aten.sort %19, %int0, %false : !torch.vtensor<[3],si64>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %values, %int0, %int0_1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %20 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %arg0, %21 : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor
  // CHECK: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %values, %int0, %int1_2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %23 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %22, %24 : !torch.vtensor, !torch.int -> !torch.vtensor
  // CHECK: %[[INT2_3:.*]] = torch.constant.int 2
  // CHECK: torch.aten.select.int %values, %int0, %int2_3 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %26 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %25, %27 : !torch.vtensor, !torch.int -> !torch.vtensor<[3,4,1,5,1,1],f32>
  %0 = torch.operator "onnx.Unsqueeze"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,1,5,1,1],f32>
  return %0 : !torch.vtensor<[3,4,1,5,1,1],f32>
}

// CHECK-LABEL: func.func @test_unsqueeze_unsorted_axes
func.func @test_unsqueeze_unsorted_axes(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,1,5,1,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %arg1, %int0, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %6 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %8 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %9, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %7, %10 : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %12 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %13, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %14 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %15, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %13, %16 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5, %11, %17 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.tensor %18, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[3],si64>
  // CHECK: torch.aten.sort %19, %int0, %false : !torch.vtensor<[3],si64>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %values, %int0, %int0_1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %20 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %arg0, %21 : !torch.vtensor<[3,4,5],f32>, !torch.int -> !torch.vtensor
  // CHECK: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK: torch.aten.select.int %values, %int0, %int1_2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %23 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %22, %24 : !torch.vtensor, !torch.int -> !torch.vtensor
  // CHECK: %[[INT2_3:.*]] = torch.constant.int 2
  // CHECK: torch.aten.select.int %values, %int0, %int2_3 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %26 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.unsqueeze %25, %27 : !torch.vtensor, !torch.int -> !torch.vtensor<[3,4,1,5,1,1],f32>
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

// CHECK-LABEL: func.func @test_softmax_axis_1
func.func @test_softmax_axis_1(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int1, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: func.func @test_softmax_axis_2
func.func @test_softmax_axis_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int2, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: func.func @test_softmax_default_axis
func.func @test_softmax_default_axis(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int2, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: func.func @test_softmax_large_number
func.func @test_softmax_large_number(%arg0: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int1, %none : !torch.vtensor<[2,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) : (!torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @test_softmax_negative_axis
func.func @test_softmax_negative_axis(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.softmax.int %arg0, %int2, %none : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
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

// CHECK-LABEL: func.func @test_reduce_max_keepdims_example
func.func @test_reduce_max_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[INT0:.*]] = torch.constant.int 0
    // CHECK: %[[RANK:.*]] = torch.constant.int 3
    // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
    // CHECK: %[[SELECT_DIM0:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    // CHECK: %[[ITEM0:.*]] = torch.aten.item %[[SELECT_DIM0]] : !torch.vtensor<[1],si64> -> !torch.int
    // CHECK: %[[LTZERO_0:.*]] = torch.aten.lt.int %[[ITEM0]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
    // CHECK: %[[ISNEG_0:.*]] = torch.aten.Int.bool %[[LTZERO_0]] : !torch.bool -> !torch.int
    // CHECK: %[[ADJUSTMENT_0:.*]] = torch.aten.mul.int %[[ISNEG_0]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[FINAL_0:.*]] = torch.aten.add.int %[[ITEM0]], %[[ADJUSTMENT_0]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[INT1:.*]] = torch.constant.int 1
    // CHECK: %[[SELECT_DIM1:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    // CHECK: %[[ITEM1:.*]] = torch.aten.item %[[SELECT_DIM1]] : !torch.vtensor<[1],si64> -> !torch.int
    // CHECK: %[[LTZERO_1:.*]] = torch.aten.lt.int %[[ITEM1]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
    // CHECK: %[[ISNEG_1:.*]] = torch.aten.Int.bool %[[LTZERO_1]] : !torch.bool -> !torch.int
    // CHECK: %[[ADJUSTMENT_1:.*]] = torch.aten.mul.int %[[ISNEG_1]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[FINAL_1:.*]] = torch.aten.add.int %[[ITEM1]], %[[ADJUSTMENT_1]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[FINAL_0]], %[[FINAL_1]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[KEEPDIMS:.*]] = torch.constant.bool true
    // CHECK: torch.aten.amax %arg0, %[[DIMS]], %[[KEEPDIMS]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1,1],f32>
    %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,1,1],f32>
    return %0 : !torch.vtensor<[3,1,1],f32>
  }

// -----

// CHECK-LABEL: func.func @test_reduce_max_default_axes_keepdim_example
func.func @test_reduce_max_default_axes_keepdim_example(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[INT0:.*]] = torch.constant.int 0
    // CHECK: %[[INT1:.*]] = torch.constant.int 1
    // CHECK: %[[INT2:.*]] = torch.constant.int 2
    // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT1]], %[[INT2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[INT1_0:.*]] = torch.constant.int 1
    // CHECK: %[[KEEPDIMS:.*]] = torch.aten.Bool.int %[[INT1_0]] : !torch.int -> !torch.bool
    // CHECK: torch.aten.amax %arg0, %[[DIMS]], %[[KEEPDIMS]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,1],f32>
    %0 = torch.operator "onnx.ReduceMax"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32>
    return %0 : !torch.vtensor<[1,1,1],f32>
  }

// -----

// CHECK-LABEL: func.func @test_reduce_max_do_not_keepdims_example
  func.func @test_reduce_max_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[INT0:.*]] = torch.constant.int 0
    // CHECK: %[[RANK:.*]] = torch.constant.int 3
    // CHECK: %[[INT0_0:.*]] = torch.constant.int 0
    // CHECK: %[[SELECT_DIM:.*]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    // CHECK: %[[ITEM:.*]] = torch.aten.item %[[SELECT_DIM]] : !torch.vtensor<[1],si64> -> !torch.int
    // CHECK: %[[LTZERO:.*]] = torch.aten.lt.int %[[ITEM]], %[[INT0]] : !torch.int, !torch.int -> !torch.bool
    // CHECK: %[[ISNEG:.*]] = torch.aten.Int.bool %[[LTZERO]] : !torch.bool -> !torch.int
    // CHECK: %[[ADJUSTMENT:.*]] = torch.aten.mul.int %[[ISNEG]], %[[RANK]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[FINAL:.*]] = torch.aten.add.int %[[ITEM]], %[[ADJUSTMENT]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[FINAL]] : (!torch.int) -> !torch.list<int>
    // CHECK: %[[FALSE:.*]] = torch.constant.bool false
    // CHECK: torch.aten.amax %arg0, %[[DIMS]], %[[FALSE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2],f32>
    %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
    return %0 : !torch.vtensor<[3,2],f32>
  }

// -----

// CHECK-LABEL: func.func @test_reduce_sum_default_axes_keepdims_example
func.func @test_reduce_sum_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: torch.aten.Bool.int %int1 : !torch.int -> !torch.bool
  // CHECK: torch.aten.sum.dim_IntList %arg0, %[[NONE]], %0, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_do_not_keepdims_example
func.func @test_reduce_sum_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %false, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_empty_axes_input_noop_example
func.func @test_reduce_sum_empty_axes_input_noop_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[3,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64, torch.onnx.noop_with_empty_axes = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[3,2,2],f32>
  return %0 : !torch.vtensor<[3,2,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_empty_set_non_reduced_axis_zero
func.func @test_reduce_sum_empty_set_non_reduced_axis_zero(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %true, %[[NONE]] : !torch.vtensor<[2,0,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,0,1],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32>
  return %0 : !torch.vtensor<[2,0,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_keepdims_example
func.func @test_reduce_sum_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %true, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_negative_axes_keepdims_example
func.func @test_reduce_sum_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %true, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_mean_default_axes_keepdims_example
func.func @test_reduce_mean_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: torch.aten.Bool.int %int1 : !torch.int -> !torch.bool
  // CHECK: torch.aten.mean.dim %arg0, %[[NONE]], %0, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_mean_do_not_keepdims_example
func.func @test_reduce_mean_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: torch.aten.mean.dim %arg0, %6, %false, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_mean_keepdims_example
func.func @test_reduce_mean_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.mean.dim %arg0, %6, %true, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_mean_negative_axes_keepdims_example
func.func @test_reduce_mean_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.mean.dim %arg0, %6, %true, %[[NONE]] : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_bool_inputs
func.func @test_reduce_min_bool_inputs(%arg0: !torch.vtensor<[4,2],i1>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int2 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,1],i1>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[4,2],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1>
  return %0 : !torch.vtensor<[4,1],i1>
}

// CHECK-LABEL: func.func @test_reduce_min_default_axes_keepdims_example
func.func @test_reduce_min_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: torch.aten.Bool.int %int1 : !torch.int -> !torch.bool
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: torch.prim.ListConstruct %int0, %int1_0, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.amin %arg0, %1, %0 : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_do_not_keepdims_example
func.func @test_reduce_min_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: torch.aten.amin %arg0, %6, %false : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_empty_set
func.func @test_reduce_min_empty_set(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[2,0,4],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,1,4],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32>
  return %0 : !torch.vtensor<[2,1,4],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_keepdims_example
func.func @test_reduce_min_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_negative_axes_keepdims_example
func.func @test_reduce_min_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
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
// CHECK: %[[AXIS:.*]] = torch.constant.int 1
// CHECK: %[[SPLIT_SIZE:.*]] = torch.constant.int 3
// CHECK: %[[SPLIT_RESULT:.*]] = torch.aten.split.Tensor %[[INPUT_TENSOR]], %[[SPLIT_SIZE]], %[[AXIS]] : !torch.vtensor<[2,8],f32>, !torch.int, !torch.int -> !torch.list<vtensor<[2,?],f32>>
// CHECK: %[[UNPACKED_TENSORS:.*]]:3 = torch.prim.ListUnpack %[[SPLIT_RESULT]] : !torch.list<vtensor<[2,?],f32>> -> !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,2],f32>
// CHECK: return %[[UNPACKED_TENSORS]]#0, %[[UNPACKED_TENSORS]]#1, %[[UNPACKED_TENSORS]]#2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,2],f32>
// CHECK: }
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
// CHECK-NEXT: %[[SELECT0:.*]] = torch.aten.index_select %[[ARG1]], %[[ZERO]], %[[SCALAR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
// CHECK-NEXT: %[[ITEM0:.*]] = torch.aten.item %[[SELECT0]] : !torch.vtensor<[1],si64> -> !torch.int
// CHECK-NEXT: %[[SELECT1:.*]] = torch.aten.index_select %[[ARG2]], %[[ZERO]], %[[SCALAR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
// CHECK-NEXT: %[[ITEM1:.*]] = torch.aten.item %[[SELECT1]] : !torch.vtensor<[1],si64> -> !torch.int
// CHECK-NEXT: %[[SELECT3:.*]] = torch.aten.index_select %{{.*}}, %[[ZERO]], %[[SCALAR]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
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
    // CHECK: %[[RESULTS:.*]]:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
    // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
    %0:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
    return %0#0, %0#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
  }

// -----

// CHECK-LABEL: func.func @test_top_k_smallest
  func.func @test_top_k_smallest(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[RESULTS:.*]]:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = 1 : si64, torch.onnx.largest = 0 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
    // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
    %0:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = 1 : si64, torch.onnx.largest = 0 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
    return %0#0, %0#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
  }

// -----

// CHECK-LABEL: func.func @test_top_k_negative_axis
  func.func @test_top_k_negative_axis(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
    // CHECK: %[[RESULTS:.*]]:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>)
    // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
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
