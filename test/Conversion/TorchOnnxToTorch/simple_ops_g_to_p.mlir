// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch --split-input-file | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.

// CHECK-LABEL: func.func @test_greater
func.func @test_greater(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.gt.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Greater"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_greater_or_equal
func.func @test_greater_or_equal(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.ge.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.GreaterOrEqual"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_less
func.func @test_less(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.lt.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Less"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_gather
func.func @test_gather(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[8,10,20,40], si64>) -> !torch.vtensor<[8,10,20,40,4,5],f32> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[AXIS:.+]] = torch.constant.int 0
  // CHECK: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK: %[[LT:.+]] = torch.aten.le.Scalar %arg1, %[[ZERO]]
  // CHECK: %[[SZ:.+]] = torch.aten.size.int %arg0, %[[AXIS]]
  // CHECK: %[[ADD:.+]] = torch.aten.add.Scalar %arg1, %[[SZ]], %[[ONE]]
  // CHECK: %[[SEL:.+]] = torch.aten.where.self %[[LT]], %[[ADD]], %arg1
  // CHECK: %[[SZ:.+]] = torch.aten.size %[[SEL]]
  // CHECK: %[[DIM:.+]] = torch.aten.dim %[[SEL]]
  // CHECK: %[[SUB:.+]] = torch.aten.sub.int %[[DIM]], %[[ONE]]
  // CHECK: %[[FLAT:.+]] = torch.aten.flatten.using_ints %[[SEL]], %[[ZERO]], %[[SUB]]
  // CHECK: %[[ISEL:.+]] = torch.aten.index_select %arg0, %[[AXIS]], %[[FLAT]]
  // CHECK: %[[RES:.+]] = torch.aten.unflatten.int %[[ISEL]], %[[AXIS]], %[[SZ]]
  // CHECK: return %[[RES]]
  %0 = torch.operator "onnx.Gather"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[8,10,20,40], si64>) -> !torch.vtensor<[8,10,20,40,4,5],f32>
  return %0 : !torch.vtensor<[8,10,20,40,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_gather_elements
func.func @test_gather_elements(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5], si64>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[INT0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[GATHER:.+]] = torch.aten.gather %arg0, %[[INT0]], %arg1, %[[FALSE]]
  %0 = torch.operator "onnx.GatherElements"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5], si64>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_default
func.func @test_gemm_default(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[I1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_transposeA
func.func @test_gemm_transposeA(%arg0: !torch.vtensor<[5,3],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[I0:.+]] = torch.constant.int 0
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[TRANS:.+]] = torch.aten.transpose.int %arg0, %[[I0]], %[[I1]] : !torch.vtensor<[5,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.mm %[[TRANS]], %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[I1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.transA = 1 : si64} : (!torch.vtensor<[5,3],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_transposeB
func.func @test_gemm_transposeB(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[4,5],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[I0:.+]] = torch.constant.int 0
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[TRANS:.+]] = torch.aten.transpose.int %arg1, %[[I0]], %[[I1]] : !torch.vtensor<[4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[5,4],f32>
  // CHECK: %[[MM:.+]] = torch.aten.mm %arg0, %[[TRANS]] : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[I1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.transB = 1 : si64} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_alpha
func.func @test_gemm_alpha(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK-DAG: %[[ALPHA:.+]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.add.Tensor %arg2, %[[MM]], %[[ALPHA]] : !torch.vtensor<[1,4],f32>, !torch.vtensor<[3,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.alpha = 5.000000e-01 : f32} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_beta
func.func @test_gemm_beta(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK-DAG: %[[BETA:.+]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[BETA]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.beta = 5.000000e-01 : f32} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_alpha_beta
func.func @test_gemm_alpha_beta(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[I1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK-DAG: %[[ALPHA:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[BETA:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[MUL:.+]] = torch.aten.mul.Scalar %[[MM]], %[[ALPHA]] : !torch.vtensor<[3,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MUL]], %arg2, %[[BETA]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.alpha = 5.000000e-01 : f32, torch.onnx.beta = 2.500000e-01 : f32} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL : func.func @test_layer_norm
func.func @test_layer_norm(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[3,4],f32>, %arg2: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4], f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>)
                           attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %int3 = torch.constant.int 3
  // CHECK: %int4 = torch.constant.int 4
  // CHECK: %0 = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %0, %arg1, %arg2
  %0:3 = torch.operator "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>
}

// -----

// CHECK-LABEL : func.func @test_layer_norm_single_result
func.func @test_layer_norm_single_result(%arg0: !torch.vtensor<[1,4,768],f32>, %arg1: !torch.vtensor<[768],f32>, %arg2: !torch.vtensor<[768],f32>) -> (!torch.vtensor<[1,4,768], f32>)
                           attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %float9.999990e-06 = torch.constant.float 9.9999997473787516E-6
  // CHECK: %int768 = torch.constant.int 768
  // CHECK: %0 = torch.prim.ListConstruct %int768 : (!torch.int) -> !torch.list<int>
  // CHECK: %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %0, %arg1, %arg2
  %0 = torch.operator "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 9.99999974E-6 : f32} : (!torch.vtensor<[1,4,768],f32>, !torch.vtensor<[768],f32>, !torch.vtensor<[768],f32>) -> !torch.vtensor<[1,4,768],f32>
  return %0 : !torch.vtensor<[1,4,768],f32>
}

// -----

// CHECK-LABEL: func.func @test_leaky_relu
func.func @test_leaky_relu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 16 : si64} {
  // CHECK-DAG: %[[F2:.+]] = torch.constant.float 2
  // CHECK: %[[LRELU:.+]] = torch.aten.leaky_relu %arg0, %[[F2]]
  %0 = torch.operator "onnx.LeakyRelu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_matmul_2d
func.func @test_matmul_2d(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,3],f32> -> !torch.vtensor<[3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,3],f32>
  return %0 : !torch.vtensor<[3,3],f32>
}

// -----

// CHECK-LABEL: @test_matmul_3d
func.func @test_matmul_3d(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,4,3],f32> -> !torch.vtensor<[2,3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,3,3],f32>
  return %0 : !torch.vtensor<[2,3,3],f32>
}

// -----

// CHECK-LABEL: @test_matmul_4d
func.func @test_matmul_4d(%arg0: !torch.vtensor<[1,2,3,4],f32>, %arg1: !torch.vtensor<[1,2,4,3],f32>) -> !torch.vtensor<[1,2,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,4,3],f32> -> !torch.vtensor<[1,2,3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,4,3],f32>) -> !torch.vtensor<[1,2,3,3],f32>
  return %0 : !torch.vtensor<[1,2,3,3],f32>
}

// -----

// CHECK-LABEL: @test_matmulinteger
func.func @test_matmulinteger(%arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],ui8>, %arg2: !torch.vtensor<[1],ui8>, %arg3: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,2],si32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],ui8>, !torch.vtensor<[1],ui8>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,2],si32>
  // CHECK: %[[LITEM:.+]] = torch.aten.item %arg2
  // CHECK: %[[RITEM:.+]] = torch.aten.item %arg3
  // CHECK: %[[SCALE:.+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[LMAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[LITEM]] : !torch.vtensor<[4,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[4,3],!torch.quint8>
  // CHECK: %[[RMAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[SCALE]], %[[RITEM]] : !torch.vtensor<[3,2],ui8>, !torch.float, !torch.int -> !torch.vtensor<[3,2],!torch.quint8>
  // CHECK: %[[MM:.+]] = torch.aten.mm %[[LMAKE]], %[[RMAKE]]
  // CHECK: return %[[MM]]
  return %0 : !torch.vtensor<[4,2],si32>
}

// -----

// CHECK-LABEL: func.func @test_mul
  func.func @test_mul(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.Mul"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_maxpool_2d_default
func.func @test_maxpool_2d_default(%arg0: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,31,31],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[I2_1:.*]] = torch.constant.int 2
  // CHECK: %[[LIST22:.*]] = torch.prim.ListConstruct %[[I2]], %[[I2_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I0_0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[LIST0:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_0:.*]] = torch.constant.int 1
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[LIST1:.*]] = torch.prim.ListConstruct %[[I1_0]], %[[I1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_2:.*]] = torch.constant.int 1
  // CHECK: %[[I1_3:.*]] = torch.constant.int 1
  // CHECK: %[[LIST3:.*]] = torch.prim.ListConstruct %[[I1_2]], %[[I1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.max_pool2d %arg0, %[[LIST22]], %[[LIST1]], %[[LIST0]], %[[LIST3]], %[[FALSE]] : !torch.vtensor<[1,3,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,31,31],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,31,31],f32>
  return %0 : !torch.vtensor<[1,3,31,31],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_2d_ceil
func.func @test_maxpool_2d_ceil(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[I3:.*]] = torch.constant.int 3
  // CHECK: %[[I3_1:.*]] = torch.constant.int 3
  // CHECK: %[[LIST33:.*]] = torch.prim.ListConstruct %[[I3]], %[[I3_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I0_0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[LIST0:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[I2_1:.*]] = torch.constant.int 2
  // CHECK: %[[LIST22:.*]] = torch.prim.ListConstruct %[[I2]], %[[I2_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_0:.*]] = torch.constant.int 1
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[I1_0]], %[[I1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.max_pool2d %arg0, %[[LIST33]], %[[LIST22]], %[[LIST0]], %[[LIST]], %[[TRUE]] : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,2,2],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 1 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32>
  return %0 : !torch.vtensor<[1,1,2,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_3d_default
func.func @test_maxpool_3d_default(%arg0: !torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[I2_1:.*]] = torch.constant.int 2
  // CHECK: %[[I2_2:.*]] = torch.constant.int 2
  // CHECK: %[[LIST222:.*]] = torch.prim.ListConstruct %[[I2]], %[[I2_1]], %[[I2_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I0_0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[I0_2:.*]] = torch.constant.int 0
  // CHECK: %[[LIST0:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]], %[[I0_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_0:.*]] = torch.constant.int 1
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[I1_2:.*]] = torch.constant.int 1
  // CHECK: %[[LIST1:.*]] = torch.prim.ListConstruct %[[I1_0]], %[[I1_1]], %[[I1_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_3:.*]] = torch.constant.int 1
  // CHECK: %[[I1_4:.*]] = torch.constant.int 1
  // CHECK: %[[I1_5:.*]] = torch.constant.int 1
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[I1_3]], %[[I1_4]], %[[I1_5]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.max_pool3d %arg0, %[[LIST222]], %[[LIST1]], %[[LIST0]], %[[LIST]], %[[FALSE]] : !torch.vtensor<[1,3,32,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,31,31,31],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32>
  return %0 : !torch.vtensor<[1,3,31,31,31],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_pad
func.func @test_maxpool_pad(%arg0: !torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,56,56],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INT2_0:.+]] = torch.constant.int 2
  // CHECK: %[[INT2_1:.+]] = torch.constant.int 2
  // CHECK: %[[PADI:.+]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]], %[[INT2_0]], %[[INT2_1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[MIN:.+]] = torch.constant.float -1.7976931348623157E+308
  // CHECK: %[[PADDED:.+]] = torch.aten.constant_pad_nd %arg0, %[[PADI]], %[[MIN]] : !torch.vtensor<[1,64,111,111],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,64,114,114],f32>
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT3_0:.*]] = torch.constant.int 3
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: %[[LIST2:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[INT2_4:.*]] = torch.constant.int 2
  // CHECK: %[[LIST3:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2_4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK: %[[EMPTY_LIST:.*]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[OUT:.*]] = torch.aten.max_pool2d %[[PADDED]], %[[LIST]], %[[LIST3]], %[[LIST2]], %[[EMPTY_LIST]], %[[FALSE]] : !torch.vtensor<[1,64,114,114],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
  // CHECK: return %[[OUT]] : !torch.vtensor<[1,64,56,56],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 2 : si64, 2 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,56,56],f32>
  return %0 : !torch.vtensor<[1,64,56,56],f32>
}


// -----

// CHECK-LABEL: func.func @test_maxpool_symmetric_pad
func.func @test_maxpool_symmetric_pad(%arg0: !torch.vtensor<[1,64,112,112],f32>) -> !torch.vtensor<[1,64,56,56],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT3_0:.*]] = torch.constant.int 3
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK: %[[LIST2:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[INT2_4:.*]] = torch.constant.int 2
  // CHECK: %[[LIST3:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2_4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_3:.*]] = torch.constant.int 1
  // CHECK: %[[DILATION:.*]] = torch.prim.ListConstruct %[[INT1_2]], %[[INT1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[OUT:.*]] = torch.aten.max_pool2d %arg0, %[[LIST]], %[[LIST3]], %[[LIST2]], %[[DILATION]], %[[FALSE]] : !torch.vtensor<[1,64,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
  // CHECK: return %[[OUT]] : !torch.vtensor<[1,64,56,56],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,64,112,112],f32>) -> !torch.vtensor<[1,64,56,56],f32>
  return %0 : !torch.vtensor<[1,64,56,56],f32>
}

// -----

// CHECK-LABEL: @test_gelu_default_1
func.func @test_gelu_default_1(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "none"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3],f32>, !torch.str -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_gelu_default_2
func.func @test_gelu_default_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "none"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3,4,5],f32>, !torch.str -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_gelu_tanh_1
func.func @test_gelu_tanh_1(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "tanh"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3],f32>, !torch.str -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) {torch.onnx.approximate = "tanh"} : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_gelu_tanh_2
func.func @test_gelu_tanh_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "tanh"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3,4,5],f32>, !torch.str -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) {torch.onnx.approximate = "tanh"} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_less_or_equal
func.func @test_less_or_equal(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.le.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.LessOrEqual"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_pad
func.func @test_pad(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4], si64>, %arg2: !torch.vtensor<[], f32>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT_0:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[SELECT_1:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[SELECT_2:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT2]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[SELECT_3:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT3]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[LIST:.+]] = torch.prim.tolist(%[[SELECT_1]], %[[SELECT_3]], %[[SELECT_0]], %[[SELECT_2]]) : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64> -> !torch.list<int>
  // CHECK: %[[STR:.+]] = torch.constant.str "constant"
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[INT7:.+]] = torch.constant.int 7
  // CHECK: %[[CONVERT:.+]] = torch.aten.to.dtype %arg2, %[[INT7]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
  // CHECK: %[[ITEM:.+]] = torch.aten.item %[[CONVERT]] : !torch.vtensor<[],f64> -> !torch.float
  // CHECK: %[[PAD:.+]] = torch.aten.pad %arg0, %[[LIST]], %[[STR]], %[[ITEM]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[5,4],f32>
  // CHECK: return %[[PAD]] : !torch.vtensor<[5,4],f32>
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1, %arg2) {torch.onnx.mode = "constant"} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4], si64>, !torch.vtensor<[], f32>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

// CHECK-LABEL: @test_pad_optional_constant
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>
// CHECK-SAME:    %[[ARG1:.*]]: !torch.vtensor<[4],si64>
// CHECK: %[[CONST_STR:.*]] = torch.constant.str "constant"
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[FALSE:.*]] = torch.constant.bool false
// CHECK: %[[SEVEN:.*]] = torch.constant.int 7
// CHECK: %[[DTYPE:.*]] = torch.aten.to.dtype %0, %[[SEVEN]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],f64>
// CHECK: %[[ITEM:.*]] = torch.aten.item %[[DTYPE]] : !torch.vtensor<[],f64> -> !torch.float
// CHECK: torch.aten.pad %[[ARG0]], %{{.*}}, %[[CONST_STR]], %[[ITEM]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[5,4],f32>

func.func @test_pad_optional_constant(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1) {torch.onnx.mode = "constant"} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_pow
  func.func @test_pow(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.Pow"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: @test_hardsigmoid_example
func.func @test_hardsigmoid_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[ALPHA_FLOAT:.*]] = torch.constant.float 5.000000e-01
  // CHECK: %[[BETA_FLOAT:.*]] = torch.constant.float 0.60000002384185791
  // CHECK: %[[ALPHA_MULTI_X_PLUS_BETA:.*]] = torch.aten.add.Scalar %arg0, %[[BETA_FLOAT:.*]], %[[ALPHA_FLOAT:.*]] : !torch.vtensor<[3],f32>, !torch.float, !torch.float -> !torch.vtensor<[3],f32>
  // CHECK: %[[INT_1:.*]] = torch.constant.int 1
  // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[NONE_FOR_ONE:.*]] = torch.constant.none
  // CHECK: %[[INT_TYPE_FOR_TENSOR_ONE:.*]] = torch.constant.int 6
  // CHECK: %[[ONE_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]], %[[INT_1:.*]], %[[INT_TYPE_FOR_TENSOR_ONE:.*]], %[[NONE_FOR_ONE:.*]], %[[NONE_1:.*]], %[[NONE_1:.*]] : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[MIN_EXPRESSION:.*]] = torch.aten.minimum %[[ONE_TENSOR:.*]], %[[ALPHA_MULTI_X_PLUS_BETA:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  // CHECK: %[[INT_0:.*]] = torch.constant.int 0
  // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[NONE_FOR_ZERO:.*]] = torch.constant.none
  // CHECK: %[[INT_TYPE_FOR_TENSOR_ZERO:.*]] = torch.constant.int 6
  // CHECK: %[[ZERO_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]], %[[INT_0:.*]], %[[INT_TYPE_FOR_TENSOR_ZERO:.*]], %[[NONE_FOR_ZERO:.*]], %none_0, %none_0 : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[RESULT:.*]] = torch.aten.maximum %[[ZERO_TENSOR:.*]], %[[MIN_EXPRESSION:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  // CHECK: return %[[RESULT:.*]] : !torch.vtensor<[3],f32>

  %0 = torch.operator "onnx.HardSigmoid"(%arg0) {torch.onnx.alpha = 5.000000e-01 : f32, torch.onnx.beta = 6.000000e-01 : f32} : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_hardsigmoid
func.func @test_hardsigmoid(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[ALPHA_FLOAT:.*]] = torch.constant.float 5.000000e-01
    // CHECK: %[[BETA_FLOAT:.*]] = torch.constant.float 0.60000002384185791
    // CHECK: %[[ALPHA_MULTI_X_PLUS_BETA:.*]] = torch.aten.add.Scalar %arg0, %[[BETA_FLOAT:.*]], %[[ALPHA_FLOAT:.*]] : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[INT_1:.*]] = torch.constant.int 1
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ONE:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ONE:.*]] = torch.constant.int 6
    // CHECK: %[[ONE_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]], %[[INT_1:.*]], %[[INT_TYPE_FOR_TENSOR_ONE:.*]], %[[NONE_FOR_ONE:.*]], %[[NONE_1:.*]], %[[NONE_1:.*]] : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: %[[MIN_EXPRESSION:.*]] = torch.aten.minimum %[[ONE_TENSOR:.*]], %[[ALPHA_MULTI_X_PLUS_BETA:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[INT_0:.*]] = torch.constant.int 0
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ZERO:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ZERO:.*]] = torch.constant.int 6
    // CHECK: %[[ZERO_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]], %[[INT_0:.*]], %[[INT_TYPE_FOR_TENSOR_ZERO:.*]], %[[NONE_FOR_ZERO:.*]], %none_0, %none_0 : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: %[[RESULT:.*]] = torch.aten.maximum %[[ZERO_TENSOR:.*]], %[[MIN_EXPRESSION:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    // CHECK: return %[[RESULT:.*]] : !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.HardSigmoid"(%arg0) {torch.onnx.alpha = 5.000000e-01 : f32, torch.onnx.beta = 6.000000e-01 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_hardsigmoid_default
func.func @test_hardsigmoid_default(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[ALPHA_FLOAT:.*]] = torch.constant.float 0.20000000298023224
    // CHECK: %[[BETA_FLOAT:.*]] = torch.constant.float 5.000000e-01
    // CHECK: %[[ALPHA_MULTI_X_PLUS_BETA:.*]] = torch.aten.add.Scalar %arg0, %[[BETA_FLOAT:.*]], %[[ALPHA_FLOAT:.*]] : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[INT_1:.*]] = torch.constant.int 1
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ONE:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ONE:.*]] = torch.constant.int 6
    // CHECK: %[[ONE_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]], %[[INT_1:.*]], %[[INT_TYPE_FOR_TENSOR_ONE:.*]], %[[NONE_FOR_ONE:.*]], %[[NONE_1:.*]], %[[NONE_1:.*]] : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: %[[MIN_EXPRESSION:.*]] = torch.aten.minimum %[[ONE_TENSOR:.*]], %[[ALPHA_MULTI_X_PLUS_BETA:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[INT_0:.*]] = torch.constant.int 0
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ZERO:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ZERO:.*]] = torch.constant.int 6
    // CHECK: %[[ZERO_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]], %[[INT_0:.*]], %[[INT_TYPE_FOR_TENSOR_ZERO:.*]], %[[NONE_FOR_ZERO:.*]], %none_0, %none_0 : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: torch.aten.maximum %[[ZERO_TENSOR:.*]], %[[MIN_EXPRESSION:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.HardSigmoid"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_globalaveragepool
func.func @test_globalaveragepool(%arg0: !torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[C5_0:.*]] = torch.constant.int 5
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C5]], %[[C5_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.avg_pool2d %arg0, %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[1,3,5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,1,1],f32>
  %0 = torch.operator "onnx.GlobalAveragePool"(%arg0) : (!torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32>
  return %0 : !torch.vtensor<[1,3,1,1],f32>
}

// -----

// CHECK-LABEL: @test_globalaveragepool_precomputed
func.func @test_globalaveragepool_precomputed(%arg0: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C3]], %[[C3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.avg_pool2d %arg0, %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[1,1,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1,1],f32>
  %0 = torch.operator "onnx.GlobalAveragePool"(%arg0) : (!torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_max_example
  func.func @test_max_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.maximum %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Max"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_min_example
  func.func @test_min_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.minimum %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Min"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_log
  func.func @test_log(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.log %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.Log"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_neg
  func.func @test_neg(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.neg %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.Neg"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_instancenorm
  func.func @test_instancenorm(%arg0: !torch.vtensor<[1,2,1,3],f32>, %arg1: !torch.vtensor<[2],f32>, %arg2: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2,1,3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.instance_norm %arg0, %arg1, %arg2, %none, %none, %true, %float0.000000e00, %float9.999990e-06, %false : !torch.vtensor<[1,2,1,3],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.none, !torch.none, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[1,2,1,3],f32>
    %0 = torch.operator "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,2,1,3],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2,1,3],f32>
    return %0 : !torch.vtensor<[1,2,1,3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_not_2d
func.func @test_not_2d(%arg0: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.bitwise_not %arg0 : !torch.vtensor<[3,4],i1> -> !torch.vtensor<[3,4],i1>
    %0 = torch.operator "onnx.Not"(%arg0) : (!torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1>
    return %0 : !torch.vtensor<[3,4],i1>
  }

// -----

// CHECK-LABEL: func.func @test_nonzero
  func.func @test_nonzero(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.nonzero %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],si64>
    %0 = torch.operator "onnx.NonZero"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],si64>
    return %0 : !torch.vtensor<[3,4,5],si64>
  }

// -----

// CHECK-LABEL: func.func @test_or2d
  func.func @test_or2d(%arg0: !torch.vtensor<[3,4],i1>, %arg1: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],i1>, !torch.vtensor<[3,4],i1> -> !torch.vtensor<[3,4],i1>
    %0 = torch.operator "onnx.Or"(%arg0, %arg1) : (!torch.vtensor<[3,4],i1>, !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1>
    return %0 : !torch.vtensor<[3,4],i1>
  }

// CHECK-LABEL: func.func @test_identity
  func.func @test_identity(%arg0: !torch.vtensor<[3,4], f32>) -> !torch.vtensor<[3,4], f32> attributes {torch.onnx_meta.ir_version = 14 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[NONE:.*]] = torch.constant.none
    // CHECK: %0 = torch.aten.clone %arg0, %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.none -> !torch.vtensor<[3,4],f32>
    %0 = torch.operator "onnx.Identity"(%arg0) : (!torch.vtensor<[3,4], f32>) -> !torch.vtensor<[3,4], f32>
    return %0 : !torch.vtensor<[3,4], f32>
  }

// CHECK-LABEL: func.func @test_mean_one_input
  func.func @test_mean_one_input(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Mean"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// CHECK-LABEL: func.func @test_mean_two_inputs
  func.func @test_mean_two_inputs(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[INT2:.*]] = torch.constant.int 2
    // CHECK: %[[INT1:.*]] = torch.constant.int 1
    // CHECK: torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
    // CHECK: torch.aten.div.Scalar %0, %int2 : !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Mean"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// CHECK-LABEL: func.func @test_isinf_negative
  func.func @test_isinf_negative(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.neg %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],f32>
    // CHECK: torch.aten.relu %0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],f32>
    // CHECK: torch.aten.isinf %1 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],i1>
    %0 = torch.operator "onnx.IsInf"(%arg0) {torch.onnx.detect_positive = 0 : si64} : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1>
    return %0 : !torch.vtensor<[6],i1>
  }

// CHECK-LABEL: func.func @test_isinf_positive
  func.func @test_isinf_positive(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.relu %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],f32>
    // CHECK: torch.aten.isinf %0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],i1>
    %0 = torch.operator "onnx.IsInf"(%arg0) {torch.onnx.detect_negative = 0 : si64} : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1>
    return %0 : !torch.vtensor<[6],i1>
  }

// CHECK-LABEL: func.func @test_isnan
  func.func @test_isnan(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.isnan %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],i1>
    %0 = torch.operator "onnx.IsNaN"(%arg0) : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1>
    return %0 : !torch.vtensor<[6],i1>
  }

// CHECK-LABEL: func.func @test_prelu_example
  func.func @test_prelu_example(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.prelu %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.PRelu"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// CHECK-LABEL: func.func @test_prelu_broadcast
  func.func @test_prelu_broadcast(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.prelu %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.PRelu"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
