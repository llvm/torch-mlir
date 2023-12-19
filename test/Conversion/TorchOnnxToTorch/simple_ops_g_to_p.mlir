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

// CHECK-LABEL: func.func @test_greater_or_equal
func.func @test_greater_or_equal(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.ge.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.GreaterOrEqual"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// CHECK-LABEL: func.func @test_less
func.func @test_less(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.lt.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Less"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

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

// CHECK-LABEL: func.func @test_leaky_relu
func.func @test_leaky_relu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 16 : si64} {
  // CHECK-DAG: %[[F2:.+]] = torch.constant.float 2
  // CHECK: %[[LRELU:.+]] = torch.aten.leaky_relu %arg0, %[[F2]]
  %0 = torch.operator "onnx.LeakyRelu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

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