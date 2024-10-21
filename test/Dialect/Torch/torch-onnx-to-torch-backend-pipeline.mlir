// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-onnx-to-torch-backend-pipeline{backend-legal-ops=aten.flatten.using_ints,aten.unflatten.int})' -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_reshape_negative_dim_decompose
func.func @test_reshape_negative_dim_decompose(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,6,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT6:.+]] = torch.constant.int 6
  // CHECK: %[[RESULT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT2]], %[[INT6]], %[[INT2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.view %arg0, %[[RESULT_SHAPE]] : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[2,6,2],f32>
  %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,6,2],f32>
  return %0 : !torch.vtensor<[2,6,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_triu_decompose
func.func @test_triu_decompose(%arg0: !torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[ZERO_TENSOR:.+]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT4:.+]] = torch.constant.int 4
  // CHECK: %[[INT5:.+]] = torch.constant.int 5
  // CHECK: %[[ARANGE:.+]] = torch.aten.arange.start_step %[[INT0]], %[[INT4]], %[[INT1]], %[[INT4]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4],si64>
  // CHECK: %[[ARANGE_0:.+]] = torch.aten.arange.start_step %[[INT0]], %[[INT5]], %[[INT1]], %[[INT4]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[5],si64>
  // CHECK: %[[UNSQUEEZE:.+]] = torch.aten.unsqueeze %[[ARANGE]], %[[INT1]] : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
  // CHECK: %[[UNSQUEEZE_0:.+]] = torch.aten.unsqueeze %[[ARANGE_0]], %[[INT0]] : !torch.vtensor<[5],si64>, !torch.int -> !torch.vtensor<[1,5],si64>
  // CHECK: %[[ADD:.+]] = torch.aten.add.Scalar %[[UNSQUEEZE]], %[[INT0]], %[[INT1]] : !torch.vtensor<[4,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,1],si64>
  // CHECK: %[[COND:.+]] = torch.aten.ge.Tensor %[[UNSQUEEZE_0]], %[[ADD]] : !torch.vtensor<[1,5],si64>, !torch.vtensor<[4,1],si64> -> !torch.vtensor<[4,5],i1>
  // CHECK: %[[RESULT:.+]] = torch.aten.where.self %[[COND]], %arg0, %[[ZERO_TENSOR]] : !torch.vtensor<[4,5],i1>, !torch.vtensor<[4,5],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[4,5],si64>
  %0 = torch.operator "onnx.Trilu"(%arg0) : (!torch.vtensor<[4,5],si64>) -> !torch.vtensor<[4,5],si64>
  return %0 : !torch.vtensor<[4,5],si64>
}

// -----

module {
// CHECK-LABEL: func.func @test_scalarize
  func.func @test_scalarize(%arg0: !torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.11.0"} {
    // CHECK: %[[INT2:.+]] = torch.constant.int 2
    // CHECK: %[[INT3:.+]] = torch.constant.int 3
    // CHECK: %[[ADD:.+]] = torch.aten.flatten.using_ints %arg0, %[[INT2]], %[[INT3]] : !torch.vtensor<[?,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,1024],f32>
    %0 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[4],si64>
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__21> : tensor<si64>} : () -> !torch.vtensor<[],si64>
    %2 = torch.operator "onnx.Gather"(%0, %1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[4],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64>
    %3 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[4],si64>
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__22> : tensor<si64>} : () -> !torch.vtensor<[],si64>
    %5 = torch.operator "onnx.Gather"(%3, %4) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[4],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64>
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64>
    %7 = torch.operator "onnx.Unsqueeze"(%2, %6) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64>
    %8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64>
    %9 = torch.operator "onnx.Unsqueeze"(%5, %8) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64>
    %10 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_onnx__Concat_3209> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64>
    %11 = torch.operator "onnx.Concat"(%7, %9, %10) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],si64>
    %12 = torch.operator "onnx.Reshape"(%arg0, %11) : (!torch.vtensor<[?,?,16,64],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[?,?,?],f32>
    return %12 : !torch.vtensor<[?,?,?],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __21: "0x080000000000000000000000",
      __22: "0x080000000100000000000000",
      _onnx__Concat_3209: "0x080000000004000000000000"
    }
  }
#-}
