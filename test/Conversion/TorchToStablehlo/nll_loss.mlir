// RUN: torch-mlir-opt <%s --torchdynamo-export-to-torch-backend-pipeline --torch-backend-to-stablehlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

module {
  func.func @main(%arg0: !torch.vtensor<[3,5],bf16>, %arg1: !torch.vtensor<[3],si64>, %arg2: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> {
    // CHECK: %[[CST_0:.*]] = stablehlo.constant dense<0> : tensor<1xi64>
    %int1 = torch.constant.int 1
    %int-100 = torch.constant.int -100
    // CHECK: %[[VAL_1:.*]] = stablehlo.convert %[[CST_0]] : (tensor<1xi64>) -> tensor<1xbf16>
    // CHECK: %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_1]], dims = [0] : (tensor<1xbf16>) -> tensor<3xbf16>
    // CHECK: %{{.*}} = stablehlo.select %{{.*}}, %{{.*}}, %[[VAL_2]] : tensor<3xi1>, tensor<3xbf16>
    %output, %total_weight = torch.aten.nll_loss_forward %arg0, %arg1, %arg2, %int1, %int-100 : !torch.vtensor<[3,5],bf16>, !torch.vtensor<[3],si64>, !torch.vtensor<[5],f32>, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
    return %output : !torch.vtensor<[],f32>
  }
}