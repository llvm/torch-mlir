// RUN: torch-mlir-opt --split-input-file --evaluate-randn-ops="random-values="1.1,2.1,3.1,4.1,5.1,6.1"" %s | FileCheck %s

// CHECK-LABEL: func.func @evaluateRandnOp
func.func @evaluateRandnOp() -> !torch.vtensor<[2,3],f32> {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %none_0 = torch.constant.none
  %cpu = torch.constant.device "cpu"
  %false = torch.constant.bool false
  // CHECK-DAG: torch.vtensor.literal(dense<{{\[\[}}1.100000e+00, 2.100000e+00, 3.100000e+00], [4.100000e+00, 5.100000e+00, 6.100000e+00{{\]\]}}> : tensor<2x3xf32>) : !torch.vtensor<{{\[}}2,3{{\]}},f32>
  %1 = torch.aten.randn %0, %none, %none_0, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[2,3],f32>           
  return %1 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL: func.func @evaluateTwoRandnOps
func.func @evaluateTwoRandnOps() -> !torch.vtensor<[1,3],f32> {
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int1, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %none_0 = torch.constant.none
  %cpu = torch.constant.device "cpu"
  %false = torch.constant.bool false
  
  // CHECK-DAG: torch.vtensor.literal(dense<{{\[\[}}5.200000e+00, 7.1999998, 9.1999998{{\]\]}}> : tensor<1x3xf32>) : !torch.vtensor<{{\[}}1,3{{\]}},f32>
  %1 = torch.aten.randn %0, %none, %none_0, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1,3],f32>           
  %2 = torch.aten.randn %0, %none, %none_0, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1,3],f32>           
  %3 = torch.aten.add.Tensor %1, %2, %int1 : !torch.vtensor<[1,3],f32>, !torch.vtensor<[1,3],f32>, !torch.int -> !torch.vtensor<[1,3],f32>

  return %3 : !torch.vtensor<[1,3],f32>
}
