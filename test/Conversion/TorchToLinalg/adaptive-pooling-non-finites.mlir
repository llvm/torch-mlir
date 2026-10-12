// RUN: torch-mlir-opt %s -convert-torch-to-linalg -canonicalize | FileCheck %s --check-prefix=INF
// RUN: torch-mlir-opt %s -convert-torch-to-linalg=allow-non-finites=false -canonicalize | FileCheck %s --check-prefix=FINITE

// Both modes must reach the adaptive-pooling pattern's constructor.
// INF-LABEL: func.func @adaptive_max_1d_f16
// INF: arith.constant 0xFC00 : f16
// FINITE-LABEL: func.func @adaptive_max_1d_f16
// FINITE: arith.constant -6.550400e+04 : f16
func.func @adaptive_max_1d_f16(%input: !torch.vtensor<[1,2,5],f16>) -> !torch.vtensor<[1,2,3],f16> {
  %three = torch.constant.int 3
  %size = torch.prim.ListConstruct %three : (!torch.int) -> !torch.list<int>
  %value, %index = torch.aten.adaptive_max_pool1d %input, %size : !torch.vtensor<[1,2,5],f16>, !torch.list<int> -> !torch.vtensor<[1,2,3],f16>, !torch.vtensor<[1,2,3],si64>
  return %value : !torch.vtensor<[1,2,3],f16>
}

// INF-LABEL: func.func @adaptive_max_2d_f32
// INF: arith.constant 0xFF800000 : f32
// FINITE-LABEL: func.func @adaptive_max_2d_f32
// FINITE: arith.constant -3.40282347E+38 : f32
func.func @adaptive_max_2d_f32(%input: !torch.vtensor<[1,2,5,5],f32>) -> !torch.vtensor<[1,2,3,3],f32> {
  %three = torch.constant.int 3
  %size = torch.prim.ListConstruct %three, %three : (!torch.int, !torch.int) -> !torch.list<int>
  %value, %index = torch.aten.adaptive_max_pool2d %input, %size : !torch.vtensor<[1,2,5,5],f32>, !torch.list<int> -> !torch.vtensor<[1,2,3,3],f32>, !torch.vtensor<[1,2,3,3],si64>
  return %value : !torch.vtensor<[1,2,3,3],f32>
}

// INF-LABEL: func.func @adaptive_max_3d_bf16
// INF: arith.constant 0xFF80 : bf16
// FINITE-LABEL: func.func @adaptive_max_3d_bf16
// FINITE: arith.constant -3.389530e+38 : bf16
func.func @adaptive_max_3d_bf16(%input: !torch.vtensor<[1,2,5,5,5],bf16>) -> !torch.vtensor<[1,2,3,3,3],bf16> {
  %three = torch.constant.int 3
  %size = torch.prim.ListConstruct %three, %three, %three : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %value, %index = torch.aten.adaptive_max_pool3d %input, %size : !torch.vtensor<[1,2,5,5,5],bf16>, !torch.list<int> -> !torch.vtensor<[1,2,3,3,3],bf16>, !torch.vtensor<[1,2,3,3,3],si64>
  return %value : !torch.vtensor<[1,2,3,3,3],bf16>
}
