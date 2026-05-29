// RUN: torch-mlir-opt -torch-decompose-complex-ops="legal-ops=aten.softmax.int" -split-input-file %s | FileCheck %s --check-prefix=SOFTMAX
// RUN: torch-mlir-opt -torch-decompose-complex-ops="legal-ops=aten.as_strided" -split-input-file %s | FileCheck %s --check-prefix=ASSTRIDED
// RUN: torch-mlir-opt -torch-decompose-complex-ops="legal-ops=aten.as_strided,aten.select.int" -split-input-file %s | FileCheck %s --check-prefix=BOTHLEGAL

// SOFTMAX-LABEL: func.func @torch.aten.softmax.int$cst_dim
func.func @torch.aten.softmax.int$cst_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.tensor<[2,3],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  // SOFTMAX: torch.aten.softmax.int
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}

// -----

// ASSTRIDED-LABEL: func.func @torch.aten.as_strided$legal_after_select(
// ASSTRIDED-SAME: %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
// ASSTRIDED: %[[C4:.*]] = torch.constant.int 4
// ASSTRIDED: %[[SIZES:.*]] = torch.prim.ListConstruct
// ASSTRIDED: %[[STRIDES:.*]] = torch.prim.ListConstruct
// ASSTRIDED-NOT: torch.aten.select.int
// ASSTRIDED: torch.aten.as_strided %[[ARG0]], %[[SIZES]], %[[STRIDES]], %[[C4]]
// ASSTRIDED-NOT: torch.aten.index.Tensor_hacked_twin
// BOTHLEGAL-LABEL: func.func @torch.aten.as_strided$legal_after_select
// BOTHLEGAL: %[[SELECT:.*]] = torch.aten.select.int
// BOTHLEGAL: torch.aten.as_strided %[[SELECT]],
// BOTHLEGAL-NOT: torch.aten.index.Tensor_hacked_twin
func.func @torch.aten.as_strided$legal_after_select(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %select = torch.aten.select.int %arg0, %int0, %int1 : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4],f32>
  %r = torch.aten.as_strided %select, %sizes, %strides, %none : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}
