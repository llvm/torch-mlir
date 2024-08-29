// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// -----

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func.func @sum(
// CHECK-SAME:  %[[A:.*]]: !torch.vtensor<[64,64],f32,#[[$CSR]]>) -> !torch.vtensor<[],f32>
// CHECK:       %[[S:.*]] = torch_c.to_builtin_tensor %[[A]] : !torch.vtensor<[64,64],f32,#[[$CSR]]> -> tensor<64x64xf32, #[[$CSR]]>
// CHECK:       linalg.generic {{{.*}}} ins(%[[S]] : tensor<64x64xf32, #[[$CSR]]>)
func.func @sum(%arg0: !torch.vtensor<[64,64],f32,#CSR>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none
     : !torch.vtensor<[64,64],f32,#CSR>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func.func @SpMM(
// CHECK-SAME:  %[[A:.*]]: !torch.vtensor<[8,16],f32,#[[$CSR]]>,
// CHECK-SAME:  %[[B:.*]]: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32>
// CHECK-DAG:   %[[S:.*]] = torch_c.to_builtin_tensor %[[A]] : !torch.vtensor<[8,16],f32,#[[$CSR]]> -> tensor<8x16xf32, #[[$CSR]]>
// CHECK-DAG:   %[[T:.*]] = torch_c.to_builtin_tensor %[[B]] : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
// CHECK:       linalg.matmul ins(%[[S]], %[[T]] : tensor<8x16xf32, #[[$CSR]]>, tensor<16x8xf32>)
func.func @SpMM(%arg0: !torch.vtensor<[8,16],f32,#CSR>,
                %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
  %0 = torch.aten.matmul %arg0, %arg1
     : !torch.vtensor<[8,16],f32,#CSR>,
       !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}

// -----

#sparse = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3, d4) ->
    (d0 : compressed(nonunique),
     d1 : singleton(nonunique, soa),
     d2 : singleton(nonunique, soa),
     d3 : singleton(nonunique, soa),
     d4 : singleton(soa)
    ),
    posWidth = 64,
    crdWidth = 64
}>

// CHECK: #[[$ST:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3, d4) -> (d0 : compressed(nonunique), d1 : singleton(nonunique, soa), d2 : singleton(nonunique, soa), d3 : singleton(nonunique, soa), d4 : singleton(soa)), posWidth = 64, crdWidth = 64 }>
// CHECK-LABEL: func.func @activate(
// CHECK-SAME:  %[[A:.*]]: !torch.vtensor<[128,64,30,30,6],f32>)
// CHECK:       %[[D:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[128,64,30,30,6],f32> -> tensor<128x64x30x30x6xf32>
// CHECK:       %[[C:.*]] = sparse_tensor.convert %0 : tensor<128x64x30x30x6xf32> to tensor<128x64x30x30x6xf32, #[[$ST]]>
// CHECK:       %[[R:.*]] = torch_c.from_builtin_tensor %[[C]] : tensor<128x64x30x30x6xf32, #[[$ST]]> -> !torch.vtensor<[128,64,30,30,6],f32,#[[$ST]]>
// CHECK:       return %[[R]] : !torch.vtensor<[128,64,30,30,6],f32,#[[$ST]]>
func.func @activate(%arg0: !torch.vtensor<[128,64,30,30,6],f32>)
                        -> !torch.vtensor<[128,64,30,30,6],f32,#sparse> {
  %none_0 = torch.constant.none
  %none_1 = torch.constant.none
  %none_2 = torch.constant.none
  %result = torch.operator "torch.aten._to_sparse"(%arg0, %none_0, %none_1, %none_2)
    : (!torch.vtensor<[128,64,30,30,6],f32>, !torch.none, !torch.none, !torch.none)
    -> !torch.vtensor<[128,64,30,30,6],f32,#sparse>
  return %result : !torch.vtensor<[128,64,30,30,6],f32,#sparse>
}

// -----

#sparse = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3, d4) ->
    (d0 : compressed(nonunique),
     d1 : singleton(nonunique, soa),
     d2 : singleton(nonunique, soa),
     d3 : singleton(nonunique, soa),
     d4 : singleton(soa)
    ),
    posWidth = 64,
    crdWidth = 64
}>

// CHECK: #[[$ST:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3, d4) -> (d0 : compressed(nonunique), d1 : singleton(nonunique, soa), d2 : singleton(nonunique, soa), d3 : singleton(nonunique, soa), d4 : singleton(soa)), posWidth = 64, crdWidth = 64 }>
// CHECK-LABEL: func.func @deactivate(
// CHECK-SAME:  %[[A:.*]]: !torch.vtensor<[128,64,30,30,6],f32,#[[$ST]]>)
// CHECK:       %[[D:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[128,64,30,30,6],f32,#[[$ST]]> -> tensor<128x64x30x30x6xf32, #[[$ST]]>
// CHECK:       %[[C:.*]] = sparse_tensor.convert %0 : tensor<128x64x30x30x6xf32, #[[$ST]]> to tensor<128x64x30x30x6xf32>
// CHECK:       %[[R:.*]] = torch_c.from_builtin_tensor %[[C]] : tensor<128x64x30x30x6xf32> -> !torch.vtensor<[128,64,30,30,6],f32>
// CHECK:       return %[[R]] : !torch.vtensor<[128,64,30,30,6],f32>
func.func @deactivate(%arg0: !torch.vtensor<[128,64,30,30,6],f32,#sparse>)
                          -> !torch.vtensor<[128,64,30,30,6],f32> {
  %none_0 = torch.constant.none
  %none_1 = torch.constant.none
  %none_2 = torch.constant.none
  %result = torch.operator "torch.aten._to_dense"(%arg0, %none_0, %none_1, %none_2)
    : (!torch.vtensor<[128,64,30,30,6],f32,#sparse>, !torch.none, !torch.none, !torch.none)
    -> !torch.vtensor<[128,64,30,30,6],f32>
  return %result : !torch.vtensor<[128,64,30,30,6],f32>
}
