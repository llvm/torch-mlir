// RUN: torch-mlir-opt %s -convert-torch-to-linalg -split-input-file -verify-diagnostics -mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: func.func @fma_broadcast
// CHECK: linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[FMA:.*]] = math.fma %[[A]], %[[B]], %[[C]] : f32
// CHECK-NEXT: linalg.yield %[[FMA]] : f32
func.func @fma_broadcast(%a: !torch.vtensor<[2,3],f32>, %b: !torch.vtensor<[],f32>, %c: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.prims.fma %a, %b, %c : !torch.vtensor<[2,3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL: func.func @fma_dynamic
// CHECK: linalg.generic
// CHECK: math.fma {{.*}} : f64
func.func @fma_dynamic(%a: !torch.vtensor<[?,?],f64>, %b: !torch.vtensor<[1,?],f64>, %c: !torch.vtensor<[],f64>) -> !torch.vtensor<[?,?],f64> {
  %0 = torch.prims.fma %a, %b, %c : !torch.vtensor<[?,?],f64>, !torch.vtensor<[1,?],f64>, !torch.vtensor<[],f64> -> !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// -----

// CHECK-LABEL: func.func @fma_f16
// CHECK: math.fma {{.*}} : f16
func.func @fma_f16(%a: !torch.vtensor<[3],f16>, %b: !torch.vtensor<[3],f16>, %c: !torch.vtensor<[3],f16>) -> !torch.vtensor<[3],f16> {
  %0 = torch.prims.fma %a, %b, %c : !torch.vtensor<[3],f16>, !torch.vtensor<[3],f16>, !torch.vtensor<[3],f16> -> !torch.vtensor<[3],f16>
  return %0 : !torch.vtensor<[3],f16>
}

// -----

// CHECK-LABEL: func.func @fma_bf16
// CHECK: math.fma {{.*}} : bf16
func.func @fma_bf16(%a: !torch.vtensor<[3],bf16>, %b: !torch.vtensor<[3],bf16>, %c: !torch.vtensor<[3],bf16>) -> !torch.vtensor<[3],bf16> {
  %0 = torch.prims.fma %a, %b, %c : !torch.vtensor<[3],bf16>, !torch.vtensor<[3],bf16>, !torch.vtensor<[3],bf16> -> !torch.vtensor<[3],bf16>
  return %0 : !torch.vtensor<[3],bf16>
}

// -----

// CHECK-LABEL: func.func @fma_promote
// CHECK: linalg.generic
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f64, %[[C:.*]]: f32, %{{.*}}: f64):
// CHECK-NEXT: %[[CAST_A:.*]] = arith.extf %[[A]] : f32 to f64
// CHECK-NEXT: %[[CAST_C:.*]] = arith.extf %[[C]] : f32 to f64
// CHECK-NEXT: %[[FMA:.*]] = math.fma %[[CAST_A]], %[[B]], %[[CAST_C]] : f64
// CHECK-NEXT: linalg.yield %[[FMA]] : f64
func.func @fma_promote(%a: !torch.vtensor<[3],f32>, %b: !torch.vtensor<[3],f64>, %c: !torch.vtensor<[],f32>) -> !torch.vtensor<[3],f64> {
  %0 = torch.prims.fma %a, %b, %c : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f64>, !torch.vtensor<[],f32> -> !torch.vtensor<[3],f64>
  return %0 : !torch.vtensor<[3],f64>
}

// -----

func.func @fma_integer(%a: !torch.vtensor<[3],si32>, %b: !torch.vtensor<[3],si32>, %c: !torch.vtensor<[3],si32>) -> !torch.vtensor<[3],si32> {
  // expected-error @+2 {{unimplemented: non-floating point dtype}}
  // expected-error @+1 {{failed to legalize operation 'torch.prims.fma'}}
  %0 = torch.prims.fma %a, %b, %c : !torch.vtensor<[3],si32>, !torch.vtensor<[3],si32>, !torch.vtensor<[3],si32> -> !torch.vtensor<[3],si32>
  return %0 : !torch.vtensor<[3],si32>
}
