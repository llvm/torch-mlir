// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// A finite, nonzero ord is lowered by this pass with the generic
// (sum |x|^ord)^(1/ord) formula: a reduction that accumulates |x|^ord, followed
// by raising the sum to the 1/ord power.
// CHECK-LABEL:   func.func @torch.aten.linalg_vector_norm$finite(
// CHECK:           %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "reduction"]
// CHECK:             %[[ABS:.*]] = math.absf
// CHECK:             %[[POW:.*]] = math.powf %[[ABS]],
// CHECK:             %[[ACC:.*]] = arith.addf %[[POW]],
// CHECK:             linalg.yield %[[ACC]]
// CHECK:           %[[ROOT:.*]] = linalg.generic
// CHECK-SAME:        iterator_types = ["parallel"]
// CHECK:             math.powf
// CHECK:             linalg.yield
func.func @torch.aten.linalg_vector_norm$finite(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3],f32> {
  %ord = torch.constant.float 2.000000e+00
  %dim = torch.constant.int 1
  %dimlist = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
  %keepdim = torch.constant.bool false
  %dtype = torch.constant.none
  %0 = torch.aten.linalg_vector_norm %arg0, %ord, %dimlist, %keepdim, %dtype : !torch.vtensor<[3,4],f32>, !torch.float, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// Reducing over multiple dims with keepdim = true: the reduction op has one
// reduction iterator per reduced dim, and the reduced dims are kept as size-1
// (the output indexing map pins them to 0), so the result rank matches the
// input rank.
// CHECK-LABEL:   func.func @torch.aten.linalg_vector_norm$keepdim_multidim(
// CHECK:           %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "reduction", "reduction"]
// CHECK:             %[[ABS:.*]] = math.absf
// CHECK:             %[[POW:.*]] = math.powf %[[ABS]],
// CHECK:             %[[ACC:.*]] = arith.addf %[[POW]],
// CHECK:             linalg.yield %[[ACC]]
// CHECK:           %[[ROOT:.*]] = linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]
// CHECK:             math.powf
// CHECK:             linalg.yield
func.func @torch.aten.linalg_vector_norm$keepdim_multidim(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,1,1],f32> {
  %ord = torch.constant.float 2.000000e+00
  %d0 = torch.constant.int 1
  %d1 = torch.constant.int 2
  %dimlist = torch.prim.ListConstruct %d0, %d1 : (!torch.int, !torch.int) -> !torch.list<int>
  %keepdim = torch.constant.bool true
  %dtype = torch.constant.none
  %0 = torch.aten.linalg_vector_norm %arg0, %ord, %dimlist, %keepdim, %dtype : !torch.vtensor<[2,3,4],f32>, !torch.float, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,1,1],f32>
  return %0 : !torch.vtensor<[2,1,1],f32>
}

// -----

// The ord = 0 / +-inf vector norms are handled by
// DecomposeAtenLinalgVectorNormOp before this conversion runs; the generic
// (sum |x|^ord)^(1/ord) lowering here is undefined for them. If the op reaches
// this pass undecomposed (e.g. decomposition disabled), the conversion declines
// rather than miscompiling, so the op fails to legalize.

func.func @torch.aten.linalg_vector_norm$pos_inf(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> {
  %ord = torch.constant.float 0x7FF0000000000000
  %dim = torch.constant.none
  %keepdim = torch.constant.bool false
  %dtype = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.aten.linalg_vector_norm'}}
  %0 = torch.aten.linalg_vector_norm %arg0, %ord, %dim, %keepdim, %dtype : !torch.vtensor<[5],f32>, !torch.float, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

func.func @torch.aten.linalg_vector_norm$neg_inf(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> {
  %ord = torch.constant.float 0xFFF0000000000000
  %dim = torch.constant.none
  %keepdim = torch.constant.bool false
  %dtype = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.aten.linalg_vector_norm'}}
  %0 = torch.aten.linalg_vector_norm %arg0, %ord, %dim, %keepdim, %dtype : !torch.vtensor<[5],f32>, !torch.float, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// ord is a Scalar, so an integer ord = 0 imports as a torch.constant.int.
func.func @torch.aten.linalg_vector_norm$zero_int(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> {
  %ord = torch.constant.int 0
  %dim = torch.constant.none
  %keepdim = torch.constant.bool false
  %dtype = torch.constant.none
  // expected-error @+1 {{failed to legalize operation 'torch.aten.linalg_vector_norm'}}
  %0 = torch.aten.linalg_vector_norm %arg0, %ord, %dim, %keepdim, %dtype : !torch.vtensor<[5],f32>, !torch.int, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
