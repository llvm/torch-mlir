// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #map
// CHECK-LABEL: func @grid_sampler
// CHECK-DAG: %[[TC0:.*]] = torch_c.to_builtin_tensor %[[ARG0:.*]] : !torch.vtensor<[4,10,10,4],f32> -> tensor<4x10x10x4xf32>
// CHECK-DAG: %[[TC1:.*]] = torch_c.to_builtin_tensor %[[ARG1:.*]] : !torch.vtensor<[4,6,8,2],f32> -> tensor<4x6x8x2xf32>
// CHECK-DAG: %[[FALSE:.*]] = torch.constant.bool false
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[CST2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[C2_3:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[DIM:.*]] = tensor.dim %[[TC0]], %[[C2_3]] : tensor<4x10x10x4xf32>
// CHECK-DAG: %[[X73:.*]] = arith.cmpi eq, %[[X3:.*]], %[[C27:.*]] : i64
// CHECK-DAG: %[[X74:.*]] = arith.select %[[X73:.*]], %[[X70:.*]], %[[X72:.*]] : f32
// CHECK-DAG: %[[X75:.*]] = arith.subf %[[Xcst_1:.*]], %[[X57:.*]] : f32
// CHECK-DAG: %[[X76:.*]] = arith.mulf %[[X66:.*]], %[[X75:.*]] : f32
// CHECK-DAG: %[[X77:.*]] = arith.mulf %[[X74:.*]], %[[X57:.*]] : f32
// CHECK-DAG: %[[X78:.*]] = arith.addf %[[X76:.*]], %[[X77:.*]] : f32
// CHECK-DAG: %[[C28:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[X79:.*]] = arith.cmpf olt, %[[X57:.*]], %[[X28:.*]] : f32
// CHECK-DAG: %[[X80:.*]] = arith.select %[[X79:.*]], %[[X66:.*]], %[[X74:.*]] : f32
// CHECK-DAG: %[[C29:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[X81:.*]] = arith.cmpi eq, %[[X3:.*]], %[[C29:.*]] : i64
// CHECK-DAG: %[[X82:.*]] = arith.select %[[X81:.*]], %[[X78:.*]], %[[X80:.*]] : f32
// CHECK-DAG: linalg.yield %[[X82:.*]] : f32
// CHECK-DAG: %[[X14:.*]] = torch_c.from_builtin_tensor %[[X13:.*]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>

func.func @grid_sampler(%arg0: !torch.vtensor<[4,10,10,4],f32>, %arg1: !torch.vtensor<[4,6,8,2],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %true = torch.constant.bool 0
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 0
  %4 = torch.aten.grid_sampler %arg0, %arg1, %int0, %int1, %true : !torch.vtensor<[4,10,10,4],f32>, !torch.vtensor<[4,6,8,2],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: func @grid_sampler2
// CHECK: #map
// CHECK-DAG: %[[X70:.*]] = arith.addf %[[X68:.*]], %[[X69:.*]] : f32
// CHECK-DAG: %[[X29:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[X71:.*]] = arith.cmpf olt, %[[X58:.*]], %[[X29:.*]] : f32
// CHECK-DAG: %[[X72:.*]] = arith.select %[[X71:.*]], %[[X52:.*]], %[[X54:.*]] : f32
// CHECK-DAG: %[[X30:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[X73:.*]] = arith.cmpi eq, %[[X3:.*]], %[[X30:.*]] : i64
// CHECK-DAG: %[[X74:.*]] = arith.select %[[X73:.*]], %[[X70:.*]], %[[X72:.*]] : f32
// CHECK-DAG: %[[X75:.*]] = arith.subf %[[X1:.*]], %[[X57:.*]] : f32
// CHECK-DAG: %[[X76:.*]] = arith.mulf %[[X66:.*]], %[[X75:.*]] : f32
// CHECK-DAG: %[[X77:.*]] = arith.mulf %[[X74:.*]], %[[X57:.*]] : f32
// CHECK-DAG: %[[X78:.*]] = arith.addf %[[X76:.*]], %[[X77:.*]] : f32
// CHECK-DAG: %[[X31:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[X79:.*]] = arith.cmpf olt, %[[X57:.*]], %[[X31:.*]] : f32
// CHECK-DAG: %[[X80:.*]] = arith.select %[[X79:.*]], %[[X66:.*]], %[[X74:.*]] : f32
// CHECK-DAG: %[[X32:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[X81:.*]] = arith.cmpi eq, %[[X3:.*]], %[[X32:.*]] : i64
// CHECK-DAG: %[[X82:.*]] = arith.select %[[X81:.*]], %[[X78:.*]], %[[X80:.*]] : f32
// CHECK-DAG: linalg.yield %[[X50:.*]] : f32
// CHECK: return %[[X12:.*]] : !torch.vtensor<[?,?,?,?],f32>
func.func @grid_sampler2(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %true = torch.constant.bool 0
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 0
  %4 = torch.aten.grid_sampler %arg0, %arg1, %int0, %int1, %true : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: func @grid_sampler3
// CHECK: #map
// CHECK-DAG:  %[[X15:.*]] = arith.mulf %[[X13:.*]], %[[X8:.*]] : f32
// CHECK-DAG:      %[[Y60:.*]] = arith.mulf %[[X48:.*]], %[[X59:.*]] : f32
// CHECK-DAG:      %[[Y61:.*]] = arith.mulf %[[X50:.*]], %[[X58:.*]] : f32
// CHECK-DAG:      %[[Y62:.*]] = arith.addf %[[X60:.*]], %[[X61:.*]] : f32
// CHECK-DAG:      %[[Y28:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:      %[[Y64:.*]] = arith.select %[[X63:.*]], %[[X48:.*]], %[[X50:.*]] : f32
// CHECK-DAG:      %[[Y29:.*]] = arith.constant 0 : i6
// CHECK-DAG:      %[[Y65:.*]] = arith.cmpi eq, %[[X3:.*]], %[[X28:.*]] : i64
// CHECK-DAG:      %[[Y66:.*]] = arith.select %[[X65:.*]], %[[X62:.*]], %[[X64:.*]] : f32
// CHECK-DAG:      %[[Y67:.*]] = arith.subf %[[X1:.*]], %[[X58:.*]] : f32
// CHECK-DAG:      %[[Y68:.*]] = arith.mulf %[[X52:.*]], %[[X67:.*]] : f32
// CHECK-DAG:      %[[Y69:.*]] = arith.mulf %[[X54:.*]], %[[X58:.*]] : f32
// CHECK-DAG:      %[[Y70:.*]] = arith.addf %[[X68:.*]], %[[X69:.*]] : f32
// CHECK-DAG:      %[[Y30:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:      %[[Y31:.*]] = arith.constant 0 : i64
// CHECK: return %[[X12:.*]] : !torch.vtensor<[?,?,?,?],f32>
func.func @grid_sampler3(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %false = torch.constant.bool 1
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 0
  %4 = torch.aten.grid_sampler %arg0, %arg1, %int0, %int1, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: func @grid_sampler4
func.func @grid_sampler4(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %false = torch.constant.bool 1
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %4 = torch.aten.grid_sampler %arg0, %arg1, %int0, %int1, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}
