// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -verify-diagnostics | FileCheck %s --check-prefixes=COORD,LARGE

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

// -----

// Round before testing bounds, and guard both conversions and loads.
// COORD-LABEL: func.func @nearest_zeros(
// COORD-DAG: %[[SIZE:.*]] = arith.constant 4.000000e+00 : f32
// COORD-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// COORD-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// COORD: linalg.generic
// COORD: %[[PLUS_ONE:.*]] = arith.addf {{.*}}, %[[ONE]] : f32
// COORD-NEXT: %[[SCALED:.*]] = arith.mulf %[[PLUS_ONE]], %[[SIZE]] : f32
// COORD-NEXT: %[[SHIFTED:.*]] = arith.subf %[[SCALED]], %[[ONE]] : f32
// COORD-NEXT: %[[COORD:.*]] = arith.divf %[[SHIFTED]], %[[TWO]] : f32
// COORD-NEXT: math.roundeven %[[COORD]] : f32
// CHECK-LABEL: func.func @nearest_zeros
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.generic
// CHECK: %[[ROW:.*]] = math.roundeven {{.*}} : f32
// CHECK: %[[COL:.*]] = math.roundeven {{.*}} : f32
// CHECK: %[[LIMIT:.*]] = arith.constant 9.22337203E+18 : f32
// CHECK: %[[RL:.*]] = arith.cmpf oge, %[[ROW]], %[[ZERO]] : f32
// CHECK: %[[RU:.*]] = arith.cmpf olt, %[[ROW]], %[[LIMIT]] : f32
// CHECK: %[[CL:.*]] = arith.cmpf oge, %[[COL]], %[[ZERO]] : f32
// CHECK: %[[CU:.*]] = arith.cmpf olt, %[[COL]], %[[LIMIT]] : f32
// CHECK: %[[RV:.*]] = arith.andi %[[RL]], %[[RU]] : i1
// CHECK: %[[CV:.*]] = arith.andi %[[CL]], %[[CU]] : i1
// CHECK: %[[CONVERTIBLE:.*]] = arith.andi %[[RV]], %[[CV]] : i1
// CHECK: %[[N:.*]] = linalg.index 0 : index
// CHECK: %[[C:.*]] = linalg.index 1 : index
// CHECK: %[[SAMPLED:.*]] = scf.if %[[CONVERTIBLE]] -> (f32) {
// CHECK-NEXT: %[[RI:.*]] = arith.fptosi %[[ROW]] : f32 to i64
// CHECK-NEXT: %[[CI:.*]] = arith.fptosi %[[COL]] : f32 to i64
// CHECK-NEXT: %[[RB:.*]] = arith.cmpi sle, %[[RI]], {{.*}} : i64
// CHECK-NEXT: %[[CB:.*]] = arith.cmpi sle, %[[CI]], {{.*}} : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = arith.andi %[[RB]], %[[CB]] : i1
// CHECK-NEXT: %[[PIXEL:.*]] = scf.if %[[BOUNDS]] -> (f32) {
// CHECK-NEXT: %[[R:.*]] = arith.index_cast %[[RI]] : i64 to index
// CHECK-NEXT: %[[K:.*]] = arith.index_cast %[[CI]] : i64 to index
// CHECK-NEXT: %[[VALUE:.*]] = tensor.extract {{.*}}[%[[N]], %[[C]], %[[R]], %[[K]]] : tensor<2x3x4x4xf32>
// CHECK-NEXT: scf.yield %[[VALUE]] : f32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[ZERO]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[PIXEL]] : f32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[ZERO]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: linalg.yield %[[SAMPLED]] : f32
func.func @nearest_zeros(%input: !torch.vtensor<[2,3,4,4],f32>, %grid: !torch.vtensor<[2,1,13,2],f32>) -> !torch.vtensor<[2,3,1,13],f32> {
  %nearest = torch.constant.int 1
  %zeros = torch.constant.int 0
  %align = torch.constant.bool false
  %result = torch.aten.grid_sampler %input, %grid, %nearest, %zeros, %align : !torch.vtensor<[2,3,4,4],f32>, !torch.vtensor<[2,1,13,2],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[2,3,1,13],f32>
  return %result : !torch.vtensor<[2,3,1,13],f32>
}

// -----

// CHECK-LABEL: func.func @nearest_zeros_dynamic
// CHECK: %[[INPUT:.*]] = torch_c.to_builtin_tensor {{.*}} : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[HEIGHT_AXIS:.*]] = arith.constant 2 : index
// CHECK: %[[HEIGHT:.*]] = tensor.dim %[[INPUT]], %[[HEIGHT_AXIS]] : tensor<?x?x?x?xf32>
// CHECK: %[[WIDTH_AXIS:.*]] = arith.constant 3 : index
// CHECK: %[[WIDTH:.*]] = tensor.dim %[[INPUT]], %[[WIDTH_AXIS]] : tensor<?x?x?x?xf32>
// CHECK: %[[HEIGHT_LAST:.*]] = arith.subi %[[HEIGHT]], %[[ONE]] : index
// CHECK: %[[WIDTH_LAST:.*]] = arith.subi %[[WIDTH]], %[[ONE]] : index
// CHECK: %[[ROW_MAX:.*]] = arith.index_cast %[[HEIGHT_LAST]] : index to i64
// CHECK: %[[COL_MAX:.*]] = arith.index_cast %[[WIDTH_LAST]] : index to i64
// CHECK: tensor.empty({{.*}}) : tensor<?x?x?x?xf32>
// CHECK: linalg.generic
// CHECK: %[[ROW:.*]] = math.roundeven {{.*}} : f32
// CHECK: %[[COL:.*]] = math.roundeven {{.*}} : f32
// CHECK: %[[SAMPLED:.*]] = scf.if {{.*}} -> (f32) {
// CHECK: %[[RI:.*]] = arith.fptosi %[[ROW]] : f32 to i64
// CHECK: %[[CI:.*]] = arith.fptosi %[[COL]] : f32 to i64
// CHECK: %[[RB:.*]] = arith.cmpi sle, %[[RI]], %[[ROW_MAX]] : i64
// CHECK: %[[CB:.*]] = arith.cmpi sle, %[[CI]], %[[COL_MAX]] : i64
// CHECK: %[[BOUNDS:.*]] = arith.andi %[[RB]], %[[CB]] : i1
// CHECK: %[[PIXEL:.*]] = scf.if %[[BOUNDS]] -> (f32) {
// CHECK: %[[VALUE:.*]] = tensor.extract %[[INPUT]][{{.*}}] : tensor<?x?x?x?xf32>
// CHECK-NEXT: scf.yield %[[VALUE]] : f32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[ZERO]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[PIXEL]] : f32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[ZERO]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: linalg.yield %[[SAMPLED]] : f32
func.func @nearest_zeros_dynamic(%input: !torch.vtensor<[?,?,?,?],f32>, %grid: !torch.vtensor<[?,?,?,2],f32>, %align: !torch.bool) -> !torch.vtensor<[?,?,?,?],f32> {
  %nearest = torch.constant.int 1
  %zeros = torch.constant.int 0
  %result = torch.aten.grid_sampler %input, %grid, %nearest, %zeros, %align : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,2],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %result : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// 16777219 rounds up to 16777220 in f32. Check the exact integer bound.
// LARGE-LABEL: func.func @nearest_zeros_large_width
// LARGE-DAG: %[[LAST:.*]] = arith.constant 16777219 : i64
// LARGE: scf.if
// LARGE: arith.fptosi
// LARGE: %[[COL:.*]] = arith.fptosi {{.*}} : f32 to i64
// LARGE: %[[COL_VALID:.*]] = arith.cmpi sle, %[[COL]], %[[LAST]] : i64
// LARGE: %[[VALID:.*]] = arith.andi {{.*}}, %[[COL_VALID]] : i1
// LARGE: scf.if %[[VALID]] -> (f32) {
// LARGE: tensor.extract {{.*}} : tensor<1x1x1x16777220xf32>
func.func @nearest_zeros_large_width(%input: !torch.vtensor<[1,1,1,16777220],f32>, %grid: !torch.vtensor<[1,1,1,2],f32>) -> !torch.vtensor<[1,1,1,1],f32> {
  %nearest = torch.constant.int 1
  %zeros = torch.constant.int 0
  %align = torch.constant.bool true
  %result = torch.aten.grid_sampler %input, %grid, %nearest, %zeros, %align : !torch.vtensor<[1,1,1,16777220],f32>, !torch.vtensor<[1,1,1,2],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[1,1,1,1],f32>
  return %result : !torch.vtensor<[1,1,1,1],f32>
}
