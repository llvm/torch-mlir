// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_resize_sizes_linear
func.func @test_resize_sizes_linear(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4]
,si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[x0:.*]] = torch_c.to_builtin_tensor %arg0
    // CHECK: %[[generic:.*]] = linalg.generic
    // CHECK-DAG: %[[cst:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK-DAG: %[[cst_4:.*]] = arith.constant 5.000000e-01 : f32
    // CHECK-DAG: %[[cst_5:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-DAG: %[[x15:.*]] = linalg.index 0 : index
    // CHECK-DAG: %[[x16:.*]] = linalg.index 1 : index
    // CHECK-DAG: %[[x17:.*]] = linalg.index 2 : index
    // CHECK-DAG: %[[x18:.*]] = linalg.index 3 : index
    // CHECK: %[[x19:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
    // CHECK: %[[x20:.*]] = arith.sitofp %[[x8:.*]] : i64 to f32
    // CHECK-DAG: %[[x21:.*]] = arith.divf %[[x20]], %[[x19]] : f32
    // CHECK-DAG: %[[x22:.*]] = arith.index_cast %[[x17]] : index to i64
    // CHECK-DAG: %[[x23:.*]] = arith.sitofp %[[x22]] : i64 to f32
    // CHECK-DAG: %[[x24:.*]] = arith.addf %[[x23]], %[[cst_4]] : f32
    // CHECK-DAG: %[[x25:.*]] = arith.divf %[[x24]], %[[x21]] : f32
    // CHECK-DAG: %[[x26:.*]] = arith.subf %[[x25]], %[[cst_4]] : f32
    // CHECK-DAG: %[[x27:.*]] = arith.maximumf %[[x26]], %[[cst_5]] : f32
    // CHECK-DAG: %[[x28:.*]] = arith.subf %[[x19]], %cst_4 : f32
    // CHECK-DAG: %[[x29:.*]] = arith.minimumf %[[x27]], %[[x28]] : f32
    // CHECK-DAG: %[[x30:.*]] = math.floor %[[x29]] : f32
    // CHECK-DAG: %[[x31:.*]] = arith.addf %[[cst]], %[[x29]] : f32
    // CHECK-DAG: %[[x32:.*]] = math.floor %[[x31]] : f32
    // CHECK-DAG: %[[x33:.*]] = arith.fptosi %[[x30]] : f32 to i64
    // CHECK-DAG: %[[x34:.*]] = arith.index_cast %[[x33]] : i64 to index
    // CHECK-DAG: %[[x35:.*]] = arith.minimumf %44, %42 : f32
    // CHECK-DAG: %[[x36:.*]] = arith.fptosi %[[x35]] : f32 to i64
    // CHECK-DAG: %[[x37:.*]] = arith.index_cast %[[x36]] : i64 to index
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0]][%[[x15]], %[[x16]], %[[x34]], %[[low:.*]]] : tensor<1x1x2x4xf32>
    // CHECK: %[[extracted_7:.*]] = tensor.extract %[[x0]][%[[x15]], %[[x16]], %[[x34]], %[[high:.*]]] : tensor<1x1x2x4xf32>
    // CHECK: %[[extracted_8:.*]] = tensor.extract %[[x0]][%[[x15]], %[[x16]], %[[x37]], %[[low]]] : tensor<1x1x2x4xf32>
    // CHECK: %[[extracted_9:.*]] = tensor.extract %[[x0]][%[[x15]], %[[x16]], %[[x37]], %[[high]]] : tensor<1x1x2x4xf32>
    // CHECK: %[[dx0p00:.*]] = arith.mulf %[[dx0:.*]], %[[extracted]]
    // CHECK: %[[dx1p01:.*]] = arith.mulf %[[dx1:.*]], %[[extracted_7]]
    // CHECK: %[[sum:.*]] = arith.addf %[[dx0p00]], %[[dx1p01]]
    // CHECK: %[[left:.*]] = arith.mulf %[[dy0:.*]], %[[sum]]
    // CHECK: %[[dx0p10:.*]] = arith.mulf %[[dx0]], %[[extracted_8]]
    // CHECK: %[[dx1p11:.*]] = arith.mulf %[[dx1]], %[[extracted_9]]
    // CHECK: %[[sum2:.*]] = arith.addf %[[dx0p10]], %[[dx1p11]]
    // CHECK: %[[right:.*]] = arith.mulf %[[dy1:.*]], %[[sum2]]
    // CHECK: %[[retval:.*]] = arith.addf %[[left]], %[[right]]
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "bilinear"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %int3 = torch.constant.int 3
    %2 = torch.aten.select.int %arg1, %int0, %int3 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %3 = torch.aten.item %2 : !torch.vtensor<[1],si64> -> !torch.int
    %4 = torch.prim.ListConstruct %1, %3 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[1,1,2,4],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?,?],f32>
  }

// -----

// CHECK-LABEL: func.func @test_resize_sizes_nearest
func.func @test_resize_sizes_nearest(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[GENERIC:.*]] = linalg.generic
    // CHECK: %[[x11:.*]] = linalg.index 0 : index
    // CHECK: %[[x12:.*]] = linalg.index 1 : index
    // CHECK: %[[x13:.*]] = linalg.index 2 : index
    // CHECK: %[[x14:.*]] = linalg.index 3 : index
    // CHECK: %[[x15:.*]] = arith.sitofp %[[c2_i64:.*]] : i64 to f32
    // CHECK: %[[x19:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
    // CHECK: %[[x21:.*]] = arith.divf %[[x19]], %[[x15]] : f32
    // CHECK: %[[x23:.*]] = arith.index_cast %[[x13]] : index to i64
    // CHECK: %[[x24:.*]] = arith.sitofp %[[x23]] : i64 to f32
    // CHECK: %[[x25:.*]] = arith.divf %[[x24]], %[[x21]] : f32
    // CHECK: %[[floorH:.*]] = math.floor %[[x25]] : f32
    // CHECK: %[[c1H:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0H:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1H:.*]] = arith.subf %[[x15]], %[[c1H]] : f32
    // CHECK: %[[clH1:.*]] = arith.maximumf %[[floorH]], %[[c0H]] : f32
    // CHECK: %[[clH2:.*]] = arith.minimumf %[[clH1]], %[[inLm1H]] : f32
    // CHECK: %[[x31:.*]] = arith.fptosi %[[clH2]] : f32 to i64
    // CHECK: %[[x32:.*]] = arith.index_cast %[[x31]] : i64 to index
    // CHECK: %[[x16:.*]] = arith.sitofp %[[c4_i64:.*]] : i64 to f32
    // CHECK: %[[x20:.*]] = arith.sitofp %[[x7:.*]] : i64 to f32
    // CHECK: %[[x22:.*]] = arith.divf %[[x20]], %[[x16]] : f32
    // CHECK: %[[x26:.*]] = arith.index_cast %[[x14]] : index to i64
    // CHECK: %[[x27:.*]] = arith.sitofp %[[x26]] : i64 to f32
    // CHECK: %[[x28:.*]] = arith.divf %[[x27]], %[[x22]] : f32
    // CHECK: %[[floorW:.*]] = math.floor %[[x28]] : f32
    // CHECK: %[[c1W:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0W:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1W:.*]] = arith.subf %[[x16]], %[[c1W]] : f32
    // CHECK: %[[clW1:.*]] = arith.maximumf %[[floorW]], %[[c0W]] : f32
    // CHECK: %[[clW2:.*]] = arith.minimumf %[[clW1]], %[[inLm1W]] : f32
    // CHECK: %[[x33:.*]] = arith.fptosi %[[clW2]] : f32 to i64
    // CHECK: %[[x34:.*]] = arith.index_cast %[[x33]] : i64 to index
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x11]], %[[x12]], %[[x32]], %[[x34]]] : tensor<1x1x2x4xf32>
    // CHECK: linalg.yield %[[extracted]] : f32
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "nearest"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %int3 = torch.constant.int 3
    %2 = torch.aten.select.int %arg1, %int0, %int3 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %3 = torch.aten.item %2 : !torch.vtensor<[1],si64> -> !torch.int
    %4 = torch.prim.ListConstruct %1, %3 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[1,1,2,4],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?,?],f32>
  }

// -----

// CHECK-LABEL: func.func @test_resize_nearest_1d
func.func @test_resize_nearest_1d(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[?,?,?],f32> {
    // CHECK: %[[GENERIC:.*]] = linalg.generic
    // CHECK: %[[x11:.*]] = linalg.index 0 : index
    // CHECK: %[[x12:.*]] = linalg.index 1 : index
    // CHECK: %[[x13:.*]] = linalg.index 2 : index
    // CHECK: %[[x15:.*]] = arith.sitofp %[[c2_i64:.*]] : i64 to f32
    // CHECK: %[[x19:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
    // CHECK: %[[x21:.*]] = arith.divf %[[x19]], %[[x15]] : f32
    // CHECK: %[[x23:.*]] = arith.index_cast %[[x13]] : index to i64
    // CHECK: %[[x24:.*]] = arith.sitofp %[[x23]] : i64 to f32
    // CHECK: %[[x25:.*]] = arith.divf %[[x24]], %[[x21]] : f32
    // CHECK: %[[flo1d:.*]] = math.floor %[[x25]] : f32
    // CHECK: %[[c1_1d:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0_1d:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1_1d:.*]] = arith.subf %[[x15]], %[[c1_1d]] : f32
    // CHECK: %[[cl1d1:.*]] = arith.maximumf %[[flo1d]], %[[c0_1d]] : f32
    // CHECK: %[[cl1d2:.*]] = arith.minimumf %[[cl1d1]], %[[inLm1_1d]] : f32
    // CHECK: %[[x31:.*]] = arith.fptosi %[[cl1d2]] : f32 to i64
    // CHECK: %[[x32:.*]] = arith.index_cast %[[x31]] : i64 to index
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x11]], %[[x12]], %[[x32]]] : tensor<?x?x?xf32>
    // CHECK: linalg.yield %[[extracted]] : f32
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "nearest,floor"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %4 = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
    %5 = torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL: func.func @test_resize_nearest_3d
func.func @test_resize_nearest_3d(%arg0: !torch.vtensor<[?,?,?,?,?],f32>, %arg1: !torch.vtensor<[5],si64>) -> !torch.vtensor<[?,?,?,?,?],f32> {
    // CHECK: %[[GENERIC:.*]] = linalg.generic
    // CHECK: %[[x11:.*]] = linalg.index 0 : index
    // CHECK: %[[x12:.*]] = linalg.index 1 : index
    // CHECK: %[[x13:.*]] = linalg.index 2 : index
    // CHECK: %[[x14:.*]] = linalg.index 3 : index
    // CHECK: %[[index4:.*]] = linalg.index 4 : index
    // CHECK: %[[x15:.*]] = arith.sitofp %[[c2_i64:.*]] : i64 to f32
    // CHECK: %[[x19:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
    // CHECK: %[[x21:.*]] = arith.divf %[[x19]], %[[x15]] : f32
    // CHECK: %[[x23:.*]] = arith.index_cast %[[x13]] : index to i64
    // CHECK: %[[x24:.*]] = arith.sitofp %[[x23]] : i64 to f32
    // CHECK: %[[x25:.*]] = arith.divf %[[x24]], %[[x21]] : f32
    // CHECK: %[[floor:.*]] = math.floor %[[x25]] : f32
    // CHECK: %[[c1_3da:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0_3da:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1_3da:.*]] = arith.subf %[[x15]], %[[c1_3da]] : f32
    // CHECK: %[[cl3da1:.*]] = arith.maximumf %[[floor]], %[[c0_3da]] : f32
    // CHECK: %[[cl3da2:.*]] = arith.minimumf %[[cl3da1]], %[[inLm1_3da]] : f32
    // CHECK: %[[x31:.*]] = arith.fptosi %[[cl3da2]] : f32 to i64
    // CHECK: %[[x32:.*]] = arith.index_cast %[[x31]] : i64 to index
    // CHECK: %[[x16w:.*]] = arith.sitofp %[[c3_i64:.*]] : i64 to f32
    // CHECK: %[[x20w:.*]] = arith.sitofp %[[x7:.*]] : i64 to f32
    // CHECK: %[[x22w:.*]] = arith.divf %[[x20w]], %[[x16w]] : f32
    // CHECK: %[[x26w:.*]] = arith.index_cast %[[x14]] : index to i64
    // CHECK: %[[x27w:.*]] = arith.sitofp %[[x26w]] : i64 to f32
    // CHECK: %[[x28w:.*]] = arith.divf %[[x27w]], %[[x22w]] : f32
    // CHECK: %[[floorW2:.*]] = math.floor %[[x28w]] : f32
    // CHECK: %[[c1W2:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0W2:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1W2:.*]] = arith.subf %[[x16w]], %[[c1W2]] : f32
    // CHECK: %[[clW2a:.*]] = arith.maximumf %[[floorW2]], %[[c0W2]] : f32
    // CHECK: %[[clW2b:.*]] = arith.minimumf %[[clW2a]], %[[inLm1W2]] : f32
    // CHECK: %[[Wfptosi:.*]] = arith.fptosi %[[clW2b]] : f32 to i64
    // CHECK: %[[x34:.*]] = arith.index_cast %[[Wfptosi]] : i64 to index
    // CHECK: %[[x16d:.*]] = arith.sitofp %[[c4_i64:.*]] : i64 to f32
    // CHECK: %[[x20d:.*]] = arith.sitofp %[[x8:.*]] : i64 to f32
    // CHECK: %[[x22d:.*]] = arith.divf %[[x20d]], %[[x16d]] : f32
    // CHECK: %[[x26d:.*]] = arith.index_cast %[[index4]] : index to i64
    // CHECK: %[[x27d:.*]] = arith.sitofp %[[x26d]] : i64 to f32
    // CHECK: %[[x28d:.*]] = arith.divf %[[x27d]], %[[x22d]] : f32
    // CHECK: %[[floorD2:.*]] = math.floor %[[x28d]] : f32
    // CHECK: %[[c1D2:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0D2:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1D2:.*]] = arith.subf %[[x16d]], %[[c1D2]] : f32
    // CHECK: %[[clD2a:.*]] = arith.maximumf %[[floorD2]], %[[c0D2]] : f32
    // CHECK: %[[clD2b:.*]] = arith.minimumf %[[clD2a]], %[[inLm1D2]] : f32
    // CHECK: %[[Dfptosi:.*]] = arith.fptosi %[[clD2b]] : f32 to i64
    // CHECK: %[[x35:.*]] = arith.index_cast %[[Dfptosi]] : i64 to index
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x11]], %[[x12]], %[[x32]], %[[x34]], %[[x35]]] : tensor<?x?x?x?x?xf32>
    // CHECK: linalg.yield %[[extracted]] : f32
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "nearest"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[5],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %int3 = torch.constant.int 3
    %2 = torch.aten.select.int %arg1, %int0, %int3 : !torch.vtensor<[5],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %3 = torch.aten.item %2 : !torch.vtensor<[1],si64> -> !torch.int
    %int4 = torch.constant.int 4
    %4 = torch.aten.select.int %arg1, %int0, %int4 : !torch.vtensor<[5],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %5 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %6 = torch.prim.ListConstruct %1, %3, %5: (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.__interpolate.size_list_scale_list %arg0, %6, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?,?],f32>
    return %7 : !torch.vtensor<[?,?,?,?,?],f32>
  }

// -----

// CHECK-LABEL: func.func @test_resize_nearest_ceil
func.func @test_resize_nearest_ceil(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[?,?,?],f32> {
    // CHECK: %[[GENERIC:.*]] = linalg.generic
    // CHECK: %[[x11:.*]] = linalg.index 0 : index
    // CHECK: %[[x12:.*]] = linalg.index 1 : index
    // CHECK: %[[x13:.*]] = linalg.index 2 : index
    // CHECK: %[[x15:.*]] = arith.sitofp %[[c2_i64:.*]] : i64 to f32
    // CHECK: %[[x19:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
    // CHECK: %[[x21:.*]] = arith.divf %[[x19]], %[[x15]] : f32
    // CHECK: %[[x23:.*]] = arith.index_cast %[[x13]] : index to i64
    // CHECK: %[[x24:.*]] = arith.sitofp %[[x23]] : i64 to f32
    // CHECK: %[[cst:.*]] = arith.constant 5.000000e-01 : f32
    // CHECK: %[[add:.*]] = arith.addf %[[x24]], %[[cst]] : f32
    // CHECK: %[[x25:.*]] = arith.divf %[[add]], %[[x21]] : f32
    // CHECK: %[[sub:.*]] = arith.subf %[[x25]], %[[cst]] : f32
    // CHECK: %[[ceil:.*]] = math.ceil %[[sub]] : f32
    // CHECK: %[[c1ceil:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0ceil:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1ceil:.*]] = arith.subf %[[x15]], %[[c1ceil]] : f32
    // CHECK: %[[minCl1:.*]] = arith.maximumf %[[ceil]], %[[c0ceil]] : f32
    // CHECK: %[[minCl2:.*]] = arith.minimumf %[[minCl1]], %[[inLm1ceil]] : f32
    // CHECK: %[[x31:.*]] = arith.fptosi %[[minCl2]] : f32 to i64
    // CHECK: %[[x32:.*]] = arith.index_cast %[[x31]] : i64 to index
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x11]], %[[x12]], %[[x32]]] : tensor<?x?x?xf32>
    // CHECK: linalg.yield %[[extracted]] : f32
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "nearest_half_pixel,ceil"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %4 = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
    %5 = torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL: func.func @test_resize_scales_linear_half_pixel_symmetric
func.func @test_resize_scales_linear_half_pixel_symmetric(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4]
,f64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[generic:.*]] = linalg.generic
    // CHECK: %[[cst7:.*]] = arith.constant 2.0
    // CHECK: %[[halfsize:.*]] = arith.divf %[[sizefp:.*]], %[[cst7]]
    // CHECK: %[[modifier:.*]] = arith.subf %[[cstOne:.*]], %[[adjustment:.*]]
    // CHECK: %[[offset:.*]] = arith.mulf %[[halfsize]], %[[modifier]]
    // CHECK: %[[preClip:.*]] = arith.addf %[[offset]], %[[halfpixelbase:.*]]
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x1:.*]], %[[x2:.*]], %[[x3:.*]], %[[x4:.*]]] : tensor<1x1x2x4xf32>
    // CHECK: %[[extracted_7:.*]] = tensor.extract %[[x0]][%[[x1]], %[[x2]]
    // CHECK: %[[extracted_8:.*]] = tensor.extract %[[x0]][%[[x1]], %[[x2]]
    // CHECK: %[[extracted_9:.*]] = tensor.extract %[[x0]][%[[x1]], %[[x2]]
    // CHECK: %[[dx0p00:.*]] = arith.mulf %[[dx0:.*]], %[[extracted]]
    // CHECK: %[[dx1p01:.*]] = arith.mulf %[[dx1:.*]], %[[extracted_7]]
    // CHECK: %[[sum:.*]] = arith.addf %[[dx0p00]], %[[dx1p01]]
    // CHECK: %[[left:.*]] = arith.mulf %[[dy0:.*]], %[[sum]]
    // CHECK: %[[dx0p10:.*]] = arith.mulf %[[dx0]], %[[extracted_8]]
    // CHECK: %[[dx1p11:.*]] = arith.mulf %[[dx1]], %[[extracted_9]]
    // CHECK: %[[sum2:.*]] = arith.addf %[[dx0p10]], %[[dx1p11]]
    // CHECK: %[[right:.*]] = arith.mulf %[[dy1:.*]], %[[sum2]]
    // CHECK: %[[retval:.*]] = arith.addf %[[left]], %[[right]]
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "bilinear_half_pixel_symmetric"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[4],f64>, !torch.int, !torch.int -> !torch.vtensor<[1],f64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],f64> -> !torch.float
    %int3 = torch.constant.int 3
    %2 = torch.aten.select.int %arg1, %int0, %int3 : !torch.vtensor<[4],f64>, !torch.int, !torch.int -> !torch.vtensor<[1],f64>
    %3 = torch.aten.item %2 : !torch.vtensor<[1],f64> -> !torch.float
    %4 = torch.prim.ListConstruct %1, %3 : (!torch.float, !torch.float) -> !torch.list<float>
    %5 = torch.aten.__interpolate.size_list_scale_list %arg0, %none_0, %4, %str, %false, %none_0, %false : !torch.vtensor<[1,1,2,4],f32>, !torch.none, !torch.list<float>, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?,?],f32>
  }

// -----

// CHECK-LABEL: func.func @test_resize_nearest_half_pixel_round_prefer_floor
func.func @test_resize_nearest_half_pixel_round_prefer_floor(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[?,?,?],f32> {
    // CHECK: %[[GENERIC:.*]] = linalg.generic
    // CHECK: %[[x11:.*]] = linalg.index 0 : index
    // CHECK: %[[x12:.*]] = linalg.index 1 : index
    // CHECK: %[[x13:.*]] = linalg.index 2 : index
    // CHECK: %[[x15:.*]] = arith.sitofp %[[c2_i64:.*]] : i64 to f32
    // CHECK: %[[x19:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
    // CHECK: %[[x21:.*]] = arith.divf %[[x19]], %[[x15]] : f32
    // CHECK: %[[x23:.*]] = arith.index_cast %[[x13]] : index to i64
    // CHECK: %[[x24:.*]] = arith.sitofp %[[x23]] : i64 to f32
    // CHECK: %[[cst:.*]] = arith.constant 5.000000e-01 : f32
    // CHECK: %[[add:.*]] = arith.addf %[[x24]], %[[cst]] : f32
    // CHECK: %[[x25:.*]] = arith.divf %[[add]], %[[x21]] : f32
    // CHECK: %[[sub:.*]] = arith.subf %[[x25]], %[[cst]] : f32
    // CHECK: %[[cst3:.*]] = arith.constant 5.000000e-01 : f32
    // CHECK: %[[floor:.*]] = math.floor %[[sub]] : f32
    // CHECK: %[[ceil:.*]] = math.ceil %[[sub]] : f32
    // CHECK: %[[sub2:.*]] = arith.subf %[[sub]], %[[floor]] : f32
    // CHECK: %[[cmpf:.*]] = arith.cmpf ule, %[[sub2]], %[[cst3]] : f32
    // CHECK: %[[select:.*]] =  arith.select %[[cmpf]], %[[floor]], %[[ceil]] : f32
    // CHECK: %[[c1_clamp:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[c0_clamp:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[inLm1:.*]] = arith.subf %[[x15]], %[[c1_clamp]] : f32
    // CHECK: %[[selMax0:.*]] = arith.maximumf %[[select]], %[[c0_clamp]] : f32
    // CHECK: %[[selClamped:.*]] = arith.minimumf %[[selMax0]], %[[inLm1]] : f32
    // CHECK: %[[x31:.*]] = arith.fptosi %[[selClamped]] : f32 to i64
    // CHECK: %[[x32:.*]] = arith.index_cast %[[x31]] : i64 to index
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x11]], %[[x12]], %[[x32]]] : tensor<?x?x?xf32>
    // CHECK: linalg.yield %[[extracted]] : f32
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "nearest_half_pixel,round_prefer_floor"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %4 = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
    %5 = torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL: func.func @test_resize_sizes_cubic
func.func @test_resize_sizes_cubic(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4]
,si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19
: si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK-DAG: %[[x1:.*]] = math.ceil %36 : f32
    // CHECK-DAG: %[[x_1:.*]] = arith.subf %[[x1]], %cst_5 : f32
    // CHECK-DAG: %[[x_2:.*]] = arith.subf %[[x_1]], %cst_5 : f32
    // CHECK-DAG: %[[x2:.*]] = arith.addf %[[x1]], %cst_5 : f32
    // CHECK-DAG: %[[y1:.*]] = math.ceil %28 : f32
    // CHECK-DAG: %[[y_1:.*]] = arith.subf %[[y1]], %cst_5 : f32
    // CHECK-DAG: %[[y_2:.*]] = arith.subf %[[y_1]], %cst_5 : f32
    // CHECK-DAG: %[[y2:.*]] = arith.addf %[[y1]], %cst_5 : f32
    // CHECK-DAG: %[[y2D:.*]] = arith.subf %28, %[[y2]] : f32
    // CHECK-DAG: %[[y2Dist:.*]] = math.absf %[[y2D]] : f32
    // CHECK-DAG: %[[y1D:.*]] = arith.subf %28, %[[y1]] : f32
    // CHECK-DAG: %[[y1Dist:.*]] = math.absf %[[y1D]] : f32
    // CHECK-DAG: %[[y_1D:.*]] = arith.subf %28, %[[y_1]] : f32
    // CHECK-DAG: %[[y_1Dist:.*]] = math.absf %[[y_1D]] : f32
    // CHECK-DAG: %[[y_2D:.*]] = arith.subf %28, %[[y_2]] : f32
    // CHECK-DAG: %[[y_2Dist:.*]] = math.absf %[[y_2D]] : f32
    // CHECK-DAG: %[[x2D:.*]] = arith.subf %36, %[[x2]] : f32
    // CHECK-DAG: %[[x2Dist:.*]] = math.absf %[[x2D]] : f32
    // CHECK-DAG: %[[x1D:.*]] = arith.subf %36, %[[x1]] : f32
    // CHECK-DAG: %[[x1Dist:.*]] = math.absf %[[x1D]] : f32
    // CHECK-DAG: %[[x_1D:.*]] = arith.subf %36, %[[x_1]] : f32
    // CHECK-DAG: %[[x_1Dist:.*]] = math.absf %[[x_1D]] : f32
    // CHECK-DAG: %[[x_2D:.*]] = arith.subf %36, %[[x_2]] : f32
    // CHECK-DAG: %[[x_2Dist:.*]] = math.absf %[[x_2D]] : f32
    // CHECK-DAG: %[[distSQ:.*]] = arith.mulf %52, %52 : f32
    // CHECK-DAG: %[[distCubed:.*]] = arith.mulf %[[distSQ]], %52 : f32
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %str = torch.constant.str "cubic"
    %int2 = torch.constant.int 2
    %0 = torch.aten.select.int %arg1, %int0, %int2 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
    %int3 = torch.constant.int 3
    %2 = torch.aten.select.int %arg1, %int0, %int3 : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %3 = torch.aten.item %2 : !torch.vtensor<[1],si64> -> !torch.int
    %4 = torch.prim.ListConstruct %1, %3 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.__interpolate.size_list_scale_list %arg0, %4, %none_0, %str, %false, %none_0, %false : !torch.vtensor<[1,1,2,4],f32>, !torch.list<int>, !torch.none, !torch.str, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
    return %5 : !torch.vtensor<[?,?,?,?],f32>
}

// -----
