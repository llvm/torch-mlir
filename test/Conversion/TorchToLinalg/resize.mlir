// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_resize_sizes_linear
func.func @test_resize_sizes_linear(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4]
,si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
     // CHECK: %[[generic:.*]] = linalg.generic
     // CHECK: %[[cst:.*]] = arith.constant 1.001000e+00 : f32
     // CHECK: %[[cst_4:.*]] = arith.constant 1.000000e+00 : f32
     // CHECK: %[[cst_5:.*]] = arith.constant 5.000000e-01 : f32
     // CHECK: %[[cst_6:.*]] = arith.constant 0.000000e+00 : f32
     // CHECK: %[[x13:.*]] = linalg.index 2 : index
     // CHECK: %[[x14:.*]] = linalg.index 3 : index
     // CHECK: %[[x15:.*]] = arith.sitofp %[[c2_i64:.*]] : i64 to f32
     // CHECK: %[[x16:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
     // CHECK: %[[x17:.*]] = arith.divf %[[x16]], %[[x15]] : f32
     // CHECK: %[[x18:.*]] = arith.index_cast %[[x13]] : index to i64
     // CHECK: %[[x19:.*]] = arith.sitofp %[[x18]] : i64 to f32
     // CHECK: %[[x20:.*]] = arith.addf %[[x19]], %[[cst_5]] : f32
     // CHECK: %[[x21:.*]] = arith.divf %[[x20]], %[[x17]] : f32
     // CHECK: %[[x22:.*]] = arith.subf %[[x21]], %[[cst_5]] : f32
     // CHECK: %[[x23:.*]] = arith.maximumf %[[x22]], %[[cst_6]] : f32
     // CHECK: %[[x24:.*]] = arith.subf %[[x15]], %[[cst]] : f32
     // CHECK: %[[x25:.*]] = arith.minimumf %[[x23]], %[[x24]] : f32
     // CHECK: %[[x26:.*]] = arith.sitofp %[[c4_i64:.*]] : i64 to f32
     // CHECK: %[[x27:.*]] = arith.sitofp %[[x7:.*]] : i64 to f32
     // CHECK: %[[x28:.*]] = arith.divf %[[x27]], %[[x26]] : f32
     // CHECK: %[[x29:.*]] = arith.index_cast %[[x14]] : index to i64
     // CHECK: %[[x30:.*]] = arith.sitofp %[[x29]] : i64 to f32
     // CHECK: %[[x31:.*]] = arith.addf %[[x30]], %[[cst_5]] : f32
     // CHECK: %[[x32:.*]] = arith.divf %[[x31]], %[[x28]] : f32
     // CHECK: %[[x33:.*]] = arith.subf %[[x32]], %[[cst_5]] : f32
     // CHECK: %[[x34:.*]] = arith.maximumf %[[x33]], %[[cst_6]] : f32
     // CHECK: %[[x35:.*]] = arith.subf %[[x26]], %[[cst]] : f32
     // CHECK: %[[x36:.*]] = arith.minimumf %[[x34]], %[[x35]] : f32
     // CHECK: %[[x37:.*]] = math.floor %[[x25]] : f32
     // CHECK: %[[x38:.*]] = arith.addf %[[cst_4]], %[[x25]] : f32
     // CHECK: %[[x39:.*]] = math.floor %[[x38]] : f32
     // CHECK: %[[x40:.*]] = math.floor %[[x36]] : f32
     // CHECK: %[[x41:.*]] = arith.addf %[[cst_4]], %[[x36]] : f32
     // CHECK: %[[x42:.*]] = math.floor %[[x41]] : f32
     // CHECK: %[[x43:.*]] = linalg.index 0 : index
     // CHECK: %[[x44:.*]] = linalg.index 1 : index
     // CHECK: %[[x45:.*]] = linalg.index 2 : index
     // CHECK: %[[x46:.*]] = linalg.index 3 : index
     // CHECK: %[[x47:.*]] = arith.fptosi %[[x37]] : f32 to i64
     // CHECK: %[[x48:.*]] = arith.index_cast %[[x47]] : i64 to index
     // CHECK: %[[x49:.*]] = arith.fptosi %[[x40]] : f32 to i64
     // CHECK: %[[x50:.*]] = arith.index_cast %[[x49]] : i64 to index
     // CHECK: %[[x51:.*]] = arith.fptosi %[[x39]] : f32 to i64
     // CHECK: %[[x52:.*]] = arith.index_cast %[[x51]] : i64 to index
     // CHECK: %[[x53:.*]] = arith.fptosi %[[x42]] : f32 to i64
     // CHECK: %[[x54:.*]] = arith.index_cast %[[x53]] : i64 to index
     // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x43]], %[[x44]], %[[x48]], %[[x50]]] : tensor<1x1x2x4xf32>
     // CHECK: %[[extracted_7:.*]] = tensor.extract %[[x0]][%[[x43]], %[[x44]], %[[x48]], %[[x54]]] : tensor<1x1x2x4xf32>
     // CHECK: %[[extracted_8:.*]] = tensor.extract %[[x0]][%[[x43]], %[[x44]], %[[x52]], %[[x50]]] : tensor<1x1x2x4xf32>
     // CHECK: %[[extracted_9:.*]] = tensor.extract %[[x0]][%[[x43]], %[[x44]], %[[x52]], %[[x54]]] : tensor<1x1x2x4xf32>
     // CHECK: %[[x55:.*]] = arith.subf %[[x42]], %[[x36]] : f32
     // CHECK: %[[x56:.*]] = arith.subf %[[x42]], %[[x40]] : f32
     // CHECK: %[[x57:.*]] = arith.divf %[[x55]], %[[x56]] : f32
     // CHECK: %[[x58:.*]] = arith.mulf %[[x57]], %extracted : f32
     // CHECK: %[[x59:.*]] = arith.subf %[[x36]], %[[x40]] : f32
     // CHECK: %[[x60:.*]] = arith.divf %[[x59]], %[[x56]] : f32
     // CHECK: %[[x61:.*]] = arith.mulf %[[x60]], %[[extracted_7]] : f32
     // CHECK: %[[x62:.*]] = arith.addf %[[x58]], %[[x61]] : f32
     // CHECK: %[[x63:.*]] = arith.mulf %[[x57]], %[[extracted_8]] : f32
     // CHECK: %[[x64:.*]] = arith.mulf %[[x60]], %[[extracted_9]] : f32
     // CHECK: %[[x65:.*]] = arith.addf %[[x63]], %[[x64]] : f32
     // CHECK: %[[x66:.*]] = arith.subf %[[x39]], %[[x25]] : f32
     // CHECK: %[[x67:.*]] = arith.subf %[[x39]], %[[x37]] : f32
     // CHECK: %[[x68:.*]] = arith.divf %[[x66]], %[[x67]] : f32
     // CHECK: %[[x69:.*]] = arith.mulf %[[x68]], %[[x62]] : f32
     // CHECK: %[[x70:.*]] = arith.subf %[[x25]], %[[x37]] : f32
     // CHECK: %[[x71:.*]] = arith.divf %[[x70]], %[[x67]] : f32
     // CHECK: %[[x72:.*]] = arith.mulf %[[x71]], %[[x65]] : f32
     // CHECK: %[[x73:.*]] = arith.addf %[[x69]], %[[x72]] : f32
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

func.func @test_resize_sizes_nearest(%arg0: !torch.vtensor<[1,1,2,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[GENERIC:.*]] = linalg.generic
    // CHECK: %[[x13:.*]] = linalg.index 2 : index
    // CHECK: %[[x14:.*]] = linalg.index 3 : index
    // CHECK: %[[x15:.*]] = arith.sitofp %[[c2_i64:.*]] : i64 to f32
    // CHECK: %[[x16:.*]] = arith.sitofp %[[c4_i64:.*]] : i64 to f32
    // CHECK: %[[x19:.*]] = arith.sitofp %[[x6:.*]] : i64 to f32
    // CHECK: %[[x20:.*]] = arith.sitofp %[[x7:.*]] : i64 to f32
    // CHECK: %[[x21:.*]] = arith.divf %[[x19]], %[[x15]] : f32
    // CHECK: %[[x22:.*]] = arith.divf %[[x20]], %[[x16]] : f32
    // CHECK: %[[x23:.*]] = arith.index_cast %[[x13]] : index to i64
    // CHECK: %[[x24:.*]] = arith.sitofp %[[x23]] : i64 to f32
    // CHECK: %[[x25:.*]] = arith.divf %[[x24]], %[[x21]] : f32
    // CHECK: %[[x26:.*]] = arith.index_cast %[[x14]] : index to i64
    // CHECK: %[[x27:.*]] = arith.sitofp %[[x26]] : i64 to f32
    // CHECK: %[[x28:.*]] = arith.divf %[[x27]], %[[x22]] : f32
    // CHECK: %[[x29:.*]] = math.floor %[[x25]] : f32
    // CHECK: %[[x30:.*]] = math.floor %[[x28]] : f32
    // CHECK: %[[x31:.*]] = arith.fptosi %[[x29]] : f32 to i64
    // CHECK: %[[x32:.*]] = arith.index_cast %[[x31]] : i64 to index
    // CHECK: %[[x33:.*]] = arith.fptosi %[[x30]] : f32 to i64
    // CHECK: %[[x34:.*]] = arith.index_cast %[[x33]] : i64 to index
    // CHECK: %[[x35:.*]] = linalg.index 0 : index
    // CHECK: %[[x36:.*]] = linalg.index 1 : index
    // CHECK: %[[x37:.*]] = linalg.index 2 : index
    // CHECK: %[[x38:.*]] = linalg.index 3 : index
    // CHECK: %[[extracted:.*]] = tensor.extract %[[x0:.*]][%[[x35]], %[[x36]], %[[x32]], %[[x34]]] : tensor<1x1x2x4xf32>
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
