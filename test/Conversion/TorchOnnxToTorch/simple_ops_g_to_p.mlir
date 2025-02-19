// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch --split-input-file | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.

// CHECK-LABEL: func.func @test_greater
func.func @test_greater(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.gt.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Greater"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_greater_or_equal
func.func @test_greater_or_equal(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.ge.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.GreaterOrEqual"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_less
func.func @test_less(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4,5],f32>
  // CHECK: torch.aten.lt.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],i1>
  %0 = torch.operator "onnx.Less"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}

// -----

// CHECK-LABEL: func.func @test_gather
func.func @test_gather(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[8,10,20,40], si64>) -> !torch.vtensor<[8,10,20,40,4,5],f32> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[AXIS:.+]] = torch.constant.int 0
  // CHECK: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK: %[[LT:.+]] = torch.aten.lt.Scalar %arg1, %[[ZERO]]
  // CHECK: %[[SZ:.+]] = torch.aten.size.int %arg0, %[[AXIS]]
  // CHECK: %[[ADD:.+]] = torch.aten.add.Scalar %arg1, %[[SZ]], %[[ONE]]
  // CHECK: %[[SEL:.+]] = torch.aten.where.self %[[LT]], %[[ADD]], %arg1
  // CHECK: %[[D0:.+]] = torch.constant.int 0
  // CHECK: %[[SZ0:.+]] = torch.aten.size.int %[[SEL]], %[[D0]]
  // CHECK: %[[D1:.+]] = torch.constant.int 1
  // CHECK: %[[SZ1:.+]] = torch.aten.size.int %[[SEL]], %[[D1]]
  // CHECK: %[[D2:.+]] = torch.constant.int 2
  // CHECK: %[[SZ2:.+]] = torch.aten.size.int %[[SEL]], %[[D2]]
  // CHECK: %[[D3:.+]] = torch.constant.int 3
  // CHECK: %[[SZ3:.+]] = torch.aten.size.int %[[SEL]], %[[D3]]
  // CHECK: %[[SZ:.+]] = torch.prim.ListConstruct %[[SZ0]], %[[SZ1]], %[[SZ2]], %[[SZ3]]
  // CHECK: %[[DIM:.+]] = torch.aten.dim %[[SEL]]
  // CHECK: %[[SUB:.+]] = torch.aten.sub.int %[[DIM]], %[[ONE]]
  // CHECK: %[[FLAT:.+]] = torch.aten.flatten.using_ints %[[SEL]], %[[ZERO]], %[[SUB]]
  // CHECK: %[[ISEL:.+]] = torch.aten.index_select %arg0, %[[AXIS]], %[[FLAT]]
  // CHECK: %[[RES:.+]] = torch.aten.unflatten.int %[[ISEL]], %[[AXIS]], %[[SZ]]
  // CHECK: return %[[RES]]
  %0 = torch.operator "onnx.Gather"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[8,10,20,40], si64>) -> !torch.vtensor<[8,10,20,40,4,5],f32>
  return %0 : !torch.vtensor<[8,10,20,40,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_gather_scalar
func.func @test_gather_scalar(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[], si64>) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[AXIS:.+]] = torch.constant.int 0
  // CHECK: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK: %[[LT:.+]] = torch.aten.lt.Scalar %arg1, %[[ZERO]]
  // CHECK: %[[SZ:.+]] = torch.aten.size.int %arg0, %[[AXIS]]
  // CHECK: %[[ADD:.+]] = torch.aten.add.Scalar %arg1, %[[SZ]], %[[ONE]]
  // CHECK: %[[SEL:.+]] = torch.aten.where.self %[[LT]], %[[ADD]], %arg1
  // CHECK: %[[FLAT:.+]] = torch.aten.unsqueeze %[[SEL]], %[[ZERO]] : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[ISEL:.+]] = torch.aten.index_select %arg0, %[[AXIS]], %[[FLAT]]
  // CHECK: %[[RES:.+]] = torch.aten.squeeze.dim %[[ISEL]], %[[AXIS]] : !torch.vtensor<[1,4,5],f32>, !torch.int -> !torch.vtensor<[4,5],f32>
  // CHECK: return %[[RES]]
  %0 = torch.operator "onnx.Gather"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[], si64>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_gather_nd
func.func @test_gather_nd(%arg0: !torch.vtensor<[2,6,8,5],f32>, %arg1: !torch.vtensor<[2,4,3,2], si64>) -> !torch.vtensor<[2,4,3,5],f32> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[SIZE0:.+]] = torch.aten.size.int %arg0, %[[INT0]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[SIZE1:.+]] = torch.aten.size.int %arg0, %[[INT1]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[SIZE2:.+]] = torch.aten.size.int %arg0, %[[INT2]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[SIZE3:.+]] = torch.aten.size.int %arg0, %[[INT3]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT0_3:.+]] = torch.constant.int 0
  // CHECK: %[[INT1_4:.+]] = torch.constant.int 1
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INDSIZE0:.+]] = torch.aten.size.int %arg1, %[[INT0_0]] : !torch.vtensor<[2,4,3,2],si64>, !torch.int -> !torch.int
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INDSIZE1:.+]] = torch.aten.size.int %arg1, %[[INT1_1]] : !torch.vtensor<[2,4,3,2],si64>, !torch.int -> !torch.int
  // CHECK: %[[MUL1:.+]] = torch.aten.mul.int %[[INT1_4]], %[[INDSIZE1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[INT2_2:.+]] = torch.constant.int 2
  // CHECK: %[[INDSIZE2:.+]] = torch.aten.size.int %arg1, %[[INT2_2]] : !torch.vtensor<[2,4,3,2],si64>, !torch.int -> !torch.int
  // CHECK: %[[MUL2:.+]] = torch.aten.mul.int %[[MUL1]], %[[INDSIZE2]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[INT3_5:.+]] = torch.constant.int 3
  // CHECK: %[[INT1_6:.+]] = torch.constant.int 1
  // CHECK: %[[SLICE0:.+]] = torch.aten.slice.Tensor %arg1, %[[INT3_5]], %[[INT0_3]], %[[INT1_6]], %[[INT1_4]] : !torch.vtensor<[2,4,3,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[LT0:.+]] = torch.aten.lt.Scalar %[[SLICE0]], %[[INT0_3]] : !torch.vtensor<[2,4,3,1],si64>, !torch.int -> !torch.vtensor<[2,4,3,1],i1>
  // CHECK: %[[ADD0:.+]] = torch.aten.add.Scalar %[[SLICE0]], %[[SIZE1]], %[[INT1_4]] : !torch.vtensor<[2,4,3,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[WHERE0:.+]] = torch.aten.where.self %[[LT0]], %[[ADD0]], %[[SLICE0]] : !torch.vtensor<[2,4,3,1],i1>, !torch.vtensor<[2,4,3,1],si64>, !torch.vtensor<[2,4,3,1],si64> -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[INT2_7:.+]] = torch.constant.int 2
  // CHECK: %[[SLICE1:.+]] = torch.aten.slice.Tensor %arg1, %[[INT3_5]], %[[INT1_6]], %[[INT2_7]], %[[INT1_4]] : !torch.vtensor<[2,4,3,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[LT1:.+]] = torch.aten.lt.Scalar %[[SLICE1]], %[[INT0_3]] : !torch.vtensor<[2,4,3,1],si64>, !torch.int -> !torch.vtensor<[2,4,3,1],i1>
  // CHECK: %[[ADD1:.+]] = torch.aten.add.Scalar %[[SLICE1]], %[[SIZE2]], %[[INT1_4]] : !torch.vtensor<[2,4,3,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[WHERE1:.+]] = torch.aten.where.self %[[LT1]], %[[ADD1]], %[[SLICE1]] : !torch.vtensor<[2,4,3,1],i1>, !torch.vtensor<[2,4,3,1],si64>, !torch.vtensor<[2,4,3,1],si64> -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[NEWIND:.+]] = torch.aten.add.Tensor %[[WHERE1]], %[[WHERE0]], %[[SIZE2]] : !torch.vtensor<[2,4,3,1],si64>, !torch.vtensor<[2,4,3,1],si64>, !torch.int -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[LIST0:.+]] = torch.prim.ListConstruct %[[INDSIZE0]], %[[INDSIZE1]], %[[INDSIZE2]], %[[INT1_4]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[VIEW:.+]] = torch.aten.view %[[NEWIND]], %[[LIST0]] : !torch.vtensor<[2,4,3,1],si64>, !torch.list<int> -> !torch.vtensor<[2,4,3,1],si64>
  // CHECK: %[[INT1_8:.+]] = torch.constant.int 1
  // CHECK: %[[INT2_9:.+]] = torch.constant.int 2
  // CHECK: %[[FLATIND:.+]] = torch.aten.flatten.using_ints %[[VIEW]], %[[INT1_8]], %[[INT2_9]] : !torch.vtensor<[2,4,3,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,12,1],si64>
  // CHECK: %[[EXLIST:.+]] = torch.prim.ListConstruct %[[SIZE0]], %[[MUL2]], %[[SIZE3]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[EXPANDIND:.+]] = torch.aten.expand %[[FLATIND]], %[[EXLIST]], %[[FALSE]] : !torch.vtensor<[2,12,1],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,12,5],si64>
  // CHECK: %[[INT2_10:.+]] = torch.constant.int 2
  // CHECK: %[[FLATDATA:.+]] = torch.aten.flatten.using_ints %arg0, %[[INT1_8]], %[[INT2_10]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,48,5],f32>
  // CHECK: %[[GATHER:.+]] = torch.aten.gather %[[FLATDATA]], %[[INT1_8]], %[[EXPANDIND]], %[[FALSE]]
  // CHECK: %[[LIST1:.+]] = torch.prim.ListConstruct %[[INDSIZE1]], %[[INDSIZE2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLAT:.+]] = torch.aten.unflatten.int %[[GATHER]], %[[INT1_8]], %[[LIST1]] : !torch.vtensor<[2,12,5],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[2,4,3,5],f32>
  // CHECK: return %[[UNFLAT]] : !torch.vtensor<[2,4,3,5],f32>
  %0 = torch.operator "onnx.GatherND"(%arg0, %arg1) {torch.onnx.batch_dims = 1 : si64} : (!torch.vtensor<[2,6,8,5],f32>, !torch.vtensor<[2,4,3,2], si64>) -> !torch.vtensor<[2,4,3,5],f32>
  return %0 : !torch.vtensor<[2,4,3,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_gather_nd_1D_indices
func.func @test_gather_nd_1D_indices(%arg0: !torch.vtensor<[2,6,8,5],f32>, %arg1: !torch.vtensor<[2], si64>) -> !torch.vtensor<[8,5],f32> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[SIZE0:.+]] = torch.aten.size.int %arg0, %[[INT0]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[SIZE1:.+]] = torch.aten.size.int %arg0, %[[INT1]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[SIZE2:.+]] = torch.aten.size.int %arg0, %[[INT2]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[SIZE3:.+]] = torch.aten.size.int %arg0, %[[INT3]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int -> !torch.int
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INT0_2:.+]] = torch.constant.int 0
  // CHECK: %[[INT1_3:.+]] = torch.constant.int 1
  // CHECK: %[[SLICE0:.+]] = torch.aten.slice.Tensor %arg1, %[[INT0_2]], %[[INT0_0]], %[[INT1_3]], %[[INT1_1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[LT0:.+]] = torch.aten.lt.Scalar %[[SLICE0]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],i1>
  // CHECK: %[[ADD0:.+]] = torch.aten.add.Scalar %[[SLICE0]], %[[SIZE0]], %[[INT1_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[WHERE0:.+]] = torch.aten.where.self %[[LT0]], %[[ADD0]], %[[SLICE0]] : !torch.vtensor<[1],i1>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[INT2_4:.+]] = torch.constant.int 2
  // CHECK: %[[SLICE1:.+]] = torch.aten.slice.Tensor %arg1, %[[INT0_2]], %[[INT1_3]], %[[INT2_4]], %[[INT1_1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[LT1:.+]] = torch.aten.lt.Scalar %[[SLICE1]], %[[INT0_0]] : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],i1>
  // CHECK: %[[ADD1:.+]] = torch.aten.add.Scalar %[[SLICE1]], %[[SIZE1]], %[[INT1_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[WHERE1:.+]] = torch.aten.where.self %[[LT1]], %[[ADD1]], %[[SLICE1]] : !torch.vtensor<[1],i1>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK: %[[NEWIND:.+]] = torch.aten.add.Tensor %[[WHERE1]], %[[WHERE0]], %[[SIZE1]] : !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: %[[LIST0:.+]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[VIEW:.+]] = torch.aten.view %[[NEWIND]], %[[LIST0]] : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
  // CHECK: %[[INT0_5:.+]] = torch.constant.int 0
  // CHECK: %[[FLATIND:.+]] = torch.aten.unsqueeze %[[VIEW]], %[[INT0_0]] : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1,1],si64>
  // CHECK: %[[EXLIST:.+]] = torch.prim.ListConstruct %[[INT1_1]], %[[SIZE2]], %[[SIZE3]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[EXPANDIND:.+]] = torch.aten.expand %[[FLATIND]], %[[EXLIST]], %[[FALSE]] : !torch.vtensor<[1,1,1],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,5],si64>
  // CHECK: %[[INT1_6:.+]] = torch.constant.int 1
  // CHECK: %[[FLATDATA:.+]] = torch.aten.flatten.using_ints %arg0, %[[INT0_5]], %[[INT1_6]] : !torch.vtensor<[2,6,8,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,8,5],f32>
  // CHECK: %[[GATHER:.+]] = torch.aten.gather %[[FLATDATA]], %[[INT0_5]], %[[EXPANDIND]], %[[FALSE]]
  // CHECK: %[[SQUEEZE:.+]] = torch.aten.squeeze.dim %[[GATHER]], %[[INT0_0]] : !torch.vtensor<[1,8,5],f32>, !torch.int -> !torch.vtensor<[8,5],f32>
  // CHECK: return %[[SQUEEZE]] : !torch.vtensor<[8,5],f32>
  %0 = torch.operator "onnx.GatherND"(%arg0, %arg1) {torch.onnx.batch_dims = 0 : si64} : (!torch.vtensor<[2,6,8,5],f32>, !torch.vtensor<[2], si64>) -> !torch.vtensor<[8,5],f32>
  return %0 : !torch.vtensor<[8,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_gathernd_example_int32_batch_dim1
func.func @test_gathernd_example_int32_batch_dim1(%arg0: !torch.vtensor<[2,2,2],si32>, %arg1: !torch.vtensor<[2,1],si64>) -> !torch.vtensor<[2,2],si32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[DIM0:.+]] = torch.aten.size.int %arg0, %[[INT0]]
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[DIM1:.+]] = torch.aten.size.int %arg0, %[[INT1]]
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[DIM2:.+]] = torch.aten.size.int %arg0, %[[INT2]]
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 1
  // CHECK: %[[INT0_2:.+]] = torch.constant.int 0
  // CHECK: %[[B0:.+]] = torch.aten.size.int %arg1, %[[INT0_2]]
  // CHECK: %[[INT1_3:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_4:.+]] = torch.constant.int 1
  // CHECK: %[[SLICE:.+]] = torch.aten.slice.Tensor %arg1, %[[INT1_3]], %[[INT0_0]], %[[INT1_4]], %[[INT1_1]]
  // CHECK: %[[LT:.+]] = torch.aten.lt.Scalar %[[SLICE]], %[[INT0_0]]
  // CHECK: %[[ADD:.+]] = torch.aten.add.Scalar %[[SLICE]], %[[DIM1]], %[[INT1_1]]
  // CHECK: %[[WHERE:.+]] = torch.aten.where.self %[[LT]], %[[ADD]], %[[SLICE]]
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[B0]], %[[INT1_1]]
  // CHECK: %[[VIEW:.+]] = torch.aten.view %[[WHERE]], %[[LIST]]
  // CHECK: %[[INT1_5:.+]] = torch.constant.int 1
  // CHECK: %[[UNSQ:.+]] = torch.aten.unsqueeze %[[VIEW]], %[[INT1_5]]
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[DIM0]], %[[INT1_1]], %[[DIM2]]
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[EXPAND:.+]] = torch.aten.expand %[[UNSQ]], %[[LIST]], %[[FALSE]]
  // CHECK: %[[INT1_6:.+]] = torch.constant.int 1
  // CHECK: %[[GATHER:.+]] = torch.aten.gather %arg0, %[[INT1_5]], %[[EXPAND]], %[[FALSE]]
  // CHECK: %[[SQ:.+]] = torch.aten.squeeze.dim %[[GATHER]], %[[INT1_5]]
  %none = torch.constant.none
  %0 = torch.operator "onnx.GatherND"(%arg0, %arg1) {torch.onnx.batch_dims = 1 : si64} : (!torch.vtensor<[2,2,2],si32>, !torch.vtensor<[2,1],si64>) -> !torch.vtensor<[2,2],si32>
  return %0 : !torch.vtensor<[2,2],si32>
}

// -----

// CHECK-LABEL: func.func @test_gather_elements
func.func @test_gather_elements(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5], si64>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[INT0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[DIM:.+]] = torch.aten.size.int %arg0, %[[INT0]]
  // CHECK-DAG: %[[ADD:.+]] = torch.aten.add.Scalar %arg1, %[[DIM]], %[[ONE]]
  // CHECK-DAG: %[[LT:.+]] = torch.aten.lt.Scalar %arg1, %[[INT0]]
  // CHECK-DAG: %[[WHERE:.+]] = torch.aten.where.self %[[LT]], %[[ADD]], %arg1
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[GATHER:.+]] = torch.aten.gather %arg0, %[[INT0]], %[[WHERE]], %[[FALSE]]
  %0 = torch.operator "onnx.GatherElements"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5], si64>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_defaultA
func.func @test_gemm_defaultA(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1) : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_defaultB
func.func @test_gemm_defaultB(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[I1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_transposeA
func.func @test_gemm_transposeA(%arg0: !torch.vtensor<[5,3],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[I0:.+]] = torch.constant.int 0
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[TRANS:.+]] = torch.aten.transpose.int %arg0, %[[I0]], %[[I1]] : !torch.vtensor<[5,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,5],f32>
  // CHECK: %[[MM:.+]] = torch.aten.mm %[[TRANS]], %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[I1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.transA = 1 : si64} : (!torch.vtensor<[5,3],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_transposeB
func.func @test_gemm_transposeB(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[4,5],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK: %[[I0:.+]] = torch.constant.int 0
  // CHECK: %[[I1:.+]] = torch.constant.int 1
  // CHECK: %[[TRANS:.+]] = torch.aten.transpose.int %arg1, %[[I0]], %[[I1]] : !torch.vtensor<[4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[5,4],f32>
  // CHECK: %[[MM:.+]] = torch.aten.mm %arg0, %[[TRANS]] : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[I1]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.transB = 1 : si64} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_alpha
func.func @test_gemm_alpha(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK-DAG: %[[ALPHA:.+]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.add.Tensor %arg2, %[[MM]], %[[ALPHA]] : !torch.vtensor<[1,4],f32>, !torch.vtensor<[3,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.alpha = 5.000000e-01 : f32} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_beta
func.func @test_gemm_beta(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK-DAG: %[[BETA:.+]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.add.Tensor %[[MM]], %arg2, %[[BETA]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.beta = 5.000000e-01 : f32} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_gemm_alpha_beta
func.func @test_gemm_alpha_beta(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64} {
  // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[I1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
  // CHECK-DAG: %[[ALPHA:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[BETA:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[MUL:.+]] = torch.aten.mul.Scalar %[[MM]], %[[ALPHA]] : !torch.vtensor<[3,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  // CHECK: torch.aten.add.Tensor %[[MUL]], %arg2, %[[BETA]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
  %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.alpha = 5.000000e-01 : f32, torch.onnx.beta = 2.500000e-01 : f32} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_lppool_2d
func.func @test_lppool_2d(%arg0: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,31,31],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  // CHECK: %[[I1:.*]] = torch.constant.int 1
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[NE:.*]] = torch.aten.mul %[[I2]], %[[I1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[I2_1:.*]] = torch.constant.int 2
  // CHECK: %[[NE1:.*]] = torch.aten.mul %[[I2_1]], %[[NE]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[K:.*]] = torch.prim.ListConstruct %[[I2]], %[[I2_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[I0_2:.*]] = torch.constant.int 0
  // CHECK: %[[I0_3:.*]] = torch.constant.int 0
  // CHECK: %[[PAD:.*]] = torch.prim.ListConstruct %[[I0]], %[[I0_1]], %[[I0_2]], %[[I0_3]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[I1_2:.*]] = torch.constant.int 1
  // CHECK: %[[STR:.*]] = torch.prim.ListConstruct %[[I1_1]], %[[I1_2]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[CEIL:.*]] = torch.constant.bool false
  // CHECK: %[[CIP:.*]] = torch.constant.bool true
  // CHECK: %[[P:.*]] = torch.constant.int 2
  // CHECK: %[[ABS:.*]] = torch.aten.abs %arg0 : !torch.vtensor<[1,3,32,32],f32> -> !torch.vtensor<[1,3,32,32],f32>
  // CHECK: %[[POW:.*]] = torch.aten.pow.Tensor_Scalar %[[ABS]], %[[P]] : !torch.vtensor<[1,3,32,32],f32>, !torch.int -> !torch.vtensor<[1,3,32,32],f32>
  // CHECK: %[[AVG:.*]] = torch.aten.avg_pool2d %[[POW]], %[[K]], %[[STR]], %[[PAD]], %[[CEIL]], %[[CIP]], %[[I1]] : !torch.vtensor<[1,3,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.int -> !torch.vtensor<[1,3,31,31],f32>
  // CHECK: %[[INVP:.*]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.pow.Tensor_Scalar %[[AVG]], %[[INVP]] : !torch.vtensor<[1,3,31,31],f32>, !torch.float -> !torch.vtensor<[1,3,31,31],f32>
  %0 = torch.operator "onnx.LpPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,31,31],f32>
  return %0 : !torch.vtensor<[1,3,31,31],f32>
}

// -----

// CHECK-LABEL: func.func @test_lppool_1d
func.func @test_lppool_1d(%arg0: !torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64} {
  // CHECK: %[[I1:.*]] = torch.constant.int 1
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[NE:.*]] = torch.aten.mul %[[I2]], %[[I1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[K:.*]] = torch.prim.ListConstruct %[[I2]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[I0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[PAD:.*]] = torch.prim.ListConstruct %[[I0]], %[[I0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[STR:.*]] = torch.prim.ListConstruct %[[I1_1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[CEIL:.*]] = torch.constant.bool false
  // CHECK: %[[CIP:.*]] = torch.constant.bool true
  // CHECK: %[[P:.*]] = torch.constant.int 2
  // CHECK: %[[ABS:.*]] = torch.aten.abs %arg0 : !torch.vtensor<[1,3,32],f32> -> !torch.vtensor<[1,3,32],f32>
  // CHECK: %[[POW:.*]] = torch.aten.pow.Tensor_Scalar %[[ABS]], %[[P]] : !torch.vtensor<[1,3,32],f32>, !torch.int -> !torch.vtensor<[1,3,32],f32>
  // CHECK: %[[AVG:.*]] = torch.aten.avg_pool1d %[[POW]], %[[K]], %[[STR]], %[[PAD]], %[[CEIL]], %[[CIP]] : !torch.vtensor<[1,3,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,3,31],f32>
  // CHECK: %[[POW_0:.*]] = torch.aten.mul.Scalar %[[AVG]], %[[NE]] : !torch.vtensor<[1,3,31],f32>, !torch.int -> !torch.vtensor<[1,3,31],f32>
  // CHECK: %[[INVP:.*]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.pow.Tensor_Scalar %[[POW_0]], %[[INVP]] : !torch.vtensor<[1,3,31],f32>, !torch.float -> !torch.vtensor<[1,3,31],f32>
  %0 = torch.operator "onnx.LpPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64]} : (!torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32>
  return %0 : !torch.vtensor<[1,3,31],f32>
}

// -----

// CHECK-LABEL : func.func @test_layer_norm
func.func @test_layer_norm(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[3,4],f32>, %arg2: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4], f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>)
                           attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %int3 = torch.constant.int 3
  // CHECK: %int4 = torch.constant.int 4
  // CHECK: %0 = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %0, %arg1, %arg2
  %0:3 = torch.operator "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>
}

// -----

// CHECK-LABEL : func.func @test_layer_norm_single_result
func.func @test_layer_norm_single_result(%arg0: !torch.vtensor<[1,4,768],f32>, %arg1: !torch.vtensor<[768],f32>, %arg2: !torch.vtensor<[768],f32>) -> (!torch.vtensor<[1,4,768], f32>)
                           attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %float9.999990e-06 = torch.constant.float 9.9999997473787516E-6
  // CHECK: %int768 = torch.constant.int 768
  // CHECK: %0 = torch.prim.ListConstruct %int768 : (!torch.int) -> !torch.list<int>
  // CHECK: %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %0, %arg1, %arg2
  %0 = torch.operator "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 9.99999974E-6 : f32} : (!torch.vtensor<[1,4,768],f32>, !torch.vtensor<[768],f32>, !torch.vtensor<[768],f32>) -> !torch.vtensor<[1,4,768],f32>
  return %0 : !torch.vtensor<[1,4,768],f32>
}

// -----

// CHECK-LABEL: func.func @test_leaky_relu
func.func @test_leaky_relu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 16 : si64} {
  // CHECK-DAG: %[[F2:.+]] = torch.constant.float 2
  // CHECK: %[[LRELU:.+]] = torch.aten.leaky_relu %arg0, %[[F2]]
  %0 = torch.operator "onnx.LeakyRelu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_lrn_default
func.func @test_lrn_default(%arg0: !torch.vtensor<[20,10,3,50],f32>) -> !torch.vtensor<[20,10,3,50],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
    // CHECK-DAG: %[[TRUE:.+]] = torch.constant.bool true
    // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
    // CHECK-DAG: %[[F0:.+]] = torch.constant.float 0.000000e+00
    // CHECK-DAG: %[[ALPHA:.*]] = torch.constant.float 9.9999997473787516E-5
    // CHECK-DAG: %[[BETA:.*]] = torch.constant.float 7.500000e-01
    // CHECK-DAG: %[[BIAS:.*]] = torch.constant.float 1.000000e+00
    // CHECK-DAG: %[[INSQ:.*]] = torch.aten.mul.Tensor %arg0, %arg0

    // CHECK-DAG: %[[I20:.*]] = torch.constant.int 20
    // CHECK-DAG: %[[I1:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I10:.*]] = torch.constant.int 10
    // CHECK-DAG: %[[I3:.+]] = torch.constant.int 3
    // CHECK-DAG: %[[IMINUS1:.+]] = torch.constant.int -1
    // CHECK-DAG: %[[VIEWSHAPE:.*]] = torch.prim.ListConstruct %[[I20]], %[[I1]], %[[I10]], %[[I3]], %[[IMINUS1]]

    // CHECK-DAG: %[[VIEW1:.*]] = torch.aten.view %[[INSQ]], %[[VIEWSHAPE]]

    // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_2:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_3:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_4:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I1_2:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I1_3:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[PADDING:.*]] = torch.prim.ListConstruct %[[I0]], %[[I0_2]], %[[I0_3]], %[[I0_4]], %[[I1_2]], %[[I1_3]]

    // CHECK-DAG: %[[PADDED:.*]] = torch.aten.constant_pad_nd %[[VIEW1]], %[[PADDING]], %[[F0]]

    // CHECK-DAG: %[[I3_2:.+]] = torch.constant.int 3
    // CHECK-DAG: %[[I1_4:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I1_5:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[I3_2]], %[[I1_4]], %[[I1_5]]

    // CHECK-DAG: %[[I1_6:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I1_7:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I1_8:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[I1_6]], %[[I1_7]], %[[I1_8]]

    // CHECK-DAG: %[[I0_5:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_6:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_7:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[POOLPADDING:.*]] = torch.prim.ListConstruct %[[I0_5]], %[[I0_6]], %[[I0_7]]

    // CHECK-DAG: %[[POOL3D:.*]] = torch.aten.avg_pool3d %[[PADDED]], %[[KERNELSIZE]], %[[STRIDES]], %[[POOLPADDING]], %[[FALSE]], %[[TRUE]]
    // CHECK-DAG: %[[SQUEEZED:.*]] = torch.aten.squeeze.dim %[[POOL3D]], %[[I1]]

    // CHECK-DAG: %[[I20_2:.*]] = torch.constant.int 20
    // CHECK-DAG: %[[I10_2:.*]] = torch.constant.int 10
    // CHECK-DAG: %[[I3_2:.+]] = torch.constant.int 3
    // CHECK-DAG: %[[I50_2:.+]] = torch.constant.int 50
    // CHECK-DAG: %[[ISHAPE:.*]] = torch.prim.ListConstruct %[[I20_2]], %[[I10_2]], %[[I3_2]], %[[I50_2]]

    // CHECK-DAG: %[[VIEW2:.*]] = torch.aten.view %[[SQUEEZED]], %[[ISHAPE]]
    // CHECK-DAG: %[[POSTALPHA:.*]] = torch.aten.mul.Scalar %[[VIEW2]], %[[ALPHA]]
    // CHECK-DAG: %[[POSTBIAS:.*]] = torch.aten.add.Scalar %[[POSTALPHA]], %[[BIAS]], %[[I1]]
    // CHECK-DAG: %[[POSTBETA:.*]] = torch.aten.pow.Tensor_Scalar %[[POSTBIAS]], %[[BETA]]
    // CHECK-DAG: %[[OUTPUT:.*]] = torch.aten.div.Tensor %arg0, %[[POSTBETA]]
    // CHECK: return  %[[OUTPUT]]
    %0 = torch.operator "onnx.LRN"(%arg0) {torch.onnx.size = 3 : si64} : (!torch.vtensor<[20,10,3,50],f32>) -> !torch.vtensor<[20,10,3,50],f32>
    return %0 : !torch.vtensor<[20,10,3,50],f32>
}

// -----

// CHECK-LABEL: func.func @test_lrn_with_optionals
func.func @test_lrn_with_optionals(%arg0: !torch.vtensor<[13,19,100,200],f32>) -> !torch.vtensor<[13,19,100,200],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
    // CHECK-DAG: %[[TRUE:.+]] = torch.constant.bool true
    // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
    // CHECK-DAG: %[[F0:.+]] = torch.constant.float 0.000000e+00
    // CHECK-DAG: %[[ALPHA:.*]] = torch.constant.float 0.0020000000949949026
    // CHECK-DAG: %[[BETA:.*]] = torch.constant.float 0.64999997615814209
    // CHECK-DAG: %[[BIAS:.*]] = torch.constant.float 3.000000e+00
    // CHECK-DAG: %[[INSQ:.*]] = torch.aten.mul.Tensor %arg0, %arg0

    // CHECK-DAG: %[[I13:.*]] = torch.constant.int 13
    // CHECK-DAG: %[[I1:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I19:.*]] = torch.constant.int 19
    // CHECK-DAG: %[[I100:.+]] = torch.constant.int 100
    // CHECK-DAG: %[[IMINUS1:.+]] = torch.constant.int -1
    // CHECK-DAG: %[[VIEWSHAPE:.*]] = torch.prim.ListConstruct %[[I13]], %[[I1]], %[[I19]], %[[I100]], %[[IMINUS1]]

    // CHECK-DAG: %[[VIEW1:.*]] = torch.aten.view %[[INSQ]], %[[VIEWSHAPE]]

    // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_2:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_3:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_4:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I2:.*]] = torch.constant.int 2
    // CHECK-DAG: %[[I2_2:.*]] = torch.constant.int 2
    // CHECK-DAG: %[[PADDING:.*]] = torch.prim.ListConstruct %[[I0]], %[[I0_2]], %[[I0_3]], %[[I0_4]], %[[I2]], %[[I2_2]]

    // CHECK-DAG: %[[PADDED:.*]] = torch.aten.constant_pad_nd %[[VIEW1]], %[[PADDING]], %[[F0]]

    // CHECK-DAG: %[[I5:.+]] = torch.constant.int 5
    // CHECK-DAG: %[[I1_4:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I1_5:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[I5]], %[[I1_4]], %[[I1_5]]

    // CHECK-DAG: %[[I1_6:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I1_7:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[I1_8:.*]] = torch.constant.int 1
    // CHECK-DAG: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[I1_6]], %[[I1_7]], %[[I1_8]]

    // CHECK-DAG: %[[I0_5:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_6:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[I0_7:.+]] = torch.constant.int 0
    // CHECK-DAG: %[[POOLPADDING:.*]] = torch.prim.ListConstruct %[[I0_5]], %[[I0_6]], %[[I0_7]]

    // CHECK-DAG: %[[POOL3D:.*]] = torch.aten.avg_pool3d %[[PADDED]], %[[KERNELSIZE]], %[[STRIDES]], %[[POOLPADDING]], %[[FALSE]], %[[TRUE]]
    // CHECK-DAG: %[[SQUEEZED:.*]] = torch.aten.squeeze.dim %[[POOL3D]], %[[I1]]

    // CHECK-DAG: %[[I13_2:.*]] = torch.constant.int 13
    // CHECK-DAG: %[[I19_2:.*]] = torch.constant.int 19
    // CHECK-DAG: %[[I100_2:.+]] = torch.constant.int 100
    // CHECK-DAG: %[[I200_2:.+]] = torch.constant.int 200
    // CHECK-DAG: %[[ISHAPE:.*]] = torch.prim.ListConstruct %[[I13_2]], %[[I19_2]], %[[I100_2]], %[[I200_2]]

    // CHECK-DAG: %[[VIEW2:.*]] = torch.aten.view %[[SQUEEZED]], %[[ISHAPE]]
    // CHECK-DAG: %[[POSTALPHA:.*]] = torch.aten.mul.Scalar %[[VIEW2]], %[[ALPHA]]
    // CHECK-DAG: %[[POSTBIAS:.*]] = torch.aten.add.Scalar %[[POSTALPHA]], %[[BIAS]], %[[I1]]
    // CHECK-DAG: %[[POSTBETA:.*]] = torch.aten.pow.Tensor_Scalar %[[POSTBIAS]], %[[BETA]]
    // CHECK-DAG: %[[OUTPUT:.*]] = torch.aten.div.Tensor %arg0, %[[POSTBETA]]
    // CHECK: return  %[[OUTPUT]]
    %none = torch.constant.none
    %0 = torch.operator "onnx.LRN"(%arg0) {torch.onnx.alpha = 2.000000e-03 : f32, torch.onnx.beta = 6.500000e-01 : f32, torch.onnx.bias = 3.000000e+00 : f32, torch.onnx.size = 5 : si64} : (!torch.vtensor<[13,19,100,200],f32>) -> !torch.vtensor<[13,19,100,200],f32>
    return %0 : !torch.vtensor<[13,19,100,200],f32>
}

// -----

// CHECK-LABEL: @test_matmul_2d
func.func @test_matmul_2d(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,3],f32> -> !torch.vtensor<[3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,3],f32>
  return %0 : !torch.vtensor<[3,3],f32>
}

// -----

// CHECK-LABEL: @test_matmul_3d
func.func @test_matmul_3d(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,4,3],f32> -> !torch.vtensor<[2,3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,3,3],f32>
  return %0 : !torch.vtensor<[2,3,3],f32>
}

// -----

// CHECK-LABEL: @test_matmul_4d
func.func @test_matmul_4d(%arg0: !torch.vtensor<[1,2,3,4],f32>, %arg1: !torch.vtensor<[1,2,4,3],f32>) -> !torch.vtensor<[1,2,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,4,3],f32> -> !torch.vtensor<[1,2,3,3],f32>
  %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,4,3],f32>) -> !torch.vtensor<[1,2,3,3],f32>
  return %0 : !torch.vtensor<[1,2,3,3],f32>
}

// -----

// CHECK-LABEL: @test_matmulinteger
func.func @test_matmulinteger(%arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],ui8>, %arg2: !torch.vtensor<[1],ui8>, %arg3: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,2],si32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],ui8>, !torch.vtensor<[1],ui8>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,2],si32>
  // CHECK: %[[LITEM:.+]] = torch.aten.item %arg2
  // CHECK: %[[RITEM:.+]] = torch.aten.item %arg3
  // CHECK: %[[SCALE:.+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[LMAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[LITEM]] : !torch.vtensor<[4,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[4,3],!torch.quint8>
  // CHECK: %[[RMAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[SCALE]], %[[RITEM]] : !torch.vtensor<[3,2],ui8>, !torch.float, !torch.int -> !torch.vtensor<[3,2],!torch.quint8>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[LMAKE]], %[[RMAKE]]
  // CHECK: return %[[MM]]
  return %0 : !torch.vtensor<[4,2],si32>
}

// -----

// CHECK-LABEL: @test_matmulinteger_batched
func.func @test_matmulinteger_batched(%arg0: !torch.vtensor<[7,4,3],ui8>, %arg1: !torch.vtensor<[3,2],ui8>, %arg2: !torch.vtensor<[1],ui8>, %arg3: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[7,4,2],si32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[7,4,3],ui8>, !torch.vtensor<[3,2],ui8>, !torch.vtensor<[1],ui8>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[7,4,2],si32>
  // CHECK: %[[LITEM:.+]] = torch.aten.item %arg2
  // CHECK: %[[RITEM:.+]] = torch.aten.item %arg3
  // CHECK: %[[SCALE:.+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[LMAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[SCALE]], %[[LITEM]] : !torch.vtensor<[7,4,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[7,4,3],!torch.quint8>
  // CHECK: %[[RMAKE:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[SCALE]], %[[RITEM]] : !torch.vtensor<[3,2],ui8>, !torch.float, !torch.int -> !torch.vtensor<[3,2],!torch.quint8>
  // CHECK: %[[MM:.+]] = torch.aten.matmul %[[LMAKE]], %[[RMAKE]]
  // CHECK: return %[[MM]]
  return %0 : !torch.vtensor<[7,4,2],si32>
}
// -----

// CHECK-LABEL: func.func @test_mul
  func.func @test_mul(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.Mul"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL:   func.func @test_multinomial_default
func.func @test_multinomial_default(%arg0: !torch.vtensor<[3,5],f64>) -> !torch.vtensor<[3, 1],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK:           %[[VAL_1:.*]] = torch.constant.int 3
    // CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
    // CHECK:           %[[VAL_3:.*]] = torch.constant.none
    // CHECK:           %[[VAL_4:.*]] = torch.constant.bool true
    // CHECK:           %[[VAL_5:.*]] = torch.aten.multinomial %arg0, %[[VAL_2]], %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[3,5],f64>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],si64>
    // CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
    // CHECK:           %[[VAL_7:.*]] = torch.aten.to.dtype %[[VAL_5]], %[[VAL_1]], %[[VAL_6]], %[[VAL_6]], %[[VAL_3]] : !torch.vtensor<[3,1],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,1],si32>
    // CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,1],si32>
    %0 = torch.operator "onnx.Multinomial"(%arg0) : (!torch.vtensor<[3,5],f64>) -> !torch.vtensor<[3,1],si32>
    return %0 : !torch.vtensor<[3,1],si32>
}

// CHECK-LABEL:   func.func @test_multinomial_dtype_double_samplenum_4
func.func @test_multinomial_dtype_double_samplenum_4(%arg0: !torch.vtensor<[3,5],f64>) -> !torch.vtensor<[3, 4],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK:           %[[VAL_1:.*]] = torch.constant.int 7
    // CHECK:           %[[VAL_2:.*]] = torch.constant.int 4
    // CHECK:           %[[VAL_3:.*]] = torch.constant.none
    // CHECK:           %[[VAL_4:.*]] = torch.constant.bool true
    // CHECK:           %[[VAL_5:.*]] = torch.aten.multinomial %arg0, %[[VAL_2]], %[[VAL_4]], %[[VAL_3]] : !torch.vtensor<[3,5],f64>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,4],si64>
    // CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
    // CHECK:           %[[VAL_7:.*]] = torch.aten.to.dtype %[[VAL_5]], %[[VAL_1]], %[[VAL_6]], %[[VAL_6]], %[[VAL_3]] : !torch.vtensor<[3,4],si64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f64>
    // CHECK:           return %[[VAL_7]] : !torch.vtensor<[3,4],f64>
    %0 = torch.operator "onnx.Multinomial"(%arg0) {torch.onnx.dtype = 11 : si64, torch.onnx.sample_size = 4 : si64} : (!torch.vtensor<[3,5],f64>) -> !torch.vtensor<[3,4],f64>
    return %0 : !torch.vtensor<[3,4],f64>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_2d_default
func.func @test_maxpool_2d_default(%arg0: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,31,31],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[I2_1:.*]] = torch.constant.int 2
  // CHECK: %[[LIST22:.*]] = torch.prim.ListConstruct %[[I2]], %[[I2_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I0_0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[LIST0:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_0:.*]] = torch.constant.int 1
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[LIST1:.*]] = torch.prim.ListConstruct %[[I1_0]], %[[I1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_2:.*]] = torch.constant.int 1
  // CHECK: %[[I1_3:.*]] = torch.constant.int 1
  // CHECK: %[[LIST3:.*]] = torch.prim.ListConstruct %[[I1_2]], %[[I1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.max_pool2d %arg0, %[[LIST22]], %[[LIST1]], %[[LIST0]], %[[LIST3]], %[[FALSE]] : !torch.vtensor<[1,3,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,31,31],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,31,31],f32>
  return %0 : !torch.vtensor<[1,3,31,31],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_2d_ceil
func.func @test_maxpool_2d_ceil(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[I3:.*]] = torch.constant.int 3
  // CHECK: %[[I3_1:.*]] = torch.constant.int 3
  // CHECK: %[[LIST33:.*]] = torch.prim.ListConstruct %[[I3]], %[[I3_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I0_0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[LIST0:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[I2_1:.*]] = torch.constant.int 2
  // CHECK: %[[LIST22:.*]] = torch.prim.ListConstruct %[[I2]], %[[I2_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_0:.*]] = torch.constant.int 1
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[I1_0]], %[[I1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: torch.aten.max_pool2d %arg0, %[[LIST33]], %[[LIST22]], %[[LIST0]], %[[LIST]], %[[TRUE]] : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,2,2],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 1 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32>
  return %0 : !torch.vtensor<[1,1,2,2],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_3d_default
func.func @test_maxpool_3d_default(%arg0: !torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[I2:.*]] = torch.constant.int 2
  // CHECK: %[[I2_1:.*]] = torch.constant.int 2
  // CHECK: %[[I2_2:.*]] = torch.constant.int 2
  // CHECK: %[[LIST222:.*]] = torch.prim.ListConstruct %[[I2]], %[[I2_1]], %[[I2_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I0_0:.*]] = torch.constant.int 0
  // CHECK: %[[I0_1:.*]] = torch.constant.int 0
  // CHECK: %[[I0_2:.*]] = torch.constant.int 0
  // CHECK: %[[LIST0:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]], %[[I0_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_0:.*]] = torch.constant.int 1
  // CHECK: %[[I1_1:.*]] = torch.constant.int 1
  // CHECK: %[[I1_2:.*]] = torch.constant.int 1
  // CHECK: %[[LIST1:.*]] = torch.prim.ListConstruct %[[I1_0]], %[[I1_1]], %[[I1_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[I1_3:.*]] = torch.constant.int 1
  // CHECK: %[[I1_4:.*]] = torch.constant.int 1
  // CHECK: %[[I1_5:.*]] = torch.constant.int 1
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[I1_3]], %[[I1_4]], %[[I1_5]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.max_pool3d %arg0, %[[LIST222]], %[[LIST1]], %[[LIST0]], %[[LIST]], %[[FALSE]] : !torch.vtensor<[1,3,32,32,32],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,31,31,31],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32>
  return %0 : !torch.vtensor<[1,3,31,31,31],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_pad
func.func @test_maxpool_pad(%arg0: !torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,56,56],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.+]] = torch.constant.int 2
  // CHECK: %[[INT2_0:.+]] = torch.constant.int 1
  // CHECK: %[[INT2_1:.+]] = torch.constant.int 2
  // CHECK: %[[PADI:.+]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]], %[[INT2_0]], %[[INT2_1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[MIN:.+]] = torch.constant.float -1.7976931348623157E+308
  // CHECK: %[[PADDED:.+]] = torch.aten.constant_pad_nd %arg0, %[[PADI]], %[[MIN]] : !torch.vtensor<[1,64,111,111],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,64,114,114],f32>
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT3_0:.*]] = torch.constant.int 3
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: %[[LIST2:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT0_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[INT2_4:.*]] = torch.constant.int 2
  // CHECK: %[[LIST3:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2_4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK: %[[EMPTY_LIST:.*]] = torch.prim.ListConstruct %[[INT1_0]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[OUT:.*]] = torch.aten.max_pool2d %[[PADDED]], %[[LIST]], %[[LIST3]], %[[LIST2]], %[[EMPTY_LIST]], %[[FALSE]] : !torch.vtensor<[1,64,114,114],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
  // CHECK: return %[[OUT]] : !torch.vtensor<[1,64,56,56],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 2 : si64, 2 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,64,111,111],f32>) -> !torch.vtensor<[1,64,56,56],f32>
  return %0 : !torch.vtensor<[1,64,56,56],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_2d_same_lower
func.func @test_maxpool_2d_same_lower(%arg0: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
    // CHECK: %[[int1:.*]] = torch.constant.int 1
    // CHECK: %[[int0:.*]] = torch.constant.int 0
    // CHECK: %[[int1_0:.*]] = torch.constant.int 1
    // CHECK: %[[int0_1:.*]] = torch.constant.int 0
    // CHECK: %[[list0:.*]] = torch.prim.ListConstruct %[[int1]], %[[int0]], %[[int1_0]], %[[int0_1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[FLOAT0:.*]] = torch.constant.float -1.7976931348623157E+308
    // CHECK: %[[FUNC1:.*]] = torch.aten.constant_pad_nd %arg0, %[[list0]], %[[FLOAT0]] : !torch.vtensor<[1,3,32,32],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,3,33,33],f32>
    // CHECK: %[[int2:.*]] = torch.constant.int 2
    // CHECK: %[[int2_2:.*]] = torch.constant.int 2
    // CHECK: %[[list1:.*]] = torch.prim.ListConstruct %[[int2]], %[[int2_2]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[int0_3:.*]] = torch.constant.int 0
    // CHECK: %[[int0_4:.*]] = torch.constant.int 0
    // CHECK: %[[list2:.*]] = torch.prim.ListConstruct %[[int0_3]], %[[int0_4]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[int1_5:.*]] = torch.constant.int 1
    // CHECK: %[[int1_6:.*]] = torch.constant.int 1
    // CHECK: %[[list3:.*]] = torch.prim.ListConstruct %[[int1_5]], %[[int1_6]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[int1_7:.*]] = torch.constant.int 1
    // CHECK: %[[int1_8:.*]] = torch.constant.int 1
    // CHECK: %[[list4:.*]] = torch.prim.ListConstruct %[[int1_7]], %[[int1_8]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[FALSE:.*]] = torch.constant.bool false
    // CHECK: %[[FUNC6:.*]] = torch.aten.max_pool2d %[[FUNC1]], %[[list1]], %[[list3]], %[[list2]], %[[list4]], %[[FALSE]] : !torch.vtensor<[1,3,33,33],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,32,32],f32>
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.auto_pad = "SAME_LOWER", torch.onnx.kernel_shape = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32>
    return %0 : !torch.vtensor<[1,3,32,32],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_2d_same_upper
func.func @test_maxpool_2d_same_upper(%arg0: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[int0:.*]] = torch.constant.int 0
    // CHECK: %[[int1:.*]] = torch.constant.int 1
    // CHECK: %[[int0_0:.*]] = torch.constant.int 0
    // CHECK: %[[int1_1:.*]] = torch.constant.int 1
    // CHECK: %[[list0:.*]] = torch.prim.ListConstruct %[[int0]], %[[int1]], %[[int0_0]], %[[int1_1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[FLOAT0:.*]] = torch.constant.float -1.7976931348623157E+308
    // CHECK: %[[FUNC1:.*]] = torch.aten.constant_pad_nd %arg0, %[[list0]], %[[FLOAT0]] : !torch.vtensor<[1,3,32,32],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,3,33,33],f32>
    // CHECK: %[[int2:.*]] = torch.constant.int 2
    // CHECK: %[[int2_2:.*]] = torch.constant.int 2
    // CHECK: %[[list1:.*]] = torch.prim.ListConstruct %[[int2]], %[[int2_2]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[int0_3:.*]] = torch.constant.int 0
    // CHECK: %[[int0_4:.*]] = torch.constant.int 0
    // CHECK: %[[list2:.*]] = torch.prim.ListConstruct %[[int0_3]], %[[int0_4]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[int1_5:.*]] = torch.constant.int 1
    // CHECK: %[[int1_6:.*]] = torch.constant.int 1
    // CHECK: %[[list3:.*]] = torch.prim.ListConstruct %[[int1_5]], %[[int1_6]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[int1_7:.*]] = torch.constant.int 1
    // CHECK: %[[int1_8:.*]] = torch.constant.int 1
    // CHECK: %[[list4:.*]] = torch.prim.ListConstruct %[[int1_7]], %[[int1_8]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[FALSE:.*]] = torch.constant.bool false
    // CHECK: %[[FUNC6:.*]] = torch.aten.max_pool2d %[[FUNC1]], %[[list1]], %[[list3]], %[[list2]], %[[list4]], %[[FALSE]] : !torch.vtensor<[1,3,33,33],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,32,32],f32>
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.auto_pad = "SAME_UPPER", torch.onnx.kernel_shape = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32>
    return %0 : !torch.vtensor<[1,3,32,32],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxpool_2d_precomputed_same_upper
func.func @test_maxpool_2d_precomputed_same_upper(%arg0: !torch.vtensor<[1,1,5,5],f32>) -> !torch.vtensor<[1,1,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64}{
	// CHECK: %[[int3:.*]] = torch.constant.int 3
	// CHECK: %[[int3_0:.*]] = torch.constant.int 3
	// CHECK: %[[list0:.*]] = torch.prim.ListConstruct %[[int3]], %[[int3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
	// CHECK: %[[int1:.*]] = torch.constant.int 1
	// CHECK: %[[int1_1:.*]] = torch.constant.int 1
	// CHECK: %[[list1:.*]] = torch.prim.ListConstruct %[[int1]], %[[int1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
	// CHECK: %[[int2:.*]] = torch.constant.int 2
	// CHECK: %[[int2_2:.*]] = torch.constant.int 2
	// CHECK: %[[list2:.*]] = torch.prim.ListConstruct %[[int2]], %[[int2_2]] : (!torch.int, !torch.int) -> !torch.list<int>
	// CHECK: %[[int1_3:.*]] = torch.constant.int 1
	// CHECK: %[[int1_4:.*]] = torch.constant.int 1
	// CHECK: %[[list3:.*]] = torch.prim.ListConstruct %[[int1_3]], %[[int1_4]] : (!torch.int, !torch.int) -> !torch.list<int>
	// CHECK: %[[FALSE:.*]] = torch.constant.bool false
	// CHECK: %[[FUNC4:.*]] = torch.aten.max_pool2d %arg0, %[[list0]], %[[list2]], %[[list1]], %[[list3]], %[[FALSE]] : !torch.vtensor<[1,1,5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,3,3],f32>
%0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.auto_pad = "SAME_UPPER", torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,5,5],f32>) -> !torch.vtensor<[1,1,3,3],f32>
return %0 : !torch.vtensor<[1,1,3,3],f32>
}


// -----

// CHECK-LABEL: func.func @test_maxpool_symmetric_pad
func.func @test_maxpool_symmetric_pad(%arg0: !torch.vtensor<[1,64,112,112],f32>) -> !torch.vtensor<[1,64,56,56],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64} {
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT3_0:.*]] = torch.constant.int 3
  // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_1:.*]] = torch.constant.int 1
  // CHECK: %[[LIST2:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[INT2_4:.*]] = torch.constant.int 2
  // CHECK: %[[LIST3:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2_4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT1_2:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_3:.*]] = torch.constant.int 1
  // CHECK: %[[DILATION:.*]] = torch.prim.ListConstruct %[[INT1_2]], %[[INT1_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[OUT:.*]] = torch.aten.max_pool2d %arg0, %[[LIST]], %[[LIST3]], %[[LIST2]], %[[DILATION]], %[[FALSE]] : !torch.vtensor<[1,64,112,112],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
  // CHECK: return %[[OUT]] : !torch.vtensor<[1,64,56,56],f32>
  %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,64,112,112],f32>) -> !torch.vtensor<[1,64,56,56],f32>
  return %0 : !torch.vtensor<[1,64,56,56],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxroipool
func.func @test_maxroipool(%arg0: !torch.vtensor<[8,3,32,32],f32>, %arg1: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,3,2,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[INT2_0:.*]] = torch.constant.int 2
  // CHECK: %[[LIST0:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[INT2_1:.*]] = torch.constant.int 2
  // CHECK: %[[INT3:.*]] = torch.constant.int 3
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[INT5:.*]] = torch.constant.int 5
  // CHECK: %[[INT2_2:.*]] = torch.constant.int 2
  // CHECK: %[[INT1_3:.*]] = torch.constant.int 1
  // CHECK: %[[SELECT1:.*]] = torch.aten.select.int %arg1, %[[INT1]], %[[INT0]] : !torch.vtensor<[2,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
  // CHECK: %[[CAST1:.*]] = torch.aten._cast_Long %[[SELECT1]], %[[TRUE]] : !torch.vtensor<[?],f32>, !torch.bool -> !torch.vtensor<[?],si64>
  // CHECK: %[[SLICE1:.*]] = torch.aten.slice.Tensor %arg1, %[[INT1]], %[[INT1]], %[[INT5]], %[[INT1]] : !torch.vtensor<[2,5],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
  // CHECK: %[[MUL1:.*]] = torch.aten.mul.Scalar %[[SLICE1]], %[[FLOAT1]] : !torch.vtensor<[?,4],f32>, !torch.float -> !torch.vtensor<[?,4],f32>
  // CHECK: %[[CAST2:.*]] = torch.aten._cast_Long %[[MUL1]], %[[TRUE]] : !torch.vtensor<[?,4],f32>, !torch.bool -> !torch.vtensor<[?,4],si64>
  // CHECK: %[[INT0_4:.*]] = torch.constant.int 0
  // CHECK: %[[SELECT2:.*]] = torch.aten.select.int %[[CAST2]], %[[INT0]], %[[INT0_4]] : !torch.vtensor<[?,4],si64>, !torch.int, !torch.int -> !torch.vtensor<[4],si64>
  // CHECK: %[[SELECT3:.*]] = torch.aten.select.int %[[CAST1]], %[[INT0]], %[[INT0_4]] : !torch.vtensor<[?],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM1:.*]] = torch.aten.item %[[SELECT3]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT4:.*]] = torch.aten.select.int %[[SELECT2]], %[[INT0]], %[[INT0]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM2:.*]] = torch.aten.item %[[SELECT4]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT5:.*]] = torch.aten.select.int %[[SELECT2]], %[[INT0]], %[[INT1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM3:.*]] = torch.aten.item %[[SELECT5]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT6:.*]] = torch.aten.select.int %[[SELECT2]], %[[INT0]], %[[INT2_1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM4:.*]] = torch.aten.item %[[SELECT6]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT7:.*]] = torch.aten.select.int %[[SELECT2]], %[[INT0]], %[[INT3]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM5:.*]] = torch.aten.item %[[SELECT7]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[ADD1:.*]] = torch.aten.add %[[ITEM4]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD2:.*]] = torch.aten.add %[[ITEM5]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SELECT8:.*]] = torch.aten.select.int %arg0, %[[INT0]], %[[ITEM1]] : !torch.vtensor<[8,3,32,32],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,32,32],f32>
  // CHECK: %[[SLICE2:.*]] = torch.aten.slice.Tensor %[[SELECT8]], %[[INT1_3]], %[[ITEM3]], %[[ADD2]], %[[INT1]] : !torch.vtensor<[3,32,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?,?],f32>
  // CHECK: %[[SLICE3:.*]] = torch.aten.slice.Tensor %[[SLICE2]], %[[INT2_2]], %[[ITEM2]], %[[ADD1]], %[[INT1]] : !torch.vtensor<[3,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?,?],f32>
  // CHECK: %[[RESULT0:.*]], %[[RESULT1:.*]] = torch.aten.adaptive_max_pool2d %[[SLICE3]], %[[LIST0]] : !torch.vtensor<[3,?,?],f32>, !torch.list<int> -> !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],si64>
  // CHECK: %[[INT1_5:.*]] = torch.constant.int 1
  // CHECK: %[[SELECT9:.*]] = torch.aten.select.int %[[CAST2]], %[[INT0]], %[[INT1_5]] : !torch.vtensor<[?,4],si64>, !torch.int, !torch.int -> !torch.vtensor<[4],si64>
  // CHECK: %[[SELECT10:.*]] = torch.aten.select.int %[[CAST1]], %[[INT0]], %[[INT1_5]] : !torch.vtensor<[?],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM6:.*]] = torch.aten.item %[[SELECT10]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT11:.*]] = torch.aten.select.int %[[SELECT9]], %[[INT0]], %[[INT0]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM7:.*]] = torch.aten.item %[[SELECT11]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT12:.*]] = torch.aten.select.int %[[SELECT9]], %[[INT0]], %[[INT1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM8:.*]] = torch.aten.item %[[SELECT12]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT13:.*]] = torch.aten.select.int %[[SELECT9]], %[[INT0]], %[[INT2_1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM9:.*]] = torch.aten.item %[[SELECT13]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[SELECT14:.*]] = torch.aten.select.int %[[SELECT9]], %[[INT0]], %[[INT3]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM10:.*]] = torch.aten.item %[[SELECT14]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[ADD3:.*]] = torch.aten.add %[[ITEM9]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[ADD4:.*]] = torch.aten.add %[[ITEM10]], %[[INT1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[SELECT15:.*]] = torch.aten.select.int %arg0, %[[INT0]], %[[ITEM6]] : !torch.vtensor<[8,3,32,32],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,32,32],f32>
  // CHECK: %[[SLICE4:.*]] = torch.aten.slice.Tensor %[[SELECT15]], %[[INT1_3]], %[[ITEM8]], %[[ADD4]], %[[INT1]] : !torch.vtensor<[3,32,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?,?],f32>
  // CHECK: %[[SLICE5:.*]] = torch.aten.slice.Tensor %[[SLICE4]], %[[INT2_2]], %[[ITEM7]], %[[ADD3]], %[[INT1]] : !torch.vtensor<[3,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[3,?,?],f32>
  // CHECK: %[[RESULT0_6:.*]], %[[RESULT1_7:.*]] = torch.aten.adaptive_max_pool2d %[[SLICE5]], %[[LIST0]] : !torch.vtensor<[3,?,?],f32>, !torch.list<int> -> !torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],si64>
  // CHECK: %[[LIST1:.*]] = torch.prim.ListConstruct %[[RESULT0]], %[[RESULT0_6]] : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32>) -> !torch.list<vtensor<[3,2,2],f32>>
  // CHECK: %[[STACK:.*]] = torch.aten.stack %[[LIST1]], %[[INT0]] : !torch.list<vtensor<[3,2,2],f32>>, !torch.int -> !torch.vtensor<[2,3,2,2],f32>
  // CHECK: return %[[STACK]] : !torch.vtensor<[2,3,2,2],f32>
  %0 = torch.operator "onnx.MaxRoiPool"(%arg0, %arg1) {torch.onnx.pooled_shape = [2 : si64, 2 : si64], torch.onnx.spatial_scale = 1.000000e+00 : f32} : (!torch.vtensor<[8,3,32,32],f32>, !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,3,2,2],f32>
  return %0 : !torch.vtensor<[2,3,2,2],f32>
}

// -----

// CHECK-LABEL: @test_gelu_default_1
func.func @test_gelu_default_1(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "none"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3],f32>, !torch.str -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_gelu_default_2
func.func @test_gelu_default_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "none"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3,4,5],f32>, !torch.str -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_gelu_tanh_1
func.func @test_gelu_tanh_1(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "tanh"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3],f32>, !torch.str -> !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) {torch.onnx.approximate = "tanh"} : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_gelu_tanh_2
func.func @test_gelu_tanh_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[STR1:.*]] = torch.constant.str "tanh"
  // CHECK: torch.aten.gelu %arg0, %[[STR1]] : !torch.vtensor<[3,4,5],f32>, !torch.str -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Gelu"(%arg0) {torch.onnx.approximate = "tanh"} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_grid_sampler01
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT0_0:.*]] = torch.constant.int 0
// CHECK: %[[B0:.*]] = torch.constant.bool false
// CHECK: %[[A0:.*]] = torch.aten.grid_sampler %arg0, %arg1, %[[INT0]], %[[INT0_0]], %[[B0]] : !torch.vtensor<[5,10,10,4],f32>
func.func @test_grid_sampler01(%arg0: !torch.vtensor<[5,10,10,4],f32>, %arg1: !torch.vtensor<[5,7,8,2],f32>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.GridSample" (%arg0, %arg1) {torch.onnx.align_corners = 0 : si64, torch.onnx.mode = "linear", torch.onnx.padding_mode = "zeros"} : (!torch.vtensor<[5,10,10,4],f32>, !torch.vtensor<[5,7,8,2],f32>) -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: @test_grid_sampler02
// CHECK: %[[INT0:.*]] = torch.constant.int 0
// CHECK: %[[INT0_0:.*]] = torch.constant.int 0
// CHECK: %[[B0:.*]] = torch.constant.bool true
// CHECK: %[[A0:.*]] = torch.aten.grid_sampler %arg0, %arg1, %[[INT0]], %[[INT0_0]], %[[B0]] : !torch.vtensor<[5,10,10,4],f32>
func.func @test_grid_sampler02(%arg0: !torch.vtensor<[5,10,10,4],f32>, %arg1: !torch.vtensor<[5,7,8,2],f32>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.GridSample" (%arg0, %arg1) {torch.onnx.align_corners = 1 : si64, torch.onnx.padding_mode = "zeros"} : (!torch.vtensor<[5,10,10,4],f32>, !torch.vtensor<[5,7,8,2],f32>) -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: func.func @test_oldest_pad
func.func @test_oldest_pad(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 1 : si64} {
  // CHECK: %[[int0:.*]] = torch.constant.int 0
  // CHECK: %[[int0_0:.*]] = torch.constant.int 0
  // CHECK: %[[int2:.*]] = torch.constant.int 2
  // CHECK: %[[int0_1:.*]] = torch.constant.int 0
  // CHECK: %[[float0:.*]] = torch.constant.float 0.000000e+00
  // CHECK: %[[list:.*]] = torch.prim.ListConstruct %[[int0_0]], %[[int0_1]], %[[int0]], %[[int2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[pad:.*]] = torch.aten.constant_pad_nd %arg0, %[[list]], %[[float0]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[5,4],f32>
  // CHECK: return %[[pad]] : !torch.vtensor<[5,4],f32>
  %0 = torch.operator "onnx.Pad"(%arg0) {torch.onnx.mode = "constant", torch.onnx.paddings = [0 : si64, 0 : si64, 2 : si64, 0 : si64], torch.onnx.value = 0.000000e+00 : f32} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_old_pad
func.func @test_old_pad(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 11 : si64} {
  // CHECK: %[[int0:.*]] = torch.constant.int 0
  // CHECK: %[[int0_0:.*]] = torch.constant.int 0
  // CHECK: %[[int2:.*]] = torch.constant.int 2
  // CHECK: %[[int0_1:.*]] = torch.constant.int 0
  // CHECK: %[[float0:.*]] = torch.constant.float 0.000000e+00
  // CHECK: %[[list:.*]] = torch.prim.ListConstruct %[[int0_0]], %[[int0_1]], %[[int0]], %[[int2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[pad:.*]] = torch.aten.constant_pad_nd %arg0, %[[list]], %[[float0]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[5,4],f32>
  // CHECK: return %[[pad]] : !torch.vtensor<[5,4],f32>
  %0 = torch.operator "onnx.Pad"(%arg0) {torch.onnx.mode = "constant", torch.onnx.pads = [0 : si64, 0 : si64, 2 : si64, 0 : si64], torch.onnx.value = 0.000000e+00 : f32} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_pad
func.func @test_pad(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4], si64>, %arg2: !torch.vtensor<[], f32>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT_0:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_0:.+]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[SELECT_1:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_1:.+]] = torch.aten.item %[[SELECT_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[SELECT_2:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT2]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_2:.+]] = torch.aten.item %[[SELECT_2]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[SELECT_3:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT3]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_3:.+]] = torch.aten.item %[[SELECT_3]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[VAL:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[ITEM_1]], %[[ITEM_3]], %[[ITEM_0]], %[[ITEM_2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PAD:.+]] = torch.aten.constant_pad_nd %arg0, %[[LIST]], %[[VAL]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[5,4],f32>
  // CHECK: return %[[PAD]] : !torch.vtensor<[5,4],f32>
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1, %arg2) {torch.onnx.mode = "constant"} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4], si64>, !torch.vtensor<[], f32>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_i32pad
func.func @test_i32pad(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[4], si64>, %arg2: !torch.vtensor<[], si32>) -> !torch.vtensor<[5,4],si32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: %[[SELECT_0:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT0_0]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_0:.+]] = torch.aten.item %[[SELECT_0]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[SELECT_1:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT1]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_1:.+]] = torch.aten.item %[[SELECT_1]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[SELECT_2:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT2]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_2:.+]] = torch.aten.item %[[SELECT_2]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[SELECT_3:.+]] = torch.aten.select.int %arg1, %[[INT0]], %[[INT3]] : !torch.vtensor<[4],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK: %[[ITEM_3:.+]] = torch.aten.item %[[SELECT_3]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK: %[[VAL:.+]] = torch.aten.item %arg2 : !torch.vtensor<[],si32> -> !torch.int
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[ITEM_1]], %[[ITEM_3]], %[[ITEM_0]], %[[ITEM_2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PAD:.+]] = torch.aten.constant_pad_nd %arg0, %[[LIST]], %[[VAL]] : !torch.vtensor<[3,4],si32>, !torch.list<int>, !torch.int -> !torch.vtensor<[5,4],si32>
  // CHECK: return %[[PAD]] : !torch.vtensor<[5,4],si32>
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1, %arg2) {torch.onnx.mode = "constant"} : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[4], si64>, !torch.vtensor<[], si32>) -> !torch.vtensor<[5,4],si32>
  return %0 : !torch.vtensor<[5,4],si32>
}

// -----

// CHECK-LABEL: @test_pad_optional_constant
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>
// CHECK-SAME:    %[[ARG1:.*]]: !torch.vtensor<[4],si64>
// CHECK: %[[VAL:.+]] = torch.constant.float 0
// CHECK: torch.aten.constant_pad_nd %[[ARG0]], %{{.*}}, %[[VAL]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[5,4],f32>

func.func @test_pad_optional_constant(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1) {torch.onnx.mode = "constant"} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

// CHECK-LABEL: @test_pad_wrap
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>
// CHECK-SAME:    %[[ARG1:.*]]: !torch.vtensor<[4],si64>
// CHECK: %[[VAL:.+]] = torch.constant.none
// CHECK: %[[STR:.+]] = torch.constant.str "circular"
// CHECK: torch.aten.pad %[[ARG0]], %{{.*}}, %[[STR]], %[[VAL]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.str, !torch.none -> !torch.vtensor<[5,4],f32>

func.func @test_pad_wrap(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1) {torch.onnx.mode = "wrap"} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

// CHECK-LABEL: @test_pad_edge
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>
// CHECK-SAME:    %[[ARG1:.*]]: !torch.vtensor<[4],si64>
// CHECK: %[[VAL:.+]] = torch.constant.none
// CHECK: %[[STR:.+]] = torch.constant.str "replicate"
// CHECK: torch.aten.pad %[[ARG0]], %{{.*}}, %[[STR]], %[[VAL]] : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.str, !torch.none -> !torch.vtensor<[5,4],f32>

func.func @test_pad_edge(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1) {torch.onnx.mode = "edge"} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[4], si64>) -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}

// -----

func.func @test_center_crop_pad_crop_axes_chw_expanded(%arg0: !torch.vtensor<[4,5],f32>, %arg1: !torch.vtensor<[4],si64>, %arg2: !torch.vtensor<[2],si64>) -> !torch.vtensor<[?,?],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64} {

  // CHECK:  %[[NONE:.+]] = torch.constant.none
  // CHECK:  %[[ZERO:.+]] = torch.constant.int 0

  // CHECK:  %[[IDX:.+]] = torch.constant.int 0
  // CHECK:  %[[SEL:.+]] = torch.aten.select.int %arg1, %[[ZERO]], %[[IDX]]
  // CHECK:  %[[PAD0:.+]] = torch.aten.item %[[SEL]]

  // CHECK:  %[[IDX:.+]] = torch.constant.int 1
  // CHECK:  %[[SEL:.+]] = torch.aten.select.int %arg1, %[[ZERO]], %[[IDX]]
  // CHECK:  %[[PAD1:.+]] = torch.aten.item %[[SEL]]

  // CHECK:  %[[IDX:.+]] = torch.constant.int 2
  // CHECK:  %[[SEL:.+]] = torch.aten.select.int %arg1, %[[ZERO]], %[[IDX]]
  // CHECK:  %[[PAD2:.+]] = torch.aten.item %[[SEL]]

  // CHECK:  %[[IDX:.+]] = torch.constant.int 3
  // CHECK:  %[[SEL:.+]] = torch.aten.select.int %arg1, %[[ZERO]], %[[IDX]]
  // CHECK:  %[[PAD3:.+]] = torch.aten.item %[[SEL]]

  // CHECK:  %[[ZERO:.+]] = torch.constant.int 0
  // CHECK:  %[[RANK:.+]] = torch.constant.int 2

  // CHECK:  %[[IDX:.+]] = torch.constant.int 0
  // CHECK:  %[[SEL:.+]] = torch.aten.select.int %arg2, %[[ZERO]], %[[IDX]]
  // CHECK:  %[[ITEM:.+]] = torch.aten.item %[[SEL]]
  // CHECK:  %[[LT:.+]] = torch.aten.lt.int %[[ITEM]], %[[ZERO]]
  // CHECK:  %[[INT:.+]] = torch.aten.Int.bool %[[LT]]
  // CHECK:  %[[MUL:.+]] = torch.aten.mul.int %[[INT]], %[[RANK]]
  // CHECK:  %[[AXIS0:.+]] = torch.aten.add.int %[[MUL]], %[[ITEM]]

  // CHECK:  %[[IDX:.+]] = torch.constant.int 1
  // CHECK:  %[[SEL:.+]] = torch.aten.select.int %arg2, %[[ZERO]], %[[IDX]]
  // CHECK:  %[[ITEM:.+]] = torch.aten.item %[[SEL]]
  // CHECK:  %[[LT:.+]] = torch.aten.lt.int %[[ITEM]], %[[ZERO]]
  // CHECK:  %[[INT:.+]] = torch.aten.Int.bool %[[LT]]
  // CHECK:  %[[MUL:.+]] = torch.aten.mul.int %[[INT]], %[[RANK]]
  // CHECK:  %[[AXIS1:.+]] = torch.aten.add.int %[[MUL]], %[[ITEM]]


  // CHECK:  %[[AX:.+]] = torch.constant.int 0
  // CHECK:  %[[EQ:.+]] = torch.aten.eq.int %[[AXIS0]], %[[AX]]
  // CHECK:  %[[INT:.+]] = torch.aten.Int.bool %[[EQ]]
  // CHECK:  %[[MUL0:.+]] = torch.aten.mul.int %[[INT]], %[[PAD0]]
  // CHECK:  %[[MUL1:.+]] = torch.aten.mul.int %[[INT]], %[[PAD2]]
  // CHECK:  %[[ADD0:.+]] = torch.aten.add.int %[[ZERO]], %[[MUL0]]
  // CHECK:  %[[ADD1:.+]] = torch.aten.add.int %[[ZERO]], %[[MUL1]]

  // CHECK:  %[[EQ:.+]] = torch.aten.eq.int %[[AXIS1]], %[[AX]]
  // CHECK:  %[[INT:.+]] = torch.aten.Int.bool %[[EQ]]
  // CHECK:  %[[MUL0:.+]] = torch.aten.mul.int %[[INT]], %[[PAD1]]
  // CHECK:  %[[MUL1:.+]] = torch.aten.mul.int %[[INT]], %[[PAD3]]
  // CHECK:  %[[BEGIN0:.+]] = torch.aten.add.int %[[ADD0]], %[[MUL0]]
  // CHECK:  %[[END0:.+]] = torch.aten.add.int %[[ADD1]], %[[MUL1]]

  // CHECK:  %[[AX:.+]] = torch.constant.int 1
  // CHECK:  %[[EQ:.+]] = torch.aten.eq.int %[[AXIS0]], %[[AX]]
  // CHECK:  %[[INT:.+]] = torch.aten.Int.bool %[[EQ]]
  // CHECK:  %[[MUL0:.+]] = torch.aten.mul.int %[[INT]], %[[PAD0]]
  // CHECK:  %[[MUL1:.+]] = torch.aten.mul.int %[[INT]], %[[PAD2]]
  // CHECK:  %[[ADD0:.+]] = torch.aten.add.int %[[ZERO]], %[[MUL0]]
  // CHECK:  %[[ADD1:.+]] = torch.aten.add.int %[[ZERO]], %[[MUL1]]

  // CHECK:  %[[EQ:.+]] = torch.aten.eq.int %[[AXIS1]], %[[AX]]
  // CHECK:  %[[INT:.+]] = torch.aten.Int.bool %[[EQ]]
  // CHECK:  %[[MUL0:.+]] = torch.aten.mul.int %[[INT]], %[[PAD1]]
  // CHECK:  %[[MUL1:.+]] = torch.aten.mul.int %[[INT]], %[[PAD3]]
  // CHECK:  %[[BEGIN1:.+]] = torch.aten.add.int %[[ADD0]], %[[MUL0]]
  // CHECK:  %[[END1:.+]] = torch.aten.add.int %[[ADD1]], %[[MUL1]]

  // CHECK:  %[[LIST:.+]] = torch.prim.ListConstruct %[[BEGIN1]], %[[END1]], %[[BEGIN0]], %[[END0]]
  // CHECK:  %[[MODE:.+]] = torch.constant.str "constant"
  // CHECK:  %[[PAD:.+]] = torch.aten.pad %arg0, %[[LIST]], %[[MODE]], %[[NONE]]
  %none = torch.constant.none
  %0 = torch.operator "onnx.Pad"(%arg0, %arg1, %none, %arg2) : (!torch.vtensor<[4,5],f32>, !torch.vtensor<[4],si64>, !torch.none, !torch.vtensor<[2],si64>) -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL: func.func @test_pow
func.func @test_pow(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.Pow"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_pow_i32
func.func @test_pow_i32(%arg0: !torch.vtensor<[3,4,5],si32>, %arg1: !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[POW:.+]] =  torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32> -> !torch.vtensor<[3,4,5],f64>
  // CHECK: %[[DTY:.+]] = torch.constant.int 3
  // CHECK: %[[RES:.+]] = torch.aten.to.dtype %[[POW]], %[[DTY]], %[[FALSE]], %[[FALSE]], %[[NONE]]
  // CHECK: return %[[RES]]
  %0 = torch.operator "onnx.Pow"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si32>, !torch.vtensor<[3,4,5],si32>) -> !torch.vtensor<[3,4,5],si32>
  return %0 : !torch.vtensor<[3,4,5],si32>
}

// -----

// CHECK-LABEL: @test_hardsigmoid_example
func.func @test_hardsigmoid_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[ALPHA_FLOAT:.*]] = torch.constant.float 5.000000e-01
  // CHECK: %[[BETA_FLOAT:.*]] = torch.constant.float 0.60000002384185791
  // CHECK: %[[ALPHA_MULTI_X:.*]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_FLOAT]] : !torch.vtensor<[3],f32>, !torch.float -> !torch.vtensor<[3],f32>
  // CHECK: %[[F1:.*]] = torch.constant.float 1.000000e+00
  // CHECK: %[[ALPHA_MULTI_X_PLUS_BETA:.*]] = torch.aten.add.Scalar %[[ALPHA_MULTI_X]], %[[BETA_FLOAT]], %[[F1]] : !torch.vtensor<[3],f32>, !torch.float, !torch.float -> !torch.vtensor<[3],f32>
  // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[NONE_FOR_ONE:.*]] = torch.constant.none
  // CHECK: %[[INT_TYPE_FOR_TENSOR_ONE:.*]] = torch.constant.int 6
  // CHECK: %[[ONE_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]], %[[F1:.*]], %[[INT_TYPE_FOR_TENSOR_ONE:.*]], %[[NONE_FOR_ONE:.*]], %[[NONE_1:.*]], %[[NONE_1:.*]] : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[MIN_EXPRESSION:.*]] = torch.aten.minimum %[[ONE_TENSOR:.*]], %[[ALPHA_MULTI_X_PLUS_BETA:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  // CHECK: %[[F0:.*]] = torch.constant.float 0.000000e+00
  // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: %[[NONE_FOR_ZERO:.*]] = torch.constant.none
  // CHECK: %[[INT_TYPE_FOR_TENSOR_ZERO:.*]] = torch.constant.int 6
  // CHECK: %[[ZERO_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]], %[[F0:.*]], %[[INT_TYPE_FOR_TENSOR_ZERO:.*]], %[[NONE_FOR_ZERO:.*]], %none_0, %none_0 : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK: %[[RESULT:.*]] = torch.aten.maximum %[[ZERO_TENSOR:.*]], %[[MIN_EXPRESSION:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  // CHECK: return %[[RESULT:.*]] : !torch.vtensor<[3],f32>
  %0 = torch.operator "onnx.HardSigmoid"(%arg0) {torch.onnx.alpha = 5.000000e-01 : f32, torch.onnx.beta = 6.000000e-01 : f32} : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: @test_hardsigmoid
func.func @test_hardsigmoid(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[ALPHA_FLOAT:.*]] = torch.constant.float 5.000000e-01
    // CHECK: %[[BETA_FLOAT:.*]] = torch.constant.float 0.60000002384185791
    // CHECK: %[[ALPHA_MULTI_X:.*]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_FLOAT]] : !torch.vtensor<[3,4,5],f32>, !torch.float -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[F1:.*]] = torch.constant.float 1.000000e+00
    // CHECK: %[[ALPHA_MULTI_X_PLUS_BETA:.*]] = torch.aten.add.Scalar %[[ALPHA_MULTI_X]], %[[BETA_FLOAT]], %[[F1]] : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ONE:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ONE:.*]] = torch.constant.int 6
    // CHECK: %[[ONE_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]], %[[F1:.*]], %[[INT_TYPE_FOR_TENSOR_ONE:.*]], %[[NONE_FOR_ONE:.*]], %[[NONE_1:.*]], %[[NONE_1:.*]] : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: %[[MIN_EXPRESSION:.*]] = torch.aten.minimum %[[ONE_TENSOR:.*]], %[[ALPHA_MULTI_X_PLUS_BETA:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[F0:.*]] = torch.constant.float 0.000000e+00
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ZERO:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ZERO:.*]] = torch.constant.int 6
    // CHECK: %[[ZERO_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]], %[[F0:.*]], %[[INT_TYPE_FOR_TENSOR_ZERO:.*]], %[[NONE_FOR_ZERO:.*]], %none_0, %none_0 : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: %[[RESULT:.*]] = torch.aten.maximum %[[ZERO_TENSOR:.*]], %[[MIN_EXPRESSION:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    // CHECK: return %[[RESULT:.*]] : !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.HardSigmoid"(%arg0) {torch.onnx.alpha = 5.000000e-01 : f32, torch.onnx.beta = 6.000000e-01 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_hardsigmoid_default
func.func @test_hardsigmoid_default(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[ALPHA_FLOAT:.*]] = torch.constant.float 0.20000000298023224
    // CHECK: %[[BETA_FLOAT:.*]] = torch.constant.float 5.000000e-01
    // CHECK: %[[ALPHA_MULTI_X:.*]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_FLOAT]] : !torch.vtensor<[3,4,5],f32>, !torch.float -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[F1:.*]] = torch.constant.float 1.000000e+00
    // CHECK: %[[ALPHA_MULTI_X_PLUS_BETA:.*]] = torch.aten.add.Scalar %[[ALPHA_MULTI_X]], %[[BETA_FLOAT]], %[[F1]] : !torch.vtensor<[3,4,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ONE:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ONE:.*]] = torch.constant.int 6
    // CHECK: %[[ONE_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ONE:.*]], %[[F1:.*]], %[[INT_TYPE_FOR_TENSOR_ONE:.*]], %[[NONE_FOR_ONE:.*]], %[[NONE_1:.*]], %[[NONE_1:.*]] : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: %[[MIN_EXPRESSION:.*]] = torch.aten.minimum %[[ONE_TENSOR:.*]], %[[ALPHA_MULTI_X_PLUS_BETA:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    // CHECK: %[[F0:.*]] = torch.constant.float 0.000000e+00
    // CHECK: %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
    // CHECK: %[[NONE_FOR_ZERO:.*]] = torch.constant.none
    // CHECK: %[[INT_TYPE_FOR_TENSOR_ZERO:.*]] = torch.constant.int 6
    // CHECK: %[[ZERO_TENSOR:.*]] = torch.aten.full %[[TENSOR_DIMENSION_LIST_FOR_ZERO:.*]], %[[F0:.*]], %[[INT_TYPE_FOR_TENSOR_ZERO:.*]], %[[NONE_FOR_ZERO:.*]], %none_0, %none_0 : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    // CHECK: torch.aten.maximum %[[ZERO_TENSOR:.*]], %[[MIN_EXPRESSION:.*]] : !torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.HardSigmoid"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @test_globalaveragepool
func.func @test_globalaveragepool(%arg0: !torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[C5_0:.*]] = torch.constant.int 5
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C5]], %[[C5_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.avg_pool2d %arg0, %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[1,3,5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,3,1,1],f32>
  %0 = torch.operator "onnx.GlobalAveragePool"(%arg0) : (!torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32>
  return %0 : !torch.vtensor<[1,3,1,1],f32>
}

// -----

// CHECK-LABEL: @test_globalaveragepool_precomputed
func.func @test_globalaveragepool_precomputed(%arg0: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C3]], %[[C3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: torch.aten.avg_pool2d %arg0, %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[1,1,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1,1],f32>
  %0 = torch.operator "onnx.GlobalAveragePool"(%arg0) : (!torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1,1],f32>
}

// -----

// CHECK-LABEL: @test_globalmaxpool
func.func @test_globalmaxpool(%arg0: !torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[C5_0:.*]] = torch.constant.int 5
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C5]], %[[C5_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[DILATION:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.max_pool2d %arg0, %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[DILATION]], %[[FALSE]] : !torch.vtensor<[1,3,5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,1,1],f32>
  %0 = torch.operator "onnx.GlobalMaxPool"(%arg0) : (!torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32>
  return %0 : !torch.vtensor<[1,3,1,1],f32>
}

// -----

// CHECK-LABEL: @test_globalmaxpool_precomputed
func.func @test_globalmaxpool_precomputed(%arg0: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C3:.*]] = torch.constant.int 3
  // CHECK: %[[C3_0:.*]] = torch.constant.int 3
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C3]], %[[C3_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[DILATION:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: torch.aten.max_pool2d %arg0, %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[DILATION]], %[[FALSE]] : !torch.vtensor<[1,1,3,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,1,1],f32>
  %0 = torch.operator "onnx.GlobalMaxPool"(%arg0) : (!torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1,1],f32>
}

// -----

// CHECK-LABEL: @test_globallppool
func.func @test_globallppool(%arg0: !torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 2 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[E1:.*]] = torch.aten.mul %[[C5]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[C5_0:.*]] = torch.constant.int 5
  // CHECK: %[[E2:.*]] = torch.aten.mul %[[C5_0]], %[[E1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C5]], %[[C5_0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]], %[[C0]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[ABS:.*]] = torch.aten.abs %arg0 : !torch.vtensor<[1,3,5,5],f32> -> !torch.vtensor<[1,3,5,5],f32>
  // CHECK: %[[CP:.*]] = torch.constant.int 2
  // CHECK: %[[POW1:.*]] = torch.aten.pow.Tensor_Scalar %[[ABS]], %[[CP]] : !torch.vtensor<[1,3,5,5],f32>, !torch.int -> !torch.vtensor<[1,3,5,5],f32>
  // CHECK: %[[AVGPOOL:.*]] = torch.aten.avg_pool2d %[[POW1]], %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[FALSE]], %[[C1]] : !torch.vtensor<[1,3,5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.int -> !torch.vtensor<[1,3,1,1],f32>
  // CHECK: %[[INVP:.*]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.pow.Tensor_Scalar %[[AVGPOOL]], %[[INVP]] : !torch.vtensor<[1,3,1,1],f32>, !torch.float -> !torch.vtensor<[1,3,1,1],f32>
  %0 = torch.operator "onnx.GlobalLpPool"(%arg0) : (!torch.vtensor<[1,3,5,5],f32>) -> !torch.vtensor<[1,3,1,1],f32>
  return %0 : !torch.vtensor<[1,3,1,1],f32>
}

// -----

// CHECK-LABEL: @test_globallppool_1d
func.func @test_globallppool_1d(%arg0: !torch.vtensor<[1,3,5],f32>) -> !torch.vtensor<[1,3,1],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 2 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]] = torch.constant.int 1
  // CHECK: %[[C5:.*]] = torch.constant.int 5
  // CHECK: %[[E1:.*]] = torch.aten.mul %[[C5]], %[[C1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[KERNELSIZE:.*]] = torch.prim.ListConstruct %[[C5]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[C0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[ABS:.*]] = torch.aten.abs %arg0 : !torch.vtensor<[1,3,5],f32> -> !torch.vtensor<[1,3,5],f32>
  // CHECK: %[[CP:.*]] = torch.constant.int 2
  // CHECK: %[[POW1:.*]] = torch.aten.pow.Tensor_Scalar %[[ABS]], %[[CP]] : !torch.vtensor<[1,3,5],f32>, !torch.int -> !torch.vtensor<[1,3,5],f32>
  // CHECK: %[[AVGPOOL:.*]] = torch.aten.avg_pool1d %[[POW1]], %[[KERNELSIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[FALSE]] : !torch.vtensor<[1,3,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool -> !torch.vtensor<[1,3,1],f32>
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[AVGPOOL]], %[[E1]] : !torch.vtensor<[1,3,1],f32>, !torch.int -> !torch.vtensor<[1,3,1],f32>
  // CHECK: %[[INVP:.*]] = torch.constant.float 5.000000e-01
  // CHECK: torch.aten.pow.Tensor_Scalar %[[MUL]], %[[INVP]] : !torch.vtensor<[1,3,1],f32>, !torch.float -> !torch.vtensor<[1,3,1],f32>
  %0 = torch.operator "onnx.GlobalLpPool"(%arg0) : (!torch.vtensor<[1,3,5],f32>) -> !torch.vtensor<[1,3,1],f32>
  return %0 : !torch.vtensor<[1,3,1],f32>
}

// -----

// CHECK-LABEL: func.func @test_max_example
  func.func @test_max_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.maximum %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Max"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_max_one_input_example
  func.func @test_max_one_input_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: return %arg0 : !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Max"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_min_example
  func.func @test_min_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.minimum %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Min"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_min_one_input_example
  func.func @test_min_one_input_example(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: return %arg0 : !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Min"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_mod_int64_fmod
func.func @test_mod_int64_fmod(%arg0: !torch.vtensor<[6],si64>, %arg1: !torch.vtensor<[6],si64>) -> !torch.vtensor<[6],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.fmod.Tensor %arg0, %arg1 : !torch.vtensor<[6],si64>, !torch.vtensor<[6],si64> -> !torch.vtensor<[6],si64>
  %0 = torch.operator "onnx.Mod"(%arg0, %arg1) {torch.onnx.fmod = 1 : si64} : (!torch.vtensor<[6],si64>, !torch.vtensor<[6],si64>) -> !torch.vtensor<[6],si64>
  return %0 : !torch.vtensor<[6],si64>
}

// -----

// CHECK-LABEL: func.func @test_mod_int64_no_fmod
func.func @test_mod_int64_no_fmod(%arg0: !torch.vtensor<[6],si64>, %arg1: !torch.vtensor<[6],si64>) -> !torch.vtensor<[6],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.remainder.Tensor %arg0, %arg1 : !torch.vtensor<[6],si64>, !torch.vtensor<[6],si64> -> !torch.vtensor<[6],si64>
  %0 = torch.operator "onnx.Mod"(%arg0, %arg1) : (!torch.vtensor<[6],si64>, !torch.vtensor<[6],si64>) -> !torch.vtensor<[6],si64>
  return %0 : !torch.vtensor<[6],si64>
}

// -----

// CHECK-LABEL: func.func @test_log
  func.func @test_log(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.log %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.Log"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_log_softmax_default_axis
  func.func @test_log_softmax_default_axis(%arg0: !torch.vtensor<[1,3],f32>) -> !torch.vtensor<[1,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[CIM1:.*]] = torch.constant.int -1
    // CHECK: %[[NONE:.*]] = torch.constant.none
    // CHECK: %[[LSM:.*]] = torch.aten.log_softmax.int %arg0, %[[CIM1]], %[[NONE]] : !torch.vtensor<[1,3],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,3],f32>
    // CHECK: return %[[LSM]] : !torch.vtensor<[1,3],f32>
    %0 = torch.operator "onnx.LogSoftmax"(%arg0) : (!torch.vtensor<[1,3],f32>) -> !torch.vtensor<[1,3],f32>
    return %0 : !torch.vtensor<[1,3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_log_softmax_axis_2
  func.func @test_log_softmax_axis_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[CI2:.*]] = torch.constant.int 2
    // CHECK: %[[NONE:.*]] = torch.constant.none
    // CHECK: %[[LSM:.*]] = torch.aten.log_softmax.int %arg0, %[[CI2]], %[[NONE]] : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4,5],f32>
    // CHECK: return %[[LSM]] : !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.LogSoftmax"(%arg0) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_logsoftmax_old_axis_1_dynamic_dim
  func.func @test_logsoftmax_old_axis_1_dynamic_dim(%arg0: !torch.vtensor<[3,4,?],f32>) -> !torch.vtensor<[3,4,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[CI1:.*]] = torch.constant.int 1
    // CHECK: %[[NONE:.*]] = torch.constant.none
    // CHECK: %[[CI2:.*]] = torch.constant.int 2
    // CHECK: %[[CI4:.*]] = torch.constant.int 4
    // CHECK: %[[CIM1:.*]] = torch.constant.int -1
    // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[CI4]], %[[CIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[FLAT_IN:.*]] = torch.aten.flatten.using_ints %arg0, %[[CI1]], %[[CI2]] : !torch.vtensor<[3,4,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,?],f32>
    // CHECK: %[[LSM:.*]] = torch.aten.log_softmax.int %[[FLAT_IN]], %[[CI1]], %[[NONE]] : !torch.vtensor<[3,?],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,?],f32>
    // CHECK: %[[UNFLAT:.*]] = torch.aten.unflatten.int %[[LSM]], %[[CI1]], %[[LIST]] : !torch.vtensor<[3,?],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[3,4,?],f32>
    // CHECK: return %[[UNFLAT]] : !torch.vtensor<[3,4,?],f32>
    %0 = torch.operator "onnx.LogSoftmax"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,4,?],f32>) -> !torch.vtensor<[3,4,?],f32>
    return %0 : !torch.vtensor<[3,4,?],f32>
  }

// -----

// CHECK-LABEL: func.func @test_logsoftmax_old_axis_1_static
  func.func @test_logsoftmax_old_axis_1_static(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[CI1:.*]] = torch.constant.int 1
    // CHECK: %[[NONE:.*]] = torch.constant.none
    // CHECK: %[[CI2:.*]] = torch.constant.int 2
    // CHECK: %[[CI4:.*]] = torch.constant.int 4
    // CHECK: %[[CI5:.*]] = torch.constant.int 5
    // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[CI4]], %[[CI5]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[FLAT_IN:.*]] = torch.aten.flatten.using_ints %arg0, %[[CI1]], %[[CI2]] : !torch.vtensor<[3,4,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,20],f32>
    // CHECK: %[[LSM:.*]] = torch.aten.log_softmax.int %[[FLAT_IN]], %[[CI1]], %[[NONE]] : !torch.vtensor<[3,20],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,20],f32>
    // CHECK: %[[UNFLAT:.*]] = torch.aten.unflatten.int %[[LSM]], %[[CI1]], %[[LIST]] : !torch.vtensor<[3,20],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[3,4,5],f32>
    // CHECK: return %[[UNFLAT]] : !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.LogSoftmax"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_neg
  func.func @test_neg(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.neg %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.Neg"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_instancenorm
  func.func @test_instancenorm(%arg0: !torch.vtensor<[1,2,1,3],f32>, %arg1: !torch.vtensor<[2],f32>, %arg2: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2,1,3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 6 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.instance_norm %arg0, %arg1, %arg2, %none, %none, %true, %float0.000000e00, %float9.999990e-06, %false : !torch.vtensor<[1,2,1,3],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.none, !torch.none, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[1,2,1,3],f32>
    %0 = torch.operator "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,2,1,3],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2,1,3],f32>
    return %0 : !torch.vtensor<[1,2,1,3],f32>
  }

// -----

// CHECK-LABEL: func.func @test_not_2d
func.func @test_not_2d(%arg0: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.bitwise_not %arg0 : !torch.vtensor<[3,4],i1> -> !torch.vtensor<[3,4],i1>
    %0 = torch.operator "onnx.Not"(%arg0) : (!torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1>
    return %0 : !torch.vtensor<[3,4],i1>
  }

// -----

// CHECK-LABEL:   func.func @test_nllloss_ii
func.func @test_nllloss_ii(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK:           %[[VAL_3:.*]] = torch.constant.none
    // CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
    // CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
    // CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = torch.aten.nll_loss_forward %arg0, %arg1, %[[VAL_3]], %[[VAL_5]], %[[VAL_4]] : !torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
    // CHECK:           return %[[VAL_6]] : !torch.vtensor<[],f32>
    %0 = torch.operator "onnx.NegativeLogLikelihoodLoss"(%arg0, %arg1) {torch.onnx.ignore_index = 1 : si64, torch.onnx.reduction = "mean"} : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32>
    return %0 : !torch.vtensor<[],f32>
  }

// CHECK-LABEL:   func.func @test_nllloss_ii_ignore_default
func.func @test_nllloss_ii_ignore_default(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_3:.*]] = torch.constant.none
  // CHECK:           %[[VAL_4:.*]] = torch.constant.int -100
  // CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = torch.aten.nll_loss_forward %arg0, %arg1, %[[VAL_3]], %[[VAL_5]], %[[VAL_4]] : !torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  // CHECK:           return %[[VAL_6]] : !torch.vtensor<[],f32>
  %0 = torch.operator "onnx.NegativeLogLikelihoodLoss"(%arg0, %arg1) {torch.onnx.reduction = "mean"} : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @test_nllloss_ii_reduction_sum
func.func @test_nllloss_ii_reduction_sum(%arg0: !torch.vtensor<[3,5,6,6],f32>, %arg1: !torch.vtensor<[3,6,6],si64>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK:           %[[VAL_3:.*]] = torch.constant.none
    // CHECK:           %[[VAL_4:.*]] = torch.constant.int -100
    // CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
    // CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = torch.aten.nll_loss_forward %arg0, %arg1, %[[VAL_3]], %[[VAL_5]], %[[VAL_4]] : !torch.vtensor<[3,5,6,6],f32>, !torch.vtensor<[3,6,6],si64>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
    // CHECK:           return %[[VAL_6]] : !torch.vtensor<[],f32>
    %0 = torch.operator "onnx.NegativeLogLikelihoodLoss"(%arg0, %arg1) {torch.onnx.reduction = "sum"} : (!torch.vtensor<[3,5,6,6],f32>, !torch.vtensor<[3,6,6],si64>) -> !torch.vtensor<[],f32>
    return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @test_nllloss_iii_reduction_none_ignore_negative
func.func @test_nllloss_iii_reduction_none_ignore_negative(%arg0: !torch.vtensor<[3,5,6],f32>, %arg1: !torch.vtensor<[3,6],si64>, %arg2: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK:           %[[VAL_4:.*]] = torch.constant.int -1
  // CHECK:           %[[VAL_5:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = torch.aten.nll_loss_forward %arg0, %arg1, %arg2, %[[VAL_5]], %[[VAL_4]] : !torch.vtensor<[3,5,6],f32>, !torch.vtensor<[3,6],si64>, !torch.vtensor<[5],f32>, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  // CHECK:           return %[[VAL_6]] : !torch.vtensor<[],f32>
  %0 = torch.operator "onnx.NegativeLogLikelihoodLoss"(%arg0, %arg1, %arg2) {torch.onnx.ignore_index = -1 : si64, torch.onnx.reduction = "none"} : (!torch.vtensor<[3,5,6],f32>, !torch.vtensor<[3,6],si64>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

func.func @test_nonzero(%arg0: !torch.vtensor<[?],f32>) -> !torch.vtensor<[1,?],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[ZERO:.*]] = torch.constant.int 0
  // CHECK: %[[ONE:.*]] = torch.constant.int 1
  // CHECK: %[[NONZERO:.*]] = torch.aten.nonzero %arg0 : !torch.vtensor<[?],f32> -> !torch.vtensor<[?,1],si64>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[NONZERO]], %[[ZERO]], %[[ONE]] : !torch.vtensor<[?,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
  %0 = torch.operator "onnx.NonZero"(%arg0) : (!torch.vtensor<[?],f32>) -> !torch.vtensor<[1,?],si64>
  return %0 : !torch.vtensor<[1,?],si64>
}

// -----

// CHECK-LABEL: func.func @test_or2d
  func.func @test_or2d(%arg0: !torch.vtensor<[3,4],i1>, %arg1: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.bitwise_or.Tensor %arg0, %arg1 : !torch.vtensor<[3,4],i1>, !torch.vtensor<[3,4],i1> -> !torch.vtensor<[3,4],i1>
    %0 = torch.operator "onnx.Or"(%arg0, %arg1) : (!torch.vtensor<[3,4],i1>, !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1>
    return %0 : !torch.vtensor<[3,4],i1>
  }

// CHECK-LABEL: func.func @test_identity
  func.func @test_identity(%arg0: !torch.vtensor<[3,4], f32>) -> !torch.vtensor<[3,4], f32> attributes {torch.onnx_meta.ir_version = 14 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[NONE:.*]] = torch.constant.none
    // CHECK: %0 = torch.aten.clone %arg0, %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.none -> !torch.vtensor<[3,4],f32>
    %0 = torch.operator "onnx.Identity"(%arg0) : (!torch.vtensor<[3,4], f32>) -> !torch.vtensor<[3,4], f32>
    return %0 : !torch.vtensor<[3,4], f32>
  }

// CHECK-LABEL: func.func @test_mean_one_input
  func.func @test_mean_one_input(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Mean"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// CHECK-LABEL: func.func @test_mean_two_inputs
  func.func @test_mean_two_inputs(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: %[[INT2:.*]] = torch.constant.int 2
    // CHECK: %[[INT1:.*]] = torch.constant.int 1
    // CHECK: torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
    // CHECK: torch.aten.div.Scalar %0, %int2 : !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
    %0 = torch.operator "onnx.Mean"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }

// CHECK-LABEL: func.func @test_isinf_negative
  func.func @test_isinf_negative(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.neg %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],f32>
    // CHECK: torch.aten.relu %0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],f32>
    // CHECK: torch.aten.isinf %1 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],i1>
    %0 = torch.operator "onnx.IsInf"(%arg0) {torch.onnx.detect_positive = 0 : si64} : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1>
    return %0 : !torch.vtensor<[6],i1>
  }

// CHECK-LABEL: func.func @test_isinf_positive
  func.func @test_isinf_positive(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.relu %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],f32>
    // CHECK: torch.aten.isinf %0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],i1>
    %0 = torch.operator "onnx.IsInf"(%arg0) {torch.onnx.detect_negative = 0 : si64} : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1>
    return %0 : !torch.vtensor<[6],i1>
  }

// CHECK-LABEL: func.func @test_isnan
  func.func @test_isnan(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.isnan %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[6],i1>
    %0 = torch.operator "onnx.IsNaN"(%arg0) : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1>
    return %0 : !torch.vtensor<[6],i1>
  }

// CHECK-LABEL: func.func @test_prelu_example
  func.func @test_prelu_example(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.prelu %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.PRelu"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// CHECK-LABEL: func.func @test_prelu_broadcast
  func.func @test_prelu_broadcast(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // CHECK: torch.aten.prelu %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[3,4,5],f32>
    %0 = torch.operator "onnx.PRelu"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }

// -----

// CHECK-LABEL: func.func @test_mish
func.func @test_mish(%arg0: !torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  // CHECK: torch.aten.mish %arg0 : !torch.vtensor<[10000],f32> -> !torch.vtensor<[10000],f32>
  %0 = torch.operator "onnx.Mish"(%arg0) : (!torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32>
  return %0 : !torch.vtensor<[10000],f32>
}

// -----

// CHECK-LABEL: func.func @test_hardswish
func.func @test_hardswish(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: torch.aten.hardswish %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
  %0 = torch.operator "onnx.HardSwish"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_hardmax
func.func @test_hardmax(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[AXIS:.+]] = torch.constant.int 1
  // CHECK: %[[FALSE]] = torch.constant.bool false
  // CHECK: %[[ARGMAX:.+]] = torch.aten.argmax %arg0, %[[AXIS]], %[[FALSE]]
  // CHECK: %[[CLASSES:.+]] = torch.aten.size.int %arg0, %[[AXIS]]
  // CHECK: %[[ONEHOT:.+]] = torch.aten.one_hot %[[ARGMAX]], %[[CLASSES]]
  // CHECK: %[[PERM0:.+]] = torch.constant.int 0
  // CHECK: %[[PERM2:.+]] = torch.constant.int 2
  // CHECK: %[[PERM1:.+]] = torch.constant.int 1
  // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[PERM0]], %[[PERM2]], %[[PERM1]]
  // CHECK: %[[PERMUTE:.+]] = torch.aten.permute %[[ONEHOT]], %[[LIST]]
  // CHECK: return %[[PERMUTE]]
  %0 = torch.operator "onnx.Hardmax"(%arg0) {torch.onnx.axis = -2 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: func.func @test_onehot_negative_indices
func.func @test_onehot_negative_indices(%arg0: !torch.vtensor<[3],si64>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[2],f32>) -> !torch.vtensor<[10,3],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[ITEM:.*]] = torch.aten.item %arg1 : !torch.vtensor<[],f32> -> !torch.float
  // CHECK: %[[INT:.*]] = torch.aten.Int.Scalar %[[ITEM]] : !torch.float -> !torch.int
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C1:.*]]= torch.constant.int 1
  // CHECK: %[[LT:.*]] = torch.aten.lt.Scalar %arg0, %[[C0]] : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],i1>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Scalar %arg0, %[[INT]], %[[C1]]: !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[3],si64>
  // CHECK: %[[WHERE:.*]] = torch.aten.where.self %[[LT]], %[[ADD]], %arg0 : !torch.vtensor<[3],i1>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64> -> !torch.vtensor<[3],si64>
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %arg2, %[[C0]], %[[C0]] : !torch.vtensor<[2],f32>, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  // CHECK: %[[ITEM_0:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %arg2, %[[C0]], %[[C1]]: !torch.vtensor<[2],f32>, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  // CHECK: %[[ITEM_1:.*]] = torch.aten.item %[[SELECT]] : !torch.vtensor<[1],f32> -> !torch.float
  // CHECK: %[[ONEHOT:.*]] = torch.aten.one_hot %[[WHERE]], %[[INT]] : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3,?],si32>
  // CHECK: %[[D0:.+]] = torch.constant.int 1
  // CHECK: %[[D1:.+]] = torch.constant.int 0
  // CHECK: %[[TRANS:.+]] = torch.aten.transpose.int %[[ONEHOT]], %[[D1]], %[[D0]]
  // CHECK: %[[C11:.*]] = torch.constant.int 11
  // CHECK: %[[NONE_0:.*]] = torch.constant.none
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[DTYPE:.*]] = torch.aten.to.dtype %[[TRANS]], %[[C11]], %[[FALSE]], %[[FALSE]], %[[NONE_0]] : !torch.vtensor<[?,3],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,3],i1>
  // CHECK: %[[RESULT:.*]] = torch.aten.where.Scalar %[[DTYPE]], %[[ITEM_1]], %[[ITEM_0]] : !torch.vtensor<[?,3],i1>, !torch.float, !torch.float -> !torch.vtensor<[10,3],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[10,3],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.OneHot"(%arg0, %arg1, %arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3],si64>, !torch.vtensor<[],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[10,3],f32>
  return %0 : !torch.vtensor<[10,3],f32>
}

// -----

// CHECK-LABEL: @test_lpnormalization
func.func @test_lpnormalization(%arg0: !torch.vtensor<[3,4,5,6,7],f32>) -> !torch.vtensor<[3,4,5,6,7],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[CST2:.*]] = torch.constant.int 2
  // CHECK: %[[CST2_0:.*]] = torch.constant.int 2
  // CHECK: %[[TRUE:.*]] = torch.constant.bool true
  // CHECK: %[[DIMS:.*]] = torch.prim.ListConstruct %[[CST2]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[NORM:.*]] = torch.aten.norm.ScalarOpt_dim %arg0, %[[CST2_0]], %[[DIMS]], %[[TRUE]] : !torch.vtensor<[3,4,5,6,7],f32>, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,4,1,6,7],f32>
  // CHECK: %[[OUT:.*]] = torch.aten.div.Tensor %arg0, %[[NORM]] : !torch.vtensor<[3,4,5,6,7],f32>, !torch.vtensor<[3,4,1,6,7],f32> -> !torch.vtensor<[3,4,5,6,7],f32>
  // CHECK: return %[[OUT]] : !torch.vtensor<[3,4,5,6,7],f32>
  %0 = torch.operator "onnx.LpNormalization"(%arg0) {torch.onnx.axis = 2 : si64, torch.onnx.p = 2 : si64} : (!torch.vtensor<[3,4,5,6,7],f32>) -> !torch.vtensor<[3,4,5,6,7],f32>
  return %0 : !torch.vtensor<[3,4,5,6,7],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxunpool_export_without_output_shape
func.func @test_maxunpool_export_without_output_shape(%arg0: !torch.vtensor<[1,1,2,2],f32>, %arg1: !torch.vtensor<[1,1,2,2],si64>) -> !torch.vtensor<[1,1,4,4],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[INT4_0:.*]] = torch.constant.int 4
  // CHECK: %[[OUTPUT_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1_0]], %[[INT4]], %[[INT4_0]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.max_unpool2d %arg0, %arg1, %[[OUTPUT_SHAPE]] : !torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,1,2,2],si64>, !torch.list<int> -> !torch.vtensor<[1,1,4,4],f32>
  // return %[[RESULT]] : !torch.vtensor<[1,1,4,4],f32>
  %0 = torch.operator "onnx.MaxUnpool"(%arg0, %arg1) {torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,1,2,2],si64>) -> !torch.vtensor<[1,1,4,4],f32>
  return %0 : !torch.vtensor<[1,1,4,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_maxunpool3d_export_without_output_shape
func.func @test_maxunpool3d_export_without_output_shape(%arg0: !torch.vtensor<[1,1,2,2,2],f32>, %arg1: !torch.vtensor<[1,1,2,2,2],si64>) -> !torch.vtensor<[1,1,4,4,4],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[INT1_0:.*]] = torch.constant.int 1
  // CHECK: %[[INT4:.*]] = torch.constant.int 4
  // CHECK: %[[INT4_0:.*]] = torch.constant.int 4
  // CHECK: %[[INT4_1:.*]] = torch.constant.int 4
  // CHECK: %[[OUTPUT_SHAPE:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1_0]], %[[INT4]], %[[INT4_0]], %[[INT4_1]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_1:.*]] = torch.constant.int 0
  // CHECK: %[[INT0_2:.*]] = torch.constant.int 0
  // CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT0_1]], %[[INT0_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[INT2_1:.*]] = torch.constant.int 2
  // CHECK: %[[INT2_2:.*]] = torch.constant.int 2
  // CHECK: %[[STRIDE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2_1]], %[[INT2_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.max_unpool3d %arg0, %arg1, %[[OUTPUT_SHAPE]], %[[STRIDE]], %[[PADDING]] : !torch.vtensor<[1,1,2,2,2],f32>, !torch.vtensor<[1,1,2,2,2],si64>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,1,4,4,4],f32>
  // return %[[RESULT]] : !torch.vtensor<[1,1,4,4,4],f32>
  %0 = torch.operator "onnx.MaxUnpool"(%arg0, %arg1) {torch.onnx.kernel_shape = [2 : si64, 2 : si64, 2 : si64], torch.onnx.strides = [2 : si64, 2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,2,2,2],f32>, !torch.vtensor<[1,1,2,2,2],si64>) -> !torch.vtensor<[1,1,4,4,4],f32>
  return %0 : !torch.vtensor<[1,1,4,4,4],f32>
}

// -----

// CHECK-LABEL: func.func @test_group_normalization
func.func @test_group_normalization(%arg0: !torch.vtensor<[3,4,2,2],f32>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4,2,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[EPSILON:.*]] = torch.constant.float 9.9999997473787516E-6
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[RESULT:.*]] = torch.aten.group_norm %arg0, %int2, %arg1, %arg2, %[[EPSILON]], %[[FALSE:.*]] : !torch.vtensor<[3,4,2,2],f32>, !torch.int, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,2,2],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[3,4,2,2],f32>
  %0 = torch.operator "onnx.GroupNormalization"(%arg0, %arg1, %arg2) {torch.onnx.num_groups = 2 : si64} : (!torch.vtensor<[3,4,2,2],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4,2,2],f32>
  return %0 : !torch.vtensor<[3,4,2,2],f32>
}

// -----

func.func @test_group_normalization_epsilon(%arg0: !torch.vtensor<[3,4,2,2],f32>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4,2,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[EPSILON:.*]] = torch.constant.float 0.0099999997764825821
  // CHECK: %[[INT2:.*]] = torch.constant.int 2
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[RESULT:.*]] = torch.aten.group_norm %arg0, %int2, %arg1, %arg2, %[[EPSILON]], %[[FALSE:.*]] : !torch.vtensor<[3,4,2,2],f32>, !torch.int, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float, !torch.bool -> !torch.vtensor<[3,4,2,2],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[3,4,2,2],f32>
  %0 = torch.operator "onnx.GroupNormalization"(%arg0, %arg1, %arg2) {torch.onnx.epsilon = 0.00999999977 : f32, torch.onnx.num_groups = 2 : si64} : (!torch.vtensor<[3,4,2,2],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4,2,2],f32>
  return %0 : !torch.vtensor<[3,4,2,2],f32>
}

// -----

// CHECK-LABEL: @test_optional
func.func @test_optional(%arg0: !torch.list<vtensor<[5],f32>>) -> !torch.optional<list<vtensor<[5],f32>>> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64} {
  // CHECK: %[[RESULT:.*]] = torch.derefine %arg0 : !torch.list<vtensor<[5],f32>> to !torch.optional<list<vtensor<[5],f32>>>
  // CHECK: return %[[RESULT]] : !torch.optional<list<vtensor<[5],f32>>>
  %0 = torch.operator "onnx.Optional"(%arg0) : (!torch.list<vtensor<[5],f32>>) -> !torch.optional<list<vtensor<[5],f32>>>
  return %0 : !torch.optional<list<vtensor<[5],f32>>>
}

// -----

// CHECK-LABEL: @test_optional_get_element_optional_sequence
func.func @test_optional_get_element_optional_sequence(%arg0: !torch.optional<list<vtensor<[4],si32>>>) -> !torch.list<vtensor<[4],si32>> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[RESULT:.*]] = torch.prim.unchecked_cast %arg0 : !torch.optional<list<vtensor<[4],si32>>> -> !torch.list<vtensor<[4],si32>>
  // CHECK: return %[[RESULT]] : !torch.list<vtensor<[4],si32>>
  %0 = torch.operator "onnx.OptionalGetElement"(%arg0) : (!torch.optional<list<vtensor<[4],si32>>>) -> !torch.list<vtensor<[4],si32>>
  return %0 : !torch.list<vtensor<[4],si32>>
}

// -----

// CHECK-LABEL: @test_optional_get_element_optional_tensor
func.func @test_optional_get_element_optional_tensor(%arg0: !torch.optional<vtensor<[4],f32>>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[RESULT:.*]] = torch.prim.unchecked_cast %arg0 : !torch.optional<vtensor<[4],f32>> -> !torch.vtensor<[4],f32>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[4],f32>
  %0 = torch.operator "onnx.OptionalGetElement"(%arg0) : (!torch.optional<vtensor<[4],f32>>) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @test_optional_get_element_sequence
func.func @test_optional_get_element_sequence(%arg0: !torch.list<vtensor<[4],si32>>) -> !torch.list<vtensor<[4],si32>> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: return %arg0 : !torch.list<vtensor<[4],si32>>
  %0 = torch.operator "onnx.OptionalGetElement"(%arg0) : (!torch.list<vtensor<[4],si32>>) -> !torch.list<vtensor<[4],si32>>
  return %0 : !torch.list<vtensor<[4],si32>>
}

// -----

// CHECK-LABEL: @test_optional_get_element_tensor
func.func @test_optional_get_element_tensor(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: return %arg0 : !torch.vtensor<[4],f32>
  %0 = torch.operator "onnx.OptionalGetElement"(%arg0) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @test_optional_has_element_empty_none_input
func.func @test_optional_has_element_empty_none_input() -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 11
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE_0:.*]] = torch.constant.none
  // CHECK: %[[DATA_LIST:.*]] = torch.prim.ListConstruct %[[INT0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.tensor %[[DATA_LIST]], %[[DTYPE]], %[[NONE_0]], %[[FALSE]] : !torch.list<int>, !torch.int, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[],i1>
  %none = torch.constant.none
  %0 = torch.operator "onnx.OptionalHasElement"(%none) : (!torch.none) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: @test_optional_has_element_empty_no_input
func.func @test_optional_has_element_empty_no_input() -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.*]] = torch.constant.int 0
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 11
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DATA_LIST:.*]] = torch.prim.ListConstruct %[[INT0]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.tensor %[[DATA_LIST]], %[[DTYPE]], %[[NONE]], %[[FALSE]] : !torch.list<int>, !torch.int, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.OptionalHasElement"() : () -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: @test_optional_has_element_empty_optional_input
func.func @test_optional_has_element_empty_optional_input(%arg0: !torch.optional<vtensor<[],si32>>) -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 11
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DATA_LIST:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.tensor %[[DATA_LIST]], %[[DTYPE]], %[[NONE]], %[[FALSE]] : !torch.list<int>, !torch.int, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.OptionalHasElement"(%arg0) : (!torch.optional<vtensor<[],si32>>) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: @test_optional_has_element_optional_tensor_input
func.func @test_optional_has_element_optional_tensor_input(%arg0: !torch.optional<vtensor<[4],f32>>) -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 11
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DATA_LIST:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.tensor %[[DATA_LIST]], %[[DTYPE]], %[[NONE]], %[[FALSE]] : !torch.list<int>, !torch.int, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.OptionalHasElement"(%arg0) : (!torch.optional<vtensor<[4],f32>>) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: @test_optional_has_element_optional_list_tensor_input
func.func @test_optional_has_element_optional_list_tensor_input(%arg0: !torch.optional<list<vtensor<[4],f32>>>) -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 11
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DATA_LIST:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.tensor %[[DATA_LIST]], %[[DTYPE]], %[[NONE]], %[[FALSE]] : !torch.list<int>, !torch.int, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.OptionalHasElement"(%arg0) : (!torch.optional<list<vtensor<[4],f32>>>) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: @test_optional_has_element_tensor_input
func.func @test_optional_has_element_tensor_input(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 11
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DATA_LIST:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.tensor %[[DATA_LIST]], %[[DTYPE]], %[[NONE]], %[[FALSE]] : !torch.list<int>, !torch.int, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.OptionalHasElement"(%arg0) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL: @test_optional_has_element_list_tensor_input
func.func @test_optional_has_element_list_tensor_input(%arg0: !torch.list<vtensor<[4],f32>>) -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT1:.*]] = torch.constant.int 1
  // CHECK: %[[DTYPE:.*]] = torch.constant.int 11
  // CHECK: %[[FALSE:.*]] = torch.constant.bool false
  // CHECK: %[[NONE:.*]] = torch.constant.none
  // CHECK: %[[DATA_LIST:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.*]] = torch.aten.tensor %[[DATA_LIST]], %[[DTYPE]], %[[NONE]], %[[FALSE]] : !torch.list<int>, !torch.int, !torch.none, !torch.bool -> !torch.vtensor<[],i1>
  // CHECK: return %[[RESULT]] : !torch.vtensor<[],i1>
  %0 = torch.operator "onnx.OptionalHasElement"(%arg0) : (!torch.list<vtensor<[4],f32>>) -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----

// CHECK-LABEL:   func.func @test_loop_forlike
func.func @test_loop_forlike(%arg0: !torch.vtensor<[],si64>, %arg1: !torch.vtensor<[],i1>, %arg2: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "loop_example", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME:                                 %[[MAX_TRIP_COUNT_INP:.*]]: !torch.vtensor<[],si64>,
  // CHECK-SAME:                                 %[[CONDITION_INP:.*]]: !torch.vtensor<[],i1>,
  // CHECK-SAME:                                 %[[LCD_1:.*]]: !torch.vtensor<[1],f32>
  // CHECK:           %[[NONE_0:.*]] = torch.constant.none
  // CHECK:           %[[MAX_TRIP_COUNT_INT:.*]] = torch.aten.item %[[MAX_TRIP_COUNT_INP]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:           %[[CONDITION_INT:.*]] = torch.aten.item %[[CONDITION_INP]] : !torch.vtensor<[],i1> -> !torch.int
  // CHECK:           %[[CONDITION_BOOL:.*]] = torch.aten.Bool.int %[[CONDITION_INT]] : !torch.int -> !torch.bool
  // CHECK:           %[[TRUE:.*]] = torch.constant.bool true
  // CHECK:           %[[LOOP:.*]] = torch.prim.Loop %[[MAX_TRIP_COUNT_INT]], %[[TRUE]], init(%[[LCD_1]]) {
  // CHECK:           ^bb0(%[[ITER_NUM:.*]]: !torch.int, %[[LCD_1_BODY:.*]]: !torch.vtensor<[1],f32>):
  // CHECK:             %[[ITER_NUM_T:.*]] = torch.prim.NumToTensor.Scalar %[[ITER_NUM]] : !torch.int -> !torch.vtensor<[],si64>
  // CHECK:             %[[NONE_1:.*]] = torch.constant.none
  // CHECK:             %[[CLONE_INP_COND:.*]] = torch.aten.clone %[[CONDITION_INP]], %[[NONE_1]] : !torch.vtensor<[],i1>, !torch.none -> !torch.vtensor<[],i1>
  // CHECK:             %[[CONST_ARR:.*]] = torch.vtensor.literal(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>) : !torch.vtensor<[5],f32>
  // CHECK:             %[[ONE_T:.*]] = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK:             %[[ONE_0:.*]] = torch.constant.int 1
  // CHECK:             %[[ADD_ONE_T:.*]] = torch.aten.add.Tensor %[[ITER_NUM_T]], %[[ONE_T]], %[[ONE_0]] : !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
  // CHECK:             %[[ZERO_T:.*]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  // CHECK:             %[[ZERO_0:.*]] = torch.constant.int 0
  // CHECK:             %[[ITER_NUM_RT:.*]] = torch.aten.unsqueeze %[[ITER_NUM_T]], %[[ZERO_0]] : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK:             %[[ZERO_1:.*]] = torch.constant.int 0
  // CHECK:             %[[ADD_ONE_RT:.*]] = torch.aten.unsqueeze %[[ADD_ONE_T]], %[[ZERO_1]] : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK:             %[[NONE_2:.*]] = torch.constant.none
  // CHECK:             %[[ONE_1:.*]] = torch.constant.int 1
  // CHECK:             %[[ONE_SIZE_LIST:.*]] = torch.prim.ListConstruct %[[ONE_1]] : (!torch.int) -> !torch.list<int>
  // CHECK:             %[[ONES_T:.*]] = torch.aten.ones %[[ONE_SIZE_LIST]], %[[NONE_2]], %[[NONE_2]], %[[NONE_2]], %[[NONE_2]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si64>
  // CHECK:             %[[ZERO_2:.*]] = torch.constant.int 0
  // CHECK:             %[[ZERO_3:.*]] = torch.constant.int 0
  // CHECK:             %[[ZERO_T_1:.*]] = torch.prim.NumToTensor.Scalar %[[ZERO_3]] : !torch.int -> !torch.vtensor<[1],si64>
  // CHECK:             %[[ITER_NUM_INDEXED:.*]] = torch.aten.index_select %[[ITER_NUM_RT]], %[[ZERO_2]], %[[ZERO_T_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK:             %[[ITER_NUM_INT:.*]] = torch.aten.item %[[ITER_NUM_INDEXED]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK:             %[[INC_INDEXED:.*]] = torch.aten.index_select %[[ADD_ONE_RT]], %[[ZERO_2]], %[[ZERO_T_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK:             %[[INC_INT:.*]] = torch.aten.item %[[INC_INDEXED]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK:             %[[SLICE_INDEX_T:.*]] = torch.aten.index_select %[[ONES_T]], %[[ZERO_2]], %[[ZERO_T_1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
  // CHECK:             %[[INDEX_INT:.*]] = torch.aten.item %[[SLICE_INDEX_T]] : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK:             %[[INPUT_SLICE:.*]] = torch.aten.slice.Tensor %[[CONST_ARR]], %[[ZERO_3]], %[[ITER_NUM_INT]], %[[INC_INT]], %[[INDEX_INT]] : !torch.vtensor<[5],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
  // CHECK:             %[[ONE_2:.*]] = torch.constant.int 1
  // CHECK:             %[[INTERM_RES:.*]] = torch.aten.add.Tensor %[[LCD_1_BODY]], %[[INPUT_SLICE]], %[[ONE_2]] : !torch.vtensor<[1],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[1],f32>
  // CHECK:             torch.prim.Loop.condition %[[TRUE]], iter(%[[INTERM_RES]] : !torch.vtensor<[1],f32>)
  // CHECK:           } : (!torch.int, !torch.bool, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32>
  // CHECK:           return %[[LOOP]] : !torch.vtensor<[1],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.Loop"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],si64>, !torch.vtensor<[],i1>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> {
  ^bb0(%arg3: !torch.vtensor<[],si64>, %arg4: !torch.vtensor<[],i1>, %arg5: !torch.vtensor<[1],f32>):
    %1 = torch.operator "onnx.Identity"(%arg4) : (!torch.vtensor<[],i1>) -> !torch.vtensor<[],i1>
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>} : () -> !torch.vtensor<[5],f32>
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<si64>} : () -> !torch.vtensor<[],si64>
    %4 = torch.operator "onnx.Add"(%arg3, %3) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64>
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si64>} : () -> !torch.vtensor<[],si64>
    %6 = torch.operator "onnx.Unsqueeze"(%arg3, %5) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[1],si64>
    %7 = torch.operator "onnx.Unsqueeze"(%4, %5) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[1],si64>
    %8 = torch.operator "onnx.Slice"(%2, %6, %7) : (!torch.vtensor<[5],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[?],f32>
    %9 = torch.operator "onnx.Add"(%arg5, %8) : (!torch.vtensor<[1],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[1],f32>
    torch.operator_terminator %1, %9 : !torch.vtensor<[],i1>, !torch.vtensor<[1],f32>
  }
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:   func.func @test_nonmaxsuppression_identical_boxes(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[1,10,4],f32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: !torch.vtensor<[1,1,10],f32>,
// CHECK-SAME:                                                      %[[VAL_2:.*]]: !torch.vtensor<[1],si64>,
// CHECK-SAME:                                                      %[[VAL_3:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                                      %[[VAL_4:.*]]: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_nonmaxsuppression_identical_boxes(%arg0: !torch.vtensor<[1,10,4],f32>, %arg1: !torch.vtensor<[1,1,10],f32>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1],f32>, %arg4: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK:           %[[RES:.*]] = torch.aten.empty.memory_format
// CHECK:           %[[LOOP_BATCH:.*]]:3 = torch.prim.Loop
// CHECK:             %[[LOOP_CLASS:.*]]:3 = torch.prim.Loop
// CHECK:               %[[NMS:.*]] = torch.torchvision.nms
// CHECK:               %[[MIN_RES:.*]] = torch.aten.minimum
// CHECK:               %[[NUM_ELE:.*]] = torch.aten.item %[[MIN_RES]] : !torch.vtensor<[],si64> -> !torch.int
// CHECK:               %[[LOOP_NMS:.*]]:2 = torch.prim.Loop %[[NUM_ELE]]
// CHECK:                 %[[SEL_1:.*]] = torch.aten.select.int
// CHECK:                 %[[SEL_2:.*]] = torch.aten.select.int
// CHECK:                 %[[COPY:.*]] = torch.aten.copy
// CHECK:                 %[[SEL_3:.*]] = torch.aten.select.int
// CHECK-COUNT-6:         torch.aten.select_scatter
// CHECK:                 %[[ADD_INDEX:.*]] = torch.aten.add.int
// CHECK:                 torch.prim.Loop.condition
// CHECK:                 %[[ADD_INDEX_1:.*]] = torch.aten.add.int
// CHECK:               torch.prim.Loop.condition
// CHECK:             torch.prim.Loop.condition
// CHECK:           %[[SLICE_RES:.*]] = torch.aten.slice.Tensor
// CHECK:           return %[[SLICE_RES]] : !torch.vtensor<[1,3],si64>
  %0 = torch.operator "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[1,10,4],f32>, !torch.vtensor<[1,1,10],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64>
  return %0 : !torch.vtensor<[1,3],si64>
}

// -----

// CHECK-LABEL:   func.func @test_nonmaxsuppression_single_box(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !torch.vtensor<[1,1,4],f32>,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: !torch.vtensor<[1,1,1],f32>,
// CHECK-SAME:                                                 %[[VAL_2:.*]]: !torch.vtensor<[1],si64>,
// CHECK-SAME:                                                 %[[VAL_3:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                                 %[[VAL_4:.*]]: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""}
func.func @test_nonmaxsuppression_single_box(%arg0: !torch.vtensor<[1,1,4],f32>, %arg1: !torch.vtensor<[1,1,1],f32>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1],f32>, %arg4: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK:           %[[RES:.*]] = torch.aten.empty.memory_format
// CHECK:           %[[LOOP_BATCH:.*]]:3 = torch.prim.Loop
// CHECK:             %[[LOOP_CLASS:.*]]:3 = torch.prim.Loop
// CHECK:               %[[NMS:.*]] = torch.torchvision.nms
// CHECK:               %[[MIN_RES:.*]] = torch.aten.minimum
// CHECK:               %[[NUM_ELE:.*]] = torch.aten.item %[[MIN_RES]] : !torch.vtensor<[],si64> -> !torch.int
// CHECK:               %[[LOOP_NMS:.*]]:2 = torch.prim.Loop %[[NUM_ELE]]
// CHECK:                 %[[SEL_1:.*]] = torch.aten.select.int
// CHECK:                 %[[SEL_2:.*]] = torch.aten.select.int
// CHECK:                 %[[COPY:.*]] = torch.aten.copy
// CHECK:                 %[[SEL_3:.*]] = torch.aten.select.int
// CHECK-COUNT-6:         torch.aten.select_scatter
// CHECK:                 %[[ADD_INDEX:.*]] = torch.aten.add.int
// CHECK:                 torch.prim.Loop.condition
// CHECK:                 %[[ADD_INDEX_1:.*]] = torch.aten.add.int
// CHECK:               torch.prim.Loop.condition
// CHECK:             torch.prim.Loop.condition
// CHECK:           %[[SLICE_RES:.*]] = torch.aten.slice.Tensor
// CHECK:           return %[[SLICE_RES]] : !torch.vtensor<[1,3],si64>
  %0 = torch.operator "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[1,1,4],f32>, !torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64>
  return %0 : !torch.vtensor<[1,3],si64>
}

// CHECK-LABEL:   func.func @test_nonmaxsuppression_center_point_box(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: !torch.vtensor<[1,1,4],f32>,
// CHECK-SAME:                                                       %[[VAL_1:.*]]: !torch.vtensor<[1,1,1],f32>,
// CHECK-SAME:                                                       %[[VAL_2:.*]]: !torch.vtensor<[1],si64>,
// CHECK-SAME:                                                       %[[VAL_3:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                                       %[[VAL_4:.*]]: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_nonmaxsuppression_center_point_box(%arg0: !torch.vtensor<[1,1,4],f32>, %arg1: !torch.vtensor<[1,1,1],f32>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1],f32>, %arg4: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_10:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_11:.*]] = torch.constant.none
// CHECK:           %[[VAL_12:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_13:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_14:.*]] = torch.aten.slice.Tensor %[[VAL_0]], %[[VAL_7]], %[[VAL_5]], %[[VAL_7]], %[[VAL_6]] : !torch.vtensor<[1,1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,2],f32>
// CHECK:           %[[VAL_15:.*]] = torch.aten.slice.Tensor %[[VAL_0]], %[[VAL_7]], %[[VAL_7]], %[[VAL_9]], %[[VAL_6]] : !torch.vtensor<[1,1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,2],f32>
// CHECK:           %[[VAL_16:.*]] = torch.aten.div.Scalar %[[VAL_15]], %[[VAL_10]] : !torch.vtensor<[?,?,2],f32>, !torch.float -> !torch.vtensor<[?,?,2],f32>
// CHECK:           %[[VAL_17:.*]] = torch.aten.sub.Tensor %[[VAL_14]], %[[VAL_16]], %[[VAL_6]] : !torch.vtensor<[?,?,2],f32>, !torch.vtensor<[?,?,2],f32>, !torch.int -> !torch.vtensor<[?,?,2],f32>
// CHECK:           %[[VAL_18:.*]] = torch.aten.add.Tensor %[[VAL_14]], %[[VAL_16]], %[[VAL_6]] : !torch.vtensor<[?,?,2],f32>, !torch.vtensor<[?,?,2],f32>, !torch.int -> !torch.vtensor<[?,?,2],f32>
// CHECK:           %[[VAL_19:.*]] = torch.prim.ListConstruct %[[VAL_17]], %[[VAL_18]] : (!torch.vtensor<[?,?,2],f32>, !torch.vtensor<[?,?,2],f32>) -> !torch.list<vtensor>
// CHECK:           %[[VAL_20:.*]] = torch.aten.cat %[[VAL_19]], %[[VAL_7]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,1,4],f32>
// CHECK:           %[[VAL_21:.*]] = torch.aten.item %[[VAL_4]] : !torch.vtensor<[1],f32> -> !torch.float
// CHECK:           %[[VAL_22:.*]] = torch.aten.min %[[VAL_1]] : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[VAL_23:.*]] = torch.aten.item %[[VAL_22]] : !torch.vtensor<[],f32> -> !torch.float
// CHECK:           %[[VAL_24:.*]] = torch.aten.ge.float %[[VAL_23]], %[[VAL_21]] : !torch.float, !torch.float -> !torch.bool
// CHECK:           torch.runtime.assert %[[VAL_24]], "unimplemented: score_threshold should be <= min(scores)"
// CHECK:           %[[RES:.*]] = torch.aten.empty.memory_format
// CHECK:           %[[LOOP_BATCH:.*]]:3 = torch.prim.Loop
// CHECK:             %[[LOOP_CLASS:.*]]:3 = torch.prim.Loop
// CHECK:               %[[NMS:.*]] = torch.torchvision.nms
// CHECK:               %[[MIN_RES:.*]] = torch.aten.minimum
// CHECK:               %[[NUM_ELE:.*]] = torch.aten.item %[[MIN_RES]] : !torch.vtensor<[],si64> -> !torch.int
// CHECK:               %[[LOOP_NMS:.*]]:2 = torch.prim.Loop %[[NUM_ELE]]
// CHECK:                 %[[SEL_1:.*]] = torch.aten.select.int
// CHECK:                 %[[SEL_2:.*]] = torch.aten.select.int
// CHECK:                 %[[COPY:.*]] = torch.aten.copy
// CHECK:                 %[[SEL_3:.*]] = torch.aten.select.int
// CHECK-COUNT-6:         torch.aten.select_scatter
// CHECK:                 %[[ADD_INDEX:.*]] = torch.aten.add.int
// CHECK:                 torch.prim.Loop.condition
// CHECK:                 %[[ADD_INDEX_1:.*]] = torch.aten.add.int
// CHECK:               torch.prim.Loop.condition
// CHECK:             torch.prim.Loop.condition
// CHECK:           %[[SLICE_RES:.*]] = torch.aten.slice.Tensor
// CHECK:           return %[[SLICE_RES]] : !torch.vtensor<[1,3],si64>
  %0 = torch.operator "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.center_point_box = 1 : si64} : (!torch.vtensor<[1,1,4],f32>, !torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64>
  return %0 : !torch.vtensor<[1,3],si64>
}

// -----

// CHECK-LABEL:   func.func @test_nonmaxsuppression_multiple_batch_class(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !torch.vtensor<[3,8,4],f32>,
// CHECK-SAME:                                                           %[[VAL_1:.*]]: !torch.vtensor<[3,5,8],f32>,
// CHECK-SAME:                                                           %[[VAL_2:.*]]: !torch.vtensor<[1],si64>,
// CHECK-SAME:                                                           %[[VAL_3:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                                           %[[VAL_4:.*]]: !torch.vtensor<[1],f32>) -> !torch.vtensor<[?,3],si64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
func.func @test_nonmaxsuppression_multiple_batch_class(%arg0: !torch.vtensor<[3,8,4],f32>, %arg1: !torch.vtensor<[3,5,8],f32>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1],f32>, %arg4: !torch.vtensor<[1],f32>) -> !torch.vtensor<[?,3],si64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
// CHECK:           %[[RES:.*]] = torch.aten.empty.memory_format
// CHECK:           %[[LOOP_BATCH:.*]]:3 = torch.prim.Loop
// CHECK:             %[[LOOP_CLASS:.*]]:3 = torch.prim.Loop
// CHECK:               %[[NMS:.*]] = torch.torchvision.nms
// CHECK:               %[[MIN_RES:.*]] = torch.aten.minimum
// CHECK:               %[[NUM_ELE:.*]] = torch.aten.item %[[MIN_RES]] : !torch.vtensor<[],si64> -> !torch.int
// CHECK:               %[[LOOP_NMS:.*]]:2 = torch.prim.Loop %[[NUM_ELE]]
// CHECK:                 %[[SEL_1:.*]] = torch.aten.select.int
// CHECK:                 %[[SEL_2:.*]] = torch.aten.select.int
// CHECK:                 %[[COPY:.*]] = torch.aten.copy
// CHECK:                 %[[SEL_3:.*]] = torch.aten.select.int
// CHECK-COUNT-6:         torch.aten.select_scatter
// CHECK:                 %[[ADD_INDEX:.*]] = torch.aten.add.int
// CHECK:                 torch.prim.Loop.condition
// CHECK:                 %[[ADD_INDEX_1:.*]] = torch.aten.add.int
// CHECK:               torch.prim.Loop.condition
// CHECK:             torch.prim.Loop.condition
// CHECK:           %[[SLICE_RES:.*]] = torch.aten.slice.Tensor
// CHECK:           return %[[SLICE_RES]] : !torch.vtensor<[?,3],si64>
  %0 = torch.operator "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.center_point_box = 0 : si64} : (!torch.vtensor<[3,8,4],f32>, !torch.vtensor<[3,5,8],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[?,3],si64>
  return %0 : !torch.vtensor<[?,3],si64>
}

// -----

// CHECK-LABEL:   func.func @test_mwm
func.func @test_mwm(%arg0: !torch.vtensor<[],si64>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[],si64>, %arg3: !torch.vtensor<[],f32>, %arg4: !torch.vtensor<[],f32>) -> !torch.vtensor<[9,8],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "test_mwm", torch.onnx_meta.producer_version = ""} {
  // CHECK-SAME:                           %[[NUM_MEL_BINS_ARG:.*]]: !torch.vtensor<[],si64>, %[[DFT_LENGTH_ARG:.*]]: !torch.vtensor<[],si64>, %[[SAMPLE_RATE_ARG:.*]]: !torch.vtensor<[],si64>,
  // CHECK-SAME:                           %[[LOWER_EDGE_HZ_ARG:.*]]: !torch.vtensor<[],f32>,
  // CHECK-SAME:                           %[[UPPER_EDGE_HZ_ARG:.*]]: !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_5:.*]] = torch.constant.none
  // CHECK:           %[[NUM_MEL_BINS_ITEM:.*]] = torch.aten.item %[[NUM_MEL_BINS_ARG]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:           %[[SAMPLE_RATE_ITEM:.*]] = torch.aten.item %[[SAMPLE_RATE_ARG]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:           %[[LOWER_EDGE_HZ_ITEM:.*]] = torch.aten.item %[[LOWER_EDGE_HZ_ARG]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK:           %[[UPPER_EDGE_HZ_ITEM:.*]] = torch.aten.item %[[UPPER_EDGE_HZ_ARG]] : !torch.vtensor<[],f32> -> !torch.float
  // CHECK:           %[[VAL_10:.*]] = torch.constant.none
  // CHECK:           %[[VAL_11:.*]] = torch.constant.int 0
  // CHECK:           %[[VAL_12:.*]] = torch.constant.int 1
  // CHECK:           %[[VAL_13:.*]] = torch.constant.int 2
  // CHECK:           %[[VAL_14:.*]] = torch.constant.int 3
  // CHECK:           %[[VAL_15:.*]] = torch.constant.int 6
  // CHECK:           %[[VAL_16:.*]] = torch.aten.floor_divide.Scalar %[[DFT_LENGTH_ARG]], %[[VAL_13]] : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],si64>
  // CHECK:           %[[VAL_17:.*]] = torch.aten.add.Scalar %[[VAL_16]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK:           %[[NUM_SPECTROGRAM_BINS_ITEM:.*]] = torch.aten.item %[[VAL_17]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:           %[[VAL_19:.*]] = torch.constant.float 2.595000e+03
  // CHECK:           %[[VAL_20:.*]] = torch.constant.float 7.000000e+02
  // CHECK:           %[[VAL_21:.*]] = torch.constant.float 1.000000e+01
  // CHECK:           %[[VAL_22:.*]] = torch.constant.float 1.000000e+00
  // CHECK:           %[[CONST_LN_TO_LOG10:.*]] = torch.constant.float 0.43429448190325182
  // CHECK:           %[[VAL_24:.*]] = torch.aten.div.float %[[LOWER_EDGE_HZ_ITEM]], %[[VAL_20]] : !torch.float, !torch.float -> !torch.float
  // CHECK:           %[[VAL_25:.*]] = torch.prim.NumToTensor.Scalar %[[VAL_24]] : !torch.float -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_26:.*]] = torch.aten.add.Scalar %[[VAL_25]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[],f32>, !torch.int, !torch.int -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_27:.*]] = torch.aten.log %[[VAL_26]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_28:.*]] = torch.aten.mul.Scalar %[[VAL_27]], %[[CONST_LN_TO_LOG10]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK:           %[[LOW_FREQ_MEL:.*]] = torch.aten.mul.Scalar %[[VAL_28]], %[[VAL_19]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_30:.*]] = torch.aten.div.float %[[UPPER_EDGE_HZ_ITEM]], %[[VAL_20]] : !torch.float, !torch.float -> !torch.float
  // CHECK:           %[[VAL_31:.*]] = torch.prim.NumToTensor.Scalar %[[VAL_30]] : !torch.float -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_32:.*]] = torch.aten.add.Scalar %[[VAL_31]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[],f32>, !torch.int, !torch.int -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_33:.*]] = torch.aten.log %[[VAL_32]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_34:.*]] = torch.aten.mul.Scalar %[[VAL_33]], %[[CONST_LN_TO_LOG10]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK:           %[[HIGH_FREQ_MEL:.*]] = torch.aten.mul.Scalar %[[VAL_34]], %[[VAL_19]] : !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_36:.*]] = torch.aten.sub.Tensor %[[HIGH_FREQ_MEL]], %[[LOW_FREQ_MEL]], %[[VAL_12]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_37:.*]] = torch.aten.add.int %[[NUM_MEL_BINS_ITEM]], %[[VAL_13]] : !torch.int, !torch.int -> !torch.int
  // CHECK:           %[[MEL_STEP:.*]] = torch.aten.div.Scalar %[[VAL_36]], %[[VAL_37]] : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  // CHECK:           %[[LOW_BINS_INIT:.*]] = torch.aten.arange %[[NUM_MEL_BINS_ITEM]], %[[VAL_14]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]] : !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[8],si32>
  // CHECK:           %[[CENTER_BINS_INIT:.*]] = torch.aten.arange %[[NUM_MEL_BINS_ITEM]], %[[VAL_14]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]] : !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[8],si32>
  // CHECK:           %[[HIGH_BINS_INIT:.*]] = torch.aten.arange %[[NUM_MEL_BINS_ITEM]], %[[VAL_14]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]] : !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[8],si32>
  // CHECK:           %[[VAL_42:.*]] = torch.aten.add.Scalar %[[DFT_LENGTH_ARG]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  // CHECK:           %[[VAL_43:.*]] = torch.aten.item %[[VAL_42]] : !torch.vtensor<[],si64> -> !torch.int
  // CHECK:           %[[VAL_44:.*]] = torch.constant.bool false
  // CHECK:           %[[VAL_45:.*]] = torch.aten.mul.Tensor %[[LOW_BINS_INIT]], %[[MEL_STEP]] : !torch.vtensor<[8],si32>, !torch.vtensor<[],f32> -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_46:.*]] = torch.aten.add.Tensor %[[VAL_45]], %[[LOW_FREQ_MEL]], %[[VAL_12]] : !torch.vtensor<[8],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_47:.*]] = torch.aten.div.Scalar %[[VAL_46]], %[[VAL_19]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_48:.*]] = torch.aten.clone %[[VAL_46]], %[[VAL_10]] : !torch.vtensor<[8],f32>, !torch.none -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_49:.*]] = torch.aten.fill.Scalar %[[VAL_48]], %[[VAL_21]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_50:.*]] = torch.aten.pow.Tensor_Tensor %[[VAL_49]], %[[VAL_47]] : !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32> -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_51:.*]] = torch.aten.sub.Scalar %[[VAL_50]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[8],f32>, !torch.int, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_52:.*]] = torch.aten.mul.Scalar %[[VAL_51]], %[[VAL_20]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_53:.*]] = torch.aten.mul.Scalar %[[VAL_52]], %[[VAL_43]] : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_54:.*]] = torch.aten.div.Scalar %[[VAL_53]], %[[SAMPLE_RATE_ITEM]] : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_55:.*]] = torch.aten.to.dtype %[[VAL_54]], %[[VAL_14]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[8],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[8],si32>
  // CHECK:           %[[LOW_BINS:.*]] = torch.aten.unsqueeze %[[VAL_55]], %[[VAL_11]] : !torch.vtensor<[8],si32>, !torch.int -> !torch.vtensor<[1,8],si32>
  // CHECK:           %[[VAL_57:.*]] = torch.aten.add.Scalar %[[CENTER_BINS_INIT]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[8],si32>, !torch.int, !torch.int -> !torch.vtensor<[8],si32>
  // CHECK:           %[[VAL_58:.*]] = torch.aten.mul.Tensor %[[VAL_57]], %[[MEL_STEP]] : !torch.vtensor<[8],si32>, !torch.vtensor<[],f32> -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_59:.*]] = torch.aten.add.Tensor %[[VAL_58]], %[[LOW_FREQ_MEL]], %[[VAL_12]] : !torch.vtensor<[8],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_60:.*]] = torch.aten.div.Scalar %[[VAL_59]], %[[VAL_19]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_61:.*]] = torch.aten.clone %[[VAL_59]], %[[VAL_10]] : !torch.vtensor<[8],f32>, !torch.none -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_62:.*]] = torch.aten.fill.Scalar %[[VAL_61]], %[[VAL_21]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_63:.*]] = torch.aten.pow.Tensor_Tensor %[[VAL_62]], %[[VAL_60]] : !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32> -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_64:.*]] = torch.aten.sub.Scalar %[[VAL_63]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[8],f32>, !torch.int, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_65:.*]] = torch.aten.mul.Scalar %[[VAL_64]], %[[VAL_20]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_66:.*]] = torch.aten.mul.Scalar %[[VAL_65]], %[[VAL_43]] : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_67:.*]] = torch.aten.div.Scalar %[[VAL_66]], %[[SAMPLE_RATE_ITEM]] : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_68:.*]] = torch.aten.to.dtype %[[VAL_67]], %[[VAL_14]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[8],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[8],si32>
  // CHECK:           %[[CENTER_BINS:.*]] = torch.aten.unsqueeze %[[VAL_68]], %[[VAL_11]] : !torch.vtensor<[8],si32>, !torch.int -> !torch.vtensor<[1,8],si32>
  // CHECK:           %[[VAL_70:.*]] = torch.aten.add.Scalar %[[HIGH_BINS_INIT]], %[[VAL_13]], %[[VAL_12]] : !torch.vtensor<[8],si32>, !torch.int, !torch.int -> !torch.vtensor<[8],si32>
  // CHECK:           %[[VAL_71:.*]] = torch.aten.mul.Tensor %[[VAL_70]], %[[MEL_STEP]] : !torch.vtensor<[8],si32>, !torch.vtensor<[],f32> -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_72:.*]] = torch.aten.add.Tensor %[[VAL_71]], %[[LOW_FREQ_MEL]], %[[VAL_12]] : !torch.vtensor<[8],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_73:.*]] = torch.aten.div.Scalar %[[VAL_72]], %[[VAL_19]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_74:.*]] = torch.aten.clone %[[VAL_72]], %[[VAL_10]] : !torch.vtensor<[8],f32>, !torch.none -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_75:.*]] = torch.aten.fill.Scalar %[[VAL_74]], %[[VAL_21]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_76:.*]] = torch.aten.pow.Tensor_Tensor %[[VAL_75]], %[[VAL_73]] : !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32> -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_77:.*]] = torch.aten.sub.Scalar %[[VAL_76]], %[[VAL_12]], %[[VAL_12]] : !torch.vtensor<[8],f32>, !torch.int, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_78:.*]] = torch.aten.mul.Scalar %[[VAL_77]], %[[VAL_20]] : !torch.vtensor<[8],f32>, !torch.float -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_79:.*]] = torch.aten.mul.Scalar %[[VAL_78]], %[[VAL_43]] : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_80:.*]] = torch.aten.div.Scalar %[[VAL_79]], %[[SAMPLE_RATE_ITEM]] : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8],f32>
  // CHECK:           %[[VAL_81:.*]] = torch.aten.to.dtype %[[VAL_80]], %[[VAL_14]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[8],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[8],si32>
  // CHECK:           %[[HIGH_BINS:.*]] = torch.aten.unsqueeze %[[VAL_81]], %[[VAL_11]] : !torch.vtensor<[8],si32>, !torch.int -> !torch.vtensor<[1,8],si32>
  // CHECK:           %[[IOTA_INIT:.*]] = torch.aten.arange %[[NUM_SPECTROGRAM_BINS_ITEM]], %[[VAL_14]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]] : !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[9],si32>
  // CHECK:           %[[IOTA:.*]] = torch.aten.unsqueeze %[[IOTA_INIT]], %[[VAL_12]] : !torch.vtensor<[9],si32>, !torch.int -> !torch.vtensor<[9,1],si32>
  // CHECK:           %[[LOW_TO_CENTER:.*]] = torch.aten.sub.Tensor %[[CENTER_BINS]], %[[LOW_BINS]], %[[VAL_12]] : !torch.vtensor<[1,8],si32>, !torch.vtensor<[1,8],si32>, !torch.int -> !torch.vtensor<[1,8],si32>
  // CHECK:           %[[CENTER_TO_HIGH:.*]] = torch.aten.sub.Tensor %[[HIGH_BINS]], %[[CENTER_BINS]], %[[VAL_12]] : !torch.vtensor<[1,8],si32>, !torch.vtensor<[1,8],si32>, !torch.int -> !torch.vtensor<[1,8],si32>
  // CHECK:           %[[VAL_87:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK:           %[[VAL_88:.*]] = torch.constant.none
  // CHECK:           %[[VAL_89:.*]] = torch.constant.int 6
  // CHECK:           %[[VAL_90:.*]] = torch.aten.full %[[VAL_87]], %[[VAL_12]], %[[VAL_89]], %[[VAL_88]], %[[VAL_88]], %[[VAL_88]] : !torch.list<int>, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
  // CHECK:           %[[VAL_91:.*]] = torch.aten.maximum %[[VAL_90]], %[[LOW_TO_CENTER]] : !torch.vtensor<[],f32>, !torch.vtensor<[1,8],si32> -> !torch.vtensor<[1,8],si32>
  // CHECK:           %[[UP_SCALE:.*]] = torch.aten.to.dtype %[[VAL_91]], %[[VAL_15]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[1,8],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,8],f32>
  // CHECK:           %[[VAL_93:.*]] = torch.aten.maximum %[[VAL_90]], %[[CENTER_TO_HIGH]] : !torch.vtensor<[],f32>, !torch.vtensor<[1,8],si32> -> !torch.vtensor<[1,8],si32>
  // CHECK:           %[[DOWN_SCALE:.*]] = torch.aten.to.dtype %[[VAL_93]], %[[VAL_15]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[1,8],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,8],f32>
  // CHECK:           %[[VAL_95:.*]] = torch.aten.sub.Tensor %[[IOTA]], %[[LOW_BINS]], %[[VAL_12]] : !torch.vtensor<[9,1],si32>, !torch.vtensor<[1,8],si32>, !torch.int -> !torch.vtensor<[9,8],si32>
  // CHECK:           %[[VAL_96:.*]] = torch.aten.to.dtype %[[VAL_95]], %[[VAL_15]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[9,8],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[RAMP_UP:.*]] = torch.aten.div.Tensor %[[VAL_96]], %[[UP_SCALE]] : !torch.vtensor<[9,8],f32>, !torch.vtensor<[1,8],f32> -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[VAL_98:.*]] = torch.aten.sub.Tensor %[[HIGH_BINS]], %[[IOTA]], %[[VAL_12]] : !torch.vtensor<[1,8],si32>, !torch.vtensor<[9,1],si32>, !torch.int -> !torch.vtensor<[9,8],si32>
  // CHECK:           %[[VAL_99:.*]] = torch.aten.to.dtype %[[VAL_98]], %[[VAL_15]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[9,8],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[RAMP_DOWN:.*]] = torch.aten.div.Tensor %[[VAL_99]], %[[DOWN_SCALE]] : !torch.vtensor<[9,8],f32>, !torch.vtensor<[1,8],f32> -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[VAL_101:.*]] = torch.aten.ge.Tensor %[[IOTA]], %[[CENTER_BINS]] : !torch.vtensor<[9,1],si32>, !torch.vtensor<[1,8],si32> -> !torch.vtensor<[9,8],i1>
  // CHECK:           %[[VAL_102:.*]] = torch.aten.eq.Tensor %[[IOTA]], %[[CENTER_BINS]] : !torch.vtensor<[9,1],si32>, !torch.vtensor<[1,8],si32> -> !torch.vtensor<[9,8],i1>
  // CHECK:           %[[VAL_103:.*]] = torch.aten.lt.Tensor %[[IOTA]], %[[LOW_BINS]] : !torch.vtensor<[9,1],si32>, !torch.vtensor<[1,8],si32> -> !torch.vtensor<[9,8],i1>
  // CHECK:           %[[VAL_104:.*]] = torch.aten.gt.Tensor %[[IOTA]], %[[HIGH_BINS]] : !torch.vtensor<[9,1],si32>, !torch.vtensor<[1,8],si32> -> !torch.vtensor<[9,8],i1>
  // CHECK:           %[[RAMP_INIT:.*]] = torch.aten.where.self %[[VAL_101]], %[[RAMP_DOWN]], %[[RAMP_UP]] : !torch.vtensor<[9,8],i1>, !torch.vtensor<[9,8],f32>, !torch.vtensor<[9,8],f32> -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[VAL_106:.*]] = torch.aten.where.ScalarSelf %[[VAL_103]], %[[VAL_11]], %[[RAMP_INIT]] : !torch.vtensor<[9,8],i1>, !torch.int, !torch.vtensor<[9,8],f32> -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[VAL_107:.*]] = torch.aten.where.ScalarSelf %[[VAL_104]], %[[VAL_11]], %[[VAL_106]] : !torch.vtensor<[9,8],i1>, !torch.int, !torch.vtensor<[9,8],f32> -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[VAL_108:.*]] = torch.aten.eq.Scalar %[[CENTER_TO_HIGH]], %[[VAL_11]] : !torch.vtensor<[1,8],si32>, !torch.int -> !torch.vtensor<[1,8],i1>
  // CHECK:           %[[CORNER_CASES:.*]] = torch.aten.logical_and %[[VAL_102]], %[[VAL_108]] : !torch.vtensor<[9,8],i1>, !torch.vtensor<[1,8],i1> -> !torch.vtensor<[9,8],i1>
  // CHECK:           %[[RAMP:.*]] = torch.aten.where.ScalarSelf %[[CORNER_CASES]], %[[VAL_22]], %[[VAL_107]] : !torch.vtensor<[9,8],i1>, !torch.float, !torch.vtensor<[9,8],f32> -> !torch.vtensor<[9,8],f32>
  // CHECK:           %[[VAL_111:.*]] = torch.constant.int 6
  // CHECK:           %[[OUTPUT:.*]] = torch.aten.to.dtype %[[RAMP]], %[[VAL_111]], %[[VAL_44]], %[[VAL_44]], %[[VAL_10]] : !torch.vtensor<[9,8],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[9,8],f32>
  // CHECK:           return %[[OUTPUT]] : !torch.vtensor<[9,8],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.MelWeightMatrix"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[9,8],f32>
  return %0 : !torch.vtensor<[9,8],f32>
}
