// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s
// Generally, the test cases accumulated here come from running the importer
// over all included backend tests that involve simple ops with no model
// level constants. This is a pragmatic choice which lets us have a lot
// of tests in this file, whereas the others tend to be more bespoke.


// CHECK-LABEL: func.func @test_selu
func.func @test_selu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 6 : si64} {
  // CHECK-DAG: %[[F1:.+]] = torch.constant.float 1
  // CHECK-DAG: %[[F2:.+]] = torch.constant.float 2
  // CHECK-DAG: %[[F3:.+]] = torch.constant.float 3
  // CHECK: %[[ELU:.+]] = torch.aten.elu %arg0, %[[F2]], %[[F3]], %[[F1]]
  %0 = torch.operator "onnx.Selu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32, torch.onnx.gamma = 3.000000e+00 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_default_axes_keepdims_example
func.func @test_reduce_sum_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %arg0, %none, %true, %none : !torch.vtensor<[3,2,2],f32>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_do_not_keepdims_example
func.func @test_reduce_sum_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %false, %none : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_empty_axes_input_noop_example
func.func @test_reduce_sum_empty_axes_input_noop_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[3,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64, torch.onnx.noop_with_empty_axes = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[3,2,2],f32>
  return %0 : !torch.vtensor<[3,2,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_empty_set_non_reduced_axis_zero
func.func @test_reduce_sum_empty_set_non_reduced_axis_zero(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %true, %none : !torch.vtensor<[2,0,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,0,1],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,0,1],f32>
  return %0 : !torch.vtensor<[2,0,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_keepdims_example
func.func @test_reduce_sum_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %true, %none : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_negative_axes_keepdims_example
func.func @test_reduce_sum_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.sum.dim_IntList %arg0, %6, %true, %none : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_mean_default_axes_keepdims_example
func.func @test_reduce_mean_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.mean.dim %arg0, %none, %true, %none : !torch.vtensor<[3,2,2],f32>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_mean_do_not_keepdims_example
func.func @test_reduce_mean_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: torch.aten.mean.dim %arg0, %6, %false, %none : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_mean_keepdims_example
func.func @test_reduce_mean_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.mean.dim %arg0, %6, %true, %none : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_mean_negative_axes_keepdims_example
func.func @test_reduce_mean_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[NONE:.+]] = torch.constant.none
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.mean.dim %arg0, %6, %true, %none : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_bool_inputs
func.func @test_reduce_min_bool_inputs(%arg0: !torch.vtensor<[4,2],i1>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int2 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[4,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,1],i1>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[4,2],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1>
  return %0 : !torch.vtensor<[4,1],i1>
}

// CHECK-LABEL: func.func @test_reduce_min_default_axes_keepdims_example
func.func @test_reduce_min_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT1:.+]] = torch.constant.int 1
  // CHECK: %[[INT2:.+]] = torch.constant.int 2
  // CHECK: torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,1],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32>
  return %0 : !torch.vtensor<[1,1,1],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_do_not_keepdims_example
func.func @test_reduce_min_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK: torch.aten.amin %arg0, %6, %false : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_empty_set
func.func @test_reduce_min_empty_set(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[2,0,4],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,1,4],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32>
  return %0 : !torch.vtensor<[2,1,4],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_keepdims_example
func.func @test_reduce_min_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}

// CHECK-LABEL: func.func @test_reduce_min_negative_axes_keepdims_example
func.func @test_reduce_min_negative_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[INT0:.+]] = torch.constant.int 0
  // CHECK: %[[INT3:.+]] = torch.constant.int 3
  // CHECK: %[[INT0_0:.+]] = torch.constant.int 0
  // CHECK: torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  // CHECK: torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  // CHECK: torch.aten.lt.int %1, %int0 : !torch.int, !torch.int -> !torch.bool
  // CHECK: torch.aten.Int.bool %2 : !torch.bool -> !torch.int
  // CHECK: torch.aten.mul.int %3, %int3 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.aten.add.int %1, %4 : !torch.int, !torch.int -> !torch.int
  // CHECK: torch.prim.ListConstruct %5 : (!torch.int) -> !torch.list<int>
  // CHECK: %[[TRUE:.+]] = torch.constant.bool true
  // CHECK: torch.aten.amin %arg0, %6, %true : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1,2],f32>
  %0 = torch.operator "onnx.ReduceMin"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}