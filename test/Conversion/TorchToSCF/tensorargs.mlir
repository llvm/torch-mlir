// RUN: torch-mlir-opt <%s -convert-torch-to-scf -split-input-file -verify-diagnostics | FileCheck %s
// CHECK-LABEL:   func.func @test_forloop_tensorargs() -> !torch.vtensor<[2,3],f32> {
// CHECK:           %[[LOOP_RESULT:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_13:.*]] to %[[VAL_15:.*]] step %[[VAL_14:.*]] iter_args(%[[LOOP_TENSOR_ARG:.*]] = %[[VAL_11:.*]]) -> (tensor<2x3xf32>) {
// CHECK:             %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[LOOP_TENSOR_ARG]]
// CHECK:           }
// CHECK:         }
func.func @test_forloop_tensorargs() -> (!torch.vtensor<[2,3],f32>) {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int5 = torch.constant.int 5
    %int6 = torch.constant.int 6
    %none = torch.constant.none
    %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.zeros %0, %int6, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
    %2 = torch.aten.ones %0, %int6, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
    %3:1 = torch.prim.Loop %int5, %true, init(%1) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.vtensor<[2,3],f32>):
        %4 = torch.aten.add.Tensor %arg2, %2, %int1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[2,3],f32>
        torch.prim.Loop.condition %true, iter(%4 : !torch.vtensor<[2,3],f32>)
    } : (!torch.int, !torch.bool, !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[2,3],f32>)
    return %3#0 : !torch.vtensor<[2,3],f32>
}
