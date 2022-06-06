func.func @graph(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.float, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.int, %arg6: !torch.vtensor<[10,5],f32>, %arg7: !torch.vtensor<[10],f32>, %arg8: !torch.vtensor<[1],si64>, %arg9: !torch.vtensor<[],f32>, %arg10: !torch.vtensor<[10,5],f32>, %arg11: !torch.float, %arg12: !torch.int, %arg13: !torch.vtensor<[10],f32>) -> (!torch.vtensor<[1,5],f32>, !torch.vtensor<[10,5],f32>, !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.vtensor<[10,5],f32>, !torch.vtensor<[1,10],f32>, !torch.vtensor<[],f32>) {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1_0 = torch.constant.int 1
  %int0_1 = torch.constant.int 0
  %1 = torch.prim.ListConstruct %int1_0, %int0_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.permute %arg6, %1 : !torch.vtensor<[10,5],f32>, !torch.list<int> -> !torch.vtensor<[5,10],f32>
  %3 = torch.aten.addmm %arg7, %arg0, %2, %arg5, %arg4 : !torch.vtensor<[10],f32>, !torch.vtensor<[1,5],f32>, !torch.vtensor<[5,10],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,10],f32>
  %4 = torch.aten.relu %3 : !torch.vtensor<[1,10],f32> -> !torch.vtensor<[1,10],f32>
  %int1_2 = torch.constant.int 1
  %false = torch.constant.bool false
  %5 = torch.aten._log_softmax %4, %int1_2, %false : !torch.vtensor<[1,10],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,10],f32>
  %none = torch.constant.none
  %int1_3 = torch.constant.int 1
  %int-100 = torch.constant.int -100
  %output, %total_weight = torch.aten.nll_loss_forward %5, %arg8, %none, %int1_3, %int-100 : !torch.vtensor<[1,10],f32>, !torch.vtensor<[1],si64>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  %6 = torch.prim.TupleConstruct %output, %total_weight : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.tuple<vtensor<[],f32>, vtensor<[],f32>>
  %none_4 = torch.constant.none
  %int1_5 = torch.constant.int 1
  %int-100_6 = torch.constant.int -100
  %7 = torch.aten.nll_loss_backward %arg9, %5, %arg8, %none_4, %int1_5, %int-100_6, %total_weight : !torch.vtensor<[],f32>, !torch.vtensor<[1,10],f32>, !torch.vtensor<[1],si64>, !torch.none, !torch.int, !torch.int, !torch.vtensor<[],f32> -> !torch.vtensor<[1,10],f32>
  %int1_7 = torch.constant.int 1
  %int6 = torch.constant.int 6
  %8 = torch.aten._log_softmax_backward_data %7, %5, %int1_7, %int6 : !torch.vtensor<[1,10],f32>, !torch.vtensor<[1,10],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,10],f32>
  %9 = torch.aten.threshold_backward %8, %4, %arg3 : !torch.vtensor<[1,10],f32>, !torch.vtensor<[1,10],f32>, !torch.int -> !torch.vtensor<[1,10],f32>
  %int1_8 = torch.constant.int 1
  %int0_9 = torch.constant.int 0
  %10 = torch.prim.ListConstruct %int1_8, %int0_9 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1_10 = torch.constant.int 1
  %int0_11 = torch.constant.int 0
  %11 = torch.prim.ListConstruct %int1_10, %int0_11 : (!torch.int, !torch.int) -> !torch.list<int>
  %12 = torch.aten.permute %arg0, %11 : !torch.vtensor<[1,5],f32>, !torch.list<int> -> !torch.vtensor<[5,1],f32>
  %13 = torch.aten.mm %12, %9 : !torch.vtensor<[5,1],f32>, !torch.vtensor<[1,10],f32> -> !torch.vtensor<[5,10],f32>
  %int1_12 = torch.constant.int 1
  %int0_13 = torch.constant.int 0
  %14 = torch.prim.ListConstruct %int1_12, %int0_13 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1_14 = torch.constant.int 1
  %int0_15 = torch.constant.int 0
  %15 = torch.prim.ListConstruct %int1_14, %int0_15 : (!torch.int, !torch.int) -> !torch.list<int>
  %16 = torch.aten.permute %13, %15 : !torch.vtensor<[5,10],f32>, !torch.list<int> -> !torch.vtensor<[10,5],f32>
  %17 = torch.aten.zero.functional %arg10 : !torch.vtensor<[10,5],f32> -> !torch.vtensor<[10,5],f32>
  %18 = torch.aten.add.Tensor %17, %16, %arg2 : !torch.vtensor<[10,5],f32>, !torch.vtensor<[10,5],f32>, !torch.int -> !torch.vtensor<[10,5],f32>
  %19 = torch.aten.add.Tensor %arg6, %18, %arg1 : !torch.vtensor<[10,5],f32>, !torch.vtensor<[10,5],f32>, !torch.float -> !torch.vtensor<[10,5],f32>
  %int0_16 = torch.constant.int 0
  %20 = torch.prim.ListConstruct %int0_16 : (!torch.int) -> !torch.list<int>
  %true = torch.constant.bool true
  %none_17 = torch.constant.none
  %21 = torch.aten.sum.dim_IntList %9, %20, %true, %none_17 : !torch.vtensor<[1,10],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,10],f32>
  %int10 = torch.constant.int 10
  %22 = torch.prim.ListConstruct %int10 : (!torch.int) -> !torch.list<int>
  %int10_18 = torch.constant.int 10
  %23 = torch.prim.ListConstruct %int10_18 : (!torch.int) -> !torch.list<int>
  %24 = torch.aten.reshape %21, %23 : !torch.vtensor<[1,10],f32>, !torch.list<int> -> !torch.vtensor<[10],f32>
  %25 = torch.aten.zero.functional %arg13 : !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
  %26 = torch.aten.add.Tensor %25, %24, %arg12 : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.int -> !torch.vtensor<[10],f32>
  %27 = torch.aten.add.Tensor %arg7, %26, %arg11 : !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[10],f32>
  return %arg0, %19, %27, %26, %18, %4, %output : !torch.vtensor<[1,5],f32>, !torch.vtensor<[10,5],f32>, !torch.vtensor<[10],f32>, !torch.vtensor<[10],f32>, !torch.vtensor<[10,5],f32>, !torch.vtensor<[1,10],f32>, !torch.vtensor<[],f32>
}
