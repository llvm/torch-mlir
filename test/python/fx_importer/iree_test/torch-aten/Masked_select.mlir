"builtin.module"() ({
  "func.func"() <{function_type = (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],i1>) -> !torch.vtensor<[?],f32>, sym_name = "main"}> ({
  ^bb0(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],i1>):
    %0 = "torch.aten.masked_select"(%arg0, %arg1) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],i1>) -> !torch.vtensor<[?],f32>
    %1 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %2 = "torch.aten.size.int"(%0, %1) : (!torch.vtensor<[?],f32>, !torch.int) -> !torch.int
    %3 = "torch.constant.int"() <{value = 0 : i64}> : () -> !torch.int
    %4 = "torch.aten.ge.int"(%2, %3) : (!torch.int, !torch.int) -> !torch.bool
    %5 = "torch.constant.none"() : () -> !torch.none
    %6 = "torch.constant.none"() : () -> !torch.none
    %7 = "torch.constant.none"() : () -> !torch.none
    %8 = "torch.constant.none"() : () -> !torch.none
    %9 = "torch.aten.scalar_tensor"(%4, %5, %6, %7, %8) : (!torch.bool, !torch.none, !torch.none, !torch.none, !torch.none) -> !torch.vtensor<[],f32>
    %10 = "torch.constant.str"() <{value = "masked_select.shape[0] is outside of inline constraint [0, 16384]."}> : () -> !torch.str
    "torch.operator"(%9, %10) <{name = "torch.aten._assert_async.msg"}> : (!torch.vtensor<[],f32>, !torch.str) -> ()
    %11 = "torch.constant.int"() <{value = 16384 : i64}> : () -> !torch.int
    %12 = "torch.aten.le.int"(%2, %11) : (!torch.int, !torch.int) -> !torch.bool
    %13 = "torch.constant.none"() : () -> !torch.none
    %14 = "torch.constant.none"() : () -> !torch.none
    %15 = "torch.constant.none"() : () -> !torch.none
    %16 = "torch.constant.none"() : () -> !torch.none
    %17 = "torch.aten.scalar_tensor"(%12, %13, %14, %15, %16) : (!torch.bool, !torch.none, !torch.none, !torch.none, !torch.none) -> !torch.vtensor<[],f32>
    %18 = "torch.constant.str"() <{value = "masked_select.shape[0] is outside of inline constraint [0, 16384]."}> : () -> !torch.str
    "torch.operator"(%17, %18) <{name = "torch.aten._assert_async.msg"}> : (!torch.vtensor<[],f32>, !torch.str) -> ()
    "func.return"(%0) : (!torch.vtensor<[?],f32>) -> ()
  }) : () -> ()
}) : () -> ()
