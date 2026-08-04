module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[],f32> {
    %int1 = torch.constant.int 1
    %int1_0 = torch.constant.int 1
    %0 = torch.aten.sub.Scalar %arg1, %int1, %int1_0 : !torch.vtensor<[128,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[128,128],f32>
    %1 = torch.aten.neg %arg0 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %2 = torch.aten.log1p %1 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %int-100 = torch.constant.int -100
    %none = torch.constant.none
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %false = torch.constant.bool false
    %4 = torch.aten.new_full %arg0, %3, %int-100, %none, %none_1, %none_2, %false : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %5 = torch.aten.maximum %2, %4 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[128,128],f32>
    %6 = torch.aten.mul.Tensor %0, %5 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %7 = torch.aten.log %arg0 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %8 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %int-100_3 = torch.constant.int -100
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %false_7 = torch.constant.bool false
    %9 = torch.aten.new_full %arg0, %8, %int-100_3, %none_4, %none_5, %none_6, %false_7 : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %10 = torch.aten.maximum %7, %9 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[128,128],f32>
    %11 = torch.aten.mul.Tensor %arg1, %10 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %int1_8 = torch.constant.int 1
    %12 = torch.aten.sub.Tensor %6, %11, %int1_8 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    %none_9 = torch.constant.none
    %13 = torch.aten.mean %12, %none_9 : !torch.vtensor<[128,128],f32>, !torch.none -> !torch.vtensor<[],f32>
    return %13 : !torch.vtensor<[],f32>
  }
}
