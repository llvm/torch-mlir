module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %false = torch.constant.bool false
    %1 = torch.aten.new_zeros %arg0, %0, %none, %none_0, %none_1, %false : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %2 = torch.aten.minimum %1, %arg0 : !torch.vtensor<[],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %3 = torch.aten.abs %arg0 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %4 = torch.aten.neg %3 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %5 = torch.aten.exp %4 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %6 = torch.aten.log1p %5 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %int1 = torch.constant.int 1
    %7 = torch.aten.sub.Tensor %2, %6, %int1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    return %7 : !torch.vtensor<[128,128],f32>
  }
}
