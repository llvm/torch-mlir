module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[32768],f32> {
    %int16384 = torch.constant.int 16384
    %0 = torch.prim.ListConstruct %int16384 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[128,128],f32>, !torch.list<int> -> !torch.vtensor<[16384],f32>
    %int1 = torch.constant.int 1
    %2 = torch.aten.unsqueeze %1, %int1 : !torch.vtensor<[16384],f32>, !torch.int -> !torch.vtensor<[16384,1],f32>
    %int16384_0 = torch.constant.int 16384
    %int2 = torch.constant.int 2
    %3 = torch.prim.ListConstruct %int16384_0, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %4 = torch.aten.expand %2, %3, %false : !torch.vtensor<[16384,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[16384,2],f32>
    %int0 = torch.constant.int 0
    %5 = torch.aten.clone %4, %int0 : !torch.vtensor<[16384,2],f32>, !torch.int -> !torch.vtensor<[16384,2],f32>
    %int32768 = torch.constant.int 32768
    %6 = torch.prim.ListConstruct %int32768 : (!torch.int) -> !torch.list<int>
    %7 = torch.aten.view %5, %6 : !torch.vtensor<[16384,2],f32>, !torch.list<int> -> !torch.vtensor<[32768],f32>
    return %7 : !torch.vtensor<[32768],f32>
  }
}
