module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[2],f32>, %arg2: !torch.vtensor<[2],f32>, %arg3: !torch.vtensor<[1,1],f32>, %arg4: !torch.bool) -> !torch.vtensor<[128,128],f32> {
    %0 = torch.prim.ListConstruct %arg1, %arg2 : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.list<optional<vtensor>>
    %false = torch.constant.bool false
    %1 = torch.aten.index_put %arg0, %0, %arg3, %false : !torch.vtensor<[128,128],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1],f32>, !torch.bool -> !torch.vtensor<[128,128],f32>
    return %1 : !torch.vtensor<[128,128],f32>
  }
}
