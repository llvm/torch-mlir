module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.vtensor<[3],si64>, %arg3: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[128,128],f32> {
    %0 = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[3],si64>) -> !torch.list<optional<vtensor>>
    %false = torch.constant.bool false
    %1 = torch.aten.index_put %arg0, %0, %arg3, %false : !torch.vtensor<[128,128],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[3,3],f32>, !torch.bool -> !torch.vtensor<[128,128],f32>
    return %1 : !torch.vtensor<[128,128],f32>
  }
}
