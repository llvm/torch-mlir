module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],i1>, %arg2: !torch.vtensor<[],si64>) -> (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) {
    %int6 = torch.constant.int 6
    %0 = torch.prims.convert_element_type %arg2, %int6 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %1 = torch.aten.where.self %arg1, %0, %arg0 : !torch.vtensor<[128,128],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    return %1, %1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>
  }
}
