module {
  func.func @main(%arg0: !torch.float, %arg1: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %none = torch.constant.none
    %0 = torch.operator "torch.aten.normal.float_Tensor"(%float0.000000e00, %arg1, %none) : (!torch.float, !torch.vtensor<[128,128],f32>, !torch.none) -> !torch.vtensor<[128,128],f32> 
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
