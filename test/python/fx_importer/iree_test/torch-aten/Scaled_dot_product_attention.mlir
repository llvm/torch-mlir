module {
  func.func @main(%arg0: !torch.vtensor<[32,8,128,64],f32>, %arg1: !torch.vtensor<[32,8,128,64],f32>, %arg2: !torch.vtensor<[32,8,128,64],f32>) -> !torch.vtensor<[32,8,128,64],f32> {
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false = torch.constant.bool false
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %0:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%arg0, %arg1, %arg2, %float0.000000e00, %false, %none, %none_0) : (!torch.vtensor<[32,8,128,64],f32>, !torch.vtensor<[32,8,128,64],f32>, !torch.vtensor<[32,8,128,64],f32>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[32,8,128,64],f32>, !torch.vtensor<[32,8,128],f32>) 
    return %0#0 : !torch.vtensor<[32,8,128,64],f32>
  }
}
