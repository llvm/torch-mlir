// RUN: torch-mlir-opt <%s -convert-torch-to-tosa -split-input-file

// CHECK:  %{{.*}} = tosa.cast %{{.*}} : (tensor<1x32x220x220xf32>) -> tensor<1x32x220x220xf16> 
func.func @forward(%arg0: !torch.vtensor<[1,32,220,220],f32>) -> !torch.vtensor<[1,32,220,220],f16> {
  %int5 = torch.constant.int 5
  %false = torch.constant.bool false
  %none = torch.constant.none
  %out = torch.aten.to.dtype %arg0, %int5, %false, %false, %none : !torch.vtensor<[1,32,220,220],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,32,220,220],f16>
  return %out : !torch.vtensor<[1,32,220,220],f16>
}


