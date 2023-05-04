func.func @torch.aten.addscalar$basic(%arg0: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
  %f9 = torch.constant.int 9
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %f9, %int1 : !torch.vtensor<[?,?],si32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],si32>
  return %0 : !torch.vtensor<[?,?],si32>
}

// func.func @torch.aten.atan2$diffrank(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
//   %0 = torch.aten.atan2 %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?],f32>
//   return %0 : !torch.vtensor<[?,?],f32>
// }


// func.func @torch.aten.atan2$rank0(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[?,?],f32> {
//   %0 = torch.aten.atan2 %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[?,?],f32>
//   return %0 : !torch.vtensor<[?,?],f32>
// }