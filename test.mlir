func.func @torch.aten.avg_pool2d$count_include_pad(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}
