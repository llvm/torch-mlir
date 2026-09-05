module {
  func.func @main(%arg0: !torch.vtensor<[1,2],i1>) -> !torch.vtensor<[],i1> {
    %0 = torch.aten.any %arg0 : !torch.vtensor<[1,2],i1> -> !torch.vtensor<[],i1>
    return %0 : !torch.vtensor<[],i1>
  }
}
