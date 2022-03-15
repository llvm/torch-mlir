// RUN: torch-mlir-opt -torch-drop-shape-calculations -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                 %[[ARG:.*]]: !torch.vtensor<[2,?],unk>) -> !torch.vtensor {
// CHECK:           %[[TANH:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor<[2,?],unk> -> !torch.vtensor<[2,?],unk>
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[TANH]] : !torch.vtensor<[2,?],unk> to !torch.vtensor
// CHECK:           return %[[ERASED]] : !torch.vtensor
func @basic(%arg0: !torch.vtensor<[2,?],unk>) -> !torch.vtensor {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.shape.calculate  {
    %2 = torch.aten.tanh %arg0 : !torch.vtensor<[2,?],unk> -> !torch.vtensor<[2,?],unk>
    torch.shape.calculate.yield %2 : !torch.vtensor<[2,?],unk>
  } shapes  {
    %2 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[2,?],unk>, !torch.int -> !torch.int
    %3 = torch.prim.ListConstruct %int2, %2 : (!torch.int, !torch.int) -> !torch.list<int>
    torch.shape.calculate.yield.shapes %3 : !torch.list<int>
  } : !torch.vtensor<[2,?],unk>
  %1 = torch.tensor_static_info_cast %0 : !torch.vtensor<[2,?],unk> to !torch.vtensor
  return %1 : !torch.vtensor
}
