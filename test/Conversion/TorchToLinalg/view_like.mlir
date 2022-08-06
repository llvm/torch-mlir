// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s


func.func @torch.aten.view$ViewExpandModule(%arg0: !torch.vtensor<[6,4],f32>) -> !torch.vtensor<[2,3,4],f32> {
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[6,4],f32>, !torch.list<int> -> !torch.vtensor<[2,3,4],f32>
  return %1 : !torch.vtensor<[2,3,4],f32>
}


func.func @torch.aten.view$ViewExpandOnesModule(%arg0: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,1,1,1,1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int1, %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[1,1,1,1,1],f32>
  return %1 : !torch.vtensor<[1,1,1,1,1],f32>
}


func.func @torch.aten.view$ViewZeroRankModule(%arg0: !torch.vtensor<[],si64>) -> !torch.vtensor<[1],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[],si64>, !torch.list<int> -> !torch.vtensor<[1],si64>
  return %1 : !torch.vtensor<[1],si64>
}


func.func @torch.aten.view$ViewDynamicExpandModule(%arg0: !torch.vtensor<[?,?,30,384],f32>) -> !torch.vtensor<[2,4,5,6,12,32],f32> {
  %int32 = torch.constant.int 32
  %int12 = torch.constant.int 12
  %int6 = torch.constant.int 6
  %int5 = torch.constant.int 5
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int2, %int4, %int5, %int6, %int12, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,30,384],f32>, !torch.list<int> -> !torch.vtensor<[2,4,5,6,12,32],f32>
  return %1 : !torch.vtensor<[2,4,5,6,12,32],f32>
}


func.func @torch.aten.view$ViewDynamicExpandWithAtenSizeIntModule(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,12,32],f32> {
  %int32 = torch.constant.int 32
  %int12 = torch.constant.int 12
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
  %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.int
  %2 = torch.prim.ListConstruct %0, %1, %int12, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,12,32],f32>
  return %3 : !torch.vtensor<[?,?,12,32],f32>
}


func.func @torch.aten.view$ViewExpandOnesBeforeAndAfterModule(%arg0: !torch.vtensor<[2,1,16,1,1],f32>) -> !torch.vtensor<[1,2,1,16,1,1,1,1],f32> {
  %int16 = torch.constant.int 16
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int2, %int1, %int16, %int1, %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,1,16,1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,2,1,16,1,1,1,1],f32>
  return %1 : !torch.vtensor<[1,2,1,16,1,1,1,1],f32>
}


func.func @torch.aten.view$ViewCollapseModule(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[8],f32> {
  %int8 = torch.constant.int 8
  %0 = torch.prim.ListConstruct %int8 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[8],f32>
  return %1 : !torch.vtensor<[8],f32>
}


func.func @torch.aten.view$ViewExpandCollapseModule(%arg0: !torch.vtensor<[2,4,8,16,4],f32>) -> !torch.vtensor<[8,2,4,16,2,2],f32> {
  %int16 = torch.constant.int 16
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int8 = torch.constant.int 8
  %0 = torch.prim.ListConstruct %int8, %int2, %int4, %int16, %int2, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,4,8,16,4],f32>, !torch.list<int> -> !torch.vtensor<[8,2,4,16,2,2],f32>
  return %1 : !torch.vtensor<[8,2,4,16,2,2],f32>
}


func.func @torch.aten.view$ViewDynamicInferMatchingStaticModule(%arg0: !torch.vtensor<[?,512,1,1],f32>) -> !torch.vtensor<[?,512],f32> {
  %int512 = torch.constant.int 512
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int-1, %int512 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,512,1,1],f32>, !torch.list<int> -> !torch.vtensor<[?,512],f32>
  return %1 : !torch.vtensor<[?,512],f32>
}


func.func @torch.aten.view$ViewExpandCollapseWithOnesModule(%arg0: !torch.vtensor<[2,4,8,8],f32>) -> !torch.vtensor<[2,1,1,4,64],f32> {
  %int64 = torch.constant.int 64
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int2, %int1, %int1, %int4, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,4,8,8],f32>, !torch.list<int> -> !torch.vtensor<[2,1,1,4,64],f32>
  return %1 : !torch.vtensor<[2,1,1,4,64],f32>
}


func.func @torch.aten.view$ViewCollapseDynamicWithAtenSizeIntModule(%arg0: !torch.vtensor<[?,?,?,?,?,?],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[],si64>) -> !torch.vtensor<[?,?,?,?,?],f32> {
  %int384 = torch.constant.int 384
  %int3 = torch.constant.int 3
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,?,?,?,?],f32>, !torch.int -> !torch.int
  %1 = torch.aten.Int.Tensor %arg1 : !torch.vtensor<[],si64> -> !torch.int
  %2 = torch.aten.Int.Tensor %arg2 : !torch.vtensor<[],si64> -> !torch.int
  %3 = torch.aten.size.int %arg0, %int3 : !torch.vtensor<[?,?,?,?,?,?],f32>, !torch.int -> !torch.int
  %4 = torch.prim.ListConstruct %0, %1, %2, %3, %int384 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.aten.view %arg0, %4 : !torch.vtensor<[?,?,?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?,?],f32>
  return %5 : !torch.vtensor<[?,?,?,?,?],f32>
}


func.func @torch.aten.view$ViewExpandOnesMiddleModule(%arg0: !torch.vtensor<[3,1,2],f32>) -> !torch.vtensor<[3,1,1,1,1,2],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int1, %int1, %int1, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[3,1,2],f32>, !torch.list<int> -> !torch.vtensor<[3,1,1,1,1,2],f32>
  return %1 : !torch.vtensor<[3,1,1,1,1,2],f32>
}


func.func @torch.aten.view$ViewDynamicExpandCollapseModule(%arg0: !torch.vtensor<[?,4,?,?],f32>) -> !torch.vtensor<[2,1,4,64],f32> {
  %int64 = torch.constant.int 64
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int2, %int1, %int4, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,4,?,?],f32>, !torch.list<int> -> !torch.vtensor<[2,1,4,64],f32>
  return %1 : !torch.vtensor<[2,1,4,64],f32>
}


func.func @torch.aten.view$ViewDynamicExpandCollapseWithAtenIntModule(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[2,1,?,64],f32> {
  %int64 = torch.constant.int 64
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
  %1 = torch.prim.ListConstruct %int2, %int1, %0, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.view %arg0, %1 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[2,1,?,64],f32>
  return %2 : !torch.vtensor<[2,1,?,64],f32>
}


func.func @torch.aten.view$ViewDynamicInferModule(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,64],f32> {
  %int64 = torch.constant.int 64
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int-1, %int64 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,64],f32>
  return %1 : !torch.vtensor<[?,64],f32>
}


func.func @torch.aten.view$ViewDynamicInferDynamicOutputModule(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,8],f32> {
  %int8 = torch.constant.int 8
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %0 = torch.aten.size.int %arg0, %int2 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
  %1 = torch.aten.size.int %arg0, %int2 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
  %2 = torch.prim.ListConstruct %0, %int-1, %1, %int8 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,8],f32>
  return %3 : !torch.vtensor<[?,?,?,8],f32>
}


func.func @torch.aten.view$ViewCollapseInferredDimModule(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[6,4],f32> {
  %int4 = torch.constant.int 4
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int-1, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[6,4],f32>
  return %1 : !torch.vtensor<[6,4],f32>
}



func.func @torch.aten.view$ViewExpandInferredDimModule(%arg0: !torch.vtensor<[2,6],f32>) -> !torch.vtensor<[2,3,2],f32> {
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int2, %int-1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,6],f32>, !torch.list<int> -> !torch.vtensor<[2,3,2],f32>
  return %1 : !torch.vtensor<[2,3,2],f32>
}


func.func @torch.aten.view$ViewNoChange1dModule(%arg0: !torch.vtensor<[?],f32>) -> !torch.vtensor<[6],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.prim.ListConstruct %int6 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?],f32>, !torch.list<int> -> !torch.vtensor<[6],f32>
  return %1 : !torch.vtensor<[6],f32>
}


func.func @torch.aten.view$ViewNoChangeStaticModule(%arg0: !torch.vtensor<[4,5,6],f32>) -> !torch.vtensor<[4,5,6],f32> {
  %int6 = torch.constant.int 6
  %int5 = torch.constant.int 5
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int4, %int5, %int6 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[4,5,6],f32>, !torch.list<int> -> !torch.vtensor<[4,5,6],f32>
  return %1 : !torch.vtensor<[4,5,6],f32>
}


func.func @torch.aten.view$ViewNoChange2dModule(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[5,6],f32> {
  %int6 = torch.constant.int 6
  %int5 = torch.constant.int 5
  %0 = torch.prim.ListConstruct %int5, %int6 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[5,6],f32>
  return %1 : !torch.vtensor<[5,6],f32>
}


func.func @torch.aten.view$ViewNoChange3dModule(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[4,5,6],f32> {
  %int6 = torch.constant.int 6
  %int5 = torch.constant.int 5
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int4, %int5, %int6 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[4,5,6],f32>
  return %1 : !torch.vtensor<[4,5,6],f32>
}
