// RUN: torch-mlir-opt --torch-simplification-pipeline='shape-dtype-refine=true' --split-input-file %s | FileCheck %s

// QInt16's negative dtype encoding must round-trip through dtype refinement.
// CHECK-LABEL: func.func @qint16_relu
// CHECK: torch.aten.relu {{.*}} -> !torch.vtensor<[4],!torch.qint16>
func.func @qint16_relu(%arg0: !torch.vtensor<[4],!torch.qint16>) -> !torch.vtensor<[4],unk> {
  %result = torch.aten.relu %arg0 : !torch.vtensor<[4],!torch.qint16> -> !torch.vtensor<[4],unk>
  return %result : !torch.vtensor<[4],unk>
}

// -----

// Same-type promotion must preserve QInt16 without indexing the lookup table.
// CHECK-LABEL: func.func @promote_qint16_pair
// CHECK: %[[DTYPE:.*]] = torch.constant.int -1
// CHECK: return %[[DTYPE]] : !torch.int
func.func @promote_qint16_pair(%arg0: !torch.vtensor<[4],!torch.qint16>) -> !torch.int {
  %rank = torch.constant.int 1
  %dtype = torch.prim.dtype %arg0 : !torch.vtensor<[4],!torch.qint16> -> !torch.int
  %ranks = torch.prim.ListConstruct %rank, %rank : (!torch.int, !torch.int) -> !torch.list<optional<int>>
  %dtypes = torch.prim.ListConstruct %dtype, %dtype : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
  return %result : !torch.int
}

// -----

// Unsupported mixed promotion returns Undefined, rather than indexing at -1.
// CHECK-LABEL: func.func @promote_qint16_float
// CHECK: %[[DTYPE:.*]] = torch.constant.int 47
// CHECK: return %[[DTYPE]] : !torch.int
func.func @promote_qint16_float(%arg0: !torch.vtensor<[4],!torch.qint16>) -> !torch.int {
  %rank = torch.constant.int 1
  %qint16 = torch.prim.dtype %arg0 : !torch.vtensor<[4],!torch.qint16> -> !torch.int
  %float = torch.constant.int 6
  %ranks = torch.prim.ListConstruct %rank, %rank : (!torch.int, !torch.int) -> !torch.list<optional<int>>
  %dtypes = torch.prim.ListConstruct %qint16, %float : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
  return %result : !torch.int
}

// -----

// CHECK-LABEL: func.func @promote_float_qint16
// CHECK: %[[DTYPE:.*]] = torch.constant.int 47
// CHECK: return %[[DTYPE]] : !torch.int
func.func @promote_float_qint16(%arg0: !torch.vtensor<[4],!torch.qint16>) -> !torch.int {
  %rank = torch.constant.int 1
  %qint16 = torch.prim.dtype %arg0 : !torch.vtensor<[4],!torch.qint16> -> !torch.int
  %float = torch.constant.int 6
  %ranks = torch.prim.ListConstruct %rank, %rank : (!torch.int, !torch.int) -> !torch.list<optional<int>>
  %dtypes = torch.prim.ListConstruct %float, %qint16 : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
  return %result : !torch.int
}
