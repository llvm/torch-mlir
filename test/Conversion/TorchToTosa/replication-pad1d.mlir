// RUN: torch-mlir-opt %s -convert-torch-to-tosa -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @asymmetric_rank2
// CHECK: tosa.slice {{.*}} : (tensor<2x3xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x1xf32>
// CHECK: tosa.slice {{.*}} : (tensor<2x3xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x1xf32>
// CHECK: tosa.tile {{.*}} : (tensor<2x1xf32>, !tosa.shape<2>) -> tensor<2x2xf32>
// CHECK: tosa.concat {{.*}} {axis = 1 : i32} : (tensor<2x1xf32>, tensor<2x3xf32>, tensor<2x2xf32>) -> tensor<2x6xf32>
func.func @asymmetric_rank2(%input: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,6],f32> {
  %one = torch.constant.int 1
  %two = torch.constant.int 2
  %pads = torch.prim.ListConstruct %one, %two : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,6],f32>
  return %result : !torch.vtensor<[2,6],f32>
}

// -----

// CHECK-LABEL: func.func @right_only_rank3_f16
// CHECK: tosa.slice {{.*}} : (tensor<1x2x3xf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x2x1xf16>
// CHECK: tosa.tile {{.*}} : (tensor<1x2x1xf16>, !tosa.shape<3>) -> tensor<1x2x4xf16>
// CHECK: tosa.concat {{.*}} {axis = 2 : i32} : (tensor<1x2x3xf16>, tensor<1x2x4xf16>) -> tensor<1x2x7xf16>
func.func @right_only_rank3_f16(%input: !torch.vtensor<[1,2,3],f16>) -> !torch.vtensor<[1,2,7],f16> {
  %zero = torch.constant.int 0
  %four = torch.constant.int 4
  %pads = torch.prim.ListConstruct %zero, %four : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[1,2,3],f16>, !torch.list<int> -> !torch.vtensor<[1,2,7],f16>
  return %result : !torch.vtensor<[1,2,7],f16>
}

// -----

// CHECK-LABEL: func.func @left_only_bf16
// CHECK: tosa.slice {{.*}} : (tensor<2x3xbf16>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x1xbf16>
// CHECK: tosa.tile {{.*}} : (tensor<2x1xbf16>, !tosa.shape<2>) -> tensor<2x4xbf16>
// CHECK: tosa.concat {{.*}} {axis = 1 : i32} : (tensor<2x4xbf16>, tensor<2x3xbf16>) -> tensor<2x7xbf16>
func.func @left_only_bf16(%input: !torch.vtensor<[2,3],bf16>) -> !torch.vtensor<[2,7],bf16> {
  %zero = torch.constant.int 0
  %four = torch.constant.int 4
  %pads = torch.prim.ListConstruct %four, %zero : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],bf16>, !torch.list<int> -> !torch.vtensor<[2,7],bf16>
  return %result : !torch.vtensor<[2,7],bf16>
}

// -----

// CHECK-LABEL: func.func @identity(
// CHECK-SAME: %[[INPUT:.*]]: !torch.vtensor<[2,3],f32>
// CHECK-NOT: tosa.
// CHECK: return %[[INPUT]]
func.func @identity(%input: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %zero = torch.constant.int 0
  %pads = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,3],f32>
  return %result : !torch.vtensor<[2,3],f32>
}

// -----

func.func @negative_padding(%input: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,4],f32> {
  %minus_one = torch.constant.int -1
  %two = torch.constant.int 2
  %pads = torch.prim.ListConstruct %minus_one, %two : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// -----

func.func @dynamic_shape(%input: !torch.vtensor<[?,3],f32>) -> !torch.vtensor<[?,5],f32> {
  %one = torch.constant.int 1
  %pads = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[?,3],f32>, !torch.list<int> -> !torch.vtensor<[?,5],f32>
  return %result : !torch.vtensor<[?,5],f32>
}

// -----

func.func @runtime_padding(%input: !torch.vtensor<[2,3],f32>, %pads: !torch.list<int>) -> !torch.vtensor<[2,5],f32> {
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,5],f32>
  return %result : !torch.vtensor<[2,5],f32>
}

// -----

func.func @malformed_padding(%input: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,5],f32> {
  %two = torch.constant.int 2
  %pads = torch.prim.ListConstruct %two : (!torch.int) -> !torch.list<int>
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,5],f32>
  return %result : !torch.vtensor<[2,5],f32>
}

// -----

func.func @zero_dimension(%input: !torch.vtensor<[0,2,3],f32>) -> !torch.vtensor<[0,2,5],f32> {
  %one = torch.constant.int 1
  %pads = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[0,2,3],f32>, !torch.list<int> -> !torch.vtensor<[0,2,5],f32>
  return %result : !torch.vtensor<[0,2,5],f32>
}

// -----

func.func @invalid_rank(%input: !torch.vtensor<[3],f32>) -> !torch.vtensor<[5],f32> {
  %one = torch.constant.int 1
  %pads = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[3],f32>, !torch.list<int> -> !torch.vtensor<[5],f32>
  return %result : !torch.vtensor<[5],f32>
}

// -----

func.func @padding_overflow(%input: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %max_int = torch.constant.int 9223372036854775807
  %one = torch.constant.int 1
  %pads = torch.prim.ListConstruct %max_int, %one : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,3],f32>
  return %result : !torch.vtensor<[2,3],f32>
}

// -----

func.func @mismatched_result_shape(%input: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,6],f32> {
  %one = torch.constant.int 1
  %pads = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @below {{failed to legalize operation 'torch.aten.replication_pad1d' that was explicitly marked illegal}}
  %result = torch.aten.replication_pad1d %input, %pads : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,6],f32>
  return %result : !torch.vtensor<[2,6],f32>
}
