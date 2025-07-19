// RUN: torch-mlir-opt -canonicalize --split-input-file -verify-diagnostics %s

func.func @torch.aten.assert_tensor_metadata_invalid_dtype() {
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %1 = tensor.empty() : tensor<1x1x128x128xi64>
  %2 = torch_c.from_builtin_tensor %1 : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
  // expected-error @+1 {{torch.aten._assert_tensor_metadata' op Failed to fold the _assert_tensor_metadata op since the dtype does not match}}
  torch.aten._assert_tensor_metadata %2, %none, %none, %int8, %none, %none : !torch.vtensor<[1,1,128,128],si64>, !torch.none, !torch.none, !torch.int, !torch.none, !torch.none
  return
}

func.func @torch.aten.assert_tensor_metadata_invalid_size() {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %sizes = torch.prim.ListConstruct %int0, %int2, %int3
        : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %int4 = torch.constant.int 4
  %none = torch.constant.none
  %1 = tensor.empty() : tensor<1x1x128x128xi64>
  %2 = torch_c.from_builtin_tensor %1 : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
  // expected-error @+1 {{'torch.aten._assert_tensor_metadata' op Failed to fold the _assert_tensor_metadata op since the sizes do not match}}
  torch.aten._assert_tensor_metadata %2, %sizes, %none, %int4, %none, %none : !torch.vtensor<[1,1,128,128],si64>, !torch.list<int>, !torch.none, !torch.int, !torch.none, !torch.none
  return
}

func.func @torch.aten.assert_tensor_metadata_invalid_size_extra_dim() {
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int128 = torch.constant.int 128
  %sizes = torch.prim.ListConstruct %int1, %int1, %int128, %int128, %int4
        : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %1 = tensor.empty() : tensor<1x1x128x128xi64>
  %2 = torch_c.from_builtin_tensor %1 : tensor<1x1x128x128xi64> -> !torch.vtensor<[1,1,128,128],si64>
  // expected-error @+1 {{'torch.aten._assert_tensor_metadata' op Failed to fold the _assert_tensor_metadata op since the sizes do not match}}
  torch.aten._assert_tensor_metadata %2, %sizes, %none, %int4, %none, %none : !torch.vtensor<[1,1,128,128],si64>, !torch.list<int>, !torch.none, !torch.int, !torch.none, !torch.none
  return
}
