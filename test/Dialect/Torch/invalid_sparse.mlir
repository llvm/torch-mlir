// RUN: torch-mlir-opt %s -verify-diagnostics

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error @+1 {{dimension-rank mismatch between encoding and tensor shape: 1 != 2}}
func.func @foo(%arg0: !torch.vtensor<[64,64],f32,#SV>) -> !torch.vtensor<[64,64],f32,#SV> {
  return %arg0 : !torch.vtensor<[64,64],f32,#SV>
}
