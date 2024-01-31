// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch
// FB OPT OPS from https://github.com/llvm/torch-mlir/issues/2689

// -----
// Fixed unecessarily high since-opset value
func.func @cast_operation(%arg0: !torch.vtensor<[?,?,?,?],si64>) -> !torch.vtensor<[?,?,?,?],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %208 = torch.operator "onnx.Cast"(%arg0) {
    torch.onnx.to = 1 : si64
  } : (!torch.vtensor<[?,?,?,?],si64>) -> !torch.vtensor<[?,?,?,?],f32>
  return %208 : !torch.vtensor<[?,?,?,?],f32>
}

// -----
func.func @div_operation(%arg0: !torch.vtensor<[1,64,768],f32>,
                          %arg1: !torch.vtensor<[1,64,1],f32>)
                          -> !torch.vtensor<[1,64,768],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %209 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[1,64,768],f32>, !torch.vtensor<[1,64,1],f32>) -> !torch.vtensor<[1,64,768],f32>
  return %209 : !torch.vtensor<[1,64,768],f32>
}

// -----
// Fixed.
// this is the onnx opset 1 version of Equal, only int types.
// this used to fail to legalize because the "since" value is set unecessarily high (19)
func.func @equal_operation(%arg0: !torch.vtensor<[4],si64>,
                            %arg1: !torch.vtensor<[4],si64>)
                            -> !torch.vtensor<[4],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %205 = torch.operator "onnx.Equal"(%arg0, %arg1) : (!torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],i1>
  return %205 : !torch.vtensor<[4],i1>
}


// -----
func.func @reduce_mean_operation(%arg0: !torch.vtensor<[1,64,768],f32>)
                                 -> !torch.vtensor<[1,64,1],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // The ReduceMean operation as provided.
  %211 = torch.operator "onnx.ReduceMean"(%arg0) {torch.onnx.axes = [-1 : si64]} : (!torch.vtensor<[1,64,768],f32>) -> !torch.vtensor<[1,64,1],f32>
  return %211 : !torch.vtensor<[1,64,1],f32>
}

// -----
// Fixed.
func.func @cumsum_operation(%arg0: !torch.vtensor<[2,3],f64>,
                            %arg1: !torch.vtensor<[],si32>)
                            -> !torch.vtensor<[2,3],f64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %212 = torch.operator "onnx.CumSum"(%arg0, %arg1) : (!torch.vtensor<[2,3],f64>, !torch.vtensor<[],si32>) -> !torch.vtensor<[2,3],f64>
  return %212 : !torch.vtensor<[2,3],f64>
}