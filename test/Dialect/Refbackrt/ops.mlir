// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s --dump-input=fail

// CHECK: refbackrt.module_metadata
refbackrt.module_metadata {
  // CHECK: refbackrt.func_metadata
  // TODO(brycearden): Encode unranked memrefs in the ABI
  refbackrt.func_metadata {
    funcName = @f,
    numInputs = 1 : i32,
    numOutputs = 0 : i32,
    inputArgTypes = dense<1> : tensor<1xi32>,
    inputElementTypes = dense<1> : tensor<1xi32>,
    inputRanks = dense<-1> : tensor<1xi32>,
    inputShapes = dense<1> : tensor<4xi32>}
}

func @f(%arg0: tensor<*xf32>) {
  return
}
