add_mlir_library(TorchMLIRTMTensorPasses
  ConvertToLoops.cpp
  Passes.cpp

  DEPENDS
  TorchMLIRTMTensorTransformsPassesIncGen

  LINK_LIBS PUBLIC
  TorchMLIRTMTensorDialect
  MLIRAffine
  MLIRIR
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRMath
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTensor
  MLIRTransforms
)
