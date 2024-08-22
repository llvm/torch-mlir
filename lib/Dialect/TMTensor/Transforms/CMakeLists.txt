add_mlir_library(TorchMLIRTMTensorPasses
  ConvertToLoops.cpp
  Bufferize.cpp
  Passes.cpp

  DEPENDS
  TorchMLIRTMTensorTransformsPassesIncGen

  LINK_LIBS PUBLIC
  TorchMLIRTMTensorDialect
  MLIRAffineDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

torch_mlir_target_includes(TorchMLIRTMTensorPasses)
