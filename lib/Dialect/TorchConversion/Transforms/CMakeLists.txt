add_mlir_library(TorchMLIRTorchConversionPasses
  BackendTypeConversion.cpp
  BackendTypeConversionPasses.cpp  
  Passes.cpp
  VerifyInvariantsBeforeBackendLowering.cpp
  VerifyLinalgOnTensorsBackendContract.cpp
  VerifyTosaBackendContract.cpp


  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/TorchConversion/Transforms

  DEPENDS
  TorchMLIRTorchConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncTransforms
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRTorchToLinalg
  TorchMLIRTorchToTMTensor
  TorchMLIRTorchToStd
  TorchMLIRTorchToSCF
  MLIRMemRefTransforms
)
