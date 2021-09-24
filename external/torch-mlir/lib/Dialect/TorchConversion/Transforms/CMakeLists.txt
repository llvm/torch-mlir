add_mlir_library(TorchMLIRTorchConversionPasses
  BackendTypeConversion.cpp
  Passes.cpp
  VerifyInvariantsBeforeBackendLowering.cpp
  VerifyLinalgOnTensorsBackendContract.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/TorchConversion/Transforms

  DEPENDS
  TorchMLIRTorchConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandardOpsTransforms
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRTorchToLinalg
  TorchMLIRTorchToStd
  TorchMLIRTorchToSCF
  MLIRMemRefTransforms
)
