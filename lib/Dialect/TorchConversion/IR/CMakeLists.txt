add_mlir_dialect_library(TorchMLIRTorchConversionDialect
  TorchConversionDialect.cpp
  TorchConversionOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/TorchConversion

  DEPENDS
  MLIRTorchConversionOpsIncGen
  MLIRTorchTypesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)

torch_mlir_target_includes(TorchMLIRTorchConversionDialect)
