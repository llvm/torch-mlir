add_mlir_conversion_library(TorchMLIRTorchToSCF
  TorchToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToSCF

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCF
  MLIRStandard
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
)

torch_mlir_target_includes(TorchMLIRTorchToSCF)
