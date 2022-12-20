add_mlir_conversion_library(TorchMLIRTorchToMLProgram
  TorchToMLProgram.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToMLProgram

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMLProgramDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToMLProgram)