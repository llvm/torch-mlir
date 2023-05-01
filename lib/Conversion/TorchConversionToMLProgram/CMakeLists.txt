add_mlir_conversion_library(TorchMLIRTorchConversionToMLProgram
  TorchConversionToMLProgram.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchConversionToMLProgram

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRMLProgramDialect
  MLIRMathDialect
  MLIRPass
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchConversionToMLProgram)
