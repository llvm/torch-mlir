add_mlir_conversion_library(TorchMLIRTorchToStd
  TorchToStd.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToStd

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFunc
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToStd)
