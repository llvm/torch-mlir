add_mlir_conversion_library(TorchMLIRTorchToMhlo
  TorchToMhlo.cpp
  MhloLegalizeUtils.cpp
  BasicOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToMhlo

  DEPENDS
  MhloDialect
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MhloDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToMhlo)
