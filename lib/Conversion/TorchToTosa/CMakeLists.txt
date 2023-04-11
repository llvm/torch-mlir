add_mlir_conversion_library(TorchMLIRTorchToTosa
  TorchToTosa.cpp
  TosaLegalizeUtils.cpp
  TosaLegalizeCommon.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToTosa

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToTosa)
