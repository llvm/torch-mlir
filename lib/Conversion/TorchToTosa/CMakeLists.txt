add_mlir_conversion_library(TorchMLIRTorchToTosa
  Passes.cpp
  TorchToTosa.cpp
  TosaLegalizeUtils.cpp
  TosaLegalizeCommon.cpp
  VerifyTosaBackendContract.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToTosa

  DEPENDS
  TorchMLIRConversionTosaPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRTransforms
  TorchMLIRTorchConversionPasses
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToTosa)
