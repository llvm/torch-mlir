add_mlir_conversion_library(TorchMLIRTorchToStablehlo
  Passes.cpp
  TorchToStablehlo.cpp
  StablehloLegalizeUtils.cpp
  Basic.cpp
  GatherScatter.cpp
  Linear.cpp
  ViewLike.cpp
  Reduction.cpp
  Pooling.cpp
  VerifyStablehloBackendContract.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToStablehlo

  DEPENDS
  TorchMLIRConversionStablehloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRBufferTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  StablehloOps
  TorchMLIRConversionCommon
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchDialect
  TorchMLIRConversionUtils
)

torch_mlir_target_includes(TorchMLIRTorchToStablehlo)
