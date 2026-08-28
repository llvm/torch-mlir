add_mlir_conversion_library(TorchMLIRTorchToStablehlo
  TorchToStablehlo.cpp
  StablehloLegalizeUtils.cpp
  Basic.cpp
  GatherScatter.cpp
  Linear.cpp
  ViewLike.cpp
  Reduction.cpp
  Rng.cpp
  Pooling.cpp
  Uncategorized.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToStablehlo

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRComplexDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  ChloOps
  StablehloOps
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchDialect
  TorchMLIRConversionUtils
)

torch_mlir_target_includes(TorchMLIRTorchToStablehlo)
