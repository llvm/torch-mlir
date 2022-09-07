add_mlir_conversion_library(TorchMLIRTorchToMhlo
  TorchToMhlo.cpp
  MhloLegalizeUtils.cpp
  Basic.cpp
  Gather.cpp
  Linear.cpp
  ViewLike.cpp
  Reduction.cpp
  Pooling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToMhlo

  DEPENDS
  MhloDialect
  MhloToLinalg
  MLIRMhloPassIncGen
  LMHLOTransformsPassIncGen
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ChloOps
  MLIRIR
  MLIRPass
  MhloDialect
  MhloToLinalg
  MLIRBufferTransforms
  StablehloBase
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToMhlo)
