add_mlir_conversion_library(TorchMLIRStablehloToTcp
  StablehloToTcp.cpp

  ADDITIONAL_HEADER_DIRS
  ${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  TorchMLIRDialectsConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  StablehloOps
  TorchMLIRTcpDialect
)

torch_mlir_dialects_target_includes(TorchMLIRStablehloToTcp)
