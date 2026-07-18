add_mlir_conversion_library(TorchMLIRTorchToSCF
  TorchToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToSCF

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTransformUtils
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToSCF)
