add_mlir_conversion_library(TorchMLIRTorchToTensor
  TorchToTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToTensor

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTensorDialect
  MLIRTransformUtils
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchDialect
  TorchMLIRConversionUtils
)

torch_mlir_target_includes(TorchMLIRTorchToTensor)
