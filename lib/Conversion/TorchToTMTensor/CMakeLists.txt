add_mlir_conversion_library(TorchMLIRTorchToTMTensor
TorchToTMTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToTMTensor

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  TorchMLIRConversionUtils
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchDialect
  TorchMLIRTMTensorDialect
)

torch_mlir_target_includes(TorchMLIRTorchToTMTensor)
