add_mlir_conversion_library(TorchMLIRTorchToArith
  TorchToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToArith

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRRewrite
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRTransformUtils
  TorchMLIRConversionUtils
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToArith)
