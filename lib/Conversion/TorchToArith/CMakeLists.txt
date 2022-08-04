add_mlir_conversion_library(TorchMLIRTorchToArith
  TorchToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToArith

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToArith)
