add_mlir_conversion_library(TorchMLIRTorchToTosaLinalg
  TorchToTosaLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToTosaLinalg

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRLinalgDialect
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToTosaLinalg)
