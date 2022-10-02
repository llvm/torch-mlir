add_mlir_conversion_library(TorchMLIRConversionUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/Utils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLinalgDialect
  TorchMLIRTorchDialect
)
