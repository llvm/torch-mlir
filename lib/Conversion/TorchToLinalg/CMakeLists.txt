add_mlir_conversion_library(TorchMLIRTorchToLinalg
  TorchToLinalg.cpp
  CommonTorchToLinalg.cpp
  Math/Matmul.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToLinalg

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalg
  MLIRMath
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToLinalg)
