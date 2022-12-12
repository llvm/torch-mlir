add_mlir_conversion_library(TorchMLIRTorchToTcp
  Elementwise.cpp
  Misc.cpp
  TorchToTcp.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToTcp
  
  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TorchMLIRTcpDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToTcp)